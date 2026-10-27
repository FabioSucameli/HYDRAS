# Closed-loop, independent one-step IPP versus the unchanged continuous random walk.

from dataclasses import replace
import json
from time import perf_counter

import numpy as np
from sklearn.preprocessing import StandardScaler

from pollutant_gp.model import integrated_variance_reduction, latent_variance
from pollutant_gp.peak import measure_peak
from pollutant_gp.peak_sampling import common_unsampled_mask
from pollutant_gp.positivity import prediction_metrics
from pollutant_gp.reconstruction import reconstruct_field
from pollutant_gp.robot.study import fit_robot_model, write_rows
from pollutant_gp.robot.informative_checkpoint import load_checkpoint, restore_model, save_checkpoint
from pollutant_gp.robot.trajectories import (
    cell_at, observations_up_to, segment_is_navigable, simulate_random_walk,
)
from pollutant_gp.robot.visualization import plot_robot_results, plot_robot_uncertainty


def uncertainty_summary(std, valid, white_level, target_scale, count, layout):
    """Mean of variances, not square of mean std; same full valid domain for both policies."""
    observed = np.asarray(std)[valid]**2
    latent = latent_variance(np.asarray(std)[valid], white_level, target_scale)
    return dict(checkpoint=count, layout=layout, valid_count=int(valid.sum()),
                mean_latent_variance=float(latent.mean()),
                mean_predictive_variance=float(observed.mean()),
                white_variance=float(white_level * target_scale**2), target_scale=float(target_scale))

# Score actual measurement cells; all robots use the same pre-movement GP.
# No concentration values are read here. Previously observed cells yield zero
# new information because the noiseless simulation deduplicates acquisitions.
def choose_informative_moves(grid, positions, step_length, observed, model, scaled_xy, rng, batch_size=2000):
    
    angles = np.arange(8) * np.pi / 4
    ends = positions[:, None, :] + step_length * np.column_stack((np.cos(angles), np.sin(angles)))
    cells = cell_at(grid, ends)
    valid = np.zeros(cells.shape, dtype=bool)
    for robot, start in enumerate(positions):
        for direction, end in enumerate(ends[robot]):
            valid[robot, direction] = cells[robot, direction] >= 0 and segment_is_navigable(grid, start, end)
    new_cells = np.setdiff1d(np.unique(cells[valid]), observed)
    gains = integrated_variance_reduction(
        model, scaled_xy[grid.valid_mask.ravel()], scaled_xy[new_cells], batch_size)
    gain_by_cell = dict(zip(new_cells.tolist(), gains.tolist()))
    scores = np.full(cells.shape, np.nan)
    for robot, direction in zip(*np.nonzero(valid)):
        scores[robot, direction] = gain_by_cell.get(int(cells[robot, direction]), 0.)
    selected = np.full(len(positions), -1, dtype=int)
    moved = positions.copy()
    blocked = ~valid.any(axis=1)
    decisions, candidates = [], []
    observed_set = set(np.asarray(observed).tolist())
    for robot in range(len(positions)):
        options = np.flatnonzero(valid[robot])
        if len(options):
            best = np.max(scores[robot, options])
            ties = options[np.isclose(scores[robot, options], best, rtol=1e-10, atol=0.)]
            selected[robot] = int(rng.choice(ties))
            moved[robot] = ends[robot, selected[robot]]
        direction = selected[robot]
        cell = int(cell_at(grid, moved[robot]))
        decisions.append(dict(robot=robot + 1, direction_index=int(direction),
                              angle_radians=float(angles[direction]) if direction >= 0 else float("nan"),
                              x=float(moved[robot, 0]), y=float(moved[robot, 1]), cell=cell,
                              gain_mean_variance=float(scores[robot, direction]) if direction >= 0 else 0.,
                              valid_candidates=len(options), already_observed=cell in observed_set,
                              blocked=bool(blocked[robot])))
        for d in range(8):
            candidates.append(dict(robot=robot + 1, direction_index=d, angle_radians=float(angles[d]),
                                   x=float(ends[robot, d, 0]), y=float(ends[robot, d, 1]),
                                   cell=int(cells[robot, d]), navigable=bool(valid[robot, d]),
                                   already_observed=int(cells[robot, d]) in observed_set,
                                   gain_mean_variance=float(scores[robot, d]), selected=d == direction))
    return moved, blocked, decisions, candidates


# Fit after each acquisition round; evaluate both policies on fixed common domains.
# The historical random walk is regenerated with its original RNG/protocol and
# fitted only at requested checkpoints, not replaced by an eight-direction walk.
# No cached experiment is silently accepted without a provenance check.
def run_informative_study(args, grid, coordinate_transform, output_path, coordinate_unit, *,
                          save_artifacts=True, checkpoint_dir=None):
    
    if coordinate_transform is None:
        raise ValueError("IPP requires current-informed coordinates.")
    base = output_path.with_name(f"{output_path.stem}_seed_{args.random_seed}_informative")
    base.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    random_walk = simulate_random_walk(
        grid, args.n_robots, args.n_steps, args.robot_step, args.robot_start_radius,
        args.random_seed, args.robot_start_center, args.robot_deployment,
        args.robot_near_count, args.robot_annulus_inner, args.robot_annulus_outer)
    positions = np.empty_like(random_walk.positions)
    positions[0] = random_walk.positions[0]
    cells = np.empty_like(random_walk.cell_indices)
    cells[0] = random_walk.cell_indices[0]
    rejected = np.zeros_like(random_walk.rejected)
    xy = np.column_stack((grid.x_grid.ravel(), grid.y_grid.ravel()))
    rotated = coordinate_transform.transform(xy)
    scaler = StandardScaler().fit(rotated[grid.valid_mask.ravel()])
    scaled = scaler.transform(rotated)
    rng = np.random.default_rng(np.random.SeedSequence([args.random_seed, 4]))
    optimizer_rows, round_rows, decision_rows, candidate_rows = [], [], [], []
    snapshots, snapshot_fits = {}, {}
    round_theta, round_normalization = [], []
    comparison_counts = set(args.robot_checkpoints) | {args.n_robots}
    map_counts = (comparison_counts if checkpoint_dir is not None else
                  set(range(args.n_robots, args.n_samples + 1, args.n_robots)))
    round_uncertainty = []
    previous_indices = None
    model = None
    print("\n=== Informative motion versus historical random walk ===", flush=True)
    print("Eight feasible headings; full valid grid Z; posterior latent covariance; White + alpha denominator.")
    print("Direct targets + clipping; shared starts; two LML-selected starts per fit; no inducing points.")
    print("Random walk unchanged: continuous random angles, rejected moves stay in place.")
    print("Baseline fits are recomputed at checkpoints; no archived metrics with different masks are reused.")
    for step in range(args.n_steps):
        count = (step + 1) * args.n_robots
        visits = cells[:step + 1].ravel()
        _, first = np.unique(visits, return_index=True)
        indices = visits[np.sort(first)]
        checkpoint = checkpoint_dir / f"ipp_{count:06d}.npz" if checkpoint_dir is not None else None
        if checkpoint is not None and checkpoint.exists():
            cached = load_checkpoint(checkpoint)
            np.testing.assert_array_equal(indices, cached["indices"])
            name, diagnostic = cached["name"], cached["diagnostic"]
            previous_indices, model = indices.copy(), None
            round_theta.append(cached["theta"])
            round_normalization.append((diagnostic.target_mean, diagnostic.target_scale))
            round_rows.append(cached["round_row"])
            round_uncertainty.append(cached["uncertainty"])
            optimizer_rows.extend(cached["optimizer_rows"])
            if count in map_counts:
                snapshots[count], snapshot_fits[count] = cached["raw"], (name, diagnostic)
            if step + 1 < args.n_steps:
                positions[step + 1] = cached["next_positions"]
                cells[step + 1] = cell_at(grid, positions[step + 1])
                rejected[step + 1] = cached["next_rejected"]
            rng.bit_generator.state = cached["rng_state"]
            print(f"IPP N={count}: restored completed round", flush=True)
            continue
        optimizer_start = len(optimizer_rows)
        start = perf_counter()
        refitted = previous_indices is None or not np.array_equal(indices, previous_indices)
        if refitted:
            name, model, diagnostic = fit_robot_model(
                args, rotated, indices, grid.field.ravel()[indices], scaler, count,
                "Informative", optimizer_rows, base, coordinate_unit)
        elif model is None:
            model = restore_model(args, round_theta[-1], scaled[indices], grid.field.ravel()[indices], diagnostic)
        fit_seconds = perf_counter() - start
        previous_indices = indices.copy()
        round_theta.append(model.kernel_.theta.copy())
        round_normalization.append((diagnostic.target_mean, diagnostic.target_scale))
        raw = reconstruct_field(grid, model, scaler, args.prediction_batch_size, "none", False, coordinate_transform)
        round_uncertainty.append(uncertainty_summary(
            raw.std_field, grid.valid_mask, float(model.kernel_.k2.noise_level), diagnostic.target_scale,
            count, "Informative"))
        if count in map_counts:
            snapshots[count] = raw
            snapshot_fits[count] = (name, diagnostic)
        score_seconds = 0.
        if step + 1 < args.n_steps:
            start = perf_counter()
            positions[step + 1], rejected[step + 1], decisions, candidates = choose_informative_moves(
                grid, positions[step], args.robot_step, indices, model, scaled, rng, args.ipp_batch_size)
            cells[step + 1] = cell_at(grid, positions[step + 1])
            score_seconds = perf_counter() - start
            if checkpoint_dir is None:
                decision_rows.extend(dict(acquisition=step + 2, model_measurements=count, **r) for r in decisions)
                candidate_rows.extend(dict(acquisition=step + 2, model_measurements=count, **r) for r in candidates)
                write_rows(base.with_name(base.name + "_decisions.csv"), decision_rows)
                write_rows(base.with_name(base.name + "_candidates.csv"), candidate_rows)
        round_rows.append(dict(acquisition=step + 1, measurements=count, distinct_cells=len(indices),
                               refitted=refitted, initialization=name, lml=diagnostic.final_lml,
                               white_level=float(model.kernel_.k2.noise_level),
                               target_mean=diagnostic.target_mean, target_scale=diagnostic.target_scale,
                               fit_seconds=fit_seconds, scoring_seconds=score_seconds))
        if checkpoint is not None:
            save_checkpoint(checkpoint, raw if count in map_counts else None, name, diagnostic,
                            model.kernel_.theta, optimizer_rows[optimizer_start:], indices=indices,
                            round_row=round_rows[-1], uncertainty=round_uncertainty[-1],
                            rng_state=rng.bit_generator.state,
                            next_positions=positions[step + 1] if step + 1 < args.n_steps else None,
                            next_rejected=rejected[step + 1] if step + 1 < args.n_steps else None)
        write_rows(base.with_name(base.name + "_rounds.csv"), round_rows)
        print(f"IPP N={count}: {len(indices)} cells; {name}; LML={diagnostic.final_lml:.6g}; "
              f"fit={fit_seconds:.2f}s, scoring={score_seconds:.2f}s", flush=True)

    walk = replace(random_walk, positions=positions, cell_indices=cells, rejected=rejected)
    # Future visits and ground-truth peak/bands are used only after planning, for evaluation.
    common = common_unsampled_mask(grid, {"IPP": cells.ravel(), "Random": random_walk.cell_indices.ravel()})
    peak_cell = np.argmax(np.where(grid.valid_mask, grid.field, -np.inf))
    local = grid.valid_mask & (np.linalg.norm(xy - xy[peak_cell], axis=1).reshape(grid.field.shape) <= args.peak_radius)
    if not common.any() or not (common & local).any():
        raise ValueError("No common unobserved cells globally or locally; cannot report paired unseen errors.")
    saved = dict(truth=grid.field, valid=grid.valid_mask, x=grid.x_grid, y=grid.y_grid,
                 positions=positions, visits=cells, rejected=rejected,
                 random_positions=random_walk.positions, random_visits=random_walk.cell_indices,
                 random_rejected=random_walk.rejected, common_unobserved=common, local=local,
                 planning_indices=np.flatnonzero(grid.valid_mask), policy="informative",
                 deployment=walk.deployment, deployment_center=walk.center if walk.center is not None else np.empty(0),
                 annulus_inner=walk.annulus_inner, annulus_outer=walk.annulus_outer, near_count=walk.near_count,
                 coordinate_mean=scaler.mean_, coordinate_scale=scaler.scale_,
                 rotation_degrees=coordinate_transform.angle_degrees, coordinate_unit=coordinate_unit,
                 rotation_center=np.array([coordinate_transform.center_x, coordinate_transform.center_y]),
                 config=json.dumps(vars(args), default=str), checkpoints=np.array(args.robot_checkpoints),
                 map_checkpoints=np.array(sorted(map_counts)), round_theta=np.array(round_theta),
                 round_target_normalization=np.array(round_normalization))
    write_rows(base.with_name(base.name + "_rounds.csv"), round_rows)
    rows, uncertainty_rows, results = [], list(round_uncertainty), {}
    for count in sorted(map_counts):
        for layout, trajectory in (("Informative", walk), ("Random walk", random_walk)):
            if layout == "Random walk" and count not in comparison_counts:
                continue
            indices = observations_up_to(trajectory, count)
            if layout == "Informative":
                raw = snapshots[count]
                name, diagnostic = snapshot_fits[count]
            elif count == args.n_robots:
                # Identical starts, model and normalization: no redundant initial fit.
                raw = snapshots[count]
                name, diagnostic = snapshot_fits[count]
            else:
                checkpoint = checkpoint_dir / f"random_{count:06d}.npz" if checkpoint_dir is not None else None
                if checkpoint is not None and checkpoint.exists():
                    cached = load_checkpoint(checkpoint)
                    np.testing.assert_array_equal(indices, cached["indices"])
                    name, diagnostic, raw = cached["name"], cached["diagnostic"], cached["raw"]
                    optimizer_rows.extend(cached["optimizer_rows"])
                    print(f"Random walk N={count}: restored completed fit", flush=True)
                else:
                    optimizer_start = len(optimizer_rows)
                    name, model, diagnostic = fit_robot_model(
                        args, rotated, indices, grid.field.ravel()[indices], scaler, count,
                        layout, optimizer_rows, base, coordinate_unit)
                    raw = reconstruct_field(grid, model, scaler, args.prediction_batch_size,
                                            "none", False, coordinate_transform)
                    if checkpoint is not None:
                        save_checkpoint(checkpoint, raw, name, diagnostic, model.kernel_.theta,
                                        optimizer_rows[optimizer_start:], indices=indices)
            prediction = np.maximum(raw.mean_field, 0.)
            metrics = prediction_metrics(grid.field, raw.mean_field, prediction, grid.valid_mask, common, local)
            metrics["local_full_rmse"] = float(np.sqrt(np.mean((prediction[local] - grid.field[local])**2)))
            reconstruction = replace(raw, mean_field=prediction, rmse=metrics["full_rmse"],
                                     mse=metrics["full_rmse"]**2, r2=metrics["full_r2"])
            peak = measure_peak(grid, reconstruction, indices, grid.field.ravel()[indices], args.peak_radius)
            results[(layout, count)] = reconstruction
            white_level = next(p.optimized_value for p in diagnostic.hyperparameters if p.name == "k2__noise_level")
            if layout == "Random walk":
                uncertainty_rows.append(uncertainty_summary(
                    raw.std_field, grid.valid_mask, white_level, diagnostic.target_scale, count, layout))
            steps = count // args.n_robots
            row = dict(seed=args.random_seed, checkpoint=count, layout=layout, deployment=walk.deployment,
                       measurements=count, distinct_cells=len(indices), revisits=count - len(indices),
                       rejected_moves=int(trajectory.rejected[:steps].sum()),
                       distance_total=float(np.linalg.norm(np.diff(trajectory.positions[:steps], axis=0), axis=2).sum()),
                       common_global_count=int(common.sum()), common_local_count=int((common & local).sum()),
                       initialization=name, lml=diagnostic.final_lml, **metrics,
                       true_peak=peak.true_max, observed_max=peak.observed_max,
                       prediction_at_true_peak=peak.prediction_at_true_peak, predicted_max=peak.predicted_max,
                       peak_absolute_error=abs(peak.prediction_at_true_peak - peak.true_max),
                       peak_distance=peak.peak_location_error,
                       peak_relative_error_percent=100 * abs(peak.peak_underestimation) / peak.true_max
                       if peak.true_max > 0 else float("nan"),
                       nearest_sensor_distance=peak.nearest_sensor_distance,
                       local_sensor_count=peak.local_sensor_count, true_peak_sampled=peak.true_peak_sampled)
            if count in args.robot_checkpoints:
                rows.append(row)
                write_rows(base.with_name(base.name + "_metrics.csv"), rows)
                print(f"N={count} {layout}: full RMSE={metrics['full_rmse']:.6g}; "
                      f"local full={metrics['local_full_rmse']:.6g}; peak={peak.prediction_at_true_peak:.6g}", flush=True)
            tag = f"{layout.lower().replace(' ', '_')}_{count}"
            saved[tag + "_indices"] = indices
            saved[tag + "_raw_mean"] = raw.mean_field
            saved[tag + "_std"] = raw.std_field
            saved[tag + "_white_level"] = white_level
            saved[tag + "_target_scale"] = diagnostic.target_scale
    uncertainty_rows.sort(key=lambda row: (row["checkpoint"], row["layout"]))
    write_rows(base.with_name(base.name + "_uncertainty.csv"), uncertainty_rows)
    write_rows(base.with_name(base.name + "_optimizer.csv"), optimizer_rows)
    fields_path = base.with_name(base.name + "_fields.npz")
    np.savez_compressed(fields_path, **saved)
    if save_artifacts:
        paths = plot_robot_results(grid, walk, results, rows, args, base, coordinate_unit, None,
                                   robot_label="Informative", reference_label="Random walk", reference_walk=random_walk)
        paths.append(plot_robot_uncertainty(uncertainty_rows, base.with_name(base.name + "_uncertainty.png"),
                                           f"{walk.deployment} | seed {args.random_seed}"))
        for path in paths:
            print(f"Saved: {path}")
    print(f"Saved CSV diagnostics and replay fields: {fields_path}")
    return rows, walk, common


def refresh_informative_outputs(archive, output_dir):
    """Complete archived IPP maps at frozen theta/normalization, without optimization or motion."""
    import csv
    from pathlib import Path
    from shutil import copy2
    from types import SimpleNamespace
    from sklearn.gaussian_process import GaussianProcessRegressor
    from pollutant_gp.model import build_kernel
    from pollutant_gp.robot.trajectories import RobotWalk
    from pollutant_gp.spatial import RotationTransform, build_rotation_transform
    from pollutant_gp.types import GridData

    archive, output_dir = Path(archive), Path(output_dir)
    if output_dir.resolve() == archive.parent.resolve():
        raise ValueError("Use a different output directory to preserve the original experiment.")
    if not archive.name.endswith("_fields.npz"):
        raise ValueError("Expected an IPP _fields.npz archive.")
    source_base = archive.with_name(archive.name.removesuffix("_fields.npz"))
    with np.load(archive, allow_pickle=False) as stored:
        data = {name: stored[name] for name in stored.files}
    if str(data.get("policy", "")) != "informative":
        raise ValueError("Replay requires an informative-study archive.")
    args = SimpleNamespace(**json.loads(str(data["config"])))
    args.nc_file = Path(args.nc_file)
    args.robot_gif = True
    grid = GridData(data["truth"], data["valid"], data["x"], data["y"], "x", "y", None)
    center = data["deployment_center"] if data["deployment_center"].size else None
    walk = RobotWalk(data["positions"], data["visits"], data["rejected"], center,
                     str(data["deployment"]), float(data["annulus_inner"]),
                     float(data["annulus_outer"]), int(data["near_count"]))
    baseline = replace(walk, positions=data["random_positions"], cell_indices=data["random_visits"],
                       rejected=data["random_rejected"])
    transform = build_rotation_transform(grid, float(data["rotation_degrees"]), "archived current")
    if "rotation_center" in data:
        transform = RotationTransform(transform.angle_degrees, *data["rotation_center"], "archived current")
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = data["coordinate_mean"], data["coordinate_scale"]
    scaler.var_, scaler.n_features_in_ = scaler.scale_**2, 2
    xy = np.column_stack((grid.x_grid.ravel(), grid.y_grid.ravel()))
    scaled = scaler.transform(transform.transform(xy))
    kernel = build_kernel(2, "anisotropic", args.length_scale_lower_bound, args.length_scale_upper_bound,
                          args.noise_level_initial, args.noise_level_lower_bound, args.noise_level_upper_bound,
                          constant_value_initial=.5, length_scale_initial=1., second_length_scale_initial=.1)

    def read_csv(suffix):
        with source_base.with_name(source_base.name + suffix).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    metrics = read_csv("_metrics.csv")
    optimizer = read_csv("_optimizer.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / source_base.name
    counts = list(range(args.n_robots, args.n_samples + 1, args.n_robots))
    if len(data["round_theta"]) != len(counts) or len(data["round_target_normalization"]) != len(counts):
        raise ValueError("Archive does not contain every IPP fit.")
    results, uncertainty = {}, []
    for step, count in enumerate(counts):
        tag = f"informative_{count}"
        normalization = data["round_target_normalization"][step]
        frozen_kernel = kernel.clone_with_theta(data["round_theta"][step])
        missing = tag + "_raw_mean" not in data
        # Validate restoration against the first and final stored fits before trusting missing maps.
        if missing or count in (counts[0], counts[-1]):
            indices = observations_up_to(walk, count)
            model = GaussianProcessRegressor(kernel=frozen_kernel, alpha=1e-10, optimizer=None, normalize_y=False)
            targets = (grid.field.ravel()[indices].astype(float) - normalization[0]) / normalization[1]
            model.fit(scaled[indices], targets)
            raw = reconstruct_field(grid, model, scaler, args.prediction_batch_size, "none", False,
                                    transform, tuple(normalization))
            if not missing:
                np.testing.assert_allclose(raw.mean_field, data[tag + "_raw_mean"], rtol=1e-8, atol=1e-9)
                np.testing.assert_allclose(raw.std_field, data[tag + "_std"], rtol=1e-8, atol=1e-9)
            else:
                data[tag + "_indices"] = indices
                data[tag + "_raw_mean"], data[tag + "_std"] = raw.mean_field, raw.std_field
            print(f"Replay N={count}: {'predicted missing map' if missing else 'verified stored fit'}; optimizer=None", flush=True)
        data[tag + "_white_level"] = float(frozen_kernel.k2.noise_level)
        data[tag + "_target_scale"] = float(normalization[1])
        results[("Informative", count)] = SimpleNamespace(
            mean_field=np.maximum(data[tag + "_raw_mean"], 0.), std_field=data[tag + "_std"])
        uncertainty.append(uncertainty_summary(data[tag + "_std"], grid.valid_mask,
                                              data[tag + "_white_level"], normalization[1], count, "Informative"))
        tag = f"random_walk_{count}"
        if tag + "_std" in data:
            if tag + "_white_level" in data:
                white, scale = float(data[tag + "_white_level"]), float(data[tag + "_target_scale"])
            elif count == args.n_robots:
                white, scale = float(frozen_kernel.k2.noise_level), float(normalization[1])
            else:
                metric = next(r for r in metrics if r["layout"] == "Random walk" and int(r["checkpoint"]) == count)
                fit = next(r for r in optimizer if r["layout"] == "Random walk" and int(r["checkpoint"]) == count
                           and r["initialization"] == metric["initialization"] and r["success"] == "True")
                white, scale = float(fit["k2__noise_level"]), float(fit["target_scale"])
            data[tag + "_white_level"], data[tag + "_target_scale"] = white, scale
            results[("Random walk", count)] = SimpleNamespace(
                mean_field=np.maximum(data[tag + "_raw_mean"], 0.), std_field=data[tag + "_std"])
            uncertainty.append(uncertainty_summary(data[tag + "_std"], grid.valid_mask, white, scale, count, "Random walk"))
    data["map_checkpoints"] = np.array(counts)
    data["replayed_from"] = str(archive.resolve())
    np.savez_compressed(base.with_name(base.name + "_fields.npz"), **data)
    write_rows(base.with_name(base.name + "_uncertainty.csv"), uncertainty)
    # Keep experimental metrics and optimization logs byte-for-byte, rather than recomputing them.
    for suffix in ("_metrics.csv", "_optimizer.csv", "_rounds.csv", "_decisions.csv", "_candidates.csv"):
        source = source_base.with_name(source_base.name + suffix)
        if source.exists():
            copy2(source, base.with_name(base.name + suffix))
    for row in metrics:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except ValueError:
                pass
    unit = str(data["coordinate_unit"])
    paths = plot_robot_results(grid, walk, results, metrics, args, base, unit, None,
                               robot_label="Informative", reference_label="Random walk", reference_walk=baseline)
    paths.append(plot_robot_uncertainty(uncertainty, base.with_name(base.name + "_uncertainty.png"),
                                       f"{walk.deployment} | seed {args.random_seed}"))
    for path in paths:
        print(f"Saved: {path}")
    return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Refresh IPP maps/GIF and mean variance from archived fits; no optimization.")
    parser.add_argument("--replay-fields", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory; original results are preserved.")
    replay_args = parser.parse_args()
    refresh_informative_outputs(replay_args.replay_fields, replay_args.output_dir)
