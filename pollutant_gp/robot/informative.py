# Closed-loop, independent one-step IPP versus the unchanged continuous random walk.

from dataclasses import replace
import json
from time import perf_counter

import numpy as np
from sklearn.preprocessing import StandardScaler

from pollutant_gp.model import integrated_variance_reduction
from pollutant_gp.peak import measure_peak
from pollutant_gp.peak_sampling import common_unsampled_mask
from pollutant_gp.positivity import prediction_metrics
from pollutant_gp.reconstruction import reconstruct_field
from pollutant_gp.robot.study import fit_robot_model, write_rows
from pollutant_gp.robot.trajectories import (
    cell_at, observations_up_to, segment_is_navigable, simulate_random_walk,
)
from pollutant_gp.robot.visualization import plot_robot_results

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
def run_informative_study(args, grid, coordinate_transform, output_path, coordinate_unit, *, save_artifacts=True):
    
    if coordinate_transform is None:
        raise ValueError("IPP requires current-informed coordinates.")
    base = output_path.with_name(f"{output_path.stem}_seed_{args.random_seed}_informative")
    base.parent.mkdir(parents=True, exist_ok=True)
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
    map_counts = set(args.robot_checkpoints) | {args.n_robots}
    previous_indices = None
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
        start = perf_counter()
        refitted = previous_indices is None or not np.array_equal(indices, previous_indices)
        if refitted:
            name, model, diagnostic = fit_robot_model(
                args, rotated, indices, grid.field.ravel()[indices], scaler, count,
                "Informative", optimizer_rows, base, coordinate_unit)
        fit_seconds = perf_counter() - start
        previous_indices = indices.copy()
        round_theta.append(model.kernel_.theta.copy())
        round_normalization.append((diagnostic.target_mean, diagnostic.target_scale))
        if count in map_counts:
            snapshots[count] = reconstruct_field(
                grid, model, scaler, args.prediction_batch_size, "none", False, coordinate_transform)
            snapshot_fits[count] = (name, diagnostic)
        score_seconds = 0.
        if step + 1 < args.n_steps:
            start = perf_counter()
            positions[step + 1], rejected[step + 1], decisions, candidates = choose_informative_moves(
                grid, positions[step], args.robot_step, indices, model, scaled, rng, args.ipp_batch_size)
            cells[step + 1] = cell_at(grid, positions[step + 1])
            score_seconds = perf_counter() - start
            decision_rows.extend(dict(acquisition=step + 2, model_measurements=count, **r) for r in decisions)
            candidate_rows.extend(dict(acquisition=step + 2, model_measurements=count, **r) for r in candidates)
            write_rows(base.with_name(base.name + "_decisions.csv"), decision_rows)
            write_rows(base.with_name(base.name + "_candidates.csv"), candidate_rows)
        round_rows.append(dict(acquisition=step + 1, measurements=count, distinct_cells=len(indices),
                               refitted=refitted, initialization=name, lml=diagnostic.final_lml,
                               white_level=float(model.kernel_.k2.noise_level),
                               target_mean=diagnostic.target_mean, target_scale=diagnostic.target_scale,
                               fit_seconds=fit_seconds, scoring_seconds=score_seconds))
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
                 config=json.dumps(vars(args), default=str), checkpoints=np.array(args.robot_checkpoints),
                 map_checkpoints=np.array(sorted(map_counts)), round_theta=np.array(round_theta),
                 round_target_normalization=np.array(round_normalization))
    rows, results = [], {}
    for count in sorted(map_counts):
        for layout, trajectory in (("Informative", walk), ("Random walk", random_walk)):
            indices = observations_up_to(trajectory, count)
            if layout == "Informative":
                raw = snapshots[count]
                name, diagnostic = snapshot_fits[count]
            elif count == args.n_robots:
                # Identical starts, model and normalization: no redundant initial fit.
                raw = snapshots[count]
                name, diagnostic = snapshot_fits[count]
            else:
                name, model, diagnostic = fit_robot_model(
                    args, rotated, indices, grid.field.ravel()[indices], scaler, count,
                    layout, optimizer_rows, base, coordinate_unit)
                raw = reconstruct_field(grid, model, scaler, args.prediction_batch_size,
                                        "none", False, coordinate_transform)
            prediction = np.maximum(raw.mean_field, 0.)
            metrics = prediction_metrics(grid.field, raw.mean_field, prediction, grid.valid_mask, common, local)
            metrics["local_full_rmse"] = float(np.sqrt(np.mean((prediction[local] - grid.field[local])**2)))
            reconstruction = replace(raw, mean_field=prediction, rmse=metrics["full_rmse"],
                                     mse=metrics["full_rmse"]**2, r2=metrics["full_r2"])
            peak = measure_peak(grid, reconstruction, indices, grid.field.ravel()[indices], args.peak_radius)
            results[(layout, count)] = reconstruction
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
    fields_path = base.with_name(base.name + "_fields.npz")
    np.savez_compressed(fields_path, **saved)
    if save_artifacts:
        paths = plot_robot_results(grid, walk, results, rows, args, base, coordinate_unit, None,
                                   robot_label="Informative", reference_label="Random walk", reference_walk=random_walk)
        for path in paths:
            print(f"Saved: {path}")
    print(f"Saved CSV diagnostics and replay fields: {fields_path}")
    return rows, walk, common
