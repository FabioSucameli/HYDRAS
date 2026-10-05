"""Compare archived robot deployments with one independent Pure Uniform reference."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import xarray as xr

from pollutant_gp.data import prepare_grid_data
from pollutant_gp.model import build_kernel, fit_gaussian_process
from pollutant_gp.peak_kernel import INITIALIZATIONS, select_converged_fit
from pollutant_gp.robot.study import write_rows
from pollutant_gp.robot.trajectories import observations_up_to, simulate_random_walk, uniform_reference
from pollutant_gp.spatial import build_rotation_transform


LAYOUTS = ("Central", "Uniform-domain", "Corona", "Pure Uniform")
METRICS = ("full_rmse", "local_full_rmse", "prediction_at_true_peak",
           "peak_relative_error_percent", "peak_distance", "distinct_cells", "local_sensor_count")
COMMON_SETTINGS = ("variable", "time_index", "time_dim", "x_dim", "y_dim", "x_coordinate",
                   "y_coordinate", "n_robots", "n_steps", "robot_step", "peak_radius",
                   "length_scale_lower_bound", "length_scale_upper_bound", "noise_level_initial",
                   "noise_level_lower_bound", "noise_level_upper_bound", "target_transform",
                   "n_restarts", "noise_std", "clip_negative", "optimizer_seed")


# Read numeric CSVs as strings so parameter precision is preserved until conversion.
def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


# Fail closed on replay discrepancies rather than silently mixing different experiments.
def require_close(actual, expected, name, rtol=1e-6, atol=1e-8):
    if not np.isfinite(actual) or not np.isclose(actual, float(expected), rtol=rtol, atol=atol):
        raise ValueError(f"Replay mismatch for {name}: recovered={actual}, archived={expected}")


# Use a separate reproducible stream, with no inherited or excluded robot starts.
def pure_uniform_indices(grid, seed, budget):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    return rng.choice(np.flatnonzero(grid.valid_mask), budget, replace=False)


# Recover natural parameter values in the kernel's actual theta order.
def parameter_names(kernel):
    return [h.name if h.n_elements == 1 else f"{h.name}[{i}]"
            for h in kernel.hyperparameters if not h.fixed for i in range(h.n_elements)]


# Factor the archived covariance without calling any hyperparameter optimizer.
def restore_gp(coordinates, values, scaler, config, archived):
    kernel = build_kernel(2, "anisotropic", config["length_scale_lower_bound"],
                          config["length_scale_upper_bound"], config["noise_level_initial"],
                          config["noise_level_lower_bound"], config["noise_level_upper_bound"],
                          constant_value_initial=.5, length_scale_initial=1., second_length_scale_initial=.1)
    parameters = np.array([float(archived[name]) for name in parameter_names(kernel)])
    if not np.isfinite(parameters).all() or np.any(parameters <= 0):
        raise ValueError("Invalid archived kernel parameters.")
    kernel = kernel.clone_with_theta(np.log(parameters))
    model = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=1e-10, normalize_y=True)
    model.fit(scaler.transform(coordinates), values)
    require_close(float(model._y_train_mean), archived["target_mean"], "target mean", 1e-10, 1e-12)
    require_close(float(model._y_train_std), archived["target_scale"], "target scale", 1e-10, 1e-12)
    require_close(model.log_marginal_likelihood_value_, archived["lml"], "LML", 1e-8, 1e-6)
    return model


# Fit the same two controlled starts; selection uses sensor LML, never map errors.
def fit_pure(coordinates, values, scaler, config, seed, checkpoint):
    candidates, rows = [], []
    for name, first, second, amplitude in INITIALIZATIONS["Double"]:
        model, _, diag = fit_gaussian_process(
            coordinates, values, "anisotropic", config["length_scale_lower_bound"],
            config["length_scale_upper_bound"], config["noise_level_initial"],
            config["noise_level_lower_bound"], config["noise_level_upper_bound"], "none", 0,
            seed if config["optimizer_seed"] is None else config["optimizer_seed"],
            constant_value_initial=amplitude, length_scale_initial=first,
            second_length_scale_initial=second, coordinate_scaler=scaler)
        run = diag.optimizer_runs[diag.selected_run_index]
        rows.append(dict(seed=seed, checkpoint=checkpoint, initialization=name, success=run.success,
                         status=run.status, message=run.message, lml=diag.final_lml,
                         theta=json.dumps(model.kernel_.theta.tolist()),
                         target_mean=diag.target_mean, target_scale=diag.target_scale))
        candidates.append((name, model, scaler, diag))
    name, model, _, _ = select_converged_fit(candidates)
    return model, name, rows


# Predict only means in bounded batches; uncertainty is not needed for these RMSE curves.
def predict_map(model, scaled, valid, batch_size):
    indices = np.flatnonzero(valid)
    prediction = np.full(valid.size, np.nan)
    for start in range(0, len(indices), batch_size):
        cells = indices[start:start + batch_size]
        prediction[cells] = np.maximum(model.predict(scaled[cells]), 0.)
    if not np.isfinite(prediction[indices]).all():
        raise ValueError("Nonfinite reconstructed concentration.")
    return prediction.reshape(valid.shape)


# Score fixed global/local domains including training cells, independently of deployment masks.
def score(grid, prediction, indices, local):
    peak = np.argmax(np.where(grid.valid_mask, grid.field, -np.inf))
    maximum = np.argmax(np.where(grid.valid_mask, prediction, -np.inf))
    xy = np.column_stack((grid.x_grid.ravel(), grid.y_grid.ravel()))
    error = prediction - grid.field
    return dict(full_rmse=float(np.sqrt(np.mean(error[grid.valid_mask]**2))),
                local_full_rmse=float(np.sqrt(np.mean(error[local]**2))),
                prediction_at_true_peak=float(prediction.ravel()[peak]),
                peak_relative_error_percent=float(100 * abs(error.ravel()[peak]) / grid.field.ravel()[peak]),
                peak_distance=float(np.linalg.norm(xy[maximum] - xy[peak])),
                distinct_cells=len(indices), local_sensor_count=int(local.ravel()[indices].sum()))


# Aggregate complete paired seeds only, with an undefined standard deviation for a one-seed smoke test.
def summarize(rows, checkpoints):
    grouped = {}
    expected = {(n, layout) for n in checkpoints for layout in LAYOUTS}
    for row in rows:
        group = grouped.setdefault(int(row["seed"]), {})
        key = (int(row["checkpoint"]), row["layout"])
        if key in group:
            raise ValueError("Duplicate comparison row.")
        group[key] = row
    if not grouped or any(set(group) != expected for group in grouped.values()):
        raise ValueError("Aggregation requires complete paired seeds and checkpoints.")
    summary = []
    for count in checkpoints:
        for layout in LAYOUTS:
            for metric in METRICS:
                values = np.array([g[(count, layout)][metric] for g in grouped.values()], dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite metric: {metric}")
                summary.append(dict(checkpoint=count, layout=layout, metric=metric, n_seeds=len(values),
                                    mean=float(values.mean()),
                                    std=float(values.std(ddof=1)) if len(values) > 1 else float("nan")))
    return summary


# Show the four strategies on identical evaluation domains, without overlapping filled bands.
def plot_comparison(summary, output):
    styles = (("#8c564b", "o", "-"), ("#0072B2", "s", "--"),
              ("#009E73", "^", "-."), ("#CC79A7", "D", ":"))
    for metric, name in (("full_rmse", "global"), ("local_full_rmse", "local")):
        fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
        for layout, (color, marker, line) in zip(LAYOUTS, styles):
            data = [r for r in summary if r["layout"] == layout and r["metric"] == metric]
            ax.errorbar([r["checkpoint"] for r in data], [r["mean"] for r in data],
                        yerr=[r["std"] for r in data] if data[0]["n_seeds"] > 1 else None,
                        label=layout, color=color, marker=marker, linestyle=line, capsize=3, linewidth=1.7)
        n = summary[0]["n_seeds"]
        ax.set(title=f"Deployment comparison | {name} reconstruction | {n} seed(s)\n"
                     + ("Mean +/- 1 sample std" if n > 1 else "Smoke test: standard deviation unavailable"),
               xlabel="Acquisition budget", ylabel="RMSE (observed cells included)")
        ax.set_xticks(sorted({r["checkpoint"] for r in summary}))
        ax.grid(axis="y", alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.savefig(output / f"{name}_rmse.png", dpi=180)
        plt.close(fig)


# Check protocol identity before using any archive as a controlled comparison.
def load_archives(args):
    sources = (("Central", "central", args.central_dir),
               ("Uniform-domain", "uniform-domain", args.domain_dir),
               ("Corona", "peak-annulus", args.annulus_dir))
    archives = []
    for label, mode, root in sources:
        manifest = json.loads((root / "configuration.json").read_text(encoding="utf-8"))
        config = manifest["configuration"]
        if config.get("robot_deployment", "central") != mode:
            raise ValueError(f"Wrong deployment archive: {root}")
        if archives:
            reference = archives[0][2]
            for key in COMMON_SETTINGS:
                if config[key] != reference["configuration"][key]:
                    raise ValueError(f"Incompatible protocol: {key} in {root}")
            for key in ("selected_time", "coordinate_unit", "rotation_degrees"):
                if manifest[key] != reference[key]:
                    raise ValueError(f"Incompatible archive metadata: {key}")
        if (config["target_transform"] != "none" or not config["clip_negative"]
                or config["noise_std"] != 0 or config["n_restarts"] != 0):
            raise ValueError("Only the noiseless direct + clipping, zero-restart protocol is supported.")
        if not set(args.checkpoints).issubset(config["robot_checkpoints"]):
            raise ValueError("Requested checkpoints are absent from an archive.")
        metrics = read_rows(root / "metrics.csv")
        selected = {(int(r["seed"]), int(r["checkpoint"])): r for r in metrics if r["layout"] == "Robot"}
        if len(selected) != sum(r["layout"] == "Robot" for r in metrics):
            raise ValueError("Duplicate Robot rows in archive.")
        for seed in args.seeds:
            for count in args.checkpoints:
                if (seed, count) not in selected:
                    raise ValueError(f"Missing Robot result for {label}, seed={seed}, N={count}")
        archives.append((label, root, manifest, selected))
    return archives


# Recover all three robot fits and optimize only Pure Uniform, saving each completed seed.
def run(args):
    archives = load_archives(args)
    c = archives[0][2]["configuration"]
    with xr.open_dataset(args.nc_file) as ds:
        grid = prepare_grid_data(ds, c["variable"], c["time_dim"], c["time_index"],
                                 c["y_dim"], c["x_dim"], c["y_coordinate"], c["x_coordinate"])
    if grid.selected_time_label != archives[0][2]["selected_time"]:
        raise ValueError("Snapshot timestamp differs from the archives.")
    xy = np.column_stack((grid.x_grid.ravel(), grid.y_grid.ravel()))
    transform = build_rotation_transform(grid, archives[0][2]["rotation_degrees"], "archived current-informed")
    rotated = transform.transform(xy)
    scaler = StandardScaler().fit(rotated[grid.valid_mask.ravel()])
    scaled = scaler.transform(rotated)
    peak = np.argmax(np.where(grid.valid_mask, grid.field, -np.inf))
    if grid.field.ravel()[peak] <= 0:
        raise ValueError("A positive reference peak is required.")
    local = grid.valid_mask & (np.linalg.norm(xy - xy[peak], axis=1).reshape(grid.field.shape) <= c["peak_radius"])
    budget = c["n_robots"] * c["n_steps"]
    provenance = dict(seeds=args.seeds, checkpoints=args.checkpoints, archives=[a[2] for a in archives],
                      archive_paths=[str(a[1].resolve()) for a in archives],
                      field_sha256=hashlib.sha256(grid.field.tobytes() + grid.valid_mask.tobytes()
                                                  + xy.tobytes()).hexdigest(),
                      evaluation="full valid domain and full peak disk, INCLUDING observed cells",
                      robot_replay="local predictions only; other reported metrics copied from archives",
                      pure_uniform_rng="SeedSequence([seed, 3]); one full-budget sample, nested prefixes",
                      replay_tolerance="LML rtol=1e-8 atol=1e-6; map metrics rtol=1e-6 atol=1e-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "configuration.json"
    if any(args.output_dir.iterdir()):
        if not args.resume or not manifest_path.exists() or json.loads(manifest_path.read_text()) != provenance:
            raise ValueError("Output must be empty, or use --resume with the identical configuration.")
    manifest_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    all_rows = []
    for seed in args.seeds:
        destination = args.output_dir / f"seed_{seed:04d}"
        destination.mkdir(exist_ok=True)
        completed_path = destination / "complete.csv"
        if args.resume and completed_path.exists():
            seed_rows = read_rows(completed_path)
            summarize(seed_rows, args.checkpoints)
            print(f"Seed {seed}: reused completed comparison", flush=True)
        else:
            seed_rows, audits = [], []
            print(f"Seed {seed}: recovering Robot fits (optimizer disabled, local predictions only)", flush=True)
            with (destination / "run.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                for label, root, manifest, old in archives:
                    cfg = manifest["configuration"]
                    walk = simulate_random_walk(grid, cfg["n_robots"], cfg["n_steps"], cfg["robot_step"],
                                                cfg["robot_start_radius"], seed, cfg["robot_start_center"],
                                                cfg.get("robot_deployment", "central"), cfg.get("robot_near_count", 3),
                                                cfg.get("robot_annulus_inner", 50), cfg.get("robot_annulus_outer", 150))
                    uniform = uniform_reference(grid, walk.cell_indices[0], budget, seed)
                    common = grid.valid_mask.copy()
                    common.ravel()[np.concatenate((walk.cell_indices.ravel(), uniform))] = False
                    files = list((root / f"seed_{seed:04d}").glob("*_optimizer.csv"))
                    if len(files) != 1:
                        raise ValueError(f"Expected one optimizer CSV for {label}, seed {seed}")
                    optimizer = read_rows(files[0])
                    for count in args.checkpoints:
                        original = old[(seed, count)]
                        candidates = [r for r in optimizer if r["layout"] == "Robot" and int(r["checkpoint"]) == count
                                      and r["initialization"] == original["initialization"]]
                        if len(candidates) != 1 or candidates[0]["success"] != "True":
                            raise ValueError(f"Missing converged selected optimizer row: {label}/{seed}/{count}")
                        archived = candidates[0]
                        indices = observations_up_to(walk, count)
                        model = restore_gp(rotated[indices], grid.field.ravel()[indices], scaler, cfg, archived)
                        prediction = predict_map(model, scaled, local, args.batch_size)
                        metrics = {key: float(original[key]) for key in METRICS if key != "local_full_rmse"}
                        metrics.update(distinct_cells=int(original["distinct_cells"]),
                                       local_sensor_count=int(original["local_sensor_count"]))
                        checks = dict(lml=model.log_marginal_likelihood_value_,
                                      prediction_at_true_peak=float(prediction.ravel()[peak]),
                                      distinct_cells=len(indices), local_sensor_count=int(local.ravel()[indices].sum()))
                        error = prediction - grid.field
                        metrics["local_full_rmse"] = float(np.sqrt(np.mean(error[local]**2)))
                        checks["unseen_local_rmse"] = float(np.sqrt(np.mean(error[common & local]**2)))
                        for key, actual in checks.items():
                            require_close(actual, original[key], f"{label}/{seed}/{count}/{key}",
                                          1e-8 if key == "lml" else 1e-6, 1e-6 if key == "lml" else 1e-8)
                            audits.append(dict(seed=seed, checkpoint=count, layout=label, metric=key,
                                               archived=float(original[key]), recovered=actual,
                                               difference=actual-float(original[key])))
                        seed_rows.append(dict(seed=seed, checkpoint=count, layout=label,
                                              initialization=original["initialization"], lml=model.log_marginal_likelihood_value_,
                                              **metrics))
                write_rows(destination / "replay_checks.csv", audits)
            print(f"Seed {seed}: replay checks passed; fitting Pure Uniform", flush=True)
            indices = pure_uniform_indices(grid, seed, budget)
            optimizer_rows = []
            with (destination / "run.log").open("a", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                for count in args.checkpoints:
                    selected = indices[:count]
                    model, name, diagnostics = fit_pure(rotated[selected], grid.field.ravel()[selected], scaler, c, seed, count)
                    optimizer_rows.extend(diagnostics)
                    write_rows(destination / "pure_optimizer.csv", optimizer_rows)
                    prediction = predict_map(model, scaled, grid.valid_mask, args.batch_size)
                    seed_rows.append(dict(seed=seed, checkpoint=count, layout="Pure Uniform", initialization=name,
                                          lml=model.log_marginal_likelihood_value_, **score(grid, prediction, selected, local)))
            summarize(seed_rows, args.checkpoints)
            write_rows(completed_path, seed_rows)
        all_rows.extend(seed_rows)
        summary = summarize(all_rows, args.checkpoints)
        write_rows(args.output_dir / "metrics.csv", all_rows)
        write_rows(args.output_dir / "summary.csv", summary)
        write_rows(args.output_dir / "final_summary.csv", [r for r in summary if r["checkpoint"] == args.checkpoints[-1]])
        plot_comparison(summary, args.output_dir)
        print(f"Seed {seed}: complete ({len(all_rows)//(4*len(args.checkpoints))}/{len(args.seeds)})", flush=True)
    print(f"Saved comparison: {args.output_dir}", flush=True)


# Keep this archive-based experiment independent of the main CLI and original output folders.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nc-file", type=Path, default=Path("CL02_V1_SRC131_Conc_10mGrid.nc"))
    parser.add_argument("--central-dir", type=Path, default=Path("outputs/robot_multiseed_100seeds/robot_multiseed"))
    parser.add_argument("--domain-dir", type=Path, default=Path("outputs/robot_domain_100seeds/robot_domain_multiseed"))
    parser.add_argument("--annulus-dir", type=Path, default=Path("outputs/robot_peak_100seeds/robot_annulus_multiseed"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 101)))
    parser.add_argument("--checkpoints", type=int, nargs="+", default=list(range(100, 801, 100)))
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robot_deployment_comparison"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds) or args.batch_size <= 0
            or args.checkpoints != sorted(set(args.checkpoints)) or min(args.checkpoints) <= 0):
        parser.error("Seeds must be distinct/nonnegative; checkpoints strictly increasing/positive; batch size positive.")
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
