"""Controlled final-budget sensitivity to random-walk step length."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import xarray as xr

from pollutant_gp.current import compute_current_orientation
from pollutant_gp.data import DEFAULT_CONCENTRATION_VARIABLE, prepare_grid_data
from pollutant_gp.model import fit_gaussian_process
from pollutant_gp.peak_kernel import INITIALIZATIONS, select_converged_fit
from pollutant_gp.peak_sampling import common_unsampled_mask
from pollutant_gp.reconstruction import reconstruct_field
from pollutant_gp.robot.proximity import write_csv
from pollutant_gp.robot.trajectories import measure_robot_proximity, observations_up_to, simulate_random_walk
from pollutant_gp.spatial import build_rotation_transform


GEOMETRY_METRICS = ("distinct_cells", "rejected_percent", "final_radius_mean", "close_10_percent")
GP_METRICS = ("rmse_global", "rmse_local", "peak_error_percent")


# Summarize each step on the same eligible seed set, preserving paired comparisons.
def aggregate(rows, metrics, steps, seeds):
    result = []
    for step in steps:
        selected = [r for r in rows if r["step"] == step and r["seed"] in seeds]
        if len(selected) != len(seeds) or len(seeds) < 2:
            raise ValueError("At least two complete, paired seeds are required.")
        for metric in metrics:
            values = np.array([r[metric] for r in selected], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite values in {metric}.")
            result.append(dict(step=step, metric=metric, n_seeds=len(values),
                               mean=float(values.mean()), std=float(values.std(ddof=1))))
    return result


# Couple starts and angle streams through a common seed; exclude the union of observed cells.
def make_step_walks(grid, seed, steps):
    walks = {step: simulate_random_walk(grid, 20, 40, step, 150., seed) for step in steps}
    starts = next(iter(walks.values())).positions[0]
    if not all(np.array_equal(w.positions[0], starts) for w in walks.values()):
        raise AssertionError("Step comparisons must share deployment positions.")
    indices = {step: observations_up_to(walk, 800) for step, walk in walks.items()}
    return walks, indices, common_unsampled_mask(grid, indices)


# Fit exactly the two existing double-RBF starts and select by converged sensor LML.
def fit_step(grid, transform, scaler, coordinates, indices, seed, step, common, local, peak_index):
    candidates, diagnostics = [], []
    values = grid.field.ravel()[indices]
    for name, first, second, amplitude in INITIALIZATIONS["Double"]:
        with warnings.catch_warnings(record=True) as captured:
            model, _, diagnostic = fit_gaussian_process(
                coordinates[indices], values, "anisotropic", .02, 100., 1e-4, 1e-6, 10.,
                "none", 0, seed, constant_value_initial=amplitude, length_scale_initial=first,
                second_length_scale_initial=second, coordinate_scaler=scaler)
        run = diagnostic.optimizer_runs[diagnostic.selected_run_index]
        diagnostics.append(dict(seed=seed, step=step, initialization=name, success=run.success,
                                status=run.status, message=run.message, lml=diagnostic.final_lml,
                                iterations=run.iterations, evaluations=run.function_evaluations,
                                theta=json.dumps(run.optimized_theta.tolist()),
                                target_mean=diagnostic.target_mean, target_scale=diagnostic.target_scale,
                                warnings=" | ".join(str(w.message) for w in captured)))
        candidates.append((name, model, scaler, diagnostic))
    row = dict(seed=seed, step=step, distinct_cells=len(indices),
               common_global_count=int(common.sum()), common_local_count=int((common & local).sum()))
    try:
        name, model, _, diagnostic = select_converged_fit(candidates)
    except RuntimeError:
        return dict(**row, success=False, initialization="none", lml=float("nan"),
                    rmse_global=float("nan"), rmse_local=float("nan"),
                    peak_error_percent=float("nan"), prediction_at_peak=float("nan")), diagnostics, None
    raw = reconstruct_field(grid, model, scaler, 2000, "none", False, transform)
    prediction = np.maximum(raw.mean_field, 0.)
    error = prediction - grid.field
    true_peak = grid.field.ravel()[peak_index]
    row.update(success=True, initialization=name, lml=diagnostic.final_lml,
               rmse_global=float(np.sqrt(np.mean(error[common] ** 2))),
               rmse_local=float(np.sqrt(np.mean(error[common & local] ** 2))),
               peak_error_percent=float(100 * abs(error.ravel()[peak_index]) / true_peak),
               prediction_at_peak=float(prediction.ravel()[peak_index]))
    return row, diagnostics, prediction


# Save per-seed results incrementally; no intermediate-checkpoint or uniform-baseline fits.
def run_study(args):
    with xr.open_dataset(args.nc_file) as ds:
        grid = prepare_grid_data(ds, DEFAULT_CONCENTRATION_VARIABLE, "time", args.time_index,
                                 "y", "x", "y", "x")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), robots=20, acquisitions=40, deployment_radius=150,
                  length_scale_bounds=[.02, 100.], white_initial=1e-4, white_bounds=[1e-6, 10.],
                  target="none + clipping", restarts=0, current_average_hours=12,
                  initializations=INITIALIZATIONS["Double"])
    (args.output_dir / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    geometry, gp, optimizers = [], [], []
    cached = {}
    xy = np.column_stack((grid.x_grid.ravel(), grid.y_grid.ravel()))
    peak_index = np.argmax(np.where(grid.valid_mask, grid.field, -np.inf))
    local = grid.valid_mask & (np.linalg.norm(xy - xy[peak_index], axis=1).reshape(grid.field.shape) <= 150.)
    for seed in args.geometry_seeds:
        walks, indices, common = make_step_walks(grid, seed, args.steps)
        for step, walk in walks.items():
            _, proximity = measure_robot_proximity(walk)
            geometry.append(dict(seed=seed, step=step, distinct_cells=len(indices[step]),
                                 rejected_percent=100 * float(walk.rejected[1:].mean()),
                                 final_radius_mean=float(np.linalg.norm(walk.positions[-1] - walk.center, axis=1).mean()),
                                 close_10_percent=100 * proximity["lt_10_acquisitions"] / 40))
        if seed in args.gp_seeds:
            cached[seed] = (walks, indices, common)
        if seed % 10 == 0:
            print(f"Geometry: seed {seed} complete", flush=True)
    write_csv(args.output_dir / "geometry.csv", geometry)
    write_csv(args.output_dir / "geometry_summary.csv", aggregate(geometry, GEOMETRY_METRICS, args.steps, args.geometry_seeds))
    orientation = compute_current_orientation(args.current_file, datetime.fromisoformat(grid.selected_time_label),
                                             12., "u_velocity", "v_velocity", "time", grid.valid_mask)
    transform = build_rotation_transform(grid, orientation.math_angle_degrees, "current-informed")
    coordinates = transform.transform(xy)
    scaler = StandardScaler().fit(coordinates[grid.valid_mask.ravel()])
    config.update(rotation_degrees=orientation.math_angle_degrees,
                  coordinate_mean=scaler.mean_.tolist(), coordinate_scale=scaler.scale_.tolist())
    (args.output_dir / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    for seed in args.gp_seeds:
        walks, indices, common = cached[seed]
        if not common.any() or not (common & local).any() or grid.field.ravel()[peak_index] <= 0:
            raise ValueError("Missing common test cells or positive reference peak.")
        saved = dict(common=common, local=local, truth=grid.field)
        for step in args.steps:
            print(f"GP seed={seed}, step={step:g}, distinct={len(indices[step])}", flush=True)
            row, diagnostics, prediction = fit_step(grid, transform, scaler, coordinates, indices[step],
                                                    seed, step, common, local, peak_index)
            gp.append(row)
            optimizers.extend(diagnostics)
            write_csv(args.output_dir / "gp.csv", gp)
            write_csv(args.output_dir / "optimizer.csv", optimizers)
            saved[f"indices_{step:g}"] = indices[step]
            saved[f"positions_{step:g}"] = walks[step].positions
            if prediction is not None:
                saved[f"prediction_{step:g}"] = prediction
            print(f"  success={row['success']}; global={row['rmse_global']:.5f}; "
                  f"local={row['rmse_local']:.5f}; peak error={row['peak_error_percent']:.2f}%", flush=True)
        np.savez_compressed(args.output_dir / f"seed_{seed}.npz", **saved)
    eligible = [seed for seed in args.gp_seeds if all(r["success"] for r in gp if r["seed"] == seed)]
    print(f"Complete converged GP seeds: {eligible}", flush=True)
    write_csv(args.output_dir / "gp_summary.csv", aggregate(gp, GP_METRICS, args.steps, eligible))


# Keep this narrowly scoped experiment separate from ordinary robot runs.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nc-file", type=Path, required=True)
    parser.add_argument("--current-file", type=Path, default=Path("CL02_V1_SRC000_U_V_10mGrid.nc"))
    parser.add_argument("--time-index", type=int, default=729)
    parser.add_argument("--steps", nargs="+", type=float, default=[30., 60., 90., 120.])
    parser.add_argument("--geometry-seeds", nargs="+", type=int, default=list(range(1, 101)))
    parser.add_argument("--gp-seeds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robot_step_sensitivity"))
    args = parser.parse_args()
    for values in (args.steps, args.geometry_seeds, args.gp_seeds):
        if len(values) < 2 or len(set(values)) != len(values):
            parser.error("At least two distinct steps and seeds are required.")
    if (any(not np.isfinite(s) or s <= 0 for s in args.steps)
            or any(s < 0 for s in args.geometry_seeds)
            or not set(args.gp_seeds).issubset(args.geometry_seeds)):
        parser.error("Positive finite steps and nonnegative GP seeds within geometry seeds are required.")
    with threadpool_limits(limits=1):
        run_study(args)


if __name__ == "__main__":
    main()
