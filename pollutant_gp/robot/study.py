# Matched robot/static comparisons on a frozen concentration field.

import csv
from dataclasses import replace
import json

import numpy as np
from sklearn.preprocessing import StandardScaler

from pollutant_gp.model import fit_gaussian_process
from pollutant_gp.peak import measure_peak
from pollutant_gp.peak_kernel import INITIALIZATIONS, print_kernel_fit, select_converged_fit
from pollutant_gp.peak_sampling import common_unsampled_mask
from pollutant_gp.positivity import prediction_metrics
from pollutant_gp.reconstruction import reconstruct_field
from pollutant_gp.robot.trajectories import observations_up_to, simulate_random_walk, uniform_reference
from pollutant_gp.robot.visualization import plot_robot_results


# Write ordinary rows so diagnostics can be inspected without loading Python objects.
def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Reuse GP fitting and metrics; only the source of observations changes.
def run_robot_study(args, grid, coordinate_transform, output_path, coordinate_unit):
    if coordinate_transform is None:
        raise ValueError("Robot study requires current-informed coordinates.")
    seed = args.random_seed
    budget = args.n_robots * args.n_steps
    walk = simulate_random_walk(grid, args.n_robots, args.n_steps, args.robot_step,
                                args.robot_start_radius, seed, args.robot_start_center)
    uniform = uniform_reference(grid, walk.cell_indices[0], budget, seed)
    common = common_unsampled_mask(grid, {"Robot": walk.cell_indices.ravel(), "Uniform": uniform})
    peak_index = np.argmax(np.where(grid.valid_mask, grid.field, -np.inf))
    xy = np.column_stack((grid.x_grid.ravel(), grid.y_grid.ravel()))
    local = grid.valid_mask & (np.linalg.norm(xy - xy[peak_index], axis=1).reshape(grid.field.shape)
                              <= args.peak_radius)
    if not common.any() or not (common & local).any():
        raise ValueError("No common unobserved cells globally or in the diagnostic disk.")
    rotated = coordinate_transform.transform(xy)
    scaler = StandardScaler().fit(rotated[grid.valid_mask.ravel()])
    optimizer_seed = seed if args.optimizer_seed is None else args.optimizer_seed
    base = output_path.with_name(f"{output_path.stem}_seed_{seed}_robot")
    base.parent.mkdir(parents=True, exist_ok=True)
    print("\n=== Robot sampling: frozen field, two RBFs, direct + clipping ===")
    print(f"Seed={seed}; deployment center={walk.center}; radius={args.robot_start_radius:g} {coordinate_unit}")
    print(f"{args.n_robots} robots x {args.n_steps} acquisitions = {budget} measurements (initial included)")
    print(f"Step={args.robot_step:g} {coordinate_unit}; duplicates consume budget but are fitted once")
    print(f"Domain coordinate scale={scaler.scale_}; physical lower bounds={args.length_scale_lower_bound * scaler.scale_}")
    print(f"Fixed common unobserved cells: global={common.sum()}, local={(common & local).sum()}")
    print("Uniform references share initial positions, but have no travel constraints.")
    rows, optimizer_rows, results, cache = [], [], {}, {}
    saved = dict(truth=grid.field, valid=grid.valid_mask, x=grid.x_grid, y=grid.y_grid,
                 positions=walk.positions, visits=walk.cell_indices, rejected=walk.rejected,
                 deployment_center=walk.center, uniform_indices=uniform,
                 common_unobserved=common, local=local, coordinate_mean=scaler.mean_,
                 coordinate_scale=scaler.scale_, rotation_degrees=coordinate_transform.angle_degrees,
                 config=json.dumps(vars(args), default=str), coordinate_unit=coordinate_unit,
                 checkpoints=np.array(args.robot_checkpoints))
    for count in args.robot_checkpoints:
        acquired_steps = count // args.n_robots
        mobile = observations_up_to(walk, count)
        layouts = {"Robot": mobile, "Uniform budget": uniform[:count],
                   "Uniform distinct": uniform[:len(mobile)]}
        for layout, indices in layouts.items():
            key = tuple(indices)
            print(f"\nN={count}, {layout}: {len(indices)} distinct cells", flush=True)
            if key not in cache:
                values = grid.field.ravel()[indices]
                candidates = []
                for name, first, second, amplitude in INITIALIZATIONS["Double"]:
                    model, _, diagnostic = fit_gaussian_process(
                        rotated[indices], values, "anisotropic",
                        args.length_scale_lower_bound, args.length_scale_upper_bound,
                        args.noise_level_initial, args.noise_level_lower_bound, args.noise_level_upper_bound,
                        "none", 0, optimizer_seed, constant_value_initial=amplitude,
                        length_scale_initial=first, second_length_scale_initial=second,
                        coordinate_scaler=scaler,
                    )
                    run = diagnostic.optimizer_runs[diagnostic.selected_run_index]
                    entry = dict(checkpoint=count, layout=layout, initialization=name, success=run.success,
                                 status=run.status, message=run.message, iterations=run.iterations,
                                 evaluations=run.function_evaluations, lml=diagnostic.final_lml,
                                 target_mean=diagnostic.target_mean, target_scale=diagnostic.target_scale,
                                 physical_length_scales=json.dumps(diagnostic.physical_length_scales.tolist()))
                    for parameter in diagnostic.hyperparameters:
                        entry[parameter.name] = parameter.optimized_value
                        entry[parameter.name + "_initial"] = parameter.initial_value
                        entry[parameter.name + "_lower_hit"] = parameter.at_lower_bound
                        entry[parameter.name + "_upper_hit"] = parameter.at_upper_bound
                    optimizer_rows.append(entry)
                    if args.verbose_optimizer_diagnostics:
                        print_kernel_fit(name, diagnostic, coordinate_unit)
                    elif not run.success:
                        print(f"WARNING {name}: status={run.status}; {run.message}", flush=True)
                    candidates.append((name, model, scaler, diagnostic))
                write_rows(base.with_name(base.name + "_optimizer.csv"), optimizer_rows)
                name, model, _, diagnostic = select_converged_fit(candidates)
                raw = reconstruct_field(grid, model, scaler, args.prediction_batch_size,
                                        "none", False, coordinate_transform)
                prediction = np.maximum(raw.mean_field, 0)
                metrics = prediction_metrics(grid.field, raw.mean_field, prediction,
                                             grid.valid_mask, common, local)
                reconstruction = replace(raw, mean_field=prediction, rmse=metrics["full_rmse"],
                                         mse=metrics["full_rmse"]**2, r2=metrics["full_r2"])
                peak = measure_peak(grid, reconstruction, indices, values, args.peak_radius)
                cache[key] = (reconstruction, raw.mean_field, metrics, peak, name, diagnostic)
            reconstruction, raw_mean, metrics, peak, name, diagnostic = cache[key]
            mobile_layout = layout == "Robot"
            row = dict(seed=seed, checkpoint=count, layout=layout,
                       measurements=count if layout != "Uniform distinct" else len(indices),
                       distinct_cells=len(indices),
                       revisits=count - len(indices) if mobile_layout else 0,
                       rejected_moves=int(walk.rejected[:acquired_steps].sum()) if mobile_layout else 0,
                       distance_total=float(np.linalg.norm(np.diff(walk.positions[:acquired_steps], axis=0),
                                                           axis=2).sum()) if mobile_layout else float("nan"),
                       common_global_count=int(common.sum()), common_local_count=int((common & local).sum()),
                       initialization=name, lml=diagnostic.final_lml, **metrics,
                       true_peak=peak.true_max, observed_max=peak.observed_max,
                       prediction_at_true_peak=peak.prediction_at_true_peak, predicted_max=peak.predicted_max,
                       peak_distance=peak.peak_location_error,
                       peak_relative_error_percent=100 * abs(peak.peak_underestimation) / peak.true_max
                       if peak.true_max > 0 else float("nan"),
                       nearest_sensor_distance=peak.nearest_sensor_distance,
                       local_sensor_count=peak.local_sensor_count, true_peak_sampled=peak.true_peak_sampled)
            rows.append(row)
            results[(layout, count)] = reconstruction
            tag = f"{layout.lower().replace(' ', '_')}_{count}"
            saved[tag + "_indices"] = indices
            saved[tag + "_raw_mean"] = raw_mean
            saved[tag + "_std"] = reconstruction.std_field
            print(f"Selected {name}; LML={diagnostic.final_lml:.6g}; "
                  f"unseen RMSE global={row['unseen_global_rmse']:.6g}, local={row['unseen_local_rmse']:.6g}; "
                  f"prediction at true peak={peak.prediction_at_true_peak:.6g}", flush=True)
            write_rows(base.with_name(base.name + "_metrics.csv"), rows)
    fields_path = base.with_name(base.name + "_fields.npz")
    np.savez_compressed(fields_path, **saved)
    paths = plot_robot_results(grid, walk, results, rows, args, base, coordinate_unit)
    print(f"\nSaved metrics/optimizer CSV and replay fields: {fields_path}")
    for path in paths:
        print(f"Saved: {path}")
    return rows, walk, common
