# Controlled single- versus two-scale kernel comparison on identical sensor sets.

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from pollutant_gp.model import fit_gaussian_process
from pollutant_gp.peak import measure_peak, extract_peak_profiles
from pollutant_gp.peak_sampling import PeakSamplingRun, make_peak_sampling_sets, common_unsampled_mask
from pollutant_gp.peak_visualization import print_peak_diagnostics, plot_peak_kernel_comparison
from pollutant_gp.reconstruction import reconstruct_field
from pollutant_gp.sampling import sample_sensor_points
from pollutant_gp.spatial import maybe_transform_coordinates


# Keep two declared starting points per structure, with total initial signal variance one.
INITIALIZATIONS = {
    "Single": (("Default", 1.0, None, 1.0), ("Short-scale", 0.1, None, 1.0)),
    "Double": (("Broad/short", 1.0, 0.1, 0.5), ("Compact/short", 0.2, 0.075, 0.5)),
}


# Rank candidates on their common training observations, never on reconstruction error.
def select_converged_fit(candidates):
    eligible = [(name, model, scaler, diagnostic) for name, model, scaler, diagnostic in candidates
                if np.isfinite(diagnostic.final_lml)
                and diagnostic.optimizer_runs[diagnostic.selected_run_index].success]
    if not eligible:
        raise RuntimeError("No converged finite-LML candidate; no selected reconstruction is reported.")
    return max(eligible, key=lambda candidate: candidate[3].final_lml)


# Print optimized parameters and numerical status for selected and rejected candidates alike.
def print_kernel_fit(name, diagnostic, unit):
    print(f"Candidate {name}: LML={diagnostic.final_lml:.12g}")
    print(f"  ell standardized: {diagnostic.standardized_length_scales}")
    print(f"  ell physical ({unit}): {diagnostic.physical_length_scales}")
    for parameter in diagnostic.hyperparameters:
        print(f"  {parameter.name}: initial={parameter.initial_value:.9g}, "
              f"final={parameter.optimized_value:.9g}, "
              f"bounds=[{parameter.lower_bound:g}, {parameter.upper_bound:g}], "
              f"lower_hit={parameter.at_lower_bound}, upper_hit={parameter.at_upper_bound}")
    for run in diagnostic.optimizer_runs:
        print(f"  success={run.success}, status={run.status}, iterations={run.iterations}, "
              f"evaluations={run.function_evaluations}; {run.message}")


# Compare both kernels with preprocessing fixed from Uniform and common held-out masks.
def run_peak_kernel_study(args, grid, coordinate_transform, output_path, coordinate_unit):
    if coordinate_transform is None:
        raise ValueError("The kernel comparison requires a physical coordinate rotation.")
    _, _, uniform = sample_sensor_points(grid, args.n_samples, 0.0, args.random_seed)
    sets, local = make_peak_sampling_sets(
        grid, uniform, args.peak_radius, args.peak_local_replacements, args.random_seed,
    )
    # Retain the phase-2 mask, including the unused Peak-observed control, for comparability.
    common = common_unsampled_mask(grid, sets)
    local_common = common & local
    if not common.any() or not local_common.any():
        raise ValueError("The study requires common unobserved global and local cells.")
    xy = np.column_stack((grid.x_grid.ravel(), grid.y_grid.ravel()))
    uniform_xy = maybe_transform_coordinates(xy[uniform], coordinate_transform)
    scaler = StandardScaler().fit(uniform_xy)
    values_u = grid.field.ravel()[uniform].astype(float)
    normalization = (float(values_u.mean()), float(values_u.std()) or 1.0)
    seed = args.random_seed if args.optimizer_seed is None else args.optimizer_seed
    print("\n=== Single / two-scale peak kernel study ===")
    print(f"Sampling seed={args.random_seed}; N={args.n_samples}; no random restarts")
    print(f"Length-scale bounds=[{args.length_scale_lower_bound}, {args.length_scale_upper_bound}]")
    print("Two prescribed initializations per kernel; highest converged sensor LML is selected.")
    print("Both components use the SAME current rotation and Uniform preprocessing.")
    print("Component A/B labels are initialization labels, not enforced global/local roles.")
    print("Enrichment is oracle, centred on the true maximum, NOT a verified source.")
    print(f"Common unseen cells: global={common.sum()}, local={local_common.sum()}")
    print("The evaluation mask also excludes Peak-observed sensors to match the phase-2 study.")
    print(f"Frozen coordinate mean={scaler.mean_}; scale={scaler.scale_}")
    print(f"Frozen target mean/std={normalization}")
    runs = []
    for sampling_name in ("Uniform", "Locally enriched (oracle)"):
        indices = sets[sampling_name]
        coordinates = maybe_transform_coordinates(xy[indices], coordinate_transform)
        values = grid.field.ravel()[indices].astype(float)
        print(f"\n--- {sampling_name}: local sensors={local.ravel()[indices].sum()} ---", flush=True)
        for structure, initializations in INITIALIZATIONS.items():
            candidates = []
            for name, first, second, amplitude in initializations:
                print(f"\nFitting {structure} / {name} ...", flush=True)
                model, fitted_scaler, diagnostic = fit_gaussian_process(
                    coordinates, values, args.kernel_mode,
                    args.length_scale_lower_bound, args.length_scale_upper_bound,
                    args.noise_level_initial, args.noise_level_lower_bound, args.noise_level_upper_bound,
                    args.target_transform, 0, seed,
                    constant_value_initial=amplitude, length_scale_initial=first,
                    second_length_scale_initial=second, coordinate_scaler=scaler,
                    target_normalization=normalization,
                )
                print_kernel_fit(name, diagnostic, coordinate_unit)
                candidates.append((name, model, fitted_scaler, diagnostic))
            name, model, _, diagnostic = select_converged_fit(candidates)
            print(f"Selected {structure}: {name}; kernel={model.kernel_}", flush=True)
            reconstruction = reconstruct_field(
                grid, model, scaler, args.prediction_batch_size, args.target_transform,
                args.clip_negative, coordinate_transform, target_normalization=normalization,
            )
            peak = measure_peak(grid, reconstruction, indices, values, args.peak_radius)
            error = reconstruction.mean_field - grid.field
            run = PeakSamplingRun(
                f"{sampling_name} / {structure}", indices, reconstruction, peak,
                extract_peak_profiles(grid, reconstruction, peak, coordinate_transform), diagnostic,
                float(np.sqrt(np.mean(error[common] ** 2))),
                float(np.sqrt(np.mean(error[local_common] ** 2))),
            )
            runs.append(run)
            print_peak_diagnostics(peak, coordinate_unit)
            if structure == "Double":
                a, b = model.kernel_.k1.k1, model.kernel_.k1.k2
                print(f"Component A signal variance fraction: "
                      f"{a.k1.constant_value / (a.k1.constant_value + b.k1.constant_value):.6g}")
                print(f"Directional scale ratios A/B: {a.k2.length_scale / b.k2.length_scale}")
                print("Similar scales or negligible amplitude do not support distinct spatial scales.")
    print("\n=== Selected fits (sensor LML selection; full-grid truth is diagnostic only) ===")
    print("Configuration | LML | full RMSE | full R2 | unseen global RMSE | unseen local RMSE "
          "| prediction at true peak | predicted GLOBAL max | GLOBAL peak distance")
    for run in runs:
        print(f"{run.name} | {run.optimization.final_lml:.9g} | {run.reconstruction.rmse:.9g} | "
              f"{run.reconstruction.r2:.9g} | {run.common_rmse:.9g} | {run.common_local_rmse:.9g} | "
              f"{run.peak.prediction_at_true_peak:.9g} | {run.peak.predicted_max:.9g} | "
              f"{run.peak.peak_location_error:.9g}")
    print("Do not compare sensor LML across Uniform and enriched observations.")
    base = output_path.with_name(f"{output_path.stem}_seed_{args.random_seed}_peak_kernels")
    title = (f"{args.nc_file.stem} | time index {args.time_index}\n"
             f"N = {args.n_samples}, sampling seed = {args.random_seed} | Current-informed kernels")
    for path in plot_peak_kernel_comparison(grid, runs, base, title, coordinate_unit,
                                           args.peak_vmax, args.show):
        print(f"Saved kernel comparison: {path}")
    return runs, common
