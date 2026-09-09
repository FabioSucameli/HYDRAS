# Controlled oracle sampling experiments with a fixed sensor budget.

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from pollutant_gp.model import GPOptimizationDiagnostics, fit_gaussian_process
from pollutant_gp.peak import PeakDiagnostics, PeakProfile, extract_peak_profiles, measure_peak
from pollutant_gp.peak_visualization import plot_peak_sampling_comparison, print_peak_diagnostics
from pollutant_gp.reconstruction import reconstruct_field
from pollutant_gp.sampling import sample_sensor_points
from pollutant_gp.spatial import maybe_transform_coordinates
from pollutant_gp.types import GridData, ReconstructionResult


@dataclass(frozen=True)
class PeakSamplingRun:
    name: str
    indices: np.ndarray
    reconstruction: ReconstructionResult
    peak: PeakDiagnostics
    profiles: tuple[PeakProfile, PeakProfile]
    optimization: GPOptimizationDiagnostics
    common_rmse: float
    common_local_rmse: float
    reused: bool = False


# Replace sensors without changing their total number or resampling the retained cells.
def make_peak_sampling_sets(
    grid: GridData, uniform_indices: np.ndarray, radius: float,
    replacements: int, sampling_seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    uniform = np.asarray(uniform_indices)
    valid = grid.valid_mask.ravel()
    if (uniform.ndim != 1 or not np.issubdtype(uniform.dtype, np.integer)
            or uniform.size == 0 or np.unique(uniform).size != uniform.size
            or np.any(uniform < 0) or np.any(uniform >= valid.size)):
        raise ValueError("Uniform sensors must be distinct valid integer cell indices.")
    if not np.all(valid[uniform]):
        raise ValueError("Uniform sensors must be on valid cells.")
    if not np.isfinite(radius) or radius <= 0 or replacements <= 0:
        raise ValueError("The radius and replacement count must be positive.")
    peak = int(np.argmax(np.where(valid, grid.field.ravel(), -np.inf)))
    distance = np.hypot(grid.x_grid - grid.x_grid.ravel()[peak],
                        grid.y_grid - grid.y_grid.ravel()[peak])
    local = grid.valid_mask & (distance <= radius)
    outside_positions = np.flatnonzero(~local.ravel()[uniform])
    candidates = np.setdiff1d(np.flatnonzero(local), uniform)
    if replacements > outside_positions.size or replacements > candidates.size:
        raise ValueError(
            f"Cannot make {replacements} local replacements: {outside_positions.size} outside "
            f"sensors and {candidates.size} distinct unobserved local cells are available."
        )

    observed = uniform.copy()
    if peak not in uniform:
        rng = np.random.default_rng(np.random.SeedSequence([sampling_seed, 1]))
        observed[rng.choice(outside_positions)] = peak

    # Independent deterministic stream; enrichment starts from Uniform, not Peak-observed.
    rng = np.random.default_rng(np.random.SeedSequence([sampling_seed, 2]))
    enriched = uniform.copy()
    removed = rng.choice(outside_positions, replacements, replace=False)
    enriched[removed] = rng.choice(candidates, replacements, replace=False)
    return {"Uniform": uniform.copy(), "Peak-observed (oracle)": observed,
            "Locally enriched (oracle)": enriched}, local


# Evaluate all methods on the same cells, outside the union of their training sets.
def common_unsampled_mask(grid: GridData, sample_sets: dict[str, np.ndarray]) -> np.ndarray:
    common = grid.valid_mask.copy()
    common.ravel()[np.unique(np.concatenate(list(sample_sets.values())))] = False
    return common


# Run one sampling seed, keeping baseline normalization and physical rotation fixed.
def run_peak_sampling_study(args, grid, coordinate_transform, output_path, coordinate_unit):
    _, _, uniform = sample_sensor_points(grid, args.n_samples, 0.0, args.random_seed)
    sample_sets, local = make_peak_sampling_sets(
        grid, uniform, args.peak_radius, args.peak_local_replacements, args.random_seed,
    )
    common = common_unsampled_mask(grid, sample_sets)
    local_common = common & local
    if not common.any() or not local_common.any():
        raise ValueError("The comparison needs nonempty common unobserved global and local sets.")

    print("\n=== Oracle peak sampling control ===")
    print(f"Sampling seed: {args.random_seed}; total sensors in every configuration: {args.n_samples}")
    print("The disk is centred on the ground-truth maximum, NOT a verified source position.")
    print("Peak-observed and locally enriched use privileged information; neither is an operational policy.")
    print(f"Disk radius: {args.peak_radius:g} {coordinate_unit}; local valid cells: {local.sum()}")
    print(f"Local replacements from Uniform: {args.peak_local_replacements}")
    print(f"Common unobserved cells: global={common.sum()}, local={local_common.sum()}")
    print("Common evaluation masks are fixed before fitting and exclude all three sensor sets.")
    print("Default initialization and bounds are unchanged; all hyperparameters are refitted by LML.")
    print("Coordinate scaling and target mean/std are frozen from Uniform for the other configurations.")
    print("LML values belong to different observations: do not use them to rank sampling configurations.")
    optimizer_seed = args.random_seed if args.optimizer_seed is None else args.optimizer_seed
    runs = []
    scaler = None
    normalization = None
    for name, indices in sample_sets.items():
        print(f"\n--- {name} ---", flush=True)
        changed = np.setdiff1d(indices, uniform).size
        print(f"Replaced sensors: {changed}; local sensors: {local.ravel()[indices].sum()}")
        if runs and np.array_equal(indices, uniform):
            run = replace(runs[0], name=name, reused=True)
            print("The true maximum was already sampled: reusing Uniform, with no additional fit.")
        else:
            xy = np.column_stack((grid.x_grid.ravel()[indices], grid.y_grid.ravel()[indices]))
            values = grid.field.ravel()[indices].astype(float)
            model, fitted_scaler, optimization = fit_gaussian_process(
                sample_coordinates=maybe_transform_coordinates(xy, coordinate_transform),
                sample_values=values, kernel_mode=args.kernel_mode,
                length_scale_lower_bound=args.length_scale_lower_bound,
                length_scale_upper_bound=args.length_scale_upper_bound,
                noise_level_initial=args.noise_level_initial,
                noise_level_lower_bound=args.noise_level_lower_bound,
                noise_level_upper_bound=args.noise_level_upper_bound,
                target_transform=args.target_transform, n_restarts=0, optimizer_seed=optimizer_seed,
                coordinate_scaler=scaler, target_normalization=normalization,
            )
            reconstruction = reconstruct_field(
                grid, model, fitted_scaler, args.prediction_batch_size,
                args.target_transform, args.clip_negative, coordinate_transform,
                target_normalization=normalization,
            )
            peak = measure_peak(grid, reconstruction, indices, values, args.peak_radius)
            profiles = extract_peak_profiles(grid, reconstruction, peak, coordinate_transform)
            error = reconstruction.mean_field - grid.field
            run = PeakSamplingRun(
                name, indices, reconstruction, peak, profiles, optimization,
                float(np.sqrt(np.mean(error[common] ** 2))),
                float(np.sqrt(np.mean(error[local_common] ** 2))),
            )
            if not runs:
                scaler = fitted_scaler
                normalization = (optimization.target_mean, optimization.target_scale)
                print(f"Frozen coordinate mean: {scaler.mean_}; scale: {scaler.scale_}")
                print(f"Frozen target mean/std: {normalization}")
        runs.append(run)
        d = run.optimization
        print(f"LML: {d.final_lml:.9g}; ell: {d.standardized_length_scales}; "
              f"physical ell ({coordinate_unit}): {d.physical_length_scales}")
        for parameter in d.hyperparameters:
            print(f"  {parameter.name}={parameter.optimized_value:.9g}; "
                  f"lower_hit={parameter.at_lower_bound}; upper_hit={parameter.at_upper_bound}")
        for optimizer_run in d.optimizer_runs:
            print(f"  Optimizer success={optimizer_run.success}; status={optimizer_run.status}; "
                  f"{optimizer_run.message}")
        print_peak_diagnostics(run.peak, coordinate_unit)
        print(f"Full-grid MSE={run.reconstruction.mse:.9g}; R2={run.reconstruction.r2:.9g}")
        print(f"Common unobserved RMSE: global={run.common_rmse:.9g}; local={run.common_local_rmse:.9g}")

    print("\n=== RMSE comparison (same evaluation cells for every row) ===")
    print(f"{'Configuration':29s} {'All global':>12s} {'All local':>12s} "
          f"{'Unseen global':>14s} {'Unseen local':>14s}")
    for run in runs:
        print(f"{run.name:29s} {run.peak.global_rmse:12.7f} {run.peak.local_rmse:12.7f} "
              f"{run.common_rmse:14.7f} {run.common_local_rmse:14.7f}")
    title = (f"{args.nc_file.stem} | time index {args.time_index}\n"
             f"N = {args.n_samples}, sampling seed = {args.random_seed} | Oracle sampling control")
    base = output_path.with_name(f"{output_path.stem}_seed_{args.random_seed}_peak_sampling")
    for path in plot_peak_sampling_comparison(
        grid, runs, base, title, coordinate_unit, args.peak_vmax, args.show,
    ):
        print(f"Saved sampling comparison: {path}")
    return runs, common
