# Main reconstruction workflow.

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from pollutant_gp.current import compute_current_orientation
from pollutant_gp.data import (
    choose_time_index,
    prepare_grid_data,
    print_dataset_structure,
    validate_dataset_layout,
)

# Optional dataset inspection utility
from pollutant_gp.inspection import inspect_netcdf_files

# Gaussian Process model training
from pollutant_gp.model import GPOptimizationDiagnostics, fit_gaussian_process

# Field reconstruction using the trained GP
from pollutant_gp.reconstruction import reconstruct_field

# Synthetic sensor sampling
from pollutant_gp.sampling import sample_sensor_points
from pollutant_gp.spatial import (
    RotationTransform,
    build_rotation_transform,
    maybe_transform_coordinates,
)

# Visualization utilities
from pollutant_gp.visualization import (
    plot_concentration_map,
    plot_kernel_comparison_multiseed,
    plot_kernel_comparison_multiseed_panels,
    plot_length_scale_lower_bound_study,
    plot_reconstruction,
    plot_reconstruction_panels,
    plot_sample_size_study,
    plot_sample_size_study_panels,
    plot_sample_size_study_multiseed,
    plot_sample_size_study_multiseed_panels,
    plot_valid_domain,
)
from pollutant_gp.wind import compute_wind_orientation, parse_time_label


# Resolve the optimizer seed without changing the historical default behavior.
def resolve_optimizer_seed(args: argparse.Namespace, sampling_seed: int) -> int:
    if args.optimizer_seed is None:
        return sampling_seed
    return args.optimizer_seed


# Format a numeric vector compactly for terminal diagnostics.
def format_diagnostic_vector(values: np.ndarray) -> str:
    return np.array2string(
        np.asarray(values, dtype=float),
        precision=6,
        separator=", ",
        suppress_small=False,
    )


# Summarize the most important optimization outcomes on one terminal line.
def format_gp_optimization_summary(diagnostics: GPOptimizationDiagnostics) -> str:
    bound_hits: list[str] = []
    if diagnostics.length_scale_lower_bound_hit:
        bound_hits.append("lower")
    if diagnostics.length_scale_upper_bound_hit:
        bound_hits.append("upper")
    bound_status = "+".join(bound_hits) if bound_hits else "none"
    selected_run = diagnostics.optimizer_runs[diagnostics.selected_run_index]
    optimizer_status = "ok" if selected_run.success else f"failed:{selected_run.status}"
    return (
        f"LML={diagnostics.final_lml:.6g}  "
        f"length_scale={format_diagnostic_vector(diagnostics.standardized_length_scales)}  "
        f"length_bound={bound_status}  optimizer={optimizer_status}"
    )


# Print complete initialization, scaling, bound, and optimizer diagnostics.
def print_gp_optimization_diagnostics(
    diagnostics: GPOptimizationDiagnostics,
    sampling_seed: int,
) -> None:
    print("\n=== GP optimization diagnostics ===")
    print(f"Sampling seed: {sampling_seed}")
    print(f"Optimizer seed: {diagnostics.optimizer_seed}")
    print(f"Configured optimizer restarts: {diagnostics.n_restarts}")
    print(f"Total optimizer runs: {len(diagnostics.optimizer_runs)}")
    print(f"Coordinate mean: {format_diagnostic_vector(diagnostics.coordinate_mean)}")
    print(f"Coordinate scale: {format_diagnostic_vector(diagnostics.coordinate_scale)}")
    print(f"Target mean before normalize_y: {diagnostics.target_mean:.8g}")
    print(f"Target scale before normalize_y: {diagnostics.target_scale:.8g}")
    print(f"Initial LML: {diagnostics.initial_lml:.8g}")
    print(f"Final LML: {diagnostics.final_lml:.8g}")
    print(f"Final LML gradient norm: {diagnostics.final_lml_gradient_norm:.8g}")
    print(f"Selected optimizer run: {diagnostics.selected_run_index}")
    print(
        "Standardized length scales: "
        f"{format_diagnostic_vector(diagnostics.standardized_length_scales)}"
    )
    print(
        "Length scales in coordinate units: "
        f"{format_diagnostic_vector(diagnostics.physical_length_scales)}"
    )

    print("Hyperparameters:")
    for parameter in diagnostics.hyperparameters:
        if parameter.at_lower_bound:
            bound_status = "LOWER"
        elif parameter.at_upper_bound:
            bound_status = "UPPER"
        else:
            bound_status = "interior"
        print(
            f"  - {parameter.name}: initial={parameter.initial_value:.8g}, "
            f"optimized={parameter.optimized_value:.8g}, "
            f"bounds=[{parameter.lower_bound:.8g}, {parameter.upper_bound:.8g}], "
            f"status={bound_status}, "
            f"dLML/dlog(theta)={parameter.lml_gradient:.4g}"
        )

    print("Optimizer runs:")
    for run in diagnostics.optimizer_runs:
        selected = " selected" if run.run_index == diagnostics.selected_run_index else ""
        print(
            f"  - run {run.run_index}{selected}: success={run.success}, "
            f"status={run.status}, iterations={run.iterations}, "
            f"evaluations={run.function_evaluations}, "
            f"LML={run.initial_lml:.8g} -> {run.final_lml:.8g}"
        )
        print(f"    initial parameters: {format_diagnostic_vector(np.exp(run.initial_theta))}")
        print(f"    final parameters:   {format_diagnostic_vector(np.exp(run.optimized_theta))}")
        print(f"    termination: {run.message}")


# Print how strongly the GP violates non-negativity before clipping.
def print_positivity_diagnostics(reconstruction) -> None:
    percentage = 100.0 * reconstruction.negative_prediction_fraction
    print("\n=== Positivity diagnostics before clipping ===")
    print(f"Minimum prediction before clipping: {reconstruction.min_prediction_before_clipping:.8g}")
    print(
        "Negative predictions before clipping: "
        f"{reconstruction.negative_prediction_count} "
        f"({percentage:.4g}% of valid cells)"
    )
    if reconstruction.negative_prediction_count > 0:
        print(f"Mean negative prediction: {reconstruction.mean_negative_prediction:.8g}")
    else:
        print("Mean negative prediction: n/a")


# Convert optional CLI string values into Python None.
def optional_name(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() in {"none", "null", ""}:
        return None
    return value

# Build a figure path
def make_output_figure_path(
    output_dir: Path,
    figure_name: str | None,
    nc_file: Path,
    time_index: int | None,
    n_samples: int,
) -> Path:

    if figure_name:
        return output_dir / figure_name

    dataset_name = nc_file.stem
    time_part = f"time_{time_index}" if time_index is not None else "no_time"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{dataset_name}_{time_part}_samples_{n_samples}_{timestamp}.png"
    return output_dir / file_name


def make_domain_figure_path(
    output_dir: Path,
    nc_file: Path,
    time_index: int | None,
) -> Path:
    dataset_name = nc_file.stem
    time_part = f"time_{time_index}" if time_index is not None else "no_time"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{dataset_name}_{time_part}_valid_domain_{timestamp}.png"
    return output_dir / file_name


def make_concentration_map_path(
    output_dir: Path,
    figure_name: str | None,
    nc_file: Path,
    time_index: int | None,
) -> Path:
    if figure_name:
        return output_dir / figure_name

    dataset_name = nc_file.stem
    time_part = f"time_{time_index}" if time_index is not None else "no_time"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{dataset_name}_{time_part}_concentration_map_{timestamp}.png"
    return output_dir / file_name


def make_kernel_comparison_path(
    output_dir: Path,
    figure_name: str | None,
    nc_file: Path,
    time_index: int | None,
) -> Path:
    if figure_name:
        return output_dir / figure_name

    dataset_name = nc_file.stem
    time_part = f"time_{time_index}" if time_index is not None else "no_time"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{dataset_name}_{time_part}_kernel_comparison_{timestamp}.png"
    return output_dir / file_name


def make_length_scale_lower_bound_study_path(
    output_dir: Path,
    figure_name: str | None,
    nc_file: Path,
    time_index: int | None,
    n_samples: int,
) -> Path:
    if figure_name:
        return output_dir / figure_name

    dataset_name = nc_file.stem
    time_part = f"time_{time_index}" if time_index is not None else "no_time"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = (
        f"{dataset_name}_{time_part}_samples_{n_samples}_"
        f"lower_bound_study_{timestamp}.png"
    )
    return output_dir / file_name


def build_wind_coordinate_transform(
    args: argparse.Namespace,
    grid_data,
) -> RotationTransform:
    target_time = parse_time_label(grid_data.selected_time_label)
    wind_orientation = compute_wind_orientation(
        path=args.wind_file,
        target_time=target_time,
        average_hours=args.wind_average_hours,
        direction_convention=args.wind_direction_convention,
    )

    transform = build_rotation_transform(
        grid_data=grid_data,
        angle_degrees=wind_orientation.math_angle_degrees,
        description="wind-informed transport direction",
    )

    print("\n=== Physically informed coordinate transform ===")
    print(f"Wind file: {wind_orientation.source_path}")
    print(f"Target time: {wind_orientation.target_time}")
    print(f"Averaging window: {wind_orientation.average_hours:g} h")
    print(f"Wind vector speed: {wind_orientation.vector_speed:.6g}")
    print(f"Wind direction FROM: {wind_orientation.direction_from_degrees:.3f} deg")
    print(f"Transport direction TOWARD: {wind_orientation.direction_toward_degrees:.3f} deg")
    print(f"Rotation angle from +x axis: {wind_orientation.math_angle_degrees:.3f} deg")
    print(f"Rotation center: x={transform.center_x:.3f}, y={transform.center_y:.3f}")

    return transform


def build_current_coordinate_transform(
    args: argparse.Namespace,
    grid_data,
) -> RotationTransform:
    target_time = parse_time_label(grid_data.selected_time_label)
    current_orientation = compute_current_orientation(
        path=args.current_file,
        target_time=target_time,
        average_hours=args.current_average_hours,
        u_variable=args.current_u_variable,
        v_variable=args.current_v_variable,
        time_dim=args.current_time_dim,
        valid_mask=grid_data.valid_mask,
    )

    transform = build_rotation_transform(
        grid_data=grid_data,
        angle_degrees=current_orientation.math_angle_degrees,
        description="current-informed transport direction",
    )

    print("\n=== Physically informed coordinate transform ===")
    print(f"Current file: {current_orientation.source_path}")
    print(f"Target time: {current_orientation.target_time}")
    print(f"Averaging window: {current_orientation.average_hours:g} h")
    print(f"Selected current time steps: {current_orientation.selected_time_count}")
    print(f"Valid current vectors used: {current_orientation.valid_vector_count}")
    print(f"Mean current u: {current_orientation.mean_u:.6g} m/s")
    print(f"Mean current v: {current_orientation.mean_v:.6g} m/s")
    print(f"Mean current vector speed: {current_orientation.vector_speed:.6g} m/s")
    print(f"Current direction TOWARD: {current_orientation.direction_toward_degrees:.3f} deg")
    print(f"Rotation angle from +x axis: {current_orientation.math_angle_degrees:.3f} deg")
    print(f"Rotation center: x={transform.center_x:.3f}, y={transform.center_y:.3f}")

    return transform


def build_coordinate_transform(
    args: argparse.Namespace,
    grid_data,
) -> RotationTransform | None:
    if not args.physically_informed:
        return None

    if args.physics_source == "wind":
        transform = build_wind_coordinate_transform(args, grid_data)
    elif args.physics_source == "current":
        transform = build_current_coordinate_transform(args, grid_data)
    else:
        raise ValueError(f"Unknown physics source: {args.physics_source}")

    if args.kernel_mode != "anisotropic":
        print("Note: physically informed rotations are most meaningful with --kernel-mode anisotropic.")

    return transform


# Main workflow function
def run_workflow(args: argparse.Namespace) -> None:
    # Inspection mode: scan NetCDF files and print finite-domain patterns, then exit.
    if args.inspect_netcdf:
        inspect_netcdf_files(
            folder=args.netcdf_dir,
            time_dim=optional_name(args.time_dim),
            time_index=args.time_index,
        )
        return

    if not args.nc_file.exists():
        raise FileNotFoundError(f"NetCDF file not found: {args.nc_file}")

    # Dataset structure printing mode: print the structure, save a valid-domain map, and exit.
    if args.print_dataset:
        ds = xr.open_dataset(args.nc_file)
        try:
            print_dataset_structure(ds)

            variable_name = args.variable
            time_dim = optional_name(args.time_dim)
            y_dim = args.y_dim
            x_dim = args.x_dim
            y_coordinate = optional_name(args.y_coordinate)
            x_coordinate = optional_name(args.x_coordinate)

            try:
                validate_dataset_layout(
                    ds=ds,
                    variable_name=variable_name,
                    time_dim=time_dim,
                    y_dim=y_dim,
                    x_dim=x_dim,
                    y_coordinate=y_coordinate,
                    x_coordinate=x_coordinate,
                )

                data_array = ds[variable_name]
                time_index = choose_time_index(data_array, time_dim, args.time_index)
                grid_data = prepare_grid_data(
                    ds=ds,
                    variable_name=variable_name,
                    time_dim=time_dim,
                    time_index=time_index,
                    y_dim=y_dim,
                    x_dim=x_dim,
                    y_coordinate=y_coordinate,
                    x_coordinate=x_coordinate,
                )

                domain_path = make_domain_figure_path(
                    output_dir=args.output_dir,
                    nc_file=args.nc_file,
                    time_index=time_index,
                )
                plot_valid_domain(
                    grid_data=grid_data,
                    output_path=domain_path,
                    show=args.show,
                )
                print(f"\nSaved valid domain map: {domain_path}")
            except ValueError as exc:
                print(f"\nValid domain map was not created: {exc}")
        finally:
            ds.close()
        return
    
    # Main workflow: load data, prepare grid, sample sensors, fit GP, reconstruct field, compute metrics, and plot results.
    # Load the NetCDF dataset and print its structure.
    ds = xr.open_dataset(args.nc_file)
    try:
        print_dataset_structure(ds)

        variable_name = args.variable
        time_dim = optional_name(args.time_dim)
        y_dim = args.y_dim
        x_dim = args.x_dim
        y_coordinate = optional_name(args.y_coordinate)
        x_coordinate = optional_name(args.x_coordinate)

        # Check that the variable exists in the dataset
        validate_dataset_layout(
            ds=ds,
            variable_name=variable_name,
            time_dim=time_dim,
            y_dim=y_dim,
            x_dim=x_dim,
            y_coordinate=y_coordinate,
            x_coordinate=x_coordinate,
        )

        # Extract the concentration variable.
        data_array = ds[variable_name]
        time_index = choose_time_index(data_array, time_dim, args.time_index)

        print("\n=== Selected configuration ===")
        print(f"NetCDF file: {args.nc_file}")
        print(f"Concentration variable: {variable_name}")
        print(f"Time dimension: {time_dim}")
        print(f"Time index: {time_index}")
        print(f"Spatial dimensions: y='{y_dim}', x='{x_dim}'")
        print(f"Spatial coordinates: y='{y_coordinate}', x='{x_coordinate}'")
        print("Valid domain: finite concentration values; NaN cells are ignored")
        print(f"Number of sensor samples: {args.n_samples}")
        print(f"Sensor noise standard deviation: {args.noise_std}")
        if args.optimizer_seed is None:
            print("Optimizer seed: reuse the sampling seed for each fit")
        else:
            print(f"Optimizer seed: {args.optimizer_seed}")
        print(f"Optimizer restarts: {args.n_restarts}")
        print(f"Kernel mode: {args.kernel_mode}")
        print(f"Length-scale lower-bound study: {args.length_scale_lower_bound_study}")
        print(f"Kernel comparison study: {args.kernel_comparison_study}")
        if args.kernel_comparison_study:
            print("Physically informed rotation: evaluated inside the kernel comparison study")
        else:
            print(f"Physically informed rotation: {args.physically_informed}")
        if args.physically_informed:
            print(f"Physics source: {args.physics_source}")
            if args.physics_source == "wind":
                print(f"Wind file: {args.wind_file}")
                print(f"Wind averaging window: {args.wind_average_hours:g} h")
                print(f"Wind direction convention: {args.wind_direction_convention}")
            elif args.physics_source == "current":
                print(f"Current file: {args.current_file}")
                print(f"Current averaging window: {args.current_average_hours:g} h")
                print(f"Current u variable: {args.current_u_variable}")
                print(f"Current v variable: {args.current_v_variable}")
        print(f"Target transform: {args.target_transform}")
        print(f"Clip negative predictions: {args.clip_negative}")

        # Prepare 2D grid data
        grid_data = prepare_grid_data(
            ds=ds,
            variable_name=variable_name,
            time_dim=time_dim,
            time_index=time_index,
            y_dim=y_dim,
            x_dim=x_dim,
            y_coordinate=y_coordinate,
            x_coordinate=x_coordinate,
        )
    finally:
        ds.close()

    # Ground-truth field summary statistics
    valid_values = grid_data.field[grid_data.valid_mask]
    cells_above_threshold = int(np.sum(valid_values > 1.0))
    print("\n=== Selected field summary ===")
    print(f"Field shape: {grid_data.field.shape}")
    print(f"Valid cells: {grid_data.valid_mask.sum()} / {grid_data.valid_mask.size}")
    print(f"Ground-truth min: {np.nanmin(valid_values):.6g}")
    print(f"Ground-truth max: {np.nanmax(valid_values):.6g}")
    print(f"Ground-truth mean: {np.nanmean(valid_values):.6g}")
    print(f"Cells with concentration > 1: {cells_above_threshold}")

    if args.plot_concentration_map:
        concentration_map_path = make_concentration_map_path(
            output_dir=args.output_dir,
            figure_name=args.figure_name,
            nc_file=args.nc_file,
            time_index=time_index,
        )
        plot_concentration_map(
            grid_data=grid_data,
            output_path=concentration_map_path,
            show=args.show,
            display_threshold=args.concentration_display_threshold,
        )
        print(f"\nSaved concentration map: {concentration_map_path}")
        return

    if args.kernel_comparison_study:
        comparison_path = make_kernel_comparison_path(
            output_dir=args.output_dir,
            figure_name=args.figure_name,
            nc_file=args.nc_file,
            time_index=time_index,
        )
        run_kernel_comparison_study(args, grid_data, comparison_path)
        return

    coordinate_transform = build_coordinate_transform(args, grid_data)

    if args.length_scale_lower_bound_study:
        study_path = make_length_scale_lower_bound_study_path(
            output_dir=args.output_dir,
            figure_name=args.figure_name,
            nc_file=args.nc_file,
            time_index=time_index,
            n_samples=args.n_samples,
        )
        run_length_scale_lower_bound_study(
            args=args,
            grid_data=grid_data,
            output_path=study_path,
            coordinate_transform=coordinate_transform,
        )
        return

    # Sample synthetic sensor measurements
    sample_coordinates, sample_values, _ = sample_sensor_points(
        grid_data=grid_data,
        n_samples=args.n_samples,
        noise_std=args.noise_std,
        random_seed=args.random_seed,
    )
    print("\n=== Sample summary ===")
    print(f"Sample value min: {np.min(sample_values):.6g}")
    print(f"Sample value max: {np.max(sample_values):.6g}")
    print(f"Sample value mean: {np.mean(sample_values):.6g}")
    model_sample_coordinates = maybe_transform_coordinates(
        sample_coordinates,
        coordinate_transform,
    )

    # Train Gaussian Process model
    # The GP learns a mapping from spatial coordinates to concentration: (x, y) -> C
    # using only the sparse synthetic measurements.
    print("\n=== Fitting Gaussian Process ===")
    optimizer_seed = resolve_optimizer_seed(args, args.random_seed)
    model, coordinate_scaler, optimization_diagnostics = fit_gaussian_process(
        sample_coordinates=model_sample_coordinates,
        sample_values=sample_values,
        kernel_mode=args.kernel_mode,
        length_scale_lower_bound=args.length_scale_lower_bound,
        length_scale_upper_bound=args.length_scale_upper_bound,
        noise_level_initial=args.noise_level_initial,
        noise_level_lower_bound=args.noise_level_lower_bound,
        noise_level_upper_bound=args.noise_level_upper_bound,
        target_transform=args.target_transform,
        n_restarts=args.n_restarts,
        optimizer_seed=optimizer_seed,
    )
    print(f"Learned kernel: {model.kernel_}")
    print_gp_optimization_diagnostics(
        optimization_diagnostics,
        sampling_seed=args.random_seed,
    )

    # Reconstruct the full concentration field on the grid using the trained GP.
    print("\n=== Reconstructing field ===")
    reconstruction = reconstruct_field(
        grid_data=grid_data,
        model=model,
        coordinate_scaler=coordinate_scaler,
        batch_size=args.prediction_batch_size,
        target_transform=args.target_transform,
        clip_negative=args.clip_negative,
        coordinate_transform=coordinate_transform,
    )

    print("\n=== Reconstruction metrics on valid sea cells ===")
    print(f"MSE:  {reconstruction.mse:.8g}")
    print(f"RMSE: {reconstruction.rmse:.8g}")
    print(f"MAE:  {reconstruction.mae:.8g}")
    print(f"R^2:  {reconstruction.r2:.8g}")
    print_positivity_diagnostics(reconstruction)

    # Save visualization
    figure_path = make_output_figure_path(
        output_dir=args.output_dir,
        figure_name=args.figure_name,
        nc_file=args.nc_file,
        time_index=time_index,
        n_samples=args.n_samples,
    )
    plot_reconstruction(
        grid_data=grid_data,
        reconstruction=reconstruction,
        sample_coordinates=sample_coordinates,
        output_path=figure_path,
        show=args.show,
    )
    print(f"\nSaved figure: {figure_path}")
    panel_paths = plot_reconstruction_panels(
        grid_data=grid_data,
        reconstruction=reconstruction,
        sample_coordinates=sample_coordinates,
        output_path=figure_path,
        show=args.show,
    )
    print("Saved separate reconstruction panels:")
    for panel_path in panel_paths:
        print(f"  - {panel_path}")

    # Optional sample size study
    if args.sample_size_study:
        run_sample_size_study(args, grid_data, figure_path, coordinate_transform)

    # Optional multi-seed sample size study
    if args.sample_size_study_multiseed:
        run_sample_size_study_multiseed(args, grid_data, figure_path, coordinate_transform)


# Run a controlled sensitivity study in which only the RBF lower bound changes.
def run_length_scale_lower_bound_study(
    args: argparse.Namespace,
    grid_data,
    output_path: Path,
    coordinate_transform: RotationTransform | None,
) -> None:
    if args.n_restarts != 0:
        raise ValueError(
            "The controlled lower-bound study requires --n-restarts 0 so random restart "
            "initializations do not change with the bounds."
        )

    lower_bounds = sorted(set(args.length_scale_lower_bound_study_values))
    if not lower_bounds:
        raise ValueError("At least one lower-bound study value is required.")
    if any(value <= 0.0 for value in lower_bounds):
        raise ValueError("All lower-bound study values must be positive.")
    if any(value >= args.length_scale_upper_bound for value in lower_bounds):
        raise ValueError(
            "Every lower-bound study value must be smaller than --length-scale-upper-bound."
        )

    sample_coordinates, sample_values, _ = sample_sensor_points(
        grid_data=grid_data,
        n_samples=args.n_samples,
        noise_std=args.noise_std,
        random_seed=args.random_seed,
    )
    model_sample_coordinates = maybe_transform_coordinates(
        sample_coordinates,
        coordinate_transform,
    )
    optimizer_seed = resolve_optimizer_seed(args, args.random_seed)

    lml_values: list[float] = []
    rmse_values: list[float] = []
    r2_values: list[float] = []
    constant_kernel_values: list[float] = []
    white_kernel_values: list[float] = []
    standardized_length_scale_rows: list[np.ndarray] = []
    physical_length_scale_rows: list[np.ndarray] = []
    length_scale_bound_hit_rows: list[np.ndarray] = []

    print("\n=== Controlled length-scale lower-bound study ===")
    print(f"Lower bounds: {format_diagnostic_vector(np.asarray(lower_bounds))}")
    print(f"Shared sampling seed: {args.random_seed}")
    print(f"Shared optimizer seed: {optimizer_seed}")
    print(f"Shared sensor count: {args.n_samples}")
    print(f"Shared upper bound: {args.length_scale_upper_bound:g}")
    print("Optimizer restarts: 0")
    print("Sampling design: the same sensor coordinates and values are reused in every fit.")

    for lower_bound in lower_bounds:
        print(f"\n  Lower bound = {lower_bound:g}")
        model, coordinate_scaler, diagnostics = fit_gaussian_process(
            sample_coordinates=model_sample_coordinates,
            sample_values=sample_values,
            kernel_mode=args.kernel_mode,
            length_scale_lower_bound=lower_bound,
            length_scale_upper_bound=args.length_scale_upper_bound,
            noise_level_initial=args.noise_level_initial,
            noise_level_lower_bound=args.noise_level_lower_bound,
            noise_level_upper_bound=args.noise_level_upper_bound,
            target_transform=args.target_transform,
            n_restarts=0,
            optimizer_seed=optimizer_seed,
        )
        reconstruction = reconstruct_field(
            grid_data=grid_data,
            model=model,
            coordinate_scaler=coordinate_scaler,
            batch_size=args.prediction_batch_size,
            target_transform=args.target_transform,
            clip_negative=args.clip_negative,
            coordinate_transform=coordinate_transform,
        )

        length_parameters = [
            parameter
            for parameter in diagnostics.hyperparameters
            if "length_scale" in parameter.name
        ]
        length_gradients = np.asarray(
            [parameter.lml_gradient for parameter in length_parameters],
            dtype=float,
        )
        length_bound_hits = np.asarray(
            [parameter.at_lower_bound for parameter in length_parameters],
            dtype=bool,
        )
        if length_bound_hits.size == 1:
            length_bound_hits = np.repeat(
                length_bound_hits,
                diagnostics.physical_length_scales.size,
            )

        selected_run = diagnostics.optimizer_runs[diagnostics.selected_run_index]
        print(f"    Learned kernel: {model.kernel_}")
        print(
            f"    LML: {diagnostics.initial_lml:.8g} -> {diagnostics.final_lml:.8g}; "
            f"gradient norm={diagnostics.final_lml_gradient_norm:.6g}"
        )
        print(
            "    Standardized length scales: "
            f"{format_diagnostic_vector(diagnostics.standardized_length_scales)}"
        )
        print(
            "    Length scales in coordinate units: "
            f"{format_diagnostic_vector(diagnostics.physical_length_scales)}"
        )
        print(
            "    Length-scale LML gradients: "
            f"{format_diagnostic_vector(length_gradients)}"
        )
        print(
            f"    Bound hit: lower={diagnostics.length_scale_lower_bound_hit}, "
            f"upper={diagnostics.length_scale_upper_bound_hit}"
        )
        print(
            f"    Optimizer: success={selected_run.success}, status={selected_run.status}, "
            f"iterations={selected_run.iterations}, evaluations={selected_run.function_evaluations}"
        )
        print(f"    Termination: {selected_run.message}")
        print(
            f"    Metrics: RMSE={reconstruction.rmse:.8g}, "
            f"MAE={reconstruction.mae:.8g}, R2={reconstruction.r2:.8g}"
        )

        lml_values.append(diagnostics.final_lml)
        rmse_values.append(reconstruction.rmse)
        r2_values.append(reconstruction.r2)
        constant_kernel_values.append(float(model.kernel_.k1.k1.constant_value))
        white_kernel_values.append(float(model.kernel_.k2.noise_level))
        standardized_length_scale_rows.append(diagnostics.standardized_length_scales)
        physical_length_scale_rows.append(diagnostics.physical_length_scales)
        length_scale_bound_hit_rows.append(length_bound_hits)

    if coordinate_transform is None:
        length_scale_axis_labels = ("x", "y")
        table_length_scale_labels = ("ell_x", "ell_y")
    else:
        length_scale_axis_labels = ("along transport", "across transport")
        table_length_scale_labels = ("ell_parallel", "ell_perp")

    standardized_length_scales = np.vstack(standardized_length_scale_rows)
    physical_length_scales = np.vstack(physical_length_scale_rows)
    print("\n=== Lower-bound study summary ===")
    print(
        "  lower      final LML       RMSE         R2   "
        f"{table_length_scale_labels[0]} std   "
        f"{table_length_scale_labels[1]} std   "
        f"{table_length_scale_labels[0]} units   "
        f"{table_length_scale_labels[1]} units   "
        "ConstantKernel   WhiteKernel"
    )
    for index, lower_bound in enumerate(lower_bounds):
        standardized_row = standardized_length_scales[index]
        physical_row = physical_length_scales[index]
        standardized_values = (
            np.repeat(standardized_row, 2)
            if standardized_row.size == 1
            else standardized_row
        )
        print(
            f"  {lower_bound:5.3f}  {lml_values[index]:13.5f}  "
            f"{rmse_values[index]:9.6f}  {r2_values[index]:9.6f}  "
            f"{standardized_values[0]:9.6f}  {standardized_values[1]:9.6f}  "
            f"{physical_row[0]:11.4f}  {physical_row[1]:11.4f}  "
            f"{constant_kernel_values[index]:14.6g}  "
            f"{white_kernel_values[index]:11.6g}"
        )

    model_fit_path, length_scales_path = plot_length_scale_lower_bound_study(
        lower_bounds=lower_bounds,
        lml_values=lml_values,
        rmse_values=rmse_values,
        standardized_length_scales=standardized_length_scales,
        length_scale_bound_hits=np.vstack(length_scale_bound_hit_rows),
        length_scale_axis_labels=length_scale_axis_labels[:standardized_length_scales.shape[1]],
        output_path=output_path,
        show=args.show,
    )
    print("\nSaved lower-bound sensitivity figures:")
    print(f"  - {model_fit_path}")
    print(f"  - {length_scales_path}")


def fit_and_reconstruct_samples(
    args: argparse.Namespace,
    grid_data,
    sample_coordinates: np.ndarray,
    sample_values: np.ndarray,
    random_seed: int,
    kernel_mode: str,
    coordinate_transform: RotationTransform | None,
):
    model_sample_coordinates = maybe_transform_coordinates(
        sample_coordinates,
        coordinate_transform,
    )
    optimizer_seed = resolve_optimizer_seed(args, random_seed)
    model, coordinate_scaler, optimization_diagnostics = fit_gaussian_process(
        sample_coordinates=model_sample_coordinates,
        sample_values=sample_values,
        kernel_mode=kernel_mode,
        length_scale_lower_bound=args.length_scale_lower_bound,
        length_scale_upper_bound=args.length_scale_upper_bound,
        noise_level_initial=args.noise_level_initial,
        noise_level_lower_bound=args.noise_level_lower_bound,
        noise_level_upper_bound=args.noise_level_upper_bound,
        target_transform=args.target_transform,
        n_restarts=args.n_restarts,
        optimizer_seed=optimizer_seed,
    )
    reconstruction = reconstruct_field(
        grid_data=grid_data,
        model=model,
        coordinate_scaler=coordinate_scaler,
        batch_size=args.prediction_batch_size,
        target_transform=args.target_transform,
        clip_negative=args.clip_negative,
        coordinate_transform=coordinate_transform,
    )
    return reconstruction, model, optimization_diagnostics


def run_kernel_comparison_study(
    args: argparse.Namespace,
    grid_data,
    output_path: Path,
) -> None:
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    sample_counts = list(args.sample_size_study_counts)
    seeds = list(args.sample_size_study_seeds)

    model_configs = [
        ("Isotropic", "isotropic", None),
        ("Axis-aligned anisotropic", "anisotropic", None),
    ]
    physical_builders = [
        ("Wind-informed anisotropic", build_wind_coordinate_transform),
        ("Current-informed anisotropic", build_current_coordinate_transform),
    ]
    for label, builder in physical_builders:
        try:
            model_configs.append((label, "anisotropic", builder(args, grid_data)))
        except (FileNotFoundError, ValueError) as exc:
            print(f"\nSkipping {label}: {exc}")

    print(
        "\n=== Kernel comparison study "
        f"({len(model_configs)} models x {len(seeds)} seeds x {len(sample_counts)} counts) ==="
    )
    print("Sampling design: identical sensor locations and values are shared by all models.")

    shared_samples: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for seed in seeds:
        for n_samples in sample_counts:
            sample_coordinates, sample_values, _ = sample_sensor_points(
                grid_data=grid_data,
                n_samples=n_samples,
                noise_std=args.noise_std,
                random_seed=seed,
            )
            shared_samples[(seed, n_samples)] = (sample_coordinates, sample_values)

    results_by_model: dict[str, dict[str, np.ndarray]] = {}
    for label, kernel_mode, coordinate_transform in model_configs:
        rmse_matrix = np.full((len(seeds), len(sample_counts)), np.nan)
        mae_matrix = np.full((len(seeds), len(sample_counts)), np.nan)
        r2_matrix = np.full((len(seeds), len(sample_counts)), np.nan)
        lower_bound_hit_matrix = np.zeros((len(seeds), len(sample_counts)), dtype=bool)
        upper_bound_hit_matrix = np.zeros((len(seeds), len(sample_counts)), dtype=bool)
        optimizer_failure_matrix = np.zeros((len(seeds), len(sample_counts)), dtype=bool)

        print(f"\n  Model: {label}")
        for seed_index, seed in enumerate(seeds):
            print(f"    Seed {seed}:")
            for count_index, n_samples in enumerate(sample_counts):
                print(f"      n_samples={n_samples} ...", end=" ", flush=True)
                try:
                    sample_coordinates, sample_values = shared_samples[(seed, n_samples)]
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        reconstruction, model, optimization_diagnostics = fit_and_reconstruct_samples(
                            args=args,
                            grid_data=grid_data,
                            sample_coordinates=sample_coordinates,
                            sample_values=sample_values,
                            random_seed=seed,
                            kernel_mode=kernel_mode,
                            coordinate_transform=coordinate_transform,
                        )
                    rmse_matrix[seed_index, count_index] = reconstruction.rmse
                    mae_matrix[seed_index, count_index] = reconstruction.mae
                    r2_matrix[seed_index, count_index] = reconstruction.r2
                    lower_bound_hit = optimization_diagnostics.length_scale_lower_bound_hit
                    upper_bound_hit = optimization_diagnostics.length_scale_upper_bound_hit
                    lower_bound_hit_matrix[seed_index, count_index] = lower_bound_hit
                    upper_bound_hit_matrix[seed_index, count_index] = upper_bound_hit
                    optimizer_failure_matrix[seed_index, count_index] = (
                        optimization_diagnostics.optimizer_failure_count > 0
                    )
                    print(
                        f"RMSE={reconstruction.rmse:.6g}  "
                        f"R^2={reconstruction.r2:.4f}  "
                        f"neg={100.0 * reconstruction.negative_prediction_fraction:.3g}%"
                    )
                    print(
                        "        optimization: "
                        f"{format_gp_optimization_summary(optimization_diagnostics)}"
                    )
                except ValueError as exc:
                    print(f"skipped ({exc})")

        results_by_model[label] = {
            "rmse": rmse_matrix,
            "mae": mae_matrix,
            "r2": r2_matrix,
            "length_scale_lower_bound_hit": lower_bound_hit_matrix,
            "length_scale_upper_bound_hit": upper_bound_hit_matrix,
            "optimizer_failure": optimizer_failure_matrix,
        }

    print("\n=== Kernel comparison summary ===")
    for label, model_results in results_by_model.items():
        print(f"\n  {label}:")
        for count_index, n_samples in enumerate(sample_counts):
            rmse_values = model_results["rmse"][:, count_index]
            r2_values = model_results["r2"][:, count_index]
            std_ddof = 1 if len(seeds) > 1 else 0
            print(
                f"    n_samples={n_samples}: "
                f"RMSE={np.nanmean(rmse_values):.6g} +/- {np.nanstd(rmse_values, ddof=std_ddof):.6g}, "
                f"R^2={np.nanmean(r2_values):.6g} +/- {np.nanstd(r2_values, ddof=std_ddof):.6g}"
            )
        bound_hits = int(np.count_nonzero(model_results["length_scale_lower_bound_hit"]))
        if bound_hits:
            print(
                f"    Diagnostic: {bound_hits} fit(s) reached the length-scale lower bound."
            )
        upper_bound_hits = int(
            np.count_nonzero(model_results["length_scale_upper_bound_hit"])
        )
        if upper_bound_hits:
            print(
                f"    Diagnostic: {upper_bound_hits} fit(s) reached the length-scale upper bound."
            )
        optimizer_failures = int(np.count_nonzero(model_results["optimizer_failure"]))
        if optimizer_failures:
            print(
                f"    Diagnostic: {optimizer_failures} fit(s) had at least one failed optimizer run."
            )

    plot_kernel_comparison_multiseed(
        n_samples_list=sample_counts,
        results_by_model=results_by_model,
        output_path=output_path,
        show=args.show,
    )
    print(f"\nSaved kernel comparison study: {output_path}")

    panel_paths = plot_kernel_comparison_multiseed_panels(
        n_samples_list=sample_counts,
        results_by_model=results_by_model,
        output_path=output_path,
        show=args.show,
    )
    print("Saved separate kernel comparison panels:")
    for panel_path in panel_paths:
        print(f"  - {panel_path}")


# Run the GP pipeline for multiple sample counts and plot reconstruction metrics vs n_samples.
def run_sample_size_study(
    args: argparse.Namespace,
    grid_data,
    figure_path: Path,
    coordinate_transform: RotationTransform | None,
) -> None:
    sample_counts = list(args.sample_size_study_counts)
    rmse_list, mae_list, r2_list, valid_counts = [], [], [], []

    print("\n=== Sample size study ===")
    for n in sample_counts:
        print(f"  n_samples={n} ...", end=" ", flush=True)
        try:
            sample_coordinates, sample_values, _ = sample_sensor_points(
                grid_data=grid_data,
                n_samples=n,
                noise_std=args.noise_std,
                random_seed=args.random_seed,
            )
            model_sample_coordinates = maybe_transform_coordinates(
                sample_coordinates,
                coordinate_transform,
            )
            optimizer_seed = resolve_optimizer_seed(args, args.random_seed)
            model, coordinate_scaler, optimization_diagnostics = fit_gaussian_process(
                sample_coordinates=model_sample_coordinates,
                sample_values=sample_values,
                kernel_mode=args.kernel_mode,
                length_scale_lower_bound=args.length_scale_lower_bound,
                length_scale_upper_bound=args.length_scale_upper_bound,
                noise_level_initial=args.noise_level_initial,
                noise_level_lower_bound=args.noise_level_lower_bound,
                noise_level_upper_bound=args.noise_level_upper_bound,
                target_transform=args.target_transform,
                n_restarts=args.n_restarts,
                optimizer_seed=optimizer_seed,
            )
            reconstruction = reconstruct_field(
                grid_data=grid_data,
                model=model,
                coordinate_scaler=coordinate_scaler,
                batch_size=args.prediction_batch_size,
                target_transform=args.target_transform,
                clip_negative=args.clip_negative,
                coordinate_transform=coordinate_transform,
            )
            rmse_list.append(reconstruction.rmse)
            mae_list.append(reconstruction.mae)
            r2_list.append(reconstruction.r2)
            valid_counts.append(n)
            print(
                f"RMSE={reconstruction.rmse:.6g}  "
                f"R^2={reconstruction.r2:.4f}  "
                f"neg={100.0 * reconstruction.negative_prediction_fraction:.3g}%"
            )
            print(
                "    optimization: "
                f"{format_gp_optimization_summary(optimization_diagnostics)}"
            )
        except ValueError as exc:
            print(f"skipped ({exc})")

    study_path = figure_path.parent / f"{figure_path.stem}_sample_size_study.png"
    plot_sample_size_study(
        n_samples_list=valid_counts,
        rmse_list=rmse_list,
        mae_list=mae_list,
        r2_list=r2_list,
        output_path=study_path,
        show=args.show,
    )
    print(f"Saved sample size study: {study_path}")
    study_panel_paths = plot_sample_size_study_panels(
        n_samples_list=valid_counts,
        rmse_list=rmse_list,
        mae_list=mae_list,
        r2_list=r2_list,
        output_path=study_path,
        show=args.show,
    )
    print("Saved separate sample size study panels:")
    for panel_path in study_panel_paths:
        print(f"  - {panel_path}")


# Run the GP pipeline for multiple sample counts AND multiple random seeds.
# Produces a plot with mean ± 1 std bands across seeds.
def run_sample_size_study_multiseed(
    args: argparse.Namespace,
    grid_data,
    figure_path: Path,
    coordinate_transform: RotationTransform | None,
) -> None:
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    sample_counts = list(args.sample_size_study_counts)
    seeds = list(args.sample_size_study_seeds)

    # matrices: rows = seeds, columns = sample counts
    rmse_matrix = np.full((len(seeds), len(sample_counts)), np.nan)
    mae_matrix  = np.full((len(seeds), len(sample_counts)), np.nan)
    r2_matrix   = np.full((len(seeds), len(sample_counts)), np.nan)

    print(f"\n=== Multi-seed sample size study ({len(seeds)} seeds × {len(sample_counts)} counts) ===")

    for s_idx, seed in enumerate(seeds):
        print(f"\n  Seed {seed}:")
        for n_idx, n in enumerate(sample_counts):
            print(f"    n_samples={n} ...", end=" ", flush=True)
            try:
                sample_coordinates, sample_values, _ = sample_sensor_points(
                    grid_data=grid_data,
                    n_samples=n,
                    noise_std=args.noise_std,
                    random_seed=seed,
                )
                model_sample_coordinates = maybe_transform_coordinates(
                    sample_coordinates,
                    coordinate_transform,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    optimizer_seed = resolve_optimizer_seed(args, seed)
                    model, coordinate_scaler, optimization_diagnostics = fit_gaussian_process(
                        sample_coordinates=model_sample_coordinates,
                        sample_values=sample_values,
                        kernel_mode=args.kernel_mode,
                        length_scale_lower_bound=args.length_scale_lower_bound,
                        length_scale_upper_bound=args.length_scale_upper_bound,
                        noise_level_initial=args.noise_level_initial,
                        noise_level_lower_bound=args.noise_level_lower_bound,
                        noise_level_upper_bound=args.noise_level_upper_bound,
                        target_transform=args.target_transform,
                        n_restarts=args.n_restarts,
                        optimizer_seed=optimizer_seed,
                    )
                reconstruction = reconstruct_field(
                    grid_data=grid_data,
                    model=model,
                    coordinate_scaler=coordinate_scaler,
                    batch_size=args.prediction_batch_size,
                    target_transform=args.target_transform,
                    clip_negative=args.clip_negative,
                    coordinate_transform=coordinate_transform,
                )
                rmse_matrix[s_idx, n_idx] = reconstruction.rmse
                mae_matrix[s_idx, n_idx]  = reconstruction.mae
                r2_matrix[s_idx, n_idx]   = reconstruction.r2
                print(
                    f"RMSE={reconstruction.rmse:.6g}  "
                    f"R^2={reconstruction.r2:.4f}  "
                    f"neg={100.0 * reconstruction.negative_prediction_fraction:.3g}%"
                )
                print(
                    "      optimization: "
                    f"{format_gp_optimization_summary(optimization_diagnostics)}"
                )
            except ValueError as exc:
                print(f"skipped ({exc})")

    # Drop columns where all seeds failed
    valid_mask = ~np.all(np.isnan(rmse_matrix), axis=0)
    valid_counts = [n for n, v in zip(sample_counts, valid_mask) if v]
    rmse_matrix = rmse_matrix[:, valid_mask]
    mae_matrix  = mae_matrix[:, valid_mask]
    r2_matrix   = r2_matrix[:, valid_mask]

    study_path = figure_path.parent / f"{figure_path.stem}_multiseed_study.png"
    plot_sample_size_study_multiseed(
        n_samples_list=valid_counts,
        rmse_matrix=rmse_matrix,
        mae_matrix=mae_matrix,
        r2_matrix=r2_matrix,
        output_path=study_path,
        show=args.show,
    )
    print(f"\nSaved multi-seed study: {study_path}")

    panel_paths = plot_sample_size_study_multiseed_panels(
        n_samples_list=valid_counts,
        rmse_matrix=rmse_matrix,
        mae_matrix=mae_matrix,
        r2_matrix=r2_matrix,
        output_path=study_path,
        show=args.show,
    )
    print("Saved separate multi-seed panels:")
    for panel_path in panel_paths:
        print(f"  - {panel_path}")
