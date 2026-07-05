# Main reconstruction workflow.

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from pollutant_gp.data import (
    choose_time_index,
    prepare_grid_data,
    print_dataset_structure,
    validate_dataset_layout,
)

# Optional dataset inspection utility
from pollutant_gp.inspection import inspect_netcdf_files

# Gaussian Process model training
from pollutant_gp.model import fit_gaussian_process

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
    plot_reconstruction,
    plot_reconstruction_panels,
    plot_sample_size_study,
    plot_sample_size_study_panels,
    plot_sample_size_study_multiseed,
    plot_sample_size_study_multiseed_panels,
    plot_valid_domain,
)
from pollutant_gp.wind import compute_wind_orientation, parse_time_label

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


def build_coordinate_transform(
    args: argparse.Namespace,
    grid_data,
) -> RotationTransform | None:
    if not args.physically_informed:
        return None

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

    if args.kernel_mode != "anisotropic":
        print("Note: wind-informed rotations are most meaningful with --kernel-mode anisotropic.")

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
        print(f"Kernel mode: {args.kernel_mode}")
        print(f"Physically informed rotation: {args.physically_informed}")
        if args.physically_informed:
            print(f"Wind file: {args.wind_file}")
            print(f"Wind averaging window: {args.wind_average_hours:g} h")
            print(f"Wind direction convention: {args.wind_direction_convention}")
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

    coordinate_transform = build_coordinate_transform(args, grid_data)

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
    model, coordinate_scaler = fit_gaussian_process(
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
        random_seed=args.random_seed,
    )
    print(f"Learned kernel: {model.kernel_}")

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
            model, coordinate_scaler = fit_gaussian_process(
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
                random_seed=args.random_seed,
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
            print(f"RMSE={reconstruction.rmse:.6g}  R^2={reconstruction.r2:.4f}")
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
                    model, coordinate_scaler = fit_gaussian_process(
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
                        random_seed=seed,
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
                print(f"RMSE={reconstruction.rmse:.6g}  R^2={reconstruction.r2:.4f}")
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
