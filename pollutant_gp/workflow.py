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

# Visualization utilities
from pollutant_gp.visualization import (
    plot_reconstruction,
    plot_reconstruction_panels,
    plot_sample_size_study,
    plot_sample_size_study_panels,
)

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

    # Dataset structure printing mode: load the specified NetCDF file, print its structure, and exit.
    if args.print_dataset:
        ds = xr.open_dataset(args.nc_file)
        print(ds)
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

        # # Extract the concentration variable.
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
    print("\n=== Selected field summary ===")
    print(f"Field shape: {grid_data.field.shape}")
    print(f"Valid cells: {grid_data.valid_mask.sum()} / {grid_data.valid_mask.size}")
    print(f"Ground-truth min: {np.nanmin(valid_values):.6g}")
    print(f"Ground-truth max: {np.nanmax(valid_values):.6g}")
    print(f"Ground-truth mean: {np.nanmean(valid_values):.6g}")

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

    # Train Gaussian Process model
    # The GP learns a mapping from spatial coordinates to concentration: (x, y) -> C
    # using only the sparse synthetic measurements.
    print("\n=== Fitting Gaussian Process ===")
    model, coordinate_scaler = fit_gaussian_process(
        sample_coordinates=sample_coordinates,
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

    # --- Optional sample size study ---
    if args.sample_size_study:
        run_sample_size_study(args, grid_data, figure_path)


# Run the GP pipeline for multiple sample counts and plot reconstruction metrics vs n_samples.
def run_sample_size_study(
    args: argparse.Namespace,
    grid_data,
    figure_path: Path,
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
            model, coordinate_scaler = fit_gaussian_process(
                sample_coordinates=sample_coordinates,
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
