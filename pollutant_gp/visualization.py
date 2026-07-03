# Plotting helpers.

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from pollutant_gp.types import GridData, ReconstructionResult

# Helper function to generate output paths for individual panels.
def _panel_output_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")

# Prepare the data arrays and shared color limits for the 2x2 reconstruction panel plot.
def _reconstruction_panel_data(
    grid_data: GridData,
    reconstruction: ReconstructionResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    truth = np.where(grid_data.valid_mask, grid_data.field, np.nan)
    prediction = np.where(grid_data.valid_mask, reconstruction.mean_field, np.nan)
    uncertainty = np.where(grid_data.valid_mask, reconstruction.std_field, np.nan)
    absolute_error = np.abs(prediction - truth)

    finite_truth = truth[np.isfinite(truth)]
    shared_vmin = float(np.nanmin(finite_truth))
    shared_vmax = float(np.nanmax(finite_truth))

    return truth, prediction, uncertainty, absolute_error, shared_vmin, shared_vmax

# Draw a single spatial panel with the given data and formatting. Optionally overlay sample locations.
def _draw_spatial_panel(
    axis: plt.Axes,
    grid_data: GridData,
    values: np.ndarray,
    title: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    sample_coordinates: np.ndarray | None = None,
):
    mesh = axis.pcolormesh(
        grid_data.x_grid,
        grid_data.y_grid,
        values,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_title(title)
    axis.set_xlabel(grid_data.x_dim)
    axis.set_ylabel(grid_data.y_dim)
    axis.set_aspect("equal", adjustable="box")

    if sample_coordinates is not None:
        axis.scatter(
            sample_coordinates[:, 0],
            sample_coordinates[:, 1],
            s=14,
            c="white",
            edgecolors="black",
            linewidths=0.5,
            label="Synthetic sensors",
        )
        axis.legend(loc="upper right", fontsize=9)

    return mesh

# Plot the valid domain mask as a binary colormap, with a legend for land vs sea.
def plot_valid_domain(
    grid_data: GridData,
    output_path: Path,
    show: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    domain = np.where(grid_data.valid_mask, 1.0, 0.0)
    cmap = ListedColormap(["#f2f2f2", "#2f80ed"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    mesh = axis.pcolormesh(
        grid_data.x_grid,
        grid_data.y_grid,
        domain,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    axis.set_title("Valid marine domain")
    axis.set_xlabel(grid_data.x_dim)
    axis.set_ylabel(grid_data.y_dim)
    axis.set_aspect("equal", adjustable="box")

    colorbar = fig.colorbar(mesh, ax=axis, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["NaN / land", "Finite / sea"])

    fig.savefig(output_path, dpi=220)

    if show:
        plt.show()

    plt.close(fig)


# Plot a standalone concentration map for dataset inspection and report figures.
def plot_concentration_map(
    grid_data: GridData,
    output_path: Path,
    show: bool,
    display_threshold: float = 0.0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    field = np.where(grid_data.valid_mask, grid_data.field, np.nan)
    visible_concentration = np.where(field > display_threshold, field, np.nan)
    finite_values = field[np.isfinite(field)]
    vmax = float(np.nanmax(finite_values))

    domain = np.where(grid_data.valid_mask, 1.0, 0.0)
    domain_cmap = ListedColormap(["white", "#86cce3"])
    domain_norm = BoundaryNorm([-0.5, 0.5, 1.5], domain_cmap.N)

    concentration_cmap = plt.get_cmap("YlOrRd").copy()
    concentration_cmap.set_bad((0.0, 0.0, 0.0, 0.0))

    fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)

    axis.pcolormesh(
        grid_data.x_grid,
        grid_data.y_grid,
        domain,
        shading="auto",
        cmap=domain_cmap,
        norm=domain_norm,
    )
    mesh = axis.pcolormesh(
        grid_data.x_grid,
        grid_data.y_grid,
        visible_concentration,
        shading="auto",
        cmap=concentration_cmap,
        vmin=0.0,
        vmax=vmax,
    )

    if finite_values.size:
        max_flat_index = int(np.nanargmax(field))
        max_y_index, max_x_index = np.unravel_index(max_flat_index, field.shape)
        axis.scatter(
            grid_data.x_grid[max_y_index, max_x_index],
            grid_data.y_grid[max_y_index, max_x_index],
            marker="*",
            s=120,
            c="yellow",
            edgecolors="black",
            linewidths=0.8,
            label="Maximum concentration",
            zorder=5,
        )
        axis.legend(loc="upper right")

    title_parts = ["Concentration field"]
    if grid_data.selected_time_label is not None:
        title_parts.append(f"time = {grid_data.selected_time_label}")
    if display_threshold > 0.0:
        title_parts.append(f"display threshold = {display_threshold:g}")
    axis.set_title(" | ".join(title_parts), fontsize=11)
    axis.set_xlabel(f"{grid_data.x_dim} (m)")
    axis.set_ylabel(f"{grid_data.y_dim} (m)")
    axis.set_aspect("equal", adjustable="box")

    colorbar = fig.colorbar(mesh, ax=axis)
    colorbar.set_label("Concentration")

    fig.savefig(output_path, dpi=220)

    if show:
        plt.show()

    plt.close(fig)


# Generate a 2x2 panel plot showing the ground truth, GP reconstruction, predictive uncertainty, and absolute error.
def plot_reconstruction(
    grid_data: GridData,
    reconstruction: ReconstructionResult,
    sample_coordinates: np.ndarray,
    output_path: Path,
    show: bool,
) -> None:
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    truth, prediction, uncertainty, absolute_error, shared_vmin, shared_vmax = (
        _reconstruction_panel_data(grid_data, reconstruction)
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    axes = axes.ravel()

    panels = [
        ("Ground truth", truth, "viridis", shared_vmin, shared_vmax, True),
        ("Gaussian Process reconstruction", prediction, "viridis", shared_vmin, shared_vmax, True),
        ("Predictive uncertainty (standard deviation)", uncertainty, "magma", None, None, False),
        ("Absolute reconstruction error", absolute_error, "inferno", None, None, False),
    ]

    for axis, (title, values, cmap, vmin, vmax, draw_samples) in zip(axes, panels):
        samples = sample_coordinates if draw_samples else None
        mesh = _draw_spatial_panel(axis, grid_data, values, title, cmap, vmin, vmax, samples)
        fig.colorbar(mesh, ax=axis)

    subtitle = []
    if grid_data.selected_time_label is not None:
        subtitle.append(f"time = {grid_data.selected_time_label}")
    subtitle.append(f"MSE = {reconstruction.mse:.6g}")
    subtitle.append(f"RMSE = {reconstruction.rmse:.6g}")
    fig.suptitle("Stationary concentration field reconstruction | " + " | ".join(subtitle))
    fig.savefig(output_path, dpi=200)

    if show:
        plt.show()

    plt.close(fig)

# Generate separate panel plots for each of the reconstruction metrics.
def plot_reconstruction_panels(
    grid_data: GridData,
    reconstruction: ReconstructionResult,
    sample_coordinates: np.ndarray,
    output_path: Path,
    show: bool,
) -> list[Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    truth, prediction, uncertainty, absolute_error, shared_vmin, shared_vmax = (
        _reconstruction_panel_data(grid_data, reconstruction)
    )
    panels = [
        ("Ground truth", truth, "viridis", shared_vmin, shared_vmax, True, "ground_truth"),
        (
            "Gaussian Process reconstruction",
            prediction,
            "viridis",
            shared_vmin,
            shared_vmax,
            True,
            "gp_reconstruction",
        ),
        (
            "Predictive uncertainty (standard deviation)",
            uncertainty,
            "magma",
            None,
            None,
            False,
            "predictive_uncertainty",
        ),
        (
            "Absolute reconstruction error",
            absolute_error,
            "inferno",
            None,
            None,
            False,
            "absolute_error",
        ),
    ]

    saved_paths = []
    for title, values, cmap, vmin, vmax, draw_samples, suffix in panels:
        panel_path = _panel_output_path(output_path, suffix)
        fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
        samples = sample_coordinates if draw_samples else None
        mesh = _draw_spatial_panel(axis, grid_data, values, title, cmap, vmin, vmax, samples)
        fig.colorbar(mesh, ax=axis)
        fig.savefig(panel_path, dpi=220)

        if show:
            plt.show()

        plt.close(fig)
        saved_paths.append(panel_path)

    return saved_paths


# Curve of reconstruction error metrics as a function of the number of sensor samples.
# Runs the full GP pipeline for each sample count in n_samples_list and plots RMSE, MAE, R^2.
def plot_sample_size_study(
    n_samples_list: Sequence[int],
    rmse_list: Sequence[float],
    mae_list: Sequence[float],
    r2_list: Sequence[float],
    output_path: Path,
    show: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    for ax, values, label, color in zip(
        axes,
        [rmse_list, mae_list, r2_list],
        ["RMSE", "MAE", "R²"],
        ["steelblue", "darkorange", "seagreen"],
    ):
        ax.plot(n_samples_list, values, "o-", color=color, linewidth=1.5, markersize=5)
        ax.set_xlabel("Number of sensor samples")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs number of samples")
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Sample size study: reconstruction quality vs sensor count")
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)

# Plot individual panels for each reconstruction error metric
def plot_sample_size_study_panels(
    n_samples_list: Sequence[int],
    rmse_list: Sequence[float],
    mae_list: Sequence[float],
    r2_list: Sequence[float],
    output_path: Path,
    show: bool,
) -> list[Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("RMSE", rmse_list, "steelblue", "rmse"),
        ("MAE", mae_list, "darkorange", "mae"),
        ("R²", r2_list, "seagreen", "r2"),
    ]

    saved_paths = []
    for label, values, color, suffix in metrics:
        panel_path = _panel_output_path(output_path, suffix)
        fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        axis.plot(n_samples_list, values, "o-", color=color, linewidth=1.8, markersize=6)
        axis.set_xlabel("Number of sensor samples")
        axis.set_ylabel(label)
        axis.set_title(f"{label} vs number of samples")
        axis.grid(True, linestyle="--", alpha=0.5)
        fig.savefig(panel_path, dpi=220)

        if show:
            plt.show()

        plt.close(fig)
        saved_paths.append(panel_path)

    return saved_paths


# Multi-seed sample size study

# Plot the sample size study averaged over multiple random seeds.
# For each metric, draws the mean curve with a shaded ±1 std band and individual seed curves.
def plot_sample_size_study_multiseed(
    n_samples_list: Sequence[int],
    rmse_matrix: np.ndarray,
    mae_matrix: np.ndarray,
    r2_matrix: np.ndarray,
    output_path: Path,
    show: bool,
) -> None:
  
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(n_samples_list)
    n_seeds = rmse_matrix.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    metrics = [
        ("RMSE", rmse_matrix, "steelblue"),
        ("MAE",  mae_matrix,  "darkorange"),
        ("R²",   r2_matrix,   "seagreen"),
    ]

    for ax, (label, matrix, color) in zip(axes, metrics):
        mean = matrix.mean(axis=0)
        std  = matrix.std(axis=0)

        for row in matrix:
            ax.plot(x, row, color=color, alpha=0.20, linewidth=0.8)

        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.25, label="±1 std")
        ax.plot(x, mean, "o-", color=color, linewidth=2.0, markersize=6,
                label=f"Mean ({n_seeds} seeds)")

        ax.set_xlabel("Number of sensor samples")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs number of samples")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle(f"Sample size study — {n_seeds} random seeds  |  mean ± 1 std")
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


# Same as above but saves one figure per metric.
def plot_sample_size_study_multiseed_panels(
    n_samples_list: Sequence[int],
    rmse_matrix: np.ndarray,
    mae_matrix: np.ndarray,
    r2_matrix: np.ndarray,
    output_path: Path,
    show: bool,
) -> list[Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(n_samples_list)
    n_seeds = rmse_matrix.shape[0]

    metrics = [
        ("RMSE", rmse_matrix, "steelblue",  "rmse"),
        ("MAE",  mae_matrix,  "darkorange", "mae"),
        ("R²",   r2_matrix,   "seagreen",   "r2"),
    ]

    saved_paths = []
    for label, matrix, color, suffix in metrics:
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)

        panel_path = _panel_output_path(output_path, f"multiseed_{suffix}")
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

        for row in matrix:
            ax.plot(x, row, color=color, alpha=0.20, linewidth=0.8)

        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.25, label="±1 std")
        ax.plot(x, mean, "o-", color=color, linewidth=2.0, markersize=6,
                label=f"Mean ({n_seeds} seeds)")

        ax.set_xlabel("Number of sensor samples")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs number of samples  [{n_seeds} seeds]")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

        fig.savefig(panel_path, dpi=220)
        if show:
            plt.show()
        plt.close(fig)
        saved_paths.append(panel_path)

    return saved_paths
