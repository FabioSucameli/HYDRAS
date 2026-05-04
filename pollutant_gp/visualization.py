# Plotting helpers.

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from pollutant_gp.types import GridData, ReconstructionResult


def _panel_output_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")


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
        axis.legend(loc="upper right")

    return mesh


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
        ["RMSE", "MAE", "R2"],
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
        ("R2", r2_list, "seagreen", "r2"),
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
