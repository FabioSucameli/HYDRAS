# Plotting helpers.

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from pollutant_gp.style import KERNEL_COMPARISON_STYLES
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


# Compute a plotting standard deviation that also works with a single seed.
def _std_for_plot(matrix: np.ndarray) -> np.ndarray:
    ddof = 1 if matrix.shape[0] > 1 else 0
    return np.nanstd(matrix, axis=0, ddof=ddof)


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


# Plot model-fit metrics and standardized length scales across lower bounds.
def plot_length_scale_lower_bound_study(
    lower_bounds: Sequence[float],
    lml_values: Sequence[float],
    rmse_values: Sequence[float],
    standardized_length_scales: np.ndarray,
    length_scale_bound_hits: np.ndarray,
    length_scale_axis_labels: Sequence[str],
    output_path: Path,
    show: bool,
) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(lower_bounds, dtype=float)
    standardized_length_scales = np.asarray(standardized_length_scales, dtype=float)
    length_scale_bound_hits = np.asarray(length_scale_bound_hits, dtype=bool)

    model_fit_path = _panel_output_path(output_path, "model_fit")
    length_scales_path = _panel_output_path(output_path, "learned_length_scales")

    fit_figure, fit_axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    panels = (
        (fit_axes[0], lml_values, "Log-marginal likelihood", "#0072B2"),
        (fit_axes[1], rmse_values, "RMSE", "#D55E00"),
    )
    for axis, values, ylabel, color in panels:
        axis.plot(x, values, "o-", color=color, linewidth=2.0, markersize=6)
        axis.set_xlabel("Length-scale lower bound (standardized coordinates)")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{ylabel} vs lower bound")
        axis.grid(True, linestyle="-", alpha=0.25)

    fit_figure.suptitle("Lower-bound sensitivity of model fit")
    fit_figure.savefig(model_fit_path, dpi=220)
    if show:
        plt.show()
    plt.close(fit_figure)

    length_figure, length_axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    length_axis.plot(
        x,
        x,
        "--",
        color="#222222",
        linewidth=1.6,
        label="Imposed lower bound",
        zorder=4,
    )
    colors = ("#0072B2", "#CC79A7")
    for dimension in range(standardized_length_scales.shape[1]):
        color = colors[dimension % len(colors)]
        axis_label = length_scale_axis_labels[dimension]
        length_axis.plot(
            x,
            standardized_length_scales[:, dimension],
            "o-",
            color=color,
            linewidth=2.0,
            markersize=6,
            label=f"Learned: {axis_label}",
            zorder=2 + dimension,
        )
        hit_mask = length_scale_bound_hits[:, dimension]
        if np.any(hit_mask):
            length_axis.scatter(
                x[hit_mask],
                standardized_length_scales[hit_mask, dimension],
                s=90,
                facecolors="none",
                edgecolors="#222222",
                linewidths=1.3,
                zorder=5,
            )

    length_axis.set_xlabel("Length-scale lower bound (standardized coordinates)")
    length_axis.set_ylabel("Learned length scale (standardized coordinates)")
    length_axis.set_title("Learned length-scales vs imposed lower bound")
    length_axis.grid(True, linestyle="-", alpha=0.25)
    length_axis.legend(fontsize=9)

    length_figure.savefig(length_scales_path, dpi=220)
    if show:
        plt.show()
    plt.close(length_figure)

    return model_fit_path, length_scales_path


# Plot fit quality and solution stability across length-scale upper bounds.
def plot_length_scale_upper_bound_study(
    upper_bounds: Sequence[float],
    profile_labels: Sequence[str],
    lml_matrix: np.ndarray,
    rmse_matrix: np.ndarray,
    standardized_length_scales: np.ndarray,
    map_delta_matrix: np.ndarray,
    output_path: Path,
    show: bool,
) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(upper_bounds, dtype=float)
    lml_matrix = np.asarray(lml_matrix, dtype=float)
    rmse_matrix = np.asarray(rmse_matrix, dtype=float)
    standardized_length_scales = np.asarray(standardized_length_scales, dtype=float)
    map_delta_matrix = np.asarray(map_delta_matrix, dtype=float)
    colors = ("#0072B2", "#D55E00")
    markers = ("o", "s")

    model_fit_path = _panel_output_path(output_path, "model_fit")
    stability_path = _panel_output_path(output_path, "solution_stability")

    fit_figure, fit_axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    for profile_index, profile_label in enumerate(profile_labels):
        color = colors[profile_index % len(colors)]
        marker = markers[profile_index % len(markers)]
        fit_axes[0].plot(
            x,
            lml_matrix[profile_index],
            marker=marker,
            color=color,
            linewidth=2.0,
            markersize=7,
            label=profile_label,
        )
        fit_axes[1].plot(
            x,
            rmse_matrix[profile_index],
            marker=marker,
            color=color,
            linewidth=2.0,
            markersize=7,
            label=profile_label,
        )
    for axis, ylabel in zip(
        fit_axes,
        ("Final log-marginal likelihood", "RMSE"),
        strict=True,
    ):
        axis.set_xscale("log")
        axis.set_xticks(x)
        axis.set_xticklabels([f"{value:g}" for value in x])
        axis.set_xlabel("Length-scale upper bound (standardized coordinates)")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{ylabel} vs upper bound")
        axis.grid(True, which="both", linestyle="-", alpha=0.25)
        axis.legend(fontsize=9)
    fit_figure.suptitle("Length-scale upper-bound sensitivity of model fit")
    fit_figure.savefig(model_fit_path, dpi=220)
    if show:
        plt.show()
    plt.close(fit_figure)

    stability_figure, stability_axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.6),
        constrained_layout=True,
    )
    length_axis, map_axis = stability_axes
    length_axis.plot(
        x,
        x,
        color="#222222",
        linestyle=":",
        linewidth=1.5,
        label="Imposed upper bound",
    )
    for profile_index, profile_label in enumerate(profile_labels):
        color = colors[profile_index % len(colors)]
        marker = markers[profile_index % len(markers)]
        length_axis.plot(
            x,
            standardized_length_scales[profile_index, :, 0],
            marker=marker,
            color=color,
            linestyle="-",
            linewidth=2.0,
            markersize=7,
            label=f"{profile_label}: along transport",
        )
        length_axis.plot(
            x,
            standardized_length_scales[profile_index, :, 1],
            marker=marker,
            color=color,
            linestyle="--",
            linewidth=1.8,
            markersize=6,
            label=f"{profile_label}: across transport",
        )
        map_axis.plot(
            x,
            np.maximum(map_delta_matrix[profile_index], 1e-12),
            marker=marker,
            color=color,
            linewidth=2.0,
            markersize=7,
            label=profile_label,
        )

    length_axis.set_xscale("log")
    length_axis.set_yscale("log")
    length_axis.set_xticks(x)
    length_axis.set_xticklabels([f"{value:g}" for value in x])
    length_axis.set_xlabel("Length-scale upper bound (standardized coordinates)")
    length_axis.set_ylabel("Optimized length scale (standardized coordinates)")
    length_axis.set_title("Learned length-scales")
    length_axis.grid(True, which="both", linestyle="-", alpha=0.25)
    length_axis.legend(fontsize=8)

    map_axis.set_xscale("log")
    map_axis.set_yscale("log")
    map_axis.set_xticks(x)
    map_axis.set_xticklabels([f"{value:g}" for value in x])
    map_axis.set_xlabel("Length-scale upper bound (standardized coordinates)")
    map_axis.set_ylabel("Maximum map difference vs UB=100 (floor: 1e-12)")
    map_axis.set_title("Reconstruction stability")
    map_axis.grid(True, which="both", linestyle="-", alpha=0.25)
    map_axis.legend(fontsize=9)

    stability_figure.suptitle("Upper-bound activity and solution stability")
    stability_figure.savefig(stability_path, dpi=220)
    if show:
        plt.show()
    plt.close(stability_figure)

    return model_fit_path, stability_path


# Plot local fit sensitivity when one fixed RBF length scale is perturbed at a time.
def plot_length_scale_local_sensitivity_study(
    factors: Sequence[float],
    parallel_lml: Sequence[float],
    perpendicular_lml: Sequence[float],
    parallel_rmse: Sequence[float],
    perpendicular_rmse: Sequence[float],
    parallel_optimizer_success: Sequence[bool],
    perpendicular_optimizer_success: Sequence[bool],
    output_path: Path,
    show: bool,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(factors, dtype=float)
    parallel_optimizer_success = np.asarray(parallel_optimizer_success, dtype=bool)
    perpendicular_optimizer_success = np.asarray(
        perpendicular_optimizer_success,
        dtype=bool,
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    panels = (
        (
            axes[0],
            parallel_lml,
            perpendicular_lml,
            "Final log-marginal likelihood",
        ),
        (axes[1], parallel_rmse, perpendicular_rmse, "RMSE"),
    )
    for axis, parallel_values, perpendicular_values, ylabel in panels:
        axis.plot(
            x,
            parallel_values,
            color="#0072B2",
            marker="o",
            linewidth=2.0,
            markersize=7,
            label="Parallel length-scale perturbation",
        )
        axis.plot(
            x,
            perpendicular_values,
            color="#D55E00",
            marker="s",
            linewidth=2.0,
            markersize=7,
            label="Perpendicular length-scale perturbation",
        )
        failed_parallel = ~parallel_optimizer_success
        failed_perpendicular = ~perpendicular_optimizer_success
        if np.any(failed_parallel):
            axis.scatter(
                x[failed_parallel],
                np.asarray(parallel_values)[failed_parallel],
                color="#111111",
                marker="x",
                s=100,
                linewidths=2.0,
                zorder=5,
                label="Optimizer did not converge",
            )
        if np.any(failed_perpendicular):
            axis.scatter(
                x[failed_perpendicular],
                np.asarray(perpendicular_values)[failed_perpendicular],
                color="#111111",
                marker="x",
                s=100,
                linewidths=2.0,
                zorder=5,
                label=(
                    None
                    if np.any(failed_parallel)
                    else "Optimizer did not converge"
                ),
            )
        axis.axvline(
            1.0,
            color="#333333",
            linestyle=":",
            linewidth=1.5,
            label="Reference (alpha = 1)",
        )
        axis.set_xticks(x)
        axis.set_xticklabels([f"{value:g}" for value in x])
        axis.set_xlabel("Multiplicative factor alpha")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{ylabel} vs length-scale factor")
        axis.grid(True, linestyle="-", alpha=0.25)
        axis.legend(fontsize=8)

    figure.suptitle("Local one-at-a-time sensitivity of fixed RBF length-scales")
    figure.savefig(output_path, dpi=220)
    if show:
        plt.show()
    plt.close(figure)
    return output_path


# Plot final fit quality and optimized hyperparameters for deterministic initializations.
def plot_optimizer_initialization_study(
    profile_labels: Sequence[str],
    final_lml_values: Sequence[float],
    rmse_values: Sequence[float],
    standardized_length_scales: np.ndarray,
    constant_kernel_values: Sequence[float],
    white_kernel_values: Sequence[float],
    length_scale_lower_bound: float,
    output_path: Path,
    show: bool,
) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(profile_labels))
    standardized_length_scales = np.asarray(standardized_length_scales, dtype=float)
    model_fit_path = _panel_output_path(output_path, "model_fit")
    hyperparameter_path = _panel_output_path(output_path, "optimized_hyperparameters")

    fit_figure, fit_axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    fit_panels = (
        (fit_axes[0], final_lml_values, "Final log-marginal likelihood", "#0072B2"),
        (fit_axes[1], rmse_values, "RMSE", "#D55E00"),
    )
    for axis, values, ylabel, color in fit_panels:
        axis.plot(x, values, "o-", color=color, linewidth=2.0, markersize=7)
        axis.set_xticks(x, profile_labels, rotation=15, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel)
        axis.grid(True, axis="y", linestyle="-", alpha=0.25)

    fit_figure.suptitle("GP optimizer sensitivity to kernel initialization")
    fit_figure.savefig(model_fit_path, dpi=220)
    if show:
        plt.show()
    plt.close(fit_figure)

    hyper_figure, hyper_axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        constrained_layout=True,
    )
    length_axis, variance_axis = hyper_axes
    length_axis.axhline(
        length_scale_lower_bound,
        color="#222222",
        linestyle="--",
        linewidth=1.5,
        label="Lower bound",
    )
    length_axis.plot(
        x,
        standardized_length_scales[:, 0],
        "o-",
        color="#0072B2",
        linewidth=2.0,
        markersize=7,
        label="Along transport",
    )
    length_axis.plot(
        x,
        standardized_length_scales[:, 1],
        "o-",
        color="#CC79A7",
        linewidth=2.0,
        markersize=7,
        label="Across transport",
    )
    length_axis.set_xticks(x, profile_labels, rotation=15, ha="right")
    length_axis.set_ylabel("Optimized length scale (standardized coordinates)")
    length_axis.set_title("Optimized RBF length-scales")
    length_axis.grid(True, axis="y", linestyle="-", alpha=0.25)
    length_axis.legend(fontsize=9)

    variance_axis.plot(
        x,
        constant_kernel_values,
        "o-",
        color="#009E73",
        linewidth=2.0,
        markersize=7,
        label="ConstantKernel",
    )
    variance_axis.plot(
        x,
        white_kernel_values,
        "o-",
        color="#E69F00",
        linewidth=2.0,
        markersize=7,
        label="WhiteKernel",
    )
    variance_axis.set_xticks(x, profile_labels, rotation=15, ha="right")
    variance_axis.set_ylabel("Optimized kernel value")
    variance_axis.set_title("Optimized variance levels")
    variance_axis.set_yscale("log")
    variance_axis.grid(True, which="both", axis="y", linestyle="-", alpha=0.25)
    variance_axis.legend(fontsize=9)

    hyper_figure.suptitle("Optimized kernel hyperparameters")
    hyper_figure.savefig(hyperparameter_path, dpi=220)
    if show:
        plt.show()
    plt.close(hyper_figure)

    return model_fit_path, hyperparameter_path


# Plot internal restart outcomes, nested best-LML budgets, and LML-RMSE alignment.
def plot_optimizer_restart_study(
    optimizer_seeds: Sequence[int],
    run_lml_matrix: np.ndarray,
    run_rmse_matrix: np.ndarray,
    selected_run_indices: Sequence[int],
    restart_budgets: Sequence[int],
    best_lml_matrix: np.ndarray,
    controlled_initialization_lml: float,
    controlled_initialization_rmse: float,
    output_path: Path,
    show: bool,
) -> tuple[Path, Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_lml_matrix = np.asarray(run_lml_matrix, dtype=float)
    run_rmse_matrix = np.asarray(run_rmse_matrix, dtype=float)
    best_lml_matrix = np.asarray(best_lml_matrix, dtype=float)
    run_indices = np.arange(run_lml_matrix.shape[1])
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    markers = ("o", "s", "^", "D")
    line_styles = ("-", "--", ":", "-.")

    internal_path = _panel_output_path(output_path, "internal_run_lml")
    budget_path = _panel_output_path(output_path, "best_lml_so_far")
    alignment_path = _panel_output_path(output_path, "lml_rmse")

    internal_figure, internal_axis = plt.subplots(
        figsize=(8.2, 5.2),
        constrained_layout=True,
    )
    for seed_index, optimizer_seed in enumerate(optimizer_seeds):
        color = colors[seed_index % len(colors)]
        internal_axis.plot(
            run_indices,
            run_lml_matrix[seed_index],
            color=color,
            marker=markers[seed_index % len(markers)],
            linestyle=line_styles[seed_index % len(line_styles)],
            linewidth=1.8,
            markersize=6,
            label=f"Optimizer seed {optimizer_seed}",
        )
        selected_index = selected_run_indices[seed_index]
        internal_axis.scatter(
            selected_index,
            run_lml_matrix[seed_index, selected_index],
            marker="*",
            s=180,
            color=color,
            edgecolors="#222222",
            linewidths=0.8,
            zorder=5,
        )
    internal_axis.axhline(
        controlled_initialization_lml,
        color="#222222",
        linestyle="--",
        linewidth=1.4,
        label=f"Controlled-initialization LML ({controlled_initialization_lml:.3f})",
    )
    internal_axis.set_xlabel("Internal optimizer run (0 = default initialization)")
    internal_axis.set_ylabel("Final log-marginal likelihood")
    internal_axis.set_title("Final LML of every internal optimizer run")
    internal_axis.set_xticks(run_indices)
    internal_axis.grid(True, linestyle="-", alpha=0.25)
    internal_axis.legend(fontsize=9)
    internal_figure.savefig(internal_path, dpi=220)
    if show:
        plt.show()
    plt.close(internal_figure)

    budget_figure, budget_axis = plt.subplots(
        figsize=(7.4, 5.0),
        constrained_layout=True,
    )
    for seed_index, optimizer_seed in enumerate(optimizer_seeds):
        budget_axis.plot(
            restart_budgets,
            best_lml_matrix[seed_index],
            color=colors[seed_index % len(colors)],
            marker=markers[seed_index % len(markers)],
            linestyle=line_styles[seed_index % len(line_styles)],
            linewidth=2.0,
            markersize=7,
            label=f"Optimizer seed {optimizer_seed}",
        )
    budget_axis.axhline(
        controlled_initialization_lml,
        color="#222222",
        linestyle="--",
        linewidth=1.4,
        label=f"Controlled-initialization LML ({controlled_initialization_lml:.3f})",
    )
    budget_axis.set_xlabel("n_restarts_optimizer budget")
    budget_axis.set_ylabel("Best final log-marginal likelihood so far")
    budget_axis.set_title("Nested restart budget comparison")
    budget_axis.set_xticks(restart_budgets)
    budget_axis.grid(True, linestyle="-", alpha=0.25)
    budget_axis.legend(fontsize=9)
    budget_figure.savefig(budget_path, dpi=220)
    if show:
        plt.show()
    plt.close(budget_figure)

    alignment_figure, alignment_axis = plt.subplots(
        figsize=(7.4, 5.2),
        constrained_layout=True,
    )
    for seed_index, optimizer_seed in enumerate(optimizer_seeds):
        color = colors[seed_index % len(colors)]
        alignment_axis.scatter(
            run_lml_matrix[seed_index],
            run_rmse_matrix[seed_index],
            s=65,
            marker=markers[seed_index % len(markers)],
            facecolors=color if seed_index == 0 else "none",
            edgecolors=color,
            linewidths=1.5,
            label=f"Optimizer seed {optimizer_seed}",
        )
        coincident_groups: dict[tuple[float, float], list[int]] = {}
        for run_index, (lml_value, rmse_value) in enumerate(
            zip(run_lml_matrix[seed_index], run_rmse_matrix[seed_index], strict=True)
        ):
            group_key = (round(float(lml_value), 4), round(float(rmse_value), 4))
            coincident_groups.setdefault(group_key, []).append(run_index)

        vertical_offset = 5 if seed_index == 0 else -14
        for (lml_value, rmse_value), grouped_indices in coincident_groups.items():
            if len(grouped_indices) >= 3 and grouped_indices == list(
                range(grouped_indices[0], grouped_indices[-1] + 1)
            ):
                run_label = f"{grouped_indices[0]}-{grouped_indices[-1]}"
            else:
                run_label = ",".join(str(index) for index in grouped_indices)
            alignment_axis.annotate(
                run_label,
                (lml_value, rmse_value),
                xytext=(5, vertical_offset),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )
    alignment_axis.axvline(
        controlled_initialization_lml,
        color="#222222",
        linestyle="--",
        linewidth=1.4,
        label=f"Controlled-initialization LML ({controlled_initialization_lml:.3f})",
    )
    alignment_axis.scatter(
        controlled_initialization_lml,
        controlled_initialization_rmse,
        marker="*",
        s=260,
        color="#CC79A7",
        edgecolors="#222222",
        linewidths=0.9,
        zorder=6,
        label="Best controlled initialization",
    )
    alignment_axis.set_xlabel("Final log-marginal likelihood")
    alignment_axis.set_ylabel("Full-grid RMSE")
    alignment_axis.set_title("Optimizer objective versus reconstruction error")
    alignment_axis.grid(True, linestyle="-", alpha=0.25)
    alignment_axis.legend(fontsize=9)
    alignment_figure.savefig(alignment_path, dpi=220)
    if show:
        plt.show()
    plt.close(alignment_figure)

    return internal_path, budget_path, alignment_path


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


# Plot a multi-model kernel comparison using mean curves and +/- 1 std error bars.
def plot_kernel_comparison_multiseed(
    n_samples_list: Sequence[int],
    results_by_model: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    show: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(n_samples_list)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)
    metrics = [
        ("rmse", "RMSE", "RMSE vs number of sensors"),
        ("r2", "R2", "R2 vs number of sensors"),
    ]

    for axis, (metric_key, ylabel, title) in zip(axes, metrics):
        for label, model_results in results_by_model.items():
            matrix = model_results[metric_key]
            mean = np.nanmean(matrix, axis=0)
            std = _std_for_plot(matrix)
            style = KERNEL_COMPARISON_STYLES.get(label, {})

            axis.errorbar(
                x,
                mean,
                yerr=std,
                color=style.get("color"),
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                linewidth=2.2,
                markersize=6,
                capsize=4,
                elinewidth=1.6,
                label=label,
            )

        axis.set_xlabel("Synthetic sensors")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, linestyle="-", alpha=0.25)
        axis.legend(fontsize=8.5, loc="best", ncol=2)

    fig.suptitle("CL02_V1_SRC131, time index 729: kernel comparison over random seeds")
    fig.set_constrained_layout_pads(h_pad=0.08, hspace=0.08)
    fig.savefig(output_path, dpi=220)
    if show:
        plt.show()
    plt.close(fig)


# Save separate RMSE and R2 panels for the multi-model kernel comparison.
def plot_kernel_comparison_multiseed_panels(
    n_samples_list: Sequence[int],
    results_by_model: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    show: bool,
) -> list[Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(n_samples_list)
    metrics = [
        ("rmse", "RMSE", "RMSE vs number of sensors", "rmse"),
        ("r2", "R2", "R2 vs number of sensors", "r2"),
    ]

    saved_paths = []
    for metric_key, ylabel, title, suffix in metrics:
        panel_path = _panel_output_path(output_path, f"kernel_comparison_{suffix}")
        fig, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

        for label, model_results in results_by_model.items():
            matrix = model_results[metric_key]
            mean = np.nanmean(matrix, axis=0)
            std = _std_for_plot(matrix)
            style = KERNEL_COMPARISON_STYLES.get(label, {})

            axis.errorbar(
                x,
                mean,
                yerr=std,
                color=style.get("color"),
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                linewidth=2.2,
                markersize=6,
                capsize=4,
                elinewidth=1.6,
                label=label,
            )

        axis.set_xlabel("Synthetic sensors")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, linestyle="-", alpha=0.25)
        axis.legend(fontsize=8.5, ncol=2)

        fig.savefig(panel_path, dpi=220)
        if show:
            plt.show()
        plt.close(fig)
        saved_paths.append(panel_path)

    return saved_paths
