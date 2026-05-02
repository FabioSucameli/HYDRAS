# Plotting helpers.

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pollutant_gp.types import GridData, ReconstructionResult

# Generate a 2x2 panel plot showing the ground truth, GP reconstruction, predictive uncertainty, and absolute error.
def plot_reconstruction(
    grid_data: GridData,
    reconstruction: ReconstructionResult,
    sample_coordinates: np.ndarray,
    output_path: Path,
    show: bool,
) -> None:
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    truth = np.where(grid_data.valid_mask, grid_data.field, np.nan)
    prediction = reconstruction.mean_field
    uncertainty = reconstruction.std_field
    absolute_error = np.abs(prediction - truth)

    finite_truth = truth[np.isfinite(truth)]
    shared_vmin = float(np.nanmin(finite_truth))
    shared_vmax = float(np.nanmax(finite_truth))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    axes = axes.ravel()

    panels = [
        ("Ground truth", truth, "viridis", shared_vmin, shared_vmax, True),
        ("Gaussian Process reconstruction", prediction, "viridis", shared_vmin, shared_vmax, True),
        ("Predictive uncertainty (standard deviation)", uncertainty, "magma", None, None, False),
        ("Absolute reconstruction error", absolute_error, "inferno", None, None, False),
    ]

    for axis, (title, values, cmap, vmin, vmax, draw_samples) in zip(axes, panels):
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

        if draw_samples:
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

