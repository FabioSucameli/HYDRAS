# Terminal summary and figures for post-fit peak diagnostics.

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from pollutant_gp.peak import PeakDiagnostics, PeakProfile
from pollutant_gp.types import GridData, ReconstructionResult


# Print quantities that distinguish a missing observation from a smoothed observed peak.
def print_peak_diagnostics(diagnostics: PeakDiagnostics) -> None:
    def number(value):
        return "n/a" if value is None else f"{value:.8g}"

    print("\n=== Peak diagnostics (post-fit; original concentration units) ===")
    print("Region: disk around the ground-truth maximum, not an assumed source location.")
    print(f"Radius: {diagnostics.radius:g} coordinate units (metres for the CL02 grid)")
    print(f"Ground-truth peak coordinates: {diagnostics.true_peak_xy}")
    print(f"Predicted peak coordinates: {diagnostics.predicted_peak_xy}")
    print(f"Ground-truth maximum: {diagnostics.true_max:.8g}")
    print(f"Maximum truth at sensor cells: {diagnostics.sampled_truth_max:.8g}")
    print(f"Maximum sensor observation (including noise): {diagnostics.observed_max:.8g}")
    print(f"Predicted global maximum: {diagnostics.predicted_max:.8g}")
    print(f"Prediction at the true peak: {diagnostics.prediction_at_true_peak:.8g}")
    print(f"True peak sampled: {diagnostics.true_peak_sampled}")
    fraction = diagnostics.peak_underestimation_fraction
    print(f"Signed peak underestimation: {diagnostics.peak_underestimation:.8g}")
    print("Relative underestimation (%; negative means overshoot): "
          f"{number(None if fraction is None else 100 * fraction)}")
    print(f"Peak location error: {diagnostics.peak_location_error:.8g} coordinate units")
    print(f"Nearest sensor distance: {diagnostics.nearest_sensor_distance:.8g} coordinate units")
    print(f"Local cells / sensors: {diagnostics.local_cell_count} / {diagnostics.local_sensor_count}")
    print(f"Global RMSE (all valid cells): {diagnostics.global_rmse:.8g}")
    print(f"Local RMSE (all valid cells in disk): {diagnostics.local_rmse:.8g}")
    if diagnostics.true_peak_count > 1 or diagnostics.predicted_peak_count > 1:
        print("Tied maxima: true=" + str(diagnostics.true_peak_count)
              + ", predicted=" + str(diagnostics.predicted_peak_count)
              + "; locations use the first row-major maximum. Location error may be ambiguous.")


# Draw matched local maps and two map-based profiles without GP evaluation.
def plot_peak_diagnostics(
    grid_data: GridData,
    reconstruction: ReconstructionResult,
    sampled_flat_indices: np.ndarray,
    diagnostics: PeakDiagnostics,
    profiles: tuple[PeakProfile, PeakProfile] | None,
    output_base: Path,
    title: str,
    show: bool = False,
) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    center = np.asarray(diagnostics.true_peak_xy)
    radius = diagnostics.radius
    x = grid_data.x_grid - center[0]
    y = grid_data.y_grid - center[1]
    sensor_xy = np.column_stack((x.ravel()[sampled_flat_indices], y.ravel()[sampled_flat_indices]))
    inside_zoom = np.all(np.abs(sensor_xy) <= radius, axis=1)
    prediction = np.where(grid_data.valid_mask, reconstruction.mean_field, np.nan)
    truth = np.where(grid_data.valid_mask, grid_data.field, np.nan)
    vmin = min(0.0, float(np.nanmin(truth)), float(np.nanmin(prediction)))
    vmax = max(diagnostics.true_max, diagnostics.predicted_max, vmin + 1e-12)
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.6), constrained_layout=True)
    for axis, values, label in zip(axes, (truth, prediction), ("Ground truth", "GP reconstruction")):
        mesh = axis.pcolormesh(x, y, values, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axis.add_patch(Circle((0, 0), radius, fill=False, edgecolor="0.6", linestyle="--"))
        axis.scatter(*sensor_xy[inside_zoom].T, s=36, facecolors="none", edgecolors="white",
                     linewidths=1.2, label="Sensors", zorder=4)
        axis.scatter(0, 0, marker="*", s=150, c="#E69F00", edgecolors="black",
                     label="True maximum", zorder=6)
        predicted_offset = np.asarray(diagnostics.predicted_peak_xy) - center
        if np.all(np.abs(predicted_offset) <= radius):
            axis.scatter(*predicted_offset, marker="x", s=65, color="#D55E00",
                         linewidths=2, label="Predicted maximum", zorder=5)
        axis.set(xlim=(-radius, radius), ylim=(-radius, radius), title=label,
                 xlabel="x offset from true maximum (coordinate units)",
                 ylabel="y offset from true maximum (coordinate units)")
        axis.set_aspect("equal")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=len(labels))
    figure.colorbar(mesh, ax=axes, label="Concentration", shrink=0.85)
    figure.suptitle(title + f"\nPeak-centred zoom; diagnostic disk radius = {radius:g}", fontsize=13)
    zoom_path = output_base.with_name(output_base.name + "_zoom.png")
    figure.savefig(zoom_path, dpi=200)
    if show:
        plt.show()
    plt.close(figure)
    paths = [zoom_path]

    if profiles is not None:
        figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, sharey=True,
                                   constrained_layout=True)
        for axis, profile in zip(axes, profiles):
            axis.plot(profile.distances, profile.truth, color="#222222", linewidth=1.8,
                      drawstyle="steps-mid", label="Ground truth")
            axis.plot(profile.distances, profile.prediction, color="#0072B2", linewidth=1.8,
                      drawstyle="steps-mid", label="GP reconstruction")
            axis.axvline(0, color="0.6", linestyle=":", linewidth=1)
            axis.set(title=f"{profile.label} | angle from +x = {profile.angle_degrees:.2f} deg",
                     ylabel="Concentration", ylim=(vmin, vmax * 1.05))
            axis.grid(alpha=0.2)
            axis.legend(loc="upper right")
        axes[-1].set_xlabel("Signed distance from true maximum (coordinate units)")
        figure.suptitle(title + "\nNearest-cell profiles of the existing maps; gaps are masked/outside cells",
                       fontsize=12)
        profile_path = output_base.with_name(output_base.name + "_profiles.png")
        figure.savefig(profile_path, dpi=200)
        if show:
            plt.show()
        plt.close(figure)
        paths.append(profile_path)
    return paths

