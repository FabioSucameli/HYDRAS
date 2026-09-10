# Terminal summary and figures for post-fit peak diagnostics.

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from pollutant_gp.peak import PeakDiagnostics, PeakProfile
from pollutant_gp.types import GridData, ReconstructionResult


# Print quantities that distinguish a missing observation from a smoothed observed peak.
def print_peak_diagnostics(diagnostics: PeakDiagnostics, coordinate_unit: str = "coordinate units") -> None:
    def number(value):
        return "n/a" if value is None else f"{value:.8g}"

    print("\n=== Peak diagnostics (post-fit; original concentration units) ===")
    print("Region: disk around the ground-truth maximum, not an assumed source location.")
    print(f"Radius: {diagnostics.radius:g} {coordinate_unit}")
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
    print(f"Peak location error: {diagnostics.peak_location_error:.8g} {coordinate_unit}")
    print(f"Nearest sensor distance: {diagnostics.nearest_sensor_distance:.8g} {coordinate_unit}")
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
    colour_max: float | None = None,
    coordinate_unit: str = "coordinate units",
) -> list[Path]:
    if colour_max is not None and (not np.isfinite(colour_max) or colour_max <= 0):
        raise ValueError("The colour maximum must be finite and positive.")
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
    data_max = max(diagnostics.true_max, diagnostics.predicted_max, vmin + 1e-12)
    vmax = data_max if colour_max is None else colour_max
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
                 xlabel=f"x offset from true maximum ({coordinate_unit})",
                 ylabel=f"y offset from true maximum ({coordinate_unit})")
        axis.set_aspect("equal")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=len(labels))
    figure.colorbar(mesh, ax=axes, label="Concentration", shrink=0.85,
                    extend="max" if data_max > vmax else "neither")
    figure.suptitle(title + f"\nPeak-centred zoom; diagnostic disk radius = {radius:g} {coordinate_unit}", fontsize=13)
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
                     ylabel="Concentration", ylim=(vmin, data_max * 1.05))
            axis.grid(alpha=0.2)
            axis.legend(loc="upper right")
        axes[-1].set_xlabel(f"Signed distance from true maximum ({coordinate_unit})")
        figure.suptitle(title + "\nNearest-cell profiles of the existing maps; gaps are masked/outside cells",
                       fontsize=12)
        profile_path = output_base.with_name(output_base.name + "_profiles.png")
        figure.savefig(profile_path, dpi=200)
        if show:
            plt.show()
        plt.close(figure)
        paths.append(profile_path)
    return paths


# Compare the two kernel structures on each fixed sensor set, with shared map colours.
def plot_peak_kernel_comparison(grid, runs, output_base, title, coordinate_unit,
                                colour_max=None, show=False):
    output_base.parent.mkdir(parents=True, exist_ok=True)
    center = np.asarray(runs[0].peak.true_peak_xy)
    radius = runs[0].peak.radius
    x, y = grid.x_grid - center[0], grid.y_grid - center[1]
    fields = [grid.field] + [run.reconstruction.mean_field for run in runs]
    data_max = max(float(np.max(field[grid.valid_mask])) for field in fields)
    vmin = min(0., *(float(np.min(field[grid.valid_mask])) for field in fields))
    vmax = max(data_max, vmin + 1e-12) if colour_max is None else colour_max
    figure, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    for row, label in enumerate(("Uniform", "Locally enriched (oracle)")):
        pair = runs[row * 2:row * 2 + 2]
        for column, axis in enumerate(axes[row]):
            field = grid.field if column == 0 else pair[column - 1].reconstruction.mean_field
            mesh = axis.pcolormesh(x, y, np.where(grid.valid_mask, field, np.nan),
                                  shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            indices = pair[0].indices
            inside = (np.abs(x.ravel()[indices]) <= radius) & (np.abs(y.ravel()[indices]) <= radius)
            axis.scatter(x.ravel()[indices][inside], y.ravel()[indices][inside], s=10,
                         facecolors="none", edgecolors="white", linewidths=.6)
            axis.scatter(0, 0, marker="*", s=100, color="#E69F00", edgecolors="black", zorder=5)
            axis.add_patch(Circle((0, 0), radius, fill=False, color="0.6", linestyle="--"))
            if column:
                delta = np.asarray(pair[column - 1].peak.predicted_peak_xy) - center
                if np.all(np.abs(delta) <= radius):
                    axis.scatter(*delta, marker="x", color="#D55E00", s=45, zorder=5)
            axis.set(title=f"{label}\n{('Ground truth', 'Single RBF', 'Two RBFs')[column]}",
                     xlim=(-radius, radius), ylim=(-radius, radius),
                     xlabel=f"x offset ({coordinate_unit})", ylabel=f"y offset ({coordinate_unit})")
            axis.set_aspect("equal")
    figure.colorbar(mesh, ax=list(axes.flat), label="Concentration", shrink=.8,
                    extend="max" if data_max > vmax else "neither")
    figure.suptitle(title + "\nWhite circles: sensors; star: true maximum; cross: global predicted maximum",
                    fontsize=12)
    zoom = output_base.with_name(output_base.name + "_zoom.png")
    figure.savefig(zoom, dpi=200)
    if show:
        plt.show()
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, sampling in enumerate(("Uniform", "Locally enriched (oracle)")):
        for direction, axis in enumerate(axes[row]):
            pair = runs[row * 2:row * 2 + 2]
            profile = pair[0].profiles[direction]
            axis.plot(profile.distances, profile.truth, color="black",
                      drawstyle="steps-mid", label="Ground truth", linewidth=2)
            for run, colour, style, label in zip(pair, ("#0072B2", "#D55E00"),
                                                ("--", "-"), ("Single RBF", "Two RBFs")):
                p = run.profiles[direction]
                axis.plot(p.distances, p.prediction, color=colour, linestyle=style,
                          drawstyle="steps-mid", label=label)
            axis.set(title=f"{sampling} | {profile.label}", ylabel="Concentration",
                     xlabel=f"Signed distance from true maximum ({coordinate_unit})")
            axis.axvline(0, color=".6", linestyle=":")
            axis.grid(alpha=.2)
            axis.legend()
    figure.suptitle(title + "\nNearest-cell profiles of the reconstructed maps", fontsize=12)
    profiles = output_base.with_name(output_base.name + "_profiles.png")
    figure.savefig(profiles, dpi=200)
    if show:
        plt.show()
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, attribute, label in zip(axes, ("common_rmse", "common_local_rmse"),
                                      ("Global common unseen cells", "Local common unseen cells")):
        for offset, colour, structure in ((-.18, "#0072B2", 0), (.18, "#D55E00", 1)):
            values = [getattr(runs[row * 2 + structure], attribute) for row in range(2)]
            bars = axis.bar(np.arange(2) + offset, values, .36, color=colour,
                            label=("Single RBF", "Two RBFs")[structure])
            axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        axis.set(xticks=[0, 1], xticklabels=["Uniform", "Enriched (oracle)"],
                 ylabel="RMSE", title=label)
        axis.margins(y=.25)
        axis.legend()
    figure.suptitle(title, fontsize=11)
    metrics = output_base.with_name(output_base.name + "_rmse.png")
    figure.savefig(metrics, dpi=200)
    if show:
        plt.show()
    plt.close(figure)
    return [zoom, profiles, metrics]


# Compare oracle sampling controls with shared colour limits and overlaid directional profiles.
def plot_peak_sampling_comparison(
    grid, runs, output_base: Path, title: str, coordinate_unit: str,
    colour_max: float | None = None, show: bool = False,
) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    center = np.asarray(runs[0].peak.true_peak_xy)
    radius = runs[0].peak.radius
    x, y = grid.x_grid - center[0], grid.y_grid - center[1]
    fields = [grid.field] + [run.reconstruction.mean_field for run in runs]
    vmin = min(0.0, *(float(np.min(field[grid.valid_mask])) for field in fields))
    data_max = max(float(np.max(field[grid.valid_mask])) for field in fields)
    vmax = max(data_max, vmin + 1e-12) if colour_max is None else colour_max
    figure, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
    for i, (axis, field) in enumerate(zip(axes.flat, fields)):
        mesh = axis.pcolormesh(x, y, np.where(grid.valid_mask, field, np.nan),
                               shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axis.add_patch(Circle((0, 0), radius, fill=False, edgecolor="0.6", linestyle="--"))
        axis.scatter(0, 0, marker="*", s=120, c="#E69F00", edgecolors="black",
                     label="True maximum", zorder=6)
        label = "Ground truth"
        if i:
            run = runs[i - 1]
            xy = np.column_stack((x.ravel()[run.indices], y.ravel()[run.indices]))
            inside = np.all(np.abs(xy) <= radius, axis=1)
            added = ~np.isin(run.indices, runs[0].indices)
            for selection, colour, sensor_label in (
                (inside & ~added, "white", "Retained sensors"),
                (inside & added, "#00BFC4", "Added sensors"),
            ):
                axis.scatter(*xy[selection].T, s=16, facecolors="none", edgecolors=colour,
                             linewidths=0.8, label=sensor_label, zorder=4)
            delta = np.asarray(run.peak.predicted_peak_xy) - center
            if np.all(np.abs(delta) <= radius):
                axis.scatter(*delta, marker="x", s=55, c="#D55E00", linewidths=2,
                             label="Predicted maximum", zorder=5)
            suffix = "; same samples as Uniform" if run.reused else ""
            label = f"{run.name}\nLocal sensors: {run.peak.local_sensor_count}{suffix}"
        axis.set(title=label, xlim=(-radius, radius), ylim=(-radius, radius),
                 xlabel=f"x offset from true maximum ({coordinate_unit})",
                 ylabel=f"y offset from true maximum ({coordinate_unit})")
        axis.set_aspect("equal")
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=len(labels))
    figure.colorbar(mesh, ax=list(axes.flat), label="Concentration", shrink=0.85,
                    extend="max" if data_max > vmax else "neither")
    figure.suptitle(title + f"\nDiagnostic disk radius = {radius:g} {coordinate_unit}", fontsize=12)
    zoom_path = output_base.with_name(output_base.name + "_zoom.png")
    figure.savefig(zoom_path, dpi=200)
    if show:
        plt.show()
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True, sharey=True,
                               constrained_layout=True)
    for i, axis in enumerate(axes):
        reference = runs[0].profiles[i]
        axis.plot(reference.distances, reference.truth, color="black", drawstyle="steps-mid",
                  linewidth=2, label="Ground truth")
        for run, colour, style in zip(runs, ("#0072B2", "#D55E00", "#009E73"), ("-", "--", "-.")):
            profile = run.profiles[i]
            label = run.name + (" = Uniform" if run.reused else "")
            axis.plot(profile.distances, profile.prediction, color=colour, linestyle=style,
                      drawstyle="steps-mid", linewidth=1.5, label=label)
        axis.axvline(0, color="0.6", linestyle=":", linewidth=1)
        axis.set(title=f"{reference.label} | angle from +x = {reference.angle_degrees:.2f} deg",
                 ylabel="Concentration")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=9)
    axes[-1].set_xlabel(f"Signed distance from true maximum ({coordinate_unit})")
    figure.suptitle(title + "\nNearest-cell profiles of the reconstructed maps", fontsize=12)
    profile_path = output_base.with_name(output_base.name + "_profiles.png")
    figure.savefig(profile_path, dpi=200)
    if show:
        plt.show()
    plt.close(figure)
    return [zoom_path, profile_path]
