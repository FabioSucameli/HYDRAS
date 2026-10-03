# Fixed-scale maps and checkpoint-replay animation in the report's Figure 14 palette.

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle, Patch, Polygon, Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image

from pollutant_gp.robot.trajectories import connected_geometry, grid_geometry


# Mark privileged starts and the excluded inner disk without implying source knowledge.
def draw_oracle_deployment(axis, walk):
    for radius in (walk.annulus_inner, walk.annulus_outer):
        axis.add_patch(Circle(walk.center, radius, fill=False, color="#b2182b", linestyle="--", linewidth=1))
    axis.scatter(*walk.center, marker="+", s=70, color="#b2182b", label="True peak (oracle center)")
    axis.scatter(*walk.positions[0, :walk.near_count].T, s=48, facecolors="#fdae61",
                 edgecolors="black", label=f"{walk.near_count} annulus starts", zorder=5)


# Show categorical connectivity and the centroid-to-cell projection without concentration data.
def plot_connected_components(grid, walk, radius, base, unit):
    components, principal, centroid, central_cell = connected_geometry(grid)
    count = int(components.max())
    sizes = np.bincount(components.ravel())[1:]
    automatic = np.array([grid.x_grid.ravel()[central_cell], grid.y_grid.ravel()[central_cell]])
    _, spacing = grid_geometry(grid)
    colors = plt.get_cmap("Set3" if count <= 12 else "turbo", max(count, 2))(np.arange(count))
    cmap = ListedColormap(["white", *colors])
    norm = BoundaryNorm(np.arange(count + 2) - .5, cmap.N)
    if walk.center is None or walk.deployment == "peak-annulus":
        fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
        mesh = axis.pcolormesh(grid.x_grid, grid.y_grid, components, cmap=cmap, norm=norm, shading="auto")
        axis.scatter(*walk.positions[0].T, s=35, facecolors="white", edgecolors="black", label="Initial robots")
        axis.set(xlabel=f"x ({unit})", ylabel=f"y ({unit})", aspect="equal",
                 title=f"Uniform-domain deployment | {len(walk.positions[0])} distinct cells\n"
                       f"All {count} valid-mask components eligible; no deployment disk")
        if walk.deployment == "peak-annulus":
            draw_oracle_deployment(axis, walk)
            axis.set_title(f"Oracle peak-annulus deployment | {walk.near_count} near-peak robots\n"
                           f"Annulus {walk.annulus_inner:g}-{walk.annulus_outer:g} {unit}; remaining starts uniform-domain")
        axis.ticklabel_format(useOffset=False)
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.legend(fontsize=9)
        fig.colorbar(mesh, ax=axis, ticks=np.arange(1, count + 1), label="Component ID")
        path = base.with_name(base.name + "_components.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    for axis in axes:
        mesh = axis.pcolormesh(grid.x_grid, grid.y_grid, components, cmap=cmap, norm=norm, shading="auto")
        axis.scatter(*centroid, marker="o", s=90, facecolors="none", edgecolors="#b2182b",
                     linewidths=2, label="Principal component centroid", zorder=4)
        axis.scatter(*automatic, marker="x", s=60, color="black", linewidths=2,
                     label="Nearest valid cell center", zorder=5)
        axis.add_patch(Circle(walk.center, radius, fill=False, color="#333333", linestyle="--",
                              label=f"Deployment disk (r = {radius:g} {unit})"))
        if not np.array_equal(walk.center, automatic):
            axis.scatter(*walk.center, marker="+", s=100, color="#e66101", label="Custom deployment center")
        axis.set(xlabel=f"x ({unit})", ylabel=f"y ({unit})", aspect="equal")
        axis.ticklabel_format(useOffset=False)
        axis.xaxis.set_major_locator(MaxNLocator(3))
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.tick_params(labelsize=9)
    axes[0].set_title(f"Valid-mask components: {count}\nPrincipal component: {principal} ({sizes[principal - 1]} cells)", fontsize=11)
    half_span = 2.5 * np.abs(spacing)
    axes[1].set(xlim=(automatic[0] - half_span[0], automatic[0] + half_span[0]),
                ylim=(automatic[1] - half_span[1], automatic[1] + half_span[1]))
    axes[1].add_patch(Rectangle(automatic - np.abs(spacing) / 2, *np.abs(spacing),
                                fill=False, edgecolor="black", linewidth=1))
    axes[1].set_title(f"Center selection (zoom)\nCentroid-to-cell distance: {np.linalg.norm(centroid - automatic):.3g} {unit}", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    if count <= 6:
        handles.extend(Patch(facecolor=colors[i], label=f"Component {i + 1}") for i in range(count))
        labels.extend(f"Component {i + 1}" for i in range(count))
    else:
        fig.colorbar(mesh, ax=axes, ticks=np.arange(1, count + 1), label="Component ID", shrink=.6)
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.subplots_adjust(bottom=.23, top=.88, wspace=.4)
    path = base.with_name(base.name + "_components.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# Plot acquisition-time separation only; thresholds are diagnostics rather than safety limits.
def plot_robot_proximity(rows, base, unit, thresholds=(10., 20.)):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    times = [row["acquisition"] for row in rows]
    axes[0].plot(times, [np.nan if row["min_distance"] is None else row["min_distance"] for row in rows],
                 color="#173f5f", marker="o", markersize=3, label="Minimum pair distance")
    for threshold, color in zip(thresholds, ("#b2182b", "#d98c00")):
        axes[0].axhline(threshold, color=color, linestyle="--", label=f"{threshold:g} {unit}")
    axes[0].set(ylabel=f"Minimum distance ({unit})")
    axes[0].legend(fontsize=9)
    axes[1].step(times, [row["same_cell_pairs"] for row in rows], where="mid", color="black",
                 marker="x", markersize=4, zorder=4, label="Same cell")
    for threshold, color in zip(thresholds, ("#b2182b", "#d98c00")):
        axes[1].step(times, [row[f"pairs_lt_{threshold:g}"] for row in rows], where="mid",
                     color=color, label=f"Distance < {threshold:g} {unit}")
    axes[1].set(xlabel="Acquisition index (initial positions included)", ylabel="Simultaneous unordered pairs")
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].legend(fontsize=9)
    for axis in axes:
        axis.grid(alpha=.25)
    path = base.with_name(base.name + "_proximity.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# Extend the encoded last frame, avoiding extra renders or artificial robot states.
def save_paused_gif(figure, update, n_steps, path, dpi=90):
    fps = 4
    animation = FuncAnimation(figure, update, frames=n_steps, interval=1000 / fps, repeat=False)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    frames, durations = [], []
    with Image.open(path) as gif:
        for index in range(gif.n_frames):
            gif.seek(index)
            frames.append(gif.copy())
            durations.append(gif.info.get("duration", 250))
    durations[-1] += 2000
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, disposal=2)


# Animate stylized twin-hull robots using recorded positions, without concentration or GP maps.
def plot_robot_movement(grid, walk, args, base, unit):
    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    draw_map(axis, grid, np.full_like(grid.field, np.nan), 1, unit)
    points = walk.positions.reshape(-1, 2)
    span = np.ptp(points, axis=0)
    if walk.center is None or walk.deployment == "peak-annulus":
        # pcolormesh extends half a cell beyond the extreme grid centers.
        _, spacing = grid_geometry(grid)
        half = np.abs(spacing) / 2
        axis.set(xlim=(grid.x_grid.min() - half[0], grid.x_grid.max() + half[0]),
                 ylim=(grid.y_grid.min() - half[1], grid.y_grid.max() + half[1]))
    else:
        margin = max(float(span.max()) * .12, args.robot_start_radius * .5)
        axis.set(xlim=(points[:, 0].min() - margin, points[:, 0].max() + margin),
                 ylim=(points[:, 1].min() - margin, points[:, 1].max() + margin))
    if walk.deployment == "peak-annulus":
        draw_oracle_deployment(axis, walk)
    elif walk.center is not None:
        axis.add_patch(Circle(walk.center, args.robot_start_radius, fill=False,
                              color="#173f5f", linestyle="--", linewidth=.8, alpha=.45))
    trails = [axis.plot([], [], color="#173f5f", linewidth=.9, alpha=.45)[0]
              for _ in range(args.n_robots)]
    # Glyph dimensions are schematic, not physical vessel dimensions.
    upper_hull = np.array([[-.8, .25], [.55, .25], [1., .47], [.55, .7], [-.8, .7]])
    lower_hull = upper_hull * [1, -1]
    bridge = np.array([[-.4, -.4], [.35, -.4], [.35, .4], [-.4, .4]])
    shapes = (upper_hull, lower_hull, bridge)
    size = max(np.ptp(axis.get_xlim()), np.ptp(axis.get_ylim())) * .010
    robots = []
    for _ in range(args.n_robots):
        patches = []
        for vertices, color in zip(shapes, ("white", "white", "#fdae61")):
            patch = Polygon(vertices, facecolor=color, edgecolor="#173f5f", linewidth=.8, zorder=4)
            axis.add_patch(patch)
            patches.append(patch)
        robots.append(patches)
    headings = np.zeros((args.n_steps, args.n_robots))
    for step in range(1, args.n_steps):
        moves = walk.positions[step] - walk.positions[step - 1]
        headings[step] = np.where(np.linalg.norm(moves, axis=1) > 1e-10,
                                  np.arctan2(moves[:, 1], moves[:, 0]), headings[step - 1])
    heading = axis.set_title("", fontsize=12)
    figure.supxlabel("Recorded random walk | schematic robots, not to scale | frozen field", fontsize=9)

    def update(step):
        for robot, patches in enumerate(robots):
            angle = headings[step, robot]
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            for patch, vertices in zip(patches, shapes):
                patch.set_xy(size * vertices @ rotation.T + walk.positions[step, robot])
            trails[robot].set_data(walk.positions[:step + 1, robot, 0], walk.positions[:step + 1, robot, 1])
        heading.set_text(f"Marine robots | {walk.deployment} | seed {args.random_seed}\n"
                         f"Acquisition {step + 1}/{args.n_steps} | N={(step + 1) * args.n_robots}")

    path = base.with_name(base.name + "_robots_only.gif")
    save_paused_gif(figure, update, args.n_steps, path, dpi=120)
    plt.close(figure)
    return path


# Draw white land, blue sea and the same yellow-to-red concentration map as Figure 14.
def draw_map(axis, grid, values, vmax, unit, threshold=None):
    axis.pcolormesh(grid.x_grid, grid.y_grid, grid.valid_mask.astype(int), shading="auto",
                    cmap=ListedColormap(["white", "#86cce3"]), vmin=0, vmax=1)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad((0, 0, 0, 0))
    visible = grid.valid_mask & np.isfinite(values)
    if threshold is not None:
        visible &= values > threshold
    mesh = axis.pcolormesh(grid.x_grid, grid.y_grid, np.where(visible, values, np.nan),
                           shading="auto", cmap=cmap, vmin=0, vmax=vmax)
    axis.set(xlabel=f"x ({unit})", ylabel=f"y ({unit})", aspect="equal")
    axis.ticklabel_format(useOffset=False)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.tick_params(labelsize=9)
    return mesh


# Keep concentration limits shared across all checkpoints, methods and animation frames.
def plot_robot_results(grid, walk, results, rows, args, base, unit):
    paths = []
    threshold = args.concentration_display_threshold
    vmax = max(float(np.max(grid.field[grid.valid_mask])),
               *(float(np.max(r.mean_field[grid.valid_mask])) for r in results.values()), 1e-12)
    stdmax = max(*(float(np.max(r.std_field[grid.valid_mask])) for r in results.values()), 1e-12)
    checkpoints = args.robot_checkpoints
    title = (f"{args.nc_file.stem} | time index {args.time_index} | sampling seed {args.random_seed}"
             f" | {walk.deployment}")

    fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    mesh = draw_map(axis, grid, grid.field, vmax, unit, threshold)
    if walk.deployment == "peak-annulus":
        draw_oracle_deployment(axis, walk)
    elif walk.center is not None:
        axis.add_patch(Circle(walk.center, args.robot_start_radius, fill=False, color="#173f5f", linestyle="--"))
    axis.scatter(*walk.positions[0].T, s=24, facecolors="white", edgecolors="#173f5f", label="Initial robots")
    if walk.center is not None and walk.deployment != "peak-annulus":
        axis.scatter(*walk.center, marker="+", s=90, c="black", label="Geometric deployment center")
    axis.legend(fontsize=9)
    description = "Oracle peak-annulus deployment" if walk.deployment == "peak-annulus" else "Deployment from geometry only"
    axis.set_title(f"{title}\n{description}; display threshold {threshold:g}", fontsize=10)
    fig.colorbar(mesh, ax=axis, label="Concentration")
    path = base.with_name(base.name + "_deployment.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(len(checkpoints), 4, figsize=(17, 4.1 * len(checkpoints)),
                             squeeze=False, constrained_layout=True)
    for row, count in enumerate(checkpoints):
        step = count // args.n_robots
        mobile = results[("Robot", count)]
        reference = results[("Uniform budget", count)]
        for col, (values, limit, cutoff, label) in enumerate((
            (grid.field, vmax, threshold, "Ground truth + trajectories"),
            (mobile.mean_field, vmax, threshold, "Robot GP + clip"),
            (reference.mean_field, vmax, threshold, "Uniform budget GP + clip"),
            (mobile.std_field, stdmax, None, "Robot predictive std (incl. White)"),
        )):
            mesh = draw_map(axes[row, col], grid, values, limit, unit, cutoff)
            axes[row, col].set_title(f"N={count} | {label}", fontsize=10)
            if col == 0:
                axes[row, col].plot(walk.positions[:step, :, 0], walk.positions[:step, :, 1],
                                    color="#173f5f", alpha=.55, linewidth=.7)
                axes[row, col].scatter(*walk.positions[step - 1].T, s=16, c="white", edgecolors="#173f5f")
            if col == 2:
                fig.colorbar(mesh, ax=list(axes[row, :3]), shrink=.78, label="Concentration")
            elif col == 3:
                fig.colorbar(mesh, ax=axes[row, col], shrink=.78, label="Standard deviation")
    fig.suptitle(f"{title}\nFrozen field; shared colour limits; concentration display threshold {threshold:g}", fontsize=13)
    path = base.with_name(base.name + "_checkpoints.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    metrics = (("unseen_global_rmse", "Global RMSE (common unobserved)"),
               ("unseen_local_rmse", "Local RMSE (common unobserved)"),
               ("prediction_at_true_peak", "Prediction at true maximum"),
               ("zero_rmse", "RMSE: c = 0"), ("low_rmse", "RMSE: 0 < c <= 1"),
               ("high_rmse", "RMSE: c > 1"))
    for layout, color, marker in (("Robot", "#b2182b", "o"), ("Uniform budget", "#176d9c", "s"),
                                   ("Uniform distinct", "#b46a00", "^")):
        subset = [r for r in rows if r["layout"] == layout]
        for axis, (metric, label) in zip(axes.ravel(), metrics):
            axis.plot([r["checkpoint"] for r in subset], [r[metric] for r in subset],
                      color=color, marker=marker, label=layout)
            axis.set(xlabel="Robot measurement budget", ylabel=label)
            axis.grid(alpha=.25)
    axes[0, 2].axhline(float(np.max(grid.field[grid.valid_mask])), color="black", linestyle=":", label="True maximum")
    axes[0, 0].legend(fontsize=9)
    axes[0, 2].legend(fontsize=8)
    fig.suptitle(title)
    path = base.with_name(base.name + "_metrics.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    if args.robot_gif:
        paths.extend(plot_robot_gifs(grid, walk, results, args, base, unit))
    return paths


# Replay both GIFs from recorded arrays without saving PNGs or running any GP fits.
def plot_robot_gifs(grid, walk, results, args, base, unit):
    paths = []
    threshold = args.concentration_display_threshold
    vmax = max(float(np.max(grid.field[grid.valid_mask])),
               *(float(np.max(r.mean_field[grid.valid_mask])) for r in results.values()), 1e-12)
    stdmax = max(*(float(np.max(r.std_field[grid.valid_mask])) for r in results.values()), 1e-12)
    checkpoints = args.robot_checkpoints
    title = (f"{args.nc_file.stem} | time index {args.time_index} | sampling seed {args.random_seed}"
             f" | {walk.deployment}")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    empty = np.full_like(grid.field, np.nan)
    truth_mesh = draw_map(axes[0], grid, grid.field, vmax, unit, threshold)
    mean_mesh = draw_map(axes[1], grid, empty, vmax, unit, threshold)
    std_mesh = draw_map(axes[2], grid, empty, stdmax, unit)
    tracks = [axes[0].plot([], [], color="#173f5f", linewidth=.8, alpha=.6)[0]
              for _ in range(args.n_robots)]
    dots = axes[0].scatter(*walk.positions[0].T, s=25, c="white", edgecolors="#173f5f")
    axes[0].set_title("Ground truth + robot trajectories", fontsize=11)
    fig.colorbar(truth_mesh, ax=list(axes[:2]), label="Concentration", shrink=.8)
    fig.colorbar(std_mesh, ax=axes[2], label="Standard deviation", shrink=.8)
    heading = fig.suptitle(title, fontsize=12)

    def update(step):
        count = (step + 1) * args.n_robots
        for robot, line in enumerate(tracks):
            line.set_data(walk.positions[:step + 1, robot, 0], walk.positions[:step + 1, robot, 1])
        dots.set_offsets(walk.positions[step])
        ready = [n for n in checkpoints if n <= count]
        if ready:
            last = max(ready)
            result = results[("Robot", last)]
            mean_mesh.set_array(np.ma.masked_where(~grid.valid_mask | (result.mean_field <= threshold), result.mean_field))
            std_mesh.set_array(np.ma.masked_where(~grid.valid_mask, result.std_field))
            axes[1].set_title(f"GP + clip | last fit: N={last}", fontsize=11)
            axes[2].set_title(f"Predictive std (incl. White) | N={last}", fontsize=10)
        else:
            mean_mesh.set_array(np.ma.masked_invalid(empty))
            std_mesh.set_array(np.ma.masked_invalid(empty))
            axes[1].set_title(f"First GP fit at N={checkpoints[0]}", fontsize=11)
            axes[2].set_title("No fit yet", fontsize=11)
        unique = np.unique(walk.cell_indices[:step + 1]).size
        heading.set_text(f"{title}\nAcquisition {step + 1}/{args.n_steps} | measurements {count} | distinct cells {unique}")

    path = base.with_name(base.name + "_motion.gif")
    save_paused_gif(fig, update, args.n_steps, path)
    plt.close(fig)
    paths.append(path)
    paths.append(plot_robot_movement(grid, walk, args, base, unit))
    return paths
