# Paired multi-seed aggregation of the existing robot sampling protocol.

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json

import matplotlib.pyplot as plt
import numpy as np

from pollutant_gp.robot.study import RobotFitError, run_robot_study, write_rows


LAYOUTS = ("Robot", "Uniform budget", "Uniform distinct")
CURVE_METRICS = ("unseen_global_rmse", "unseen_local_rmse")
FINAL_METRICS = (*CURVE_METRICS, "prediction_at_true_peak", "peak_relative_error_percent",
                 "peak_distance", "true_peak", "distinct_cells", "zero_rmse", "low_rmse", "high_rmse")


# Require complete paired seeds; missing concentration bands remain explicitly undefined.
def aggregate_robot_rows(rows, checkpoints):
    expected = {(n, layout) for n in checkpoints for layout in LAYOUTS}
    groups = {}
    for row in rows:
        key = (row["checkpoint"], row["layout"])
        group = groups.setdefault(row["seed"], {})
        if key in group:
            raise ValueError("Duplicate seed/checkpoint/layout metric row.")
        group[key] = row
    for group in groups.values():
        if set(group) != expected:
            raise ValueError("Only complete seeds may enter paired aggregation.")
        if any(not np.isfinite(row[m]) for row in group.values() for m in CURVE_METRICS):
            raise ValueError("Nonfinite primary RMSE in a completed seed.")
    curves, final = [], []
    for checkpoint in checkpoints:
        for layout in LAYOUTS:
            selected = [group[(checkpoint, layout)] for group in groups.values()]
            metrics = FINAL_METRICS if checkpoint == checkpoints[-1] else CURVE_METRICS
            for metric in metrics:
                values = np.array([r[metric] for r in selected], dtype=float)
                finite = values[np.isfinite(values)]
                result = dict(checkpoint=checkpoint, layout=layout, metric=metric,
                              n_seeds=len(groups), n_valid=len(finite), n_missing=len(values)-len(finite),
                              mean=float(finite.mean()) if len(finite) else float("nan"),
                              std=float(finite.std(ddof=1)) if len(finite) > 1 else float("nan"))
                if metric in CURVE_METRICS:
                    curves.append(result)
                if checkpoint == checkpoints[-1]:
                    final.append(result)
    return curves, final


# Keep colors and markers distinct; error bars avoid overlapping shaded uncertainty bands.
def plot_robot_multiseed(summary, path, requested, completed, deployment="central"):
    styles = (("#0072B2", "o", "-"), ("#D55E00", "s", "--"), ("#009E73", "^", "-."))
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True, layout="constrained")
    for ax, metric, title in zip(axes, CURVE_METRICS, ("Global reconstruction", "Peak region (fixed diagnostic disk)")):
        for layout, (color, marker, line) in zip(LAYOUTS, styles):
            data = sorted((r for r in summary if r["layout"] == layout and r["metric"] == metric),
                          key=lambda r: r["checkpoint"])
            x = [r["checkpoint"] for r in data]
            y = [r["mean"] for r in data]
            std = [r["std"] for r in data]
            ax.errorbar(x, y, yerr=std if completed >= 2 else None, color=color, marker=marker,
                        linestyle=line, linewidth=1.8, markersize=5, capsize=3, elinewidth=1.1, label=layout)
        ax.set(title=title, ylabel="RMSE on common unobserved cells")
        ax.grid(axis="y", color="#dddddd", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=.035)
    axes[0].legend(frameon=False, fontsize=10, loc="best")
    axes[-1].set_xticks(sorted({r["checkpoint"] for r in summary}))
    axes[-1].set_xlabel("Nominal acquisition budget (distinct count matched for Uniform distinct)")
    fig.suptitle(f"Robot sampling | {deployment} | {completed}/{requested} complete seeds\n"
                 "Mean +/- 1 sample standard deviation" if completed >= 2 else
                 f"Robot sampling | {deployment} | {completed}/{requested} complete seeds\nStandard deviation unavailable",
                 fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


# Persist each seed before aggregation; failures never produce selectively missing curve points.
def run_robot_multiseed(args, grid, coordinate_transform, output_path, coordinate_unit):
    root = output_path.with_suffix("")
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(configuration=vars(args), selected_time=grid.selected_time_label,
                    coordinate_unit=coordinate_unit, rotation_degrees=coordinate_transform.angle_degrees,
                    aggregation="complete paired seeds; sample std ddof=1; undefined bands excluded with n_valid")
    (root / "configuration.json").write_text(json.dumps(manifest, default=str, indent=2), encoding="utf-8")
    rows, statuses = [], []
    print(f"\nRobot multiseed: {len(args.robot_study_seeds)} seeds; output={root}", flush=True)
    for seed in args.robot_study_seeds:
        seed_args = argparse.Namespace(**vars(args))
        seed_args.random_seed = seed
        seed_dir = root / f"seed_{seed:04d}"
        seed_dir.mkdir(exist_ok=True)
        print(f"Seed {seed}: starting (details in {seed_dir.name}/run.log)", flush=True)
        try:
            with (seed_dir / "run.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                seed_rows, _, _ = run_robot_study(seed_args, grid, coordinate_transform,
                                                 seed_dir / "robot.png", coordinate_unit, save_artifacts=False)
            aggregate_robot_rows(seed_rows, args.robot_checkpoints)
        except RobotFitError as error:
            statuses.append(dict(seed=seed, status="failed_fit", message=str(error)))
            print(f"Seed {seed}: failed fit; excluded from all aggregate checkpoints.", flush=True)
        else:
            rows.extend(seed_rows)
            statuses.append(dict(seed=seed, status="complete", message=""))
            print(f"Seed {seed}: complete", flush=True)
        write_rows(root / "seed_status.csv", statuses)
        if rows:
            write_rows(root / "metrics.csv", rows)
            curves, final = aggregate_robot_rows(rows, args.robot_checkpoints)
            write_rows(root / "checkpoint_summary.csv", curves)
            write_rows(root / "final_summary.csv", final)
    completed = sum(s["status"] == "complete" for s in statuses)
    if rows:
        plot_robot_multiseed(curves, root / "multiseed_rmse.png", len(statuses), completed, args.robot_deployment)
    print(f"Completed {completed}/{len(statuses)} seeds. Final budget: {args.robot_checkpoints[-1]}; results: {root}", flush=True)
    if completed < 2:
        raise RuntimeError("Fewer than two complete seeds: descriptive mean only, no multi-seed standard deviation.")
    return rows, curves, final
