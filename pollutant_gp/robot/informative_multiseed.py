# Paired IPP/random-walk study, resumable at acquisition-round boundaries.

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn

from pollutant_gp.robot.informative import run_informative_study
from pollutant_gp.robot.informative_checkpoint import atomic_json
from pollutant_gp.robot.study import RobotFitError, write_rows


LAYOUTS = ("Informative", "Random walk")
ERRORS = ("full_rmse", "local_full_rmse", "unseen_global_rmse", "unseen_local_rmse",
          "peak_absolute_error", "peak_distance", "zero_rmse", "low_rmse", "high_rmse")
METRICS = (*ERRORS, "prediction_at_true_peak", "peak_relative_error_percent", "distinct_cells",
           "local_sensor_count", "revisits", "rejected_moves", "distance_total")


def experiment_manifest(args, grid, transform, unit):
    ignored = {"resume", "output_dir", "figure_name", "show", "verbose_optimizer_diagnostics", "random_seed"}
    configuration = {key: value for key, value in vars(args).items() if key not in ignored}
    configuration = json.loads(json.dumps(configuration, default=str))
    digest = hashlib.sha256()
    for array in (grid.field, grid.valid_mask, grid.x_grid, grid.y_grid):
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    package = Path(__file__).resolve().parents[1]
    sources = ("model.py", "reconstruction.py", "spatial.py", "peak.py", "peak_sampling.py",
               "peak_kernel.py", "positivity.py", "robot/study.py", "robot/trajectories.py",
               "robot/informative.py", "robot/informative_checkpoint.py", "robot/informative_multiseed.py")
    return dict(version=1, configuration=configuration, snapshot_sha256=digest.hexdigest(),
                selected_time=grid.selected_time_label, coordinate_unit=unit,
                rotation=[transform.angle_degrees, transform.center_x, transform.center_y],
                source_sha256={name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in sources},
                libraries=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
                evaluation="full valid grid and full peak disk; also fixed paired common-unobserved cells; "
                           "concentration bands on those common-unobserved cells",
                aggregation="complete paired seeds only; sample standard deviation ddof=1; IPP wins iff error < RW")


def prepare_run(root, manifest, resume):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "configuration.json"
    if path.exists():
        if not resume:
            raise ValueError(f"Experiment already exists in {root}. Use --resume or a new --output-dir.")
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Resume rejected: configuration, data, scientific code or library versions changed. "
                             "Restore the original configuration or use a new output directory.")
    elif any(root.iterdir()):
        raise ValueError("Output directory is not empty and has no IPP manifest. Use a new --output-dir.")
    else:
        atomic_json(path, manifest)


def summarize(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return dict(n_valid=len(finite), n_missing=len(values) - len(finite),
                mean=float(finite.mean()) if len(finite) else float("nan"),
                std=float(finite.std(ddof=1)) if len(finite) > 1 else float("nan"))


def aggregate_informative_rows(rows, checkpoints):
    expected = {(count, layout) for count in checkpoints for layout in LAYOUTS}
    groups = {}
    for row in rows:
        group = groups.setdefault(row["seed"], {})
        key = (row["checkpoint"], row["layout"])
        if key in group:
            raise ValueError("Duplicate seed/checkpoint/layout.")
        group[key] = row
    for group in groups.values():
        if set(group) != expected:
            raise ValueError("Only complete paired seeds may enter aggregation.")
        if any(not np.isfinite(row[key]) for row in group.values() for key in ERRORS[:4]):
            raise ValueError("Nonfinite primary RMSE in a completed seed.")
    summaries, comparisons = [], []
    for count in checkpoints:
        for metric in METRICS:
            for layout in LAYOUTS:
                values = [group[(count, layout)][metric] for group in groups.values()]
                summaries.append(dict(checkpoint=count, layout=layout, metric=metric, n_seeds=len(groups),
                                      **summarize(values)))
            if metric in ERRORS:
                pairs = np.array([[group[(count, name)][metric] for name in LAYOUTS]
                                  for group in groups.values()], dtype=float).reshape(-1, 2)
                finite = pairs[np.isfinite(pairs).all(axis=1)]
                delta = finite[:, 0] - finite[:, 1]
                comparisons.append(dict(checkpoint=count, metric=metric, n_seeds=len(groups),
                                        n_pairs=len(delta), ipp_wins=int((delta < 0).sum()),
                                        random_walk_wins=int((delta > 0).sum()), ties=int((delta == 0).sum()),
                                        mean_delta_ipp_minus_rw=float(delta.mean()) if len(delta) else float("nan"),
                                        std_delta=float(delta.std(ddof=1)) if len(delta) > 1 else float("nan")))
    return summaries, comparisons


def atomic_rows(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    write_rows(temporary, rows)
    os.replace(temporary, path)


def publish(root, completed, statuses, checkpoints):
    atomic_rows(root / "seed_status.csv", statuses)
    if not completed:
        return [], []
    rows = [row for result in completed for row in result["rows"]]
    uncertainty = [row for result in completed for row in result["uncertainty"]]
    summary, comparisons = aggregate_informative_rows(rows, checkpoints)
    uncertainty_summary = []
    for count, layout in sorted({(r["checkpoint"], r["layout"]) for r in uncertainty}):
        values = [r["mean_latent_variance"] for r in uncertainty if (r["checkpoint"], r["layout"]) == (count, layout)]
        uncertainty_summary.append(dict(checkpoint=count, layout=layout, metric="mean_latent_variance",
                                        n_seeds=len(completed), **summarize(values)))
    for name, contents in (("metrics.csv", rows), ("checkpoint_summary.csv", summary),
                           ("final_summary.csv", [r for r in summary if r["checkpoint"] == checkpoints[-1]]),
                           ("paired_comparison.csv", comparisons), ("uncertainty.csv", uncertainty),
                           ("uncertainty_summary.csv", uncertainty_summary)):
        atomic_rows(root / name, contents)
    return summary, uncertainty_summary


def plot_summary(summary, metrics, labels, path, completed, requested):
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 4 * len(metrics)), squeeze=False,
                             sharex=True, layout="constrained")
    for ax, metric, label in zip(axes.ravel(), metrics, labels):
        for layout, color, marker in zip(LAYOUTS, ("#0072B2", "#D55E00"), ("o", "s")):
            selected = sorted((r for r in summary if r["metric"] == metric and r["layout"] == layout),
                              key=lambda r: r["checkpoint"])
            x = np.array([r["checkpoint"] for r in selected])
            y = np.array([r["mean"] for r in selected])
            std = np.array([r["std"] for r in selected])
            ax.plot(x, y, color=color, marker=marker, markersize=4, linewidth=1.8, label=layout)
            if completed > 1:
                ax.fill_between(x, np.maximum(0, y - std), y + std, color=color, alpha=.15)
            ax.set_ylabel(label)
        ax.grid(axis="y", color="#dddddd", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    axes[-1, 0].set_xlabel("Acquired measurements")
    fig.suptitle(f"IPP vs random walk | {completed}/{requested} complete seeds\nMean +/- 1 sample standard deviation")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_informative_multiseed(args, grid, coordinate_transform, output_path, coordinate_unit):
    # Stable output path: timestamped single-run figure names must not affect resume.
    root = Path(args.output_dir)
    prepare_run(root, experiment_manifest(args, grid, coordinate_transform, coordinate_unit), args.resume)
    completed = []
    statuses = [dict(seed=seed, status="pending", message="") for seed in args.robot_study_seeds]
    for status in statuses:
        marker = root / f"seed_{status['seed']:04d}" / "complete.json"
        if marker.exists():
            result = json.loads(marker.read_text(encoding="utf-8"))
            if {row["seed"] for row in result["rows"]} != {status["seed"]}:
                raise ValueError(f"Invalid completion marker for seed {status['seed']}")
            aggregate_informative_rows(result["rows"], args.robot_checkpoints)
            completed.append(result)
            status.update(status="complete")
    summary, uncertainty_summary = publish(root, completed, statuses, args.robot_checkpoints)
    print(f"\nIPP vs random walk: {len(statuses)} seeds; resumable output: {root}", flush=True)
    for status in statuses:
        seed = status["seed"]
        directory = root / f"seed_{seed:04d}"
        directory.mkdir(exist_ok=True)
        marker = directory / "complete.json"
        if status["status"] == "complete":
            print(f"Seed {seed}: already complete, skipped", flush=True)
            continue
        else:
            seed_args = argparse.Namespace(**vars(args))
            seed_args.random_seed = seed
            start = perf_counter()
            print(f"Seed {seed}: starting/resuming; progress in {directory / 'run.log'}", flush=True)
            try:
                with (directory / "run.log").open("a", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                    rows, _, _ = run_informative_study(
                        seed_args, grid, coordinate_transform, directory / "ipp.png", coordinate_unit,
                        save_artifacts=False, checkpoint_dir=directory / "checkpoints")
                aggregate_informative_rows(rows, args.robot_checkpoints)
                with (directory / f"ipp_seed_{seed}_informative_uncertainty.csv").open(newline="", encoding="utf-8") as stream:
                    uncertainty = [dict(seed=seed, **{key: value if key == "layout" else
                                                     int(value) if key in ("checkpoint", "valid_count") else float(value)
                                                     for key, value in row.items()}) for row in csv.DictReader(stream)]
                result = dict(rows=rows, uncertainty=uncertainty)
                atomic_json(marker, result)
            except RobotFitError as error:
                status.update(status="failed_fit", message=str(error))
                print(f"Seed {seed}: no converged candidate. Saved rounds retained; --resume retries it.", flush=True)
            except BaseException as error:
                status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "error", message=str(error))
                publish(root, completed, statuses, args.robot_checkpoints)
                print("Run stopped. Repeat the same command with --resume to reuse completed work.", flush=True)
                raise
            else:
                completed.append(result)
                status.update(status="complete")
                print(f"Seed {seed}: complete ({perf_counter() - start:.1f}s this invocation)", flush=True)
        summary, uncertainty_summary = publish(root, completed, statuses, args.robot_checkpoints)
    if completed:
        plot_summary(summary, ERRORS[:2], ("Global RMSE (all valid cells)", "Local RMSE (full peak disk)"),
                     root / "multiseed_rmse.png", len(completed), len(statuses))
        plot_summary(summary, ERRORS[2:4], ("Global RMSE (common unobserved)", "Local RMSE (common unobserved)"),
                     root / "multiseed_unseen_rmse.png", len(completed), len(statuses))
        plot_summary(uncertainty_summary, ("mean_latent_variance",), ("Mean latent variance [C\u00b2]",),
                     root / "multiseed_latent_variance.png", len(completed), len(statuses))
    print(f"Completed {len(completed)}/{len(statuses)} paired seeds. Results: {root}", flush=True)
    if len(completed) != len(statuses):
        raise RuntimeError("Study incomplete: inspect seed_status.csv and run again with --resume.")
    return summary, uncertainty_summary
