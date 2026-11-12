# Four-policy noise control; reuse reference fits and resume at round boundaries

import argparse
from contextlib import redirect_stderr, redirect_stdout
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from pollutant_gp.reconstruction import reconstruct_field
from pollutant_gp.robot.deployment_comparison import pure_uniform_indices, score
from pollutant_gp.robot.informative import run_informative_study
from pollutant_gp.robot.informative_checkpoint import atomic_json, load_checkpoint, save_checkpoint
from pollutant_gp.robot.informative_multiseed import experiment_manifest, prepare_run, atomic_rows, summarize
from pollutant_gp.robot.observations import sensor_errors, full_band_metrics
from pollutant_gp.robot.study import fit_robot_model, RobotFitError, write_rows


LAYOUTS = ("Random walk", "IPP", "IPP weighted", "Pure Uniform")
ERRORS = ("full_rmse", "local_full_rmse", "peak_absolute_error", "peak_distance",
          "full_zero_rmse", "full_low_rmse", "full_high_rmse")
METRICS = (*ERRORS, "prediction_at_true_peak", "peak_relative_error_percent", "distinct_cells",
           "local_sensor_count", "training_observations", "revisits")


def aggregate(rows, checkpoints):
    expected = {(n, layout) for n in checkpoints for layout in LAYOUTS}
    groups = {}
    for row in rows:
        key = (row["checkpoint"], row["layout"])
        group = groups.setdefault(row["seed"], {})
        if key in group:
            raise ValueError("Duplicate noise-study row.")
        group[key] = row
    if any(set(group) != expected for group in groups.values()):
        raise ValueError("Only complete four-policy seeds may enter aggregation.")
    if any(not np.isfinite(r[m]) for g in groups.values() for r in g.values() for m in ERRORS[:2]):
        raise ValueError("Nonfinite primary RMSE.")
    summary, pairs = [], []
    for count in checkpoints:
        for metric in METRICS:
            for layout in LAYOUTS:
                summary.append(dict(checkpoint=count, layout=layout, metric=metric, n_seeds=len(groups),
                                    **summarize([g[(count, layout)][metric] for g in groups.values()])))
            if metric in ERRORS:
                for left, right in combinations(LAYOUTS, 2):
                    delta = np.array([g[(count, left)][metric] - g[(count, right)][metric]
                                      for g in groups.values()])
                    finite = delta[np.isfinite(delta)]
                    pairs.append(dict(checkpoint=count, metric=metric, left=left, right=right,
                                      left_wins=int((finite < 0).sum()), right_wins=int((finite > 0).sum()),
                                      ties=int((finite == 0).sum()), **summarize(delta)))
    return summary, pairs


def publish(root, completed, statuses, checkpoints):
    atomic_rows(root / "seed_status.csv", statuses)
    if not completed:
        return []
    rows = [r for seed_rows in completed for r in seed_rows]
    summary, pairs = aggregate(rows, checkpoints)
    for name, contents in (("metrics.csv", rows), ("checkpoint_summary.csv", summary),
                           ("final_summary.csv", [r for r in summary if r["checkpoint"] == checkpoints[-1]]),
                           ("paired_comparison.csv", pairs)):
        atomic_rows(root / name, contents)
    return summary


def plot_summary(summary, path, noise_std, completed, requested):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, layout="constrained")
    styles = (("#D55E00", "s"), ("#0072B2", "o"), ("#AA3377", "D"), ("#009E73", "^"))
    for axis, metric, label in zip(axes, ERRORS[:2], ("Global RMSE (full domain)", "Local RMSE (full peak disk)")):
        for layout, (color, marker) in zip(LAYOUTS, styles):
            selected = sorted((r for r in summary if r["metric"] == metric and r["layout"] == layout),
                              key=lambda r: r["checkpoint"])
            x = np.array([r["checkpoint"] for r in selected])
            mean = np.array([r["mean"] for r in selected])
            std = np.array([r["std"] for r in selected])
            axis.plot(x, mean, color=color, marker=marker, markersize=4, label=layout)
            if completed > 1:
                axis.fill_between(x, np.maximum(0, mean - std), mean + std, color=color, alpha=.12)
        axis.set_ylabel(label)
        axis.grid(axis="y", color=".88")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Acquired measurements")
    fig.suptitle(f"Sensor noise std = {noise_std:g} | {completed}/{requested} complete seeds\nMean +/- sample standard deviation")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_pure_uniform(args, grid, transform, root, unit):
    root.mkdir(parents=True, exist_ok=True)
    sequence = pure_uniform_indices(grid, args.random_seed, args.n_samples)
    errors = sensor_errors(args.random_seed, args.n_samples, args.noise_std)
    measured = grid.field.ravel()[sequence].astype(float) + errors
    np.savez_compressed(root / "observations.npz", indices=sequence, values=measured, errors=errors)
    xy = np.column_stack((grid.x_grid.ravel(), grid.y_grid.ravel()))
    rotated = transform.transform(xy)
    scaler = StandardScaler().fit(rotated[grid.valid_mask.ravel()])
    peak = np.argmax(np.where(grid.valid_mask, grid.field, -np.inf))
    local = grid.valid_mask & (np.linalg.norm(xy - xy[peak], axis=1).reshape(grid.field.shape) <= args.peak_radius)
    rows, optimizer_rows = [], []
    for count in args.robot_checkpoints:
        indices, values = sequence[:count], measured[:count]
        checkpoint = root / f"pure_{count:06d}.npz"
        if checkpoint.exists():
            cached = load_checkpoint(checkpoint)
            np.testing.assert_array_equal(indices, cached["indices"])
            np.testing.assert_array_equal(values, cached["values"])
            raw, name, diagnostic = cached["raw"], cached["name"], cached["diagnostic"]
            optimizer_rows.extend(cached["optimizer_rows"])
        else:
            start = len(optimizer_rows)
            name, model, diagnostic = fit_robot_model(
                args, rotated, indices, values, scaler, count, "Pure Uniform", optimizer_rows, root / "pure", unit)
            raw = reconstruct_field(grid, model, scaler, args.prediction_batch_size, "none", False, transform)
            save_checkpoint(checkpoint, raw, name, diagnostic, model.kernel_.theta,
                            optimizer_rows[start:], indices=indices, values=values)
        prediction = np.maximum(raw.mean_field, 0.)
        metrics = score(grid, prediction, indices, local)
        metrics.update(full_band_metrics(grid.field, prediction, grid.valid_mask))
        metrics["peak_absolute_error"] = abs(metrics["prediction_at_true_peak"] - grid.field.ravel()[peak])
        rows.append(dict(seed=args.random_seed, checkpoint=count, layout="Pure Uniform", noise_std=args.noise_std,
                         training_observations=count, revisits=0, **metrics))
    write_rows(root / "pure_optimizer.csv", optimizer_rows)
    return rows


def run_seed(args, grid, transform, directory, unit):
    rows, references = [], {}
    for label, weight, folder in (("IPP", 0., "ipp"),
                                  ("IPP weighted", args.ipp_concentration_weight, "weighted")):
        root = directory / folder
        root.mkdir(parents=True, exist_ok=True)
        marker = root / "complete.json"
        if marker.exists():
            pair_rows = json.loads(marker.read_text(encoding="utf-8"))
        else:
            policy_args = argparse.Namespace(**vars(args))
            policy_args.ipp_concentration_weight = weight
            pair_rows, _, _ = run_informative_study(
                policy_args, grid, transform, root / "ipp.png", unit, save_artifacts=False,
                checkpoint_dir=root / "checkpoints", reference_checkpoint_dir=directory / "random_checkpoints")
            atomic_json(marker, pair_rows)
        for row in pair_rows:
            if row["layout"] == "Random walk":
                if row["checkpoint"] in references:
                    for metric in METRICS:
                        np.testing.assert_allclose(row[metric], references[row["checkpoint"]][metric],
                                                   rtol=1e-12, atol=1e-12, equal_nan=True)
                else:
                    references[row["checkpoint"]] = row
            else:
                rows.append(dict(row, layout=label))
    rows.extend(references.values())
    rows.extend(run_pure_uniform(args, grid, transform, directory / "pure", unit))
    # Only common full-domain metrics enter the four-way comparison. The paired
    # unseen masks in the IPP subarchives differ and must not be mixed here.
    return [{key: row[key] for key in ("seed", "checkpoint", "layout", "noise_std", *METRICS)} for row in rows]


def run_noise_study(args, grid, coordinate_transform, output_path, coordinate_unit):
    if coordinate_transform is None or args.robot_deployment != "uniform-domain":
        raise ValueError("Noise control requires current-informed, uniform-domain deployment.")
    if args.n_samples > int(grid.valid_mask.sum()):
        raise ValueError("Pure Uniform requires at least as many valid cells as acquisitions.")
    root = Path(args.output_dir)
    manifest = experiment_manifest(args, grid, coordinate_transform, coordinate_unit)
    manifest.update(study="four-policy sensor-noise control", policies=list(LAYOUTS),
                    evaluation="Full valid domain, full peak disk, full ground-truth bands INCLUDING observed cells.",
                    measurement_protocol="Independent Gaussian errors per acquisition; common error sequence by seed/acquisition "
                                         "across policies; separate RNG from motion. Retain noisy revisits; deduplicate at sigma=0. "
                                         "No clipping of observations; WhiteKernel learned, true sigma not supplied to the GP.")
    prepare_run(root, manifest, args.resume)
    statuses = [dict(seed=s, status="pending", message="") for s in args.robot_study_seeds]
    completed = []
    for status in statuses:
        marker = root / f"seed_{status['seed']:04d}" / "complete.json"
        if marker.exists():
            rows = json.loads(marker.read_text(encoding="utf-8"))
            if {r["seed"] for r in rows} != {status["seed"]}:
                raise ValueError("Invalid noise-study completion marker.")
            aggregate(rows, args.robot_checkpoints)
            completed.append(rows)
            status["status"] = "complete"
    summary = publish(root, completed, statuses, args.robot_checkpoints)
    for status in statuses:
        if status["status"] == "complete":
            print(f"Seed {status['seed']}: already complete, skipped", flush=True)
            continue
        directory = root / f"seed_{status['seed']:04d}"
        directory.mkdir(exist_ok=True)
        seed_args = argparse.Namespace(**vars(args))
        seed_args.random_seed = status["seed"]
        start = perf_counter()
        print(f"Seed {status['seed']}: four policies, noise std={args.noise_std:g}; log: {directory / 'run.log'}", flush=True)
        try:
            with (directory / "run.log").open("a", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                rows = run_seed(seed_args, grid, coordinate_transform, directory, coordinate_unit)
            aggregate(rows, args.robot_checkpoints)
            atomic_json(directory / "complete.json", rows)
        except RobotFitError as error:
            status.update(status="failed_fit", message=str(error))
            print(f"Seed {status['seed']}: no converged fit; saved rounds retained.", flush=True)
        except BaseException as error:
            status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "error", message=str(error))
            publish(root, completed, statuses, args.robot_checkpoints)
            print("Stopped. Repeat the same command with --resume; completed rounds are reused.", flush=True)
            raise
        else:
            completed.append(rows)
            status["status"] = "complete"
            print(f"Seed {status['seed']}: complete ({perf_counter() - start:.1f}s)", flush=True)
        summary = publish(root, completed, statuses, args.robot_checkpoints)
    if completed:
        plot_summary(summary, root / "noise_rmse.png", args.noise_std, len(completed), len(statuses))
    print(f"Completed {len(completed)}/{len(statuses)} seeds. Results: {root}", flush=True)
    if len(completed) != len(statuses):
        raise RuntimeError("Noise study incomplete: inspect seed_status.csv and resume.")
    return summary
