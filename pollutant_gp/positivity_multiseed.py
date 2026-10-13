"""Paired D/Q positivity control at one sensor budget, without per-seed plots."""

from __future__ import annotations

from copy import copy
import csv
import json

import numpy as np

from pollutant_gp.positivity import run_positivity_study


METHODS = ("Direct + clip", "Square-root mean")
SAMPLINGS = ("Uniform", "Locally enriched (oracle)")
METRICS = (
    "unseen_global_rmse", "unseen_local_rmse", "full_rmse", "local_full_rmse",
    "prediction_at_true_peak", "peak_absolute_error", "peak_relative_error_percent",
    "peak_distance", "zero_rmse", "low_rmse", "high_rmse", "local_sensor_count",
)
ERROR_METRICS = tuple(name for name in METRICS if "rmse" in name or "error" in name or name == "peak_distance")


# Write a compact rectangular table, including an empty file when no seed completed.
def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


# Aggregate complete four-way seeds; nonfinite metrics are excluded pairwise.
def summarize_positivity(rows):
    indexed = {}
    for row in rows:
        key = (int(row["seed"]), row["sampling"], row["method"])
        if key in indexed:
            raise ValueError(f"Duplicate positivity result: {key}")
        indexed[key] = row
    seeds = sorted({key[0] for key in indexed})
    complete = [seed for seed in seeds if all(
        (seed, sampling, method) in indexed for sampling in SAMPLINGS for method in METHODS)]
    summary, wins = [], []
    for sampling in SAMPLINGS:
        for metric in METRICS:
            paired = np.array([[indexed[seed, sampling, method][metric] for method in METHODS]
                               for seed in complete], dtype=float).reshape(-1, 2)
            paired = paired[np.isfinite(paired).all(axis=1)]
            count = len(paired)
            for col, method in enumerate(METHODS):
                summary.append(dict(sampling=sampling, method=method, metric=metric,
                                    n_complete=len(complete), n_valid=count,
                                    mean=float(paired[:, col].mean()) if count else float("nan"),
                                    std=float(paired[:, col].std(ddof=1)) if count > 1 else float("nan")))
            if metric in ERROR_METRICS:
                delta = paired[:, 1] - paired[:, 0]
                ties = np.isclose(paired[:, 1], paired[:, 0], rtol=1e-9, atol=1e-12)
                wins.append(dict(sampling=sampling, metric=metric, n_valid=count,
                                 q_wins=int(np.sum((delta < 0) & ~ties)),
                                 d_wins=int(np.sum((delta > 0) & ~ties)), ties=int(ties.sum()),
                                 q_win_percent=100 * np.sum((delta < 0) & ~ties) / count if count else float("nan"),
                                 mean_q_minus_d=float(delta.mean()) if count else float("nan")))
    return summary, wins


# Run only D and Q, preserving the single-seed preprocessing and LML selection.
def run_positivity_multiseed(args, grid, coordinate_transform, output_path, coordinate_unit):
    if args.n_samples != 800:
        raise ValueError("Positivity multiseed is restricted to N=800.")
    seeds = list(args.positivity_study_seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("At least two distinct nonnegative seeds are required.")
    folder = output_path.parent / "positivity_multiseed"
    folder.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        configuration=vars(args), selected_time=grid.selected_time_label,
        coordinate_unit=coordinate_unit, rotation=vars(coordinate_transform),
        methods={"D": METHODS[0], "Q": METHODS[1]},
        normalization="Coordinate scaler and transform-specific target normalization from Uniform; frozen for E.",
        evaluation="Common unobserved cells exclude U, E and the unused Peak-observed control, as in positivity study; masks vary by seed.",
        full_metrics="full_rmse/local_full_rmse include observed cells; peak prediction is evaluated even if observed.",
        selection="Two starts per method/layout; highest converged finite sensor LML, never ground-truth RMSE.",
        aggregation="Only complete paired seeds; sample std ddof=1; nonfinite bands excluded pairwise; ties rtol=1e-9 atol=1e-12.",
    )
    (folder / "configuration.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    rows, statuses = [], []
    for index, seed in enumerate(seeds, 1):
        print(f"\n=== Positivity multiseed {index}/{len(seeds)} | seed {seed} ===", flush=True)
        seed_args = copy(args)
        seed_args.random_seed = seed
        seed_path = folder / f"seed_{seed}" / "positivity.png"
        try:
            seed_rows, _, _ = run_positivity_study(
                seed_args, grid, coordinate_transform, seed_path, coordinate_unit, lightweight=True)
        except RuntimeError as error:
            statuses.append(dict(seed=seed, status="failed", message=str(error)))
            print(f"Seed {seed} excluded from paired summaries: {error}", flush=True)
        else:
            rows.extend(seed_rows)
            statuses.append(dict(seed=seed, status="complete", message=""))
        write_rows(folder / "metrics.csv", rows)
        write_rows(folder / "seed_status.csv", statuses)
        summary, wins = summarize_positivity(rows)
        write_rows(folder / "summary.csv", summary)
        write_rows(folder / "q_vs_d.csv", wins)
    completed = sum(row["status"] == "complete" for row in statuses)
    print(f"\nComplete paired seeds: {completed}/{len(seeds)}. Results: {folder}", flush=True)
    for row in wins:
        if row["metric"] in ("unseen_global_rmse", "unseen_local_rmse", "peak_absolute_error"):
            print(f"{row['sampling']} | {row['metric']}: Q wins {row['q_wins']}/{row['n_valid']}; "
                  f"ties={row['ties']}", flush=True)
    return rows, summary, wins
