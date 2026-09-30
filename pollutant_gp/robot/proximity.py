"""Reproducible multi-seed proximity diagnostics, without GP fitting."""

import argparse
import csv
from pathlib import Path

import numpy as np
import xarray as xr

from pollutant_gp.data import DEFAULT_CONCENTRATION_VARIABLE, prepare_grid_data
from pollutant_gp.robot.trajectories import measure_robot_proximity, simulate_random_walk


METRICS = (
    "minimum_distance", "same_cell_pair_instants", "same_cell_acquisitions",
    "lt_10_pair_instants", "lt_10_acquisitions", "lt_20_pair_instants", "lt_20_acquisitions",
)


# Aggregate independent seed summaries, not correlated pair-instants as replicates.
def summarize_seeds(summaries):
    if len(summaries) < 2 or len({row["seed"] for row in summaries}) != len(summaries):
        raise ValueError("At least two distinct seeds are required.")
    rows = []
    for metric in METRICS:
        values = np.array([row[metric] for row in summaries], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Multi-seed metrics must be finite; use at least two robots.")
        rows.append(dict(metric=metric, n_seeds=len(values), mean=float(values.mean()),
                         std=float(values.std(ddof=1)), minimum=float(values.min()),
                         maximum=float(values.max())))
    return rows


# Keep motion unchanged and measure proximity only at acquisition instants.
def run_proximity_study(grid, seeds, n_robots=20, n_steps=40, step=60., radius=150.):
    seeds = list(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Provide at least two distinct nonnegative seeds.")
    if n_robots < 2:
        raise ValueError("Pairwise proximity requires at least two robots.")
    summaries, acquisitions = [], []
    for seed in seeds:
        walk = simulate_random_walk(grid, n_robots, n_steps, step, radius, seed)
        rows, summary = measure_robot_proximity(walk)
        summaries.append(dict(seed=seed, step=step, deployment_radius=radius,
                              center_x=float(walk.center[0]), center_y=float(walk.center[1]),
                              **summary))
        acquisitions.extend(dict(seed=seed, **row) for row in rows)
    return summaries, acquisitions, summarize_seeds(summaries)


# Export full-precision numerical results rather than rounded report values.
def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Run geometry-only diagnostics independently of the GP workflow.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nc-file", type=Path, required=True)
    parser.add_argument("--time-index", type=int, default=729)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 101)))
    parser.add_argument("--n-robots", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=40)
    parser.add_argument("--robot-step", type=float, default=60.)
    parser.add_argument("--robot-start-radius", type=float, default=150.)
    parser.add_argument("--coordinate-unit", default="coordinate units",
                        help="Label only: does not convert coordinates or thresholds.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robot_proximity_multiseed"))
    args = parser.parse_args()
    with xr.open_dataset(args.nc_file) as ds:
        grid = prepare_grid_data(ds, DEFAULT_CONCENTRATION_VARIABLE, "time", args.time_index,
                                 "y", "x", "y", "x")
    summaries, acquisitions, aggregate = run_proximity_study(
        grid, args.seeds, args.n_robots, args.n_steps, args.robot_step, args.robot_start_radius)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(nc_file=str(args.nc_file), time_index=args.time_index,
                    distance_unit=args.coordinate_unit)
    for name, rows in (("seeds", summaries), ("acquisitions", acquisitions), ("summary", aggregate)):
        write_csv(args.output_dir / f"proximity_{name}.csv", [dict(**metadata, **row) for row in rows])
    print(f"Seeds: {args.seeds}; {args.n_robots} robots x {args.n_steps} acquisitions (initial included).")
    print("Geometry only: no GP fitting; strict <10/<20 thresholds in coordinate units.")
    for row in aggregate:
        print(f"{row['metric']}: {row['mean']:.4f} +/- {row['std']:.4f} "
              f"[range {row['minimum']:.4f}, {row['maximum']:.4f}]")
    print(f"CSV results: {args.output_dir}")


if __name__ == "__main__":
    main()
