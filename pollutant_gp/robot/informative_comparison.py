# Plot archived IPP, random walk and Pure Uniform scores without fitting or prediction.

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LAYOUTS = ("Informative", "Random walk", "Pure Uniform")
METRICS = ("full_rmse", "local_full_rmse")
SETTINGS = ("nc_file", "variable", "time_index", "time_dim", "x_dim", "y_dim", "x_coordinate",
            "y_coordinate", "current_file", "current_average_hours", "current_u_variable",
            "current_v_variable", "current_time_dim", "physics_source", "kernel_mode",
            "length_scale_lower_bound", "length_scale_upper_bound", "noise_level_initial",
            "noise_level_lower_bound", "noise_level_upper_bound", "n_restarts", "optimizer_seed",
            "noise_std", "target_transform", "clip_negative", "peak_radius", "n_samples",
            "n_robots", "n_steps", "robot_step")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_manifests(ipp, reference):
    config = ipp["configuration"]
    if config["robot_policy"] != "informative" or config["robot_deployment"] != "uniform-domain":
        raise ValueError("Expected IPP versus random walk with uniform-domain deployment.")
    seeds, checkpoints = config["robot_study_seeds"], config["robot_checkpoints"]
    if (len(seeds) < 2 or len(set(seeds)) != len(seeds) or sorted(seeds) != sorted(reference["seeds"])
            or not checkpoints or checkpoints != sorted(set(checkpoints))
            or checkpoints != reference["checkpoints"]):
        raise ValueError("Archives must contain the same seeds and increasing checkpoints (at least two seeds).")
    if "INCLUDING observed cells" not in reference["evaluation"]:
        raise ValueError("Pure Uniform must be evaluated on full domains, including observed cells.")
    if not reference["archives"]:
        raise ValueError("Missing Pure Uniform source configurations.")
    for archive in reference["archives"]:
        for key in SETTINGS:
            if key not in config or key not in archive["configuration"] or config[key] != archive["configuration"][key]:
                raise ValueError(f"Incompatible archived setting: {key}")
        if (ipp["selected_time"] != archive["selected_time"]
                or ipp["coordinate_unit"] != archive["coordinate_unit"]
                or not np.isclose(ipp["rotation"][0], archive["rotation_degrees"], rtol=0, atol=1e-10)):
            raise ValueError("Incompatible snapshot, units or current rotation.")
    return sorted(seeds), checkpoints


def index_rows(rows, layouts, seeds, checkpoints):
    indexed = {}
    for row in rows:
        if row["layout"] not in layouts:
            continue
        key = (int(row["seed"]), int(row["checkpoint"]), row["layout"])
        if key in indexed:
            raise ValueError(f"Duplicate archived metric: {key}")
        values = {metric: float(row[metric]) for metric in METRICS}
        if any(not np.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError(f"Invalid RMSE: {key}")
        indexed[key] = dict(seed=key[0], checkpoint=key[1], layout=key[2], **values)
    expected = {(seed, count, layout) for seed in seeds for count in checkpoints for layout in layouts}
    if set(indexed) != expected:
        raise ValueError("Missing or unexpected seed/checkpoint/layout; incomplete runs cannot be compared.")
    return indexed


def combine_rows(ipp_rows, reference_rows, seeds, checkpoints):
    ipp = index_rows(ipp_rows, LAYOUTS[:2], seeds, checkpoints)
    reference = index_rows(reference_rows, ("Uniform-domain", "Pure Uniform"), seeds, checkpoints)
    # The two legacy formats hash fields differently. The shared random-walk scores
    # provide an additional numerical consistency check at every seed/checkpoint.
    for seed in seeds:
        for count in checkpoints:
            for metric in METRICS:
                if not np.isclose(ipp[(seed, count, "Random walk")][metric],
                                  reference[(seed, count, "Uniform-domain")][metric], rtol=1e-6, atol=1e-8):
                    raise ValueError(f"Random-walk archive mismatch: seed={seed}, N={count}, {metric}")
            ipp[(seed, count, "Pure Uniform")] = reference[(seed, count, "Pure Uniform")]
    return [ipp[(seed, count, layout)] for seed in seeds for count in checkpoints for layout in LAYOUTS]


def aggregate(rows, checkpoints):
    summary = []
    for count in checkpoints:
        for layout in LAYOUTS:
            selected = [row for row in rows if row["checkpoint"] == count and row["layout"] == layout]
            for metric in METRICS:
                values = np.array([row[metric] for row in selected])
                summary.append(dict(checkpoint=count, layout=layout, metric=metric, n_seeds=len(values),
                                    mean=float(values.mean()), std=float(values.std(ddof=1))))
    return summary


def plot_comparison(summary, path, radius, unit):
    styles = (("#0072B2", "o", "-", "IPP"), ("#D55E00", "s", "--", "Random walk"),
              ("#009E73", "^", "-.", "Pure Uniform (static)"))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, layout="constrained")
    for ax, metric, title in zip(axes, METRICS, ("Full marine domain", f"Full peak disk | radius {radius:g} {unit}")):
        for layout, (color, marker, line, label) in zip(LAYOUTS, styles):
            data = [row for row in summary if row["layout"] == layout and row["metric"] == metric]
            x = np.array([row["checkpoint"] for row in data])
            mean = np.array([row["mean"] for row in data])
            std = np.array([row["std"] for row in data])
            ax.plot(x, mean, color=color, marker=marker, linestyle=line, linewidth=1.8,
                    markersize=4, label=label)
            ax.fill_between(x, np.maximum(0, mean - std), mean + std, color=color, alpha=.10)
        ax.set(title=title, ylabel="RMSE (including observed cells)", ylim=(0, None))
        ax.grid(axis="y", color="#dddddd", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=3, fontsize=10)
    axes[-1].set_xticks(sorted({row["checkpoint"] for row in summary}))
    axes[-1].set_xlabel("Acquired measurements")
    fig.suptitle(f"Informative trajectories and static sampling | {summary[0]['n_seeds']} seeds\n"
                 "Mean +/- 1 sample standard deviation")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_comparison(ipp_dir, uniform_dir, output_dir):
    ipp_dir, uniform_dir, output_dir = (Path(path).resolve() for path in (ipp_dir, uniform_dir, output_dir))
    for source in (ipp_dir, uniform_dir):
        if output_dir == source or source in output_dir.parents or output_dir in source.parents:
            raise ValueError("Use a separate output directory outside the input archives.")
    sources = [folder / name for folder in (ipp_dir, uniform_dir) for name in ("configuration.json", "metrics.csv")]
    ipp = json.loads(sources[0].read_text(encoding="utf-8"))
    uniform = json.loads(sources[2].read_text(encoding="utf-8"))
    seeds, checkpoints = validate_manifests(ipp, uniform)
    rows = combine_rows(read_csv(sources[1]), read_csv(sources[3]), seeds, checkpoints)
    summary = aggregate(rows, checkpoints)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metrics.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "final_summary.csv", [row for row in summary if row["checkpoint"] == checkpoints[-1]])
    provenance = dict(seeds=seeds, checkpoints=checkpoints,
                      sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                      evaluation="full valid domain and full peak disk, INCLUDING observed cells",
                      checks="matching configurations, complete seed/checkpoint coverage and shared random-walk RMSE",
                      fitting="none; archived scalar metrics only", std="sample standard deviation, ddof=1")
    (output_dir / "configuration.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    plot_comparison(summary, output_dir / "ipp_random_pure_uniform.png", ipp["configuration"]["peak_radius"],
                    ipp["coordinate_unit"])
    print(f"Compared {len(seeds)} seeds at {len(checkpoints)} checkpoints. No GP fitting or prediction.")
    print(f"Saved comparison: {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipp-dir", type=Path, default=Path("outputs/ipp_uniform_100seeds"))
    parser.add_argument("--uniform-dir", type=Path, default=Path("outputs/robot_deployment_comparison_local"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ipp_final_comparison"))
    args = parser.parse_args()
    run_comparison(args.ipp_dir, args.uniform_dir, args.output_dir)


if __name__ == "__main__":
    main()
