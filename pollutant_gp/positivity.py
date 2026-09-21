"""Controlled target-transform comparison using the shared prediction moments."""

from __future__ import annotations

import csv
from dataclasses import replace
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from pollutant_gp.model import concentration_moments, fit_gaussian_process, latent_variance
from pollutant_gp.peak import extract_peak_profiles, measure_peak
from pollutant_gp.peak_kernel import INITIALIZATIONS, print_kernel_fit, select_converged_fit
from pollutant_gp.peak_sampling import common_unsampled_mask, make_peak_sampling_sets
from pollutant_gp.reconstruction import reconstruct_field
from pollutant_gp.sampling import sample_sensor_points
from pollutant_gp.spatial import maybe_transform_coordinates


# Transform noiseless nonnegative observations, including exact zeros.
def transform_concentrations(values, transform):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("The positivity study requires finite nonnegative observations.")
    return {"none": lambda x: x.copy(), "log1p": np.log1p, "sqrt": np.sqrt}[transform](values)


# Evaluate point predictions without mixing full-grid and common-unobserved metrics.
def prediction_metrics(truth, raw, prediction, valid, common, local):
    if not np.all(np.isfinite(prediction[valid])) or not np.all(np.isfinite(raw[valid])):
        raise ValueError("Nonfinite predictions; no finite score can be reported.")
    error = prediction - truth
    negatives = raw[valid] < 0
    background = common & (truth == 0)
    raw_rmse = float(np.sqrt(np.mean((raw[common] - truth[common])**2)))
    rmse = float(np.sqrt(np.mean(error[common]**2)))
    metrics = {
        "full_rmse": float(np.sqrt(np.mean(error[valid]**2))),
        "full_r2": float(r2_score(truth[valid], prediction[valid])),
        "unseen_global_rmse": rmse,
        "raw_unseen_global_rmse": raw_rmse,
        "clipping_rmse_reduction": raw_rmse - rmse,
        "clipping_rmse_reduction_percent": 100 * (raw_rmse - rmse) / raw_rmse if raw_rmse else 0.,
        "unseen_local_rmse": float(np.sqrt(np.mean(error[common & local]**2))),
        "unseen_global_mae": float(np.mean(np.abs(error[common]))),
        "raw_negative_percent": float(100 * negatives.mean()),
        "raw_min": float(np.min(raw[valid])),
        "raw_negative_mean": float(raw[valid][negatives].mean()) if negatives.any() else 0.,
        "zero_cell_mean_prediction": float(prediction[background].mean()) if background.any() else float("nan"),
    }
    # Ground-truth bands partition the same common-unobserved global domain.
    for name, band in (("zero", truth == 0), ("low", (truth > 0) & (truth <= 1)),
                       ("high", truth > 1)):
        mask = common & band
        count = int(mask.sum())
        squared_error_sum = float(np.sum(error[mask]**2))
        metrics.update({
            f"{name}_count": count,
            f"{name}_percent": 100 * count / int(common.sum()),
            f"{name}_rmse": float(np.sqrt(squared_error_sum / count)) if count else float("nan"),
            f"{name}_weighted_mse": squared_error_sum / int(common.sum()),
        })
    return metrics


# Plot the three primary estimators on identical peak-centred profiles.
def plot_positivity_profiles(profiles, path, title, unit, show):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    styles = (("Direct + clip", "#0072B2", "--"),
              ("Log1p mean + clip", "#D55E00", "-."), ("Square-root mean", "#009E73", "-"))
    for row, sampling in enumerate(("Uniform", "Locally enriched (oracle)")):
        for col in range(2):
            axis = axes[row, col]
            reference = profiles[(sampling, styles[0][0])][col]
            axis.plot(reference.distances, reference.truth, color="black", linewidth=2,
                      drawstyle="steps-mid", label="Ground truth")
            for method, color, style in styles:
                profile = profiles[(sampling, method)][col]
                axis.plot(profile.distances, profile.prediction, color=color, linestyle=style,
                          drawstyle="steps-mid", label=method)
            axis.set(title=f"{sampling} | {reference.label}", ylabel="Concentration",
                     xlabel=f"Signed distance from true maximum ({unit})")
            axis.axvline(0, color=".6", linestyle=":")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(figure)


# Reuse matched sensor sets and two prescribed starts, selecting within each target space only.
def run_positivity_study(args, grid, coordinate_transform, output_path, coordinate_unit):
    if coordinate_transform is None:
        raise ValueError("Current-informed coordinates are required.")
    if np.any(grid.field[grid.valid_mask] < 0):
        raise ValueError("The reference field must be nonnegative.")
    _, _, uniform = sample_sensor_points(grid, args.n_samples, 0., args.random_seed)
    sets, local = make_peak_sampling_sets(grid, uniform, args.peak_radius,
                                         args.peak_local_replacements, args.random_seed)
    common = common_unsampled_mask(grid, sets)
    if not common.any() or not (common & local).any():
        raise ValueError("Common unobserved global and local cells are required.")
    xy = np.column_stack((grid.x_grid.ravel(), grid.y_grid.ravel()))
    rotated = maybe_transform_coordinates(xy, coordinate_transform)
    scaler = StandardScaler().fit(rotated[uniform])
    base = output_path.with_name(f"{output_path.stem}_seed_{args.random_seed}_positivity")
    base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base.with_name(base.name + "_metrics.csv")
    optimizer_seed = args.random_seed if args.optimizer_seed is None else args.optimizer_seed
    print("\n=== Positivity study: two-scale current-informed GP ===", flush=True)
    print(f"Settings: {vars(args)}")
    print(f"Rotation: {coordinate_transform}")
    print(f"Coordinate mean/scale: {scaler.mean_} / {scaler.scale_}")
    print(f"Common unobserved cells: global={common.sum()}, local={(common & local).sum()}")
    print("Noiseless oracle control. Two starts per target/layout; selection uses sensor LML only.")
    print("Do NOT compare LML between transforms or layouts. Predictions use latent field variance.")
    rows, profiles = [], {}
    saved_fields = dict(truth=grid.field, valid=grid.valid_mask, common_unobserved=common, local=local)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for sampling in ("Uniform", "Locally enriched (oracle)"):
            indices = sets[sampling]
            layout_key = "uniform" if sampling == "Uniform" else "enriched"
            saved_fields[f"{layout_key}_sensor_indices"] = indices
            values = grid.field.ravel()[indices].astype(float)
            print(f"\n{sampling}: sensors={len(indices)}, local={local.ravel()[indices].sum()}", flush=True)
            for transform in ("none", "log1p", "sqrt"):
                reference = transform_concentrations(grid.field.ravel()[uniform], transform)
                normalization = (float(reference.mean()), float(reference.std()) or 1.)
                transformed = transform_concentrations(values, transform)
                print(f"Target={transform}; frozen Uniform mean/std={normalization}", flush=True)
                candidates = []
                started = perf_counter()
                for name, first, second, amplitude in INITIALIZATIONS["Double"]:
                    model, fitted_scaler, diagnostic = fit_gaussian_process(
                        rotated[indices], transformed, args.kernel_mode,
                        args.length_scale_lower_bound, args.length_scale_upper_bound,
                        args.noise_level_initial, args.noise_level_lower_bound, args.noise_level_upper_bound,
                        "none", 0, optimizer_seed, constant_value_initial=amplitude,
                        length_scale_initial=first, second_length_scale_initial=second,
                        coordinate_scaler=scaler, target_normalization=normalization,
                    )
                    print_kernel_fit(name, diagnostic, coordinate_unit)
                    candidates.append((name, model, fitted_scaler, diagnostic))
                fit_seconds = perf_counter() - started
                name, model, _, diagnostic = select_converged_fit(candidates)
                print(f"Selected {transform}: {name}; LML={diagnostic.final_lml:.12g}; {model.kernel_}", flush=True)
                started = perf_counter()
                latent = reconstruct_field(grid, model, scaler, args.prediction_batch_size, "none", False,
                                           coordinate_transform, target_normalization=normalization)
                variance = latent_variance(latent.std_field, model.kernel_.k2.noise_level, normalization[1])
                mean, variance_c = concentration_moments(latent.mean_field, variance, transform)
                saved_fields[f"{layout_key}_{transform}_mean"] = mean
                prediction_seconds = perf_counter() - started
                variants = [("Direct", mean, False), ("Direct + clip", mean, True)] if transform == "none" else (
                    [("Log1p mean", mean, False), ("Log1p mean + clip", mean, True)] if transform == "log1p" else
                    [("Square-root mean", mean, False)])
                for method, raw, clip in variants:
                    postprocess_started = perf_counter()
                    prediction = np.maximum(raw, 0.) if clip else raw.copy()
                    postprocess_seconds = perf_counter() - postprocess_started
                    metrics = prediction_metrics(grid.field, raw, prediction, grid.valid_mask, common, local)
                    # Spread is for the un-clipped transformed marginal, not Gaussian error bars for a clipped map.
                    reconstructed = replace(
                        latent, mean_field=prediction, std_field=np.sqrt(variance_c),
                        mse=metrics["full_rmse"]**2, rmse=metrics["full_rmse"],
                        mae=float(np.mean(np.abs(prediction[grid.valid_mask] - grid.field[grid.valid_mask]))),
                        r2=metrics["full_r2"], min_prediction_before_clipping=metrics["raw_min"],
                        negative_prediction_count=int(np.count_nonzero(raw[grid.valid_mask] < 0)),
                        negative_prediction_fraction=metrics["raw_negative_percent"] / 100,
                        mean_negative_prediction=metrics["raw_negative_mean"],
                    )
                    peak = measure_peak(grid, reconstructed, indices, values, args.peak_radius)
                    row = dict(seed=args.random_seed, sampling=sampling, method=method,
                               initialization=name, lml=diagnostic.final_lml,
                               white_level=model.kernel_.k2.noise_level, **metrics,
                               prediction_at_true_peak=peak.prediction_at_true_peak,
                               predicted_global_max=peak.predicted_max,
                               peak_distance=peak.peak_location_error,
                               true_peak_sampled=peak.true_peak_sampled,
                               local_full_rmse=peak.local_rmse,
                               fit_seconds=fit_seconds, prediction_seconds=prediction_seconds,
                               postprocess_seconds=postprocess_seconds)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    rows.append(row)
                    print(f"{method}: unseen global RMSE={row['unseen_global_rmse']:.8g}; "
                          f"local={row['unseen_local_rmse']:.8g}; peak={peak.prediction_at_true_peak:.8g}; "
                          f"negative={row['raw_negative_percent']:.3f}%; min={row['raw_min']:.8g}", flush=True)
                    if method in ("Direct + clip", "Square-root mean"):
                        print(f"  Common unobserved bands [c=0, 0<c<=1, c>1]: "
                              f"counts={[metrics[f'{band}_count'] for band in ('zero', 'low', 'high')]}; "
                              f"RMSE={[metrics[f'{band}_rmse'] for band in ('zero', 'low', 'high')]}")
                        print(f"  Mean prediction on c=0: {metrics['zero_cell_mean_prediction']:.8g}; "
                              f"clipping RMSE reduction: {metrics['clipping_rmse_reduction']:.8g} "
                              f"({metrics['clipping_rmse_reduction_percent']:.5g}%)")
                    if method in ("Direct + clip", "Log1p mean + clip", "Square-root mean"):
                        profiles[(sampling, method)] = extract_peak_profiles(grid, reconstructed, peak, coordinate_transform)
    fields_path = base.with_name(base.name + "_fields.npz")
    np.savez_compressed(fields_path, **saved_fields)
    figure_path = base.with_name(base.name + "_profiles.png")
    plot_positivity_profiles(profiles, figure_path,
                            f"{args.nc_file.stem} | time {args.time_index} | N={args.n_samples} | seed {args.random_seed}",
                            coordinate_unit, args.show)
    print(f"Saved metrics: {csv_path}\nSaved profiles: {figure_path}\nSaved mean fields and masks: {fields_path}")
    print("Times are repeated across estimators from the same fit; clipping controls require no refit.")
    return rows, profiles, common
