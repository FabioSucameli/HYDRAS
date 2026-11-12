# Reproducible measurement noise, separate from deployment and motion RNGs.

import numpy as np


def sensor_errors(seed, count, noise_std):
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("Sensor noise standard deviation must be finite and nonnegative.")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 5]))
    return noise_std * rng.standard_normal(count)


def training_observations(field, visits, errors, noise_std):
    visits = np.asarray(visits, dtype=int).ravel()
    if noise_std > 0:
        # Every acquisition is retained, including independent noisy revisits.
        return visits.copy(), field.ravel()[visits].astype(float) + errors[:len(visits)]
    _, first = np.unique(visits, return_index=True)
    indices = visits[np.sort(first)]
    return indices, field.ravel()[indices].astype(float)


def full_band_metrics(truth, prediction, valid):
    metrics = {}
    for name, band in (("zero", truth == 0), ("low", (truth > 0) & (truth <= 1)),
                       ("high", truth > 1)):
        mask = valid & band
        metrics[f"full_{name}_count"] = int(mask.sum())
        metrics[f"full_{name}_rmse"] = (float(np.sqrt(np.mean((prediction[mask] - truth[mask])**2)))
                                         if mask.any() else float("nan"))
    return metrics
