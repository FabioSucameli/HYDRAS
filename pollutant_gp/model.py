# Gaussian Process model construction and prediction.
# This module contains the core GP logic: The GP learns a mapping:(x, y) -> concentrationfrom sparse synthetic sensor measurements.


from __future__ import annotations

import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

# Transform concentration values before fitting.
# Supported transformations:
# - "none": use raw concentration values.
# - "log1p": use log(1 + concentration), useful for highly skewed fields with many near-zero values and localized high peaks.
def transform_targets(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return values
    if transform == "log1p":
        # log1p is only meaningful for non-negative values.
        # If noisy samples become negative, clip them to zero before transforming.
        if np.any(values < 0.0):
            warnings.warn("Negative sample values were clipped to zero before log1p transform.")
        return np.log1p(np.maximum(values, 0.0))
    raise ValueError(f"Unknown target transform: {transform}")

# Map GP predictions back to the original concentration scale.
def inverse_predictions(
    mean: np.ndarray,
    std: np.ndarray,
    transform: str,
) -> tuple[np.ndarray, np.ndarray]:
    if transform == "none":
        return mean, std
    if transform == "log1p":
        raw_mean = np.expm1(mean)
        raw_std = np.exp(mean) * std
        return raw_mean, raw_std
    raise ValueError(f"Unknown target transform: {transform}")


# Build the GP kernel used for spatial reconstruction.
# The kernel is a sum of an RBF kernel (modeling spatial correlations) and a WhiteKernel (modeling noise).
# The RBF length scale can be either isotropic (one shared length scale) or anisotropic (one length scale per input dimension).
def build_kernel(
    n_features: int,
    kernel_mode: str,
    length_scale_lower_bound: float,
    length_scale_upper_bound: float,
    noise_level_initial: float,
    noise_level_lower_bound: float,
    noise_level_upper_bound: float,
):
    
    if kernel_mode == "anisotropic":
        length_scale = np.ones(n_features)
    elif kernel_mode == "isotropic":
        length_scale = 1.0
    else:
        raise ValueError(f"Unknown kernel mode: {kernel_mode}")

    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(
            length_scale=length_scale,
            length_scale_bounds=(length_scale_lower_bound, length_scale_upper_bound),
        )
        + WhiteKernel(
            noise_level=noise_level_initial,
            noise_level_bounds=(noise_level_lower_bound, noise_level_upper_bound),
        )
    )

# Fit a Gaussian Process on standardized spatial coordinates.
# The GP learns a mapping from (x, y) coordinates to concentration values using only the sparse synthetic measurements.
# The function returns the fitted GP model and the coordinate scaler used for standardization, which is needed for making predictions.
def fit_gaussian_process(
    sample_coordinates: np.ndarray,
    sample_values: np.ndarray,
    kernel_mode: str,
    length_scale_lower_bound: float,
    length_scale_upper_bound: float,
    noise_level_initial: float,
    noise_level_lower_bound: float,
    noise_level_upper_bound: float,
    target_transform: str,
    n_restarts: int,
    random_seed: int,
) -> tuple[GaussianProcessRegressor, StandardScaler]:
    
    # Standardize spatial coordinates.
    coordinate_scaler = StandardScaler()
    scaled_coordinates = coordinate_scaler.fit_transform(sample_coordinates)

    # Optionally transform target concentration values
    transformed_values = transform_targets(sample_values, target_transform)

    # Build GP covariance kernel.
    kernel = build_kernel(
        n_features=scaled_coordinates.shape[1],
        kernel_mode=kernel_mode,
        length_scale_lower_bound=length_scale_lower_bound,
        length_scale_upper_bound=length_scale_upper_bound,
        noise_level_initial=noise_level_initial,
        noise_level_lower_bound=noise_level_lower_bound,
        noise_level_upper_bound=noise_level_upper_bound,
    )

    # Create the GP regression model.
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        random_state=random_seed,
    )
    # Fit GP hyperparameters and training data.
    model.fit(scaled_coordinates, transformed_values)
    return model, coordinate_scaler

# Predict GP mean and standard deviation in smaller batches and concatenate the results to avoid memory issues.
def predict_in_batches(
    model: GaussianProcessRegressor,
    coordinate_scaler: StandardScaler,
    prediction_coordinates: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []

    for start in range(0, prediction_coordinates.shape[0], batch_size):
        stop = min(start + batch_size, prediction_coordinates.shape[0])
        scaled_batch = coordinate_scaler.transform(prediction_coordinates[start:stop])
        mean_batch, std_batch = model.predict(scaled_batch, return_std=True)
        means.append(mean_batch)
        stds.append(std_batch)

    return np.concatenate(means), np.concatenate(stds)

