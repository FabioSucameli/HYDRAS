# Gaussian Process model construction and prediction.
# This module contains the core GP logic: The GP learns a mapping:(x, y) -> concentrationfrom sparse synthetic sensor measurements.


from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

# Diagnostics collected for one bounded L-BFGS-B optimization run.
@dataclass(frozen=True)
class OptimizerRunDiagnostics:

    run_index: int
    initial_theta: np.ndarray
    optimized_theta: np.ndarray
    initial_lml: float
    final_lml: float
    lml_gradient: np.ndarray
    success: bool
    status: int
    message: str
    iterations: int | None
    function_evaluations: int | None

# Initial value, fitted value, bounds, and LML gradient for one parameter.
@dataclass(frozen=True)
class HyperparameterDiagnostics:

    name: str
    initial_value: float
    optimized_value: float
    lower_bound: float
    upper_bound: float
    at_lower_bound: bool
    at_upper_bound: bool
    lml_gradient: float

# Complete optimization diagnostics for one Gaussian Process fit.
@dataclass(frozen=True)
class GPOptimizationDiagnostics:

    optimizer_seed: int
    n_restarts: int
    coordinate_mean: np.ndarray
    coordinate_scale: np.ndarray
    target_mean: float
    target_scale: float
    initial_lml: float
    final_lml: float
    final_lml_gradient_norm: float
    selected_run_index: int
    optimizer_runs: tuple[OptimizerRunDiagnostics, ...]
    hyperparameters: tuple[HyperparameterDiagnostics, ...]
    standardized_length_scales: np.ndarray
    physical_length_scales: np.ndarray

    @property
    def length_scale_lower_bound_hit(self) -> bool:
        return any(
            parameter.at_lower_bound
            for parameter in self.hyperparameters
            if "length_scale" in parameter.name
        )

    @property
    def length_scale_upper_bound_hit(self) -> bool:
        return any(
            parameter.at_upper_bound
            for parameter in self.hyperparameters
            if "length_scale" in parameter.name
        )

    @property
    def optimizer_failure_count(self) -> int:
        return sum(not run.success for run in self.optimizer_runs)

# Run scikit-learn's default bounded optimizer while retaining its result.
class _RecordingLBFGSBOptimizer:

    def __init__(self) -> None:
        self.runs: list[OptimizerRunDiagnostics] = []

    def __call__(self, objective, initial_theta, bounds):
        initial_theta = np.asarray(initial_theta, dtype=float).copy()
        bounds = np.asarray(bounds, dtype=float)
        initial_lml = -float(objective(initial_theta, eval_gradient=False))

        result = minimize(
            objective,
            initial_theta,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
        )

        if not result.success:
            warnings.warn(
                f"L-BFGS-B failed to converge (status={result.status}): {result.message}",
                ConvergenceWarning,
                stacklevel=2,
            )

        self.runs.append(
            OptimizerRunDiagnostics(
                run_index=len(self.runs),
                initial_theta=initial_theta,
                optimized_theta=np.asarray(result.x, dtype=float).copy(),
                initial_lml=initial_lml,
                final_lml=-float(result.fun),
                lml_gradient=-np.asarray(result.jac, dtype=float).copy(),
                success=bool(result.success),
                status=int(result.status),
                message=str(result.message),
                iterations=int(result.nit) if hasattr(result, "nit") else None,
                function_evaluations=int(result.nfev) if hasattr(result, "nfev") else None,
            )
        )
        return result.x, result.fun

# Expand vector kernel hyperparameters into one name per optimized value.
def _expanded_hyperparameter_names(kernel) -> list[str]:
  
    names: list[str] = []
    for hyperparameter in kernel.hyperparameters:
        if hyperparameter.fixed:
            continue
        if hyperparameter.n_elements == 1:
            names.append(hyperparameter.name)
        else:
            names.extend(
                f"{hyperparameter.name}[{index}]"
                for index in range(hyperparameter.n_elements)
            )
    return names

# Treat values within 0.1 percent of a bound as bound hits.
def _is_at_bound(value: float, bound: float) -> bool:

    return bool(np.isclose(value, bound, rtol=1e-3, atol=np.finfo(float).eps * 10.0))

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
    constant_value_initial: float = 1.0,
    length_scale_initial: float | np.ndarray | None = None,
):
    if constant_value_initial <= 0.0:
        raise ValueError("The ConstantKernel initial value must be positive.")

    requested_length_scales = (
        None
        if length_scale_initial is None
        else np.asarray(length_scale_initial, dtype=float).reshape(-1)
    )
    if kernel_mode == "anisotropic":
        if requested_length_scales is None:
            length_scale = np.ones(n_features)
        elif requested_length_scales.size == 1:
            length_scale = np.repeat(requested_length_scales, n_features)
        elif requested_length_scales.size == n_features:
            length_scale = requested_length_scales
        else:
            raise ValueError(
                f"An anisotropic kernel with {n_features} features requires one or "
                f"{n_features} initial length scales."
            )
    elif kernel_mode == "isotropic":
        if requested_length_scales is None:
            length_scale = 1.0
        elif requested_length_scales.size == 1:
            length_scale = float(requested_length_scales[0])
        else:
            raise ValueError("An isotropic kernel requires one initial length scale.")
    else:
        raise ValueError(f"Unknown kernel mode: {kernel_mode}")

    if np.any(np.asarray(length_scale) <= 0.0):
        raise ValueError("All initial length scales must be positive.")

    return (
        ConstantKernel(constant_value_initial, (1e-3, 1e3))
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
    optimizer_seed: int,
    constant_value_initial: float = 1.0,
    length_scale_initial: float | np.ndarray | None = None,
) -> tuple[GaussianProcessRegressor, StandardScaler, GPOptimizationDiagnostics]:
    
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
        constant_value_initial=constant_value_initial,
        length_scale_initial=length_scale_initial,
    )
    initial_theta = kernel.theta.copy()
    original_bounds = np.exp(kernel.bounds)
    parameter_names = _expanded_hyperparameter_names(kernel)
    optimizer = _RecordingLBFGSBOptimizer()

    # Create the GP regression model.
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        optimizer=optimizer,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        random_state=optimizer_seed,
    )
    # Fit GP hyperparameters and training data.
    model.fit(scaled_coordinates, transformed_values)

    final_theta = model.kernel_.theta.copy()
    initial_values = np.exp(initial_theta)
    optimized_values = np.exp(final_theta)
    _, final_lml_gradient = model.log_marginal_likelihood(
        final_theta,
        eval_gradient=True,
    )
    initial_lml = float(model.log_marginal_likelihood(initial_theta))

    hyperparameters = tuple(
        HyperparameterDiagnostics(
            name=name,
            initial_value=float(initial_value),
            optimized_value=float(optimized_value),
            lower_bound=float(bounds[0]),
            upper_bound=float(bounds[1]),
            at_lower_bound=_is_at_bound(float(optimized_value), float(bounds[0])),
            at_upper_bound=_is_at_bound(float(optimized_value), float(bounds[1])),
            lml_gradient=float(gradient),
        )
        for name, initial_value, optimized_value, bounds, gradient in zip(
            parameter_names,
            initial_values,
            optimized_values,
            original_bounds,
            final_lml_gradient,
            strict=True,
        )
    )

    standardized_length_scales = np.asarray(
        model.kernel_.k1.k2.length_scale,
        dtype=float,
    ).reshape(-1)
    if standardized_length_scales.size == 1:
        physical_length_scales = standardized_length_scales[0] * coordinate_scaler.scale_
    else:
        physical_length_scales = standardized_length_scales * coordinate_scaler.scale_

    selected_run_index = int(
        np.argmax([run.final_lml for run in optimizer.runs])
    )
    target_scale = float(np.std(transformed_values))
    if target_scale == 0.0:
        target_scale = 1.0

    diagnostics = GPOptimizationDiagnostics(
        optimizer_seed=optimizer_seed,
        n_restarts=n_restarts,
        coordinate_mean=coordinate_scaler.mean_.copy(),
        coordinate_scale=coordinate_scaler.scale_.copy(),
        target_mean=float(np.mean(transformed_values)),
        target_scale=target_scale,
        initial_lml=initial_lml,
        final_lml=float(model.log_marginal_likelihood_value_),
        final_lml_gradient_norm=float(np.linalg.norm(final_lml_gradient)),
        selected_run_index=selected_run_index,
        optimizer_runs=tuple(optimizer.runs),
        hyperparameters=hyperparameters,
        standardized_length_scales=standardized_length_scales.copy(),
        physical_length_scales=np.asarray(physical_length_scales, dtype=float).copy(),
    )
    return model, coordinate_scaler, diagnostics


# Build the exact predictive GP state for a supplied theta without optimization.
def fit_gaussian_process_at_fixed_theta(
    sample_coordinates: np.ndarray,
    sample_values: np.ndarray,
    kernel_template,
    theta: np.ndarray,
    target_transform: str,
) -> tuple[GaussianProcessRegressor, StandardScaler]:
    coordinate_scaler = StandardScaler()
    scaled_coordinates = coordinate_scaler.fit_transform(sample_coordinates)
    transformed_values = transform_targets(sample_values, target_transform)
    fixed_kernel = kernel_template.clone_with_theta(np.asarray(theta, dtype=float))

    model = GaussianProcessRegressor(
        kernel=fixed_kernel,
        alpha=1e-10,
        optimizer=None,
        normalize_y=True,
    )
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
