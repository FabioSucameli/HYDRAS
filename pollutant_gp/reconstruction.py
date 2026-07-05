# Field reconstruction and metric computation.

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from pollutant_gp.model import inverse_predictions, predict_in_batches
from pollutant_gp.spatial import RotationTransform, maybe_transform_coordinates
from pollutant_gp.types import GridData, ReconstructionResult

# Reconstruct the concentration field on all valid grid cells.
# The trained Gaussian Process is evaluated on the full valid marine domain.
# The function returns both:
# - the predicted concentration field;
# - the predictive uncertainty field;
# - reconstruction metrics against the ground truth.
def reconstruct_field(
    grid_data: GridData,
    model: GaussianProcessRegressor,
    coordinate_scaler: StandardScaler,
    batch_size: int,
    target_transform: str,
    clip_negative: bool,
    coordinate_transform: RotationTransform | None = None,
) -> ReconstructionResult:
    
    # Extract coordinates of valid cells and reshape them into a 2D array of (x, y) pairs for prediction.
    valid_flat_indices = np.flatnonzero(grid_data.valid_mask.ravel())
    x_flat = grid_data.x_grid.ravel()
    y_flat = grid_data.y_grid.ravel()
    
    prediction_coordinates = np.column_stack(
        [x_flat[valid_flat_indices], y_flat[valid_flat_indices]]
    )
    prediction_coordinates = maybe_transform_coordinates(
        prediction_coordinates,
        coordinate_transform,
    )
    
    # Predict GP mean and standard deviation
    predicted_mean, predicted_std = predict_in_batches(
        model=model,
        coordinate_scaler=coordinate_scaler,
        prediction_coordinates=prediction_coordinates,
        batch_size=batch_size,
    )
    # If targets were transformed before training
    predicted_mean, predicted_std = inverse_predictions(
        predicted_mean,
        predicted_std,
        target_transform,
    )

    # Concentration cannot be negative.
    if clip_negative:
        predicted_mean = np.maximum(predicted_mean, 0.0)

    # Create flat arrays initialized with NaN.
    mean_flat = np.full(grid_data.field.size, np.nan, dtype=float)
    std_flat = np.full(grid_data.field.size, np.nan, dtype=float)

    # Insert predictions only at valid marine cells.
    mean_flat[valid_flat_indices] = predicted_mean
    std_flat[valid_flat_indices] = predicted_std

    # Extract ground-truth values on the same valid cells.
    truth = grid_data.field.ravel()[valid_flat_indices]

    # Compute reconstruction metrics against the ground truth.
    mse = mean_squared_error(truth, predicted_mean)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(truth, predicted_mean)
    r2 = r2_score(truth, predicted_mean)

    # Return reconstructed maps and scalar metrics.
    return ReconstructionResult(
        mean_field=mean_flat.reshape(grid_data.field.shape),
        std_field=std_flat.reshape(grid_data.field.shape),
        mse=float(mse),
        rmse=rmse,
        mae=float(mae),
        r2=float(r2),
    )
