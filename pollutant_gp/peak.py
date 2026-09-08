# Post-fit diagnostics of the simulated concentration maximum.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pollutant_gp.spatial import RotationTransform
from pollutant_gp.types import GridData, ReconstructionResult


@dataclass(frozen=True)
class PeakDiagnostics:
    true_peak_flat_index: int
    predicted_peak_flat_index: int
    true_peak_xy: tuple[float, float]
    predicted_peak_xy: tuple[float, float]
    true_peak_count: int
    predicted_peak_count: int
    true_max: float
    sampled_truth_max: float
    observed_max: float
    predicted_max: float
    prediction_at_true_peak: float
    peak_underestimation: float
    peak_underestimation_fraction: float | None
    peak_location_error: float
    nearest_sensor_distance: float
    true_peak_sampled: bool
    radius: float
    local_cell_count: int
    local_sensor_count: int
    global_rmse: float
    local_rmse: float


@dataclass(frozen=True)
class PeakProfile:
    label: str
    angle_degrees: float
    distances: np.ndarray
    xy: np.ndarray
    truth: np.ndarray
    prediction: np.ndarray


# Measure the peak and a fixed-radius disk without changing samples or predictions.
def measure_peak(
    grid_data: GridData,
    reconstruction: ReconstructionResult,
    sampled_flat_indices: np.ndarray,
    sample_values: np.ndarray,
    radius: float,
) -> PeakDiagnostics:
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Peak radius must be finite and positive in coordinate units.")
    shape = grid_data.field.shape
    if len(shape) != 2 or any(
        array.shape != shape
        for array in (grid_data.valid_mask, grid_data.x_grid, grid_data.y_grid,
                      reconstruction.mean_field)
    ):
        raise ValueError("Peak diagnostics require matching two-dimensional grids.")
    valid = grid_data.valid_mask.ravel()
    truth = grid_data.field.ravel()
    prediction = reconstruction.mean_field.ravel()
    xy = np.column_stack((grid_data.x_grid.ravel(), grid_data.y_grid.ravel()))
    if not np.any(valid) or not all(
        np.all(np.isfinite(array[valid])) for array in (truth, prediction, xy)
    ):
        raise ValueError("Valid cells must have finite coordinates, truth and predictions.")
    indices = np.asarray(sampled_flat_indices)
    values = np.asarray(sample_values, dtype=float)
    if (indices.ndim != 1 or indices.size == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or values.shape != indices.shape or not np.all(np.isfinite(values))):
        raise ValueError("Provide nonempty integer sensor indices and matching finite observations.")
    if (np.any(indices < 0) or np.any(indices >= truth.size)
            or np.unique(indices).size != indices.size or not np.all(valid[indices])):
        raise ValueError("Sensor indices must be unique and refer to valid grid cells.")

    valid_indices = np.flatnonzero(valid)
    true_index = int(valid_indices[np.argmax(truth[valid])])
    predicted_index = int(valid_indices[np.argmax(prediction[valid])])
    true_max = float(truth[true_index])
    true_maxima = valid & (truth == true_max)
    predicted_maxima = valid & (prediction == prediction[predicted_index])
    sampled = np.zeros(truth.size, dtype=bool)
    sampled[indices] = True
    distances = np.linalg.norm(xy - xy[true_index], axis=1)
    local = valid & (distances <= radius)
    underestimation = true_max - float(prediction[true_index])

    # For tied maxima, report the first row-major cell and retain the tie counts.
    return PeakDiagnostics(
        true_peak_flat_index=true_index,
        predicted_peak_flat_index=predicted_index,
        true_peak_xy=tuple(float(v) for v in xy[true_index]),
        predicted_peak_xy=tuple(float(v) for v in xy[predicted_index]),
        true_peak_count=int(true_maxima.sum()),
        predicted_peak_count=int(predicted_maxima.sum()),
        true_max=true_max,
        sampled_truth_max=float(np.max(truth[indices])),
        observed_max=float(np.max(values)),
        predicted_max=float(prediction[predicted_index]),
        prediction_at_true_peak=float(prediction[true_index]),
        peak_underestimation=underestimation,
        peak_underestimation_fraction=underestimation / true_max if true_max > 0 else None,
        peak_location_error=float(distances[predicted_index]),
        nearest_sensor_distance=float(np.min(distances[indices])),
        true_peak_sampled=bool(sampled[true_index]),
        radius=float(radius),
        local_cell_count=int(local.sum()),
        local_sensor_count=int(np.count_nonzero(local & sampled)),
        global_rmse=float(np.sqrt(np.mean((prediction[valid] - truth[valid]) ** 2))),
        local_rmse=float(np.sqrt(np.mean((prediction[local] - truth[local]) ** 2))),
    )


# Extract two nearest-cell map profiles; never interpolate across land or refit the GP.
def extract_peak_profiles(
    grid_data: GridData,
    reconstruction: ReconstructionResult,
    diagnostics: PeakDiagnostics,
    coordinate_transform: RotationTransform | None = None,
) -> tuple[PeakProfile, PeakProfile]:
    x = grid_data.x_grid[0, :]
    y = grid_data.y_grid[:, 0]
    if (x.size < 2 or y.size < 2
            or not np.allclose(grid_data.x_grid, x[None, :], rtol=0, atol=1e-8)
            or not np.allclose(grid_data.y_grid, y[:, None], rtol=0, atol=1e-8)):
        raise ValueError("Peak profiles require a rectilinear grid with at least two cells per axis.")
    for axis in (x, y):
        differences = np.diff(axis)
        if not np.all(np.isfinite(axis)) or not (
            np.all(differences > 0) or np.all(differences < 0)
        ):
            raise ValueError("Profile coordinates must be finite and strictly monotonic.")
    spacing = min(float(np.min(np.abs(np.diff(x)))), float(np.min(np.abs(np.diff(y)))))
    half_count = max(1, int(np.ceil(diagnostics.radius / (spacing / 2))))
    if half_count > 10000:
        raise ValueError("Peak radius is too large relative to the grid spacing for profiles.")
    distances = np.linspace(-diagnostics.radius, diagnostics.radius, 2 * half_count + 1)
    maps = np.stack((grid_data.field, reconstruction.mean_field), axis=-1)
    maps = np.where(grid_data.valid_mask[..., None], maps, np.nan)
    interpolate = RegularGridInterpolator(
        (y, x), maps, method="nearest", bounds_error=False, fill_value=np.nan,
    )
    angle = coordinate_transform.angle_degrees if coordinate_transform else 0.0
    labels = ("Parallel", "Perpendicular") if coordinate_transform else ("X direction", "Y direction")
    profiles = []
    for label, offset in zip(labels, (0.0, 90.0)):
        radians = np.deg2rad(angle + offset)
        direction = np.array([np.cos(radians), np.sin(radians)])
        points = np.asarray(diagnostics.true_peak_xy) + distances[:, None] * direction
        values = interpolate(points[:, ::-1])
        profiles.append(PeakProfile(label, angle + offset, distances.copy(), points,
                                    values[:, 0], values[:, 1]))
    return tuple(profiles)
