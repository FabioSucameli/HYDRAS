# Synthetic sensor sampling.

from __future__ import annotations

import numpy as np

from pollutant_gp.types import GridData

# Sample synthetic sensor measurements from valid grid cells.
def sample_sensor_points(
    grid_data: GridData,
    n_samples: int,
    noise_std: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    valid_flat_indices = np.flatnonzero(grid_data.valid_mask.ravel())
    if n_samples > valid_flat_indices.size:
        raise ValueError(
            f"Requested {n_samples} samples, but only {valid_flat_indices.size} valid cells exist."
        )

    rng = np.random.default_rng(random_seed)
    sampled_flat_indices = rng.choice(valid_flat_indices, size=n_samples, replace=False)

    x_flat = grid_data.x_grid.ravel()
    y_flat = grid_data.y_grid.ravel()
    field_flat = grid_data.field.ravel()

    sample_coordinates = np.column_stack(
        [x_flat[sampled_flat_indices], y_flat[sampled_flat_indices]]
    )
    sample_values = field_flat[sampled_flat_indices].astype(float)

    if noise_std > 0.0:
        sample_values = sample_values + rng.normal(0.0, noise_std, size=n_samples)

    return sample_coordinates, sample_values, sampled_flat_indices

