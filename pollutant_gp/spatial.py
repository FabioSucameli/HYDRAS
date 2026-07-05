# Spatial coordinate transformations used before Gaussian Process fitting.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pollutant_gp.types import GridData


@dataclass(frozen=True)
class RotationTransform:
    # Rotate coordinates so the first axis follows a physically meaningful direction.

    angle_degrees: float
    center_x: float
    center_y: float
    description: str

    # Apply the configured rotation to an array of spatial coordinates.
    def transform(self, coordinates: np.ndarray) -> np.ndarray:
        coordinates = np.asarray(coordinates, dtype=float)
        centered_x = coordinates[:, 0] - self.center_x
        centered_y = coordinates[:, 1] - self.center_y

        angle_radians = np.deg2rad(self.angle_degrees)
        cos_angle = np.cos(angle_radians)
        sin_angle = np.sin(angle_radians)

        rotated_x = cos_angle * centered_x + sin_angle * centered_y
        rotated_y = -sin_angle * centered_x + cos_angle * centered_y
        return np.column_stack([rotated_x, rotated_y])


# Create a rotation transform centered on the valid marine domain.
def build_rotation_transform(
    grid_data: GridData,
    angle_degrees: float,
    description: str,
) -> RotationTransform:
    valid_x = grid_data.x_grid[grid_data.valid_mask]
    valid_y = grid_data.y_grid[grid_data.valid_mask]

    return RotationTransform(
        angle_degrees=angle_degrees,
        center_x=float(np.mean(valid_x)),
        center_y=float(np.mean(valid_y)),
        description=description,
    )


# Apply a coordinate transform only when one is configured.
def maybe_transform_coordinates(
    coordinates: np.ndarray,
    transform: RotationTransform | None,
) -> np.ndarray:
    if transform is None:
        return coordinates
    return transform.transform(coordinates)
