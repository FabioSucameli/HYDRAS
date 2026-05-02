# Shared data structures.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Prepared 2D field, coordinates, and valid cells.
@dataclass(frozen=True)
class GridData:

    field: np.ndarray
    valid_mask: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    x_dim: str
    y_dim: str
    selected_time_label: str | None

# Predicted fields and scalar reconstruction metrics.
@dataclass(frozen=True)
class ReconstructionResult:

    mean_field: np.ndarray
    std_field: np.ndarray
    mse: float
    rmse: float
    mae: float
    r2: float

