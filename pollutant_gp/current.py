# Current-field utilities for physically informed coordinate rotations.

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class CurrentOrientation:
    source_path: Path
    target_time: datetime
    average_hours: float
    selected_time_count: int
    valid_vector_count: int
    mean_u: float
    mean_v: float
    vector_speed: float
    direction_toward_degrees: float
    math_angle_degrees: float


# Convert an east/north vector into current-direction and mathematical angle conventions.
def vector_to_current_orientation(
    east: float,
    north: float,
) -> tuple[float, float, float]:
    vector_speed = math.hypot(east, north)
    if vector_speed <= 0.0:
        raise ValueError("The current vector has zero magnitude and cannot define an orientation.")

    math_angle_degrees = math.degrees(math.atan2(north, east))
    direction_toward_degrees = (90.0 - math_angle_degrees) % 360.0
    return vector_speed, direction_toward_degrees, math_angle_degrees


# Convert a Python datetime into a NumPy datetime64 value for xarray selection.
def to_datetime64(value: datetime) -> np.datetime64:
    return np.datetime64(value)


# Validate current variables and the requested time coordinate.
def validate_current_dataset(
    ds: xr.Dataset,
    u_variable: str,
    v_variable: str,
    time_dim: str,
) -> None:
    if u_variable not in ds.data_vars:
        raise ValueError(f"Current u variable '{u_variable}' was not found in the dataset.")
    if v_variable not in ds.data_vars:
        raise ValueError(f"Current v variable '{v_variable}' was not found in the dataset.")
    if time_dim not in ds.coords:
        raise ValueError(f"Current time coordinate '{time_dim}' was not found in the dataset.")

    if ds[u_variable].dims != ds[v_variable].dims:
        raise ValueError(
            f"Current variables must share the same dimensions, got "
            f"{ds[u_variable].dims} and {ds[v_variable].dims}."
        )
    if time_dim not in ds[u_variable].dims:
        raise ValueError(f"Current variable '{u_variable}' does not contain time dimension '{time_dim}'.")


# Select the current field at the target time or over the preceding averaging window.
def select_current_window(
    ds: xr.Dataset,
    target_time: datetime,
    average_hours: float,
    time_dim: str,
) -> xr.Dataset:
    time_values = ds[time_dim].values
    if time_values.size == 0:
        raise ValueError("The current dataset has an empty time coordinate.")

    target_time64 = to_datetime64(target_time)
    first_time = time_values[0]
    last_time = time_values[-1]

    if average_hours <= 0.0:
        if target_time64 < first_time or target_time64 > last_time:
            raise ValueError(
                f"Target time {target_time} is outside the current file range "
                f"[{first_time}, {last_time}]."
            )
        return ds.sel({time_dim: target_time64}, method="nearest")

    start_time64 = to_datetime64(target_time - timedelta(hours=average_hours))
    if start_time64 < first_time or target_time64 > last_time:
        raise ValueError(
            f"Current averaging window [{start_time64}, {target_time64}] is outside "
            f"the current file range [{first_time}, {last_time}]."
        )

    selected = ds.sel({time_dim: slice(start_time64, target_time64)})
    if selected.sizes.get(time_dim, 0) == 0:
        raise ValueError(
            f"No current records were found in the averaging window "
            f"[{start_time64}, {target_time64}]."
        )
    return selected


# Compute the spatial or space-time mean current vector over the valid marine domain.
def compute_mean_current_vector(
    selected: xr.Dataset,
    u_variable: str,
    v_variable: str,
    time_dim: str,
    valid_mask: np.ndarray,
) -> tuple[float, float, int, int]:
    u_values = selected[u_variable].values.astype(float)
    v_values = selected[v_variable].values.astype(float)

    if u_values.shape[-2:] != valid_mask.shape:
        raise ValueError(
            f"Current grid shape {u_values.shape[-2:]} does not match "
            f"concentration grid shape {valid_mask.shape}."
        )

    if time_dim not in selected[u_variable].dims:
        u_values = u_values[np.newaxis, ...]
        v_values = v_values[np.newaxis, ...]

    marine_mask = valid_mask[np.newaxis, :, :]
    finite_mask = np.isfinite(u_values) & np.isfinite(v_values) & marine_mask
    valid_vector_count = int(np.count_nonzero(finite_mask))
    if valid_vector_count == 0:
        raise ValueError("No valid current vectors were found on the marine domain.")

    selected_time_count = int(selected.sizes.get(time_dim, 1))
    mean_u = float(np.mean(u_values[finite_mask]))
    mean_v = float(np.mean(v_values[finite_mask]))
    return mean_u, mean_v, selected_time_count, valid_vector_count


# Compute the current-informed orientation used to rotate GP coordinates.
def compute_current_orientation(
    path: Path,
    target_time: datetime,
    average_hours: float,
    u_variable: str,
    v_variable: str,
    time_dim: str,
    valid_mask: np.ndarray,
) -> CurrentOrientation:
    if not path.exists():
        raise FileNotFoundError(f"Current file not found: {path}")

    with xr.open_dataset(path) as ds:
        validate_current_dataset(
            ds=ds,
            u_variable=u_variable,
            v_variable=v_variable,
            time_dim=time_dim,
        )
        selected = select_current_window(
            ds=ds,
            target_time=target_time,
            average_hours=average_hours,
            time_dim=time_dim,
        )
        mean_u, mean_v, selected_time_count, valid_vector_count = compute_mean_current_vector(
            selected=selected,
            u_variable=u_variable,
            v_variable=v_variable,
            time_dim=time_dim,
            valid_mask=valid_mask,
        )

    vector_speed, direction_toward, math_angle = vector_to_current_orientation(mean_u, mean_v)
    return CurrentOrientation(
        source_path=path,
        target_time=target_time,
        average_hours=average_hours,
        selected_time_count=selected_time_count,
        valid_vector_count=valid_vector_count,
        mean_u=mean_u,
        mean_v=mean_v,
        vector_speed=vector_speed,
        direction_toward_degrees=direction_toward,
        math_angle_degrees=math_angle,
    )
