# Wind forcing utilities for physically informed coordinate rotations.

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WindOrientation:
    source_path: Path
    target_time: datetime
    average_hours: float
    vector_speed: float
    direction_from_degrees: float
    direction_toward_degrees: float
    math_angle_degrees: float


# Convert a dataset time label into a Python datetime.
def parse_time_label(time_label: str | None) -> datetime:
    if time_label is None:
        raise ValueError("A selected time is required when --physically-informed is used.")
    return datetime.fromisoformat(time_label.replace("T", " "))


# Read wind timestamps, speeds, and directions from a tab-separated text file.
def read_wind_file(path: Path) -> tuple[list[datetime], np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Wind file not found: {path}")

    times: list[datetime] = []
    speeds: list[float] = []
    directions: list[float] = []

    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.reader(file_obj, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            first = row[0].strip()
            if first in {"", "Time", "Unit"}:
                continue
            try:
                times.append(datetime.fromisoformat(first))
                speeds.append(float(row[1]))
                directions.append(float(row[2]))
            except ValueError:
                continue

    if not times:
        raise ValueError(f"No wind records were found in {path}")

    order = np.argsort([time.timestamp() for time in times])
    sorted_times = [times[index] for index in order]
    sorted_speeds = np.asarray(speeds, dtype=float)[order]
    sorted_directions = np.asarray(directions, dtype=float)[order]
    return sorted_times, sorted_speeds, sorted_directions


# Convert wind speed and direction into east/north vector components.
def wind_vectors(
    speeds: np.ndarray,
    directions_degrees: np.ndarray,
    direction_convention: str,
) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(directions_degrees)

    if direction_convention == "from":
        east = -speeds * np.sin(radians)
        north = -speeds * np.cos(radians)
    elif direction_convention == "toward":
        east = speeds * np.sin(radians)
        north = speeds * np.cos(radians)
    else:
        raise ValueError(f"Unknown wind direction convention: {direction_convention}")

    return east, north


# Interpolate a time series at the requested target time.
def interpolate_series(
    times: list[datetime],
    values: np.ndarray,
    target_time: datetime,
) -> float:
    seconds = np.asarray([time.timestamp() for time in times], dtype=float)
    target_seconds = target_time.timestamp()

    if target_seconds < seconds[0] or target_seconds > seconds[-1]:
        raise ValueError(
            f"Target time {target_time} is outside the wind file range "
            f"[{times[0]}, {times[-1]}]."
        )

    return float(np.interp(target_seconds, seconds, values))


# Compute a time-weighted mean component over a selected time window.
def time_weighted_component_mean(
    times: list[datetime],
    values: np.ndarray,
    start_time: datetime,
    end_time: datetime,
) -> float:
    internal_times = [time for time in times if start_time < time < end_time]
    breakpoints = [start_time, *internal_times, end_time]

    weighted_sum = 0.0
    total_seconds = 0.0
    for left, right in zip(breakpoints[:-1], breakpoints[1:]):
        left_value = interpolate_series(times, values, left)
        right_value = interpolate_series(times, values, right)
        duration_seconds = (right - left).total_seconds()
        weighted_sum += 0.5 * (left_value + right_value) * duration_seconds
        total_seconds += duration_seconds

    if total_seconds <= 0.0:
        raise ValueError("The wind averaging window must have positive duration.")

    return weighted_sum / total_seconds


# Convert an east/north vector into wind and mathematical angle conventions.
def vector_to_orientation(
    east: float,
    north: float,
) -> tuple[float, float, float, float]:
    vector_speed = math.hypot(east, north)
    if vector_speed <= 0.0:
        raise ValueError("The wind vector has zero magnitude and cannot define an orientation.")

    math_angle_degrees = math.degrees(math.atan2(north, east))
    direction_toward_degrees = (90.0 - math_angle_degrees) % 360.0
    direction_from_degrees = (direction_toward_degrees + 180.0) % 360.0
    return vector_speed, direction_from_degrees, direction_toward_degrees, math_angle_degrees


# Compute the wind-informed orientation used to rotate GP coordinates.
def compute_wind_orientation(
    path: Path,
    target_time: datetime,
    average_hours: float,
    direction_convention: str,
) -> WindOrientation:
    times, speeds, directions = read_wind_file(path)
    east_values, north_values = wind_vectors(
        speeds=speeds,
        directions_degrees=directions,
        direction_convention=direction_convention,
    )

    if average_hours <= 0.0:
        east = interpolate_series(times, east_values, target_time)
        north = interpolate_series(times, north_values, target_time)
    else:
        start_time = target_time - timedelta(hours=average_hours)
        east = time_weighted_component_mean(times, east_values, start_time, target_time)
        north = time_weighted_component_mean(times, north_values, start_time, target_time)

    vector_speed, direction_from, direction_toward, math_angle = vector_to_orientation(east, north)
    return WindOrientation(
        source_path=path,
        target_time=target_time,
        average_hours=average_hours,
        vector_speed=vector_speed,
        direction_from_degrees=direction_from,
        direction_toward_degrees=direction_toward,
        math_angle_degrees=math_angle,
    )
