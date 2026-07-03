# NetCDF loading and grid preparation for HYDRAS dataset.
# The HYDRAS concentration dataset provides pollutant concentration fields defined on a structured grid with dimensions (time, y, x).
# Land cells = NaN values. Valid marine cells = finite-value masking.

from __future__ import annotations

import numpy as np
import xarray as xr

from pollutant_gp.types import GridData

# Default names for HYDRAS dataset 
DEFAULT_CONCENTRATION_VARIABLE = "Concentration - component 1"
DEFAULT_TIME_DIM = "time"
DEFAULT_X_DIM = "x"
DEFAULT_Y_DIM = "y"
DEFAULT_X_COORDINATE = "x"
DEFAULT_Y_COORDINATE = "y"


def format_time_label(value) -> str:
    value_array = np.asarray(value)
    if np.issubdtype(value_array.dtype, np.datetime64):
        return np.datetime_as_string(value_array, unit="s").replace("T", " ")
    return str(value)


# Print a compact description of the NetCDF dataset.
def print_dataset_structure(ds: xr.Dataset) -> None:
    print("\n=== Dataset structure ===")
    print(ds)
    print("\n=== Coordinates ===")
    for name, coord in ds.coords.items():
        print(f"- {name}: dims={coord.dims}, shape={coord.shape}, dtype={coord.dtype}")
    print("\n=== Data variables ===")
    for name, data_var in ds.data_vars.items():
        print(f"- {name}: dims={data_var.dims}, shape={data_var.shape}, dtype={data_var.dtype}")

# Check that the dataset contains the required variables and dimensions.
def validate_dataset_layout(
    ds: xr.Dataset,
    variable_name: str,
    time_dim: str | None,
    y_dim: str,
    x_dim: str,
    y_coordinate: str | None,
    x_coordinate: str | None,
) -> None:
    
    if variable_name not in ds.data_vars:
        raise ValueError(f"Concentration variable '{variable_name}' was not found in the dataset.")

    data_array = ds[variable_name]
    required_dims = [y_dim, x_dim]
    if time_dim is not None:
        required_dims.append(time_dim)

    missing_dims = [dim for dim in required_dims if dim not in data_array.dims]
    if missing_dims:
        raise ValueError(
            f"Variable '{variable_name}' does not contain dimensions {missing_dims}. "
            f"Available dimensions are {data_array.dims}."
        )

    if x_coordinate is not None and x_coordinate not in ds.variables:
        raise ValueError(f"x coordinate '{x_coordinate}' was not found in the dataset.")
    if y_coordinate is not None and y_coordinate not in ds.variables:
        raise ValueError(f"y coordinate '{y_coordinate}' was not found in the dataset.")

# Select the time index to reconstruct.
# When no time index is specified, a representative snapshot is selected by evaluating a set of candidate time steps 
# and choosing the one with the largest concentration range.
def choose_time_index(
    data_array: xr.DataArray,
    time_dim: str | None,
    requested_index: int | None,
) -> int | None:
    
    if time_dim is None:
        return None

    n_times = data_array.sizes[time_dim]
    if requested_index is not None:
        if requested_index < 0 or requested_index >= n_times:
            raise ValueError(f"--time-index must be in [0, {n_times - 1}], got {requested_index}.")
        return requested_index

    # [0, 10, 100, 705, 1410, 2115, 2820] with n_times = 2821
    candidate_indices = sorted(
        {
            0,
            min(10, n_times - 1),
            min(100, n_times - 1),
            n_times // 4,
            n_times // 2,
            (3 * n_times) // 4,
            n_times - 1,
        }
    )
    best_index = candidate_indices[0]
    best_range = -np.inf

    for index in candidate_indices:
        values = data_array.isel({time_dim: index}).values
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            continue
        value_range = float(np.nanmax(finite_values) - np.nanmin(finite_values))
        if value_range > best_range:
            best_index = index
            best_range = value_range

    return best_index

# Create 2D coordinate grids matching the concentration field shape.
def build_coordinate_grids(
    ds: xr.Dataset,
    y_coordinate: str | None,
    x_coordinate: str | None,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    
    x_values = np.arange(shape[1], dtype=float) if x_coordinate is None else ds[x_coordinate].values
    y_values = np.arange(shape[0], dtype=float) if y_coordinate is None else ds[y_coordinate].values

    if x_values.ndim == 1 and y_values.ndim == 1:
        x_grid, y_grid = np.meshgrid(x_values, y_values)
    elif x_values.ndim == 2 and y_values.ndim == 2:
        x_grid, y_grid = x_values, y_values
    else:
        raise ValueError("Spatial coordinates must either both be 1D or both be 2D.")

    if x_grid.shape != shape or y_grid.shape != shape:
        raise ValueError(
            f"Coordinate grid shape mismatch. Field shape is {shape}, "
            f"x grid is {x_grid.shape}, y grid is {y_grid.shape}."
        )

    return x_grid.astype(float), y_grid.astype(float)

# Extract a 2D field, coordinates, and valid cells.
def prepare_grid_data(
    ds: xr.Dataset,
    variable_name: str,
    time_dim: str | None,
    time_index: int | None,
    y_dim: str,
    x_dim: str,
    y_coordinate: str | None,
    x_coordinate: str | None,
) -> GridData:
    
    data_array = ds[variable_name]
    selected_time_label = None

    if time_dim is not None:
        data_array = data_array.isel({time_dim: time_index})
        time_coord = ds.coords.get(time_dim)
        selected_time_label = (
            format_time_label(time_coord.isel({time_dim: time_index}).values)
            if time_coord is not None
            else str(time_index)
        )

    data_array = data_array.transpose(y_dim, x_dim)
    field = data_array.values.astype(float)
    valid_mask = np.isfinite(field)

    x_grid, y_grid = build_coordinate_grids(
        ds=ds,
        y_coordinate=y_coordinate,
        x_coordinate=x_coordinate,
        shape=field.shape,
    )

    return GridData(
        field=field,
        valid_mask=valid_mask,
        x_grid=x_grid,
        y_grid=y_grid,
        x_dim=x_dim,
        y_dim=y_dim,
        selected_time_label=selected_time_label,
    )
