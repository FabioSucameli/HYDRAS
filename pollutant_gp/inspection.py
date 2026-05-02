# Utilities for inspecting NetCDF files and finite-value domains.

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

# Return one slice of a variable and a readable description of the selected time.
def selected_time_slice(
    data_array: xr.DataArray,
    time_dim: str | None,
    requested_time_index: int | None,
) -> tuple[np.ndarray, str]:
    
    if time_dim is None or time_dim not in data_array.dims:
        return data_array.values, "no time dimension"

    n_times = data_array.sizes[time_dim]
    if requested_time_index is None:
        selected_index = 0
        selection_note = "default"
    else:
        selected_index = min(max(requested_time_index, 0), n_times - 1)
        selection_note = "requested"
        if selected_index != requested_time_index:
            selection_note = f"requested {requested_time_index} out of range"

    values = data_array.isel({time_dim: selected_index}).values
    return values, f"{time_dim} index {selected_index} ({selection_note})"

# Print dimensions, variables, and finite-domain patterns for all NetCDF files.
def inspect_netcdf_files(
    folder: Path,
    time_dim: str | None,
    time_index: int | None,
) -> None:
    
    files = sorted(folder.glob("*.nc"))
    if not files:
        print(f"No NetCDF files found in {folder}.")
        return

    print("=== NetCDF files ===")
    for file_path in files:
        print(f"- {file_path.name} ({file_path.stat().st_size / 1_000_000:.1f} MB)")

    finite_domains: dict[str, np.ndarray] = {}

    for file_path in files:
        print(f"\n=== {file_path.name} ===")
        ds = xr.open_dataset(file_path)
        try:
            print(f"Dimensions: {dict(ds.sizes)}")
            print("Coordinates:")
            for name, coord in ds.coords.items():
                print(f"  - {name}: dims={coord.dims}, shape={coord.shape}, dtype={coord.dtype}")

            print("Data variables:")
            domains_in_file: list[np.ndarray] = []
            for name, data_var in ds.data_vars.items():
                values, selected_time = selected_time_slice(data_var, time_dim, time_index)
                finite_domain = np.isfinite(values)
                domains_in_file.append(finite_domain)

                finite_values = values[finite_domain]
                if finite_values.size:
                    min_value = float(np.nanmin(finite_values))
                    max_value = float(np.nanmax(finite_values))
                    mean_value = float(np.nanmean(finite_values))
                else:
                    min_value = np.nan
                    max_value = np.nan
                    mean_value = np.nan

                print(
                    f"  - {name}: dims={data_var.dims}, shape={data_var.shape}, "
                    f"dtype={data_var.dtype}, selected={selected_time}, "
                    f"finite={int(finite_domain.sum())}, nan={int((~finite_domain).sum())}, "
                    f"min={min_value:.6g}, max={max_value:.6g}, mean={mean_value:.6g}"
                )

            if domains_in_file:
                finite_domains[file_path.name] = np.logical_or.reduce(domains_in_file)
        finally:
            ds.close()

    if len(finite_domains) > 1:
        first_name = next(iter(finite_domains))
        reference = finite_domains[first_name]
        print(f"\n=== Finite-domain comparison against {first_name} ===")
        for name, domain in finite_domains.items():
            if domain.shape != reference.shape:
                print(f"- {name}: different shape {domain.shape}, cannot compare")
                continue
            different = int(np.count_nonzero(reference != domain))
            print(
                f"- {name}: same={bool(np.array_equal(reference, domain))}, "
                f"different_cells={different}, valid_cells={int(domain.sum())}"
            )
