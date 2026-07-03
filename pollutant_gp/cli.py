# Command-line interface definition.

from __future__ import annotations

import argparse
from pathlib import Path

from pollutant_gp.data import (
    DEFAULT_CONCENTRATION_VARIABLE,
    DEFAULT_TIME_DIM,
    DEFAULT_X_COORDINATE,
    DEFAULT_X_DIM,
    DEFAULT_Y_COORDINATE,
    DEFAULT_Y_DIM,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stationary pollutant concentration field reconstruction with Gaussian Processes."
    )
    parser.add_argument(
        "--print-dataset",
        action="store_true",
        help="Print the full dataset structure, save the valid-domain map, and exit.",
    )
    parser.add_argument(
        "--plot-concentration-map",
        action="store_true",
        help="Save a standalone concentration map for the selected time index and exit.",
    )
    parser.add_argument(
        "--nc-file",
        type=Path,
        # TODO: Remove the default path and make this a required argument once the user is expected to provide their own data.
        default=Path("CMEMS_S1_01_conc_grid_10m.nc"),
        help="Path to the NetCDF file used for reconstruction.",
    )
    parser.add_argument(
        "--inspect-netcdf",
        action="store_true",
        help="Inspect all NetCDF files in --netcdf-dir and exit.",
    )
    parser.add_argument(
        "--netcdf-dir",
        type=Path,
        default=Path("."),
        help="Directory scanned when --inspect-netcdf is used.",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default=DEFAULT_CONCENTRATION_VARIABLE,
        help="Name of the concentration variable.",
    )
    parser.add_argument(
        "--time-dim",
        type=str,
        default=DEFAULT_TIME_DIM,
        help="Name of the time dimension. Use 'none' for a dataset without time.",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=None,
        help=(
            "Time index to reconstruct. In inspection mode, this is the time index used for variable statistics."
        ),
    )
    parser.add_argument(
        "--x-dim",
        type=str,
        default=DEFAULT_X_DIM,
        help="Name of the x/easting dimension.",
    )
    parser.add_argument(
        "--y-dim",
        type=str,
        default=DEFAULT_Y_DIM,
        help="Name of the y/northing dimension.",
    )
    parser.add_argument(
        "--x-coordinate",
        type=str,
        default=DEFAULT_X_COORDINATE,
        help="Name of the x coordinate variable. Use 'none' to use grid indices.",
    )
    parser.add_argument(
        "--y-coordinate",
        type=str,
        default=DEFAULT_Y_COORDINATE,
        help="Name of the y coordinate variable. Use 'none' to use grid indices.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of synthetic sensor measurements sampled from valid sea cells.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Standard deviation of additive Gaussian sensor noise.",
    )
    parser.add_argument(
        "--concentration-display-threshold",
        type=float,
        default=0.0,
        help=(
            "Values at or below this threshold are hidden in standalone concentration maps, "
            "leaving the marine-domain background visible."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=7,
        help="Random seed for reproducible sampling and noise.",
    )
    parser.add_argument(
        "--kernel-mode",
        choices=("anisotropic", "isotropic"),
        default="anisotropic",
        help="Use one length scale per axis or a single shared length scale.",
    )
    parser.add_argument(
        "--length-scale-lower-bound",
        type=float,
        default=0.05,
        help="Lower bound for GP RBF length scales in standardized coordinates.",
    )
    parser.add_argument(
        "--length-scale-upper-bound",
        type=float,
        default=100.0,
        help="Upper bound for GP RBF length scales in standardized coordinates.",
    )
    parser.add_argument(
        "--noise-level-initial",
        type=float,
        default=1e-4,
        help="Initial value for the WhiteKernel noise level.",
    )
    parser.add_argument(
        "--noise-level-lower-bound",
        type=float,
        default=1e-6,
        help="Lower bound for the WhiteKernel noise level.",
    )
    parser.add_argument(
        "--noise-level-upper-bound",
        type=float,
        default=1e1,
        help="Upper bound for the WhiteKernel noise level.",
    )
    parser.add_argument(
        "--target-transform",
        choices=("none", "log1p"),
        default="none",
        help="Optional target transform before fitting the GP.",
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=0,
        help="Number of Gaussian Process optimizer restarts.",
    )
    parser.add_argument(
        "--prediction-batch-size",
        type=int,
        default=20000,
        help="Number of grid points predicted per batch.",
    )
    parser.add_argument(
        "--clip-negative",
        dest="clip_negative",
        action="store_true",
        default=True,
        help="Clip negative mean predictions to zero. This is enabled by default.",
    )
    parser.add_argument(
        "--allow-negative-predictions",
        dest="clip_negative",
        action="store_false",
        help="Keep negative GP mean predictions instead of clipping them.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where figures are saved.",
    )
    parser.add_argument(
        "--figure-name",
        type=str,
        default=None,
        help=(
            "Name of the output reconstruction figure. Default: dataset, time index, sample count."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively in addition to saving it.",
    )
    parser.add_argument(
        "--sample-size-study",
        action="store_true",
        help="Run the GP for multiple sample counts and plot metrics vs n_samples.",
    )
    parser.add_argument(
        "--sample-size-study-counts",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100, 200, 400, 800],
        help=(
            "List of sample counts used in the sample size study. "
            "Example: --sample-size-study-counts 10 50 100 200"
        ),
    )
    parser.add_argument(
        "--sample-size-study-multiseed",
        action="store_true",
        help="Run the sample size study over multiple random seeds and plot mean +/- 1 std.",
    )
    parser.add_argument(
        "--sample-size-study-seeds",
        type=int,
        nargs="+",
        default=[7, 42, 123, 256, 512],
        help=(
            "List of random seeds used in the multi-seed sample size study. "
            "Example: --sample-size-study-seeds 7 42 123"
        ),
    )
    return parser.parse_args()
