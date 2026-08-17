# Categorical colours for the covariance-structure comparison.
KERNEL_COMPARISON_STYLES = {
    "Isotropic": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "Axis-aligned anisotropic": {"color": "#E69F00", "marker": "s", "linestyle": "--"},
    "Wind-informed anisotropic": {"color": "#009E73", "marker": "^", "linestyle": "-."},
    "Current-informed anisotropic": {"color": "#CC79A7", "marker": "D", "linestyle": ":"},
}

COLOUR_PRIMARY = "#1f6fb2"  # posterior quantities: mean, conditional density
COLOUR_ACCENT = "#d55e00"  # observations and the source location
COLOUR_THIRD = "#009e73"  # third series, where one is needed
COLOUR_INK = "#222222"  # text and annotations
COLOUR_MUTED = "#8a8a8a"  # prior, marginal, reference levels

RAMP_ORDINAL = ("#7fb0d9", "#4a8ac4", "#2a639f", "#123f6b")

# Domain map: land and valid sea.
COLOUR_LAND = "white"
COLOUR_SEA = "#86cce3"


# Rendering presets
def apply_thesis_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


# Restore the matplotlib defaults used by the interactive and report figures of the project
def apply_screen_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)
