# HYDRAS Thesis Project

**Reconstruction and localisation of marine pollutant sources using Gaussian Processes and underwater robots**

This repository contains the Python code, the working report and the thesis sources for a Master's
project on probabilistic reconstruction of marine pollutant concentration fields. The long-term goal
is to estimate the spatial distribution of a pollutant from sparse measurements collected by
underwater robots, and to use the reconstructed field to support informative exploration and source
localisation.



---

## Table of Contents

- [Project Goal](#project-goal)
- [How It Works](#how-it-works)
- [Thesis Roadmap](#thesis-roadmap)
- [Current Status](#current-status)
- [Key Result](#key-result)
- [Results Snapshot](#results-snapshot)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Command-Line Arguments](#command-line-arguments)
- [Outputs](#outputs)

---

## Project Goal

In realistic marine monitoring scenarios, pollutant concentration is not available continuously over
the full domain. It can only be observed at a limited number of positions, or along trajectories
followed by autonomous underwater robots.

This project studies how **Gaussian Processes (GPs)** can be used to:

1. reconstruct a pollutant concentration field from sparse measurements;
2. quantify predictive uncertainty over the marine domain;
3. evaluate how many measurements are needed for reliable reconstruction;
4. encode physical knowledge about pollutant transport into the covariance function;
5. guide robot motion and localise the pollutant source.

Gaussian Processes are useful here because they return both a mean prediction and a predictive
uncertainty. That makes them suitable not only for interpolation, but also for informative path
planning, since the predictive variance depends only on *where* measurements were taken and not on
*what* they returned — so a candidate measurement can be scored before it is made.

---

## How It Works

The baseline pipeline turns one snapshot of a simulated field into a probabilistic reconstruction:

```mermaid
flowchart LR
    A["NetCDF dataset<br/>Concentration - component 1"] --> B["Select time snapshot"]
    B --> C["Valid marine mask<br/>np.isfinite(field)"]
    C --> D["Sample N synthetic sensors<br/>without replacement"]
    D --> E["Standardise coordinates<br/>StandardScaler"]
    E --> F["Fit GP<br/>ConstantKernel x RBF + WhiteKernel"]
    F --> G["Predict over all valid cells<br/>in batches"]
    G --> H["Clip negatives<br/>optional"]
    H --> I["Mean map, uncertainty map,<br/>RMSE / MAE / R2"]
```

The physically informed variant changes **only the coordinate system in which the covariance is
defined**. The field, the grid and the output maps are untouched:

```mermaid
flowchart LR
    W["Wind file"] --> M["Mean transport direction<br/>over a time window"]
    U["Current file<br/>U/V NetCDF"] --> M
    M --> T["Rotation angle theta"]
    T --> R["Rotate sensor and prediction<br/>coordinates about the domain centre"]
    R --> K["Fit the anisotropic RBF<br/>in the rotated frame"]
    K --> P["Predict, then report the maps<br/>in the original coordinates"]
```

A key methodological distinction is that the orientation angle theta is derived from independently measured physical forcing, making the orientation part of the model  

---

## Thesis Roadmap

```mermaid
flowchart TD
    P1["Phase 1 - Stationary reconstruction<br/>from sparse point samples"]
    P1B["Phase 1b - Physically informed covariance<br/>oriented by wind or current"]
    P2["Phase 2 - Measurements collected<br/>along robot trajectories"]
    P3["Phase 3 - Informative exploration<br/>driven by GP uncertainty"]
    P4["Phase 4 - Pollutant source localisation"]

    P1 --> P1B --> P2 --> P3 --> P4

    classDef phase fill:#86cce3,stroke:#4a8ac4,stroke-width:1px,color:#000000
    class P1,P1B,P2,P3,P4 phase
```



The package structure is intentionally modular so that trajectory sampling, informative planning and
source localisation can be added without rewriting the Phase 1 baseline.

---

## Current Status

Implemented:

- NetCDF dataset loading and layout validation with `xarray`;
- extraction of a single time snapshot and valid-domain detection from finite values;
- synthetic sensor sampling on valid marine cells, without replacement, with optional noise;
- GP fitting with isotropic or anisotropic RBF kernels, plus `ConstantKernel` and `WhiteKernel`;
- **coordinate rotation informed by an external forcing**, from a wind time series or from a
  current U/V NetCDF file, with a configurable averaging window;
- full-field reconstruction on the original grid, with batched prediction;
- predictive uncertainty maps and error metrics: MSE, RMSE, MAE, R2;
- single-seed and multi-seed sample-size studies;
- **multi-seed comparison of four covariance structures** — isotropic, axis-aligned anisotropic,
  wind-informed and current-informed — on identical shared samples;
- positivity diagnostics before clipping, and an optional `log1p` target transform;
- length-scale bound diagnostics, reporting how often an optimised length scale reaches its bound;
- **peak diagnostics** at a fixed sensor budget: prediction at the true maximum, localisation error
  and local RMSE inside a peak-centred disk, with a zoom and two directional profiles;
- **oracle sampling controls** — peak-observed and locally enriched — that redistribute sensors
  without changing their number, and are evaluated on cells observed by no configuration;
- **single versus two-scale kernel comparison**, with prescribed initialisations selected by
  marginal likelihood on the sensors alone;
- controlled optimiser diagnostics: initialisation study, nested restart study, and lower-bound,
  upper-bound and local sensitivity studies for the length scales;
- NetCDF inspection utilities.

---

## Key Result

Anisotropy alone buys nothing; anisotropy **with the right orientation** buys a great deal.

Averaged over five random seeds on the `CL02_V1_SRC131` scenario at time index 729, with identical
samples shared by all four models and a length-scale lower bound of 0.075:

| Sensors | Isotropic | Axis-aligned | Wind-informed | Current-informed |
|---:|---:|---:|---:|---:|
| 200 | 0.2582 | 0.2589 | 0.1998 | **0.1983** |
| 400 | 0.2411 | 0.2425 | 0.1702 | **0.1568** |
| 800 | 0.1977 | 0.1909 | 0.1439 | **0.1356** |
| 1200 | 0.1739 | 0.1739 | **0.1240** | 0.1243 |
| 1600 | 0.1635 | 0.1635 | 0.1271 | **0.1248** |

Mean RMSE on the valid marine cells; lower is better. The axis-aligned anisotropic model is
indistinguishable from the isotropic one, because an ellipse constrained to the grid axes cannot
follow a plume that runs north-west to south-east. The two physically informed models — whose
transport directions are estimated independently, from wind and from current, and agree to within
about two degrees — reduce the error by roughly a quarter.

---

## Results Snapshot

A small number of curated figures. Diagnostic panels are discussed in the report, not here.

### Valid Marine Domain

![Valid marine domain](outputs/valid_domain_map_t705.png)

The valid domain is extracted from finite concentration values. Sampling, reconstruction and metric
computation are all restricted to these cells.

### Stationary Reconstruction Example

![Phase 1 reconstruction overview with 800 samples](outputs/github_t2820_n800_overview.png)

Reconstruction at time index `2820` with 800 synthetic sensors. The four panels show the ground
truth, the GP mean, the predictive standard deviation and the absolute error. This run reaches
`RMSE = 1.89` and `R2 = 0.60` on valid marine cells.

### Multi-Seed Sample-Size Study

![Multi-seed RMSE study](outputs/case_2820_multiseed_rmse.png)

The mean trend improves with the number of samples, while the spread across seeds shows that
*where* the sensors fall matters as much as how many there are — especially for localised plumes.

### Kernel Comparison

![Kernel comparison across seeds](outputs/cl02_wind_current_kernel_comparison_lb0075.png)

The four covariance structures on shared samples. The isotropic and axis-aligned curves overlap; the
two physically informed curves separate clearly from them.

---

## Recovering the Peak

Controlled experiments on `CL02_V1_SRC131` (time index 729, 800 sensors) show how local sampling
and a flexible current-informed GP can improve both peak recovery and field reconstruction.

`--peak-diagnostics` complements global metrics with peak height, location and error within a
150 m disk. These diagnostics reveal local details that a low global RMSE can hide.

### Improving Local Coverage

![Oracle sampling control at seed 123](outputs/cl02_seed123_sampling_zoom.png)

`--peak-sampling-study` compares uniform, peak-observed and locally enriched layouts at a
**fixed budget of 800 sensors**. Moving 160 sensors into the disk reduces local RMSE on common
unobserved cells from **2.94 to 0.50** for seed 123 and **0.47 to 0.39** for seed 7.
This improves the neighbourhood reconstruction, while motivating a closer look at peak height.

### Recovering the Peak with Finer Scales

`--peak-kernel-study` compares single and two-scale RBF kernels, selecting the best converged fit
from two prescribed initialisations by sensor marginal likelihood. With enriched sampling,
lowering the length-scale bound from 0.05 to 0.02 substantially improves peak recovery:

| Seed 123, enriched | Lower bound | Prediction at the true peak | Local unseen RMSE |
|---|---:|---:|---:|
| Single RBF | 0.05 | 2.975 | 0.448 |
| Single RBF | 0.02 | **7.280** | 0.269 |
| Two RBFs | 0.05 | 3.161 | 0.444 |
| Two RBFs | 0.02 | **7.381** | 0.278 |

The two-scale model recovers **96.5% of the true peak** (7.647) for seed 123 and **81.2%** for
seed 7. At the same bound of 0.02, it also reduces global RMSE on common unobserved cells by
approximately **29% and 34%**, respectively, compared with the single RBF.

![Single and two-scale kernels at lower bound 0.02](outputs/cl02_peak_kernel_seed123_lb002_zoom.png)

Ground truth, single RBF and two RBFs at lower bound 0.02. Rows show uniform and enriched sampling,
with a shared colour scale.

These results support combining local coverage with finer admissible scales and a two-scale
covariance. They concern two seeds and one snapshot: enrichment uses the known true peak
(*oracle* information), and a smaller bound does not improve every sampling configuration.
They demonstrate improved reconstruction, not autonomous source localisation.

---


## Repository Structure

```text
main.py                     Entry point: parses arguments and runs the workflow
pollutant_gp/
  cli.py                    Command-line interface
  data.py                   NetCDF loading, validation and grid preparation
  inspection.py             NetCDF inspection utilities
  model.py                  Kernel construction, GP fitting, batched prediction
  reconstruction.py         Full-field reconstruction and error metrics
  sampling.py               Synthetic sensor sampling
  spatial.py                Coordinate rotation used by the physically informed models
  wind.py                   Wind time series parsing and transport direction
  current.py                Current U/V fields and mean transport direction
  peak.py                   Peak-centred metrics and the diagnostic disk
  peak_sampling.py          Oracle sampling controls at a fixed sensor budget
  peak_kernel.py            Single versus two-scale kernel comparison
  peak_visualization.py     Peak zooms and directional profiles
  types.py                  Shared dataclasses
  style.py                  Shared palette and figure styling
  visualization.py          Plot generation
  workflow.py               Pipeline orchestration and experimental studies
outputs/                    Selected figures

requirements.txt
README.md
```

Local environments, Python caches, run-time outputs, LaTeX build artefacts and NetCDF datasets are
not tracked by Git.

---

## Installation

```bash
pip install -r requirements.txt
```

Recommended Python version: 3.10 or newer. Dependencies: `numpy`, `scipy`, `xarray`, `netCDF4`,
`scikit-learn`, `matplotlib`, `pillow`.

---

## Data

The project uses NetCDF simulation files produced with the MIKE 21 Flow Model FM for the waters
around the port of Cecina. Concentration files use:

```text
variable:    Concentration - component 1
dimensions:  time, y, x
coordinates: time, y, x
```

Velocity files carry `u_velocity` and `v_velocity` on the same grid and are used by the
current-informed model. Land and excluded cells are stored as `NaN`, so the navigable domain is
recovered from the data themselves:

```python
valid_mask = np.isfinite(field)
```

Large NetCDF files are ignored by Git. Keep them locally and pass their path with `--nc-file`.

---

## Usage

All commands are run from the repository root.

### Inspect the data

```bash
python main.py --print-dataset --time-index 705
python main.py --inspect-netcdf --netcdf-dir .
```

### Standard reconstruction

```bash
python main.py --nc-file CL02_V1_SRC131_Conc_10mGrid.nc --time-index 729 --n-samples 200
```

### Physically informed reconstruction

Rotate the covariance along the wind-derived transport direction:

```bash
python main.py --nc-file CL02_V1_SRC131_Conc_10mGrid.nc --time-index 729 \
  --n-samples 800 --physically-informed
```

Or along the mean sea current, averaged over the twelve hours preceding the snapshot:

```bash
python main.py --nc-file CL02_V1_SRC131_Conc_10mGrid.nc --time-index 729 \
  --n-samples 800 --current-informed --current-average-hours 12
```

### Compare the four covariance structures

All models are evaluated on identical samples for every `(seed, N)` pair, so differences are
attributable to the model and not to the data:

```bash
python main.py --nc-file CL02_V1_SRC131_Conc_10mGrid.nc --time-index 729 \
  --kernel-comparison-study \
  --sample-size-study-counts 200 400 800 1200 1600 \
  --sample-size-study-seeds 7 42 123 256 512 \
  --length-scale-lower-bound 0.075
```

### Sample-size studies

```bash
python main.py --time-index 2820 --sample-size-study-multiseed \
  --sample-size-study-counts 10 25 50 100 200 400 800 \
  --sample-size-study-seeds 7 42 123 256 512
```

### Positivity of the predictions

```bash
python main.py --time-index 729 --target-transform log1p
python main.py --time-index 729 --allow-negative-predictions
```


### Peak diagnostics and kernel study

Metrics and figures centred on the true maximum, after a single reconstruction:

```bash
python main.py --nc-file CL02_V1_SRC131_Conc_10mGrid.nc --time-index 729 \
  --n-samples 800 --current-informed --random-seed 123 --n-restarts 0 \
  --peak-diagnostics --peak-radius 150 --peak-vmax 11 --peak-coordinate-unit m
```

Uniform, peak-observed and locally enriched sensor layouts:

```bash
python main.py --nc-file CL02_V1_SRC131_Conc_10mGrid.nc --time-index 729 \
  --n-samples 800 --current-informed --random-seed 123 --n-restarts 0 \
  --peak-sampling-study --peak-radius 150 --peak-local-replacements 160 \
  --peak-vmax 11 --peak-coordinate-unit m
```

Single against two-scale kernel:

```bash
python main.py --nc-file CL02_V1_SRC131_Conc_10mGrid.nc --time-index 729 \
  --n-samples 800 --current-informed --random-seed 123 --n-restarts 0 \
  --peak-kernel-study --length-scale-lower-bound 0.02 \
  --peak-radius 150 --peak-local-replacements 160 \
  --peak-vmax 11 --peak-coordinate-unit m
```

---

## Command-Line Arguments

| Argument | Default | Description |
|---|---:|---|
| `--nc-file` | `CMEMS_S1_01_conc_grid_10m.nc` | NetCDF file used for reconstruction |
| `--time-index` | auto | Time index to reconstruct or inspect |
| `--n-samples` | `200` | Number of synthetic sensors |
| `--noise-std` | `0.0` | Standard deviation of the simulated sensor noise |
| `--random-seed` | `7` | Seed for the main reconstruction |
| `--kernel-mode` | `anisotropic` | One length scale per axis, or a single shared one |
| `--physically-informed` | off | Rotate coordinates along a physical transport direction |
| `--physics-source` | `wind` | Forcing used for the rotation: `wind` or `current` |
| `--current-informed` | off | Shortcut for `--physically-informed --physics-source current` |
| `--wind-file` | `CI_WIND_faseII_V1.txt` | Wind forcing time series |
| `--wind-average-hours` | `12.0` | Averaging window preceding the snapshot |
| `--current-file` | `CL02_V1_SRC000_U_V_10mGrid.nc` | Current U/V dataset |
| `--current-average-hours` | `12.0` | Averaging window for the mean current |
| `--length-scale-lower-bound` | `0.05` | Lower bound in standardised coordinates; acts as regularisation |
| `--length-scale-upper-bound` | `100.0` | Upper bound in standardised coordinates |
| `--target-transform` | `none` | `none` or `log1p` |
| `--n-restarts` | `0` | Additional optimiser restarts |
| `--clip-negative` | on | Clip negative mean predictions to zero |
| `--allow-negative-predictions` | off | Keep negative predictions instead of clipping |
| `--prediction-batch-size` | `20000` | Grid points predicted per batch |
| `--sample-size-study` | off | Single-seed sample-size study |
| `--sample-size-study-multiseed` | off | Multi-seed sample-size study |
| `--kernel-comparison-study` | off | Multi-seed comparison of the four covariance structures |
| `--sample-size-study-counts` | `10 25 50 100 200 400 800` | Sample counts used by the studies |
| `--sample-size-study-seeds` | `7 42 123 256 512` | Seeds used by the multi-seed studies |
| `--peak-diagnostics` | off | Peak metrics, zoom and directional profiles after one reconstruction |
| `--peak-sampling-study` | off | Uniform, peak-observed and locally enriched oracle layouts at a fixed budget |
| `--peak-kernel-study` | off | Single against two-scale current-informed kernel on the same sensors |
| `--peak-radius` | `150.0` | Radius of the peak-centred diagnostic disk, in grid units |
| `--peak-local-replacements` | `160` | Outside sensors moved into the disk by the enriched control |
| `--peak-vmax` | auto | Common colour maximum for peak zooms, so runs stay comparable |
| `--peak-coordinate-unit` | auto | Unit label for the peak figures; no conversion is applied |
| `--optimizer-seed` | sampling seed | Seed used only for optimiser restarts |
| `--optimizer-initialization-study` | off | Four deterministic kernel initialisations on one shared sample |
| `--optimizer-restart-study` | off | Nested restart study, evaluating every internal optimiser run |
| `--length-scale-lower-bound-study` | off | Controlled sensitivity study over length-scale lower bounds |
| `--length-scale-upper-bound-study` | off | The same for upper bounds, on two initialisations |
| `--length-scale-local-sensitivity-study` | off | One-at-a-time perturbations around the short-scale solution |
| `--output-dir` | `outputs` | Directory where figures are saved |
| `--figure-name` | auto | Custom name for the main figure |
| `--show` | off | Show figures interactively as well as saving them |

For the complete list:

```bash
python main.py --help
```

---



## Outputs



A reconstruction produces a combined overview plus the same four panels saved separately, so that a
single map can be reused in a report or a slide without cropping:

| File | Content |
|---|---|
| `<base>.png` | four-panel overview |
| `<base>_ground_truth.png` | simulated concentration field |
| `<base>_gp_reconstruction.png` | GP posterior mean |
| `<base>_predictive_uncertainty.png` | predictive standard deviation |
| `<base>_absolute_error.png` | absolute error against the ground truth |

`<base>` is built from dataset name, time index, sample count and timestamp, and can be replaced
with `--figure-name`.

The studies write their own curves. The sample-size studies plot RMSE, MAE and R2 against the number
of sensors, with one faint line per seed and the mean in bold in the multi-seed version;
`--kernel-comparison-study` plots the same metrics with one curve per covariance structure.

Results are saved locally in the `outputs/` directory.


