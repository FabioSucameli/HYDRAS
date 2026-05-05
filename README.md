# HYDRAS
This repository contains Python code developed for a thesis project focused on the probabilistic reconstruction of marine pollutant concentration fields from sparse measurements.

The work is developed within the context of the **HYDRAS** research project:
**HYdrodynamic-aware Distributed Robots for mArine Source-seeking**.

---

## Project Context

HYDRAS focuses on autonomous marine monitoring and pollutant source localization using networks of underwater or surface robots.

The general objective is to enable teams of autonomous vehicles (AUVs/ASVs) to collect environmental measurements and collaboratively reconstruct pollutant concentration fields in complex coastal environments.

In realistic scenarios, robots operate with local sensing and limited communication. The broader research framework combines hydrodynamic modeling, probabilistic reconstruction, and learning-based strategies to support efficient exploration and source localization under uncertain conditions.



This repository focuses on the reconstruction component of the problem, using **Gaussian Processes** to estimate spatial pollutant concentration fields from sparse measurements.


---

## Thesis Scope

The long-term thesis objective is to reconstruct and localize marine pollutant sources using probabilistic models and robotic measurements.

The thesis is organized as a progressive sequence of phases. Each phase increases the realism of the sensing and reconstruction problem.

| Phase | Goal 
|---|---|
| Phase 1 | Stationary field reconstruction from sparse point samples 
| Phase 2 | Stationary field reconstruction from measurements collected along robot trajectories 
| Phase 3A | Online informative exploration using GP uncertainty 
| Phase 3B | Space-time reconstruction of a time-varying concentration field 
| Phase 4 | Combination of space-time reconstruction and online trajectory optimization 
| Phase 5 | Pollutant source localization from reconstructed maps 

This repository is structured as a modular framework that will be progressively extended to include trajectory-based sensing, uncertainty-driven exploration, and source localization.


---

## Phase 1 — Stationary Field Reconstruction

In the first phase, the concentration field is treated as **stationary**: a single time snapshot is selected from the dataset and reconstructed as a fixed 2D spatial field, simulating what a robot swarm would observe at a given instant.

The robots are modelled as synthetic sensors that sample the concentration at a limited number of spatial positions within the valid marine domain (land cells, encoded as NaN, are excluded). A GP is then trained on these sparse measurements to reconstruct the full concentration field across the entire domain.

GPs are well suited for this task because they provide not only a **mean concentration estimate** at unobserved locations, but also a **predictive uncertainty** (posterior standard deviation) that quantifies the confidence of the reconstruction. Uncertainty is high far from sensor positions and low near observed points — physically meaningful behaviour for robot-driven sensing.

This phase addresses two objectives:

1. Validate the GP approach on realistic CMEMS simulation data
2. Study how reconstruction quality varies with the **number of sensor samples** — directly informing how many robots (or measurements per robot) are needed in practice

The dataset is a CMEMS NetCDF simulation output on a structured UTM grid with dimensions `(time, y, x)`.

---


## Project Structure
```
HYDRAS/
├── pollutant_gp/
│   ├── __init__.py          # Package entry point
│   ├── cli.py               # Command-line argument definitions
│   ├── data.py              # NetCDF loading, validation, and grid preparation
│   ├── inspection.py        # Utility for inspecting NetCDF files in a folder
│   ├── model.py             # GP kernel construction, training, and batch prediction
│   ├── reconstruction.py    # Field reconstruction and metrics
│   ├── sampling.py          # Synthetic sensor point sampling from valid grid cells
│   ├── types.py             # Shared dataclasses: GridData, ReconstructionResult
│   ├── visualization.py     # All plotting functions
│   └── workflow.py          # Main pipeline orchestration
├── main.py                  # Entry point
├── requirements.txt         # Python dependencies
├── .gitignore
└── LICENSE
```

---
## Dependencies

Create and activate a Python environment, then install the required dependencies:

```bash
pip install -r requirements.txt
```
---

## Usage

### Basic reconstruction

```bash
python main.py --nc-file path/to/file.nc --time-index <index>
```

Runs the full pipeline.

### Sample size study (single seed)

```bash
python main.py --nc-file path/to/file.nc --sample-size-study
```

After the main reconstruction, reruns the GP for each value in `--sample-size-study-counts` and saves a RMSE / MAE / R² vs n_samples plot.

### Sample size study (multiple seeds)

```bash
python main.py --nc-file path/to/file.nc --sample-size-study-multiseed
```

Runs the study over multiple random seeds and produces a plot with individual seed curves, a mean curve, and a shaded ±1 std band for each metric.

### Inspect NetCDF files in a folder

```bash
python main.py --inspect-netcdf --netcdf-dir path/to/folder/
```

### Print dataset structure and save valid-domain map

```bash
python main.py --nc-file path/to/file.nc --print-dataset
```
---
## Main Options

The most important command-line options are:

- `--nc-file`: path to the NetCDF dataset  
- `--time-index`: time snapshot to reconstruct (auto-selected if omitted)  
- `--n-samples`: number of synthetic sensor measurements  
- `--kernel-mode`: `anisotropic` or `isotropic`  
- `--target-transform`: optional transformation (e.g. `log1p`)  
- `--sample-size-study`: run reconstruction for multiple sample sizes  

For the complete list of options:

```bash
python main.py --help
```

---
## Outputs

The code generates diagnostic plots including:

- reconstructed concentration field  
- predictive uncertainty map  
- reconstruction error map  
- sample-size study plots  

Results are saved locally in the `outputs/` directory.


