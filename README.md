# SCLT: Spatially Coupled Low-Rank Tensor Completion

Research code and recovered experiment artifacts for **Spatially Coupled Low-Rank Tensor Completion for Zero-Observation Spatio-Temporal Data Reconstruction**, by Linchao Li, Jiahan Huang, and Hanhao Li.

The repository retains its historical name `ADMM-2B`. The current main implementation is called **SCLT-IDW**. Historical result keys `admm_2b` and `admm_idw` remain unchanged for traceability. Kriging is an auxiliary variant.

## Release Status

**v0.2.0-rc.1 is a recovery/reproduction release candidate, not a certified end-to-end reproduction.** The original cloud server expired. The available core code, later experiment drivers, selected parameters, saved results and publication figures are preserved here. Original processed tensors, the historical environment lockfile and some per-seed sweep outputs were not recovered in this package. Read [the reproducibility audit](docs/REPRODUCIBILITY.md) before rerunning or interpreting the experiments.

Two separate workflows are provided:

1. **Saved-result reconstruction:** verify checksums, export numerical tables and redraw six experimental figures. No model is fitted.
2. **Fresh experiments:** provide processed tensors in an isolated Linux cloud environment, review historical protocol caveats and explicitly launch the cloud runner. This workflow has not been revalidated after the server expired.

## Saved-Result Reproduction

Use Python 3.10 or newer in an isolated environment, from the repository root:

```bash
python -m pip install -r requirements-reproduction.txt
python reproduction/reproduce.py verify
python reproduction/reproduce.py tables
python reproduction/reproduce.py figures
```

Outputs go to `generated/`, never overwrite archived inputs, and are excluded from Git. Times New Roman is used when legally installed. Otherwise the script warns and uses STIXGeneral; numeric values remain unchanged. Use `figures --font-dir /path/to/fonts` to specify installed Times New Roman files. Fonts are not redistributed.

The original framework is preserved byte-for-byte; the rejected redrawn framework is not used. Submitted table snippets are in `publication/tables/`. Regenerated numerical tables include a provenance report identifying unavailable source rows.

## Cloud Experiments

Read [data preparation](data/README.md) and [protocol caveats](docs/REPRODUCIBILITY.md) first. The commands above do not download datasets or launch experiments.

```bash
python -m pip install -r requirements-core.txt
# Also install requirements.txt for the legacy BRITS/SAITS benchmark group.
python reproduction/reproduce.py plan
python reproduction/reproduce.py preflight --data-dir /path/to/prepared/data
python reproduction/reproduce.py cloud-run \
  --data-dir /path/to/prepared/data \
  --output-dir /path/to/new-isolated-run \
  --workers 2 --blas-threads 1 \
  --confirm-cloud --acknowledge-legacy-protocol
```

The output directory must not already exist. Each launch copies code and inputs into a new isolated directory, restores selected parameter caches, records the environment, and writes logs and a job-status report. Do not use another project's directory. Workers multiply memory demand; use one worker for runtime/memory measurement.

Select experiment groups with `--groups zero holdouts faults density ablation scalability sensitivity timeseries`. Select datasets with `--datasets guangzhou pemsd7 epa_ca`. The main plan excludes two-stage and auxiliary Kriging-coupled comparisons; historical functions remain for traceability.

## Contents

| Path | Purpose |
|---|---|
| `src/` | Existing ADMM, CP, interpolation, splitting and baseline code |
| `run_section_5_*.py` | Recovered holdout, corruption and ablation drivers |
| `run_spatial_prior_fault_experiment.py` | Density and flagged-rate experiments |
| `run_tkde_extra_experiments.py` | Scalability/sensitivity driver; historical name retained |
| `generate_zero_observation_timeseries_detail.py` | Refit and select illustrative time-series cases on the cloud |
| `configs/paper_parameters.json` | Selected parameters recovered from result metadata |
| `reproduction/` | Archive checks, table export, plots and isolated cloud runner |
| `reproduction_data/` | Recovered results, not raw training datasets |
| `paper_results/` | Previously published historical outputs |
| `publication/` | Current seven figures and four submitted table snippets |
| `docs/` | Provenance, limitations and validation record |

## Interpretation

SCLT combines spatial information for locations with no observations and low-rank structure for repairing monitored entries. The evidence is conditional: it does not establish universal superiority, anomaly detection, or guaranteed success for large contiguous outages. Zero-observation and flagged-entry errors must be evaluated separately.

## Citation and Contact

See [CITATION.cff](CITATION.cff). No publication DOI is claimed. Corresponding author: Linchao Li, `lilinchao@szu.edu.cn`.

This release introduces no new software or dataset license. Preserve existing third-party terms and seek author clarification where no license is specified.
