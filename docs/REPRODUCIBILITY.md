# Reproducibility Audit

## Scope

Recovery candidate `v0.2.0-rc.1` was assembled on 16 November 2026 from public commit `307b6ba` and locally preserved scripts/results. The original cloud server expired. Core files in `src/` are preserved, not rewritten to manufacture agreement with saved results. Their exact identity with every historical cloud working tree cannot be certified.

| Evidence | Status | Limitation |
|---|---|---|
| Core solver and baseline code | Available | Historical cloud source hash absent |
| Later experiment/case-study drivers | Available | Full rerun not performed |
| Selected parameters and seed counts | Recovered from JSON | Full HPO search history absent |
| Coordinate arrays | In original repository | Sensor ordering must match |
| Three processed tensors | Not in release | Original checksums unavailable |
| Six experimental figure inputs | Available | Rebuilding plots is not rerunning models |
| Original framework | Original PDF retained | Not regenerated |
| Four submitted tables | Snippets preserved | Some rows lack surviving per-seed evidence |
| Structured fault/holdout per-seed results | Largely available | Density/rate outputs only partly recovered |
| Exact cloud package versions/hardware | Not recovered | New validation environment is not a historical lockfile |

Archive hashes check preservation, not scientific validity. Historical JSON uses `NaN` for undefined metrics, such as Fault MAE without any faults. Python's reader accepts this extension. CSV exports use empty values for unavailable statistics, not invented zeros.

## Evidence Map

| Current manuscript item | Saved source |
|---|---|
| Figure 1 framework | `publication/figures/SCLT_Framework_HighQuality.pdf` |
| Figure 2 time-series detail | `results/zero_observation_timeseries_detail/pemsd7_*` |
| Figure 3 mask degradation | `results/section_5_fault_stress/section_5_fault_stress_summary.csv` |
| Figure 4 density trend | `section_5_revised_masked_summary.csv` |
| Figure 5 structured faults | `results/section_5_fault_stress/section_5_fault_stress_summary.csv` |
| Figure 6 runtime | `results/tkde_extra/scalability.json` |
| Figure 7 sensitivity | `results/tkde_extra/sensitivity.json` |
| Table 1 datasets | Submitted snippet and `configs/paper_parameters.json` |
| Table 2 zero observation | `paper_results/experiment1_*.json`; table exporter flags missing rows |
| Table 3 fault rates | Saved aggregate `section_5_revised_masked_summary.csv`; original sweep JSONs incomplete |
| Table 4 ablation | `results/section_5_fault_stress/*_mask_perfect.json` and `results/section_5_ablation/*.json` |

Paths except those beginning `publication/`, `configs/` or `paper_results/` are relative to `reproduction_data/`.

## Historical Protocol Caveats

These findings are documented rather than silently changing algorithms or numbers:

1. **Prefilling:** `_prefill_faults_for_joint_model` temporally fills flagged entries and passes the deployment mask to the joint solver. Filled entries consequently participate in its observation term. This differs from fitting exclusively on original valid observations.
2. **Full-tensor preprocessing:** several drivers normalize the complete tensor and fit a variogram with an all-true mask before the holdout. Evaluation-safe preprocessing must be reviewed before claiming a leakage-free benchmark. Correcting it requires new experiments, not relabeling saved values.
3. **Fault rate:** density/rate masks sample with replacement, so nominal rates can differ from the number of unique flagged entries. Stress masks sample without replacement. Do not treat these protocols as identical.
4. **Tensor order:** stored axes differ across datasets and comments. C-order `(J, K)` flattening of `burst` masks is not necessarily chronological for `(slot, day)` arrays. Interpret these archived experiments accordingly.
5. **ADMM settings:** zero-observation and fault wrappers default to different `rho` values (1.0 and 0.1). Selected-parameter records alone do not capture every default. No new convergence or optimality guarantee is claimed.
6. **Ablation:** the public solver explicitly branches for `gamma == 0`. The saved no-spatial ablation is retained; it is not an independently implemented standard CP baseline.
7. **Case selection:** the illustrated sensor/windows were selected for high measured gain, not randomly sampled or representative of all sensors. Selection metadata and full saved traces are included.
8. **Runtime:** saved timings depend on hardware. Memory is sampled process RSS, not an asymptotic bound. Concurrent jobs affect both; use `--workers 1` for measurements. A historical hardware lockfile is unavailable.
9. **Deep baselines:** exact historical PyPOTS/PyTorch versions, checkpoints and full training history were not recovered. Re-runs are not guaranteed to match the submitted table. Consult the exported table provenance report.

The cloud runner requires `--acknowledge-legacy-protocol`. This acknowledges, but does not correct, these limitations. A corrected benchmark should use a distinct version and fresh results.

## Validation Boundaries

Local checks are limited to syntax, file hashes, selected arithmetic, table export and figure rendering. No model training or scientific experiment was run locally. Missing original tensors are not replaced with synthetic or similarly shaped data. Experiment launch requires explicit confirmation on Linux.

```bash
python -m unittest discover -s tests -v
python reproduction/reproduce.py verify
```

Before a stable complete-reproduction release: recover/provenance-check processed tensors; establish a pinned cloud environment; resolve protocol discrepancies against the paper; rerun all benchmarks on the cloud; compare with archived values; publish differences and corrections.
