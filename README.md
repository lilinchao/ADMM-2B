# ADMM-2B: Joint Kriging and Tensor Decomposition for Spatial Extrapolation

Official implementation of **"Where No Sensor Goes: Kriging-Informed Low-Rank Tensor Recovery for Unobserved Location Inference"**.

## Overview

ADMM-2B is a 2-block ADMM framework that unifies ordinary kriging with low-rank CP tensor decomposition for spatial extrapolation in spatio-temporal monitoring networks. The kriging block provides spatially informed estimates at unobserved locations, while the tensor block repairs sensor faults and enforces global low-rank consistency. The two blocks reinforce each other through dual updates, creating a positive feedback loop that makes the method uniquely robust to sensor faults.

## Project Structure

```
├── src/                        # Core implementation
│   ├── admm_joint.py           # ADMM-2B and 3-block ADMM solvers
│   ├── kriging.py              # Ordinary kriging with variogram fitting
│   ├── cp_als.py               # CP tensor decomposition via ALS
│   ├── halrtc.py               # HaLRTC baseline
│   ├── interpolation_baselines.py  # IDW, linear interpolation
│   ├── data_loader.py          # Tensor data loading
│   ├── dataset_config.py       # Dataset configuration (Guangzhou, PeMSD7, EPA CA)
│   ├── data_split.py           # Train/test split utilities
│   ├── metrics.py              # MAE, RMSE evaluation
│   ├── hpo.py                  # Hyperparameter optimization
│   ├── two_stage.py            # Two-stage baseline
│   └── baselines/
│       └── brits.py            # BRITS and SAITS deep learning baselines
├── run_experiment1.py          # Exp 1: Pure spatial extrapolation benchmark
├── run_experiment2.py          # Exp 2: Ablation study
├── run_experiment3.py          # Exp 3: Parameter sensitivity
├── run_experiment4.py          # Exp 4: Convergence analysis
├── run_experiment5.py          # Exp 5: Mixed missingness (spatial + faults)
├── run_hpo.py                  # Hyperparameter optimization runner
├── run_all_experiments.sh      # Master script to reproduce all experiments
├── data/                       # Datasets (see data/README.md for download)
│   ├── tensor.mat              # Guangzhou traffic tensor
│   ├── PeMSD7/                 # PeMSD7 traffic data
│   ├── EPA_CA/                 # EPA CA air quality data
│   └── external/               # Data download/processing scripts
├── results/                    # Experiment results (JSON)
├── overleaf/                   # LaTeX paper source
└── figures/                    # Table LaTeX files
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

PyTorch with CUDA is recommended for the deep learning baselines (BRITS, SAITS). CPU-only PyTorch also works.

### 2. Prepare Data

Follow the instructions in [`data/README.md`](data/README.md) to download the three datasets. At minimum, you need the `.mat` tensor files:

- `data/tensor.mat` (Guangzhou)
- `data/PeMSD7/tensor.mat` (PeMSD7)
- `data/EPA_CA/tensor.mat` (EPA CA)

## Reproducing Results

### Quick Reproduction (All Experiments)

```bash
bash run_all_experiments.sh
```

This runs all five experiments sequentially (HPO → Exp 1-5). Total runtime is approximately 4-8 hours depending on hardware.

### Individual Experiments

```bash
# Step 1: Hyperparameter optimization (required before experiments)
python run_hpo.py --max-iter 200 --cache-dir results/hpo_cache

# Experiment 1: Pure spatial extrapolation
python run_experiment1.py --deploy-numbers 20 50 100 150 190 --n-seeds 3 --max-iter 200 \
    --hpo-cache results/hpo_cache --output results/experiment1.json

# Experiment 2: Ablation study
python run_experiment2.py --deploy-numbers 20 50 100 150 190 --n-seeds 3 --max-iter 200 \
    --hpo-cache results/hpo_cache --output results/experiment2.json

# Experiment 3: Parameter sensitivity
python run_experiment3.py --n-deploy 100 --n-seeds 3 --max-iter 200 \
    --hpo-cache results/hpo_cache --output results/experiment3.json

# Experiment 4: Convergence
python run_experiment4.py --deploy-numbers 50 100 150 --gammas 0.5 1.0 5.0 --max-iter 200 \
    --output results/experiment4.json

# Experiment 5: Mixed missingness (spatial extrapolation + sensor faults)
python run_experiment5.py --deploy-numbers 100 --fault-rates 0 0.05 0.1 0.2 0.3 \
    --n-seeds 3 --max-iter 200 --hpo-cache results/hpo_cache --output results/experiment5.json
```

### Per-Dataset Experiments

Each experiment script supports a `--dataset` flag:

```bash
python run_experiment1.py --dataset guangzhou --deploy-numbers 100 --n-seeds 3 --max-iter 200
python run_experiment1.py --dataset pemsd7   --deploy-numbers 100 --n-seeds 3 --max-iter 200
python run_experiment1.py --dataset epa_ca   --deploy-numbers 50  --n-seeds 3 --max-iter 200
```

## Key Results

| Method | Guangzhou MAE | PeMSD7 MAE | EPA CA MAE |
|--------|:---:|:---:|:---:|
| **ADMM-2B** | **6.98** | **7.42** | **4.42** |
| IDW | 6.99 | 7.96 | 4.51 |
| Kriging | 7.94 | 7.90 | 4.91 |
| CP-TD | 163.7 | 562.9 | 34.9 |
| HaLRTC | 38.5 | 55.7 | 14.8 |
| BRITS | 40.7 | 54.2 | 28.5 |
| SAITS | 45.1 | 58.7 | 267.3 |

At 30% sensor fault rate, ADMM-2B is the **only method whose MAE decreases** (−11% on Guangzhou, −8% on PeMSD7), while IDW's MAE increases by 130–216%.

## Citation

```bibtex
@article{admm2b2026,
  title={Where No Sensor Goes: Kriging-Informed Low-Rank Tensor Recovery for Unobserved Location Inference},
  author={},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2026}
}
```

## License

This project is released under the MIT License.
