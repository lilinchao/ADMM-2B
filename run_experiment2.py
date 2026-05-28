"""
Experiment 2: Ablation study.

Compares ADMM-2B against its components and variants:
  - gamma→∞ (Kriging only): pure spatial extrapolation, no low-rank constraint
  - gamma=0 (CP-TD only): pure low-rank decomposition, no spatial extrapolation
  - ADMM-2B (full): joint optimization
  - Two-stage A: Kriging → CP-TD pipeline
  - ADMM no-U: remove dual variable update
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import scipy.io
from src.dataset_config import add_dataset_arg, load_dataset, default_output_path, get_config
from src.data_split import split_sensors, make_hpo_masks, make_test_mask
from src.hpo import load_hpo_results
from src.kriging import SpatioTemporalKriging, fit_variogram_from_tensor
from src.cp_als import cp_als_best_of
from src.admm_joint import admm_two_block
from src.two_stage import two_stage_kriging_then_td


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Ablation study")
    add_dataset_arg(parser)
    parser.add_argument("--deploy-numbers", nargs="+", type=int, default=None,
                        help="Deployment numbers (default: per-dataset)")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--hpo-cache", type=str, default=None,
                        help="HPO cache directory (default: per-dataset)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: per-dataset)")
    args = parser.parse_args()

    dcfg = get_config(args.dataset)
    output_path = PROJECT_ROOT / (args.output or default_output_path(args.dataset, "results/experiment2.json"))
    hpo_cache_dir = str(PROJECT_ROOT / (args.hpo_cache or dcfg["hpo_cache_dir"]))
    deploy_numbers = args.deploy_numbers or dcfg["default_deploy_numbers"]

    # Load data
    print("Loading dataset...")
    X_raw, sensor_coords, data_range, xmin, _ = load_dataset(PROJECT_ROOT, args.dataset)
    X_norm = (X_raw - xmin) / (data_range + 1e-8)
    I, J, K = X_norm.shape

    full_mask = np.ones(X_norm.shape, dtype=bool)
    variogram_params = fit_variogram_from_tensor(X_norm, full_mask, sensor_coords)

    split = split_sensors(I, test_frac=0.3, val_frac_of_deploy=0.1, seed=42)
    print(f"  {split.summary()}")

    # Load best params from HPO for ADMM-2B
    hpo_res = load_hpo_results("admm_2b", hpo_cache_dir)
    if hpo_res is not None:
        base_params = hpo_res["best_params"]
    else:
        base_params = {"rank": 10, "gamma": 1.0, "n_neighbors": 20, "beta": 0.01, "rho": 1.0}
    print(f"  Base params: {base_params}")

    # Ablation variants
    variants = {
        "kriging_only": {
            "label": "Kriging only (γ→∞)",
            "params": {**base_params, "gamma": 1e6},
        },
        "cp_td_only": {
            "label": "CP-TD only (γ=0)",
            "params": {**base_params, "gamma": 0.0},
        },
        "admm_2b_full": {
            "label": "ADMM-2B (full)",
            "params": base_params.copy(),
        },
        "two_stage_a": {
            "label": "Two-stage A",
            "params": base_params.copy(),
        },
        "admm_no_dual": {
            "label": "ADMM no dual (ρ→∞)",
            "params": {**base_params, "rho": 1e6},
        },
    }

    all_results = {v: {n: [] for n in deploy_numbers} for v in variants}

    for n_deploy in deploy_numbers:
        print(f"\n{'='*70}")
        print(f"  Deployment: {n_deploy}/{I} sensors")
        print(f"{'='*70}")

        for seed in range(args.n_seeds):
            mask, test_positions, deployed = make_test_mask(
                X_norm.shape, split, n_deploy, seed=seed
            )
            X_obs = X_norm.copy()
            X_obs[~mask] = 0.0
            print(f"\n  Seed {seed}: {len(deployed)} deployed")

            for vname, vinfo in variants.items():
                vlabel = vinfo["label"]
                params = vinfo["params"]
                print(f"    {vlabel:<25s}...", end=" ", flush=True)
                t0 = time.time()
                try:
                    if vname == "two_stage_a":
                        X_hat = two_stage_kriging_then_td(
                            X_obs, mask, sensor_coords, {},
                            rank=params.get("rank", 10),
                            beta=params.get("beta", 0.01),
                            max_iter=args.max_iter,
                            n_neighbors=params.get("n_neighbors", 20),
                            n_restarts=1,
                            sensor_coords=sensor_coords,
                            variogram_params=variogram_params,
                            kriging_mode="idw",
                            tau_j=params.get("tau_j", 3.0),
                            tau_k=params.get("tau_k", 1.0),
                        )
                    else:
                        idw_power = params.get("idw_power", 1)
                        tau_j = params.get("tau_j", 3.0)
                        tau_k = params.get("tau_k", 1.0)
                        X_hat, info = admm_two_block(
                            X_obs, mask, sensor_coords, {},
                            rank=params.get("rank", 10),
                            gamma=params.get("gamma", 1.0),
                            beta=params.get("beta", 0.01),
                            rho=params.get("rho", 1.0),
                            max_iter=args.max_iter, tol=1e-4,
                            n_neighbors=params.get("n_neighbors", 20),
                            seed=seed,
                            sensor_coords=sensor_coords,
                            variogram_params=variogram_params,
                            kriging_mode="idw", idw_spatial_only=True, idw_power=idw_power, tau_j=tau_j, tau_k=tau_k,
                            adaptive_gamma=False,
                        )

                    elapsed = time.time() - t0
                    X_pred_raw = X_hat * data_range + xmin
                    diff = X_raw[test_positions] - X_pred_raw[test_positions]
                    test_mae = float(np.abs(diff).mean())
                    test_rmse = float(np.sqrt((diff ** 2).mean()))
                    all_results[vname][n_deploy].append({
                        "seed": seed, "mae": test_mae, "rmse": test_rmse, "elapsed": elapsed,
                    })
                    print(f"MAE={test_mae:.3f}  RMSE={test_rmse:.3f}  ({elapsed:.0f}s)")
                except Exception as e:
                    elapsed = time.time() - t0
                    all_results[vname][n_deploy].append({
                        "seed": seed, "mae": float("nan"), "rmse": float("nan"),
                        "elapsed": elapsed, "error": str(e),
                    })
                    print(f"FAILED: {e}")

    # Print tables
    print(f"\n{'='*70}")
    print("  Ablation: Test MAE (mean ± std)")
    print(f"{'='*70}")
    header = f"{'Variant':<25s}"
    for n in deploy_numbers:
        header += f" | n={n:>3d}        "
    print(header)
    print("-" * len(header))
    for vname, vinfo in variants.items():
        row = f"{vinfo['label']:<25s}"
        for n in deploy_numbers:
            vals = [v["mae"] for v in all_results[vname][n] if not np.isnan(v.get("mae", float("nan")))]
            if vals:
                row += f" | {np.mean(vals):>5.2f}±{np.std(vals):<5.2f}"
            else:
                row += f" |   N/A       "
        print(row)

    # Save
    output_data = {
        "experiment": "experiment2_ablation",
        "variants": {k: v["label"] for k, v in variants.items()},
        "base_params": base_params,
        "results": {},
    }
    for v in variants:
        output_data["results"][v] = {}
        for n in deploy_numbers:
            output_data["results"][v][str(n)] = all_results[v][n]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
