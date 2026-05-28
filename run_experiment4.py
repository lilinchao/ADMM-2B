"""
Experiment 4: Convergence analysis.

Tracks ADMM-2B primal/dual residuals across iterations for different
deployment numbers and gamma values.
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
from src.data_split import split_sensors, make_test_mask
from src.hpo import load_hpo_results
from src.kriging import fit_variogram_from_tensor
from src.admm_joint import admm_two_block


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Convergence analysis")
    add_dataset_arg(parser)
    parser.add_argument("--deploy-numbers", nargs="+", type=int, default=None,
                        help="Deployment numbers (default: per-dataset subset)")
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.5, 1.0, 5.0])
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: per-dataset)")
    args = parser.parse_args()

    dcfg = get_config(args.dataset)
    output_path = PROJECT_ROOT / (args.output or default_output_path(args.dataset, "results/experiment4.json"))
    hpo_cache_dir = str(PROJECT_ROOT / dcfg["hpo_cache_dir"])
    deploy_numbers = args.deploy_numbers or [50, 100, 150]

    # Load HPO best params
    hpo_res = load_hpo_results("admm_2b", hpo_cache_dir)
    if hpo_res is not None:
        best_params = hpo_res["best_params"]
        print(f"  HPO best params: {best_params}")
    else:
        best_params = {}
        print("  WARNING: No HPO results, using defaults")

    # Load data
    print("Loading dataset...")
    X_raw, sensor_coords, data_range, xmin, _ = load_dataset(PROJECT_ROOT, args.dataset)
    X_norm = (X_raw - xmin) / (data_range + 1e-8)
    I, J, K = X_norm.shape

    full_mask = np.ones(X_norm.shape, dtype=bool)
    variogram_params = fit_variogram_from_tensor(X_norm, full_mask, sensor_coords)

    split = split_sensors(I, test_frac=0.3, val_frac_of_deploy=0.1, seed=42)
    print(f"  {split.summary()}")

    convergence_results = []

    for n_deploy in deploy_numbers:
        for gamma in args.gammas:
            print(f"\n  n_deploy={n_deploy}, gamma={gamma}...")
            mask, test_positions, deployed = make_test_mask(
                X_norm.shape, split, n_deploy, seed=0
            )
            X_obs = X_norm.copy()
            X_obs[~mask] = 0.0

            # Run with convergence logging
            idw_power = best_params.get("idw_power", 1)
            tau_j = best_params.get("tau_j", 3.0)
            tau_k = best_params.get("tau_k", 1.0)
            X_hat, info = admm_two_block(
                X_obs, mask, sensor_coords, {},
                rank=best_params.get("rank", 10),
                gamma=gamma,
                beta=best_params.get("beta", 0.01),
                rho=1.0,
                max_iter=args.max_iter, tol=1e-4,
                n_neighbors=best_params.get("n_neighbors", 20),
                seed=0,
                sensor_coords=sensor_coords,
                variogram_params=variogram_params,
                kriging_mode="idw", idw_spatial_only=True, idw_power=idw_power, tau_j=tau_j, tau_k=tau_k,
                adaptive_gamma=False,
                log_convergence=True,
            )

            # Extract convergence history
            r_primals = info.get("primal_res", [])
            r_duals = info.get("dual_res", [])
            objectives = info.get("obj", [])
            gamma_hist = info.get("gamma_hist", [])
            n_iters = info.get("n_iter", 0)

            # Evaluate final result
            X_pred_raw = X_hat * data_range + xmin
            diff = X_raw[test_positions] - X_pred_raw[test_positions]
            test_mae = float(np.abs(diff).mean())

            convergence_results.append({
                "n_deploy": n_deploy,
                "gamma": gamma,
                "n_iters": n_iters,
                "final_mae": test_mae,
                "r_primals": r_primals,
                "r_duals": r_duals,
                "objectives": objectives,
                "gamma_hist": gamma_hist,
            })

            print(f"    Converged in {n_iters} iters, test_MAE={test_mae:.3f}")

    # Save
    output_data = {
        "experiment": "experiment4_convergence",
        "results": convergence_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
