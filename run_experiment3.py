"""
Experiment 3: Parameter sensitivity analysis.

For each parameter of ADMM-2B, sweep a range while fixing others,
and evaluate on test set.
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
    parser = argparse.ArgumentParser(description="Experiment 3: Parameter sensitivity")
    add_dataset_arg(parser)
    parser.add_argument("--n-deploy", type=int, default=None,
                        help="Fixed deployment number for sensitivity analysis (default: ~50%% of sensors)")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--hpo-cache", type=str, default=None,
                        help="HPO cache directory (default: per-dataset)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: per-dataset)")
    args = parser.parse_args()

    dcfg = get_config(args.dataset)
    output_path = PROJECT_ROOT / (args.output or default_output_path(args.dataset, "results/experiment3.json"))
    hpo_cache_dir = str(PROJECT_ROOT / (args.hpo_cache or dcfg["hpo_cache_dir"]))

    # Load data
    print("Loading dataset...")
    X_raw, sensor_coords, data_range, xmin, _ = load_dataset(PROJECT_ROOT, args.dataset)
    X_norm = (X_raw - xmin) / (data_range + 1e-8)
    I, J, K = X_norm.shape

    # Default n_deploy to ~50% of sensors
    if args.n_deploy is None:
        n_deploy_default = max(20, I // 2)
    else:
        n_deploy_default = args.n_deploy

    full_mask = np.ones(X_norm.shape, dtype=bool)
    variogram_params = fit_variogram_from_tensor(X_norm, full_mask, sensor_coords)

    split = split_sensors(I, test_frac=0.3, val_frac_of_deploy=0.1, seed=42)
    print(f"  {split.summary()}")

    # Load best params
    hpo_res = load_hpo_results("admm_2b", hpo_cache_dir)
    if hpo_res is not None:
        base = hpo_res["best_params"]
    else:
        base = {"rank": 10, "gamma": 1.0, "n_neighbors": 20, "beta": 0.01, "rho": 1.0}
    print(f"  Base params: {base}")

    # Sensitivity parameters
    sensitivity_specs = [
        {
            "name": "rank",
            "label": "Rank R",
            "values": [3, 5, 10, 15, 20, 30],
            "fixed": {"gamma": 1.0, "n_neighbors": 20, "beta": 0.01},
        },
        {
            "name": "gamma",
            "label": "γ",
            "values": [0.1, 0.5, 1, 5, 10, 50],
            "fixed": {"rank": 10, "n_neighbors": 20, "beta": 0.01},
        },
        {
            "name": "n_neighbors",
            "label": "K (neighbors)",
            "values": [5, 10, 15, 20, 30],
            "fixed": {"rank": 10, "gamma": 1.0, "beta": 0.01},
        },
        {
            "name": "beta",
            "label": "β",
            "values": [0.001, 0.01, 0.1, 1.0],
            "fixed": {"rank": 10, "gamma": 1.0, "n_neighbors": 20},
        },
    ]

    all_results = {}

    for spec in sensitivity_specs:
        pname = spec["name"]
        plabel = spec["label"]
        values = spec["values"]
        print(f"\n{'='*60}")
        print(f"  Sensitivity: {plabel}, values={values}")
        print(f"  Fixed: {spec['fixed']}, n_deploy={n_deploy_default}")
        print(f"{'='*60}")

        param_results = []

        for val in values:
            params = {**spec["fixed"], pname: val, "rho": base.get("rho", 1.0)}
            seed_results = []

            for seed in range(args.n_seeds):
                mask, test_positions, deployed = make_test_mask(
                    X_norm.shape, split, n_deploy_default, seed=seed
                )
                X_obs = X_norm.copy()
                X_obs[~mask] = 0.0

                print(f"    {plabel}={val}, seed={seed}...", end=" ", flush=True)
                t0 = time.time()
                try:
                    idw_power = params.get("idw_power", 1)
                    tau_j = params.get("tau_j", 3.0)
                    tau_k = params.get("tau_k", 1.0)
                    X_hat, info = admm_two_block(
                        X_obs, mask, sensor_coords, {},
                        rank=params["rank"],
                        gamma=params["gamma"],
                        beta=params["beta"],
                        rho=params["rho"],
                        max_iter=args.max_iter, tol=1e-4,
                        n_neighbors=params["n_neighbors"],
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
                    seed_results.append({
                        "seed": seed, "mae": test_mae, "rmse": test_rmse, "elapsed": elapsed,
                    })
                    print(f"MAE={test_mae:.3f}  ({elapsed:.0f}s)")
                except Exception as e:
                    elapsed = time.time() - t0
                    seed_results.append({
                        "seed": seed, "mae": float("nan"), "rmse": float("nan"),
                        "elapsed": elapsed, "error": str(e),
                    })
                    print(f"FAILED: {e}")

            param_results.append({"value": val, "seeds": seed_results})

        all_results[pname] = {
            "label": plabel,
            "values": values,
            "fixed": spec["fixed"],
            "results": param_results,
        }

        # Print summary for this parameter
        print(f"\n  {plabel} sensitivity:")
        for pr in param_results:
            vals = [s["mae"] for s in pr["seeds"] if not np.isnan(s.get("mae", float("nan")))]
            if vals:
                print(f"    {plabel}={pr['value']:<8}  MAE={np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # Save
    output_data = {
        "experiment": "experiment3_sensitivity",
        "n_deploy": n_deploy_default,
        "base_params": base,
        "results": all_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
