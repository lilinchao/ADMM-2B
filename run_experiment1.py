"""
Experiment 1: Multi-method benchmark with fair hyperparameter optimization.

For each method:
  1. HPO on validation set (using run_hpo.py cached results)
  2. Evaluate on test set at multiple deployment numbers × 3 seeds

Reports MAE/RMSE on test sensors only (never observed during HPO).
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
from src.halrtc import halrtc
from src.admm_joint import admm_two_block
from src.interpolation_baselines import linear_time
from src.baselines import run_brits, run_saits


# ---------------------------------------------------------------------------
# Method run functions (same as run_hpo.py but with full interface)
# ---------------------------------------------------------------------------

def run_idw(X_obs, mask, sensor_coords=None, **params):
    K = params.get("K", 10)
    power = params.get("power", 2)
    I = X_obs.shape[0]
    observed_ids = np.where(mask[:, 0, 0])[0]
    unobserved_ids = np.where(~mask[:, 0, 0])[0]
    if len(observed_ids) == 0 or len(unobserved_ids) == 0:
        return X_obs.copy()
    from scipy.spatial import KDTree
    obs_coords = sensor_coords[observed_ids]
    tree = KDTree(obs_coords)
    k_use = min(K, len(observed_ids))
    X_hat = X_obs.copy()
    for uid in unobserved_ids:
        dists, idxs = tree.query(sensor_coords[uid], k=k_use)
        if k_use == 1:
            dists = np.array([dists])
            idxs = np.array([idxs])
        dists = np.maximum(dists, 1e-10)
        weights = 1.0 / (dists ** power)
        weights /= weights.sum()
        obs_values = X_obs[observed_ids[idxs], :, :]
        X_hat[uid, :, :] = np.einsum("n,njk->jk", weights, obs_values)
    return X_hat


def run_linear_time_fn(X_obs, mask, **kw):
    return linear_time(X_obs, mask)


def run_cp_td(X_obs, mask, max_iter=200, **params):
    rank = params.get("rank", 10)
    beta = params.get("beta", 0.01)
    X_hat, _, _ = cp_als_best_of(X_obs, mask, rank=rank, beta=beta,
                                  max_iter=max_iter, tol=1e-4, n_restarts=1)
    return X_hat


def run_halrtc_fn(X_obs, mask, **params):
    rho = params.get("rho", 0.01)
    X_hat, _ = halrtc(X_obs, mask, rho=rho, max_iter=50, tol=1e-4)
    return X_hat


def run_ordinary_kriging(X_obs, mask, sensor_coords=None, **params):
    n_neighbors = params.get("n_neighbors", 20)
    krig = SpatioTemporalKriging(
        sensor_coords=sensor_coords,
        n_neighbors=n_neighbors, kriging_mode="variogram",
        tau_j=3.0, tau_k=1.0,
    )
    krig.fit(X_obs, mask)
    return krig.predict_full(X_obs, mask)


def run_admm_2b(X_obs, mask, sensor_coords=None, variogram_params=None, max_iter=200, seed=0, **params):
    rank = params.get("rank", 10)
    gamma = params.get("gamma", 1.0)
    n_neighbors = params.get("n_neighbors", 20)
    beta = params.get("beta", 0.01)
    rho = params.get("rho", 1.0)
    idw_power = params.get("idw_power", 1)
    tau_j = params.get("tau_j", 3.0)
    tau_k = params.get("tau_k", 1.0)
    X_hat, info = admm_two_block(
        X_obs, mask, sensor_coords, {}, rank=rank, gamma=gamma, beta=beta, rho=rho,
        max_iter=max_iter, tol=1e-4, n_neighbors=n_neighbors, seed=seed,
        sensor_coords=sensor_coords, variogram_params=variogram_params,
        kriging_mode="idw", idw_spatial_only=True, idw_power=idw_power, tau_j=tau_j, tau_k=tau_k,
        adaptive_gamma=False,
    )
    return X_hat


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "idw":              {"fn": run_idw,              "label": "IDW"},
    "linear_time":      {"fn": run_linear_time_fn,   "label": "Linear (time)"},
    "cp_td":            {"fn": run_cp_td,            "label": "CP-TD"},
    "halrtc":           {"fn": run_halrtc_fn,        "label": "HaLRTC"},
    "ordinary_kriging": {"fn": run_ordinary_kriging, "label": "Ordinary Kriging"},
    "admm_2b":          {"fn": run_admm_2b,          "label": "ADMM-2B"},
    "brits":            {"fn": run_brits,            "label": "BRITS"},
    "saits":            {"fn": run_saits,            "label": "SAITS"},
}


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Multi-method benchmark")
    add_dataset_arg(parser)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to evaluate (default: all)")
    parser.add_argument("--deploy-numbers", nargs="+", type=int, default=None,
                        help="Number of deployed sensors to test (default: per-dataset)")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of random seeds per deployment number")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Max iterations for iterative methods")
    parser.add_argument("--hpo-cache", type=str, default=None,
                        help="Directory with cached HPO results (default: per-dataset)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (default: per-dataset)")
    parser.add_argument("--dl-cache-dir", type=str, default=None,
                        help="Directory for caching DL model weights (default: per-dataset)")
    args = parser.parse_args()

    dcfg = get_config(args.dataset)
    method_names = args.methods or list(METHOD_REGISTRY.keys())
    output_path = PROJECT_ROOT / (args.output or default_output_path(args.dataset, "results/experiment1.json"))
    hpo_cache_dir = str(PROJECT_ROOT / (args.hpo_cache or dcfg["hpo_cache_dir"]))
    dl_cache_dir = str(PROJECT_ROOT / (args.dl_cache_dir or dcfg["dl_cache_dir"]))
    deploy_numbers = args.deploy_numbers or dcfg["default_deploy_numbers"]
    dl_methods = {"brits", "saits"}

    # Load data
    print("Loading dataset...")
    X_raw, sensor_coords, data_range, xmin, _ = load_dataset(PROJECT_ROOT, args.dataset)
    X_norm = (X_raw - xmin) / (data_range + 1e-8)
    I, J, K = X_norm.shape

    # Variogram
    full_mask = np.ones(X_norm.shape, dtype=bool)
    variogram_params = fit_variogram_from_tensor(X_norm, full_mask, sensor_coords)

    # Split sensors
    split = split_sensors(I, test_frac=0.3, val_frac_of_deploy=0.1, seed=42)
    print(f"  {split.summary()}")

    # Load HPO results for each method
    best_params = {}
    for m in method_names:
        hpo_res = load_hpo_results(m, hpo_cache_dir)
        if hpo_res is not None:
            best_params[m] = hpo_res["best_params"]
            print(f"  {m}: loaded HPO params = {hpo_res['best_params']}, val_MAE={hpo_res['best_val_mae']:.3f}")
        else:
            print(f"  WARNING: No HPO results for {m}, using defaults")
            best_params[m] = {}

    # Run experiment
    all_results = {m: {n: [] for n in deploy_numbers} for m in method_names}

    for n_deploy in deploy_numbers:
        print(f"\n{'='*80}")
        print(f"  Deployment: {n_deploy}/{I} sensors")
        print(f"{'='*80}")

        for seed in range(args.n_seeds):
            mask, test_positions, deployed = make_test_mask(
                X_norm.shape, split, n_deploy, seed=seed
            )
            X_obs = X_norm.copy()
            X_obs[~mask] = 0.0

            n_test_pts = test_positions.sum()
            print(f"\n  Seed {seed}: {len(deployed)} deployed, {n_test_pts} test positions")

            for m in method_names:
                mlabel = METHOD_REGISTRY[m]["label"]
                fn = METHOD_REGISTRY[m]["fn"]
                params = best_params[m].copy()

                print(f"    {mlabel:<18s}...", end=" ", flush=True)
                t0 = time.time()
                try:
                    extra = {}
                    if m in dl_methods:
                        extra["cache_dir"] = dl_cache_dir
                    X_hat = fn(
                        X_obs, mask,
                        sensor_coords=sensor_coords,
                        variogram_params=variogram_params,
                        max_iter=args.max_iter,
                        seed=seed,
                        **params,
                        **extra,
                    )
                    elapsed = time.time() - t0

                    # Evaluate on test positions only
                    X_pred_raw = X_hat * data_range + xmin
                    diff = X_raw[test_positions] - X_pred_raw[test_positions]
                    test_mae = float(np.abs(diff).mean())
                    test_rmse = float(np.sqrt((diff ** 2).mean()))

                    all_results[m][n_deploy].append({
                        "seed": seed,
                        "mae": test_mae,
                        "rmse": test_rmse,
                        "elapsed": elapsed,
                    })
                    print(f"MAE={test_mae:.3f}  RMSE={test_rmse:.3f}  ({elapsed:.0f}s)")
                except Exception as e:
                    elapsed = time.time() - t0
                    all_results[m][n_deploy].append({
                        "seed": seed,
                        "mae": float("nan"),
                        "rmse": float("nan"),
                        "elapsed": elapsed,
                        "error": str(e),
                    })
                    print(f"FAILED: {e}")

    # =======================================================================
    # Print results
    # =======================================================================
    for metric_name, metric_key in [("MAE", "mae"), ("RMSE", "rmse")]:
        print(f"\n{'='*80}")
        print(f"  Test {metric_name} — number of deployed sensors")
        print(f"{'='*80}")
        header = f"{'Method':<18s}"
        for n in deploy_numbers:
            header += f" | n={n:>3d}        "
        print(header)
        print("-" * len(header))
        for m in method_names:
            mlabel = METHOD_REGISTRY[m]["label"]
            row = f"{mlabel:<18s}"
            for n in deploy_numbers:
                vals = [v[metric_key] for v in all_results[m][n] if not np.isnan(v.get(metric_key, float("nan")))]
                if vals:
                    mean_v = np.mean(vals)
                    std_v = np.std(vals)
                    row += f" | {mean_v:>5.2f}±{std_v:<5.2f}"
                else:
                    row += f" |   N/A       "
            print(row)

    # =======================================================================
    # Relative improvement: ADMM-2B vs baselines
    # =======================================================================
    if "admm_2b" in method_names:
        print(f"\n{'='*80}")
        print("  ADMM-2B relative MAE improvement over baselines (%)")
        print("  Positive = ADMM-2B better")
        print(f"{'='*80}")
        baselines = [m for m in method_names if m != "admm_2b"]
        header = f"{'vs.':<18s}"
        for n in deploy_numbers:
            header += f" | n={n:>3d}   "
        print(header)
        print("-" * len(header))
        for b in baselines:
            blabel = METHOD_REGISTRY[b]["label"]
            row = f"{'vs. '+blabel:<18s}"
            for n in deploy_numbers:
                admm_vals = [v["mae"] for v in all_results["admm_2b"][n]]
                base_vals = [v["mae"] for v in all_results[b][n]]
                admm_mean = np.nanmean(admm_vals)
                base_mean = np.nanmean(base_vals)
                if base_mean > 0 and not np.isnan(base_mean) and admm_mean > 0:
                    pct = (base_mean - admm_mean) / base_mean * 100
                    sign = "+" if pct > 0 else ""
                    row += f" | {sign}{pct:>5.1f}"
                else:
                    row += f" |   N/A"
            print(row)

    # =======================================================================
    # Save results
    # =======================================================================
    output_data = {
        "experiment": "experiment1",
        "dataset": args.dataset,
        "shape": [I, J, K],
        "data_range": data_range,
        "split": {
            "train": split.train.tolist(),
            "val": split.val.tolist(),
            "test": split.test.tolist(),
            "deploy_pool": split.deploy_pool.tolist(),
        },
        "best_params": best_params,
        "deploy_numbers": deploy_numbers,
        "n_seeds": args.n_seeds,
        "results": {},
    }
    for m in method_names:
        output_data["results"][m] = {}
        for n in deploy_numbers:
            output_data["results"][m][str(n)] = all_results[m][n]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
