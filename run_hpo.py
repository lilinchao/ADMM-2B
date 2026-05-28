"""
Hyperparameter optimization for all methods using train/val/test split.

Runs HPO on validation set, caches results to JSON.
Can use joblib for parallel execution across methods.
"""

import sys
import time
import numpy as np
from pathlib import Path
import json
import argparse

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import scipy.io
from src.dataset_config import add_dataset_arg, load_dataset, get_config
from src.data_split import split_sensors, make_hpo_masks
from src.hpo import run_hpo, enumerate_configs
from src.kriging import SpatioTemporalKriging, fit_variogram_from_tensor
from src.cp_als import cp_als_best_of
from src.halrtc import halrtc
from src.admm_joint import admm_two_block
from src.interpolation_baselines import linear_time
from src.baselines import run_brits, run_saits


# ---------------------------------------------------------------------------
# Method run functions (signature: X_obs, mask, sensor_coords, variogram_params, max_iter, **params)
# ---------------------------------------------------------------------------

def run_idw(X_obs, mask, sensor_coords=None, variogram_params=None, max_iter=100, **params):
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


def run_linear_time_fn(X_obs, mask, sensor_coords=None, variogram_params=None, max_iter=100, **params):
    return linear_time(X_obs, mask)


def run_cp_td(X_obs, mask, sensor_coords=None, variogram_params=None, max_iter=100, **params):
    rank = params.get("rank", 10)
    beta = params.get("beta", 0.01)
    X_hat, _, _ = cp_als_best_of(X_obs, mask, rank=rank, beta=beta,
                                  max_iter=max_iter, tol=1e-4, n_restarts=1)
    return X_hat


def run_halrtc_fn(X_obs, mask, sensor_coords=None, variogram_params=None, max_iter=100, **params):
    rho = params.get("rho", 0.01)
    X_hat, _ = halrtc(X_obs, mask, rho=rho, max_iter=50, tol=1e-4)
    return X_hat


def run_ordinary_kriging(X_obs, mask, sensor_coords=None, variogram_params=None, max_iter=100, **params):
    n_neighbors = params.get("n_neighbors", 20)
    krig = SpatioTemporalKriging(
        sensor_coords=sensor_coords,
        n_neighbors=n_neighbors, kriging_mode="variogram",
        tau_j=3.0, tau_k=1.0,
    )
    krig.fit(X_obs, mask)
    X_hat = krig.predict_full(X_obs, mask)
    return X_hat


def run_admm_2b(X_obs, mask, sensor_coords=None, variogram_params=None, max_iter=100, **params):
    rank = params.get("rank", 10)
    gamma = params.get("gamma", 1.0)
    n_neighbors = params.get("n_neighbors", 20)
    beta = params.get("beta", 0.01)
    rho = params.get("rho", 1.0)
    idw_power = params.get("idw_power", 1)
    tau_j = params.get("tau_j", 3.0)
    tau_k = params.get("tau_k", 1.0)
    locs = sensor_coords
    X_hat, info = admm_two_block(
        X_obs, mask, locs, {}, rank=rank, gamma=gamma, beta=beta, rho=rho,
        max_iter=max_iter, tol=1e-4, n_neighbors=n_neighbors, seed=0,
        sensor_coords=sensor_coords, variogram_params=variogram_params,
        kriging_mode="idw", idw_spatial_only=True, idw_power=idw_power, tau_j=tau_j, tau_k=tau_k,
        adaptive_gamma=False,
    )
    return X_hat


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "idw": run_idw,
    "linear_time": run_linear_time_fn,
    "cp_td": run_cp_td,
    "halrtc": run_halrtc_fn,
    "ordinary_kriging": run_ordinary_kriging,
    "admm_2b": run_admm_2b,
    "brits": run_brits,
    "saits": run_saits,
}

# CPU methods that can run in parallel (non-deep-learning)
CPU_METHODS = ["idw", "linear_time", "cp_td", "halrtc", "ordinary_kriging", "admm_2b"]

# Deep learning methods (via PyPOTS)
DL_METHODS = ["brits", "saits"]


def run_hpo_single_method(method_name, X_raw, X_norm, train_mask, val_positions,
                           data_range, xmin, sensor_coords, variogram_params,
                           max_iter, cache_dir):
    """Run HPO for a single method (designed for parallel execution)."""
    run_fn = METHOD_REGISTRY[method_name]
    print(f"\n{'='*60}")
    print(f"  HPO for {method_name}")
    print(f"  Configs: {len(enumerate_configs(method_name))}")
    print(f"{'='*60}")

    best_params, best_val_mae, all_results = run_hpo(
        method_name, run_fn, X_raw, X_norm, train_mask, val_positions,
        data_range, xmin, sensor_coords=sensor_coords,
        variogram_params=variogram_params, max_iter=max_iter,
        cache_dir=cache_dir,
    )

    print(f"\n  {method_name} BEST: val_MAE={best_val_mae:.3f}, params={best_params}")
    return method_name, best_params, best_val_mae, all_results


def main():
    parser = argparse.ArgumentParser(description="Run HPO for all methods")
    add_dataset_arg(parser)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to run HPO for (default: all CPU methods)")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Max iterations for iterative methods")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory for caching HPO results (default: per-dataset)")
    parser.add_argument("--parallel", action="store_true",
                        help="Run methods in parallel using joblib")
    parser.add_argument("--n-jobs", type=int, default=4,
                        help="Number of parallel jobs")
    args = parser.parse_args()

    dcfg = get_config(args.dataset)
    methods = args.methods or CPU_METHODS
    cache_dir = str(PROJECT_ROOT / (args.cache_dir or dcfg["hpo_cache_dir"]))

    # Load data
    print("Loading dataset...")
    X_raw, sensor_coords, data_range, xmin, _ = load_dataset(PROJECT_ROOT, args.dataset)
    X_norm = (X_raw - xmin) / (data_range + 1e-8)
    I, J, K = X_norm.shape

    # Variogram (fitted from full data — only used by methods that support it)
    full_mask = np.ones(X_norm.shape, dtype=bool)
    variogram_params = fit_variogram_from_tensor(X_norm, full_mask, sensor_coords)

    # Split sensors
    split = split_sensors(I, test_frac=0.3, val_frac_of_deploy=0.1, seed=42)
    print(f"  {split.summary()}")

    # HPO masks: train sensors observed, val sensors for evaluation
    train_mask, val_positions = make_hpo_masks(X_norm.shape, split)
    print(f"  Train observed: {train_mask.sum():,} / {train_mask.size:,}")
    print(f"  Val positions:  {val_positions.sum():,}")

    # Run HPO
    print(f"\nRunning HPO for {len(methods)} methods...")
    print(f"  Methods: {methods}")
    print(f"  Cache dir: {cache_dir}")

    all_hpo = {}

    if args.parallel:
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(run_hpo_single_method)(
                    m, X_raw, X_norm, train_mask, val_positions,
                    data_range, xmin, sensor_coords, variogram_params,
                    args.max_iter, cache_dir,
                )
                for m in methods
            )
            for method_name, best_params, best_val_mae, all_results in results:
                all_hpo[method_name] = {
                    "best_params": best_params,
                    "best_val_mae": best_val_mae,
                }
        except ImportError:
            print("joblib not available, running sequentially")
            args.parallel = False

    if not args.parallel:
        for m in methods:
            method_name, best_params, best_val_mae, all_results = run_hpo_single_method(
                m, X_raw, X_norm, train_mask, val_positions,
                data_range, xmin, sensor_coords, variogram_params,
                args.max_iter, cache_dir,
            )
            all_hpo[method_name] = {
                "best_params": best_params,
                "best_val_mae": best_val_mae,
            }

    # Summary
    print(f"\n{'='*70}")
    print("  HPO Summary — Best validation MAE and parameters")
    print(f"{'='*70}")
    for m in methods:
        if m in all_hpo:
            h = all_hpo[m]
            print(f"  {m:<16s}  val_MAE={h['best_val_mae']:.3f}  params={h['best_params']}")


if __name__ == "__main__":
    main()
