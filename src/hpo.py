"""
Hyperparameter optimization framework.

Each method defines a search space. Grid search on validation set,
then report results on test set with best hyperparameters.
"""

import itertools
import time
import numpy as np
from pathlib import Path
import json


# ---------------------------------------------------------------------------
# Search space definitions
# ---------------------------------------------------------------------------

SEARCH_SPACES = {
    "idw": {
        "K": [5, 10, 15, 20, 30],
        "power": [1, 2, 3],
    },
    "linear_time": {},  # no hyperparameters
    "softimpute": {
        "lam": [0.01, 0.1, 1, 10],
    },
    "cp_td": {
        "rank": [3, 5, 10, 15, 20],
        "beta": [0.001, 0.01, 0.1],
    },
    "halrtc": {
        "rho": [0.001, 0.01, 0.1, 1],
    },
    "ordinary_kriging": {
        "n_neighbors": [5, 10, 15, 20, 30],
    },
    "admm_2b": {
        "rank": [5, 10, 15],
        "gamma": [0.5, 1.0, 5.0, 50.0],
        "n_neighbors": [10, 20, 30],
        "beta": [0.01, 0.1],
        "idw_power": [1, 2],
    },
    "brits": {
        "rnn_hidden_size": [64, 128],
        "lr": [1e-3, 1e-4],
    },
    "saits": {
        "d_model": [64, 128],
        "n_heads": [4],
        "n_layers": [2],
        "lr": [1e-3, 1e-4],
    },
}


def get_search_space(method_name):
    """Return dict of param_name -> list of values."""
    return SEARCH_SPACES.get(method_name, {})


def enumerate_configs(method_name):
    """Return list of all hyperparameter configurations for grid search."""
    space = get_search_space(method_name)
    if not space:
        return [{}]
    keys = list(space.keys())
    values = list(space.values())
    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


# ---------------------------------------------------------------------------
# HPO runner
# ---------------------------------------------------------------------------

def run_hpo(method_name, run_fn, X_raw, X_norm, train_mask, val_positions,
            data_range, xmin, sensor_coords=None, variogram_params=None,
            max_iter=100, cache_dir=None):
    """
    Run hyperparameter optimization for a method.

    Parameters
    ----------
    method_name : str
    run_fn : callable(X_obs, mask, **params) -> X_hat
        Function that runs the method and returns predicted tensor (normalized).
    X_raw : ndarray — ground truth in original scale
    X_norm : ndarray — normalized tensor
    train_mask : ndarray bool — observed positions (True = observed)
    val_positions : ndarray bool — positions for validation evaluation
    data_range, xmin : float — for denormalization
    sensor_coords : ndarray or None
    variogram_params : dict or None
    max_iter : int — max iterations for iterative methods
    cache_dir : str or None — directory to cache HPO results

    Returns
    -------
    best_params : dict
    best_val_mae : float
    all_results : list of (params, val_mae, val_rmse, elapsed)
    """
    # Check cache
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"hpo_{method_name}.json"
        if cache_path.exists():
            with open(cache_path, "r") as f:
                cached = json.load(f)
            return cached["best_params"], cached["best_val_mae"], cached["all_results"]

    configs = enumerate_configs(method_name)
    print(f"  HPO for {method_name}: {len(configs)} configurations")

    X_obs = X_norm.copy()
    X_obs[~train_mask] = 0.0

    all_results = []
    best_val_mae = float("inf")
    best_params = {}

    for i, params in enumerate(configs):
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"    [{i+1}/{len(configs)}] {param_str}...", end=" ", flush=True)
        t0 = time.time()
        try:
            X_hat = run_fn(
                X_obs, train_mask,
                sensor_coords=sensor_coords,
                variogram_params=variogram_params,
                max_iter=max_iter,
                **params,
            )
            elapsed = time.time() - t0

            # Evaluate on validation positions
            X_pred_raw = X_hat * data_range + xmin
            diff = X_raw[val_positions] - X_pred_raw[val_positions]
            val_mae = float(np.abs(diff).mean())
            val_rmse = float(np.sqrt((diff ** 2).mean()))

            all_results.append({
                "params": params,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "elapsed": elapsed,
            })

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_params = params

            print(f"val_MAE={val_mae:.3f} val_RMSE={val_rmse:.3f} ({elapsed:.0f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            all_results.append({
                "params": params,
                "val_mae": float("nan"),
                "val_rmse": float("nan"),
                "elapsed": elapsed,
            })
            print(f"FAILED: {e}")

    # Cache results
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"hpo_{method_name}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({
                "best_params": best_params,
                "best_val_mae": best_val_mae,
                "all_results": all_results,
            }, f, indent=2)

    return best_params, best_val_mae, all_results


def load_hpo_results(method_name, cache_dir):
    """Load cached HPO results."""
    cache_path = Path(cache_dir) / f"hpo_{method_name}.json"
    if not cache_path.exists():
        return None
    with open(cache_path, "r") as f:
        return json.load(f)
