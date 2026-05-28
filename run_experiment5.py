"""
Experiment 5: Mixed missingness — spatial extrapolation + temporal faults.

Real-world sensor networks face BOTH spatial gaps (undeployed locations)
AND temporal gaps (sensor failures). This experiment shows ADMM-2B's
superiority in this realistic combined scenario.

Methods: IDW, Linear, CP-TD, HaLRTC, Ordinary Kriging, ADMM-2B
(+ DL baselines: BRITS, SAITS)
Variables: fault_rate x n_deploy x 3 seeds
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
from src.kriging import SpatioTemporalKriging, fit_variogram_from_tensor
from src.cp_als import cp_als_best_of
from src.halrtc import halrtc
from src.admm_joint import admm_two_block
from src.interpolation_baselines import linear_time
from src.baselines import run_brits, run_saits


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


def run_ordinary_kriging(X_obs, mask, sensor_coords=None, variogram_params=None, max_iter=100, **params):
    n_neighbors = params.get("n_neighbors", 20)
    krig = SpatioTemporalKriging(
        sensor_coords=sensor_coords,
        n_neighbors=n_neighbors, kriging_mode="variogram",
        tau_j=3.0, tau_k=1.0,
    )
    krig.fit(X_obs, mask)
    return krig.predict_full(X_obs, mask)


def run_admm_2b(X_obs, mask, sensor_coords=None, variogram_params=None, max_iter=200, seed=0,
                base_mask=None, **params):
    rank = params.get("rank", 10)
    gamma = params.get("gamma", 1.0)
    n_neighbors = params.get("n_neighbors", 20)
    beta = params.get("beta", 0.01)
    rho = params.get("rho", 1.0)
    idw_power = params.get("idw_power", 1)
    tau_j = params.get("tau_j", 3.0)
    tau_k = params.get("tau_k", 1.0)

    # When base_mask is provided (spatial-only missing, without faults),
    # pre-fill fault positions with linear-time interpolation so CP-TD
    # sees complete deployed sensor data.
    if base_mask is not None:
        from src.interpolation_baselines import linear_time as _linear_time
        X_filled = X_obs.copy()
        X_filled[~mask] = 0.0
        Z_interp = _linear_time(X_filled.astype(float), mask)
        X_admm = Z_interp.copy()
        X_admm[~base_mask] = 0.0
        admm_mask = base_mask
    else:
        X_admm = X_obs
        admm_mask = mask

    X_hat, info = admm_two_block(
        X_admm, admm_mask, sensor_coords, {}, rank=rank, gamma=gamma, beta=beta, rho=rho,
        max_iter=max_iter, tol=1e-4, n_neighbors=n_neighbors, seed=seed,
        sensor_coords=sensor_coords, variogram_params=variogram_params,
        kriging_mode="idw", idw_spatial_only=True, idw_power=idw_power, tau_j=tau_j, tau_k=tau_k,
        adaptive_gamma=False,
    )
    return X_hat


METHOD_REGISTRY = {
    "idw":              {"fn": run_idw,              "label": "IDW"},
    "linear_time":      {"fn": run_linear_time_fn,   "label": "Linear (time)"},
    "cp_td":            {"fn": run_cp_td,            "label": "CP-TD"},
    "halrtc":           {"fn": run_halrtc_fn,        "label": "HaLRTC"},
    "ordinary_kriging": {"fn": run_ordinary_kriging, "label": "Ordinary Kriging"},
    "admm_2b":          {"fn": run_admm_2b,          "label": "ADMM-2B (ours)"},
    "brits":            {"fn": run_brits,            "label": "BRITS"},
    "saits":            {"fn": run_saits,            "label": "SAITS"},
}


def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Mixed missingness")
    add_dataset_arg(parser)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to evaluate (default: all)")
    parser.add_argument("--deploy-numbers", nargs="+", type=int, default=None,
                        help="Number of deployed sensors (default: per-dataset)")
    parser.add_argument("--fault-rates", nargs="+", type=float,
                        default=[0.0, 0.05, 0.10, 0.20, 0.30],
                        help="Fault rates on deployed sensors")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--hpo-cache", type=str, default=None,
                        help="HPO cache directory (default: per-dataset)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: per-dataset)")
    parser.add_argument("--dl-cache-dir", type=str, default=None,
                        help="Directory for caching DL model weights (default: per-dataset)")
    args = parser.parse_args()

    dcfg = get_config(args.dataset)
    method_names = args.methods or list(METHOD_REGISTRY.keys())
    output_path = PROJECT_ROOT / (args.output or default_output_path(args.dataset, "results/experiment5.json"))
    hpo_cache_dir = str(PROJECT_ROOT / (args.hpo_cache or dcfg["hpo_cache_dir"]))
    dl_cache_dir = str(PROJECT_ROOT / (args.dl_cache_dir or dcfg["dl_cache_dir"]))
    deploy_numbers = args.deploy_numbers or dcfg["default_deploy_numbers"][:4]  # use first 4
    dl_methods = {"brits", "saits"}

    # Load data
    print("Loading dataset...")
    X_raw, sensor_coords, data_range, xmin, _ = load_dataset(PROJECT_ROOT, args.dataset)
    X_norm = (X_raw - xmin) / (data_range + 1e-8)
    I, J, K = X_norm.shape

    full_mask = np.ones(X_norm.shape, dtype=bool)
    variogram_params = fit_variogram_from_tensor(X_norm, full_mask, sensor_coords)

    split = split_sensors(I, test_frac=0.3, val_frac_of_deploy=0.1, seed=42)
    print(f"  {split.summary()}")

    # Load HPO results
    best_params = {}
    for m in method_names:
        hpo_res = load_hpo_results(m, hpo_cache_dir)
        if hpo_res is not None:
            best_params[m] = hpo_res["best_params"]
        else:
            best_params[m] = {}

    # Run experiment
    all_results = {m: {n: {f: [] for f in args.fault_rates}
                       for n in deploy_numbers}
                   for m in method_names}

    for n_deploy in deploy_numbers:
        for fault_rate in args.fault_rates:
            for seed in range(args.n_seeds):
                # Base mask: spatial block missing
                base_mask, test_positions, deployed = make_test_mask(
                    X_norm.shape, split, n_deploy, seed=seed
                )
                mask = base_mask.copy()

                # Add random faults on deployed sensors
                if fault_rate > 0:
                    rng = np.random.default_rng(seed + 1000)
                    deployed_sensor_ids = np.where(mask[:, 0, 0])[0]
                    for s_id in deployed_sensor_ids:
                        n_fault = max(1, int(fault_rate * J * K))
                        fault_j = rng.choice(J, size=n_fault, replace=True)
                        fault_k = rng.choice(K, size=n_fault, replace=True)
                        mask[s_id, fault_j, fault_k] = False

                X_obs = X_norm.copy()
                X_obs[~mask] = 0.0

                n_test_pts = test_positions.sum()
                deployed_fault = ~mask & ~test_positions.reshape(X_norm.shape)

                print(f"\n  n={n_deploy} fault={fault_rate:.0%} seed={seed}: "
                      f"obs={mask.sum():,} test={n_test_pts:,} fault={deployed_fault.sum():,}")

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
                        if m == "admm_2b" and fault_rate > 0:
                            extra["base_mask"] = base_mask
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

                        # Evaluate on all missing, test only, and fault only
                        X_pred_raw = X_hat * data_range + xmin
                        diff_all = X_raw[~mask] - X_pred_raw[~mask]
                        mae_all = float(np.abs(diff_all).mean())
                        rmse_all = float(np.sqrt((diff_all ** 2).mean()))
                        diff_test = X_raw[test_positions] - X_pred_raw[test_positions]
                        mae_test = float(np.abs(diff_test).mean())
                        rmse_test = float(np.sqrt((diff_test ** 2).mean()))
                        if deployed_fault.any():
                            diff_fault = X_raw[deployed_fault] - X_pred_raw[deployed_fault]
                            mae_fault = float(np.abs(diff_fault).mean())
                            rmse_fault = float(np.sqrt((diff_fault ** 2).mean()))
                        else:
                            mae_fault = float("nan")
                            rmse_fault = float("nan")

                        all_results[m][n_deploy][fault_rate].append({
                            "seed": seed,
                            "mae_all": mae_all, "rmse_all": rmse_all,
                            "mae_test": mae_test, "rmse_test": rmse_test,
                            "mae_fault": mae_fault, "rmse_fault": rmse_fault,
                            "elapsed": elapsed,
                        })
                        print(f"MAE: all={mae_all:.3f} test={mae_test:.3f} fault={mae_fault:.3f} | "
                              f"RMSE: all={rmse_all:.3f} test={rmse_test:.3f} fault={rmse_fault:.3f} ({elapsed:.0f}s)")
                    except Exception as e:
                        elapsed = time.time() - t0
                        all_results[m][n_deploy][fault_rate].append({
                            "seed": seed,
                            "mae_all": float("nan"), "rmse_all": float("nan"),
                            "mae_test": float("nan"), "rmse_test": float("nan"),
                            "mae_fault": float("nan"), "rmse_fault": float("nan"),
                            "elapsed": elapsed,
                            "error": str(e),
                        })
                        print(f"FAILED: {e}")

    # Print results tables
    for metric_key, metric_label in [("mae_all", "MAE (all missing)"),
                                      ("rmse_all", "RMSE (all missing)"),
                                      ("mae_test", "MAE (spatial only)"),
                                      ("rmse_test", "RMSE (spatial only)"),
                                      ("mae_fault", "MAE (temporal faults)"),
                                      ("rmse_fault", "RMSE (temporal faults)")]:
        print(f"\n{'='*80}")
        print(f"  {metric_label}")
        print(f"{'='*80}")
        for n_deploy in deploy_numbers:
            print(f"\n  n_deploy={n_deploy}")
            header = f"  {'Method':<18s}"
            for f in args.fault_rates:
                header += f" | fault={f:.0%}       "
            print(header)
            print("-" * len(header))
            for m in method_names:
                mlabel = METHOD_REGISTRY[m]["label"]
                row = f"  {mlabel:<18s}"
                for f in args.fault_rates:
                    vals = [v[metric_key] for v in all_results[m][n_deploy][f]
                            if v.get(metric_key) is not None and v[metric_key] == v[metric_key]]
                    if vals:
                        mean_v = np.mean(vals)
                        std_v = np.std(vals)
                        row += f" | {mean_v:>5.2f}±{std_v:<5.2f}"
                    else:
                        row += f" |   N/A       "
                print(row)

    # Save
    output_data = {
        "experiment": "experiment5_mixed_missingness",
        "dataset": args.dataset,
        "shape": [I, J, K],
        "data_range": data_range,
        "best_params": best_params,
        "deploy_numbers": deploy_numbers,
        "fault_rates": args.fault_rates,
        "n_seeds": args.n_seeds,
        "results": {},
    }
    for m in method_names:
        output_data["results"][m] = {}
        for n in deploy_numbers:
            output_data["results"][m][str(n)] = {}
            for f in args.fault_rates:
                output_data["results"][m][str(n)][f"{f}"] = all_results[m][n_deploy][f]

    # Fix: use correct n_deploy for each entry
    for m in method_names:
        output_data["results"][m] = {}
        for n in deploy_numbers:
            output_data["results"][m][str(n)] = {}
            for f in args.fault_rates:
                output_data["results"][m][str(n)][f"{f}"] = all_results[m][n][f]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f_out:
        json.dump(output_data, f_out, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
