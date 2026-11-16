"""
Spatial-prior and coupling study for ADMM-2B.

This experiment is designed for the paper's revised Results section. It
compares IDW vs. variogram kriging as the spatial prior, and one-shot
two-stage pipelines vs. ADMM coupling under mixed missingness.

Outputs are saved as JSON and do not overwrite the existing experiment files.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.admm_joint import admm_two_block
from src.data_split import make_test_mask, split_sensors
from src.dataset_config import add_dataset_arg, default_output_path, get_config, load_dataset
from src.hpo import load_hpo_results
from src.interpolation_baselines import linear_time
from src.kriging import SpatioTemporalKriging, fit_variogram_from_tensor
from src.two_stage import two_stage_kriging_then_td


def _finite_stats(values):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def _build_fault_mask(base_mask, fault_rate, seed):
    mask = base_mask.copy()
    if fault_rate <= 0:
        return mask

    _, J, K = mask.shape
    rng = np.random.default_rng(seed + 1000)
    deployed_sensor_ids = np.where(base_mask[:, 0, 0])[0]
    n_fault = max(1, int(fault_rate * J * K))
    for sensor_id in deployed_sensor_ids:
        fault_j = rng.choice(J, size=n_fault, replace=True)
        fault_k = rng.choice(K, size=n_fault, replace=True)
        mask[sensor_id, fault_j, fault_k] = False
    return mask


def _prefill_faults_for_joint_model(X_obs, mask, base_mask):
    """Fill deployed-sensor faults but keep undeployed sensors unobserved."""
    X_filled = X_obs.copy()
    X_filled[~mask] = 0.0
    interp = linear_time(X_filled.astype(float), mask)
    X_alg = interp.copy()
    X_alg[~base_mask] = 0.0
    return X_alg, base_mask


def run_idw_spatial_only(X_obs, mask, sensor_coords, params):
    K_nn = int(params.get("K", params.get("n_neighbors", 10)))
    power = float(params.get("power", params.get("idw_power", 2)))
    observed_ids = np.where(mask[:, 0, 0])[0]
    unobserved_ids = np.where(~mask[:, 0, 0])[0]
    if len(observed_ids) == 0 or len(unobserved_ids) == 0:
        return X_obs.copy()

    from scipy.spatial import KDTree

    tree = KDTree(sensor_coords[observed_ids])
    k_use = min(K_nn, len(observed_ids))
    X_hat = X_obs.copy()
    for uid in unobserved_ids:
        dists, idxs = tree.query(sensor_coords[uid], k=k_use)
        if k_use == 1:
            dists = np.array([dists])
            idxs = np.array([idxs])
        weights = 1.0 / (np.maximum(dists, 1e-10) ** power)
        weights /= weights.sum()
        X_hat[uid, :, :] = np.einsum("n,njk->jk", weights, X_obs[observed_ids[idxs], :, :])
    return X_hat


def run_idw_masked_spatial_only(X_obs, mask, sensor_coords, params):
    """Entry-wise IDW that respects the full observed-entry mask.

    Unlike run_idw_spatial_only(), this baseline recomputes the available
    neighbors for each day/time entry and excludes fault-marked deployed
    readings from the spatial average.
    """
    K_nn = int(params.get("K", params.get("n_neighbors", 10)))
    power = float(params.get("power", params.get("idw_power", 2)))
    I, J, K = X_obs.shape
    X_hat = X_obs.copy()

    from scipy.spatial import KDTree

    for j in range(J):
        for k in range(K):
            observed_ids = np.where(mask[:, j, k])[0]
            target_ids = np.where(~mask[:, j, k])[0]
            if len(observed_ids) == 0 or len(target_ids) == 0:
                continue

            k_use = min(K_nn, len(observed_ids))
            tree = KDTree(sensor_coords[observed_ids])
            dists, idxs = tree.query(sensor_coords[target_ids], k=k_use)
            if k_use == 1:
                dists = dists[:, None]
                idxs = idxs[:, None]

            weights = 1.0 / (np.maximum(dists, 1e-10) ** power)
            weights /= weights.sum(axis=1, keepdims=True)
            neighbor_values = X_obs[observed_ids[idxs], j, k]
            X_hat[target_ids, j, k] = np.sum(weights * neighbor_values, axis=1)

    return X_hat


def run_ordinary_kriging(X_obs, mask, sensor_coords, variogram_params, params):
    n_neighbors = int(params.get("n_neighbors", 20))
    krig = SpatioTemporalKriging(
        sensor_coords=sensor_coords,
        variogram_params=variogram_params,
        n_neighbors=n_neighbors,
        kriging_mode="variogram",
        tau_j=float(params.get("tau_j", 3.0)),
        tau_k=float(params.get("tau_k", 1.0)),
    )
    krig.fit(X_obs, mask)
    return krig.predict_full(X_obs, mask)


def run_two_stage(X_alg, alg_mask, sensor_coords, variogram_params, params, spatial_mode, max_iter):
    n_neighbors = int(params.get("n_neighbors", params.get("K", 20)))
    return two_stage_kriging_then_td(
        X_alg,
        alg_mask,
        locs=None,
        kriging_params={},
        rank=int(params.get("rank", 10)),
        beta=float(params.get("beta", 0.01)),
        max_iter=max_iter,
        n_neighbors=n_neighbors,
        n_restarts=1,
        sensor_coords=sensor_coords,
        variogram_params=variogram_params if spatial_mode == "variogram" else None,
        kriging_mode=spatial_mode,
        tau_j=float(params.get("tau_j", 3.0)),
        tau_k=float(params.get("tau_k", 1.0)),
    )


def run_admm(X_alg, alg_mask, sensor_coords, variogram_params, params, spatial_mode, max_iter, seed):
    kwargs = {
        "rank": int(params.get("rank", 10)),
        "gamma": float(params.get("gamma", 1.0)),
        "beta": float(params.get("beta", 0.01)),
        "rho": float(params.get("rho", 0.1)),
        "max_iter": max_iter,
        "tol": 1e-4,
        "n_neighbors": int(params.get("n_neighbors", params.get("K", 20))),
        "seed": seed,
        "sensor_coords": sensor_coords,
        "variogram_params": variogram_params if spatial_mode == "variogram" else None,
        "kriging_mode": spatial_mode,
        "idw_power": float(params.get("idw_power", params.get("power", 1))),
        "idw_spatial_only": spatial_mode == "idw",
        "tau_j": float(params.get("tau_j", 3.0)),
        "tau_k": float(params.get("tau_k", 1.0)),
        "adaptive_gamma": False,
        "log_convergence": True,
    }
    X_hat, info = admm_two_block(X_alg, alg_mask, locs=None, kriging_params={}, **kwargs)
    return X_hat, info


def evaluate_prediction(X_raw, X_hat_norm, data_range, xmin, test_positions, deployed_fault):
    X_pred = X_hat_norm * data_range + xmin
    out = {}

    eval_positions = test_positions | deployed_fault
    diff_all = X_raw[eval_positions] - X_pred[eval_positions]
    out["mae_all"] = float(np.abs(diff_all).mean())
    out["rmse_all"] = float(np.sqrt((diff_all ** 2).mean()))

    diff_test = X_raw[test_positions] - X_pred[test_positions]
    out["mae_spatial"] = float(np.abs(diff_test).mean())
    out["rmse_spatial"] = float(np.sqrt((diff_test ** 2).mean()))

    if deployed_fault.any():
        diff_fault = X_raw[deployed_fault] - X_pred[deployed_fault]
        out["mae_fault"] = float(np.abs(diff_fault).mean())
        out["rmse_fault"] = float(np.sqrt((diff_fault ** 2).mean()))
    else:
        out["mae_fault"] = float("nan")
        out["rmse_fault"] = float("nan")
    return out


def main():
    parser = argparse.ArgumentParser(description="Spatial prior and coupling study under faults")
    add_dataset_arg(parser)
    parser.add_argument("--methods", nargs="+", default=[
        "idw",
        "ordinary_kriging",
        "two_stage_idw",
        "two_stage_kriging",
        "admm_idw",
        "admm_kriging",
    ])
    parser.add_argument("--deploy-numbers", nargs="+", type=int, default=None)
    parser.add_argument("--fault-rates", nargs="+", type=float, default=[0.0, 0.1, 0.3])
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--hpo-cache", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dcfg = get_config(args.dataset)
    output_path = PROJECT_ROOT / (
        args.output or default_output_path(args.dataset, "results/spatial_prior_fault_study.json")
    )
    hpo_cache_dir = str(PROJECT_ROOT / (args.hpo_cache or dcfg["hpo_cache_dir"]))

    if args.deploy_numbers is None:
        args.deploy_numbers = [50] if args.dataset == "epa_ca" else [100]

    print("Loading dataset...")
    X_raw, sensor_coords, data_range, xmin, _ = load_dataset(PROJECT_ROOT, args.dataset)
    X_norm = (X_raw - xmin) / (data_range + 1e-8)
    I, J, K = X_norm.shape

    print("Fitting variogram...")
    variogram_params = fit_variogram_from_tensor(
        X_norm, np.ones(X_norm.shape, dtype=bool), sensor_coords
    )
    print("  variogram:", variogram_params)

    split = split_sensors(I, test_frac=0.3, val_frac_of_deploy=0.1, seed=42)
    print(" ", split.summary())

    best_params = {}
    for method in ["idw", "ordinary_kriging", "admm_2b"]:
        hpo = load_hpo_results(method, hpo_cache_dir)
        best_params[method] = hpo["best_params"] if hpo is not None else {}
        print(f"  HPO {method}: {best_params[method]}")

    method_params = {
        "idw": best_params["idw"],
        "idw_masked": best_params["idw"],
        "ordinary_kriging": best_params["ordinary_kriging"],
        "two_stage_idw": best_params["admm_2b"],
        "two_stage_kriging": best_params["admm_2b"],
        "two_stage_idw_raw": best_params["admm_2b"],
        "two_stage_kriging_raw": best_params["admm_2b"],
        "admm_idw": best_params["admm_2b"],
        "admm_kriging": best_params["admm_2b"],
        "admm_idw_raw": best_params["admm_2b"],
        "admm_kriging_raw": best_params["admm_2b"],
    }

    results = {
        m: {str(n): {str(fr): [] for fr in args.fault_rates} for n in args.deploy_numbers}
        for m in args.methods
    }

    for n_deploy in args.deploy_numbers:
        for fault_rate in args.fault_rates:
            for seed in range(args.seed_start, args.seed_start + args.n_seeds):
                base_mask, test_positions, _ = make_test_mask(
                    X_norm.shape, split, n_deploy, seed=seed
                )
                mask = _build_fault_mask(base_mask, fault_rate, seed)
                X_obs = X_norm.copy()
                X_obs[~mask] = 0.0
                # Only injected faults on deployed sensors count as fault
                # positions. Sensors outside the sampled deployment set are not
                # evaluated as faults.
                deployed_fault = base_mask & (~mask)

                X_prefill, prefill_mask = _prefill_faults_for_joint_model(
                    X_obs, mask, base_mask
                )

                print(
                    f"\nDataset={args.dataset} n={n_deploy} fault={fault_rate:.0%} "
                    f"seed={seed} obs={mask.sum():,} test={test_positions.sum():,} "
                    f"fault={deployed_fault.sum():,}",
                    flush=True,
                )

                for method in args.methods:
                    params = method_params.get(method, {}).copy()
                    t0 = time.time()
                    info = {}
                    try:
                        if method == "idw":
                            X_hat = run_idw_spatial_only(X_obs, mask, sensor_coords, params)
                        elif method == "idw_masked":
                            X_hat = run_idw_masked_spatial_only(
                                X_obs, mask, sensor_coords, params
                            )
                        elif method == "ordinary_kriging":
                            X_hat = run_ordinary_kriging(
                                X_obs, mask, sensor_coords, variogram_params, params
                            )
                        elif method == "two_stage_idw":
                            X_hat = run_two_stage(
                                X_prefill, prefill_mask, sensor_coords, variogram_params,
                                params, "idw", args.max_iter
                            )
                        elif method == "two_stage_kriging":
                            X_hat = run_two_stage(
                                X_prefill, prefill_mask, sensor_coords, variogram_params,
                                params, "variogram", args.max_iter
                            )
                        elif method == "two_stage_idw_raw":
                            X_hat = run_two_stage(
                                X_obs, mask, sensor_coords, variogram_params,
                                params, "idw", args.max_iter
                            )
                        elif method == "two_stage_kriging_raw":
                            X_hat = run_two_stage(
                                X_obs, mask, sensor_coords, variogram_params,
                                params, "variogram", args.max_iter
                            )
                        elif method == "admm_idw":
                            X_hat, info = run_admm(
                                X_prefill, prefill_mask, sensor_coords, variogram_params,
                                params, "idw", args.max_iter, seed
                            )
                        elif method == "admm_kriging":
                            X_hat, info = run_admm(
                                X_prefill, prefill_mask, sensor_coords, variogram_params,
                                params, "variogram", args.max_iter, seed
                            )
                        elif method == "admm_idw_raw":
                            X_hat, info = run_admm(
                                X_obs, mask, sensor_coords, variogram_params,
                                params, "idw", args.max_iter, seed
                            )
                        elif method == "admm_kriging_raw":
                            X_hat, info = run_admm(
                                X_obs, mask, sensor_coords, variogram_params,
                                params, "variogram", args.max_iter, seed
                            )
                        else:
                            raise ValueError(f"Unknown method: {method}")

                        elapsed = time.time() - t0
                        row = {
                            "seed": seed,
                            "elapsed": elapsed,
                            **evaluate_prediction(
                                X_raw, X_hat, data_range, xmin,
                                test_positions, deployed_fault
                            ),
                        }
                        if info:
                            row["n_iter"] = int(info.get("n_iter", -1))
                            row["gamma_used"] = float(info.get("gamma_used", float("nan")))
                            row["primal_res_final"] = (
                                float(info["primal_res"][-1]) if info.get("primal_res") else float("nan")
                            )
                            row["dual_res_final"] = (
                                float(info["dual_res"][-1]) if info.get("dual_res") else float("nan")
                            )
                        results[method][str(n_deploy)][str(fault_rate)].append(row)
                        print(
                            f"  {method:<18s} all={row['mae_all']:.3f} "
                            f"spatial={row['mae_spatial']:.3f} fault={row['mae_fault']:.3f} "
                            f"elapsed={elapsed:.0f}s",
                            flush=True,
                        )
                    except Exception as exc:
                        elapsed = time.time() - t0
                        results[method][str(n_deploy)][str(fault_rate)].append({
                            "seed": seed,
                            "elapsed": elapsed,
                            "error": str(exc),
                            "mae_all": float("nan"),
                            "rmse_all": float("nan"),
                            "mae_spatial": float("nan"),
                            "rmse_spatial": float("nan"),
                            "mae_fault": float("nan"),
                            "rmse_fault": float("nan"),
                        })
                        print(f"  {method:<18s} FAILED after {elapsed:.0f}s: {exc}", flush=True)

    summary = {}
    for method in args.methods:
        summary[method] = {}
        for n_deploy in args.deploy_numbers:
            summary[method][str(n_deploy)] = {}
            for fault_rate in args.fault_rates:
                rows = results[method][str(n_deploy)][str(fault_rate)]
                summary[method][str(n_deploy)][str(fault_rate)] = {
                    metric: _finite_stats([r.get(metric, float("nan")) for r in rows])
                    for metric in ["mae_all", "mae_spatial", "mae_fault", "rmse_all", "rmse_spatial", "rmse_fault"]
                }

    output = {
        "experiment": "spatial_prior_fault_study",
        "dataset": args.dataset,
        "shape": [I, J, K],
        "data_range": data_range,
        "deploy_numbers": args.deploy_numbers,
        "fault_rates": args.fault_rates,
        "n_seeds": args.n_seeds,
        "max_iter": args.max_iter,
        "methods": args.methods,
        "best_params": best_params,
        "variogram_params": variogram_params,
        "results": results,
        "summary": summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
