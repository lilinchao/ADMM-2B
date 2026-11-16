"""Run Section 5 fault-stress experiments.

This script adds two reviewer-oriented robustness checks:

1. Imperfect invalid-entry masks: a fraction of true faults is left unflagged,
   so methods receive corrupted values as if they were valid observations.
2. Structured faults: faults are bursty, sensor-level, spatially clustered, or
   value-dependent rather than independent random entries.

The script runs at the medium deployment density and is intended for the cloud
project directory where ``src/`` and the datasets are available.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_spatial_prior_fault_experiment import (  # noqa: E402
    _prefill_faults_for_joint_model,
    run_admm,
    run_idw_masked_spatial_only,
)
from src.data_split import make_test_mask, split_sensors  # noqa: E402
from src.dataset_config import add_dataset_arg, get_config, load_dataset  # noqa: E402
from src.hpo import load_hpo_results  # noqa: E402
from src.kriging import SpatioTemporalKriging, fit_variogram_from_tensor  # noqa: E402


MEDIUM_DEPLOY = {
    "guangzhou": 100,
    "pemsd7": 100,
    "epa_ca": 50,
}

SCENARIOS = {
    "mask_perfect": {"family": "mask_noise", "pattern": "random_entry", "fn_rate": 0.0, "fp_rate": 0.0},
    "mask_fn05": {"family": "mask_noise", "pattern": "random_entry", "fn_rate": 0.05, "fp_rate": 0.0},
    "mask_fn10": {"family": "mask_noise", "pattern": "random_entry", "fn_rate": 0.10, "fp_rate": 0.0},
    "mask_fn20": {"family": "mask_noise", "pattern": "random_entry", "fn_rate": 0.20, "fp_rate": 0.0},
    "mask_fp10": {"family": "mask_noise", "pattern": "random_entry", "fn_rate": 0.0, "fp_rate": 0.10},
    "random_entry": {"family": "structured", "pattern": "random_entry", "fn_rate": 0.0, "fp_rate": 0.0},
    "burst": {"family": "structured", "pattern": "burst", "fn_rate": 0.0, "fp_rate": 0.0},
    "sensor_outage": {"family": "structured", "pattern": "sensor_outage", "fn_rate": 0.0, "fp_rate": 0.0},
    "spatial_cluster": {"family": "structured", "pattern": "spatial_cluster", "fn_rate": 0.0, "fp_rate": 0.0},
    "high_value": {"family": "structured", "pattern": "high_value", "fn_rate": 0.0, "fp_rate": 0.0},
}

METHOD_LABELS = {
    "idw_masked": "Masked-IDW",
    "ordinary_kriging": "Kriging",
    "admm_idw": "ADMM-IDW",
    "admm_kriging": "ADMM-Kriging",
}


def finite_stats(values):
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def random_entry_fault(base_mask, X_norm, fault_rate, seed, sensor_coords=None):
    true_fault = np.zeros_like(base_mask, dtype=bool)
    _, J, K = base_mask.shape
    rng = np.random.default_rng(seed + 1000)
    deployed = np.where(base_mask[:, 0, 0])[0]
    n_fault = max(1, int(fault_rate * J * K))
    for sensor_id in deployed:
        flat = rng.choice(J * K, size=n_fault, replace=False)
        jj, kk = np.unravel_index(flat, (J, K))
        true_fault[sensor_id, jj, kk] = True
    return true_fault


def burst_fault(base_mask, X_norm, fault_rate, seed, sensor_coords=None):
    true_fault = np.zeros_like(base_mask, dtype=bool)
    _, J, K = base_mask.shape
    rng = np.random.default_rng(seed + 2000)
    deployed = np.where(base_mask[:, 0, 0])[0]
    total = J * K
    length = max(1, int(fault_rate * total))
    for sensor_id in deployed:
        start = int(rng.integers(0, max(1, total - length + 1)))
        flat = np.arange(start, start + length)
        jj, kk = np.unravel_index(flat, (J, K))
        true_fault[sensor_id, jj, kk] = True
    return true_fault


def sensor_outage_fault(base_mask, X_norm, fault_rate, seed, sensor_coords=None):
    true_fault = np.zeros_like(base_mask, dtype=bool)
    rng = np.random.default_rng(seed + 3000)
    deployed = np.where(base_mask[:, 0, 0])[0]
    n_fault_sensors = max(1, int(round(fault_rate * len(deployed))))
    fault_sensors = rng.choice(deployed, size=n_fault_sensors, replace=False)
    true_fault[fault_sensors, :, :] = True
    return true_fault


def spatial_cluster_fault(base_mask, X_norm, fault_rate, seed, sensor_coords):
    true_fault = np.zeros_like(base_mask, dtype=bool)
    rng = np.random.default_rng(seed + 4000)
    deployed = np.where(base_mask[:, 0, 0])[0]
    n_fault_sensors = max(1, int(round(fault_rate * len(deployed))))
    center = int(rng.choice(deployed))
    dists = np.linalg.norm(sensor_coords[deployed] - sensor_coords[center], axis=1)
    fault_sensors = deployed[np.argsort(dists)[:n_fault_sensors]]
    true_fault[fault_sensors, :, :] = True
    return true_fault


def high_value_fault(base_mask, X_norm, fault_rate, seed, sensor_coords=None):
    true_fault = np.zeros_like(base_mask, dtype=bool)
    _, J, K = base_mask.shape
    deployed = np.where(base_mask[:, 0, 0])[0]
    n_fault = max(1, int(fault_rate * J * K))
    for sensor_id in deployed:
        values = X_norm[sensor_id].reshape(-1)
        # Stable tie-breaking keeps the setting deterministic while preserving
        # the value-dependent MNAR structure.
        order = np.lexsort((np.arange(values.size), values))
        flat = order[-n_fault:]
        jj, kk = np.unravel_index(flat, (J, K))
        true_fault[sensor_id, jj, kk] = True
    return true_fault


PATTERN_BUILDERS = {
    "random_entry": random_entry_fault,
    "burst": burst_fault,
    "sensor_outage": sensor_outage_fault,
    "spatial_cluster": spatial_cluster_fault,
    "high_value": high_value_fault,
}


def apply_mask_noise(base_mask, true_fault, fn_rate, fp_rate, seed):
    rng = np.random.default_rng(seed + 5000)
    flagged_fault = true_fault.copy()

    true_idx = np.flatnonzero(true_fault)
    n_fn = int(round(fn_rate * len(true_idx)))
    false_negative = np.zeros_like(true_fault, dtype=bool)
    if n_fn > 0 and len(true_idx) > 0:
        fn_idx = rng.choice(true_idx, size=min(n_fn, len(true_idx)), replace=False)
        false_negative.flat[fn_idx] = True
        flagged_fault.flat[fn_idx] = False

    false_positive = np.zeros_like(true_fault, dtype=bool)
    valid_normal = base_mask & (~true_fault)
    normal_idx = np.flatnonzero(valid_normal)
    n_fp = int(round(fp_rate * len(true_idx)))
    if n_fp > 0 and len(normal_idx) > 0:
        fp_idx = rng.choice(normal_idx, size=min(n_fp, len(normal_idx)), replace=False)
        false_positive.flat[fp_idx] = True

    provided_mask = base_mask.copy()
    provided_mask[flagged_fault | false_positive] = False
    return provided_mask, flagged_fault, false_negative, false_positive


def build_observed_tensor(X_norm, base_mask, true_fault, provided_mask, corrupt_value=0.0):
    X_obs = X_norm.copy()
    X_obs[true_fault] = corrupt_value
    X_obs[~provided_mask] = 0.0
    X_obs[~base_mask] = 0.0
    return X_obs


def run_ordinary_kriging_limited(X_obs, mask, sensor_coords, variogram_params, params, n_jobs):
    krig = SpatioTemporalKriging(
        sensor_coords=sensor_coords,
        variogram_params=variogram_params,
        n_neighbors=int(params.get("n_neighbors", 20)),
        kriging_mode="variogram",
        tau_j=float(params.get("tau_j", 3.0)),
        tau_k=float(params.get("tau_k", 1.0)),
        n_jobs=n_jobs,
    )
    krig.fit(X_obs, mask)
    return krig.predict_full(X_obs, mask)


def evaluate(X_raw, X_hat_norm, data_range, xmin, test_positions, true_fault, flagged_fault, false_negative, false_positive):
    X_pred = X_hat_norm * data_range + xmin

    def metric(mask):
        if not mask.any():
            return float("nan"), float("nan")
        diff = X_raw[mask] - X_pred[mask]
        return float(np.abs(diff).mean()), float(np.sqrt((diff**2).mean()))

    mae_spatial, rmse_spatial = metric(test_positions)
    mae_fault, rmse_fault = metric(true_fault)
    mae_flagged, rmse_flagged = metric(flagged_fault)
    mae_unflagged, rmse_unflagged = metric(false_negative)
    mae_fp, rmse_fp = metric(false_positive)
    mae_all, rmse_all = metric(test_positions | true_fault)

    return {
        "mae_all": mae_all,
        "rmse_all": rmse_all,
        "mae_spatial": mae_spatial,
        "rmse_spatial": rmse_spatial,
        "mae_fault": mae_fault,
        "rmse_fault": rmse_fault,
        "mae_flagged_fault": mae_flagged,
        "rmse_flagged_fault": rmse_flagged,
        "mae_unflagged_fault": mae_unflagged,
        "rmse_unflagged_fault": rmse_unflagged,
        "mae_false_positive": mae_fp,
        "rmse_false_positive": rmse_fp,
        "n_true_fault": int(true_fault.sum()),
        "n_flagged_fault": int(flagged_fault.sum()),
        "n_unflagged_fault": int(false_negative.sum()),
        "n_false_positive": int(false_positive.sum()),
    }


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_existing(path):
    if not path.exists():
        return {"results": []}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("results", [])
    return data


def main():
    parser = argparse.ArgumentParser(description="Section 5 fault-stress experiments")
    add_dataset_arg(parser)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--fault-rate", type=float, default=0.30)
    parser.add_argument("--methods", nargs="+", default=["idw_masked", "ordinary_kriging", "admm_idw", "admm_kriging"])
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--kriging-jobs", type=int, default=12)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]
    output_path = PROJECT_ROOT / (
        args.output
        or f"results/section_5_fault_stress/{args.dataset}_{args.scenario}.json"
    )

    print("Loading dataset...", flush=True)
    X_raw, sensor_coords, data_range, xmin, _ = load_dataset(PROJECT_ROOT, args.dataset)
    X_norm = (X_raw - xmin) / (data_range + 1e-8)
    I, _, _ = X_norm.shape

    print("Fitting variogram...", flush=True)
    variogram_params = fit_variogram_from_tensor(
        X_norm, np.ones(X_norm.shape, dtype=bool), sensor_coords
    )
    print("  variogram:", variogram_params, flush=True)

    split = split_sensors(I, test_frac=0.3, val_frac_of_deploy=0.1, seed=42)
    n_deploy = MEDIUM_DEPLOY[args.dataset]
    print(" ", split.summary(), flush=True)

    hpo_cache_dir = str(PROJECT_ROOT / get_config(args.dataset)["hpo_cache_dir"])
    best_params = {}
    for method in ["idw", "ordinary_kriging", "admm_2b"]:
        hpo = load_hpo_results(method, hpo_cache_dir)
        best_params[method] = hpo["best_params"] if hpo is not None else {}
        print(f"  HPO {method}: {best_params[method]}", flush=True)

    method_params = {
        "idw_masked": best_params["idw"],
        "ordinary_kriging": best_params["ordinary_kriging"],
        "admm_idw": best_params["admm_2b"],
        "admm_kriging": best_params["admm_2b"],
    }

    payload = load_existing(output_path)
    payload.update({
        "experiment": "section_5_fault_stress",
        "dataset": args.dataset,
        "shape": list(X_norm.shape),
        "scenario": args.scenario,
        "scenario_config": scenario,
        "fault_rate": args.fault_rate,
        "n_deploy": n_deploy,
        "seed_start": args.seed_start,
        "n_seeds": args.n_seeds,
        "methods": args.methods,
        "max_iter": args.max_iter,
        "best_params": best_params,
        "variogram_params": variogram_params,
    })

    done = {(int(r["seed"]), r["method"]) for r in payload["results"]}
    builder = PATTERN_BUILDERS[scenario["pattern"]]

    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        base_mask, test_positions, deployed_sensors = make_test_mask(
            X_norm.shape, split, n_deploy, seed=seed
        )
        true_fault = builder(base_mask, X_norm, args.fault_rate, seed, sensor_coords)
        provided_mask, flagged_fault, false_negative, false_positive = apply_mask_noise(
            base_mask,
            true_fault,
            fn_rate=float(scenario.get("fn_rate", 0.0)),
            fp_rate=float(scenario.get("fp_rate", 0.0)),
            seed=seed,
        )
        X_obs = build_observed_tensor(X_norm, base_mask, true_fault, provided_mask)
        X_prefill, prefill_mask = _prefill_faults_for_joint_model(
            X_obs, provided_mask, base_mask
        )

        print(
            f"\nDataset={args.dataset} scenario={args.scenario} seed={seed} "
            f"true_fault={true_fault.sum():,} flagged={flagged_fault.sum():,} "
            f"unflagged={false_negative.sum():,} false_positive={false_positive.sum():,}",
            flush=True,
        )

        for method in args.methods:
            if (seed, method) in done:
                print(f"  {method:<18s} SKIP", flush=True)
                continue
            params = method_params[method].copy()
            info = {}
            t0 = time.time()
            try:
                if method == "idw_masked":
                    X_hat = run_idw_masked_spatial_only(
                        X_obs, provided_mask, sensor_coords, params
                    )
                elif method == "ordinary_kriging":
                    X_hat = run_ordinary_kriging_limited(
                        X_obs, provided_mask, sensor_coords, variogram_params, params, args.kriging_jobs
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
                else:
                    raise ValueError(f"Unknown method: {method}")

                elapsed = time.time() - t0
                row = {
                    "dataset": args.dataset,
                    "scenario": args.scenario,
                    "scenario_family": scenario["family"],
                    "pattern": scenario["pattern"],
                    "fn_rate": float(scenario.get("fn_rate", 0.0)),
                    "fp_rate": float(scenario.get("fp_rate", 0.0)),
                    "seed": seed,
                    "method": method,
                    "label": METHOD_LABELS[method],
                    "elapsed": elapsed,
                    **evaluate(
                        X_raw, X_hat, data_range, xmin, test_positions,
                        true_fault, flagged_fault, false_negative, false_positive
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
                payload["results"].append(row)
                done.add((seed, method))
                write_json_atomic(output_path, payload)
                print(
                    f"  {method:<18s} all={row['mae_all']:.3f} spatial={row['mae_spatial']:.3f} "
                    f"fault={row['mae_fault']:.3f} unflagged={row['mae_unflagged_fault']:.3f} "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )
            except Exception as exc:
                elapsed = time.time() - t0
                payload["results"].append({
                    "dataset": args.dataset,
                    "scenario": args.scenario,
                    "scenario_family": scenario["family"],
                    "pattern": scenario["pattern"],
                    "fn_rate": float(scenario.get("fn_rate", 0.0)),
                    "fp_rate": float(scenario.get("fp_rate", 0.0)),
                    "seed": seed,
                    "method": method,
                    "label": METHOD_LABELS.get(method, method),
                    "elapsed": elapsed,
                    "error": str(exc),
                    "mae_all": float("nan"),
                    "rmse_all": float("nan"),
                    "mae_spatial": float("nan"),
                    "rmse_spatial": float("nan"),
                    "mae_fault": float("nan"),
                    "rmse_fault": float("nan"),
                })
                done.add((seed, method))
                write_json_atomic(output_path, payload)
                print(f"  {method:<18s} FAILED after {elapsed:.0f}s: {exc}", flush=True)

    summary = {}
    for method in args.methods:
        rows = [r for r in payload["results"] if r.get("method") == method and "error" not in r]
        summary[method] = {
            metric: finite_stats([r.get(metric, float("nan")) for r in rows])
            for metric in [
                "mae_all",
                "mae_spatial",
                "mae_fault",
                "mae_flagged_fault",
                "mae_unflagged_fault",
                "mae_false_positive",
                "elapsed",
            ]
        }
    payload["summary"] = summary
    write_json_atomic(output_path, payload)
    print(f"\nSaved {output_path}", flush=True)


if __name__ == "__main__":
    main()
