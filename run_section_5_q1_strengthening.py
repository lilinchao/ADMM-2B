"""Run strengthened zero-observation experiments for Section 5.1.

This script focuses on the spatial-prior methods under the medium deployment
setting. It adds two pieces of evidence that are missing from the original
Section 5.1 results:

1. More random seeds for paired significance tests.
2. Per-test-sensor errors with nearest-deployed-sensor distance, plus a harder
   clustered holdout protocol.

The script is intended to run on the cloud project directory where ``src/`` and
the datasets are available.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_experiment1 import (  # noqa: E402
    run_admm_2b,
    run_idw,
    run_linear_time_fn,
    run_ordinary_kriging,
)
from src.data_split import SensorSplit, make_test_mask, split_sensors  # noqa: E402
from src.dataset_config import add_dataset_arg, get_config, load_dataset  # noqa: E402
from src.hpo import load_hpo_results  # noqa: E402
from src.kriging import fit_variogram_from_tensor  # noqa: E402


MEDIUM_DEPLOY = {
    "guangzhou": 100,
    "pemsd7": 100,
    "epa_ca": 50,
}

METHODS = {
    "admm_2b": ("ADMM-IDW", run_admm_2b),
    "idw": ("IDW", run_idw),
    "ordinary_kriging": ("Kriging", run_ordinary_kriging),
    "linear_time": ("Linear Time", run_linear_time_fn),
}


def finite_mean(values):
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def finite_rmse(values):
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(arr**2)))


def make_cluster_split(n_sensors, sensor_coords, test_frac=0.3, val_frac_of_deploy=0.1, seed=0):
    """Reserve one contiguous spatial cluster as the zero-observation test set."""
    rng = np.random.default_rng(seed + 20260530)
    all_sensors = np.arange(n_sensors)
    n_test = max(1, int(n_sensors * test_frac))

    center = int(rng.choice(all_sensors))
    dists = np.linalg.norm(sensor_coords - sensor_coords[center], axis=1)
    test_sensors = np.argsort(dists)[:n_test]
    deploy_pool = np.setdiff1d(all_sensors, test_sensors)

    n_val = max(1, int(len(deploy_pool) * val_frac_of_deploy))
    val_sensors = rng.choice(deploy_pool, size=n_val, replace=False)
    train_sensors = np.setdiff1d(deploy_pool, val_sensors)

    split = SensorSplit(
        train=np.sort(train_sensors),
        val=np.sort(val_sensors),
        test=np.sort(test_sensors),
        deploy_pool=np.sort(deploy_pool),
    )
    meta = {
        "cluster_center": center,
        "cluster_radius": float(dists[test_sensors].max()),
        "cluster_mean_radius": float(dists[test_sensors].mean()),
    }
    return split, meta


def evaluate_prediction(X_raw, X_hat_norm, data_range, xmin, test_sensors, sensor_coords, deployed_sensors):
    X_pred = X_hat_norm * data_range + xmin
    diff = X_raw[test_sensors, :, :] - X_pred[test_sensors, :, :]

    per_sensor_mae = np.mean(np.abs(diff), axis=(1, 2))
    per_sensor_rmse = np.sqrt(np.mean(diff**2, axis=(1, 2)))

    deployed_coords = sensor_coords[deployed_sensors]
    test_coords = sensor_coords[test_sensors]
    nearest_dist = np.sqrt(((test_coords[:, None, :] - deployed_coords[None, :, :]) ** 2).sum(axis=2)).min(axis=1)

    return {
        "mae": float(per_sensor_mae.mean()),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "per_sensor": [
            {
                "sensor_id": int(sensor_id),
                "nearest_deployed_dist": float(dist),
                "mae": float(mae),
                "rmse": float(rmse),
            }
            for sensor_id, dist, mae, rmse in zip(test_sensors, nearest_dist, per_sensor_mae, per_sensor_rmse)
        ],
    }


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_existing(path):
    if not path.exists():
        return {"results": [], "per_sensor": []}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("results", [])
    data.setdefault("per_sensor", [])
    return data


def main():
    parser = argparse.ArgumentParser(description="Strengthened zero-observation spatial-prior experiments")
    add_dataset_arg(parser)
    parser.add_argument("--holdout-mode", choices=["random", "cluster"], default="random")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--methods", nargs="+", default=["admm_2b", "idw", "ordinary_kriging", "linear_time"])
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dcfg = get_config(args.dataset)
    n_deploy = MEDIUM_DEPLOY[args.dataset]
    output_path = PROJECT_ROOT / (
        args.output
        or f"results/section_5_q1_strengthening/{args.dataset}_{args.holdout_mode}.json"
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

    hpo_cache_dir = str(PROJECT_ROOT / dcfg["hpo_cache_dir"])
    best_params = {}
    for method in args.methods:
        hpo_key = method
        hpo = load_hpo_results(hpo_key, hpo_cache_dir)
        best_params[method] = hpo["best_params"] if hpo is not None else {}
        print(f"  HPO {method}: {best_params[method]}", flush=True)

    payload = load_existing(output_path)
    payload.update({
        "experiment": "section_5_q1_strengthening",
        "dataset": args.dataset,
        "shape": list(X_norm.shape),
        "holdout_mode": args.holdout_mode,
        "n_deploy": n_deploy,
        "seed_start": args.seed_start,
        "n_seeds": args.n_seeds,
        "methods": args.methods,
        "best_params": best_params,
        "variogram_params": variogram_params,
    })

    done = {
        (row["seed"], row["method"])
        for row in payload["results"]
        if row.get("holdout_mode") == args.holdout_mode
    }

    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        if args.holdout_mode == "random":
            split = split_sensors(I, test_frac=0.3, val_frac_of_deploy=0.1, seed=42)
            split_meta = {"cluster_center": None, "cluster_radius": None, "cluster_mean_radius": None}
        else:
            split, split_meta = make_cluster_split(
                I, sensor_coords, test_frac=0.3, val_frac_of_deploy=0.1, seed=seed
            )

        mask, test_positions, deployed_sensors = make_test_mask(
            X_norm.shape, split, n_deploy, seed=seed
        )
        test_sensors = np.asarray(split.test, dtype=int)

        X_obs = X_norm.copy()
        X_obs[~mask] = 0.0

        print(
            f"\nDataset={args.dataset} holdout={args.holdout_mode} seed={seed} "
            f"n_deploy={len(deployed_sensors)} n_test={len(test_sensors)}",
            flush=True,
        )

        for method in args.methods:
            if (seed, method) in done:
                print(f"  {method:<18s} SKIP", flush=True)
                continue
            label, fn = METHODS[method]
            params = best_params.get(method, {}).copy()
            t0 = time.time()
            try:
                X_hat = fn(
                    X_obs,
                    mask,
                    sensor_coords=sensor_coords,
                    variogram_params=variogram_params,
                    max_iter=args.max_iter,
                    seed=seed,
                    **params,
                )
                elapsed = time.time() - t0
                eval_out = evaluate_prediction(
                    X_raw,
                    X_hat,
                    data_range,
                    xmin,
                    test_sensors,
                    sensor_coords,
                    deployed_sensors,
                )
                row = {
                    "dataset": args.dataset,
                    "holdout_mode": args.holdout_mode,
                    "seed": seed,
                    "method": method,
                    "label": label,
                    "n_deploy": int(len(deployed_sensors)),
                    "n_test_sensors": int(len(test_sensors)),
                    "mae": eval_out["mae"],
                    "rmse": eval_out["rmse"],
                    "elapsed": elapsed,
                    **split_meta,
                }
                payload["results"].append(row)
                for sensor_row in eval_out["per_sensor"]:
                    payload["per_sensor"].append({
                        "dataset": args.dataset,
                        "holdout_mode": args.holdout_mode,
                        "seed": seed,
                        "method": method,
                        "label": label,
                        **sensor_row,
                    })
                done.add((seed, method))
                write_json_atomic(output_path, payload)
                print(
                    f"  {method:<18s} MAE={row['mae']:.3f} RMSE={row['rmse']:.3f} "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )
            except Exception as exc:
                elapsed = time.time() - t0
                payload["results"].append({
                    "dataset": args.dataset,
                    "holdout_mode": args.holdout_mode,
                    "seed": seed,
                    "method": method,
                    "label": label,
                    "n_deploy": int(len(deployed_sensors)),
                    "n_test_sensors": int(len(test_sensors)),
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "elapsed": elapsed,
                    "error": str(exc),
                    **split_meta,
                })
                done.add((seed, method))
                write_json_atomic(output_path, payload)
                print(f"  {method:<18s} FAILED after {elapsed:.0f}s: {exc}", flush=True)

    write_json_atomic(output_path, payload)
    print(f"\nSaved {output_path}", flush=True)


if __name__ == "__main__":
    main()
