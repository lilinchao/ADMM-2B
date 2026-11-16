"""Run low-rank-only ADMM ablation for Section 5.

The existing fault-stress results already contain the spatial-only baselines
(Masked-IDW and Kriging) and the full coupled variants (ADMM-IDW and
ADMM-Kriging). This script adds the missing low-rank-only component by setting
gamma=0 in ADMM, which disables the kriging/IDW contribution in the Z update.
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

from run_spatial_prior_fault_experiment import _prefill_faults_for_joint_model, run_admm  # noqa: E402
from src.data_split import make_test_mask, split_sensors  # noqa: E402
from src.dataset_config import add_dataset_arg, get_config, load_dataset  # noqa: E402
from src.hpo import load_hpo_results  # noqa: E402
from src.kriging import fit_variogram_from_tensor  # noqa: E402


MEDIUM_DEPLOY = {
    "guangzhou": 100,
    "pemsd7": 100,
    "epa_ca": 50,
}


def random_entry_fault(base_mask, fault_rate, seed):
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


def evaluate(X_raw, X_hat_norm, data_range, xmin, test_positions, true_fault):
    X_pred = X_hat_norm * data_range + xmin

    def metric(mask):
        if not mask.any():
            return float("nan"), float("nan")
        diff = X_raw[mask] - X_pred[mask]
        return float(np.abs(diff).mean()), float(np.sqrt((diff**2).mean()))

    mae_spatial, rmse_spatial = metric(test_positions)
    mae_fault, rmse_fault = metric(true_fault)
    mae_all, rmse_all = metric(test_positions | true_fault)
    return {
        "mae_all": mae_all,
        "rmse_all": rmse_all,
        "mae_spatial": mae_spatial,
        "rmse_spatial": rmse_spatial,
        "mae_fault": mae_fault,
        "rmse_fault": rmse_fault,
        "n_true_fault": int(true_fault.sum()),
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
    parser = argparse.ArgumentParser(description="ADMM low-rank-only ablation")
    add_dataset_arg(parser)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--fault-rate", type=float, default=0.30)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output_path = PROJECT_ROOT / (
        args.output or f"results/section_5_ablation/{args.dataset}_admm_cp_only.json"
    )

    print("Loading dataset...", flush=True)
    X_raw, sensor_coords, data_range, xmin, _ = load_dataset(PROJECT_ROOT, args.dataset)
    X_norm = (X_raw - xmin) / (data_range + 1e-8)
    I, _, _ = X_norm.shape

    print("Fitting variogram...", flush=True)
    variogram_params = fit_variogram_from_tensor(
        X_norm, np.ones(X_norm.shape, dtype=bool), sensor_coords
    )

    split = split_sensors(I, test_frac=0.3, val_frac_of_deploy=0.1, seed=42)
    n_deploy = MEDIUM_DEPLOY[args.dataset]

    hpo_cache_dir = str(PROJECT_ROOT / get_config(args.dataset)["hpo_cache_dir"])
    hpo = load_hpo_results("admm_2b", hpo_cache_dir)
    params = hpo["best_params"].copy() if hpo is not None else {}
    params["gamma"] = 0.0
    print(f"  ADMM-CP params: {params}", flush=True)

    payload = load_existing(output_path)
    payload.update({
        "experiment": "section_5_cp_only_ablation",
        "dataset": args.dataset,
        "shape": list(X_norm.shape),
        "scenario": "mask_perfect",
        "fault_rate": args.fault_rate,
        "n_deploy": n_deploy,
        "seed_start": args.seed_start,
        "n_seeds": args.n_seeds,
        "method": "admm_cp_only",
        "label": "ADMM-CP only",
        "max_iter": args.max_iter,
        "params": params,
        "variogram_params": variogram_params,
    })

    done = {int(row["seed"]) for row in payload["results"]}
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        if seed in done:
            print(f"Seed {seed} SKIP", flush=True)
            continue
        base_mask, test_positions, _ = make_test_mask(
            X_norm.shape, split, n_deploy, seed=seed
        )
        true_fault = random_entry_fault(base_mask, args.fault_rate, seed)
        provided_mask = base_mask.copy()
        provided_mask[true_fault] = False

        X_obs = X_norm.copy()
        X_obs[~provided_mask] = 0.0
        X_obs[~base_mask] = 0.0

        X_prefill, prefill_mask = _prefill_faults_for_joint_model(
            X_obs, provided_mask, base_mask
        )

        print(
            f"\nDataset={args.dataset} seed={seed} true_fault={true_fault.sum():,}",
            flush=True,
        )
        t0 = time.time()
        try:
            X_hat, info = run_admm(
                X_prefill,
                prefill_mask,
                sensor_coords,
                variogram_params,
                params,
                "idw",
                args.max_iter,
                seed,
            )
            elapsed = time.time() - t0
            row = {
                "dataset": args.dataset,
                "scenario": "mask_perfect",
                "seed": seed,
                "method": "admm_cp_only",
                "label": "ADMM-CP only",
                "elapsed": elapsed,
                **evaluate(X_raw, X_hat, data_range, xmin, test_positions, true_fault),
            }
            row["n_iter"] = int(info.get("n_iter", -1))
            row["gamma_used"] = float(info.get("gamma_used", float("nan")))
            row["primal_res_final"] = (
                float(info["primal_res"][-1]) if info.get("primal_res") else float("nan")
            )
            row["dual_res_final"] = (
                float(info["dual_res"][-1]) if info.get("dual_res") else float("nan")
            )
            payload["results"].append(row)
            done.add(seed)
            write_json_atomic(output_path, payload)
            print(
                f"  admm_cp_only      all={row['mae_all']:.3f} spatial={row['mae_spatial']:.3f} "
                f"fault={row['mae_fault']:.3f} elapsed={elapsed:.0f}s",
                flush=True,
            )
        except Exception as exc:
            elapsed = time.time() - t0
            payload["results"].append({
                "dataset": args.dataset,
                "scenario": "mask_perfect",
                "seed": seed,
                "method": "admm_cp_only",
                "label": "ADMM-CP only",
                "elapsed": elapsed,
                "error": str(exc),
            })
            done.add(seed)
            write_json_atomic(output_path, payload)
            print(f"  admm_cp_only FAILED after {elapsed:.0f}s: {exc}", flush=True)

    rows = [r for r in payload["results"] if "error" not in r]
    payload["summary"] = {
        metric: finite_stats([r.get(metric, float("nan")) for r in rows])
        for metric in [
            "mae_all",
            "rmse_all",
            "mae_spatial",
            "rmse_spatial",
            "mae_fault",
            "rmse_fault",
            "elapsed",
            "n_iter",
            "primal_res_final",
            "dual_res_final",
        ]
    }
    write_json_atomic(output_path, payload)
    print(f"\nSaved {output_path}", flush=True)


if __name__ == "__main__":
    main()
