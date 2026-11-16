"""Additional TKDE-oriented experiments.

This script is intended to run in the cloud project directory. It adds:
1. Stronger graph-regularized low-rank baselines.
2. Synthetic scalability measurements with runtime and memory.
3. SCLT-IDW parameter sensitivity for rank, spatial weight, neighbors, and mask noise.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_section_5_fault_stress import (  # noqa: E402
    apply_mask_noise,
    build_observed_tensor,
    evaluate,
    random_entry_fault,
)
from run_spatial_prior_fault_experiment import (  # noqa: E402
    _prefill_faults_for_joint_model,
    run_admm,
)
from src.data_split import make_test_mask, split_sensors  # noqa: E402
from src.dataset_config import get_config, load_dataset  # noqa: E402
from src.hpo import load_hpo_results  # noqa: E402


MEDIUM_DEPLOY = {"guangzhou": 100, "pemsd7": 100, "epa_ca": 50}
DATASETS = ["guangzhou", "pemsd7", "epa_ca"]


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def finite_stats(values):
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def load_normalized_dataset(dataset: str):
    x_raw, coords, data_range, xmin, _ = load_dataset(PROJECT_ROOT, dataset)
    x_norm = (x_raw - xmin) / (data_range + 1e-8)
    return x_raw, x_norm, coords, data_range, xmin


def graph_edges(coords, n_neighbors=10, power=1.0):
    from scipy.spatial import KDTree

    coords = np.asarray(coords, dtype=float)
    tree = KDTree(coords)
    src, dst, weights = [], [], []
    k_query = min(n_neighbors + 1, len(coords))
    for i in range(len(coords)):
        dists, idxs = tree.query(coords[i], k=k_query)
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)
        local = []
        for d, j in zip(dists, idxs):
            j = int(j)
            if j == i:
                continue
            w = float((max(float(d), 1e-6)) ** (-power))
            local.append((j, w))
        total = sum(w for _, w in local) or 1.0
        for j, w in local:
            src.append(i)
            dst.append(j)
            weights.append(w / total)
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64), np.asarray(weights, dtype=np.float32)


def run_graph_lrmf(
    x_obs,
    mask,
    coords,
    *,
    rank=12,
    n_neighbors=12,
    graph_lambda=0.2,
    ridge=1e-4,
    lr=0.05,
    epochs=450,
    robust=False,
    huber_delta=0.05,
    seed=0,
    device=None,
):
    import psutil
    import torch
    import torch.nn.functional as F

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    i_count, j_count, k_count = x_obs.shape
    t_count = j_count * k_count
    x_mat = torch.as_tensor(x_obs.reshape(i_count, t_count), dtype=torch.float32, device=device)
    m_mat = torch.as_tensor(mask.reshape(i_count, t_count), dtype=torch.bool, device=device)

    src, dst, ew = graph_edges(coords, n_neighbors=n_neighbors, power=1.0)
    src_t = torch.as_tensor(src, dtype=torch.long, device=device)
    dst_t = torch.as_tensor(dst, dtype=torch.long, device=device)
    ew_t = torch.as_tensor(ew, dtype=torch.float32, device=device)

    u = torch.nn.Parameter(torch.as_tensor(0.05 * rng.standard_normal((i_count, rank)), dtype=torch.float32, device=device))
    v = torch.nn.Parameter(torch.as_tensor(0.05 * rng.standard_normal((t_count, rank)), dtype=torch.float32, device=device))
    opt = torch.optim.Adam([u, v], lr=lr)
    proc = psutil.Process(os.getpid())
    max_rss = proc.memory_info().rss
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    for epoch in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        pred = u @ v.T
        residual = pred[m_mat] - x_mat[m_mat]
        if robust:
            data_loss = F.huber_loss(residual, torch.zeros_like(residual), delta=float(huber_delta), reduction="mean")
        else:
            data_loss = torch.mean(residual * residual)
        edge_diff = u[src_t] - u[dst_t]
        graph_loss = torch.mean(ew_t * torch.sum(edge_diff * edge_diff, dim=1))
        ridge_loss = torch.mean(u * u) + torch.mean(v * v)
        loss = data_loss + float(graph_lambda) * graph_loss + float(ridge) * ridge_loss
        loss.backward()
        opt.step()
        if epoch % 25 == 0:
            max_rss = max(max_rss, proc.memory_info().rss)

    with torch.no_grad():
        pred = torch.clamp(u @ v.T, 0.0, 1.0).detach().cpu().numpy().reshape(x_obs.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    max_rss = max(max_rss, proc.memory_info().rss)
    peak_gpu = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
    info = {"elapsed": elapsed, "peak_rss_mb": float(max_rss / (1024**2)), "peak_gpu_mb": peak_gpu, "epochs": int(epochs)}
    return pred, info


def run_graph_cp_tensor(
    x_obs,
    mask,
    coords,
    *,
    rank=12,
    n_neighbors=12,
    graph_lambda=0.2,
    ridge=1e-4,
    lr=0.04,
    epochs=450,
    robust=True,
    huber_delta=0.05,
    seed=0,
    device=None,
):
    import psutil
    import torch
    import torch.nn.functional as F

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed + 1009)
    torch.manual_seed(seed + 1009)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    i_count, j_count, k_count = x_obs.shape
    x_t = torch.as_tensor(x_obs, dtype=torch.float32, device=device)
    m_t = torch.as_tensor(mask, dtype=torch.bool, device=device)

    src, dst, ew = graph_edges(coords, n_neighbors=n_neighbors, power=1.0)
    src_t = torch.as_tensor(src, dtype=torch.long, device=device)
    dst_t = torch.as_tensor(dst, dtype=torch.long, device=device)
    ew_t = torch.as_tensor(ew, dtype=torch.float32, device=device)

    a = torch.nn.Parameter(torch.as_tensor(0.05 * rng.standard_normal((i_count, rank)), dtype=torch.float32, device=device))
    b = torch.nn.Parameter(torch.as_tensor(0.05 * rng.standard_normal((j_count, rank)), dtype=torch.float32, device=device))
    c = torch.nn.Parameter(torch.as_tensor(0.05 * rng.standard_normal((k_count, rank)), dtype=torch.float32, device=device))
    opt = torch.optim.Adam([a, b, c], lr=lr)
    proc = psutil.Process(os.getpid())
    max_rss = proc.memory_info().rss
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    for epoch in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        pred = torch.einsum("ir,jr,kr->ijk", a, b, c)
        residual = pred[m_t] - x_t[m_t]
        if robust:
            data_loss = F.huber_loss(residual, torch.zeros_like(residual), delta=float(huber_delta), reduction="mean")
        else:
            data_loss = torch.mean(residual * residual)
        edge_diff = a[src_t] - a[dst_t]
        graph_loss = torch.mean(ew_t * torch.sum(edge_diff * edge_diff, dim=1))
        ridge_loss = torch.mean(a * a) + torch.mean(b * b) + torch.mean(c * c)
        loss = data_loss + float(graph_lambda) * graph_loss + float(ridge) * ridge_loss
        loss.backward()
        opt.step()
        if epoch % 25 == 0:
            max_rss = max(max_rss, proc.memory_info().rss)

    with torch.no_grad():
        pred = torch.clamp(torch.einsum("ir,jr,kr->ijk", a, b, c), 0.0, 1.0).detach().cpu().numpy()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    max_rss = max(max_rss, proc.memory_info().rss)
    peak_gpu = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
    info = {"elapsed": elapsed, "peak_rss_mb": float(max_rss / (1024**2)), "peak_gpu_mb": peak_gpu, "epochs": int(epochs)}
    return pred, info


def best_sclt_params(dataset: str):
    hpo_cache = str(PROJECT_ROOT / get_config(dataset)["hpo_cache_dir"])
    hpo = load_hpo_results("admm_2b", hpo_cache)
    params = hpo["best_params"] if hpo is not None else {}
    out = {
        "rank": int(params.get("rank", 10)),
        "gamma": float(params.get("gamma", 1.0)),
        "beta": float(params.get("beta", 0.01)),
        "rho": float(params.get("rho", 0.1)),
        "n_neighbors": int(params.get("n_neighbors", params.get("K", 20))),
        "idw_power": float(params.get("idw_power", params.get("power", 1.0))),
    }
    return out


def run_sclt_idw(x_obs, provided_mask, base_mask, coords, params, max_iter, seed):
    import threading

    import psutil

    try:
        import torch
    except Exception:  # pragma: no cover - torch is available on the cloud server.
        torch = None

    proc = psutil.Process(os.getpid())
    peak_rss = [proc.memory_info().rss]
    stop = threading.Event()

    def _sample_memory():
        while not stop.is_set():
            peak_rss[0] = max(peak_rss[0], proc.memory_info().rss)
            time.sleep(0.05)

    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    x_prefill, prefill_mask = _prefill_faults_for_joint_model(x_obs, provided_mask, base_mask)
    sampler = threading.Thread(target=_sample_memory, daemon=True)
    sampler.start()
    t0 = time.time()
    try:
        pred, info = run_admm(
            x_prefill,
            prefill_mask,
            coords,
            variogram_params={},
            params=params,
            spatial_mode="idw",
            max_iter=max_iter,
            seed=seed,
        )
    finally:
        stop.set()
        sampler.join(timeout=1.0)
        peak_rss[0] = max(peak_rss[0], proc.memory_info().rss)
    if isinstance(info, dict):
        info = dict(info)
        info.setdefault("elapsed", time.time() - t0)
    else:
        info = {"elapsed": time.time() - t0}
    info["peak_rss_mb"] = float(peak_rss[0] / (1024**2))
    info["peak_gpu_mb"] = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch is not None and torch.cuda.is_available() else 0.0
    return pred, info


def make_fault_case(x_norm, split, n_deploy, seed, fn_rate=0.0, fault_rate=0.30):
    base_mask, test_positions, _ = make_test_mask(x_norm.shape, split, n_deploy, seed=seed)
    true_fault = random_entry_fault(base_mask, x_norm, fault_rate, seed)
    provided_mask, flagged_fault, false_negative, false_positive = apply_mask_noise(
        base_mask, true_fault, fn_rate=float(fn_rate), fp_rate=0.0, seed=seed
    )
    x_obs = build_observed_tensor(x_norm, base_mask, true_fault, provided_mask)
    return base_mask, provided_mask, x_obs, test_positions, true_fault, flagged_fault, false_negative, false_positive


def run_strong_baselines(args):
    out_path = PROJECT_ROOT / "results/tkde_extra/strong_baselines.json"
    payload = {"experiment": "strong_graph_baselines", "results": []}
    if out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        payload.setdefault("results", [])
    done = {(r["dataset"], r["scenario"], int(r["seed"]), r["method"]) for r in payload["results"]}
    methods = {
        "SCLT-IDW": {"type": "sclt"},
        "G-LRMF": {"robust": False},
        "R-G-LRMF": {"robust": True},
        "R-GTC": {"type": "graph_cp", "robust": True},
    }
    scenarios = {"perfect_flags": 0.0, "fn20": 0.20}

    for dataset in args.datasets:
        x_raw, x_norm, coords, data_range, xmin = load_normalized_dataset(dataset)
        split = split_sensors(x_norm.shape[0], test_frac=0.3, val_frac_of_deploy=0.1, seed=42)
        n_deploy = MEDIUM_DEPLOY[dataset]
        for scenario, fn_rate in scenarios.items():
            for seed in range(args.n_seeds):
                case = make_fault_case(x_norm, split, n_deploy, seed, fn_rate=fn_rate, fault_rate=args.fault_rate)
                base_mask, provided_mask, x_obs, test_positions, true_fault, flagged_fault, false_negative, false_positive = case
                for method, mcfg in methods.items():
                    key = (dataset, scenario, seed, method)
                    if key in done:
                        continue
                    if mcfg.get("type") == "sclt":
                        pred, info = run_sclt_idw(
                            x_obs,
                            provided_mask,
                            base_mask,
                            coords,
                            best_sclt_params(dataset),
                            max_iter=args.baseline_sclt_iter,
                            seed=seed,
                        )
                    elif mcfg.get("type") == "graph_cp":
                        pred, info = run_graph_cp_tensor(
                            x_obs,
                            provided_mask,
                            coords,
                            rank=args.graph_cp_rank,
                            n_neighbors=args.graph_neighbors,
                            graph_lambda=args.graph_lambda,
                            epochs=args.graph_cp_epochs,
                            robust=bool(mcfg["robust"]),
                            seed=seed,
                        )
                    else:
                        pred, info = run_graph_lrmf(
                            x_obs,
                            provided_mask,
                            coords,
                            rank=args.graph_rank,
                            n_neighbors=args.graph_neighbors,
                            graph_lambda=args.graph_lambda,
                            epochs=args.graph_epochs,
                            robust=bool(mcfg["robust"]),
                            seed=seed,
                        )
                    metrics = evaluate(
                        x_raw,
                        pred,
                        data_range,
                        xmin,
                        test_positions,
                        true_fault,
                        flagged_fault,
                        false_negative,
                        false_positive,
                    )
                    row = {
                        "dataset": dataset,
                        "scenario": scenario,
                        "seed": seed,
                        "method": method,
                        "fn_rate": fn_rate,
                        **metrics,
                        **info,
                    }
                    payload["results"].append(row)
                    done.add(key)
                    write_json(out_path, payload)
                    print(f"{dataset} {scenario} seed={seed} {method} spatial={row['mae_spatial']:.3f} fault={row['mae_fault']:.3f} time={row['elapsed']:.1f}s", flush=True)
    summarize_payload(payload)
    write_json(out_path, payload)


def summarize_payload(payload):
    groups = {}
    for row in payload.get("results", []):
        key = tuple((k, row.get(k)) for k in ["dataset", "scenario", "method"] if k in row)
        groups.setdefault(key, []).append(row)
    summary = []
    for key, rows in groups.items():
        record = {k: v for k, v in key}
        for metric in ["mae_all", "mae_spatial", "mae_fault", "rmse_all", "rmse_spatial", "rmse_fault", "elapsed", "peak_rss_mb", "peak_gpu_mb"]:
            record[metric] = finite_stats([r.get(metric, float("nan")) for r in rows])
        summary.append(record)
    payload["summary"] = summary


def synthetic_tensor(i_count, j_count, k_count, rank=5, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.random((i_count, 2))
    a = rng.normal(size=(i_count, rank))
    b = rng.normal(size=(j_count, rank))
    c = rng.normal(size=(k_count, rank))
    x = np.einsum("ir,jr,kr->ijk", a, b, c)
    x += 0.08 * rng.normal(size=x.shape)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x.astype(np.float64), coords.astype(np.float64)


def synthetic_masks(shape, observed_nodes, seed=0, fault_rate=0.10):
    i_count, j_count, k_count = shape
    rng = np.random.default_rng(seed)
    all_nodes = np.arange(i_count)
    n_test = max(1, int(0.25 * i_count))
    test_nodes = rng.choice(all_nodes, size=n_test, replace=False)
    pool = np.setdiff1d(all_nodes, test_nodes)
    obs_nodes = rng.choice(pool, size=min(observed_nodes, len(pool)), replace=False)
    base = np.zeros(shape, dtype=bool)
    base[obs_nodes, :, :] = True
    test_positions = np.zeros(shape, dtype=bool)
    test_positions[test_nodes, :, :] = True
    true_fault = random_entry_fault(base, np.zeros(shape), fault_rate, seed)
    provided = base.copy()
    provided[true_fault] = False
    return base, provided, test_positions, true_fault


def run_scalability(args):
    out_path = PROJECT_ROOT / "results/tkde_extra/scalability.json"
    payload = {"experiment": "synthetic_scalability", "results": []}
    if out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        payload.setdefault("results", [])
    done = {(int(r["nodes"]), r["method"]) for r in payload["results"]}
    sizes = [100, 200, 400, 800, 1200]
    for nodes in sizes:
        j_count, k_count = 30, 48
        x_norm, coords = synthetic_tensor(nodes, j_count, k_count, rank=5, seed=nodes)
        x_raw = x_norm.copy()
        observed_nodes = max(20, int(0.6 * nodes))
        base_mask, provided_mask, test_positions, true_fault = synthetic_masks(x_norm.shape, observed_nodes, seed=nodes, fault_rate=0.10)
        x_obs = x_norm.copy()
        x_obs[true_fault] = 0.0
        x_obs[~provided_mask] = 0.0
        false_mask = np.zeros_like(base_mask, dtype=bool)
        methods = args.scalability_methods
        for method in methods:
            if (nodes, method) in done:
                continue
            t0 = time.time()
            if method == "SCLT-IDW":
                params = {"rank": 10, "gamma": 1.0, "beta": 0.01, "rho": 0.1, "n_neighbors": 10, "idw_power": 1.0}
                pred, info = run_sclt_idw(x_obs, provided_mask, base_mask, coords, params, max_iter=args.scalability_iter, seed=nodes)
                info = {"elapsed": time.time() - t0, **{k: v for k, v in info.items() if isinstance(v, (int, float))}}
            else:
                pred, info = run_graph_lrmf(
                    x_obs,
                    provided_mask,
                    coords,
                    rank=12,
                    n_neighbors=12,
                    graph_lambda=args.graph_lambda,
                    epochs=args.scalability_epochs,
                    robust=True,
                    seed=nodes,
                )
            metrics = evaluate(x_raw, pred, 1.0, 0.0, test_positions, true_fault, true_fault, false_mask, false_mask)
            row = {
                "nodes": nodes,
                "time_slots": j_count * k_count,
                "entries": int(np.prod(x_norm.shape)),
                "observed_nodes": observed_nodes,
                "method": method,
                **metrics,
                **info,
            }
            payload["results"].append(row)
            done.add((nodes, method))
            write_json(out_path, payload)
            print(f"scale nodes={nodes} {method} time={row['elapsed']:.1f}s spatial={row['mae_spatial']:.4f} fault={row['mae_fault']:.4f}", flush=True)
    write_json(out_path, payload)


def run_sensitivity(args):
    out_path = PROJECT_ROOT / "results/tkde_extra/sensitivity.json"
    payload = {"experiment": "sclt_idw_sensitivity", "results": []}
    if out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        payload.setdefault("results", [])
    done = {(r["dataset"], r["group"], str(r["value"]), int(r["seed"])) for r in payload["results"]}
    dataset = args.sensitivity_dataset
    x_raw, x_norm, coords, data_range, xmin = load_normalized_dataset(dataset)
    split = split_sensors(x_norm.shape[0], test_frac=0.3, val_frac_of_deploy=0.1, seed=42)
    base_params = best_sclt_params(dataset)
    groups = {
        "rank": [5, 10, 15, 20],
        "gamma": [0.1, 1.0, 10.0, 50.0, 100.0],
        "neighbors": [5, 10, 20, 30],
        "mask_fn": [0.0, 0.05, 0.10, 0.20, 0.30],
    }
    for group, values in groups.items():
        for value in values:
            for seed in range(args.n_seeds):
                key = (dataset, group, str(value), seed)
                if key in done:
                    continue
                fn_rate = float(value) if group == "mask_fn" else 0.0
                case = make_fault_case(x_norm, split, MEDIUM_DEPLOY[dataset], seed, fn_rate=fn_rate, fault_rate=args.fault_rate)
                base_mask, provided_mask, x_obs, test_positions, true_fault, flagged_fault, false_negative, false_positive = case
                params = dict(base_params)
                if group == "rank":
                    params["rank"] = int(value)
                elif group == "gamma":
                    params["gamma"] = float(value)
                elif group == "neighbors":
                    params["n_neighbors"] = int(value)
                pred, info = run_sclt_idw(x_obs, provided_mask, base_mask, coords, params, max_iter=args.sensitivity_iter, seed=seed)
                metrics = evaluate(x_raw, pred, data_range, xmin, test_positions, true_fault, flagged_fault, false_negative, false_positive)
                row = {
                    "dataset": dataset,
                    "group": group,
                    "value": value,
                    "seed": seed,
                    "params": params,
                    **metrics,
                    "elapsed": float(info.get("elapsed", float("nan"))) if isinstance(info, dict) else float("nan"),
                    "n_iter": int(info.get("n_iter", -1)) if isinstance(info, dict) else -1,
                }
                payload["results"].append(row)
                done.add(key)
                write_json(out_path, payload)
                print(f"sensitivity {group}={value} seed={seed} spatial={row['mae_spatial']:.3f} fault={row['mae_fault']:.3f}", flush=True)
    summarize_payload(payload)
    write_json(out_path, payload)


def main():
    parser = argparse.ArgumentParser(description="Run TKDE extra experiments")
    parser.add_argument("--mode", choices=["baselines", "scalability", "sensitivity", "all"], default="all")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--fault-rate", type=float, default=0.30)
    parser.add_argument("--graph-rank", type=int, default=12)
    parser.add_argument("--graph-neighbors", type=int, default=12)
    parser.add_argument("--graph-lambda", type=float, default=0.2)
    parser.add_argument("--graph-epochs", type=int, default=450)
    parser.add_argument("--graph-cp-rank", type=int, default=12)
    parser.add_argument("--graph-cp-epochs", type=int, default=450)
    parser.add_argument("--baseline-sclt-iter", type=int, default=120)
    parser.add_argument("--sensitivity-dataset", default="pemsd7")
    parser.add_argument("--sensitivity-iter", type=int, default=120)
    parser.add_argument("--scalability-iter", type=int, default=60)
    parser.add_argument("--scalability-epochs", type=int, default=220)
    parser.add_argument("--scalability-methods", nargs="+", choices=["SCLT-IDW", "R-G-LRMF"],
                        default=["SCLT-IDW", "R-G-LRMF"])
    args = parser.parse_args()

    if args.mode in {"baselines", "all"}:
        run_strong_baselines(args)
    if args.mode in {"scalability", "all"}:
        run_scalability(args)
    if args.mode in {"sensitivity", "all"}:
        run_sensitivity(args)


if __name__ == "__main__":
    main()
