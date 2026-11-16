"""Archive-only operations are local; experiment execution requires explicit cloud confirmation."""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reproduction.build_tables import build
from reproduction.verify_archive import verify

DATASETS = ("guangzhou", "pemsd7", "epa_ca")
GROUPS = ("zero", "holdouts", "faults", "density", "ablation", "scalability", "sensitivity", "timeseries")
SHAPES = {"guangzhou": (214, 61, 144), "pemsd7": (228, 96, 44), "epa_ca": (101, 24, 92)}
DIRS = {"guangzhou": "", "pemsd7": "PeMSD7", "epa_ca": "EPA_CA"}
COORDS = {"guangzhou": "sensor_coords_mds.npy", "pemsd7": "sensor_coords.npy", "epa_ca": "sensor_coords.npy"}
CACHE = {"guangzhou": "hpo_cache", "pemsd7": "hpo_cache_pemsd7", "epa_ca": "hpo_cache_epa_ca"}
MEDIUM = {"guangzhou": 100, "pemsd7": 100, "epa_ca": 50}
DENSITIES = {"guangzhou": [20, 50, 100, 125, 150], "pemsd7": [20, 50, 100, 130, 160], "epa_ca": [10, 25, 50, 60, 71]}


def job_plan(datasets=DATASETS, groups=GROUPS):
    jobs = []

    def add(name, group, argv, expected):
        jobs.append({"name": name, "group": group, "argv": [str(x) for x in argv], "expected": expected})

    for ds in datasets:
        if "zero" in groups:
            out = f"results/zero/{ds}.json"
            add(f"zero-{ds}", "zero", ["run_experiment1.py", "--dataset", ds,
                "--deploy-numbers", MEDIUM[ds], "--n-seeds", 3, "--max-iter", 200,
                "--output", out], out)
        if "holdouts" in groups:
            for mode in ("random", "cluster"):
                out = f"results/section_5_q1_strengthening/{ds}_{mode}.json"
                add(f"holdout-{ds}-{mode}", "holdouts", ["run_section_5_q1_strengthening.py",
                    "--dataset", ds, "--holdout-mode", mode, "--n-seeds", 10, "--max-iter", 200], out)
        if "faults" in groups:
            for scenario in ("mask_perfect", "mask_fn05", "mask_fn10", "mask_fn20", "mask_fp10",
                             "burst", "sensor_outage", "spatial_cluster", "high_value"):
                out = f"results/section_5_fault_stress/{ds}_{scenario}.json"
                add(f"fault-{ds}-{scenario}", "faults", ["run_section_5_fault_stress.py",
                    "--dataset", ds, "--scenario", scenario, "--n-seeds", 5,
                    "--max-iter", 200, "--kriging-jobs", 1,
                    "--methods", "idw_masked", "ordinary_kriging", "admm_idw"], out)
        if "density" in groups:
            for n in DENSITIES[ds]:
                out = f"results/density/{ds}_{n}.json"
                add(f"density-{ds}-{n}", "density", ["run_spatial_prior_fault_experiment.py",
                    "--dataset", ds, "--deploy-numbers", n, "--fault-rates",
                    0, .05, .1, .15, .2, .25, .3, "--n-seeds", 3, "--max-iter", 200,
                    "--methods", "idw_masked", "ordinary_kriging", "admm_idw", "--output", out], out)
        if "ablation" in groups:
            out = f"results/section_5_ablation/{ds}_admm_cp_only.json"
            add(f"ablation-{ds}", "ablation", ["run_section_5_cp_only_ablation.py",
                "--dataset", ds, "--n-seeds", 5, "--max-iter", 200], out)
    if "scalability" in groups:
        add("scalability", "scalability", ["run_tkde_extra_experiments.py", "--mode", "scalability",
            "--scalability-methods", "SCLT-IDW", "--scalability-iter", 60], "results/tkde_extra/scalability.json")
    if "sensitivity" in groups and "pemsd7" in datasets:
        add("sensitivity", "sensitivity", ["run_tkde_extra_experiments.py", "--mode", "sensitivity",
            "--sensitivity-dataset", "pemsd7", "--n-seeds", 3, "--sensitivity-iter", 120],
            "results/tkde_extra/sensitivity.json")
    if "timeseries" in groups and "pemsd7" in datasets:
        add("timeseries", "timeseries", ["generate_zero_observation_timeseries_detail.py",
            "--dataset", "pemsd7", "--seed", 0, "--max-iter", 200],
            "results/zero_observation_timeseries_detail/pemsd7_meta.json")
    return jobs


def preflight(data_dir, datasets, groups, inspect_data=True):
    report = {"data_root": str(data_dir), "errors": [], "files": [],
              "original_data_checksums_available": False}
    required = ("numpy", "scipy", "joblib", "matplotlib", "psutil")
    if "zero" in groups:
        required += ("torch", "pypots")
    for package in required:
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["errors"].append(f"Missing dependency: {package}")
    need_data = any(g != "scalability" for g in groups)
    for ds in datasets if need_data else ():
        for name in ("tensor.mat", COORDS[ds]):
            path = data_dir / DIRS[ds] / name
            if not path.is_file():
                report["errors"].append(f"Missing data: {path}")
                continue
            report["files"].append({"dataset": ds, "file": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
        if inspect_data and not report["errors"]:
            import numpy as np
            from scipy.io import loadmat
            values = loadmat(data_dir / DIRS[ds] / "tensor.mat")["tensor"]
            coords = np.load(data_dir / DIRS[ds] / COORDS[ds], allow_pickle=False)
            if tuple(values.shape) != SHAPES[ds]:
                report["errors"].append(f"Unexpected tensor shape for {ds}: {values.shape}")
            if coords.shape != (SHAPES[ds][0], 2) or not np.isfinite(coords).all():
                report["errors"].append(f"Invalid coordinates for {ds}")
            if not np.isfinite(values).all():
                report["errors"].append(f"Non-finite source data for {ds}; explicit missing-ground-truth handling is required")
    report["ready"] = not report["errors"]
    return report


def write_parameter_cache(workspace, datasets):
    config = json.loads((REPO / "configs/paper_parameters.json").read_text())
    for ds in datasets:
        dst = workspace / "results" / CACHE[ds]
        dst.mkdir(parents=True, exist_ok=True)
        for method, params in config["datasets"][ds]["best_params"].items():
            obj = {"best_params": params, "best_val_mae": float("nan"), "all_results": [],
                   "provenance": "Recovered selected parameters; historical HPO scores/search history were not recovered."}
            (dst / f"hpo_{method}.json").write_text(json.dumps(obj, indent=2) + "\n")


def result_errors(value):
    found = []
    if isinstance(value, dict):
        if value.get("error"):
            found.append(str(value["error"]))
        for metric in ("mae", "mae_all", "mae_spatial"):
            if metric in value and isinstance(value[metric], (int, float)) and not math.isfinite(value[metric]):
                found.append(f"Non-finite primary metric: {metric}")
        for child in value.values():
            found.extend(result_errors(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(result_errors(child))
    return found


def count_result_rows(value):
    if isinstance(value, dict):
        if any(isinstance(value.get(k), (float, int)) for k in ("mae", "mae_all", "mae_spatial")):
            return 1
        return sum(count_result_rows(v) for v in value.values())
    if isinstance(value, list):
        return sum(count_result_rows(v) for v in value)
    return 0


def validate_result(group, value):
    errors = result_errors(value)
    if not isinstance(value, dict):
        return errors + ["Expected a result object"]
    if group == "timeseries":
        if not value.get("cases") or not value.get("shape"):
            errors.append("Incomplete time-series metadata")
    else:
        expected = {"zero": 24, "holdouts": 40, "faults": 15, "density": 63,
                    "ablation": 5, "scalability": 5, "sensitivity": 54}[group]
        count = count_result_rows(value.get("results"))
        if count != expected:
            errors.append(f"Incomplete result records: expected {expected}, found {count}")
    return errors


def cloud_run(args):
    if not args.confirm_cloud or platform.system() != "Linux":
        raise SystemExit("Experiments require Linux on a cloud server and --confirm-cloud. No model was run.")
    if not args.acknowledge_legacy_protocol:
        raise SystemExit("Read docs/REPRODUCIBILITY.md and pass --acknowledge-legacy-protocol to replay historical behavior.")
    jobs = job_plan(args.datasets, args.groups)
    if not jobs:
        raise SystemExit("No jobs selected.")
    check = preflight(args.data_dir, args.datasets, args.groups)
    if not check["ready"]:
        raise SystemExit(json.dumps(check, indent=2))
    if args.workers < 1 or args.blas_threads < 1:
        raise SystemExit("Worker/thread counts must be positive.")
    workspace = args.output_dir.resolve()
    # A fresh workspace prevents legacy resume/skip logic from treating cached outputs as reruns.
    workspace.mkdir(parents=True, exist_ok=False)
    shutil.copytree(REPO / "src", workspace / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for path in REPO.glob("*.py"):
        shutil.copy2(path, workspace / path.name)
    (workspace / "data").mkdir()
    if any(j["group"] != "scalability" for j in jobs):
        for ds in args.datasets:
            dst = workspace / "data" / DIRS[ds]
            dst.mkdir(parents=True, exist_ok=True)
            for name in ("tensor.mat", COORDS[ds]):
                shutil.copy2(args.data_dir / DIRS[ds] / name, dst / name)
    write_parameter_cache(workspace, args.datasets)
    env = os.environ.copy()
    env.update({k: str(args.blas_threads) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env["PYTHONHASHSEED"] = "0"
    env["MPLBACKEND"] = "Agg"
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "platform": platform.platform(), "workers": args.workers, "blas_threads": args.blas_threads,
              "jobs": jobs, "preflight": check,
              "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    (workspace / "run_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    logs = workspace / "logs"
    logs.mkdir()

    def execute(job):
        with (logs / f"{job['name']}.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable, *job["argv"]], cwd=workspace, env=env,
                                  stdout=log, stderr=subprocess.STDOUT)
        output = workspace / job["expected"]
        errors = []
        if proc.returncode:
            errors.append(f"Exit status {proc.returncode}")
        if not output.is_file():
            errors.append(f"Missing expected output: {job['expected']}")
        else:
            try:
                errors.extend(validate_result(job["group"], json.loads(output.read_text())))
            except (ValueError, OSError) as exc:
                errors.append(str(exc))
        return {"name": job["name"], "errors": errors, "ok": not errors}

    outcomes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            outcome = future.result()
            outcomes.append(outcome)
            (workspace / "run_status.json").write_text(json.dumps(outcomes, indent=2) + "\n")
            print(json.dumps(outcome), flush=True)
    if not all(r["ok"] for r in outcomes):
        raise SystemExit("Some jobs failed. Inspect run_status.json and logs; incomplete jobs are not a successful reproduction.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="Check archive hashes/arithmetic; no fitting")
    sub.add_parser("tables", help="Export available table values; no fitting")
    figures = sub.add_parser("figures", help="Rebuild the six experimental figures; no fitting")
    figures.add_argument("--font-dir", type=Path)
    for command in ("plan", "preflight", "cloud-run"):
        p = sub.add_parser(command)
        p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
        p.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
        if command != "plan":
            p.add_argument("--data-dir", type=Path, default=REPO / "data")
        if command == "cloud-run":
            p.add_argument("--output-dir", type=Path, required=True)
            p.add_argument("--workers", type=int, default=1)
            p.add_argument("--blas-threads", type=int, default=1)
            p.add_argument("--confirm-cloud", action="store_true")
            p.add_argument("--acknowledge-legacy-protocol", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        for path in list(REPO.glob("*.py")) + list((REPO / "src").rglob("*.py")) + list((REPO / "reproduction").glob("*.py")):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        print(json.dumps(verify(), indent=2))
    elif args.command == "tables":
        print(json.dumps(build(REPO / "generated/tables"), indent=2))
    elif args.command == "figures":
        verify()
        argv = [sys.executable, str(REPO / "reproduction/plot_figures.py")]
        if args.font_dir:
            argv.extend(["--font-dir", str(args.font_dir)])
        subprocess.run(argv, check=True, cwd=REPO)
    elif args.command == "plan":
        print(json.dumps(job_plan(args.datasets, args.groups), indent=2))
    elif args.command == "preflight":
        check = preflight(args.data_dir, args.datasets, args.groups)
        print(json.dumps(check, indent=2))
        if not check["ready"]:
            raise SystemExit(2)
    else:
        cloud_run(args)


if __name__ == "__main__":
    main()
