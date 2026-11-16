"""Validate saved files and selected arithmetic without running any model."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def read_json(path):
    # Historical Python JSON uses NaN for undefined evaluation sets.
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(root=REPO):
    manifest = read_json(root / "reproduction/archive_manifest.json")
    errors = []
    for relative, record in manifest["files"].items():
        path = root / relative
        if not path.is_file():
            errors.append(f"Missing: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            errors.append(f"SHA256 mismatch: {relative}")
        elif path.suffix == ".json":
            read_json(path)
    if errors:
        raise ValueError("\n".join(errors))

    data = root / "reproduction_data"
    metadata = read_json(data / "results/zero_observation_timeseries_detail/pemsd7_meta.json")
    case = next(c for c in metadata["cases"] if c["case_key"] == "gain")
    series = defaultdict(dict)
    with (data / "results/zero_observation_timeseries_detail/pemsd7_full_sensor_series.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["sensor_id"]) == case["sensor_id"]:
                series[row["method"]][int(row["time_index"])] = float(row["value"])
    truth = series["Ground truth"]
    expected = set(range(4224))
    if set(truth) != expected:
        raise ValueError("Incomplete ground-truth time axis")
    maes = {}
    for method, key in [("SCLT-IDW", "sclt_mae"), ("IDW", "idw_mae"),
                        ("Kriging", "kriging_mae"), ("Temporal-only", "temporal_mae")]:
        if set(series[method]) != expected:
            raise ValueError(f"Incomplete time axis: {method}")
        mae = sum(abs(series[method][i] - truth[i]) for i in sorted(expected)) / len(expected)
        if not math.isclose(mae, case[key], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"Trace MAE differs from metadata: {method}")
        maes[method] = mae

    params = read_json(root / "configs/paper_parameters.json")
    for ds, record in params["datasets"].items():
        for source in record["sources"]:
            saved = read_json(root / source)
            for method, values in saved["best_params"].items():
                if record["best_params"][method] != values:
                    raise ValueError(f"Parameter mismatch: {ds}/{method}")
    return {"status": "archive_integrity_passed", "files": len(manifest["files"]),
            "trace_points_per_method": 4224, "trace_mae": maes,
            "raw_data_rerun_verified": False,
            "warning": "Hash/arithmetic checks do not certify scientific validity or a fresh experiment rerun."}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
