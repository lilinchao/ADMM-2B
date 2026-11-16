"""Export available numerical table data, keeping missing evidence explicit."""
import csv
import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "reproduction_data"
METHODS = {"admm_2b": "SCLT-IDW", "admm_idw": "SCLT-IDW", "idw": "IDW",
           "idw_masked": "Masked-IDW", "ordinary_kriging": "Kriging", "linear_time": "Linear Time",
           "cp_td": "CP-TD", "halrtc": "HaLRTC", "brits": "BRITS", "saits": "SAITS",
           "admm_cp_only": "SCLT w/o spatial"}
MEDIUM = {"guangzhou": 100, "pemsd7": 100, "epa_ca": 50}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {"mean": statistics.mean(vals) if vals else None,
            "std": statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else None), "n": len(vals)}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    zero = []
    missing = []
    for ds, medium in MEDIUM.items():
        source = REPO / f"paper_results/experiment1_{ds}.json"
        obj = load(source)
        for method in ("admm_2b", "idw", "ordinary_kriging", "linear_time", "cp_td", "halrtc", "brits", "saits"):
            rows = obj.get("results", {}).get(method, {}).get(str(medium), [])
            record = {"dataset": ds, "method": METHODS[method], "observed_nodes": medium}
            for metric in ("mae", "rmse"):
                for name, value in stats([r.get(metric) for r in rows]).items():
                    record[f"{metric}_{name}"] = value
            if not record["mae_n"]:
                missing.append(f"Table 2: {ds}/{method}: no finite archived per-seed values")
            zero.append(record)
    write_csv(output / "table_02_zero_observation.csv", zero)

    with (DATA / "section_5_revised_masked_summary.csv").open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["deploy_n"]) == MEDIUM[r["dataset"]]
                and int(r["fault_pct"]) > 0 and r["method"] in {"idw_masked", "ordinary_kriging", "admm_idw"}]
    write_csv(output / "table_03_fault_rate.csv", rows)

    ablation = []
    for ds in MEDIUM:
        rows = load(DATA / f"results/section_5_fault_stress/{ds}_mask_perfect.json")["results"]
        rows += load(DATA / f"results/section_5_ablation/{ds}_admm_cp_only.json")["results"]
        for method in ("idw_masked", "ordinary_kriging", "admm_cp_only", "admm_idw"):
            record = {"dataset": ds, "method": METHODS[method]}
            selected = [r for r in rows if r["method"] == method and "error" not in r]
            for metric in ("mae_spatial", "mae_fault", "rmse_spatial", "rmse_fault"):
                for name, value in stats([r.get(metric) for r in selected]).items():
                    record[f"{metric}_{name}"] = value
            ablation.append(record)
    write_csv(output / "table_04_component_ablation.csv", ablation)
    report = {"tables_exported": 3, "missing_per_seed_evidence": missing,
              "submitted_table_snippets": "publication/tables",
              "note": "Table 3 uses saved aggregate CSV; not all original sweep JSONs survive. Submitted tables are not overwritten."}
    (output / "table_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(build(REPO / "generated/tables"), indent=2))
