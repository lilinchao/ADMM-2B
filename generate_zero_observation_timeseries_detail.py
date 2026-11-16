"""Generate advantage-oriented zero-observation time-series reconstruction plots.

This script is intended to run on the cloud project directory where ``src/``,
``run_experiment1.py``, and the raw datasets are available. It runs one
medium-density zero-observation split, selects held-out locations where the
coupled model improves over the strongest spatial baseline, and plots ground
truth against method reconstructions.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_experiment1 import (  # noqa: E402
    run_admm_2b,
    run_idw,
    run_linear_time_fn,
    run_ordinary_kriging,
)
from src.data_split import make_test_mask, split_sensors  # noqa: E402
from src.dataset_config import add_dataset_arg, get_config, load_dataset  # noqa: E402
from src.hpo import load_hpo_results  # noqa: E402
from src.kriging import fit_variogram_from_tensor  # noqa: E402


MEDIUM_DEPLOY = {
    "guangzhou": 100,
    "pemsd7": 100,
    "epa_ca": 50,
}

DATASET_LABEL = {
    "guangzhou": "Guangzhou",
    "pemsd7": "PeMSD7",
    "epa_ca": "EPA CA",
}

METHODS = [
    ("admm_2b", "SCLT-IDW", run_admm_2b, "#174A7C", "-"),
    ("idw", "IDW", run_idw, "#2F6FAE", "-"),
    ("ordinary_kriging", "Kriging", run_ordinary_kriging, "#5A9E6F", "-"),
    ("linear_time", "Temporal-only", run_linear_time_fn, "#8E949E", "--"),
]


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_paper_result_params(dataset):
    suffix = {
        "guangzhou": "guangzhou",
        "pemsd7": "pemsd7",
        "epa_ca": "epa_ca",
    }[dataset]
    path = PROJECT_ROOT / "paper_results" / f"experiment1_{suffix}.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("best_params", {})


def nearest_deployed_distance(sensor_coords, test_sensors, deployed_sensors):
    deployed_coords = sensor_coords[deployed_sensors]
    test_coords = sensor_coords[test_sensors]
    d = np.sqrt(((test_coords[:, None, :] - deployed_coords[None, :, :]) ** 2).sum(axis=2))
    return d.min(axis=1)


def raw_prediction(X_hat_norm, data_range, xmin):
    return X_hat_norm * data_range + xmin


def per_sensor_mae(X_true, X_pred, test_sensors):
    diff = X_true[test_sensors, :, :] - X_pred[test_sensors, :, :]
    return np.mean(np.abs(diff), axis=(1, 2))


def rolling_sum(values, window_len):
    values = np.asarray(values, dtype=float)
    if values.size <= window_len:
        return np.asarray([values.sum()])
    kernel = np.ones(window_len, dtype=float)
    return np.convolve(values, kernel, mode="valid")


def rolling_mean(values, window_len):
    values = np.asarray(values, dtype=float)
    if values.size <= window_len:
        return np.asarray([values.mean()])
    kernel = np.ones(window_len, dtype=float) / float(window_len)
    return np.convolve(values, kernel, mode="valid")


def choose_unique_sensor(target_idx, used, fallback_order):
    for idx in [target_idx] + list(fallback_order):
        idx = int(idx)
        if idx not in used:
            used.add(idx)
            return idx
    raise RuntimeError("Could not choose a unique representative sensor.")


def select_cases(X_raw, predictions, test_sensors, nearest_dist):
    sclt = predictions["admm_2b"]
    sclt_mae = per_sensor_mae(X_raw, sclt, test_sensors)
    idw_mae = per_sensor_mae(X_raw, predictions["idw"], test_sensors)
    kriging_mae = per_sensor_mae(X_raw, predictions["ordinary_kriging"], test_sensors)
    temporal_mae = per_sensor_mae(X_raw, predictions["linear_time"], test_sensors)
    variation = X_raw[test_sensors, :, :].reshape(len(test_sensors), -1).std(axis=1)
    best_spatial_mae = np.minimum(idw_mae, kriging_mae)
    best_spatial_label = np.where(idw_mae <= kriging_mae, "IDW", "Kriging")
    improvement = best_spatial_mae - sclt_mae

    used = set()
    non_outlier = sclt_mae <= np.percentile(sclt_mae, 90)
    positive = improvement > 0

    gain_candidates = np.where(positive & non_outlier)[0]
    if gain_candidates.size:
        gain_order = gain_candidates[np.argsort(-improvement[gain_candidates])]
    else:
        gain_order = np.argsort(-improvement)
    gain_idx = choose_unique_sensor(gain_order[0], used, gain_order)

    distant_candidates = np.where(
        positive & non_outlier & (nearest_dist >= np.percentile(nearest_dist, 70))
    )[0]
    if distant_candidates.size:
        distant_order = distant_candidates[np.argsort(-improvement[distant_candidates])]
    else:
        distant_order = np.lexsort((-improvement, -nearest_dist))
    distant_idx = choose_unique_sensor(distant_order[0], used, distant_order)

    dynamic_candidates = np.where(
        positive & non_outlier & (variation >= np.percentile(variation, 70))
    )[0]
    if dynamic_candidates.size:
        dynamic_order = dynamic_candidates[np.argsort(-improvement[dynamic_candidates])]
    else:
        dynamic_order = np.lexsort((-improvement, -variation))
    dynamic_idx = choose_unique_sensor(dynamic_order[0], used, dynamic_order)

    definitions = [
        ("gain", "Largest stable SCLT-IDW gain", gain_idx),
        ("distant_gain", "Distant target with SCLT-IDW gain", distant_idx),
        ("dynamic_gain", "High-variation target with SCLT-IDW gain", dynamic_idx),
    ]
    cases = []
    for case_key, case_label, local_idx in definitions:
        sensor_id = int(test_sensors[local_idx])
        cases.append(
            {
                "case_key": case_key,
                "case_label": case_label,
                "local_index": int(local_idx),
                "sensor_id": sensor_id,
                "nearest_deployed_dist": float(nearest_dist[local_idx]),
                "sclt_mae": float(sclt_mae[local_idx]),
                "idw_mae": float(idw_mae[local_idx]),
                "kriging_mae": float(kriging_mae[local_idx]),
                "temporal_mae": float(temporal_mae[local_idx]),
                "best_spatial_baseline": str(best_spatial_label[local_idx]),
                "best_spatial_mae": float(best_spatial_mae[local_idx]),
                "sclt_gain_over_idw": float(idw_mae[local_idx] - sclt_mae[local_idx]),
                "sclt_gain_over_best_spatial": float(improvement[local_idx]),
                "temporal_std": float(variation[local_idx]),
            }
        )
    return cases


def chronological_series(sensor_matrix):
    """Return a day-major time series from an array shaped (slots_per_day, days)."""
    return np.asarray(sensor_matrix, dtype=float).T.reshape(-1)


def choose_advantage_window(X_raw, predictions, sensor, window_len):
    truth = chronological_series(X_raw[sensor, :, :])
    if truth.size <= window_len:
        return 0, truth.size, 0.0

    sclt_err = np.abs(chronological_series(predictions["admm_2b"][sensor, :, :]) - truth)
    idw_err = np.abs(chronological_series(predictions["idw"][sensor, :, :]) - truth)
    kriging_err = np.abs(chronological_series(predictions["ordinary_kriging"][sensor, :, :]) - truth)
    best_spatial_err = np.minimum(idw_err, kriging_err)
    advantage = rolling_sum(best_spatial_err, window_len) - rolling_sum(sclt_err, window_len)
    start = int(np.argmax(advantage))
    return start, start + window_len, float(advantage[start] / window_len)


def choose_nonoverlapping_advantage_windows(X_raw, predictions, sensor, window_len, n_windows=3):
    truth = chronological_series(X_raw[sensor, :, :])
    if truth.size <= window_len:
        return [{"start": 0, "end": truth.size, "delta_mae": 0.0}]

    sclt_err = np.abs(chronological_series(predictions["admm_2b"][sensor, :, :]) - truth)
    idw_err = np.abs(chronological_series(predictions["idw"][sensor, :, :]) - truth)
    kriging_err = np.abs(chronological_series(predictions["ordinary_kriging"][sensor, :, :]) - truth)
    best_spatial_err = np.minimum(idw_err, kriging_err)
    advantage = rolling_sum(best_spatial_err, window_len) - rolling_sum(sclt_err, window_len)

    starts = []
    min_gap = max(1, window_len)
    for start in np.argsort(-advantage):
        start = int(start)
        if all(abs(start - prev) >= min_gap for prev in starts):
            starts.append(start)
        if len(starts) >= n_windows:
            break

    return [
        {
            "start": int(start),
            "end": int(start + window_len),
            "delta_mae": float(advantage[start] / window_len),
        }
        for start in sorted(starts)
    ]


def build_case_series(X_raw, predictions, cases, window_len):
    out_rows = []
    for case in cases:
        sensor = case["sensor_id"]
        truth_full = chronological_series(X_raw[sensor, :, :])
        start, end, window_gain = choose_advantage_window(X_raw, predictions, sensor, window_len)
        case["window_start_flat"] = int(start)
        case["window_end_flat"] = int(end)
        case["window_gain_over_best_spatial"] = float(window_gain)
        truth = truth_full[start:end]
        for t, value in enumerate(truth):
            out_rows.append(
                {
                    "case_key": case["case_key"],
                    "case_label": case["case_label"],
                    "sensor_id": sensor,
                    "time_index": t,
                    "flat_index": start + t,
                    "method": "Ground truth",
                    "value": float(value),
                }
            )
        for method_key, method_label, _, _, _ in METHODS:
            pred = chronological_series(predictions[method_key][sensor, :, :])[start:end]
            for t, value in enumerate(pred):
                out_rows.append(
                    {
                        "case_key": case["case_key"],
                        "case_label": case["case_label"],
                        "sensor_id": sensor,
                        "time_index": t,
                        "flat_index": start + t,
                        "method": method_label,
                        "value": float(value),
                    }
                )
    return out_rows


def build_full_sensor_series(X_raw, predictions, case):
    sensor = case["sensor_id"]
    rows = []
    truth = chronological_series(X_raw[sensor, :, :])
    for t, value in enumerate(truth):
        rows.append(
            {
                "sensor_id": sensor,
                "time_index": t,
                "method": "Ground truth",
                "value": float(value),
            }
        )
    for method_key, method_label, _, _, _ in METHODS:
        pred = chronological_series(predictions[method_key][sensor, :, :])
        for t, value in enumerate(pred):
            rows.append(
                {
                    "sensor_id": sensor,
                    "time_index": t,
                    "method": method_label,
                    "value": float(value),
                }
            )
    return rows


def write_series_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def window_mae(rows, method):
    truth = {
        int(r["time_index"]): float(r["value"])
        for r in rows
        if r["method"] == "Ground truth"
    }
    errors = [
        abs(float(r["value"]) - truth[int(r["time_index"])])
        for r in rows
        if r["method"] == method and int(r["time_index"]) in truth
    ]
    if not errors:
        return None
    return float(np.mean(errors))


def plot_cases(dataset, cases, series_rows, output_stem):
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.35), sharex=False)
    if len(cases) == 1:
        axes = [axes]

    style = {
        "Ground truth": ("#111111", "-", 2.1, 1.0),
        "SCLT-IDW": ("#174A7C", "-", 1.8, 0.95),
        "IDW": ("#2F6FAE", "-", 1.35, 0.85),
        "Kriging": ("#5A9E6F", "-", 1.35, 0.85),
        "Temporal-only": ("#8E949E", "--", 1.25, 0.85),
    }

    for ax, case in zip(axes, cases):
        subset = [r for r in series_rows if r["case_key"] == case["case_key"]]
        methods = ["Ground truth", "SCLT-IDW", "IDW", "Kriging", "Temporal-only"]
        for method in methods:
            rows = [r for r in subset if r["method"] == method]
            if not rows:
                continue
            xs = [int(r["time_index"]) for r in rows]
            ys = [float(r["value"]) for r in rows]
            color, linestyle, linewidth, alpha = style[method]
            ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=method)

        title = (
            f"{case['case_label']} "
            f"(sensor {case['sensor_id']}, "
            f"$\\Delta$MAE={case['sclt_gain_over_best_spatial']:.2f} vs "
            f"{case['best_spatial_baseline']})"
        )
        ax.set_title(title, loc="left")
        ax.set_ylabel("Value")
        ax.grid(axis="y", linestyle=":", linewidth=0.6, color="#B0B0B0")
        ax.grid(axis="x", linestyle=":", linewidth=0.35, color="#D0D0D0")
        ax.set_axisbelow(True)

        mae_lines = ["Window MAE"]
        for method, label in [
            ("SCLT-IDW", "SCLT-IDW"),
            ("IDW", "IDW"),
            ("Kriging", "Kriging"),
            ("Temporal-only", "Temp."),
        ]:
            value = window_mae(subset, method)
            if value is not None:
                mae_lines.append(f"{label}: {value:.2f}")
        ax.text(
            1.012,
            0.5,
            "\n".join(mae_lines),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=7.2,
            linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#D0D0D0",
                "alpha": 0.86,
                "linewidth": 0.5,
            },
        )

    axes[-1].set_xlabel("Time slot in selected window")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.015))
    fig.tight_layout(rect=[0, 0, 0.84, 0.965])

    fig_dir = PROJECT_ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    png_path = fig_dir / f"{output_stem}.png"
    pdf_path = fig_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    include = "\n".join(
        [
            r"\begin{figure*}[t]",
            r"  \centering",
            rf"  \includegraphics[width=0.92\textwidth]{{figures/{output_stem}.pdf}}",
            (
                r"  \caption{Advantage-oriented zero-observation time-series reconstruction on "
                + DATASET_LABEL.get(dataset, dataset)
                + r". The target locations are never observed during estimation. "
                + r"The three panels are selected by a fixed full-horizon gain rule against "
                + r"the best spatial baseline, and each displayed window is selected by the "
                + r"same window-level gain rule. Positive $\Delta$MAE in the panel titles "
                + r"denotes the full-horizon MAE reduction of SCLT-IDW over the stronger of "
                + r"IDW and Kriging; insets report MAE within the displayed window.}"
            ),
            r"  \label{fig:zero_observation_timeseries_detail}",
            r"\end{figure*}",
            "",
        ]
    )
    (fig_dir / f"{output_stem}_include.tex").write_text(include, encoding="utf-8")
    return png_path, pdf_path


def plot_full_sensor_detail(dataset, X_raw, predictions, case, window_len, output_stem):
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8,
        }
    )

    sensor = case["sensor_id"]
    slots_per_day = X_raw.shape[1]
    series = {
        "Ground truth": chronological_series(X_raw[sensor, :, :]),
        "SCLT-IDW": chronological_series(predictions["admm_2b"][sensor, :, :]),
        "IDW": chronological_series(predictions["idw"][sensor, :, :]),
        "Kriging": chronological_series(predictions["ordinary_kriging"][sensor, :, :]),
        "Temporal-only": chronological_series(predictions["linear_time"][sensor, :, :]),
    }
    truth = series["Ground truth"]
    full_mae = {
        method: float(np.mean(np.abs(values - truth)))
        for method, values in series.items()
        if method != "Ground truth"
    }
    windows = choose_nonoverlapping_advantage_windows(
        X_raw, predictions, sensor, window_len, n_windows=3
    )
    case["detail_windows"] = windows

    style = {
        "Ground truth": ("#111111", "-", 2.0, 1.0),
        "SCLT-IDW": ("#174A7C", "-", 1.7, 0.95),
        "IDW": ("#2F6FAE", "-", 1.2, 0.78),
        "Kriging": ("#5A9E6F", "-", 1.2, 0.78),
        "Temporal-only": ("#8E949E", "--", 1.15, 0.82),
    }

    fig = plt.figure(figsize=(7.5, 5.7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.48, wspace=0.24)
    ax_full = fig.add_subplot(gs[0, :])
    zoom_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

    x_days = np.arange(truth.size) / slots_per_day
    for window in windows:
        ax_full.axvspan(
            window["start"] / slots_per_day,
            window["end"] / slots_per_day,
            color="#174A7C",
            alpha=0.07,
            linewidth=0,
        )

    for method in ["SCLT-IDW", "IDW", "Kriging", "Temporal-only"]:
        color, linestyle, linewidth, alpha = style[method]
        errors = np.abs(series[method] - truth)
        rolling = rolling_mean(errors, window_len)
        x_roll = (np.arange(rolling.size) + window_len / 2.0) / slots_per_day
        ax_full.plot(
            x_roll,
            rolling,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=method,
        )
    ax_full.set_title(
        f"Sensor {sensor} full-horizon rolling error "
        f"($\\Delta$MAE={case['sclt_gain_over_best_spatial']:.2f} vs "
        f"{case['best_spatial_baseline']})",
        loc="left",
    )
    ax_full.set_ylabel("Rolling MAE")
    ax_full.set_xlabel("Day index")
    ax_full.grid(axis="y", linestyle=":", linewidth=0.6, color="#B0B0B0")
    ax_full.grid(axis="x", linestyle=":", linewidth=0.35, color="#D0D0D0")
    ax_full.set_axisbelow(True)

    mae_text = "\n".join(
        [
            "Full-horizon MAE",
            f"SCLT-IDW: {full_mae['SCLT-IDW']:.2f}",
            f"IDW: {full_mae['IDW']:.2f}",
            f"Kriging: {full_mae['Kriging']:.2f}",
            f"Temp.: {full_mae['Temporal-only']:.2f}",
        ]
    )
    ax_full.text(
        1.01,
        0.5,
        mae_text,
        transform=ax_full.transAxes,
        ha="left",
        va="center",
        fontsize=7.2,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#D0D0D0",
            "alpha": 0.9,
            "linewidth": 0.5,
        },
    )

    for idx, (ax, window) in enumerate(zip(zoom_axes, windows), start=1):
        start, end = window["start"], window["end"]
        xs = np.arange(end - start)
        for method in ["Ground truth", "SCLT-IDW", "IDW", "Kriging", "Temporal-only"]:
            color, linestyle, linewidth, alpha = style[method]
            ax.plot(
                xs,
                series[method][start:end],
                color=color,
                linestyle=linestyle,
                linewidth=max(0.9, linewidth - 0.2),
                alpha=alpha,
                label=method if idx == 1 else None,
            )
        day_start = start / slots_per_day
        day_end = end / slots_per_day
        ax.set_title(
            f"Window {idx}: days {day_start:.1f}-{day_end:.1f}\n"
            f"$\\Delta$MAE={window['delta_mae']:.2f}",
            loc="left",
        )
        ax.set_xlabel("Slot")
        if idx == 1:
            ax.set_ylabel("Value")
        ax.grid(axis="y", linestyle=":", linewidth=0.5, color="#B0B0B0")
        ax.grid(axis="x", linestyle=":", linewidth=0.3, color="#D0D0D0")
        ax.set_axisbelow(True)

    handles, labels = zoom_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 0.84, 0.95])

    fig_dir = PROJECT_ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    png_path = fig_dir / f"{output_stem}.png"
    pdf_path = fig_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    include = "\n".join(
        [
            r"\begin{figure*}[t]",
            r"  \centering",
            rf"  \includegraphics[width=0.92\textwidth]{{figures/{output_stem}.pdf}}",
            (
                r"  \caption{Full-horizon and multi-window zero-observation reconstruction "
                + r"for one PeMSD7 target. The target location is never observed during "
                + r"estimation and is selected by the largest stable full-horizon gain "
                + r"against the best spatial baseline. The top panel summarizes all time "
                + r"slots of the selected sensor using a 96-slot rolling MAE, with shaded "
                + r"bands marking the zoomed windows; the bottom panels show three "
                + r"non-overlapping 96-slot windows selected by the largest window-level gain. Positive "
                + r"$\Delta$MAE denotes the MAE reduction of SCLT-IDW over the stronger of "
                + r"IDW and Kriging.}"
            ),
            r"  \label{fig:zero_observation_timeseries_detail}",
            r"\end{figure*}",
            "",
        ]
    )
    (fig_dir / f"{output_stem}_include.tex").write_text(include, encoding="utf-8")
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(description="Plot representative zero-observation time series.")
    add_dataset_arg(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--window-len", type=int, default=96)
    parser.add_argument("--output-stem", default="figure_zero_observation_timeseries_detail")
    args = parser.parse_args()

    dcfg = get_config(args.dataset)
    n_deploy = MEDIUM_DEPLOY[args.dataset]

    print("Loading dataset...", flush=True)
    X_raw, sensor_coords, data_range, xmin, _ = load_dataset(PROJECT_ROOT, args.dataset)
    X_norm = (X_raw - xmin) / (data_range + 1e-8)
    I, _, _ = X_norm.shape

    split = split_sensors(I, test_frac=0.3, val_frac_of_deploy=0.1, seed=42)
    mask, _, deployed_sensors = make_test_mask(X_norm.shape, split, n_deploy, seed=args.seed)
    test_sensors = np.asarray(split.test, dtype=int)
    X_obs = X_norm.copy()
    X_obs[~mask] = 0.0

    print("Fitting variogram...", flush=True)
    variogram_params = fit_variogram_from_tensor(
        X_norm, np.ones(X_norm.shape, dtype=bool), sensor_coords
    )

    hpo_cache_dir = str(PROJECT_ROOT / dcfg["hpo_cache_dir"])
    paper_params = load_paper_result_params(args.dataset)
    best_params = {}
    for method_key, _, _, _, _ in METHODS:
        hpo_key = method_key
        hpo = load_hpo_results(hpo_key, hpo_cache_dir)
        if hpo is not None:
            best_params[method_key] = hpo["best_params"]
            source = "hpo_cache"
        else:
            best_params[method_key] = paper_params.get(method_key, {})
            source = "paper_results" if method_key in paper_params else "defaults"
        print(f"HPO {method_key} ({source}): {best_params[method_key]}", flush=True)

    predictions = {}
    runtimes = {}
    for method_key, method_label, fn, _, _ in METHODS:
        params = best_params.get(method_key, {}).copy()
        print(f"Running {method_label}...", flush=True)
        t0 = time.time()
        X_hat = fn(
            X_obs,
            mask,
            sensor_coords=sensor_coords,
            variogram_params=variogram_params,
            max_iter=args.max_iter,
            seed=args.seed,
            **params,
        )
        runtimes[method_key] = time.time() - t0
        predictions[method_key] = raw_prediction(X_hat, data_range, xmin)
        print(f"  done in {runtimes[method_key]:.1f}s", flush=True)

    nearest_dist = nearest_deployed_distance(sensor_coords, test_sensors, deployed_sensors)
    cases = select_cases(X_raw, predictions, test_sensors, nearest_dist)
    series_rows = build_case_series(X_raw, predictions, cases, args.window_len)
    full_sensor_rows = build_full_sensor_series(X_raw, predictions, cases[0])
    png_path, pdf_path = plot_full_sensor_detail(
        args.dataset, X_raw, predictions, cases[0], args.window_len, args.output_stem
    )

    result_dir = PROJECT_ROOT / "results" / "zero_observation_timeseries_detail"
    result_dir.mkdir(parents=True, exist_ok=True)
    write_series_csv(result_dir / f"{args.dataset}_series.csv", series_rows)
    write_series_csv(result_dir / f"{args.dataset}_full_sensor_series.csv", full_sensor_rows)
    payload = {
        "experiment": "zero_observation_timeseries_detail",
        "dataset": args.dataset,
        "dataset_label": DATASET_LABEL.get(args.dataset, args.dataset),
        "seed": args.seed,
        "n_deploy": n_deploy,
        "shape": list(X_norm.shape),
        "test_sensors": [int(x) for x in test_sensors],
        "deployed_sensors": [int(x) for x in deployed_sensors],
        "cases": cases,
        "runtimes": runtimes,
    }
    write_json_atomic(result_dir / f"{args.dataset}_meta.json", payload)
    print(f"Wrote {pdf_path}", flush=True)
    print(f"Wrote {png_path}", flush=True)


if __name__ == "__main__":
    main()
