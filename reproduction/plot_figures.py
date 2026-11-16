"""Publication figures from existing cloud experiment outputs; no model fitting."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import shutil
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import MaxNLocator, PercentFormatter
from pypdf import PdfReader, PdfWriter

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "generated"
PROJECT = REPO / "reproduction_data"
FIG = ROOT / "figures"

COL = {"SCLT-IDW": "#007C78", "IDW": "#C78526", "Masked-IDW": "#C78526",
       "Kriging": "#6578A5", "Temporal-only": "#858B93", "Ground truth": "#22272B"}
MARK = {"SCLT-IDW": "o", "IDW": "s", "Masked-IDW": "s", "Kriging": "^"}
DASH = {"SCLT-IDW": "-", "IDW": "--", "Masked-IDW": "--", "Kriging": "-.",
        "Temporal-only": ":", "Ground truth": "-"}
INK, GRID = "#30383D", "#E1E5E8"
DATASETS = [("guangzhou", "Guangzhou"), ("pemsd7", "PeMSD7"), ("epa_ca", "EPA CA")]
METHODS = [("admm_idw", "SCLT-IDW"), ("idw_masked", "Masked-IDW"), ("ordinary_kriging", "Kriging")]
AUDIT = {"source_files": {}, "figures": {}, "policy": "Existing results only; no experiments, no refitting, no curve smoothing beyond the original 96-slot rolling MAE."}

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 10, "mathtext.fontset": "stix",
    "axes.labelsize": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#9CA4AA",
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": .65,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "axes.unicode_minus": False, "lines.solid_capstyle": "round",
})


def source(relative):
    src = PROJECT / relative
    if not src.is_file():
        raise FileNotFoundError(f"Missing original result: {relative}")
    AUDIT["source_files"][relative] = hashlib.sha256(src.read_bytes()).hexdigest()
    return src


def csv_rows(relative):
    with source(relative).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def clean(ax, axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=.55)
    ax.set_axisbelow(True)
    ax.tick_params(pad=4)


def method_handle(name):
    return Line2D([], [], color=COL[name], lw=1.8 if name == "SCLT-IDW" else 1.2,
                  linestyle=DASH[name], marker=MARK.get(name), markersize=4,
                  markerfacecolor=COL[name] if name == "SCLT-IDW" else "white", label=name)


def save(fig, stem, details):
    fig.canvas.draw()
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(FIG / f"{stem}.{ext}", dpi=600 if ext == "png" else None)
    AUDIT["figures"][stem] = details
    plt.close(fig)


def timeseries():
    meta = json.loads(source("results/zero_observation_timeseries_detail/pemsd7_meta.json").read_text())
    case = next(x for x in meta["cases"] if x["case_key"] == "gain")
    rows = csv_rows("results/zero_observation_timeseries_detail/pemsd7_full_sensor_series.csv")
    grouped = defaultdict(list)
    for row in rows:
        if int(row["sensor_id"]) == case["sensor_id"]:
            grouped[row["method"]].append((int(row["time_index"]), float(row["value"])))
    series = {k: np.array([v for _, v in sorted(items)]) for k, items in grouped.items()}
    assert all(len(v) == 4224 for v in series.values())
    truth = series["Ground truth"]
    names = ["SCLT-IDW", "IDW", "Kriging", "Temporal-only"]
    means = {m: float(np.mean(np.abs(series[m] - truth))) for m in names}
    for name, key in [("SCLT-IDW", "sclt_mae"), ("IDW", "idw_mae"), ("Kriging", "kriging_mae"), ("Temporal-only", "temporal_mae")]:
        assert abs(means[name] - case[key]) < 1e-10

    fig = plt.figure(figsize=(7.2, 5.65))
    fig.legend(handles=[method_handle(m) for m in ["Ground truth"] + names],
               loc="upper center", bbox_to_anchor=(.52, .997), ncol=5, frameon=False,
               handlelength=2.1, columnspacing=1.2)
    ax = fig.add_axes([.085, .535, .89, .315])
    clean(ax)
    ax.set_title("(a)  Sensor 12: full-horizon reconstruction error", loc="left", pad=12)
    for j, win in enumerate(case["detail_windows"], 1):
        ax.axvspan(win["start"] / 96, win["end"] / 96, color="#D5E9E7", alpha=.65, lw=0)
        ax.text((win["start"] + win["end"]) / 192, 1.01, f"W{j}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color=COL["SCLT-IDW"], fontsize=8, fontweight="bold")
    for m in ["IDW", "Kriging", "Temporal-only", "SCLT-IDW"]:
        roll = np.convolve(np.abs(series[m] - truth), np.ones(96) / 96, mode="valid")
        ax.plot((np.arange(len(roll)) + 48) / 96, roll, color=COL[m], ls=DASH[m],
                lw=1.65 if m == "SCLT-IDW" else 1.05, alpha=1 if m == "SCLT-IDW" else .87)
    ax.set(xlim=(0, 44), ylim=(0, 22), xlabel="Day index", ylabel="96-slot rolling MAE")
    ax.set_xticks([0, 10, 20, 30, 40, 44])
    ax.set_yticks([0, 5, 10, 15, 20])
    fig.text(.085, .925, "Full-horizon MAE", fontsize=9, color="#6F787E")
    for x, m in zip([.30, .49, .65, .81], names):
        fig.text(x, .925, f"{m}: {means[m]:.2f}", color=COL[m], fontsize=9,
                 fontweight="bold" if m == "SCLT-IDW" else "normal")
    for j, win in enumerate(case["detail_windows"]):
        ax = fig.add_axes([.085 + j * .315, .105, .26, .265])
        clean(ax)
        a, b = win["start"], win["end"]
        for m in ["IDW", "Kriging", "Temporal-only", "Ground truth", "SCLT-IDW"]:
            ax.plot(np.arange(b-a), series[m][a:b], color=COL[m], ls=DASH[m],
                    lw=1.6 if m in ["Ground truth", "SCLT-IDW"] else 1.0)
        ax.set(xlim=(0, 95), ylim=(15, 85), xlabel="Time slot")
        ax.set_xticks([0, 24, 48, 72, 95])
        ax.set_yticks([20, 40, 60, 80])
        if j == 0:
            ax.set_ylabel("Value")
        ax.set_title(f"({chr(98+j)})  Window {j+1}", loc="left", pad=26, fontsize=10)
        ax.text(0, 1.065, f"Days {a/96:.1f}-{b/96:.1f}  |  " + r"$\Delta$" + f"MAE = {win['delta_mae']:.2f}",
                transform=ax.transAxes, fontsize=8.5, color="#657078")
    save(fig, "figure_zero_observation_timeseries_detail", {"sensor": case["sensor_id"], "full_horizon_mae": means,
         "windows": case["detail_windows"], "rolling_window": 96, "raw_values": "unchanged"})


def density():
    rows = csv_rows("section_5_revised_masked_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85))
    fig.subplots_adjust(left=.08, right=.985, top=.74, bottom=.22, wspace=.33)
    fig.legend(handles=[method_handle(m) for _, m in METHODS], loc="upper center", ncol=3,
               bbox_to_anchor=(.53, 1.0), frameon=False, columnspacing=2.5)
    record = []
    for i, (ax, (ds, name)) in enumerate(zip(axes, DATASETS)):
        clean(ax)
        ax.set_title(f"({chr(97+i)})  {name}", loc="left", pad=11, fontsize=10)
        limits = []
        xs = []
        for key, label in reversed(METHODS):
            sub = sorted([r for r in rows if r["dataset"] == ds and r["method"] == key
                          and int(r["fault_pct"]) == 30], key=lambda r: int(r["deploy_n"]))
            xs = np.array([int(r["deploy_n"]) for r in sub])
            y = np.array([float(r["mae_spatial_mean"]) for r in sub])
            sd = np.array([float(r["mae_spatial_std"]) for r in sub])
            ax.fill_between(xs, y-sd, y+sd, color=COL[label], alpha=.10, linewidth=0)
            ax.plot(xs, y, color=COL[label], ls=DASH[label], marker=MARK[label],
                    markersize=4, markeredgewidth=.9, lw=1.7 if label == "SCLT-IDW" else 1.1,
                    markerfacecolor=COL[label] if label == "SCLT-IDW" else "white")
            limits.extend((y-sd).tolist() + (y+sd).tolist())
            record.append({"dataset": ds, "method": label, "x": xs.tolist(), "mean": y.tolist(), "std": sd.tolist()})
        span = max(limits) - min(limits)
        ax.set_ylim(min(limits) - .12 * span, max(limits) + .12 * span)
        ax.set_xticks(xs)
        ax.set_xlabel("Observed locations", fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Spatial MAE", labelpad=7)
    save(fig, "figure_density_sweep_fault30_spatial_mae_masked", {"data": record, "bands": "mean +/- standard deviation"})


def stress_figures():
    rows = csv_rows("results/section_5_fault_stress/section_5_fault_stress_summary.csv")
    lookup = {(r["dataset_key"], r["scenario"], r["method_key"]): r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.05), sharey=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.22, top=.76, wspace=.16)
    fig.legend(handles=[Rectangle((0, 0), 1, 1, color=COL[label], label=label) for _, label in METHODS],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.53, 1), columnspacing=2.4)
    record = []
    order = METHODS[1:] + METHODS[:1]
    for i, (ax, (ds, ds_label)) in enumerate(zip(axes, DATASETS)):
        clean(ax)
        for j, (key, label) in enumerate(order):
            base, noisy = lookup[(ds, "mask_perfect", key)], lookup[(ds, "mask_fn20", key)]
            delta = float(noisy["mae_fault_mean"]) - float(base["mae_fault_mean"])
            se = math.sqrt(float(noisy["mae_fault_std"])**2 / int(noisy["mae_fault_n"])
                           + float(base["mae_fault_std"])**2 / int(base["mae_fault_n"]))
            ax.bar(j, delta, .54, color=COL[label], edgecolor="white", linewidth=.4)
            ax.errorbar(j, delta, yerr=se, fmt="none", ecolor=INK, lw=.8, capsize=2.5)
            ax.text(j, delta+se+.35, f"+{delta:.1f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold" if key == "admm_idw" else "normal")
            record.append({"dataset": ds, "method": key, "delta": delta, "se": se})
        ax.set_title(f"({chr(97+i)})  {ds_label}", loc="left", pad=11, fontsize=10)
        ax.set(ylim=(0, 14.3), xlim=(-.6, 2.6), xticks=[])
        ax.set_yticks([0, 3, 6, 9, 12])
    axes[0].set_ylabel("True-corruption MAE increase")
    fig.text(.53, .06, "20% of true corruptions left unflagged", ha="center", fontsize=10)
    save(fig, "figure_mask_noise_degradation", {"values": record, "error_bars": "Original independent-SE propagation retained"})

    scenarios = [("mask_perfect", "Random entries"), ("burst", "Temporal burst"),
                 ("sensor_outage", "Location outage"), ("spatial_cluster", "Spatial cluster"), ("high_value", "High-value entries")]
    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    fig.subplots_adjust(left=.23, right=.94, bottom=.18, top=.84)
    clean(ax, axis="x")
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.axvline(0, color="#747E86", lw=.85)
    ds_colors = ["#007C78", "#6578A5", "#C78526"]
    ds_markers = ["o", "s", "^"]
    record = []
    for i, ((ds, label), color, marker) in enumerate(zip(DATASETS, ds_colors, ds_markers)):
        gains = []
        y = np.arange(5) + (i-1) * .23
        for sc, _ in scenarios:
            base = float(lookup[(ds, sc, "idw_masked")]["mae_fault_mean"])
            value = float(lookup[(ds, sc, "admm_idw")]["mae_fault_mean"])
            gains.append(100 * (base-value) / base)
        ax.hlines(y, 0, gains, color=color, alpha=.25, lw=1.5)
        ax.scatter(gains, y, s=26, marker=marker, c=color, zorder=3, label=label)
        for yy, g in zip(y, gains):
            ax.text(g + (1.6 if g >= 0 else -1.6), yy, f"{g:+.1f}", fontsize=8.3,
                    va="center", ha="left" if g >= 0 else "right", color=color)
        record.append({"dataset": ds, "gain_pct": gains})
    ax.set(yticks=np.arange(5), yticklabels=[s[1] for s in scenarios], ylim=(4.6, -.6), xlim=(-42, 73))
    ax.set_xticks([-40, -20, 0, 20, 40, 60])
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel("Fault-MAE reduction relative to Masked-IDW", labelpad=9)
    fig.legend(loc="upper center", bbox_to_anchor=(.56, .99), frameon=False, ncol=3, columnspacing=2.5)
    save(fig, "figure_structured_fault_gain", {"scenarios": scenarios, "data": record,
         "definition": "100 * (baseline mean - SCLT mean) / baseline mean, unchanged"})


def scalability():
    rows = json.loads(source("results/tkde_extra/scalability.json").read_text())["results"]
    rows = sorted([r for r in rows if r["method"] == "SCLT-IDW"], key=lambda r: r["nodes"])
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    fig.subplots_adjust(left=.105, right=.96, top=.88, bottom=.24)
    clean(ax)
    x, y = [r["nodes"] for r in rows], [r["elapsed"] for r in rows]
    ax.plot(x, y, "-o", color=COL["SCLT-IDW"], lw=1.8, markersize=5)
    for xx, yy in zip(x, y):
        ax.annotate(f"{yy:.1f}", (xx, yy), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    ax.set(xlabel="Number of candidate locations", ylabel="Runtime (s)", ylim=(0, 24), xlim=(45, 1260))
    ax.set_xticks(x)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.text(.015, .96, "SCLT-IDW  |  1,440 time slots per location", transform=ax.transAxes, va="top", fontsize=9, color="#657078")
    save(fig, "figure_tkde_scalability_runtime", {"nodes": x, "runtime_s": y})


def sensitivity():
    rows = json.loads(source("results/tkde_extra/sensitivity.json").read_text())["results"]
    groups = [("rank", "Tensor rank", [5, 10, 15, 20]),
              ("gamma", r"Spatial weight $\gamma$", [.1, 1, 10, 50, 100]),
              ("neighbors", "IDW neighbors", [5, 10, 20, 30]),
              ("mask_fn", "Unflagged true corruptions", [0, .05, .1, .2, .3])]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharey=True)
    fig.subplots_adjust(left=.09, right=.98, bottom=.13, top=.81, hspace=.68, wspace=.22)
    records = []
    for i, (ax, (group, label, values)) in enumerate(zip(axes.flat, groups)):
        clean(ax)
        for metric, title, color, marker, ls in [("mae_spatial", "Spatial MAE", COL["SCLT-IDW"], "o", "-"),
                                                ("mae_fault", "Fault MAE", "#B7624E", "s", "--")]:
            means, ses = [], []
            for v in values:
                sub = np.array([r[metric] for r in rows if r["group"] == group and float(r["value"]) == v])
                assert len(sub) == 3
                means.append(float(sub.mean()))
                ses.append(float(sub.std(ddof=1) / math.sqrt(len(sub))))
            x = np.arange(len(values))
            ax.errorbar(x, means, yerr=ses, color=color, ls=ls, marker=marker, lw=1.5,
                        markersize=4, capsize=2.5, elinewidth=.85, label=title)
            records.append({"group": group, "metric": metric, "values": values, "mean": means, "se": ses})
        ax.set_xticks(np.arange(len(values)), [f"{v:.0%}" if group == "mask_fn" else f"{v:g}" for v in values])
        ax.set_xlabel(label, labelpad=6)
        ax.set_title(f"({chr(97+i)})", loc="left", fontsize=10, pad=7)
        ax.set_ylim(2.5, 11.6)
        ax.set_yticks([3, 5, 7, 9, 11])
    axes[0, 0].set_ylabel("MAE")
    axes[1, 0].set_ylabel("MAE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .985), ncol=2, frameon=False, columnspacing=3)
    save(fig, "figure_tkde_parameter_sensitivity", {"data": records, "error_bars": "Standard errors over three seeds", "x_positions": "Discrete tested settings, original order"})


def framework():
    fig = plt.figure(figsize=(7.5, 4.9))
    ax = fig.add_axes([.015, .02, .97, .95])
    ax.set(xlim=(0, 100), ylim=(0, 67))
    ax.axis("off")
    teal, blue = COL["SCLT-IDW"], COL["Kriging"]

    def text(x, y, s, size=10, color=INK, ha="left", bold=False, **kw):
        ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center", fontweight="bold" if bold else "normal", **kw)

    def box(x, y, w, h, color, title):
        ax.add_patch(Rectangle((x, y), w, h, fill=True, facecolor="white", edgecolor="#CAD2D7", lw=.7))
        ax.add_patch(Rectangle((x, y+h-.5), w, .5, facecolor=color, edgecolor="none"))
        text(x+2, y+h-3, title, size=10, color=color, bold=True)

    def arrow(a, b, color="#859099", style="-|>", **kw):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, color=color, mutation_scale=9, lw=1, **kw))

    text(0, 65, "INPUTS", 10, bold=True)
    text(28, 65, "SPATIALLY COUPLED LOW-RANK MODEL", 10, bold=True)
    text(79.5, 65, "OUTPUT", 10, bold=True)
    box(0, 40, 23, 21, "#7C878E", "Observed tensor")
    text(2, 53.4, r"$P_{\Omega_v}(\mathcal{X})$", 12)
    # Schematic mask, not experimental data.
    for r in range(4):
        for c in range(9):
            color = "#D9E9E6" if c < 6 else "#F3F4F5"
            if (r, c) in [(1, 1), (2, 4), (0, 3)]:
                color = "#D7A18F"
            ax.add_patch(Rectangle((2+c*2.05, 42+r*1.9), 1.75, 1.55, facecolor=color, edgecolor="none"))
    box(0, 19, 23, 18, "#7C878E", "Coordinates and masks")
    text(2, 29.2, r"$\{\mathbf{s}_i\}$", 11)
    text(2, 24.9, r"$\Omega_v,\quad\Omega_f,\quad\Omega_z$", 11)
    text(2, 21.2, "Valid / flagged / unobserved", 8)

    box(28, 36, 45, 25, teal, "Spatial-prior channel")
    # Compact coordinate graph depicts the IDW support only.
    target = (36, 47)
    for x, y in [(31, 42), (31, 52), (38, 53), (42, 47), (38, 40)]:
        ax.plot([x, target[0]], [y, target[1]], color="#A4C9C5", lw=.75)
        ax.plot(x, y, "o", ms=2.8, color="#8D989E")
    ax.plot(*target, marker="*", ms=9, color=teal)
    text(46, 51.2, "Coordinate-based IDW", 10)
    text(46, 46.6, "Cached neighbor weights", 9)
    text(46, 41.4, r"$\mathcal{W}^{t}=\mathcal{S}_{\Omega}(\mathcal{Z}^{t})$", 12)

    box(28, 7, 45, 25, blue, "Low-rank repair channel")
    for j, (x, h) in enumerate([(31, 12), (36, 9), (41, 6)]):
        for r in range(h//2):
            ax.add_patch(Rectangle((x, 11+r*1.5), 2.4, 1.25, facecolor="#DAE0ED", edgecolor=blue, lw=.3))
        text(x+1.2, 9, ["A", "B", "C"][j], 9, ha="center")
    text(46, 22.2, "Shared temporal structure", 10)
    text(46, 17.7, "Ridge-regularized CP factors", 9)
    text(46, 12.5, r"$\mathcal{Y}^{t+1}=[\![A,B,C]\!]$", 12)
    arrow((50.5, 36), (50.5, 32), teal, "<->")

    box(79.5, 25, 20.5, 36, teal, "Reconstruction")
    text(89.75, 51.5, r"$\widehat{\mathcal{X}}=\mathcal{Z}^{\star}$", 14, ha="center")
    ax.plot([81.5, 98], [47, 47], color=GRID, lw=.8)
    text(81.5, 42.5, r"$\Omega_z$", 11, color=teal, bold=True)
    text(81.5, 38.8, "Full temporal profiles", 8.6)
    text(81.5, 34, r"$\Omega_f$", 11, color=blue, bold=True)
    text(81.5, 30.3, "Flagged-entry repair", 8.6)
    text(79.5, 19, "Querying / monitoring", 8.4)
    text(79.5, 15.2, "Forecasting / data services", 8.4)

    # Both input groups inform both channels; avoid a false one-to-one pairing.
    ax.plot([23, 25.5], [50, 50], color="#859099", lw=.8)
    ax.plot([23, 25.5], [27, 27], color="#859099", lw=.8)
    ax.plot([25.5, 25.5], [19, 50], color="#859099", lw=.8)
    arrow((25.5, 48), (28, 48))
    arrow((25.5, 19), (28, 19))
    # Route the repeated exchange around the two model channels.
    ax.plot([75.3, 75.3], [2.2, 48], color="#8A959B", lw=.8)
    arrow((73, 48), (75.3, 48), teal)
    arrow((73, 19), (75.3, 19), blue)
    arrow((75.3, 43), (79.5, 43))
    ax.add_patch(Rectangle((0, -0.2), 73, 5.5, facecolor="#F1F5F5", edgecolor="none"))
    text(2, 2.5, "ADMM consensus", 9.5, bold=True, color=teal)
    text(27, 2.5, "Spatial estimate  /  CP factors  /  Reconstruction  /  Dual update", 8.1)
    arrow((75.3, 2.2), (73, 2.2))
    save(fig, "sclt_framework", {"type": "Vector schematic", "model": "Unchanged architecture; symbols aligned with the existing manuscript", "illustrations": "Schematic mask and coordinate graph, not measured data"})


def main():
    global ROOT, PROJECT, FIG
    parser = argparse.ArgumentParser(description="Rebuild six experimental figures from saved results; no model fitting.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--font-dir", type=Path, help="Directory containing legally installed Times New Roman TTF files")
    args = parser.parse_args()
    PROJECT, ROOT = args.data_dir.resolve(), args.output_dir.resolve()
    FIG = ROOT / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    font_dir = args.font_dir or Path("C:/Windows/Fonts")
    for file in ["times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"]:
        if (font_dir / file).is_file():
            font_manager.fontManager.addfont(str(font_dir / file))
    try:
        font_manager.findfont("Times New Roman", fallback_to_default=False)
    except ValueError:
        warnings.warn("Times New Roman is not installed; using STIXGeneral. Data are unchanged; glyph layout may differ.")
        plt.rcParams["font.family"] = "STIXGeneral"
    AUDIT["font_family"] = plt.rcParams["font.family"]
    # The author restored the original framework. Never regenerate its rejected replacement.
    original = REPO / "publication/figures/SCLT_Framework_HighQuality.pdf"
    shutil.copy2(original, FIG / original.name)
    timeseries()
    stress_figures()
    density()
    scalability()
    sensitivity()
    order = ["figure_zero_observation_timeseries_detail", "figure_mask_noise_degradation",
             "figure_density_sweep_fault30_spatial_mae_masked", "figure_structured_fault_gain",
             "figure_tkde_scalability_runtime", "figure_tkde_parameter_sensitivity"]
    gallery = PdfWriter()
    for i, stem in enumerate(order, 1):
        gallery.append(PdfReader(FIG / (stem + ".pdf")), outline_item=f"Figure {i}: " + stem.replace("figure_", "").replace("_", " "))
    gallery.write(ROOT / "Figure_gallery.pdf")
    (ROOT / "figure_data_audit.json").write_text(json.dumps(AUDIT, indent=2), encoding="utf-8")
    print(json.dumps({"figures": len(AUDIT["figures"]), "sources": len(AUDIT["source_files"])}))


if __name__ == "__main__":
    main()
