#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def configure_times_new_roman(strict=True):
    """Use Times New Roman; fail loudly rather than silently relabel another font."""
    import matplotlib.font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    if strict and "Times New Roman" not in names:
        raise RuntimeError("Times New Roman is unavailable; install the licensed font "
                           "and run fc-cache -f. For local smoke tests only use "
                           "--strict_times_new_roman 0.")
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return "Times New Roman"


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(r, k, default=np.nan):
    try:
        return float(r.get(k, default))
    except Exception:
        return float(default)


def save(fig, path: Path, dpi: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def heatmap(ax, rows: List[Dict[str, str]], key: str, title: str, *,
            diverging: bool = False, fixed_0_100: bool = False):
    rr = [r for r in rows if np.isfinite(f(r, key))]
    if not rr:
        ax.text(.5, .5, "no data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return None
    layers = sorted({int(f(r, "layer_idx")) for r in rr})
    heads = sorted({int(f(r, "kv_head_idx")) for r in rr})
    mat = np.full((len(layers), len(heads)), np.nan)
    for r in rr:
        mat[layers.index(int(f(r, "layer_idx"))), heads.index(int(f(r, "kv_head_idx")))] = 100.0 * f(r, key)
    vals = mat[np.isfinite(mat)]
    if diverging:
        vmax = max(float(np.max(np.abs(vals))) if vals.size else 1.0, 0.01)
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    elif fixed_0_100:
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=100.0)
    else:
        lo = float(np.min(vals)) if vals.size else 0.0
        hi = float(np.max(vals)) if vals.size else 1.0
        if abs(hi - lo) < 1e-12:
            hi = lo + 1.0
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=lo, vmax=hi)
    ax.set_xticks(np.arange(len(heads)), [str(x) for x in heads])
    ax.set_yticks(np.arange(len(layers)), [str(x) for x in layers])
    ax.set_xlabel("KV head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    return im


def plot_motivation1(root: Path, out: Path, dpi: int) -> List[str]:
    lh = [r for r in read_csv(root / "q1_layer_head_summary.csv") if int(f(r, "is_primary", 1)) == 1]
    overall = [r for r in read_csv(root / "q1_figure_overall.csv") if int(f(r, "is_primary", 1)) == 1]
    # Requested paper layout: top row cosine, bottom row Euclidean; three
    # statistics form the columns so the two routing metrics are easy to compare.
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 8.4), constrained_layout=True)
    notes = []
    specs = [
        ("conflict_prevalence", "Star-conflict prevalence", False, True,
         "conflict groups / eligible Star groups (%)"),
        ("intervention_gain_mean", "Conflict-Reduced gain", True, False,
         "Merge-All - Conflict-Reduced (pp)"),
        ("intervention_positive_fraction", "Positive fraction", False, True,
         "conflict groups improved (%)"),
    ]
    for row, metric in enumerate(("cosine", "euclidean")):
        rr = [r for r in lh if r.get("routing_metric") == metric]
        for col, (key, title, diverging, fixed, cb_label) in enumerate(specs):
            im = heatmap(
                axes[row, col], rr, key,
                f"{title}", diverging=diverging, fixed_0_100=fixed,
            )
            if im is not None:
                cb = fig.colorbar(im, ax=axes[row, col], shrink=.82, pad=.02)
                cb.set_label(cb_label)
            if col == 0:
                axes[row, col].text(
                    -0.34, 0.5, "Cosine routing" if metric == "cosine" else "Euclidean routing",
                    transform=axes[row, col].transAxes, rotation=90,
                    va="center", ha="center", fontsize=12, fontweight="bold",
                )
        ov = next((r for r in overall if r.get("routing_metric") == metric), None)
        if ov:
            notes.append(
                f"M1 {metric}: prevalence={100*f(ov,'conflict_prevalence'):.1f}%; "
                f"CR gain={100*f(ov,'conflict_reduced_gain'):+.4f}pp; "
                f"positive={100*f(ov,'conflict_reduced_positive_fraction'):.1f}%"
            )
    fig.suptitle(
        "Figure 1 / Motivation 1: Star-conflict generality and filtering potential",
        fontsize=14,
    )
    save(fig, out / "01_star_conflict_generality_and_reduction", dpi)
    return notes


def plot_motivation2(root: Path, out: Path, dpi: int) -> List[str]:
    route_rows = read_csv(root / "q2_routing_sweep.csv")
    cr_rows = read_csv(root / "q2_conflict_reduced_curve.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.3), constrained_layout=True)
    notes = []
    for ax, metric in zip(axes, ("euclidean", "cosine")):
        rr_route = sorted(
            [r for r in route_rows if r.get("routing_metric") == metric],
            key=lambda r: f(r, "route_param"),
        )
        rr_cr = [r for r in cr_rows if r.get("routing_metric") == metric]
        if not rr_route or not rr_cr:
            ax.text(.5, .5, "no data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(metric)
            continue

        x_ma = np.asarray([f(r, "route_param") for r in rr_route])
        y_ma = np.asarray([100*f(r, "merge_all_output_rel_error") for r in rr_route])
        ax.plot(x_ma, y_ma, marker="o", linewidth=2.2, label="Merge-All")

        k_values = sorted({int(f(r, "cr_num_removals")) for r in rr_cr})
        for k in k_values:
            kk = sorted(
                [r for r in rr_cr if int(f(r, "cr_num_removals")) == k],
                key=lambda r: f(r, "route_param"),
            )
            if not kk:
                continue
            x = np.asarray([f(r, "route_param") for r in kk])
            y = np.asarray([100*f(r, "conflict_reduced_output_rel_error") for r in kk])
            ax.plot(x, y, marker="s", linewidth=1.9, label=f"Conflict-Reduced (K={k}/Star)")

        best = next((r for r in rr_route if int(f(r, "is_best_routing_only", 0)) == 1), None)
        if best is not None:
            bx = f(best, "route_param"); by = 100*f(best, "merge_all_output_rel_error")
            ax.scatter([bx], [by], s=80, marker="*", zorder=5)
            symbol = r"$\lambda$" if metric == "euclidean" else r"$\tau_{\mathrm{cos}}$"
            ax.annotate(
                f"Best Merge-All\n{symbol}={bx:g}",
                xy=(bx, by), xytext=(7, 8), textcoords="offset points", fontsize=8,
            )

        shared_lam = f(rr_cr[0], "lambda_value")
        if metric == "euclidean":
            ax.set_xlabel(r"Shared $\lambda$ (routing and conflict), $\Delta=\lambda S$")
            ax.set_title(r"Euclidean routing and conflict: unified $\Delta$")
        else:
            ax.set_xlabel(r"Cosine routing threshold $\tau_{\mathrm{cos}}$")
            ax.set_title(f"Cosine routing ($\\lambda$={shared_lam:g} for Euclidean conflicts)")
        ax.set_ylabel("Whole-layer post-o_proj attention-output relative error (%) ↓")
        ax.grid(alpha=.22)
        ax.legend(fontsize=8)

        best_ma = min(y_ma) if y_ma.size else np.nan
        best_cr = min(100*f(r, "conflict_reduced_output_rel_error") for r in rr_cr)
        notes.append(
            f"M2 {metric}: best routing-only MA={best_ma:.4f}%; "
            f"best plotted CR={best_cr:.4f}%"
        )
    fig.suptitle(
        "Figure 2 / Motivation 2: conflict reduction beyond routing-threshold tuning",
        fontsize=14,
    )
    save(fig, out / "02_conflict_reduced_vs_routing_threshold", dpi)
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motivation_dir", required=True)
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--dpi", type=int, default=240)
    ap.add_argument("--strict_times_new_roman", type=int, choices=(0, 1), default=1)
    args = ap.parse_args()
    font = configure_times_new_roman(bool(args.strict_times_new_roman))
    mot = Path(args.motivation_dir).resolve()
    out = Path(args.output_dir).resolve() if args.output_dir else mot / "figures"
    out.mkdir(parents=True, exist_ok=True)
    notes = [f"font: {font or 'system fallback'}"]
    notes += plot_motivation1(mot, out, args.dpi)
    notes += plot_motivation2(mot, out, args.dpi)
    (out / "PLOT_SUMMARY.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes)); print(f"figures -> {out}")


if __name__ == "__main__":
    main()
