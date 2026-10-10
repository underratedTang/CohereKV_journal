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


def configure_times_new_roman():
    """Use Times New Roman for all three paper figures, with serif fallbacks."""
    import matplotlib.font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    preferred = ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"]
    chosen = next((name for name in preferred if name in names), preferred[-1])
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": preferred,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return chosen


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
            ax.annotate(
                f"Best Merge-All\n$\\lambda_r$={bx:g}",
                xy=(bx, by), xytext=(7, 8), textcoords="offset points", fontsize=8,
            )

        conflict_lambda = f(rr_cr[0], "conflict_lambda")
        ax.set_xlabel("Routing threshold")
        ax.set_ylabel("Whole-layer post-o_proj attention-output relative error (%) ↓")
        ax.set_title(f"{metric.capitalize()} routing  ($\\lambda_c$={conflict_lambda:g})")
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

def plot_method(method_root: Path, ppl_root: Path, out: Path, dpi: int) -> List[str]:
    single = [r for r in read_csv(method_root / "single_round_summary.csv") if int(f(r, "is_primary", 1)) == 1]
    ppl = [r for r in read_csv(ppl_root / "ppl_summary.csv") if int(f(r, "is_primary", 1)) == 1] if ppl_root else []
    order_left = ["eviction_only", "merge_all", "original_coherekv", "modified_coherekv"]
    order_right = ["full", "eviction_only", "merge_all", "original_coherekv", "modified_coherekv"]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.3), constrained_layout=True)
    notes = []
    smap = {r.get("variant"): r for r in single}
    vals = [100*f(smap[v], "output_rel_error") if v in smap else np.nan for v in order_left]
    x = np.arange(len(order_left))
    axes[0].bar(x, vals)
    axes[0].set_xticks(x, ["Eviction-only", "Merge-All", "Original", "Modified"])
    axes[0].set_ylabel("Whole-layer post-o_proj attention-output relative error (%) ↓")
    axes[0].set_title("(a) Single-round whole-layer attention-output error")
    axes[0].grid(axis="y", alpha=.2)
    for xi, v, y in zip(x, order_left, vals):
        if np.isfinite(y):
            axes[0].annotate(f"{y:.4f}%", (xi, y), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
            notes.append(f"Single {v}: {y:.4f}%")

    pmap = {r.get("variant"): r for r in ppl}
    pvals = [f(pmap[v], "ppl") if v in pmap else np.nan for v in order_right]
    x2 = np.arange(len(order_right))
    axes[1].bar(x2, pvals)
    axes[1].set_xticks(x2, ["Full", "Eviction-only", "Merge-All", "Original", "Modified"])
    axes[1].set_ylabel("repository fixed-sequence PPL ↓")
    axes[1].set_title("(b) PPL via evaluate_perplexity")
    axes[1].grid(axis="y", alpha=.2)
    for xi, v, y in zip(x2, order_right, pvals):
        if np.isfinite(y):
            axes[1].annotate(f"{y:.3f}", (xi, y), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
            notes.append(f"PPL {v}: {y:.4f}; wall={f(pmap[v], 'wall_seconds'):.2f}s")
    if not ppl:
        axes[1].text(.5, .5, "PPL results not available", transform=axes[1].transAxes, ha="center", va="center")

    fig.suptitle(
        "Figure 3 / Improved CohereKV: single-round attention fidelity and repository-standard PPL",
        fontsize=13,
    )
    save(fig, out / "03_modified_coherekv_attention_error_and_ppl", dpi)
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motivation_dir", required=True)
    ap.add_argument("--method_dir", required=True)
    ap.add_argument("--ppl_dir", required=True)
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--dpi", type=int, default=240)
    args = ap.parse_args()
    font = configure_times_new_roman()
    mot = Path(args.motivation_dir).resolve()
    method = Path(args.method_dir).resolve()
    ppl = Path(args.ppl_dir).resolve()
    out = Path(args.output_dir).resolve() if args.output_dir else mot / "figures"
    out.mkdir(parents=True, exist_ok=True)
    notes = [f"font: {font or 'system fallback'}"]
    notes += plot_motivation1(mot, out, args.dpi)
    notes += plot_motivation2(mot, out, args.dpi)
    notes += plot_method(method, ppl, out, args.dpi)
    (out / "PLOT_SUMMARY.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes)); print(f"figures -> {out}")


if __name__ == "__main__":
    main()
