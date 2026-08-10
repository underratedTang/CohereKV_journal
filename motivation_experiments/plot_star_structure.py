#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator


# Color-blind-friendly palette. Every curve also has an independent line style
# and marker so the figures remain identifiable after grayscale conversion.
LINE_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
LINE_MARKERS = ["o", "s", "^", "D", "v"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot IEEE-style star-structure motivation figures: group-level star prevalence, "
            "pair-level conflict rate, and horizontal two-panel composition."
        )
    )
    p.add_argument("--input_dir", required=True)
    p.add_argument("--panel_a_taus", required=True, help="Comma-separated cosine thresholds shown in panel (a).")
    p.add_argument("--panel_b_taus", required=True, help="Comma-separated cosine thresholds shown in panel (b).")
    p.add_argument("--font_family", default="Times New Roman")
    p.add_argument("--strict_times_new_roman", type=int, choices=[0, 1], default=1)
    p.add_argument("--panel_width_in", type=float, default=1.72)
    p.add_argument("--panel_height_in", type=float, default=1.58)
    p.add_argument("--combined_width_in", type=float, default=3.50)
    p.add_argument("--combined_height_in", type=float, default=1.58)
    p.add_argument("--png_dpi", type=int, default=600)
    p.add_argument("--panel_a_y_max_percent", default="auto")
    p.add_argument("--panel_b_y_max_percent", default="auto")
    return p


def parse_float_csv(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one floating-point value.")
    return values


def configure_ieee_style(font_family: str, strict: bool) -> str:
    prop = font_manager.FontProperties(family=font_family)
    try:
        path = font_manager.findfont(prop, fallback_to_default=False)
        resolved = font_manager.FontProperties(fname=path).get_name()
    except Exception as exc:
        if strict:
            raise RuntimeError(
                f"Required font {font_family!r} was not found by Matplotlib. "
                "Install Times New Roman on the experiment machine, rebuild Matplotlib's font cache, and rerun. "
                "The plotting code intentionally refuses silent font fallback."
            ) from exc
        resolved = font_family

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font_family],
            "font.size": 6.6,
            "axes.labelsize": 6.6,
            "axes.titlesize": 6.9,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 5.2,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.50,
            "ytick.major.width": 0.50,
            "xtick.major.size": 2.1,
            "ytick.major.size": 2.1,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.015,
        }
    )
    return resolved


def load_config(input_dir: Path) -> Dict:
    path = input_dir / "run_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the analysis first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_summary_overall(input_dir: Path) -> List[Dict[str, str]]:
    path = input_dir / "summary_overall.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the analysis first.")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {
        "anchor_budget",
        "cosine_threshold",
        "eligible_groups",
        "unsafe_groups",
        "unsafe_group_rate",
        "candidate_pairs",
        "conflict_pairs",
        "conflict_pair_rate",
    }
    if not rows:
        raise RuntimeError(f"{path} is empty.")
    missing = required.difference(rows[0].keys())
    if missing:
        raise KeyError(f"{path} is missing required columns: {sorted(missing)}")
    return rows


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def rounded_auto_percent_max(max_value: float) -> float:
    if not np.isfinite(max_value) or max_value <= 0.0:
        return 10.0
    return min(100.0, max(10.0, 10.0 * math.ceil((max_value + 3.0) / 10.0)))


def resolve_percent_ymax(spec: str, max_value: float, name: str) -> float:
    if spec.strip().lower() == "auto":
        return rounded_auto_percent_max(max_value)
    value = float(spec)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive or 'auto'.")
    if max_value > value + 1e-9:
        raise ValueError(
            f"{name}={value:.2f} clips the maximum plotted value ({max_value:.2f}%). "
            f"Increase {name.upper()} or set it to auto."
        )
    return value


def find_unique_row(
    summary_rows: Sequence[Dict[str, str]],
    rho: float,
    tau: float,
) -> Dict[str, str]:
    matches = [
        row
        for row in summary_rows
        if close(float(row["anchor_budget"]), rho)
        and close(float(row["cosine_threshold"]), tau)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one summary_overall row for budget={rho:.4f}, tau={tau:.4f}; "
            f"got {len(matches)}."
        )
    return matches[0]


def plot_metric_lines(
    ax,
    summary_rows: Sequence[Dict[str, str]],
    budgets: Sequence[float],
    taus: Sequence[float],
    metric_column: str,
    y_max_spec: str,
    y_max_name: str,
    ylabel: str,
    title: str,
) -> Dict[Tuple[float, float], float]:
    x = 100.0 * np.asarray(budgets, dtype=np.float64)
    plotted: Dict[Tuple[float, float], float] = {}
    max_rate = 0.0

    for idx, tau in enumerate(taus):
        y = []
        for rho in budgets:
            row = find_unique_row(summary_rows, rho, tau)
            rate = 100.0 * float(row[metric_column])
            if not np.isfinite(rate):
                raise RuntimeError(
                    f"Non-finite {metric_column} for budget={rho:.4f}, tau={tau:.4f}."
                )
            if rate < -1e-9 or rate > 100.0 + 1e-9:
                raise RuntimeError(
                    f"Invalid percentage from {metric_column}={rate:.4f}% at budget={rho:.4f}, tau={tau:.4f}."
                )
            y.append(rate)
            plotted[(rho, tau)] = rate
            max_rate = max(max_rate, rate)

        ax.plot(
            x,
            y,
            color=LINE_COLORS[idx % len(LINE_COLORS)],
            linestyle=LINE_STYLES[idx % len(LINE_STYLES)],
            marker=LINE_MARKERS[idx % len(LINE_MARKERS)],
            markerfacecolor="white",
            markeredgewidth=0.7,
            label=rf"$\tau={tau:.2f}$",
        )

    y_max = resolve_percent_ymax(y_max_spec, max_rate, y_max_name)
    ax.set_xlim(min(x) - 4.0, max(x) + 4.0)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks(x)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(round(value))}"))
    ax.yaxis.set_major_locator(MultipleLocator(10.0 if y_max <= 70.0 else 20.0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(round(value))}"))
    ax.set_xlabel("Retained KV budget (%)", labelpad=1.0)
    ax.set_ylabel(ylabel, labelpad=1.0)
    ax.set_title(title, pad=1.0, fontweight="normal")
    ax.grid(False)
    ax.legend(
        frameon=False,
        loc="best",
        handlelength=1.8,
        borderaxespad=0.25,
        labelspacing=0.12,
        handletextpad=0.35,
    )
    return plotted


def draw_panel_a(
    ax,
    summary_rows: Sequence[Dict[str, str]],
    budgets: Sequence[float],
    taus: Sequence[float],
    y_max_spec: str,
) -> Dict[Tuple[float, float], float]:
    return plot_metric_lines(
        ax=ax,
        summary_rows=summary_rows,
        budgets=budgets,
        taus=taus,
        metric_column="unsafe_group_rate",
        y_max_spec=y_max_spec,
        y_max_name="panel_a_y_max_percent",
        ylabel="Groups with star conflict (%)",
        title="(a) Star conflict prevalence",
    )


def draw_panel_b(
    ax,
    summary_rows: Sequence[Dict[str, str]],
    budgets: Sequence[float],
    taus: Sequence[float],
    y_max_spec: str,
) -> Dict[Tuple[float, float], float]:
    # Pooled pair-level statistic:
    #   total conflicting leaf-leaf pairs / total same-anchor leaf-leaf pairs.
    # It is intentionally NOT conditioned on star-conflicted groups and does
    # not average per-group densities, avoiding the size-2-group artifact.
    return plot_metric_lines(
        ax=ax,
        summary_rows=summary_rows,
        budgets=budgets,
        taus=taus,
        metric_column="conflict_pair_rate",
        y_max_spec=y_max_spec,
        y_max_name="panel_b_y_max_percent",
        ylabel="Conflicting leaf pairs (%)",
        title="(b) Pair-level conflict rate",
    )


def validate_summary_arithmetic(summary_rows: Sequence[Dict[str, str]]) -> None:
    for row in summary_rows:
        eligible = int(row["eligible_groups"])
        unsafe = int(row["unsafe_groups"])
        pairs = int(row["candidate_pairs"])
        conflicts = int(row["conflict_pairs"])
        unsafe_rate = float(row["unsafe_group_rate"])
        pair_rate = float(row["conflict_pair_rate"])

        if eligible < 0 or unsafe < 0 or unsafe > eligible:
            raise RuntimeError("Invalid eligible/unsafe group counts in summary_overall.csv.")
        if pairs < 0 or conflicts < 0 or conflicts > pairs:
            raise RuntimeError("Invalid candidate/conflict pair counts in summary_overall.csv.")
        if eligible > 0 and not close(unsafe_rate, unsafe / eligible, tol=5e-7):
            raise RuntimeError(
                "unsafe_group_rate is inconsistent with unsafe_groups / eligible_groups "
                f"at budget={row['anchor_budget']}, tau={row['cosine_threshold']}."
            )
        if pairs > 0 and not close(pair_rate, conflicts / pairs, tol=5e-7):
            raise RuntimeError(
                "conflict_pair_rate is inconsistent with conflict_pairs / candidate_pairs "
                f"at budget={row['anchor_budget']}, tau={row['cosine_threshold']}."
            )


def write_figure_statistics(
    path: Path,
    panel_a_rates: Dict[Tuple[float, float], float],
    panel_b_rates: Dict[Tuple[float, float], float],
    summary_rows: Sequence[Dict[str, str]],
) -> None:
    rows = []
    for (rho, tau), rate in sorted(panel_a_rates.items()):
        source = find_unique_row(summary_rows, rho, tau)
        rows.append(
            {
                "panel": "a_prevalence",
                "anchor_budget": rho,
                "cosine_threshold": tau,
                "metric_percent": rate,
                "eligible_groups": int(source["eligible_groups"]),
                "unsafe_groups": int(source["unsafe_groups"]),
            }
        )
    for (rho, tau), rate in sorted(panel_b_rates.items()):
        source = find_unique_row(summary_rows, rho, tau)
        rows.append(
            {
                "panel": "b_pair_conflict_rate",
                "anchor_budget": rho,
                "cosine_threshold": tau,
                "metric_percent": rate,
                "candidate_pairs": int(source["candidate_pairs"]),
                "conflict_pairs": int(source["conflict_pairs"]),
            }
        )

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, pdf_path: Path, png_path: Path, png_dpi: int) -> None:
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=png_dpi)
    plt.close(fig)
    print(f"Saved: {pdf_path}", flush=True)
    print(f"Saved: {png_path}", flush=True)


def remove_legacy_outputs(fig_dir: Path) -> None:
    legacy_names = [
        "panel_b_conflict_density.pdf",
        "panel_b_conflict_density.png",
        "panel_b_worst_leaf_similarity.pdf",
        "panel_b_worst_leaf_similarity.png",
        "panel_b_star_prevalence_extent.pdf",
        "panel_b_star_prevalence_extent.png",
        "panel_b_star_ccdf.pdf",
        "panel_b_star_ccdf.png",
        "panel_b_star_conflict_severity.pdf",
        "panel_b_star_conflict_severity.png",
        "panel_a_star_schematic.pdf",
        "panel_a_star_schematic.png",
        "star_structure_two_panel.pdf",
        "star_structure_two_panel.png",
    ]
    for name in legacy_names:
        path = fig_dir / name
        if path.exists():
            path.unlink()


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).resolve()
    config = load_config(input_dir)
    summary_rows = load_summary_overall(input_dir)
    validate_summary_arithmetic(summary_rows)

    panel_a_taus = parse_float_csv(args.panel_a_taus)
    panel_b_taus = parse_float_csv(args.panel_b_taus)
    budgets = [float(x) for x in config["anchor_budgets"]]
    available_taus = [float(x) for x in config["cosine_thresholds"]]

    for name, taus in (("PANEL_A_TAUS", panel_a_taus), ("PANEL_B_TAUS", panel_b_taus)):
        for tau in taus:
            if not any(close(tau, x) for x in available_taus):
                raise ValueError(
                    f"{name} contains tau={tau:.4f}, but analysis only contains {available_taus}. "
                    "Add the threshold to COSINE_THRESHOLDS and rerun analysis."
                )

    resolved_font = configure_ieee_style(args.font_family, bool(args.strict_times_new_roman))
    print(f"Resolved plotting font: {resolved_font}", flush=True)

    fig_dir = input_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    remove_legacy_outputs(fig_dir)

    fig_a, ax_a = plt.subplots(figsize=(args.panel_width_in, args.panel_height_in))
    rates_a = draw_panel_a(
        ax=ax_a,
        summary_rows=summary_rows,
        budgets=budgets,
        taus=panel_a_taus,
        y_max_spec=args.panel_a_y_max_percent,
    )
    fig_a.tight_layout(pad=0.35)
    save_figure(
        fig_a,
        fig_dir / "panel_a_star_prevalence.pdf",
        fig_dir / "panel_a_star_prevalence.png",
        args.png_dpi,
    )

    fig_b, ax_b = plt.subplots(figsize=(args.panel_width_in, args.panel_height_in))
    rates_b = draw_panel_b(
        ax=ax_b,
        summary_rows=summary_rows,
        budgets=budgets,
        taus=panel_b_taus,
        y_max_spec=args.panel_b_y_max_percent,
    )
    fig_b.tight_layout(pad=0.35)
    save_figure(
        fig_b,
        fig_dir / "panel_b_pair_conflict_rate.pdf",
        fig_dir / "panel_b_pair_conflict_rate.png",
        args.png_dpi,
    )

    fig_c, axes = plt.subplots(
        1,
        2,
        figsize=(args.combined_width_in, args.combined_height_in),
        gridspec_kw={"wspace": 0.34},
    )
    rates_c_a = draw_panel_a(
        ax=axes[0],
        summary_rows=summary_rows,
        budgets=budgets,
        taus=panel_a_taus,
        y_max_spec=args.panel_a_y_max_percent,
    )
    rates_c_b = draw_panel_b(
        ax=axes[1],
        summary_rows=summary_rows,
        budgets=budgets,
        taus=panel_b_taus,
        y_max_spec=args.panel_b_y_max_percent,
    )
    fig_c.subplots_adjust(left=0.105, right=0.995, bottom=0.20, top=0.90, wspace=0.38)
    save_figure(
        fig_c,
        fig_dir / "star_motivation_two_panel.pdf",
        fig_dir / "star_motivation_two_panel.png",
        args.png_dpi,
    )

    stats_path = fig_dir / "figure_statistics.csv"
    write_figure_statistics(stats_path, rates_c_a, rates_c_b, summary_rows)
    print(f"Saved: {stats_path}", flush=True)

    print("\nPanel (b) pooled pair-level conflict rates:", flush=True)
    for tau in panel_b_taus:
        pieces = []
        for rho in budgets:
            pieces.append(f"budget={rho:.2f}: {rates_b[(rho, tau)]:.2f}%")
        print(f"  tau={tau:.2f} | " + " | ".join(pieces), flush=True)


if __name__ == "__main__":
    main()
