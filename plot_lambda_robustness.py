#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import List, Sequence

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


PPL_COLOR = "#0072B2"
LONGBENCH_COLOR = "#D55E00"
TEXT_COLOR = "black"
MISSING_TOKENS = {"", "--", "-", "na", "n/a", "nan", "none", "null"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot IEEE-style lambda robustness figures. Each model is shown in one panel "
            "with a PPL line on the left y-axis and LongBench Avg. bars on the right y-axis. "
            "The script saves one figure per model plus one vertically stacked combined figure."
        )
    )
    p.add_argument("--lambdas", nargs="+", required=True)
    p.add_argument("--model", action="append", required=True)
    p.add_argument("--ppl", action="append", required=True)
    p.add_argument("--longbench", action="append", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--combined_output_stem", default="lambda_robustness_combined")

    p.add_argument("--font_family", default="Times New Roman")
    p.add_argument("--strict_times_new_roman", type=int, choices=[0, 1], default=1)
    p.add_argument("--panel_width_in", type=float, default=1.72)
    p.add_argument("--panel_height_in", type=float, default=1.25)
    p.add_argument("--combined_width_in", type=float, default=3.50)
    p.add_argument("--combined_height_in", type=float, default=1.25)
    p.add_argument("--combined_hspace", type=float, default=0.45)
    p.add_argument("--png_dpi", type=int, default=600)
    p.add_argument("--annotation_fontsize", type=float, default=5.0)
    p.add_argument("--bar_hatch", default="//")

    p.add_argument("--ppl_y_min", default="auto")
    p.add_argument("--ppl_y_max", default="auto")
    p.add_argument("--longbench_y_min", default="0")
    p.add_argument("--longbench_y_max", default="auto")
    return p


def configure_ieee_style(font_family: str, strict: bool) -> str:
    prop = font_manager.FontProperties(family=font_family)
    try:
        path = font_manager.findfont(prop, fallback_to_default=False)
        resolved = font_manager.FontProperties(fname=path).get_name()
    except Exception as exc:
        if strict:
            raise RuntimeError(
                f"Required font {font_family!r} was not found by Matplotlib. "
                "Install Times New Roman on the machine, rebuild Matplotlib's font cache, and rerun. "
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
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.50,
            "ytick.major.width": 0.50,
            "xtick.major.size": 2.1,
            "ytick.major.size": 2.1,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
            "hatch.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.015,
        }
    )
    return resolved


def parse_lambdas(values: Sequence[str]) -> np.ndarray:
    if not values:
        raise ValueError("At least one lambda value is required.")
    lambdas = np.asarray([float(x) for x in values], dtype=np.float64)
    if not np.all(np.isfinite(lambdas)):
        raise ValueError("All lambda values must be finite numbers.")
    if len(np.unique(lambdas)) != len(lambdas):
        raise ValueError("Lambda values must be unique.")
    if len(lambdas) > 1 and np.any(np.diff(lambdas) <= 0.0):
        raise ValueError("Lambda values must be strictly increasing.")
    return lambdas


def parse_metric_series(
    text: str,
    expected_len: int,
    metric_name: str,
    model_name: str,
) -> np.ndarray:
    pieces = [piece.strip() for piece in text.split(",")]
    if len(pieces) != expected_len:
        raise ValueError(
            f"{metric_name} for model {model_name!r} has {len(pieces)} values, "
            f"but {expected_len} lambda values were provided."
        )

    values: List[float] = []
    for piece in pieces:
        if piece.lower() in MISSING_TOKENS:
            values.append(float("nan"))
            continue
        value = float(piece)
        values.append(value if math.isfinite(value) else float("nan"))

    array = np.asarray(values, dtype=np.float64)
    if not np.any(np.isfinite(array)):
        raise ValueError(f"{metric_name} for model {model_name!r} contains no finite value.")
    return array


def validate_inputs(args, lambdas: np.ndarray) -> tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    models = args.model
    if not (len(models) == len(args.ppl) == len(args.longbench)):
        raise ValueError(
            "The numbers of --model, --ppl, and --longbench arguments must be identical."
        )
    if len(set(models)) != len(models):
        raise ValueError("Model display names must be unique.")

    ppl_values = [
        parse_metric_series(text, len(lambdas), "PPL", model)
        for model, text in zip(models, args.ppl)
    ]
    longbench_values = [
        parse_metric_series(text, len(lambdas), "LongBench Avg.", model)
        for model, text in zip(models, args.longbench)
    ]
    return models, ppl_values, longbench_values


def format_lambda_tick(value: float, _position: int) -> str:
    return f"{value:g}"


def model_slug(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_name.strip()).strip("_").lower()
    return slug or "model"


def finite_min_max(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError("No finite values are available.")
    return float(np.min(finite)), float(np.max(finite))


def resolve_ppl_limits(values: np.ndarray, lower_spec: str, upper_spec: str) -> tuple[float, float]:
    data_min, data_max = finite_min_max(values)
    span = data_max - data_min
    if span <= 1e-12:
        pad = max(abs(data_max) * 0.06, 0.25)
    else:
        pad = max(0.12 * span, 0.08)

    auto_lower = data_min - 0.65 * pad
    auto_upper = data_max + 1.35 * pad

    lower = auto_lower if lower_spec.strip().lower() == "auto" else float(lower_spec)
    upper = auto_upper if upper_spec.strip().lower() == "auto" else float(upper_spec)

    if lower >= upper:
        raise ValueError("PPL y_min must be smaller than y_max.")
    if lower > data_min + 1e-12 or upper < data_max - 1e-12:
        raise ValueError("Requested PPL y range clips the data.")
    return lower, upper


def resolve_longbench_limits(
    values: np.ndarray,
    lower_spec: str,
    upper_spec: str,
) -> tuple[float, float]:
    data_min, data_max = finite_min_max(values)

    if lower_spec.strip().lower() == "auto":
        lower = 0.0 if data_min >= 0.0 else data_min - 0.08 * max(abs(data_min), 1.0)
    else:
        lower = float(lower_spec)

    if upper_spec.strip().lower() == "auto":
        span_from_base = data_max - lower
        upper = data_max + max(0.10 * span_from_base, 0.75)
    else:
        upper = float(upper_spec)

    if lower >= upper:
        raise ValueError("LongBench y_min must be smaller than y_max.")
    if lower > data_min + 1e-12 or upper < data_max - 1e-12:
        raise ValueError("Requested LongBench y range clips the data.")
    return lower, upper


def bar_width_for_lambdas(lambdas: np.ndarray) -> float:
    if len(lambdas) == 1:
        return max(abs(float(lambdas[0])) * 0.18, 0.10)
    return 0.30 * float(np.min(np.diff(lambdas)))


def _expanded_bbox(bbox: Bbox, pad_px: float = 1.3) -> Bbox:
    return Bbox.from_extents(
        bbox.x0 - pad_px,
        bbox.y0 - pad_px,
        bbox.x1 + pad_px,
        bbox.y1 + pad_px,
    )


def _annotation_boxes(fig, annotations):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return [
        _expanded_bbox(a.get_window_extent(renderer=renderer))
        for a in annotations
    ]


def resolve_annotation_overlaps(
    fig,
    ax,
    movable_annotations,
    fixed_annotations=(),
) -> None:
    """
    Resolve numeric-label collisions while preserving semantic placement.

    Hard constraints
    ----------------
    * LongBench labels (`fixed_annotations`) never move.
    * PPL labels (`movable_annotations`) remain horizontally centered on
      their markers: x offset is always exactly 0 pt.
    * PPL labels prefer the marker's direct-below position.
    * If a below-marker label collides with a fixed LongBench label, move
      the PPL label farther downward, still directly below the marker.
    * If there is insufficient room below the marker, place the PPL label
      directly above the marker instead.

    All collision checks are performed in display coordinates, so twin-y
    axes are handled correctly.
    """
    movable_annotations = list(movable_annotations)
    fixed_annotations = list(fixed_annotations)

    if not movable_annotations:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)

    # Small visual clearance between two text boxes / between text and frame.
    bbox_pad_px = 0.9
    frame_margin_px = 1.5

    def ann_bbox(ann):
        return _expanded_bbox(
            ann.get_window_extent(renderer=renderer),
            pad_px=bbox_pad_px,
        )

    def inside_vertical_frame(bbox):
        return (
            bbox.y0 >= axes_bbox.y0 + frame_margin_px
            and bbox.y1 <= axes_bbox.y1 - frame_margin_px
        )

    def overlaps_any(bbox, obstacles):
        for obstacle in obstacles:
            if bbox.overlaps(ann_bbox(obstacle)):
                return True
        return False

    # Once a PPL label has been positioned, treat it as another fixed
    # obstacle for subsequent PPL labels.
    placed_ppl = []

    # Candidate offsets in pt.  Start close to the marker and gradually
    # move farther away while remaining vertically aligned with it.
    below_offsets = [-4.0, -5.5, -7.0, -8.5, -10.0, -11.5, -13.0, -14.5]
    above_offsets = [4.0, 5.5, 7.0, 8.5, 10.0, 11.5, 13.0, 14.5]

    for ann in movable_annotations:
        # annotate_ppl() may already have flipped a label above because it
        # would cross the bottom frame. Preserve that as the preferred side.
        _, current_yoff = ann.get_position()
        prefer_below = current_yoff <= 0.0

        candidate_groups = (
            (below_offsets, "top"),
            (above_offsets, "bottom"),
        )
        if not prefer_below:
            candidate_groups = (
                (above_offsets, "bottom"),
                (below_offsets, "top"),
            )

        obstacles = fixed_annotations + placed_ppl
        chosen = False

        for offsets, va in candidate_groups:
            for yoff in offsets:
                # x offset is deliberately fixed to 0:
                # the number stays directly below/above the marker.
                ann.set_position((0.0, yoff))
                ann.set_horizontalalignment("center")
                ann.set_verticalalignment(va)

                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                bbox = ann_bbox(ann)

                if not inside_vertical_frame(bbox):
                    continue
                if overlaps_any(bbox, obstacles):
                    continue

                chosen = True
                break

            if chosen:
                break

        # If the layout is exceptionally dense and no fully collision-free
        # candidate exists, retain the closest in-frame position rather than
        # moving the label horizontally or aborting the plotting job.
        if not chosen:
            fallback_groups = (
                (below_offsets, "top"),
                (above_offsets, "bottom"),
            )
            for offsets, va in fallback_groups:
                for yoff in offsets:
                    ann.set_position((0.0, yoff))
                    ann.set_horizontalalignment("center")
                    ann.set_verticalalignment(va)
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    if inside_vertical_frame(ann_bbox(ann)):
                        chosen = True
                        break
                if chosen:
                    break

        placed_ppl.append(ann)

    fig.canvas.draw()

def annotate_ppl(ax, lambdas: np.ndarray, values: np.ndarray, fontsize: float):
    """
    Annotate PPL values directly below each blue marker.

    Placement policy:
      1) Default: horizontally centered directly below the marker.
      2) If the rendered text would cross the bottom boundary of the axes,
         automatically flip it directly above the marker.
    """
    annotations = []

    for x, y in zip(lambdas, values):
        if not np.isfinite(y):
            continue

        ann = ax.annotate(
            f"{y:.2f}",
            xy=(x, y),
            xytext=(0.0, -4.0),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=fontsize,
            color=TEXT_COLOR,
            annotation_clip=True,
            clip_on=True,
            zorder=10,
        )
        annotations.append(ann)

    # Render once and inspect the true text bounding boxes.
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)

    margin_px = 1.5

    for ann in annotations:
        bbox = ann.get_window_extent(renderer=renderer)

        # Only flip vertically. Horizontal alignment always remains centered
        # exactly under/above the marker.
        if bbox.y0 < axes_bbox.y0 + margin_px:
            ann.set_position((0.0, 4.0))
            ann.set_verticalalignment("bottom")

    ax.figure.canvas.draw()
    return annotations

def annotate_longbench(ax, bars, values: np.ndarray, fontsize: float):
    """
    Align all LongBench value labels on one horizontal line per subplot.

    The common label height is determined by the highest finite LongBench
    value in the current model. Therefore, all LongBench numbers are aligned
    with the label above the tallest bar, while each number remains centered
    over its own bar.
    """
    annotations = []

    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return annotations

    # All LongBench labels share the y coordinate of the tallest bar.
    label_y = float(np.max(finite_values))

    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            continue

        annotations.append(
            ax.annotate(
                f"{value:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2.0, label_y),
                xytext=(0.0, 0.38),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color=TEXT_COLOR,
                annotation_clip=True,
                clip_on=True,
                zorder=9,
            )
        )

    return annotations

def draw_model_panel(
    ax_ppl,
    lambdas: np.ndarray,
    ppl: np.ndarray,
    longbench: np.ndarray,
    title: str,
    ppl_y_min_spec: str,
    ppl_y_max_spec: str,
    longbench_y_min_spec: str,
    longbench_y_max_spec: str,
    annotation_fontsize: float,
    bar_hatch: str,
) -> None:
    ax_lb = ax_ppl.twinx()

    width = bar_width_for_lambdas(lambdas)
    finite_lb = np.isfinite(longbench)

    bars = ax_lb.bar(
        lambdas[finite_lb],
        longbench[finite_lb],
        width=width,
        facecolor="white",
        edgecolor=LONGBENCH_COLOR,
        linewidth=0.7,
        hatch=bar_hatch,
        zorder=1,
    )

    ax_ppl.plot(
        lambdas,
        ppl,
        color=PPL_COLOR,
        linestyle="-",
        marker="o",
        markerfacecolor="white",
        markeredgecolor=PPL_COLOR,
        markeredgewidth=0.75,
        linewidth=1.0,
        markersize=3.2,
        zorder=5,
    )

    ax_ppl.set_zorder(2)
    ax_lb.set_zorder(1)
    ax_ppl.patch.set_visible(False)
    ax_lb.patch.set_visible(False)

    x_span = float(lambdas[-1] - lambdas[0]) if len(lambdas) > 1 else max(abs(float(lambdas[0])), 1.0)
    x_pad = max(0.13 * x_span, 1.00 * width)
    ax_ppl.set_xlim(float(lambdas[0]) - x_pad, float(lambdas[-1]) + x_pad)
    ax_ppl.set_xticks(lambdas)
    ax_ppl.xaxis.set_major_formatter(FuncFormatter(format_lambda_tick))

    ppl_ymin, ppl_ymax = resolve_ppl_limits(ppl, ppl_y_min_spec, ppl_y_max_spec)
    lb_ymin, lb_ymax = resolve_longbench_limits(
        longbench,
        longbench_y_min_spec,
        longbench_y_max_spec,
    )
    ax_ppl.set_ylim(ppl_ymin, ppl_ymax)
    ax_lb.set_ylim(lb_ymin, lb_ymax)

    ax_ppl.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_lb.yaxis.set_major_locator(MaxNLocator(nbins=4))

    ax_ppl.set_xlabel(r"Grouping threshold $\lambda$", color=TEXT_COLOR, labelpad=1.0)
    ax_ppl.set_ylabel("PPL", color=TEXT_COLOR, labelpad=1.0)
    ax_lb.set_ylabel("LongBench Avg.", color=TEXT_COLOR, labelpad=2.0)
    ax_ppl.tick_params(axis="x", colors=TEXT_COLOR)
    ax_ppl.tick_params(axis="y", colors=TEXT_COLOR)
    ax_lb.tick_params(axis="y", colors=TEXT_COLOR)
    for spine in ax_ppl.spines.values():
        spine.set_color(TEXT_COLOR)
    for spine in ax_lb.spines.values():
        spine.set_color(TEXT_COLOR)

    ax_ppl.set_title(title, color=TEXT_COLOR, pad=1.4, fontweight="normal")
    ax_ppl.grid(False)
    ax_lb.grid(False)

    ppl_annotations = annotate_ppl(ax_ppl, lambdas, ppl, annotation_fontsize)

    # LongBench labels are a hard layout constraint:
    # they always stay centered above the corresponding bar top.
    # Do NOT pass them to the overlap resolver.
    lb_annotations = annotate_longbench(
        ax_lb, bars, longbench[finite_lb], annotation_fontsize
    )

    # Only PPL labels may be moved to resolve label-label collisions.
    resolve_annotation_overlaps(
        ax_ppl.figure,
        ax_ppl,
        ppl_annotations,
        fixed_annotations=lb_annotations,
    )


def save_figure(fig, pdf_path: Path, png_path: Path, png_dpi: int) -> None:
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=png_dpi)
    plt.close(fig)
    print(f"Saved: {pdf_path}", flush=True)
    print(f"Saved: {png_path}", flush=True)


def save_individual_model_figure(
    output_dir: Path,
    model: str,
    lambdas: np.ndarray,
    ppl: np.ndarray,
    longbench: np.ndarray,
    args,
) -> None:
    fig, ax = plt.subplots(figsize=(args.panel_width_in, args.panel_height_in))
    draw_model_panel(
        ax_ppl=ax,
        lambdas=lambdas,
        ppl=ppl,
        longbench=longbench,
        title=model,
        ppl_y_min_spec=args.ppl_y_min,
        ppl_y_max_spec=args.ppl_y_max,
        longbench_y_min_spec=args.longbench_y_min,
        longbench_y_max_spec=args.longbench_y_max,
        annotation_fontsize=args.annotation_fontsize,
        bar_hatch=args.bar_hatch,
    )
    fig.subplots_adjust(left=0.17, right=0.83, bottom=0.23, top=0.84)

    stem = f"lambda_robustness_{model_slug(model)}"
    save_figure(
        fig,
        output_dir / f"{stem}.pdf",
        output_dir / f"{stem}.png",
        args.png_dpi,
    )


def save_combined_figure(
    output_dir: Path,
    models: Sequence[str],
    lambdas: np.ndarray,
    ppl_values: Sequence[np.ndarray],
    longbench_values: Sequence[np.ndarray],
    args,
) -> None:

    n_models = len(models)

    width = args.combined_width_in
    height = args.combined_height_in * n_models

    fig, axes = plt.subplots(
        n_models,
        1,
        figsize=(width, height),
        squeeze=False,
    )

    axes = axes[:, 0]

    for idx, (ax, model, ppl, longbench) in enumerate(
        zip(axes, models, ppl_values, longbench_values)
    ):
        panel_letter = chr(ord("a") + idx)

        draw_model_panel(
            ax_ppl=ax,
            lambdas=lambdas,
            ppl=ppl,
            longbench=longbench,
            title=f"({panel_letter}) {model}",
            ppl_y_min_spec=args.ppl_y_min,
            ppl_y_max_spec=args.ppl_y_max,
            longbench_y_min_spec=args.longbench_y_min,
            longbench_y_max_spec=args.longbench_y_max,
            annotation_fontsize=args.annotation_fontsize,
            bar_hatch=args.bar_hatch,
        )

    fig.subplots_adjust(
        left=0.15,
        right=0.85,
        bottom=0.10,
        top=0.95,
        hspace=args.combined_hspace,
    )

    save_figure(
        fig,
        output_dir / f"{args.combined_output_stem}.pdf",
        output_dir / f"{args.combined_output_stem}.png",
        args.png_dpi,
    )

def main() -> None:
    args = build_parser().parse_args()
    lambdas = parse_lambdas(args.lambdas)
    models, ppl_values, longbench_values = validate_inputs(args, lambdas)

    resolved_font = configure_ieee_style(
        args.font_family,
        bool(args.strict_times_new_roman),
    )
    print(f"Resolved plotting font: {resolved_font}", flush=True)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for model, ppl, longbench in zip(models, ppl_values, longbench_values):
        save_individual_model_figure(
            output_dir=output_dir,
            model=model,
            lambdas=lambdas,
            ppl=ppl,
            longbench=longbench,
            args=args,
        )

    save_combined_figure(
        output_dir=output_dir,
        models=models,
        lambdas=lambdas,
        ppl_values=ppl_values,
        longbench_values=longbench_values,
        args=args,
    )

    print("\nPlotted data:", flush=True)
    print("  lambdas = " + ", ".join(f"{x:g}" for x in lambdas), flush=True)
    for model, ppl, lb in zip(models, ppl_values, longbench_values):
        ppl_s = ", ".join("--" if not np.isfinite(x) else f"{x:g}" for x in ppl)
        lb_s = ", ".join("--" if not np.isfinite(x) else f"{x:g}" for x in lb)
        print(f"  {model}: PPL=[{ppl_s}] | LongBench=[{lb_s}]", flush=True)


if __name__ == "__main__":
    main()
