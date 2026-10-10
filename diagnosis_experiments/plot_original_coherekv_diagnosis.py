#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parser():
    p = argparse.ArgumentParser(description="Plot original CohereKV degeneration / fixed-Delta diagnosis.")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--font_family", default="Times New Roman")
    p.add_argument("--strict_times_new_roman", type=int, choices=[0, 1], default=0)
    p.add_argument("--png_dpi", type=int, default=600)
    p.add_argument("--primary_lambda", type=float, default=0.7)
    return p


def read(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cfg_font(name: str, strict: int) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [name], "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    if strict:
        from matplotlib import font_manager
        font_manager.findfont(name, fallback_to_default=False)


def save(fig, base: Path, dpi: int) -> None:
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {base.with_suffix('.pdf')}")


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


def f(r: Dict[str, str], key: str) -> float:
    return float(r[key])


def main():
    a = parser().parse_args()
    root = Path(a.input_dir).resolve()
    figs = root / "figures"
    figs.mkdir(exist_ok=True)
    cfg = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    if cfg.get("experiment") != "original_coherekv_diagnosis_v1":
        raise RuntimeError("Unexpected experiment config")
    single = read(root / "single_round_budget_diagnosis_overall.csv")
    rounds = read(root / "repeated_round_diagnosis_overall.csv")
    cfg_font(a.font_family, a.strict_times_new_roman)

    lambdas = sorted({float(r["lambda_mymethod"]) for r in rounds} or {float(r["lambda_mymethod"]) for r in single})
    if not lambdas:
        raise RuntimeError("No diagnosis rows found")
    primary = min(lambdas, key=lambda x: abs(x - a.primary_lambda))
    rr = sorted([r for r in rounds if close(r["lambda_mymethod"], primary)], key=lambda r: int(float(r["round_idx"])))
    sr = sorted([r for r in single if close(r["lambda_mymethod"], primary)], key=lambda r: float(r["anchor_budget"]))
    xround = np.asarray([int(float(r["round_idx"])) for r in rr], dtype=float)

    # 1) First-round system comparison: same raw cache, before certificate accumulation.
    if sr:
        xb = np.asarray([100.0 * f(r, "anchor_budget") for r in sr], dtype=float)
        fig, ax = plt.subplots(figsize=(6.8, 3.2))
        ax.plot(xb, [100*f(r, "mean_eviction_only_output_rel_error") for r in sr], marker="o", label="Eviction-Only")
        ax.plot(xb, [100*f(r, "mean_compatible_merge_all_output_rel_error") for r in sr], marker="o", label="Compatible Merge-All (routing only)")
        ax.plot(xb, [100*f(r, "mean_coherekv_output_rel_error") for r in sr], marker="o", label="Original CohereKV (routing + conflict graph + MIS)")
        ax.set_xlabel("Retained KV budget in first compression round (%)")
        ax.set_ylabel("Mean attention-output relative error (%)")
        ax.set_title(f"First compression round, λ={primary:g}: Original CohereKV vs Compatible Merge-All vs Eviction-Only")
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        fig.tight_layout()
        save(fig, figs / "first_round_original_coherekv_vs_merge_all_vs_eviction", a.png_dpi)

        fig, ax = plt.subplots(figsize=(6.8, 3.1))
        ax.plot(xb, [100*f(r, "candidate_admission_rate") for r in sr], marker="o", label="Candidate admission: Γ(source, anchor) ≤ Δ")
        ax.plot(xb, [100*f(r, "mis_survival_given_routed") for r in sr], marker="o", label="MIS survival among routed candidates")
        ax.plot(xb, [100*f(r, "overall_recovery_rate") for r in sr], marker="o", label="Final recovered evicted states")
        ax.set_xlabel("Retained KV budget in first compression round (%)")
        ax.set_ylabel("Evicted states (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"Where Original CohereKV loses recovery in the first round, λ={primary:g}: routing vs MIS filtering")
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        fig.tight_layout()
        save(fig, figs / "first_round_candidate_admission_vs_mis_recovery", a.png_dpi)

    if rr:
        # 2) Direct degeneration test.
        fig, ax = plt.subplots(figsize=(7.2, 3.3))
        ax.plot(xround, [100*f(r, "mean_eviction_only_output_rel_error") for r in rr], marker="o", label="Matched Eviction-Only")
        ax.plot(xround, [100*f(r, "mean_compatible_merge_all_output_rel_error") for r in rr], marker="o", label="Compatible Merge-All (same routing, no conflict filter)")
        ax.plot(xround, [100*f(r, "mean_coherekv_output_rel_error") for r in rr], marker="o", label="Original CohereKV")
        ax.plot(xround, [100*f(r, "mean_fresh_certificate_output_rel_error") for r in rr], marker="o", linestyle="--", label="Fresh-Certificate shadow (R=D=0 before decision)")
        ax.set_xlabel("Repeated compression round")
        ax.set_ylabel("Mean attention-output relative error (%)")
        ax.set_title(f"Does Original CohereKV approach Eviction-Only across repeated compression? λ={primary:g}")
        ax.legend(frameon=False, ncol=2)
        ax.grid(alpha=.2)
        fig.tight_layout()
        save(fig, figs / "original_coherekv_vs_eviction_only_attention_error_by_round", a.png_dpi)

        # 3) Benefit over eviction: zero means behavior has converged to the eviction baseline.
        fig, ax = plt.subplots(figsize=(6.8, 3.1))
        ax.plot(xround, [100*f(r, "coherekv_error_reduction_vs_eviction") for r in rr], marker="o", label="Original CohereKV")
        ax.plot(xround, [100*f(r, "merge_all_error_reduction_vs_eviction") for r in rr], marker="o", label="Compatible Merge-All")
        fresh_gain = []
        for r in rr:
            ev = f(r, "mean_eviction_only_output_rel_error")
            fresh = f(r, "mean_fresh_certificate_output_rel_error")
            fresh_gain.append(100*(ev-fresh)/max(ev, 1e-20))
        ax.plot(xround, fresh_gain, marker="o", linestyle="--", label="Fresh-Certificate shadow")
        ax.axhline(0.0, linewidth=.8)
        ax.set_xlabel("Repeated compression round")
        ax.set_ylabel("Attention-error reduction vs matched Eviction-Only (%)")
        ax.set_title(f"Degeneration indicator, λ={primary:g}: zero benefit means no fidelity gain over Eviction-Only")
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        fig.tight_layout()
        save(fig, figs / "fidelity_benefit_over_eviction_by_round", a.png_dpi)

        # 4) Candidate admission: fixed certificates vs same-state diagnostic shadows.
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        ax.plot(xround, [100*f(r, "original_candidate_admission_rate") for r in rr], marker="o", label="Original: fixed Δ + accumulated R,D")
        ax.plot(xround, [100*f(r, "fresh_cert_candidate_admission_rate") for r in rr], marker="o", linestyle="--", label="Same Δ, but reset R=D=0 for decision")
        ax.plot(xround, [100*f(r, "refreshed_scale_candidate_admission_rate") for r in rr], marker="o", linestyle=":", label="Accumulated R,D, but recompute S_ref and Δ")
        ax.set_xlabel("Repeated compression round")
        ax.set_ylabel("Evicted states admitted to a compatible anchor (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"Why routing collapses, λ={primary:g}: accumulated certificates vs frozen global Δ")
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        fig.tight_layout()
        save(fig, figs / "candidate_admission_fixed_delta_vs_diagnostic_shadows", a.png_dpi)

        # 5) MIS filtering and graph density.
        fig, ax = plt.subplots(figsize=(6.9, 3.1))
        ax.plot(xround, [100*f(r, "original_mis_survival_given_routed") for r in rr], marker="o", label="Original MIS survival among routed candidates")
        ax.plot(xround, [100*f(r, "fresh_cert_mis_survival_given_routed") for r in rr], marker="o", linestyle="--", label="Fresh-Certificate shadow MIS survival")
        ax.set_xlabel("Repeated compression round")
        ax.set_ylabel("Routed candidates surviving conflict filtering (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"Conflict-filter severity across rounds, λ={primary:g}: Original vs fresh-certificate shadow")
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        fig.tight_layout()
        save(fig, figs / "mis_survival_original_vs_fresh_certificates_by_round", a.png_dpi)

        fig, ax = plt.subplots(figsize=(6.9, 3.1))
        ax.plot(xround, [100*f(r, "original_conflict_graph_density") for r in rr], marker="o", label="Original conflict graph")
        ax.plot(xround, [100*f(r, "fresh_cert_conflict_graph_density") for r in rr], marker="o", linestyle="--", label="Fresh-Certificate shadow graph")
        ax.set_xlabel("Repeated compression round")
        ax.set_ylabel("Conflict edges / possible candidate-pair edges (%)")
        ax.set_title(f"Does accumulated C/R/D make the conflict graph denser? λ={primary:g}")
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        fig.tight_layout()
        save(fig, figs / "conflict_graph_density_original_vs_fresh_certificates", a.png_dpi)

        # 6) Explicit failure decomposition.
        route_rej = np.asarray([100*f(r, "original_route_rejection_fraction") for r in rr])
        mis_rej = np.asarray([100*f(r, "original_mis_rejection_fraction") for r in rr])
        recovered = np.asarray([100*f(r, "original_overall_recovery_rate") for r in rr])
        fig, ax = plt.subplots(figsize=(6.9, 3.2))
        ax.bar(xround, route_rej, label="Rejected before graph: no Γ≤Δ anchor")
        ax.bar(xround, mis_rej, bottom=route_rej, label="Routed but rejected by conflict-graph MIS")
        ax.bar(xround, recovered, bottom=route_rej+mis_rej, label="Recovered by merge")
        ax.set_xlabel("Repeated compression round")
        ax.set_ylabel("All evicted physical states (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"Where evicted states are lost in Original CohereKV, λ={primary:g}: routing rejection vs MIS rejection")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        save(fig, figs / "evicted_state_loss_decomposition_by_round", a.png_dpi)

        # 7) Certificate saturation itself.
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        ax.plot(xround, [f(r, "original_rhat_over_delta_p50") for r in rr], marker="o", label="median effective radius R̂/Δ before round")
        ax.plot(xround, [f(r, "original_rhat_over_delta_p90") for r in rr], marker="o", label="P90 effective radius R̂/Δ before round")
        ax.plot(xround, [f(r, "original_diameter_over_delta_p50") for r in rr], marker="s", label="median diameter D/Δ before round")
        ax.plot(xround, [f(r, "original_diameter_over_delta_p90") for r in rr], marker="s", label="P90 diameter D/Δ before round")
        ax.axhline(1.0, linestyle="--", linewidth=.8, label="Certificate budget Δ")
        ax.set_xlabel("Repeated compression round")
        ax.set_ylabel("Certificate value / fixed Δ")
        ax.set_title(f"Fixed-budget certificate saturation, λ={primary:g}: R̂/Δ and D/Δ across rounds")
        ax.legend(frameon=False, ncol=2)
        ax.grid(alpha=.2)
        fig.tight_layout()
        save(fig, figs / "certificate_saturation_rhat_and_diameter_over_delta", a.png_dpi)

        # 8) Remaining center-distance budget and exact minimum Gamma pressure.
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        ax.plot(xround, [f(r, "original_best_radius_only_slack_p50") for r in rr], marker="o", label="Median best radius-only slack: (Δ-R̂_src-min R̂_anchor)/Δ")
        ax.plot(xround, [f(r, "original_best_radius_only_slack_p10") for r in rr], marker="o", label="P10 best radius-only slack")
        ax.axhline(0.0, linewidth=.8, label="No center-distance budget remains")
        ax.set_xlabel("Repeated compression round")
        ax.set_ylabel("Remaining normalized center-distance budget")
        ax.set_title(f"How much fixed Δ is left for new center distance? λ={primary:g}")
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        fig.tight_layout()
        save(fig, figs / "remaining_merge_slack_delta_minus_radii_by_round", a.png_dpi)

        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        ax.plot(xround, [f(r, "original_min_gamma_over_delta_p50") for r in rr], marker="o", label="Median min-anchor Γ/Δ")
        ax.plot(xround, [f(r, "original_min_gamma_over_delta_p90") for r in rr], marker="o", label="P90 min-anchor Γ/Δ")
        ax.axhline(1.0, linestyle="--", linewidth=.8, label="Routing feasibility boundary Γ/Δ = 1")
        ax.set_xlabel("Repeated compression round")
        ax.set_ylabel("Best available anchor certificate Γ / fixed Δ")
        ax.set_title(f"Routing feasibility pressure, λ={primary:g}: even the best anchor increasingly exceeds fixed Δ")
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        fig.tight_layout()
        save(fig, figs / "minimum_anchor_gamma_over_delta_by_round", a.png_dpi)

    # 9) Lambda sensitivity at the final available repeated round.
    if rounds:
        final_by_lambda = []
        for lam in lambdas:
            sub = [r for r in rounds if close(r["lambda_mymethod"], lam)]
            if sub:
                final_by_lambda.append(max(sub, key=lambda r: int(float(r["round_idx"]))))
        fig, ax = plt.subplots(figsize=(6.8, 3.1))
        lx = [float(r["lambda_mymethod"]) for r in final_by_lambda]
        ax.plot(lx, [100*f(r, "original_candidate_admission_rate") for r in final_by_lambda], marker="o", label="Original candidate admission")
        ax.plot(lx, [100*f(r, "original_overall_recovery_rate") for r in final_by_lambda], marker="o", label="Original final recovery")
        ax.plot(lx, [100*f(r, "fresh_cert_candidate_admission_rate") for r in final_by_lambda], marker="o", linestyle="--", label="Fresh-Certificate candidate admission")
        ax.set_xlabel("Global compactness coefficient λ in Δ=λ·S_ref")
        ax.set_ylabel("Evicted states (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Final repeated-compression round: does a larger global Δ delay the collapse?")
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        fig.tight_layout()
        save(fig, figs / "final_round_recovery_vs_global_lambda", a.png_dpi)

    # Compact statistics used by paper drafting / sanity checking.
    selected_fields = [
        "lambda_mymethod", "round_idx", "input_length", "retained_length", "cumulative_retained_fraction",
        "mean_eviction_only_output_rel_error", "mean_compatible_merge_all_output_rel_error", "mean_coherekv_output_rel_error",
        "mean_fresh_certificate_output_rel_error", "mean_refreshed_scale_output_rel_error",
        "coherekv_error_reduction_vs_eviction", "original_candidate_admission_rate", "original_mis_survival_given_routed",
        "original_overall_recovery_rate", "original_conflict_graph_density", "original_rhat_over_delta_p90",
        "original_diameter_over_delta_p90", "original_best_radius_only_slack_p50", "original_min_gamma_over_delta_p90",
        "fresh_cert_candidate_admission_rate", "fresh_cert_mis_survival_given_routed", "fresh_cert_conflict_graph_density",
        "refreshed_scale_candidate_admission_rate", "current_scale_over_frozen_scale",
    ]
    with (figs / "figure_statistics.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=selected_fields)
        writer.writeheader()
        for r in sorted(rounds, key=lambda z: (float(z["lambda_mymethod"]), int(float(z["round_idx"])))):
            writer.writerow({k: r.get(k, "") for k in selected_fields})


if __name__ == "__main__":
    main()
