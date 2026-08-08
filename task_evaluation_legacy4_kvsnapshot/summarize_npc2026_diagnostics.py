#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize NPC 2026 diagnostic outputs into reviewer-facing checks.

The script intentionally reports measurements and warning thresholds rather
than selecting or hiding unfavorable results.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def safe_max(values: Iterable[float]) -> float | None:
    xs = list(values)
    return max(xs) if xs else None


def safe_mean(values: Iterable[float]) -> float | None:
    xs = list(values)
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_md", required=True)
    args = p.parse_args()

    root = Path(args.results_dir)
    summary: dict[str, Any] = {
        "results_dir": str(root.resolve()),
        "grouping": {},
        "ppl": {},
        "coverage": {},
        "microbenchmark": {},
        "warnings": [],
    }

    grouping_path = root / "grouping_diagnostics.json"
    if grouping_path.exists():
        payload = json.loads(grouping_path.read_text(encoding="utf-8"))
        rows = payload.get("rows", [])
        one_round = [r for r in rows if r.get("analysis") == "one_round"]
        multiround = [r for r in rows if r.get("analysis") == "multiround"]
        by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in one_round:
            by_variant[str(row.get("variant"))].append(row)
        for variant, values in sorted(by_variant.items()):
            summary["grouping"][variant] = {
                "rows": len(values),
                "max_group_violation_rate": safe_max(
                    float(v.get("group_violation_rate", 0.0)) for v in values
                ),
                "max_diameter_over_delta": safe_max(
                    float(v.get("max_diameter_over_delta", 0.0)) for v in values
                ),
                "mean_accepted_entry_ratio": safe_mean(
                    float(v.get("accepted_entry_ratio", 0.0)) for v in values
                    if "accepted_entry_ratio" in v
                ),
                "mean_rescued_entry_ratio": safe_mean(
                    float(v.get("rescued_entry_ratio", 0.0)) for v in values
                    if "rescued_entry_ratio" in v
                ),
            }
        cert_multi = [r for r in multiround if r.get("variant") == "mymethod_certified"]
        summary["grouping"]["mymethod_multiround"] = {
            "rows": len(cert_multi),
            "max_exact_diameter_violation_count": safe_max(
                float(r.get("diameter_violation_count", 0.0)) for r in cert_multi
            ),
            "max_radius_certificate_underestimate_count": safe_max(
                float(r.get("radius_certificate_underestimate_count", 0.0)) for r in cert_multi
            ),
            "max_diameter_certificate_underestimate_count": safe_max(
                float(r.get("diameter_certificate_underestimate_count", 0.0)) for r in cert_multi
            ),
        }
        cert = summary["grouping"].get("mymethod_certified", {})
        if float(cert.get("max_group_violation_rate") or 0.0) > 0.0:
            summary["warnings"].append("Certified one-round groups contain exact diameter violations.")
        if any(float(summary["grouping"]["mymethod_multiround"].get(k) or 0.0) > 0.0 for k in (
            "max_exact_diameter_violation_count",
            "max_radius_certificate_underestimate_count",
            "max_diameter_certificate_underestimate_count",
        )):
            summary["warnings"].append("Multiround certificate invariant failed; do not use main quality results yet.")

    for path in sorted(root.glob("ppl_*.jsonl")):
        if path.name.endswith("_diagnostics.jsonl"):
            continue
        rows = read_jsonl(path)
        summary["ppl"][path.stem] = [
            {
                "dataset": r.get("dataset"),
                "ppl": r.get("ppl"),
                "accepted_entry_ratio": r.get("mymethod_mean_accepted_entry_ratio"),
                "accepted_count_mass_ratio": r.get("mymethod_mean_accepted_count_mass_ratio"),
                "rejected_count_mass": r.get("mymethod_total_rejected_count_mass"),
            }
            for r in rows
        ]
        masses = [
            float(r["mymethod_mean_accepted_count_mass_ratio"])
            for r in rows
            if r.get("mymethod_mean_accepted_count_mass_ratio") is not None
        ]
        if masses and min(masses) < 0.2:
            summary["warnings"].append(
                f"{path.stem}: accepted count-mass ratio falls below 0.2; quality may be certificate-coverage limited."
            )

    for path in sorted(root.glob("ppl_*_diagnostics.jsonl")):
        rows = read_jsonl(path)
        evicted = sum(float(r.get("evicted_count_mass", 0.0)) for r in rows)
        accepted = sum(float(r.get("accepted_count_mass", 0.0)) for r in rows)
        summary["coverage"][path.stem] = {
            "events": len(rows),
            "accepted_count_mass_ratio": accepted / evicted if evicted > 0 else None,
            "total_rejected_count_mass": sum(float(r.get("rejected_count_mass", 0.0)) for r in rows),
            "max_post_diameter_over_delta": safe_max(
                float(r.get("post_diameter_over_delta_max", 0.0)) for r in rows
            ),
        }

    bench_path = root / "compression_microbenchmark.json"
    if bench_path.exists():
        payload = json.loads(bench_path.read_text(encoding="utf-8"))
        rows = payload.get("rows", [])
        variants: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            variants[str(row.get("variant"))].append(row)
        for variant, values in sorted(variants.items()):
            summary["microbenchmark"][variant] = {
                "rows": len(values),
                "median_latency_ms_across_cases": safe_mean(
                    float(v.get("latency_ms_median", 0.0)) for v in values
                ),
                "mean_million_pairs_per_second": safe_mean(
                    float(v.get("million_pairs_per_second", 0.0)) for v in values
                    if "million_pairs_per_second" in v
                ),
                "max_peak_temporary_mib": safe_max(
                    float(v.get("peak_temporary_mib", 0.0)) for v in values
                ),
            }

    # De-duplicate warning text without changing order.
    summary["warnings"] = list(dict.fromkeys(summary["warnings"]))

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# NPC 2026 Diagnostic Summary", ""]
    if summary["warnings"]:
        lines += ["## Blocking or high-priority warnings", ""]
        lines += [f"- {x}" for x in summary["warnings"]]
    else:
        lines += ["## Blocking or high-priority warnings", "", "- None detected from available files."]
    lines += ["", "## Grouping diagnostics", "", "```json", json.dumps(summary["grouping"], ensure_ascii=False, indent=2), "```"]
    lines += ["", "## PPL and coverage", "", "```json", json.dumps({"ppl": summary["ppl"], "coverage": summary["coverage"]}, ensure_ascii=False, indent=2), "```"]
    lines += ["", "## Compression microbenchmark", "", "```json", json.dumps(summary["microbenchmark"], ensure_ascii=False, indent=2), "```", ""]
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
