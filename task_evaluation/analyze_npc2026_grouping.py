#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze the empirical necessity and behavior of certified complete groups.

Input is produced by ``capture_npc2026_kv_snapshot.py``. The analysis fixes the
retained/evicted split to the GroupLogitKV selector so grouping mechanisms are
compared under identical cache budgets and selectors.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


def parse_floats(spec: str) -> List[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def q(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    x = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(x, p).item())


def exact_assignment_metrics(
    keys: torch.Tensor,
    keep_idx: torch.Tensor,
    evict_idx: torch.Tensor,
    best_idx: torch.Tensor,
    accepted: torch.Tensor,
    delta: float,
    scale: float,
) -> Dict[str, float]:
    """Exact one-round metrics when all incoming entries are singleton tokens."""
    keys = keys.float().cpu()
    keep_idx = keep_idx.long().cpu()
    evict_idx = evict_idx.long().cpu()
    best_idx = best_idx.long().cpu()
    accepted = accepted.bool().cpu()
    groups: List[List[int]] = [[int(x)] for x in keep_idx.tolist()]
    for src_local in torch.nonzero(accepted, as_tuple=False).flatten().tolist():
        groups[int(best_idx[src_local])].append(int(evict_idx[src_local]))

    diameters: List[float] = []
    radii: List[float] = []
    sizes: List[int] = []
    candidate_counts: List[int] = []
    violations = 0
    multi_candidate_violations = 0
    tol = max(1e-6, abs(delta) * 2e-5)
    for members in groups:
        if len(members) <= 1:
            continue
        group = keys[members]
        if group.shape[0] <= 1:
            diameter = 0.0
        else:
            diameter = float(torch.pdist(group, p=2).max().item())
        centroid = group.mean(dim=0)
        radius = float(torch.linalg.vector_norm(group - centroid, dim=-1).max().item())
        diameters.append(diameter)
        radii.append(radius)
        sizes.append(len(members))
        candidate_counts.append(len(members) - 1)
        bad = diameter > delta + tol
        violations += int(bad)
        multi_candidate_violations += int(bad and len(members) >= 3)

    merged_groups = len(diameters)
    accepted_count = int(accepted.sum().item())
    return {
        "accepted_entries": accepted_count,
        "evicted_entries": int(evict_idx.numel()),
        "accepted_entry_ratio": accepted_count / max(int(evict_idx.numel()), 1),
        "merged_groups": merged_groups,
        "group_violation_count": violations,
        "group_violation_rate": violations / max(merged_groups, 1),
        "multi_candidate_violation_count": multi_candidate_violations,
        "mean_group_size": sum(sizes) / max(len(sizes), 1),
        "max_group_size": max(sizes, default=1),
        "mean_candidates_per_merged_group": sum(candidate_counts) / max(len(candidate_counts), 1),
        "mean_diameter": sum(diameters) / max(len(diameters), 1),
        "p50_diameter": q(diameters, 0.50),
        "p95_diameter": q(diameters, 0.95),
        "max_diameter": max(diameters, default=0.0),
        "p95_diameter_over_delta": q([d / max(delta, 1e-12) for d in diameters], 0.95),
        "max_diameter_over_delta": max([d / max(delta, 1e-12) for d in diameters], default=0.0),
        "p95_radius_over_scale": q([r / max(scale, 1e-12) for r in radii], 0.95),
    }


def aggregate(rows: List[dict]) -> List[dict]:
    grouped: Dict[Tuple, List[dict]] = defaultdict(list)
    key_fields = ("analysis", "variant", "compress_ratio", "tau", "round")
    for row in rows:
        grouped[tuple(row.get(k) for k in key_fields)].append(row)
    out = []
    for key, items in grouped.items():
        summary = {k: v for k, v in zip(key_fields, key)}
        numeric = sorted({k for r in items for k, v in r.items() if isinstance(v, (int, float))})
        for name in numeric:
            vals = [float(r[name]) for r in items if isinstance(r.get(name), (int, float))]
            summary[f"mean_{name}"] = sum(vals) / max(len(vals), 1)
        summary["num_rows"] = len(items)
        out.append(summary)
    return out


def update_members(
    members: List[List[int]], keep_idx: torch.Tensor, evict_idx: torch.Tensor,
    best_idx: torch.Tensor, accepted: torch.Tensor,
) -> List[List[int]]:
    keep = keep_idx.long().cpu().tolist()
    evict = evict_idx.long().cpu().tolist()
    best = best_idx.long().cpu().tolist()
    valid = accepted.bool().cpu().tolist()
    new_members = [list(members[i]) for i in keep]
    for src_local, ok in enumerate(valid):
        if ok:
            new_members[best[src_local]].extend(members[evict[src_local]])
    return new_members


def exact_state_metrics(
    original_keys: torch.Tensor,
    members: List[List[int]],
    delta: float,
    diameter_certificate: torch.Tensor | None = None,
    radius_certificate: torch.Tensor | None = None,
    centroids: torch.Tensor | None = None,
) -> Dict[str, float]:
    original_keys = original_keys.float().cpu()
    dvals, rvals, dratios = [], [], []
    diameter_under = 0
    radius_under = 0
    violations = 0
    cert_d = None if diameter_certificate is None else diameter_certificate.float().cpu().flatten()
    cert_r = None if radius_certificate is None else radius_certificate.float().cpu().flatten()
    centers = None if centroids is None else centroids.float().cpu().reshape(len(members), -1)
    tol = max(1e-6, abs(delta) * 2e-5)
    for idx, group_ids in enumerate(members):
        group = original_keys[group_ids]
        diameter = 0.0 if len(group_ids) <= 1 else float(torch.pdist(group, p=2).max().item())
        center = group.mean(dim=0) if centers is None else centers[idx]
        radius = float(torch.linalg.vector_norm(group - center, dim=-1).max().item())
        dvals.append(diameter)
        rvals.append(radius)
        dratios.append(diameter / max(delta, 1e-12))
        violations += int(diameter > delta + tol)
        if cert_d is not None:
            diameter_under += int(float(cert_d[idx]) + tol < diameter)
        if cert_r is not None:
            radius_under += int(float(cert_r[idx]) + tol < radius)
    return {
        "entries": len(members),
        "diameter_violation_count": violations,
        "diameter_violation_rate": violations / max(len(members), 1),
        "max_exact_diameter_over_delta": max(dratios, default=0.0),
        "p95_exact_diameter_over_delta": q(dratios, 0.95),
        "max_exact_diameter": max(dvals, default=0.0),
        "max_exact_radius": max(rvals, default=0.0),
        "diameter_certificate_underestimate_count": diameter_under,
        "radius_certificate_underestimate_count": radius_under,
    }


def multiround_one_head(
    common,
    original_keys: torch.Tensor,
    original_importance: torch.Tensor,
    target_ratios: Sequence[float],
    tau: float,
    device: torch.device,
) -> List[dict]:
    """Compare certificate propagation with an ordinary centroid-distance star gate."""
    rows: List[dict] = []
    original = original_keys.float().cpu()
    original_len, dim = original.shape

    # mymethod state
    mk = original.to(device).view(1, 1, original_len, dim)
    mv = torch.zeros_like(mk)
    mc = torch.ones((1, 1, original_len), dtype=torch.int32, device=device)
    mr = torch.zeros((1, 1, original_len), dtype=torch.float32, device=device)
    md = torch.zeros_like(mr)
    mp = torch.arange(original_len, dtype=torch.int32, device=device).view(1, 1, -1)
    mh = torch.zeros((1, 1, 0, original_len), dtype=torch.float32, device=device)
    mi = original_importance.float().to(device).view(1, 1, -1)
    mmembers = [[i] for i in range(original_len)]

    # ordinary star-gate state
    sk = original.to(device)
    sc = torch.ones((original_len,), dtype=torch.float32, device=device)
    si = original_importance.float().to(device)
    smembers = [[i] for i in range(original_len)]

    for round_idx, retained_ratio in enumerate(target_ratios, start=1):
        target = max(1, int(round(original_len * retained_ratio)))
        if target >= mk.shape[-2]:
            continue

        # --- certified mymethod ---
        cur_len = int(mk.shape[-2])
        flat_k = mk.reshape(1, cur_len, dim)
        flat_c = mc.reshape(1, cur_len).float()
        flat_r = mr.reshape(1, cur_len)
        flat_d = md.reshape(1, cur_len)
        flat_i = mi.reshape(1, cur_len)
        flat_p = mp.reshape(1, cur_len)
        plan = common.build_merge_plan(
            flat_k, flat_c, flat_r, flat_d, flat_i, flat_p,
            keep_len=target, tau=tau, selector="grouplogitkv",
            sink_size=0, recent_window=0, distance_chunk_size=0,
            certify_groups=True, route_mode="nearest_feasible",
        )
        old_counts = mc.reshape(1, cur_len).float()
        imp_m = common.gather_seq(flat_i, plan.keep_idx)
        imp_n = common.gather_seq(flat_i, plan.evict_idx)
        cm = common.gather_seq(old_counts, plan.keep_idx)
        cn = common.gather_seq(old_counts, plan.evict_idx)
        accepted_weights = cn * plan.accepted.float()
        add_imp = torch.zeros((1, target), device=device, dtype=torch.float32)
        add_imp.scatter_add_(1, plan.best_idx, accepted_weights * imp_n)
        new_count_float = cm + torch.zeros_like(cm).scatter_add_(1, plan.best_idx, accepted_weights)
        new_imp = (cm * imp_m + add_imp) / new_count_float.clamp_min(1.0)
        mmembers = update_members(
            mmembers, plan.keep_idx[0], plan.evict_idx[0],
            plan.best_idx[0], plan.accepted[0],
        )
        mk, mv, mc, mr, md, mh, mp = common.merge_layer(
            mk, mv, mc, mr, md, mh, 0, mp,
            ratio=0.0, target_length=target, tau=tau,
            sink_size=0, recent_window=0, recency_beta=1.0,
            distance_chunk_size=0, certify_groups=True,
            sink_can_absorb=True, selector="grouplogitkv",
            current_importance=mi, route_mode="nearest_feasible",
        )
        mi = new_imp.view(1, 1, target)
        delta_m = float(plan.delta.item())
        row = {
            "analysis": "multiround",
            "variant": "mymethod_certified",
            "round": round_idx,
            "retained_ratio": retained_ratio,
            "tau": tau,
            "compress_ratio": 1.0 - retained_ratio,
            "delta": delta_m,
            "accepted_entry_ratio": float(plan.accepted.float().mean().item()),
        }
        row.update(exact_state_metrics(
            original, mmembers, delta_m,
            diameter_certificate=md[0, 0], radius_certificate=mr[0, 0],
            centroids=mk[0, 0],
        ))
        rows.append(row)

        # --- ordinary star gate d(source centroid, anchor centroid) <= Delta ---
        cur_len = int(sk.shape[0])
        if target >= cur_len:
            continue
        keep = common.select_keep_indices_grouplogitkv(si.view(1, -1), target)[0]
        evict = common._complement_indices(keep.view(1, -1), cur_len)[0]
        kk = sk.index_select(0, keep)
        ek = sk.index_select(0, evict)
        numerator = (sc * sk.square().sum(dim=-1)).sum()
        scale = torch.sqrt(numerator / sc.sum().clamp_min(1.0)) + 1e-6
        delta = float((scale * (1.0 - tau)).item())
        dist = torch.cdist(ek.float(), kk.float(), p=2)
        best_dist, best = dist.min(dim=-1)
        accepted = best_dist <= delta
        cm_s = sc.index_select(0, keep)
        cn_s = sc.index_select(0, evict)
        add_c = torch.zeros((target,), device=device).scatter_add_(0, best, cn_s * accepted.float())
        new_c = cm_s + add_c
        add_k = torch.zeros((target, dim), device=device)
        add_k.scatter_add_(0, best[:, None].expand(-1, dim),
                           ek * (cn_s * accepted.float())[:, None])
        sk = (kk * cm_s[:, None] + add_k) / new_c[:, None].clamp_min(1.0)
        imp_m_s = si.index_select(0, keep)
        imp_n_s = si.index_select(0, evict)
        add_i = torch.zeros((target,), device=device).scatter_add_(
            0, best, imp_n_s * cn_s * accepted.float()
        )
        si = (imp_m_s * cm_s + add_i) / new_c.clamp_min(1.0)
        sc = new_c
        smembers = update_members(smembers, keep, evict, best, accepted)
        row = {
            "analysis": "multiround",
            "variant": "ordinary_star_delta",
            "round": round_idx,
            "retained_ratio": retained_ratio,
            "tau": tau,
            "compress_ratio": 1.0 - retained_ratio,
            "delta": delta,
            "accepted_entry_ratio": float(accepted.float().mean().item()),
        }
        row.update(exact_state_metrics(original, smembers, delta, centroids=sk))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--task_eval_dir", default=str(Path(__file__).resolve().parent))
    p.add_argument("--compress_ratios", default="0.2,0.4,0.6")
    p.add_argument("--taus", default="0.6,0.7,0.8,0.9")
    p.add_argument("--grouplogitkv_sim_thr", type=float, default=0.8)
    p.add_argument("--max_heads_per_layer", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--multiround_retained_ratios", default="0.8,0.6,0.4")
    p.add_argument("--output_json", required=True)
    args = p.parse_args()

    task_dir = Path(args.task_eval_dir).resolve()
    if str(task_dir) not in sys.path:
        sys.path.insert(0, str(task_dir))
    from models.common import mymethod_common as common

    try:
        payload = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions predating the weights_only argument.
        payload = torch.load(args.snapshot, map_location="cpu")
    ratios = parse_floats(args.compress_ratios)
    taus = parse_floats(args.taus)
    target_ratios = parse_floats(args.multiround_retained_ratios)
    device = torch.device(args.device)
    rows: List[dict] = []

    first_multiround = None
    for layer in payload["layers"]:
        layer_idx = int(layer["layer_idx"])
        keys = layer["keys"]
        importance = layer["importance"]
        bsz, heads, length, dim = keys.shape
        use_heads = min(heads, max(1, args.max_heads_per_layer))
        for head_idx in range(use_heads):
            row_keys = keys[0, head_idx].to(device).float()
            row_imp = importance[0, head_idx].to(device).float()
            if first_multiround is None:
                first_multiround = (row_keys.detach().cpu(), row_imp.detach().cpu(), layer_idx, head_idx)
            k3 = row_keys.view(1, length, dim)
            counts = torch.ones((1, length), device=device)
            zeros = torch.zeros((1, length), device=device)
            positions = torch.arange(length, device=device, dtype=torch.int32).view(1, -1)
            imp2 = row_imp.view(1, -1)
            for ratio in ratios:
                keep_len = max(1, min(length, int(round(length * (1.0 - ratio)))))
                if keep_len >= length:
                    continue
                for tau in taus:
                    cert = common.build_merge_plan(
                        k3, counts, zeros, zeros, imp2, positions,
                        keep_len=keep_len, tau=tau, selector="grouplogitkv",
                        sink_size=0, recent_window=0, distance_chunk_size=0,
                        certify_groups=True, route_mode="nearest_feasible",
                    )
                    nearest = common.build_merge_plan(
                        k3, counts, zeros, zeros, imp2, positions,
                        keep_len=keep_len, tau=tau, selector="grouplogitkv",
                        sink_size=0, recent_window=0, distance_chunk_size=0,
                        certify_groups=False, route_mode="nearest_feasible",
                    )
                    nearest_then = common.build_merge_plan(
                        k3, counts, zeros, zeros, imp2, positions,
                        keep_len=keep_len, tau=tau, selector="grouplogitkv",
                        sink_size=0, recent_window=0, distance_chunk_size=0,
                        certify_groups=True, route_mode="nearest_then_certify",
                    )
                    delta = float(cert.delta[0, 0].item())
                    scale = float(cert.scale[0, 0].item())
                    common_meta = {
                        "analysis": "one_round",
                        "layer": layer_idx,
                        "head": head_idx,
                        "compress_ratio": ratio,
                        "tau": tau,
                        "keep_len": keep_len,
                        "length": length,
                        "delta": delta,
                        "scale": scale,
                    }

                    variants = []
                    variants.append(("mymethod_certified", cert.best_idx[0], cert.accepted[0]))
                    variants.append(("nearest_then_certify", nearest_then.best_idx[0], nearest_then.accepted[0]))
                    star_accept = nearest.assigned_distance[0] <= delta
                    variants.append(("ordinary_star_delta", nearest.best_idx[0], star_accept))

                    keep_keys = row_keys.index_select(0, cert.keep_idx[0])
                    evict_keys = row_keys.index_select(0, cert.evict_idx[0])
                    cosine = torch.matmul(
                        F.normalize(evict_keys, dim=-1),
                        F.normalize(keep_keys, dim=-1).transpose(0, 1),
                    )
                    best_sim, best_cos_idx = cosine.max(dim=-1)
                    gl_accept = (
                        torch.ones_like(best_sim, dtype=torch.bool)
                        if args.grouplogitkv_sim_thr <= -1.0
                        else best_sim >= args.grouplogitkv_sim_thr
                    )
                    variants.append(("grouplogitkv_cosine", best_cos_idx, gl_accept))

                    for name, best_idx, accepted in variants:
                        result = dict(common_meta)
                        result["variant"] = name
                        result.update(exact_assignment_metrics(
                            row_keys, cert.keep_idx[0], cert.evict_idx[0],
                            best_idx, accepted, delta, scale,
                        ))
                        if name == "ordinary_star_delta":
                            result["false_safe_group_rate"] = result["group_violation_rate"]
                        rows.append(result)

                    rescued = cert.accepted[0] & ~nearest_then.accepted[0]
                    rows.append({
                        **common_meta,
                        "variant": "nearest_feasible_rescue",
                        "evicted_entries": int(cert.evict_idx.shape[1]),
                        "rescued_entries": int(rescued.sum().item()),
                        "rescued_entry_ratio": float(rescued.float().mean().item()),
                        "certified_acceptance_gain": float(
                            cert.accepted.float().mean().item()
                            - nearest_then.accepted.float().mean().item()
                        ),
                    })

    if first_multiround is not None and target_ratios and taus:
        keys0, imp0, layer_idx, head_idx = first_multiround
        multi = multiround_one_head(common, keys0, imp0, target_ratios, taus[0], device)
        for row in multi:
            row["layer"] = layer_idx
            row["head"] = head_idx
        rows.extend(multi)

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = aggregate(rows)
    payload_out = {
        "snapshot_metadata": {k: v for k, v in payload.items() if k != "layers"},
        "config": vars(args),
        "rows": rows,
        "aggregate": summary,
    }
    output.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2))
    write_csv(output.with_suffix(".csv"), rows)
    output.with_name(output.stem + "_aggregate.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote: {output}")
    print(f"Wrote: {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
