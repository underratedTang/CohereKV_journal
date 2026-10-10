#!/usr/bin/env python3
"""Generate deterministic focused synthetic CSVs for plot smoke tests only."""
from pathlib import Path
import csv
import json
import shutil


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

base = Path(__file__).resolve().parent / "synthetic_focused"
if base.exists(): shutil.rmtree(base)
mot=base/"motivation"; method=base/"method"; ppl=base/"ppl"
for d in (mot,method,ppl): d.mkdir(parents=True)
layers=[0,8,16,24,31]; heads=list(range(8)); metrics=["euclidean","cosine"]

lh=[]; overall=[]
for mi,metric in enumerate(metrics):
    gains=[]; prevs=[]
    route=.40 if metric=="euclidean" else .80
    for li,l in enumerate(layers):
        for h in heads:
            prevalence=.88+.01*((li+h)%4)
            gain=.007-.0005*mi+.0004*((2*li+h)%5)
            prevs.append(prevalence); gains.append(gain)
            lh.append({
                "routing_metric":metric,"config_id":"primary","is_primary":1,"route_param":route,"conflict_lambda":.10,"layer_idx":l,"kv_head_idx":h,
                "prefix_observations":4,"candidate_groups":40,"conflict_groups":int(round(40*prevalence)),
                "conflict_prevalence":prevalence,"intervention_groups":24,
                "intervention_gain_mean":gain,"intervention_positive_fraction":.82,
            })
    overall.append({
        "routing_metric":metric,"config_id":"primary","is_primary":1,"route_param":route,"conflict_lambda":.10,
        "candidate_groups":1600,"conflict_groups":1450,
        "conflict_prevalence":sum(prevs)/len(prevs),
        "conflict_reduced_gain":sum(gains)/len(gains),
        "conflict_reduced_positive_fraction":.82,"intervention_groups":960,
    })
write(mot/"q1_layer_head_summary.csv",lh)
write(mot/"q1_figure_overall.csv",overall)

route_rows=[]; cr_rows=[]
for metric, grid in [
    ("euclidean", [.30,.40,.50,.60,.70,.80]),
    ("cosine", [.70,.80,.90]),
]:
    mid = .60 if metric == "euclidean" else .80
    means=[]
    for p in grid:
        base_err=.105 + .020*(p-mid)**2 + (0.002 if metric=="cosine" else 0.0)
        means.append((p,base_err))
    best_p,best_err=min(means,key=lambda x:x[1])
    for p,e in means:
        route_rows.append({
            "routing_metric":metric,"route_param":p,"observations":160,
            "merge_all_output_rel_error":e,"is_best_routing_only":int(p==best_p),
            "best_route_param":best_p,"best_merge_all_output_rel_error":best_err,
        })
        for k in [1,2,4]:
            cr=e - (.0008*k) * (1.0 - 0.15*abs(p-mid))
            cr_rows.append({
                "routing_metric":metric,"route_param":p,"conflict_lambda":.50,
                "cr_num_removals":k,"observations":160,
                "conflict_reduced_output_rel_error":cr,
                "merge_all_output_rel_error":e,
                "cr_gain_vs_same_route":e-cr,
                "mean_actual_removed_sources":25.0*k,
                "mean_affected_star_groups":20.0,
                "best_route_param":best_p,"best_merge_all_output_rel_error":best_err,
                "is_best_routing_only":int(p==best_p),
            })
write(mot/"q2_routing_sweep.csv",route_rows)
write(mot/"q2_conflict_reduced_curve.csv",cr_rows)
(mot/"run_config.json").write_text(json.dumps({"experiment":"focused synthetic"},indent=2),encoding="utf-8")

single=[]
vals={"eviction_only":.095,"merge_all":.084,"original_coherekv":.091,"modified_coherekv":.079}
for v,e in vals.items():
    single.append({"config_id":"primary","is_primary":1,"variant":v,"observations":160,"output_rel_error":e,"output_rel_error_std":.01,
                   "output_p90_error":e*1.4,"accepted_entry_given_routed":.9,"conflict_edges":6})
write(method/"single_round_summary.csv",single)
(method/"run_config.json").write_text(json.dumps({"experiment":"focused synthetic"},indent=2),encoding="utf-8")

ppls={"full":7.80,"eviction_only":9.11,"merge_all":8.92,"original_coherekv":8.67,"modified_coherekv":8.19}
write(ppl/"ppl_summary.csv",[
    {"config_id":"primary","is_primary":1,"variant":v,"ppl":x,"wall_seconds":20+2*i,"model_seqlen":8192,"chunk_size":128,
     "compress_ratio":0.8,"retained_ratio":0.2,"num_compressions":18,"mean_compressed_kv_len":1638}
    for i,(v,x) in enumerate(ppls.items())
])
(ppl/"run_config.json").write_text(json.dumps({"experiment":"focused synthetic"},indent=2),encoding="utf-8")
print(mot); print(method); print(ppl)
