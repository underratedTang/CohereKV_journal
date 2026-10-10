# Focused CohereKV preliminary experiments — v12 whole-layer Figure-2/3-left metrics

Only `motivation_experiments/` is changed. Production code and `task_evaluation/` remain untouched.

This version leaves Figure 1, Figure 2, Modified CohereKV, Figure 3-right/PPL, and all thresholds unchanged from v11. The only experiment change is Figure 3-left: its single-round metric now matches Figure 2 exactly by evaluating the complete attention-layer output after the real `o_proj`. All three paper figures request Times New Roman through the shared plotting entry point.

## Frozen calibration

Calibration keeps the validated multi-sequence protocol:

- 16 reproducible, non-overlapping WikiText-2 train windows;
- 8192 tokens/window;
- seed 42;
- per-sequence, per-layer/head scale `S_{l,h,m}`;
- frozen operational scale

\[
S_{l,h}^{calib}=\sqrt{\frac1M\sum_m S_{l,h,m}^{2}}.
\]

Figure 1/2/3 share this frozen `S_{l,h}^{calib}`. Original CohereKV remains unchanged.

## Motivation-only Conflict-Reduced counterfactual

Figure 1/2 do **not** run Modified CohereKV, MIS, degree priority, or any deployable source-selection rule.

For a routed Star group, conflicts are first identified only from K-space:

\[
conflict(u,v)\iff \|K_u-K_v\|_2/S_{l,h}^{calib}>\lambda_{conflict}.
\]

At each local counterfactual step:

1. find the currently most severe conflict edge
   \[
   (u,v)=\arg\max_{(i,j)\in E_{conflict}} \|K_i-K_j\|_2/S_{l,h}^{calib};
   \]
2. evaluate the two legal filtering outcomes:
   - evict `u`, keep `v` eligible for Merge-All;
   - evict `v`, keep `u` eligible for Merge-All;
3. follow the branch with lower attention-output error;
4. recompute conflicts inside that same Star group and repeat for at most `K` steps.

A filtered source is **evicted**; it is not retained as an independent KV entry. Attention-output error is used only for this offline counterfactual branch comparison. This measures the potential of conflict-aware filtering; it is not the Figure-3 source-selection rule.

## Figure 1 — Motivation 1

Goal: show that Star conflicts are widespread across layers/KV heads and that resolving a small number of severe conflicts can improve Merge-All.

Default 8B preliminary operating point:

- Euclidean route lambda: `0.60`;
- cosine route lambda: `0.80`;
- conflict lambda: `0.50`;
- primary per-Star-group counterfactual budget: `K=1`.

Optional compact OFAT values remain available:

- Euclidean route: `0.40,0.50,0.60,0.70`;
- cosine route: `0.70,0.80,0.90`;
- conflict: `0.40,0.50,0.60`;
- counterfactual steps `K`: `1,2,4`.

Figure 1 uses the requested **2 x 3** layout:

- top row: cosine routing;
- bottom row: Euclidean routing;
- column 1: Star-conflict prevalence;
- column 2: mean `Merge-All error - Conflict-Reduced error`;
- column 3: fraction of sampled conflict groups improved by the counterfactual.

Prevalence uses all eligible Star groups. The intervention analysis remains bounded by `FIG1_MAX_GROUPS_PER_HEAD` sampled conflict groups/head.

## Figure 2 — Motivation 2

Goal: show directly that explicit conflict reduction can outperform **only tuning the Merge-All routing threshold**.

The x-axis is now the routing threshold itself. For every threshold in the compact routing grid, Figure 2 computes:

- `Merge-All(route_threshold)`;
- `Conflict-Reduced(route_threshold, K)`.

The per-Star definition of `K` remains unchanged from v10 and has **exactly the same meaning as Figure 1**:

\[
\boxed{K=\text{maximum conflicts resolved independently in each Star group}}
\]

It is no longer a single budget shared by the whole KV head. For every Star group in every KV head, the Figure-1 local counterfactual is run independently. Figure 2 then reconstructs the compressed attention output for **all query heads in the layer**, concatenates them in model order, applies that layer's real attention `o_proj`, and computes a single whole-layer relative error. It does **not** average per-head relative errors.

Default Figure-2 setup:

- Euclidean routing grid: `0.30,0.40,0.50,0.60,0.70,0.80`;
- cosine routing grid: `0.70,0.80,0.90`;
- default conflict lambda: `0.50`;
- primary CR curve: `K=1` per Star group.

By default the plot therefore contains only two curves per routing metric:

- `Merge-All`;
- `Conflict-Reduced (K=1/Star)`.

Set:

```bash
FIG2_RUN_CR_GRID="1"
```

to additionally evaluate and plot the compact grid:

```bash
FIG2_CR_NUM_REMOVALS_GRID="1,2,4"
```

which produces one CR curve per K. The best Merge-All routing point is marked directly on the Merge-All curve, so the reviewer can see the tuned baseline from the same plot rather than relying on a separate horizontal reference.

`FIG2_RUN_HPARAM_SWEEP=1` remains a separate OFAT diagnostic for the conflict threshold and is not part of the main Figure-2 plot.

### Figure-2 error metric

For one prefix/layer/configuration, let the pre-`o_proj` query-head outputs be concatenated as

\[
O_{\mathrm{cat}} = \operatorname{Concat}(O_1,\ldots,O_{H_q}),
\]

and let `o_proj` be the model's actual output projection for that attention layer. Figure 2 reports

\[
E_{\mathrm{layer}} =
\frac{\|W_O O_{\mathrm{cat}}^{\mathrm{compressed}} - W_O O_{\mathrm{cat}}^{\mathrm{full}}\|_2}
{\|W_O O_{\mathrm{cat}}^{\mathrm{full}}\|_2}.
\]

Thus one observation is `prefix x layer x routing configuration`, not `prefix x layer x KV-head`. `kv_head_selection` continues to control Figure 1 only; Figure 2 always evaluates all KV/query heads so the metric is a true whole-layer output.

## Figure 3 — Improved CohereKV

### Left: single-round whole-layer attention-output error

Compares:

- Eviction-only;
- Merge-All;
- Original CohereKV;
- Modified CohereKV.

The compression methods, partitions, thresholds, routing/conflict logic, and one-round protocol are unchanged. Only the evaluation metric is changed to match Figure 2 exactly. For every selected layer, Figure 3-left always evaluates **all** KV/query heads, concatenates the pre-`o_proj` query-head outputs in model order, applies the layer's real `o_proj`, and reports

\[
E_{\mathrm{layer}} =
\frac{\|W_O O_{\mathrm{cat}}^{\mathrm{compressed}}-W_O O_{\mathrm{cat}}^{\mathrm{full}}\|_2}
{\|W_O O_{\mathrm{cat}}^{\mathrm{full}}\|_2}.
\]

Thus one Figure-3-left observation is `prefix x layer x method`, not `prefix x layer x KV-head`. `kv_head_selection` is retained for launch-script compatibility but is intentionally ignored by Figure 3-left because a true whole-layer metric requires every head. Raw observations are written to `single_round_by_layer.csv`.

### Right: repository PPL

`ppl_continuation_experiment.py` still directly calls:

`task_evaluation/evaluate_ppl_kvmerge.py::evaluate_perplexity`

and compares:

- Full;
- Eviction-only;
- Merge-All;
- Original CohereKV;
- Modified CohereKV.

Figure-3 PPL settings and implementation are unchanged from v9.

## Modified CohereKV — unchanged

Modified still uses frozen `S_{l,h}^{calib}` and tensorized MIS priority:

`degree -> anchor distance -> stable hash -> physical position`.

No importance term is used in MIS.

## Plotting style

`plot_motivation_attention.py` applies one shared publication style to Figure 1/2/3:

- font family requested: **Times New Roman**;
- STIX math text;
- PDF/PS font type 42.

If Times New Roman is unavailable in the execution environment, Matplotlib falls back to the configured serif fallback; on the target plotting machine install/enable Times New Roman to obtain the exact requested face.

## Run

Edit `motivation_experiments/run_motivation_attention.sh`, then run:

```bash
bash /home/tlxr/CohereKV/motivation_experiments/run_motivation_attention.sh
```

The default preliminary suite keeps all optional sweeps off, except the mandatory compact Figure-2 routing-threshold curve.
