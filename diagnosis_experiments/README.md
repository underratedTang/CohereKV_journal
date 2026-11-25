# CohereKV journal v9 — independent geometry p and decay gamma

Replacement of `diagnosis_experiments/` only, based on the uploaded v8 source.
No external repository code is changed. Eight method switches default to 1.
Use `p` for historical geometry and `gamma` for shared EA/EE decay (alpha=beta=gamma).
They are independent, not required to be different numerically.

## Run

Edit the literal settings in `run_soft_mis.sh` (no environment-input overrides).
The uploaded source's `ROOT_DIR="/root/CohereKV"`, active model
`meta-llama/Llama-3.2-3B` and GPU selection are retained. Change ROOT_DIR and MODEL
in that file to match your server and target model (e.g. Qwen/Qwen2.5-7B).

```bash
cd /root/CohereKV
python diagnosis_experiments/test_soft_mis.py
python diagnosis_experiments/test_soft_mis_cache.py
python diagnosis_experiments/test_sweep_framework.py
bash diagnosis_experiments/run_soft_mis.sh
```

## Algorithm

Nearest physical-Key routing is unchanged for journal methods. Historical EA
eligibility additionally checks the cross bound and existing diameters against
history_limit=max(Delta_EA,Delta_EE).
For finite gamma, define EA risk=max(cross_EA/Delta_EA, D_i/history_limit,
D_A/history_limit), a_i=min(1,risk^(-gamma)), and
c_ij=1-min(1,(Delta_EE/cross_EE)^gamma) on the conflict graph.
For gamma=inf use the exact Boolean EA eligibility and EE adjacency, with
the original threshold tolerance. The symmetrized threshold graph is used
consistently for suppression and DAG orientation (including boundary roundoff).
Priority and fixed ordering, including the hard-EA priority pool, remain the
v8 design; importance costs use the new gamma-dependent c_ij.
g_i=a_i*product_{higher priority j}(1-c_ij*g_j).
All historical geometry updates use p, never gamma. For p=inf and finite gamma,
the max-radius bound uses the weighted mean and Minkowski update without
hard-Count radius tightening; tightening by (C-1)/C is used only when both
p and gamma are infinite. A tiny positive weight still represents a full
historical member for the maximum geometry bound.
For finite gamma, strict maximum-diameter compactness is not guaranteed even
when p=inf. For finite p, geometry describes the recursively renormalized
distribution (RMS at p=2, moment upper bounds otherwise).

Fusion mass is Count*g, with true Count for next round. Contribution A
is the last successful fusion denominator for attention only; unchanged
anchors preserve their old contribution. The _log_count arms use true Count
for attention. All methods reset their history and caches at each sequence.

## Defaults

WikiText-2 validation; frozen train calibration (four 4096-token sequences).
U=256, S=32, cycles=24, 6912 logical tokens/window, 768 scored targets/window,
budget .10 (26 post-compression slots/head), score/recent protection 8,
sink 4, chunk 128. Eight non-overlapping indices 0..7 requested. Preflight
excludes incomplete windows; actual indices are recorded. All methods use
the same windows. No padding, wrap or repeated text.

- p = 1, 2, inf
- gamma = 1, 2, 4, inf
- EA = .30, .50; EE = .30, .50, .70
- 252 actual candidate configurations + 6 baseline configurations = 258
  actual evaluations/window with all switches enabled, at most 2064 for
  eight windows. Calibration is additional. Grid can be reduced in the SH.
- Change EXPERIMENT_NAME for independent reruns; RUN_PPL=1 replaces that
  experiment's sequence/aggregate results. RUN_CALIBRATION=1 with RUN_PPL=1
  and compressed methods enabled removes the same named old output first.

## Methods

- `full_cache`: native unpatched model, same scored positions; no pruning.
- `eviction_only`: KeepKV, no fusion; only budget matters.
- `merge_all`: physical nearest EA gate, CountMean/log Count; only budget,
  EA and reference mode matter.
- `coherekv_conference`: paper-based reimplementation, not asserted byte-exact
  original runtime; online-first reference, shared EA=EE, history-compatible
  nearest anchor, max geometry, degree-first maximal independent set,
  CountMean/log Count. Always actual_p=inf, actual_gamma=inf.
- `coherekv_soft_mis`, `coherekv_soft_mis_importance`: independent p/gamma,
  Count-based fusion, log Contribution. Importance uses q_i=C_i*I_i and
  continuous conflict cost within the hard-EA priority pool.
- `coherekv_soft_mis_log_count`, `coherekv_soft_mis_importance_log_count`:
  identical corresponding candidates except log Count attention. For ANY
  geometry p and finite gamma these remain separate runs. For gamma=inf,
  all fusion weights are binary, so Count=Contribution from fresh singleton
  initialization; skip the suffix arm only when its equivalent main arm is
  enabled. A suffix-only run still evaluates gamma=inf. No equivalence is
  assumed solely from p=inf.

Baseline results are reused across irrelevant grid dimensions. Disabled
baselines are not evaluated. Missing paired baselines produce blank deltas.
Full-cache-only runs require no calibration.

## Scoring, logs and outputs

U-1 unscored inputs, compression (except Full Cache), one unscored
post-compression warm-up, S scored targets. No replay, no reset within
a window. Pooled Overall PPL=exp(sum NLL/sum scored tokens), not arithmetic
mean PPL. Arithmetic mean and sample SD are also reported.

Terminal and run.log show one table per sequence, one row per actual
evaluation, with B,p,Gamma,EA,EE,sequence PPL, tokens and seconds.
FINAL SUMMARY ranks every actual configuration by pooled PPL and reports
mean/std. Verbose logs remain in sequence evaluation_detail.log and logs/.

`aggregate/ppl_sweep_unique.csv`: actual configurations without duplicates;
actual_p, actual_gamma, actual_lambda_ea, actual_lambda_ee, actual_budget,
actual_reference_mode. Blank = irrelevant. `ppl_sweep_best_by_method_p_gamma.csv`:
best lambda pair per method/p/gamma/budget. `ppl_continuous.csv` contains the
rectangular grid with p/gamma/budget/EA/EE, including baseline replicas.
Per-sequence, cycle, paired NLL, and head-round CSVs are retained. All
aggregation keys and completion metadata include gamma.

Plots: EA/EE heatmaps separated by gamma, per-sequence plots, paired NLL
heatmaps, best p/gamma heatmap (minimum over the lambda grid), baseline
and selected-setting comparisons. Strict Times New Roman, PNG 600 DPI and PDF.
Install Times New Roman on the server; no silent fallback. Numeric ZIP
is packaged before plotting and repackaged after successful plotting.
Selection on validation only; freeze configurations before test.

Upload:
`diagnosis_experiments/results/<model basename>/journal_soft_mis_sweep_v9_validation/cohere_p_continuous_results.zip`
(or corresponding test split). One ZIP per model. No checkpoint/weights.
CPU tests use synthetic histories, mock cache/logits and mock aggregation;
real GPU inference requires the original repository and model.
