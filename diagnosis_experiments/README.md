# CohereKV diagnosis experiments: why can the original graph-MIS method approach Eviction-Only?

This folder is deliberately separated from `motivation_experiments/`. It does **not** propose a new algorithm. It diagnoses the released/original CohereKV mechanism while preserving the production planner:


the original retained/evicted selector → fixed request-local `S_ref` → `Delta = lambda * S_ref` → `C/R/D/Gamma` compatible routing → per-anchor conflict graph → degree-priority Parallel MIS → count-weighted merge.

The central hypothesis is **fixed-budget certificate saturation**: `Delta` stays frozen across rounds while propagated `R`/`D` consume an increasing fraction of the same budget. The remaining center-distance allowance

\[
\frac{\Delta-\hat R_i-\hat R_j}{\Delta}
\]

can therefore shrink, candidate routing can fail, conflict graphs can become restrictive, and the method can recover so few evicted states that its behavior approaches Eviction-Only.

## 1. Two complementary diagnoses

### A. First-round budget sweep

The first compression starts from singleton states (`C=1, R=0, D=0`). This isolates the **same-Delta routing/filtering geometry before certificate accumulation**.

For each retained-KV budget and each `lambda`, the script compares:

- **Eviction-Only**: retained states only;
- **Compatible Merge-All**: every source satisfying the original source-anchor `Gamma <= Delta` test is merged, but source-source conflict filtering is disabled;
- **Original CohereKV**: the same compatible routing followed by the original conflict graph and Parallel MIS.

If Original CohereKV is already much closer to Eviction-Only than Compatible Merge-All in round 1, the initial conflict graph is already sacrificing substantial recovery.

### B. Controlled repeated-compression diagnosis

Starting from the same uncompressed prefix, the experiment repeatedly compresses the **actual Original CohereKV state**. This is a controlled mechanism diagnosis rather than a claim that every runtime request compresses at exactly this schedule.

The production original state advances each round. Three same-state counterfactuals are evaluated but never alter the trajectory:

1. **Compatible Merge-All** — same original routing, but accept all routed sources. This isolates source-source conflict/MIS filtering.
2. **Fresh-Certificate shadow** — keep the same current representatives and the same fixed `Delta`, but set `R=D=0` only while building the diagnostic plan. If admission/recovery returns, accumulated certificates are a major cause.
3. **Refreshed-Scale shadow** — keep accumulated `R,D`, but recompute `S_ref` from the current representatives before the diagnostic plan. This separates the frozen-reference-scale part of the fixed-global-Delta hypothesis from certificate accumulation.

Neither shadow is proposed as the final method.

## 2. Direct Eviction-Only degeneration test

The experiment uses short future queries exactly as post-hoc probes. Probe queries never generate any mask.

For any multi-round partition of original prefix tokens, it evaluates the same count-mean K/V + `log(C)` attention-mass construction used in `motivation_experiments`, generalized to the current partition. The main plot is:

```text
figures/original_coherekv_vs_eviction_only_attention_error_by_round.pdf
```

The title explicitly compares **Original CohereKV vs Compatible Merge-All vs matched Eviction-Only**. If Original CohereKV's error approaches Eviction-Only and its error reduction over Eviction-Only approaches zero while recovery also collapses, that is direct evidence of degeneration toward eviction behavior.

## 3. Where are evicted states lost?

For every round the script decomposes all evicted physical states into three mutually exclusive outcomes:

- **routing rejection**: no retained anchor satisfies the original `Gamma <= Delta` certificate;
- **MIS rejection**: routed into a candidate pool but removed by the source-source conflict graph / Parallel MIS;
- **recovered**: selected and merged.

The plot is:

```text
figures/evicted_state_loss_decomposition_by_round.pdf
```

This separates a collapse caused by compatible-routing scarcity from one caused primarily by dense candidate conflict graphs.

## 4. Certificate saturation measurements

For every physical state the experiment records the production effective radius

\[
\hat R_i = \min\left(R_i, \frac{C_i-1}{C_i}D_i\right)
\]

and diameter `D_i`, normalized by the same fixed `Delta`.

It reports mean/median/P90/max values and plots:

```text
figures/certificate_saturation_rhat_and_diameter_over_delta.pdf
```

The experiment also reports the best radius-only center-distance budget available to each evicted state,

\[
\max_j \frac{\Delta-\hat R_i-\hat R_j}{\Delta}
=\frac{\Delta-\hat R_i-\min_j \hat R_j}{\Delta},
\]

and plots its median/P10:

```text
figures/remaining_merge_slack_delta_minus_radii_by_round.pdf
```

A value at or below zero means that **even before considering center distance**, accumulated radii have consumed the entire fixed budget for that source.

## 5. Exact best-anchor certificate pressure

For diagnosis only, the script computes for every evicted physical state the minimum certificate over every retained anchor:

\[
\min_j \Gamma_{ij}
=\min_j\max\left(D_i,D_j,\hat R_i+\|k_i-k_j\|_2+\hat R_j\right).
\]

It plots median and P90 `min_j Gamma_ij / Delta`:

```text
figures/minimum_anchor_gamma_over_delta_by_round.pdf
```

When this quantity moves above 1, even the best available retained anchor fails the original fixed-Delta admission rule.

## 6. Conflict-graph diagnosis

The experiment records:

- candidate admission rate;
- MIS survival among routed candidates;
- overall recovered-entry and recovered-count-mass ratios;
- candidate-pool size;
- conflict edges;
- conflict graph density = conflict edges / possible within-pool edges;
- mean conflict degree;
- MIS rounds.

Two direct comparisons are produced:

```text
figures/mis_survival_original_vs_fresh_certificates_by_round.pdf
figures/conflict_graph_density_original_vs_fresh_certificates.pdf
```

If graph density rises and MIS survival falls in the original trajectory but not in the fresh-certificate shadow, accumulated `R/D` are making the pair certificate increasingly restrictive.

## 7. Frozen global Delta diagnosis

The key figure is:

```text
figures/candidate_admission_fixed_delta_vs_diagnostic_shadows.pdf
```

It compares:

- **Original: fixed Delta + accumulated R,D**;
- **Same Delta, fresh R=D=0 shadow**;
- **Accumulated R,D, refreshed S_ref/Delta shadow**.

Interpretation:

- fresh certificates help strongly, refreshed scale does little → accumulation of `R/D` is the dominant cause;
- refreshed scale helps strongly → freezing `S_ref`/`Delta` is itself a significant cause;
- neither helps → current representative geometry itself is outside the original compactness budget;
- both help → both effects matter.

## 8. QUICK configuration

`run_diagnosis.sh` uses explicit literal values; experiment parameters are not read from environment variables.

QUICK mode:

- model: `meta-llama/Meta-Llama-3.1-8B`
- dataset: WikiText-2
- prefix length: 4096
- probe tokens: 8
- layers: `0,8,16,24`
- KV heads: `0,2,4,6`
- first-round retained budgets: `20%,40%,60%,80%`
- `lambda`: `0.5,0.7,0.9`
- repeated rounds: 5
- per-round physical keep ratio: 0.8

Full mode increases sequence length, sequences, layers/heads, and rounds.

## 9. Run

```bash
cd /home/tlxr/CohereKV
bash diagnosis_experiments/run_diagnosis.sh
```

## 10. CSV outputs

```text
single_round_budget_diagnosis_by_head.csv
single_round_budget_diagnosis_overall.csv
repeated_round_diagnosis_by_head.csv
repeated_round_diagnosis_overall.csv
figures/figure_statistics.csv
```

## 11. Main figures with explicit comparisons

```text
figures/first_round_original_coherekv_vs_merge_all_vs_eviction.pdf
```
Original CohereKV vs Compatible Merge-All vs Eviction-Only before certificate accumulation.

```text
figures/first_round_candidate_admission_vs_mis_recovery.pdf
```
Candidate admission vs MIS survival vs final recovery in the first round.

```text
figures/original_coherekv_vs_eviction_only_attention_error_by_round.pdf
```
Original CohereKV vs Compatible Merge-All vs Eviction-Only across repeated compression rounds.

```text
figures/fidelity_benefit_over_eviction_by_round.pdf
```
Attention-error reduction relative to Eviction-Only; values approaching zero indicate degeneration.

```text
figures/candidate_admission_fixed_delta_vs_diagnostic_shadows.pdf
```
Original accumulated certificates/fixed Delta vs fresh-certificate and refreshed-scale shadows.

```text
figures/evicted_state_loss_decomposition_by_round.pdf
```
Routing rejection vs MIS rejection vs successfully recovered states.

```text
figures/certificate_saturation_rhat_and_diameter_over_delta.pdf
```
Effective radius and diameter consumption of the same fixed Delta budget.

```text
figures/remaining_merge_slack_delta_minus_radii_by_round.pdf
```
Remaining normalized center-distance budget after certificate radii consume part of Delta.

```text
figures/minimum_anchor_gamma_over_delta_by_round.pdf
```
Best possible retained-anchor certificate relative to the fixed feasibility boundary.

```text
figures/final_round_recovery_vs_global_lambda.pdf
```
Whether a larger global compactness coefficient delays the late-round recovery collapse.
