# CohereKV — Euclidean vs Cosine E–E MIS ablation

This release modifies **only** `diagnosis_experiments/`. The target method is
`k_only_importance_mis_drop` (Count × Importance D-first priority, Count Mean
K/V fusion, rejected KV/Count Drop), but the ablation applies to every
independently enabled MIS method (including sink-gated and I/IW fusion arms).

## What's changed

- **Removed:** `k_only_mis_drop_headcal`, `k_only_mis_drop_modelcal`, the corresponding
  RUN switches, offline Head λ calibration, plotter, utility and unused test.
  Train-based per-head **S** calibration is retained for the Euclidean baseline.
- **Added:** paired `_cosee` methods for every enabled MIS arm, automatically
  when `RUN_COSINE_EE_ABLATION=1`. For a sink-gated method, suffix order is
  `_cosee_sink`. All other algorithmic behavior is held fixed.
- **E–E conflict definition:**

  - Original Euclidean: `||K_i − K_j||_2 > lambda * S_ref[layer,head]`;
    `PPL_EE_LAMBDAS` / `MIS_EE_LAMBDAS` configure λ.
  - Cosine: `1 − cos(K_i,K_j) > delta`;
    `PPL_COSINE_EE_THRESHOLDS` / `MIS_COSINE_EE_THRESHOLDS` configure δ.
    The cosine graph **does not multiply by S_ref** and does not alter K/V.
    Zero vectors use a stable zero-normalization convention.

  The threshold scales are **not interchangeable**. E–A Routing remains
  independent (`ROUTING_COSINE_THRESHOLDS` or `PPL_COSINE_THRESHOLDS`, e.g. 0.80).

- **Unchanged:** Importance scoring; Retained/Evicted selection; R–E Routing;
  each family's MIS node priority (including Importance D-first); Count Mean,
  I Mean or C×I Mean as selected; Drop/AllCount and Count/log-count behavior;
  uncompressed Prefix → one bulk compression → post-compression continuation PPL.
- **Output identity:** `ee_metric` is `l2` or `cosine`, `ee_threshold` records the
  numeric threshold, while legacy `ee_lambda` stores the same numeric sweep
  coordinate for downstream plotting compatibility. PPL/resume keys include the
  metric; baseline evaluations are explicitly reused across metrics.
- **Plots:** separate `figures/l2/` and `figures/cosine/` subdirectories with
  PDF and PNG outputs, plus `ppl_ee_metric_best_observed.csv` and a paired
  best-*observed*-PPL plot. Minima on the PPL test sweep are **descriptive**:
  they are not an independent validation of tuned thresholds.

## Run

Edit settings **inside** `diagnosis_experiments/run_cosine_mis.sh`; do not pass
experiment settings as environment-variable overrides.

The default is Importance-D MIS with L2 vs Cosine E–E pairs:

```bash
RUN_K_ONLY_MIS_DROP=0
RUN_K_ONLY_IMPORTANCE_MIS_DROP=1
RUN_COSINE_EE_ABLATION=1
RUN_CALIBRATION=1
RUN_ANALYSIS=0
RUN_PPL=1
RUN_PLOT=1
PPL_EE_LAMBDAS=("0.10" "0.30" "0.50" "0.70")
PPL_COSINE_EE_THRESHOLDS=("0.01" "0.03" "0.05" "0.10" "0.20")
```

Specify the target model, dataset, context, budget and number of windows in
that same shell (`MODEL`, `PPL_DATASET`, `PPL_CONTEXT_LEN`, `PPL_BUDGETS`,
`PPL_MAX_SEQUENCES`). To switch from WikiText-2 to C4, change only
`PPL_DATASET="c4"`; keep the calibration dataset (e.g. WikiText-2 Train)
unchanged to test cross-dataset transfer of the same frozen scale S.

```bash
cd /home/tlxr/CohereKV
python diagnosis_experiments/test_cosine_ee_mis.py
python diagnosis_experiments/test_degree_anchor_priorities.py
python diagnosis_experiments/test_konly_importance_mis.py
python diagnosis_experiments/test_bulk_prefix_ppl.py
bash diagnosis_experiments/run_cosine_mis.sh
```

For single-round historical/future-Query Attention Replay, set
`RUN_ANALYSIS=1` and control the **independent** replay threshold arrays
`MIS_EE_LAMBDAS` and `MIS_COSINE_EE_THRESHOLDS`. The experiment uses the same
routed KV groups for both E–E metrics within each configuration.

### Result package to upload

After the run, upload the ZIP named on the final `[UPLOAD]` line, normally:

```
diagnosis_experiments/results/<model>/<experiment>/journal_ia_mis_results.zip
```

At minimum, the ZIP should contain:

- `calibration/median_reference.json`
- `ppl_sweep/ppl_konly_importance_mis.csv`
- `ppl_sweep/ppl_comparison_by_config.csv`
- `ppl_sweep/ppl_routing_load_diagnostics.csv`
- `ppl_sweep/ppl_run_config.json`
- `ppl_sweep/ppl_ee_metric_best_observed.csv` (if plotting completed)

Plots enforce **Times New Roman** when `STRICT_TIMES_NEW_ROMAN=1` and are saved
in vector PDF and high-resolution PNG (`PNG_DPI=600`). Times New Roman must
already be installed on the machine. If not installed, plotting fails clearly;
prior numeric ZIP packaging remains available. No font files are bundled.

### Validation and limitations

CPU regression tests verify cosine edge construction, positive key rescaling
invariance, identical conflict graphs across MIS priorities, fusion selection,
sink gating, and metric-safe grid/result merging. The bulk PPL test verifies
that no prefix/warm-up losses are scored. GPU execution with a full model and
runtime costs must still be validated in the target research environment.
The default experiments use relatively few scored tokens; differences near
0.01 PPL should be assessed across more independent windows.
