# CohereKV journal v6 — multi-sequence lambda sweep

This is a replacement of `diagnosis_experiments/` only, based on v5 Contribution A. All external repository modules are imported without changing them. Core journal code in `soft_mis.py` and `p_geometry.py` is unchanged. The old `_log_count` ablation methods are omitted from this sweep, not repeated under infinity order.

## Run

```bash
cd /home/tlxr/CohereKV
python diagnosis_experiments/test_soft_mis.py
python diagnosis_experiments/test_soft_mis_cache.py
python diagnosis_experiments/test_sweep_framework.py
bash diagnosis_experiments/run_soft_mis.sh
```

Edit the literal configuration inside `run_soft_mis.sh` (no environment-input overrides). MODEL defaults to Qwen/Qwen2.5-7B. Other suggested IDs: Qwen/Qwen2.5-3B, meta-llama/Llama-3.2-3B, meta-llama/Llama-3.2-1B. Run one model at a time. GPU is selected by the literal CUDA_VISIBLE_DEVICES export. If rerunning a changed grid, change EXPERIMENT_NAME to avoid replacing earlier output. RUN_CALIBRATION=1 removes the same named experiment's old output. Times New Roman must be installed on the server; no silent fallback. Numeric ZIP is created before plotting.

## Defaults

WikiText-2 validation split; calibration train split (four 4096-token windows); U=2048, S=64, cycles=12, 25,344 input tokens/window, 768 scored targets/window, budget fraction .10 (205 post-compression physical slots per head), score/recent windows 32, sink 4, chunk 128. First eight complete non-overlapping token windows requested (indices 0..7). The tokenizer-specific preflight writes `sequence_windows.json`, excludes incomplete requested windows, reports the actual indices and fails if none exist. No padding, repeat or wrap. All methods for a model share the same windows. Different tokenizers may produce different raw-text window boundaries. For final evaluation change PPL_SPLIT to test and use parameters selected on validation; do not select parameters on test.

Grid: p=2,inf; EA=.30,.50,.70; EE=.10,.30,.50,.70. 56 actual inference runs per window (48 journal configurations + 3 conference + 3 MergeAll + 1 eviction + 1 full); at most 448 for eight windows per model. Calibration is separate. With limited resources, reduce the grid or requested sequence list identically for all methods.

## Methods

- `full_cache`: native unpatched model, native DynamicCache, no pruning/merging or log correction. Exact same U/S input schedule/target locations; verifies physical length equals logical length at every cycle. No lambda, p or budget dependency.
- `eviction_only`: shared KeepKV split and budget, no fusion. Reuse across p/lambda.
- `merge_all`: nearest physical anchor with EA gate; CountMean and log C. Reuse across p/EE.
- `coherekv_conference`: paper-based reimplementation, equations (7), (9)–(18), not a byte-for-byte assertion about the original model runtime. Online-first frozen per-head reference, single lambda (each EA sweep value; EE=EA), certified nearest compatible anchor (tie smallest slot index), full max certificate, fixed degree / anchor certificate / hash / index priority, parallel maximal independent set, CountMean and log Count. Shares KeepKV, U/S evaluator and numerical conventions of this diagnostic; always p=inf. Reuse across requested p/EE. The stored high-precision geometry center represents the mean; actual K/V use model dtype. Tests cover compatible routing, ties, graph maximality, multi-round cache. Conference diameter violations beyond numerical tolerance raise.
- `coherekv_soft_mis`, `coherekv_soft_mis_importance`: unchanged v5 routing, history, soft weights, count fusion, Contribution A and priority; both p=2 and inf. Frozen train reference. Importance uses the v5 continuous conflict cost and its E-A compatible priority pool. At inf Contribution equals Count.

## Scoring and results

U-1 input tokens, compress (except Full Cache), one post-compression warm-up query, then S scored targets. No token replay; one independent cache and history per method/window; no resets within a window. First S target is predicted by the post-compression warm-up. Across windows, sum exact NLL sums and scored token counts, then exp(total NLL/total scored tokens). This is geometric mean of per-window PPL when scored counts equal. Report arithmetic mean and sample SD separately. Full Cache uses the same target positions, not all-token PPL. Sweep ranking uses overall PPL. Do not pick a separate lambda for each sequence.

`aggregate/ppl_sweep_unique.csv`: one row per *actual* method/config, no replicated baseline rows. `actual_p`, `actual_lambda_ea`, `actual_lambda_ee`, `actual_budget`, `actual_reference_mode` describe actual settings. Blank means unused. `overall_ppl`/`ppl`, `mean_ppl`, `std_ppl`, `nll_sum`, `scored_tokens`, `delta_nll_vs_full`, `std_delta_nll_vs_full`. `ppl_sweep_best_by_method_p.csv`: min pooled NLL per method/p/budget on validation; not independent test evidence.

`aggregate/ppl_continuous.csv` is the rectangular plotting/join table: grid `p`, `budget`, `lambda_ea`, `lambda_ee` may replicate baselines; do not count replicated rows as independent runs. `actual_*` fields distinguish actual settings. `ppl_continuous_per_sequence.csv`, `ppl_continuous_paired_deltas.csv`, `ppl_continuous_cycles.csv` keep window and round diagnostics. Per-window head-round logs record actual executions only, not duplicated baseline rows. Figures: combined pooled heatmaps, delta-NLL heatmaps, per-window heatmaps, baseline sweep, selected-setting PPL and paired NLL boxplot; PNG 600dpi and PDF, Times New Roman. Selection and calibration manifests included.

Upload `diagnosis_experiments/results/<model basename>/journal_soft_mis_sweep_v6_validation/cohere_p_continuous_results.zip` (or corresponding test directory). One ZIP per model. No checkpoint files or weights included. The CPU correctness tests use synthetic keys/cache and mock logits; actual GPU model forward requires the original repository, its compatible dependencies and the pretrained model.
