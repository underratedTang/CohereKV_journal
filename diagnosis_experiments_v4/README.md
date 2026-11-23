# CohereKV continuous p-history MIS + pure eviction-only + multi-window evaluation (v8)

This code package contains **only `diagnosis_experiments/`** and is intended as
an overlay on the user's existing CohereKV repository. It **does not modify**
`models/`, `task_evaluation/`, `motivation_experiments/`, weights, or any other
repository path. Replace the old `diagnosis_experiments/` directory as a whole
(keep its old `results/` separately if needed); unzipping over the old directory
will **not delete** obsolete scripts.

The main experiment is `run_cosine_mis.sh`, a literal-variable Bash configuration
file. **No environment variables are used as experiment-parameter inputs.**
Edit settings in the file itself. The original, useful runtime exports
(`PYTORCH_ALLOC_CONF=expandable_segments:True`, `HF_ENDPOINT=https://hf-mirror.com`,
`CUDA_VISIBLE_DEVICES=1`) are restored in the shell file. They configure
PyTorch/Hugging Face/GPU visibility; they are **not** environment-variable
inputs for experiment hyperparameters. Legacy *one-time bulk* and *cosine replay*
entry points and tests are removed. Active MIS priority regression tests are
retained because they validate all original MIS families.

## New mandatory `eviction_only` baseline (v7)

`eviction_only` is **always enabled** alongside `full_cache` and `merge_all`.
For each head and compression round, it calls the **same** KeepKV `select_split`
with identical attention-importance scoring, logical sink/recent protection,
recency weighting and retained-slot budget. It copies ONLY `K[keep]`,
`V[keep]`, their positional indices and their singleton counts to the
post-compression cache. Every `K[evict]` / `V[evict]` is dropped outright.
There is **no** E-A route, E-E conflict graph/MIS, Key or Value fusion,
or importance/count mass reassignment. History-geometry fields are unused and
reported blank; `history_tracked=false`, `history_certified=false`,
`ea_certificate_used=false`, `ee_mis_used=false`. Kept singleton counts
remain 1 through successive rounds, and their attention semantics use the
same underlying KeepKV compression implementation as the other methods.

Key comparison: `NLL_merge - NLL_eviction_only` (negative = beneficial merge),
not an average of per-cycle PPL. The runner exports `dropped_slots` in
per-cycle, overall and per-head-round CSV outputs; unlike `rejected_slots`,
it includes *all* evicted slots that were not actually merged. For pure
`eviction_only`, `dropped_slots = evicted_slots`, `accepted_slots=0`.

The zero-merge reference depends on KV budget and selection parameters but
**not** `p`, `lambda_ea` or `lambda_ee`; the evaluator reuses its numerical
result across those sweep settings while rewriting threshold metadata.
As previous merged KV states differ across arms, KeepKV *selection rules*
are matched but retained token indices need not match in later rounds.

## Exact definitions

- Reference scale `S_ref` is the conference-style RMS of **pairwise Key L2
  distances**, i.e. a **p=2 reference scale**, frozen at the first compression
  (`online_first`). The alternative `frozen_train` loads the independent
  WikiText-2 train-median RMS reference. **S_ref is NOT added to any distance:**
  it is multiplied by each lambda to get a distance cutoff.
- **Two independent controls:**
  `Delta_EA = lambda_ea * S_ref` controls E-A routing;
  `Delta_EE = lambda_ee * S_ref` controls E-E MIS conflict edges.
  A valid historical p-MIS group is guaranteed only within
  `Delta_cert = max(Delta_EA, Delta_EE)`, which is checked after each merge.
  If the intended theorem is `D_p <= Delta_EE`, use
  `lambda_ea <= lambda_ee` (or implement an explicit stricter group gate).
- **`merge_all` / `importance_weighted_merge_all`:** E-A routing uses ONLY
  `||K_evicted - K_anchor||_2 <= Delta_EA`, choosing the nearest compatible
  retained physical Key. Historical R_p / D_p are not used for gating.
  Both baselines accept ALL routed states and use their existing fusion
  conventions. They do NOT claim historical p-compactness.
  A comparison between two *physical* Keys is independent of p; `merge_all`
  also does not use lambda_ee. One run is cached/reused for all such settings.
- **`coherekv_wo_mis` (conference CohereKV w/o MIS ablation):** E-A routing
  uses exactly the **historical p-certificate** used by the MIS methods,
  `Rhat_p(evicted) + ||mu_evicted-mu_anchor||_2 + Rhat_p(anchor) <= Delta_EA`.
  All successfully routed candidates enter CountMean merging, with NO E-E
  conflict graph or MIS filtering. Historical R_p/D_p upper certificates are
  propagated across rounds even when the resulting diameter EXCEEDS Delta_EA;
  this arm therefore does **NOT** have a cross-round compactness guarantee.
  The next round's certified routing checks those potentially oversized
  historical groups. It depends on p and lambda_ea but **not** lambda_ee.
  In the p=inf, equal-lambda case, this reproduces the geometric
  E-A / no-E-E principle of the conference ablation. Distinguish it from
  physical-Key `merge_all` in every result table and discussion.
- **Paired `*_wo_ea_certificate` methods (ALL MIS arms):** Every original
  MIS method has its own independently switchable E-A-certificate ablation;
  no standalone, implicitly Degree-first `coherekv_wo_ea_certificate` exists.
  Example comparisons:

  | Original | Paired physical-E-A control |
  |---|---|
  | `k_only_mis_drop` | `k_only_mis_drop_wo_ea_certificate` |
  | `k_only_importance_mis_drop` | `k_only_importance_mis_drop_wo_ea_certificate` |
  | `k_only_importance_iwmean_drop` | `k_only_importance_iwmean_drop_wo_ea_certificate` |
  | `k_only_mis_drop_sink` | `k_only_mis_drop_sink_wo_ea_certificate` |

  All **15 existing MIS arms**, including Degree-first, degree-anchor-product,
  anchor-first, importance-D-first, AllCount, I-mean, IW-mean and sink-gated
  controls, have a paired ablation (30 registered MIS names in total).
  Exactly ONE decision changes within each matched pair:

  - **Original E-A routing:**
    `Rhat_p(evicted) + ||mu_evicted-mu_anchor||_2 + Rhat_p(anchor) <= Delta_EA`.
  - **Paired w/o E-A certificate:** physical representative-Key
    `||K_evicted-K_anchor||_2 <= Delta_EA`, without historical R/D gating.
  - **E-E conflict for BOTH:**
    `Rhat_p(u) + ||mu_u-mu_v||_2 + Rhat_p(v) > Delta_EE`;
    both run the **same original MIS priority and fusion/Count semantics**.

  Every w/o-EA ablation still propagates valid historical p-radius / p-diameter
  **upper bounds**, including values ABOVE `Delta_cert`. These experimental
  controls do **not** satisfy the original cross-round p-compactness guarantee,
  because the anchor–candidate pair is not certified. Exports include
  `ea_certificate_used=false`, `ee_mis_used=true`, `history_tracked=true`,
  `history_certified=false` and over-limit statistics. For sink-gated arms,
  the non-sink KV heads still use `merge_all`, exactly as in the original arm.
- **Unmodified native MIS arms:** E-A uses the historical p-certificate;
  E-E uses the same certificate at `Delta_EE`. The existing priority choices,
  K/V fusion weights, AllCount, and sink-gated behavior are unchanged.
  Certified heads must pass the group-diameter threshold after compression.
- p is a freely configurable real **p >= 1** (or `inf`); **original vector
  distances are Euclidean**. p determines how historical point-to-point L2
  distances are aggregated. Finite-p radius and diameter upper certificates
  are updated by Minkowski / point-pair weighted p-mean. At `p=inf` they
  specialize to the conference R/D certificate recurrence. No special
  p=2 exact update is performed in the main experimental comparison.
- `merge_all` has no valid history geometry by design, so its exported
  radius/diameter statistics are **blank** and its history-certified flag is
  false. Do not mistake the internal zero placeholders for geometric bounds.
- `coherekv_wo_mis` and **every** `*_wo_ea_certificate` MIS arm propagate
  historical geometry bounds. Their exported p-radius/diameter may exceed the
  threshold. They set `history_tracked=true`, `history_certified=false`, and
  count over-limit post-merge groups through `geometry_over_delta`.
- A method with `_sink` before any `_wo_ea_certificate` suffix uses MIS only
  for calibrated sink heads and MergeAll elsewhere. Only certified native MIS
  heads (not physical-E-A ablations or MergeAll heads) retain the p guarantee.

## KeepKV sink / recent-slot protection and attention importance

In `run_cosine_mis.sh`, independently configure the restored original defaults:

```bash
KEEPKV_SINK_SIZE=4
KEEPKV_RECENT_PROTECT_SIZE=128
KEEPKV_SCORE_WINDOW=128
KEEPKV_RECENCY_BETA=1.0
```

- `KEEPKV_SINK_SIZE` is any nonnegative integer, **not fixed to 1**.
  It force-retains the cache entries representing original logical token
  positions `[0, KEEPKV_SINK_SIZE)`, across every compression round.
- `KEEPKV_RECENT_PROTECT_SIZE` force-retains the newest physical slots as
  identified by the **original absolute logical positions**, rather than
  the index in the compacted physical cache. The original KeepKV selector
  caps this at `post_compression_budget - KEEPKV_SINK_SIZE`; if capping
  occurs, the program prints the effective recent protection size.
- `KEEPKV_SCORE_WINDOW` controls the number of recent query attention rows
  used for importance scoring, **not** the number of protected KV slots.
- `KEEPKV_RECENCY_BETA` controls importance recency weighting (`>= 1`).
- These settings affect the retained/evicted split identically for MergeAll
  and every MIS variant, including their sink-head gated versions. The
  `*_sink` suffix independently selects which **heads** use MIS, and must
  not be confused with `KEEPKV_SINK_SIZE` (mandatory **token** protection).
- A post-compression kept-slot budget smaller than the mandatory sink count
  is invalid and raises a clear error; reduce the sink count or increase
  the budget. For very short smoke tests, reduce the recent protection size
  to avoid having virtually no KeepKV importance-selected slots.

## Continuous scoring and cache lifetime

For each cycle: `[U-1 unscored tokens] -> compress -> [1 unscored warmup]
-> [S scored tokens]`; one contiguous ordered WikiText-2 test-token window
per sequence. Within that window there is no replay, no cache reset and no
RoPE position reset. Across windows, caches and RoPE positions restart. The last unscored
query runs AFTER compression and predicts the first scored token, so exactly S
**post-compression targets** are scored per cycle. Each cycle compresses
back down to `round((U-1)*budget)` physical slots, with history-carrying MIS
certificates propagated across rounds. This is a fixed **post-compression**
capacity, NOT a fixed peak-memory budget; KV grows before the next compression.
FullCache shares the same input sequence and scoring positions, with no
compression. Pure eviction-only uses the same budgets and scoring positions but no merging.

## Running in the original CohereKV repository

1. Back up the old folder (in particular old `results/`) and replace the folder
   wholesale so the removed obsolete scripts really disappear:

   ```bash
   cd /home/tlxr/CohereKV
   # Optional: back up the original folder OUTSIDE the repository (preserves results).
   cp -a diagnosis_experiments /home/tlxr/diagnosis_experiments_backup
   rm -rf diagnosis_experiments
   unzip -q /PATH/TO/CohereKV_diagnosis_eviction_only_multiseq_v8.zip -d .
   ```

2. Edit the literal settings in `diagnosis_experiments/run_cosine_mis.sh`:
   `ROOT_DIR`, `MODEL`, `PPL_CYCLES`, `PPL_UNSCORED_LEN`, `PPL_SCORED_LEN`,
   `PPL_SEQUENCE_INDICES`,
   `PPL_P_VALUES`, `PPL_LAMBDA_EA`, `PPL_LAMBDA_EE`, `PPL_BUDGETS`, and the
   `RUN_COHEREKV_WO_MIS` and each original/paired `RUN_*` switch.
   `eviction_only` is mandatory, with **no switch** to forget to enable.
   For example, set `RUN_K_ONLY_MIS_DROP=1` and
   `RUN_K_ONLY_MIS_DROP_WO_EA_CERTIFICATE=1` for a Degree-first matched pair;
   set `RUN_K_ONLY_IMPORTANCE_MIS_DROP=1` and
   `RUN_K_ONLY_IMPORTANCE_MIS_DROP_WO_EA_CERTIFICATE=1` for an Importance-first
   matched pair. Enable any other matching pair individually. No general
   `RUN_COHEREKV_WO_EA_CERTIFICATE` exists. Physical `merge_all` is always
   present as the no-certificate/no-MIS baseline. Set `RUN_CALIBRATION=0` **only if** an appropriate
   previously generated `calibration/median_reference.json` exists at the
   exact selected experiment output path; otherwise leave it at 1.

3. Before a long run, run the CPU/logic tests:

   ```bash
   python diagnosis_experiments/test_eviction_only.py
   python diagnosis_experiments/test_p_geometry.py
   python diagnosis_experiments/test_dual_lambda_merge_all.py
   python diagnosis_experiments/test_coherekv_wo_mis.py
   python diagnosis_experiments/test_wo_ea_per_mis.py
   python diagnosis_experiments/test_continuous_cache_rounds.py
   python diagnosis_experiments/test_keep_protection.py
   python diagnosis_experiments/test_continuous_ppl.py
   python diagnosis_experiments/test_degree_anchor_priorities.py
   python diagnosis_experiments/test_konly_importance_mis.py
   python diagnosis_experiments/test_multi_sequence.py
   ```

4. Run from the CohereKV repository (uses the same model dependencies and
   runtime patches as the original installation):

   ```bash
   bash diagnosis_experiments/run_cosine_mis.sh
   ```

For the first GPU smoke run, edit U=512, S=16, cycles=2, p=("2" "inf"),
`lambda_ea=("0.5")`, `lambda_ee=("0.5")`, and set
`KEEPKV_RECENT_PROTECT_SIZE=16` for a meaningful tiny-budget test;
start with two matched pairs: `k_only_mis_drop` +
`k_only_mis_drop_wo_ea_certificate`, and `k_only_importance_mis_drop` +
`k_only_importance_mis_drop_wo_ea_certificate`.

**Context limit:** `cycles*(U+S)` cannot exceed the model's configured
`max_position_embeddings`; the evaluator stops rather than resetting positions.
For example, 4 x (16384+256) = 66,560 logical tokens. Some Qwen2.5
configurations have a smaller default context, so reduce cycles/U or use a
validated long-context model configuration. FullCache may consume substantial
GPU RAM at 66K.

## Figures / upload for subsequent analysis

The runner writes all outputs under:

`diagnosis_experiments/results/<MODEL_NAME>/<EXPERIMENT_NAME>/`

Upload **the single archive**:

`diagnosis_experiments/results/<MODEL_NAME>/<EXPERIMENT_NAME>/cohere_p_continuous_results.zip`

It is produced by `package_continuous.py` and contains:

- `calibration/median_reference.json` (ONE shared frozen TRAIN reference);
- `sequences/seq_<index>/ppl_continuous/continuous_config.json` with
  exact `[token_start, token_end)` and SHA-256 of the **whole input window**;
- `sequences/seq_<index>/ppl_continuous/ppl_continuous.csv`,
  `ppl_continuous_cycles.csv`, and `ppl_continuous_head_rounds.csv`;
- `sequences/seq_<index>/ppl_continuous/continuous_run_complete.json`;
- `aggregate/aggregate_complete.json` (only after all windows validated);
- `aggregate/ppl_continuous.csv` (pooled NLL/token and exp(pooled NLL) PPL);
- `aggregate/ppl_continuous_cycles.csv` (scored-token-weighted pooled cycle curves);
- `aggregate/ppl_continuous_per_sequence.csv` (method results for each window);
- `aggregate/ppl_continuous_paired_deltas.csv` (within-window NLL differences
  relative to both eviction_only and merge_all);
- `aggregate/figures/*.png` and `*.pdf` (both formats, including pooled
  NLL deltas relative to eviction_only);
- `run.log` (commands and progress).

**Multi-window semantics:** `PPL_SEQUENCE_INDICES=("1" "2" "3")` with
`PPL_CYCLES=32`, `U=1024` and `S=32` reproduces **v3's token spans**:
`[33792,67584)`, `[67584,101376)` and `[101376,135168)` in the
concatenated WikiText-2 test token stream. Each window is evaluated in a
fresh Python process, with a new KV cache and RoPE logical position starting
at zero; the cache persists across all 32 compression cycles **inside** each
window. These are contiguous token windows, **not separately sampled articles**.
Sequence 0 is omitted by default because v3 treated 1/2/3 as held-out windows;
you may change the literal indices to include 0. Keeping the same window
indices with a different cycle count DOES NOT use the same source tokens.

No arithmetic mean of per-window PPL is taken. For a method, the pooled value
is `exp(sum(N_i * log(PPL_i))/sum(N_i))`, where `N_i=cycles*scored_len`.
Every window is checked for complete cycles, complete configured arms, matching
TRAIN reference hash and model/parameter settings, disjoint token spans,
and identical scoring-count invariants. The aggregate completion marker is
written only after all checks pass. The runner packages numeric data before
optional drawing so a missing Times New Roman font cannot destroy results.

`plot_continuous_ppl.py` uses **Times New Roman only**, embeds TrueType glyphs
in PDF (`pdf.fonttype=42`) and writes 600-DPI PNG by default. It fails
explicitly if Times New Roman is not installed; no substitute font is used.
Raw data are packaged *before* plotting so they survive a font error. To rerun
plotting after installing Times New Roman, call the plot script manually with
`--input_dir .../ppl_continuous --font_family 'Times New Roman' --png_dpi 600`
and then rerun `package_continuous.py --input_dir .../<EXPERIMENT_NAME>`.
Do not send or embed copyrighted font files in the result archive.

## Important limitations

- p-MIS pairwise compatibility is a **sufficient**, not necessary, criterion
  for whole-group p-compactness. Finite-p general certificates can be
  conservative; no claim is made that all admissible groups are accepted.
- Finite-p `D_p <= Delta_cert` is **not** a maximum-pairwise-distance theorem.
  Attention/Value approximation quality must be tested empirically.
- Every w/o-EA-certificate method is a **paired diagnostic ablation**, not
  a replacement for p-compactness. It keeps its own native E-E graph, MIS
  priority, head gating and fusion but bypasses the E-A history certificate.
- `eviction_only` is strict no-fusion, **not** `merge_all` with a zero
  threshold: the latter could still invoke a routing/fusion path and does not
  define the same guarantee. Here no matching or merging kernel is called.
- `*_allcount` is an artificial ablation: rejected K/V do NOT reenter
  geometry or fusion, even if their Count affects attention.
- All original Merging / MIS fusion variants remain distinct; the reference
  `merge_all` is plain CountMean, not importance-weighted fusion.
- No pretrained Qwen/Llama forward pass was run when this archive was built:
  the build environment lacks Hugging Face Transformers and model weights.
  The no-model-weight math, multi-round cache, MIS, and scoring-index tests pass.
