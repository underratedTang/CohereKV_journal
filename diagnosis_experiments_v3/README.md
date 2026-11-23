# CohereKV journal Group-D2: cross-round mechanism validation (v20)

**Scope**: This package replaces **only** `/home/tlxr/CohereKV/diagnosis_experiments/`.
It does not change `models/`, `task_evaluation/`, model weights, or any other source.
The journal *theory* concerns general historical group-distance `D_p`; the
low-auxiliary-state operational selector here remains exact `p=2`, **not** a
claim that arbitrary finite p is implemented by the Group-D2 selector.

## Changes for the next experiment

Six arms share the same model, physical KV budget, input stream and frozen TRAIN
reference. They do **not** share identical candidate IDs after their KV states
diverge across rounds:

| method | role | selector/history |
|---|---|---|
| `full_cache` | uncompressed quality reference | no compression |
| `eviction_only` | indispensible eviction baseline | KeepKV selection, no merging |
| `merge_all` | physical routing baseline | all routed candidates, CountMean |
| `coherekv_inf_certificate` | conference-*mechanism* comparator | **p=inf** `minkowski` R/D certificate, historical E-A, degree-priority E-E MIS, CountMean |
| `group_d2` | journal efficient instance | physical E-A, exact historical D2 full-group constraint, bounded pair recovery |
| `group_d2_importance_matched` | same-anchor quota control **on its own trajectory** | shadow Group-D2 cardinality; choose by Count×Importance |

The conference comparator is **not** asserted to reproduce the exact conference
paper *numbers*: both it and Group-D2 use the same frozen TRAIN calibration and
explicit EA:EE thresholds for fair side-by-side mechanism comparison; published
experiments may use different reference/threshold settings. The original
conference certificate invariant is preserved within this comparator: it uses
`p=inf`, `minkowski`, and the existing original MIS degree ordering. Neither
`R+d+R` nor its certificate propagates into the new Group-D2 method.

Additional **same-state** shadow diagnostics run inside the `group_d2` head:
`group_d2_same_state_matched_swaps`, `..._matched_violating_anchors`,
`..._matched_anchors`. This measures selection differences at the same physical
pre-merge state, but **is not** a counterfactual PPL trajectory. The original
matched arm is retained as a separate PPL control and must not be described as
having identical counts across independent trajectories.

## Default next test (Qwen2.5-7B)

The literal parameters in `run_cosine_mis.sh` are fixed using the previous
Qwen-7B pilot's sequence 0; the new test uses **held-out windows 1,2,3**.
`PPL_CYCLES=12`, `PPL_UNSCORED_LEN=1024`, `PPL_SCORED_LEN=128`,
`SWEEP_LAMBDA_PAIRS=("0.60:0.60")`, `SWEEP_LAMBDA_HISTORY=("0.80")`,
`SWEEP_BUDGETS=("0.10")`, `SWEEP_PAIR_WINDOWS=("8")`.

Each window has 12 compression rounds, 1,536 scored targets, 13,824 logical
tokens. Three held-out windows provide **4,608 scored targets per arm**.
There are 3 windows × 6 arms = **18 inference evaluations** (plus optional
frozen calibration if the old reference file is absent). Do not select a new
threshold based on held-out windows and then claim an unbiased test on the same
windows. On a different model, edit the *literal* MODEL and thresholds together,
and select its own disjoint windows and calibration file; do not reuse Qwen's
calibration on another model.

## Installation (only the diagnosis folder)

From your GPU server:

```bash
cd /home/tlxr/CohereKV
# Extract the provided ZIP at this repository root (it contains diagnosis_experiments/ only).
unzip -o CohereKV_diagnosis_crossround_v20.zip -d /home/tlxr/CohereKV
# Remove these obsolete scripts from older versions if present:
rm -f diagnosis_experiments/package_continuous.py \
      diagnosis_experiments/cosine_mis_fusion.py \
      diagnosis_experiments/test_konly_importance_mis.py \
      diagnosis_experiments/test_keep_protection.py \
      diagnosis_experiments/test_group_d2_quick.py \
      diagnosis_experiments/aggregate_group_d2_sequences.py \
      diagnosis_experiments/aggregate_group_d2_sweep.py \
      diagnosis_experiments/test_group_d2_sweep.py \
      diagnosis_experiments/mis_sweep_settings.py \
      diagnosis_experiments/ppl_sweep_utils.py
python diagnosis_experiments/test_group_d2_selection.py
python diagnosis_experiments/test_p_geometry.py
python diagnosis_experiments/test_continuous_cache_rounds.py
python diagnosis_experiments/test_crossround_validation.py
bash diagnosis_experiments/run_cosine_mis.sh
```

The shell file has **literal assignments only** for all hyperparameters. No
experiment input is read from environment variables. The fixed CUDA/HF exports
are runtime defaults, not externally provided hyperparameter inputs.

The script does not delete existing output, and refuses to overwrite an
existing `settings` directory. To rerun, edit the literal `EXPERIMENT_PREFIX`
**before** starting another independent experiment.

## Required output to send back

Upload **exactly one ZIP**:

```text
/home/tlxr/CohereKV/diagnosis_experiments/results/Qwen2.5-7B/journal_group_d2_crossround_v20_qwen7b_heldout/cohere_p_continuous_results.zip
```

The ZIP includes `run.log`, the frozen calibration, the sweep plan,
per-method PPL/NLL, 12-round NLL and full layer/head diagnostics, strict paired
window comparisons, per-round history-remerge/acceptance/radius-saturation
metrics, and **PNG + PDF** figures. Plots require genuine **Times New Roman**;
if the font is unavailable the plot phase fails explicitly (without silently
substituting a font). The numeric result ZIP is created before plotting; after
installing the font, rerun the aggregate plot + packaging commands (see script).

## Interpretation limits

- Conference R/D radius saturation and the Group-D2 remerge count test the
  proposed mechanism; PPL alone does not prove relief of geometric blocking.
- `eviction_only` measures whether *merging itself* helps relative to selecting
  a small KV cache.
- Matching is per-group within a given trajectory; it is not equal total
  acceptance across independent trajectories.
- Successful test suite and compilation do not replace full model GPU tests.
- For any general-p journal claim beyond p=2, mathematical proofs and/or
  separate general-p implementations are needed.
