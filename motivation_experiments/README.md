# CohereKV — motivation experiments (single-round only)

Scope: **existence and consequences of Star Conflict**, not full Conference/Journal method comparison. Production code is unchanged. Figure 1/2 use real frozen prefix K/V and naturally observed future queries for *post-hoc evaluation*.

## Calibration (Journal manuscript Eq. 24)

Use 16 disjoint WikiText-2 **train** token windows (8192 tokens each; seed 42) by default. For every calibration window `m`, compute at every `(layer l, KV head h)` the uncompressed-key scale from Conference Eq. 7:

\[
S_{l,h}^{(m)}=\sqrt{2\sum_{i=1}^L\|k_i-\bar k\|_2^2/(L-1)+\epsilon_s^2},\qquad
S_{l,h}^{calib}=\mathrm{Median}_{m}S_{l,h}^{(m)},\qquad
\Delta_{l,h}=\lambda S_{l,h}^{calib}.
\]

The 16-sequence median is the usual statistical median, **including averaging the two middle values** (`torch.quantile(..., 0.5, dim=0)`). Previous releases used RMS *across sequences*: **old calibration JSON is invalid for this run**. Existing calibration JSON with an explicit non-median `aggregation` field is rejected by `calibration_scale.load_calibration_scales`; rerun calibration once per model. The calibration window selection and within-window Eq. 7 formula are unchanged. The fixed scale is shared between the motivation and diagnosis folders.

## Figure 1 — prevalence and filtering potential

Both cosine and Euclidean anchor routing are tested across selected layers and KV heads. Set **one** `LAMBDA` in `run_motivation_attention.sh`. Euclidean routing **and** candidate-candidate conflict edges use the same `Delta=LAMBDA*S_calib`. Cosine routing uses a separate **dimensionless** `FIG1_COSINE_TAU`; its conflict edges are Euclidean and still use the same `Delta`. The code has no independent Euclidean routing/conflict lambda inputs. Counterfactual Conflict-Reduced takes the most severe source-source conflict edge in every Star group, tries dropping either endpoint, and uses the *real next query* **only** to choose the lower-error counterfactual. For `K>1` repeat up to K drops per Star group. This is an **oracle motivation**, never a deployable source-selection policy.

## Figure 2 — filtering vs tuning routing alone

Plot Merge-All as the routing threshold changes, alongside oracle Conflict-Reduced at the same workpoint. For **Euclidean** routing, the horizontal axis is the *shared* `lambda` and `Delta=lambda*S_calib` changes for both routing and conflicts at every point. For **cosine** routing, the horizontal axis is `tau_cos` while the Euclidean conflict `Delta=LAMBDA*S_calib` remains fixed. Thus there is no separate `lambda_r` / `lambda_c` anywhere in the Euclidean experiment. Do not interpret a query-guided oracle gain as direct performance of Journal CohereKV. Group-local numerator errors and whole-layer post-`o_proj` output errors must not be conflated.

## Run

1. Edit the **literal assignments** at the top of `run_motivation_attention.sh` (`PROJECT_ROOT`, `MODEL`, `MODEL_DIR_NAME`, `DEVICE`, etc.). There are no environment-variable configuration inputs.
2. For a new model or old RMS cache, leave `RUN_CALIBRATION="1"`. Later you may use `RUN_CALIBRATION="0"` to reuse the median-calibrated JSON.
3. Run from repository root:

```bash
bash motivation_experiments/run_motivation_attention.sh
```

Outputs:

```
motivation_experiments/results/<MODEL_DIR_NAME>/
    calibration/layer_head_scales.json
    motivation/q1_*.csv, q2_*.csv, run_config.json, run.log
    figures/01_*.png, 01_*.pdf, 02_*.png, 02_*.pdf
```

Plots are saved in both PDF and PNG with `Times New Roman` explicitly requested. By default, plotting **fails** if Times New Roman is unavailable rather than silently falling back to another font. Install a licensed TNR font and refresh the font cache on the plotting host.

**Full/eviction/Merge-All/Conference/Journal method comparisons are in `diagnosis_experiments/`**. The original single-round `method_design_experiment.py` and `ppl_continuation_experiment.py` were moved there as named `legacy_*` baselines and are not run as motivation results.

### Unified-threshold CLI

The motivation Python CLI takes `--lambda_value` (the global Euclidean coefficient), `--fig1_cosine_tau`, `--fig2_cosine_tau`, and optional Euclidean shared-`lambda` / cosine `tau` grids. Old flags `--fig1_euclidean_route_lambda`, `--fig1_conflict_lambda`, `--fig2_euclidean_route_lambda`, and `--fig2_conflict_lambda` are intentionally removed. Re-run Figure 1/2: **old CSVs were generated with different threshold semantics** and must not be mixed with new results.

Output CSVs record `lambda_value` and the metric-dependent `route_param`; `route_param` means `lambda` for Euclidean routing and `tau_cos` for cosine routing. For Euclidean rows, `route_param == lambda_value` by construction.

## Scope after Conference/Journal fair-comparison revision

This directory performs single-round Star Conflict motivation only. The plots
remain PNG and PDF with strict Times New Roman checks. The model shell script
contains grouped, manually commentable model/layer assignments. The prior
Python 3.10 f-string backslash syntax issue is corrected. Median calibration
JSON should be regenerated before rerunning `diagnosis_experiments` in the
absence of an already verified median scale file. Historic results are not
shipped in the revised ZIP.
