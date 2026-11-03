# Dual Cosine/L2 Routing Sweep + Independent E–E MIS — Real-Q diagnosis

**Only `diagnosis_experiments/` changes.** This is the Qwen2.5-3B/7B + Llama,
low-VRAM native-MIS, low-memory-plot variant of the v3 three-arm diagnostic.
All three arms share the same KeepKV retained partition, Routed Tokens, Value,
Count, and original future Query for each routing setting:

- `merge_all`: keep every routed original K.
- `mis_to_anchor`: original CohereKV degree-priority **E–E** maximal independent
  set; replace **only rejected routed Keys** by their retained Anchor Key.
- `all_to_anchor`: replace all routed Keys by Anchor Key.

Rejected Values and represented-token multiplicities are never dropped.
No extra R–E filtering is applied after either routing decision.

## Routing modes and independently swept thresholds

Edit **only** `diagnosis_experiments/run_cosine_mis.sh`, not environment variables:

```bash
MODEL="Qwen/Qwen2.5-7B"      # or Qwen/Qwen2.5-3B / a supported Llama
QUICK_MODE=1
ROUTING_MODES=("cosine" "l2")  # one or both modes
ROUTING_COSINE_THRESHOLDS=("0.80" "0.85")
ROUTING_L2_LAMBDAS=("0.40" "0.60" "0.80")
ROUTING_L2_DISTANCES=()       # optional physical L2 cutoff sweep
MIS_EE_LAMBDAS=("0.40" "0.50" "0.60" "0.70")
GRAPH_BACKEND="cpu_batched"  # memory-limited original native MIS on CPU
```

**Routing** is performed by Top-1 over retained Keys for every evicted Key:

- Cosine: nearest by **maximum cosine**; accepted if `cos >= routing_cosine_threshold`.
  No Euclidean gate. Zero-norm keys are unroutable (same as the prior v3).
- L2: nearest by **minimum physical Euclidean distance**; accepted if
  `distance <= routing_l2_cutoff`. No cosine gate.
- For a calibrated L2 routing lambda,
  `routing_l2_cutoff = lambda_route * S_calib[layer, KV_head]`.
  The scale `S_calib` is frozen Median across independent **train** sequences.
- For absolute distances (`ROUTING_L2_DISTANCES`), the cutoff is that literal
  physical Key distance. Both kinds may be swept simultaneously. Empty arrays
  are permitted if at least one L2 threshold set remains nonempty.

**E–E MIS is independent from the R–E Routing choice:**

`(i, j)` is a conflict iff `||K_i - K_j||_2 > lambda_EE * S_calib[layer, KV_head]`.
All configurations in a given routing setting reuse the same Top-1 candidate
pools across E–E lambda values. This preserves the original conflict definition
and degree-priority/hash tie-breaking. No Query enters routing/MIS/fusion.

For efficiency, each (layer, head, budget) computes Top-1 once per routing
**mode**, then applies the acceptance threshold sweep to the cached scores.
The three methods are paired **within** each routing setting only. The
Routing sweep changes how many Evicted Tokens are represented; do not compare
raw method errors across different Routing modes without also reporting coverage.

## Run

Upload the patch ZIP into `/home/tlxr/` and apply it **without deleting the
existing shell script** (you may have edited its model/budget settings):

```bash
cd /home/tlxr
unzip -o CohereKV_Dual_Routing_MIS_Qwen25_LowVRAM_patch.zip
cd /home/tlxr/CohereKV
python diagnosis_experiments/test_cosine_mis.py
python diagnosis_experiments/test_qwen_capture.py
python diagnosis_experiments/test_low_memory_mis.py
python diagnosis_experiments/test_dual_routing.py
bash diagnosis_experiments/run_cosine_mis.sh
```

Note: the patch `unzip` command **replaces** the SH file. Back up your own
custom settings first, then copy their values into the revised sweep arrays.

`RUN_CALIBRATION=1` uses the existing Qwen/Llama independent train-set Median
calibration; `RUN_CALIBRATION=0` requires a calibration JSON **inside the new
output directory** matching model, dtype, dataset, seed and lengths. The runner
cleans **only its own** result directory. Use the output directory printed by
`[UPLOAD]`, not the old v3 directory.

Quick result bundle examples:

```text
/home/tlxr/CohereKV/diagnosis_experiments/results/Qwen2.5-3B/dual_routing_mis_median_realq_v4_quick/cosine_mis_median_results.zip
/home/tlxr/CohereKV/diagnosis_experiments/results/Qwen2.5-7B/dual_routing_mis_median_realq_v4_quick/cosine_mis_median_results.zip
```

Change `QUICK_MODE=0` for `_full` in the result directory. The ZIP includes
`cosine_mis_by_config.csv`, `cosine_mis_overall.csv`, `cosine_mis_by_probe.csv`,
`cosine_mis_by_group.csv`, `cosine_mis_rejected.csv`, actual Key NPZ examples,
`run_config.json`, run log, frozen train-calibration JSON, and figures.

Every config and probe row has: `routing_mode`, `routing_parameter_type`,
`routing_parameter`, `routing_l2_cutoff`, `ee_lambda`. In Cosine mode the
`routing_l2_cutoff` column is 0 (not an extra gate); in L2 mode it is the
actual physical cutoff for that layer/head. Existing `cosine_threshold` is
filled **only for cosine routing**, blank for L2.

## Publication figures and sweep presentation

`plot_cosine_mis.py` makes focus-routing comparisons; independent E–E
lambda sensitivity; coverage/rejection; per-layer and per-Query summaries;
**Routing-threshold sweep** (coverage, rejection, real-Q errors) and a 2D
**Routing x E–E heatmap**. Real high-dimensional Key PCA examples include the
routing mode, threshold type/value, the **actual L2 cutoff** when applicable,
and the independent E–E λ in the title. PCA is for **illustration only**:
all decisions use the actual original Key dimensions, not projected points.

The plotting script uses isolated subprocesses and streams potentially massive
per-probe CSVs. It produces Times New Roman, **600 DPI PNG** and **vector PDF**
without high-resolution Matplotlib Agg allocations; `pdftoppm` or `gs` is
required for PNG. Strict Times New Roman rejects missing fonts rather than
silently substituting. Numerics are packaged *before* plotting if a font is
missing.

## Limitations and validation

The diagnostic evaluates actual post-RoPE real Q from the original uncompressed
Llama or Qwen2.5 `eager` model via teacher-forced attention replay, **not PPL
or compressed autoregressive rollout**. Qwen future Cache K is cross-checked.
Original CohereKV common/source files remain untouched. CPU native MIS vs GPU
could differ at nearly exact threshold ties due to FP32 arithmetic; default
`cpu_batched` avoids padded GPU OOM. Tests need no model weights/CUDA; full
Qwen model runs must be validated on your server with CUDA/model weights.

To avoid too many 600-DPI PNGs during wide sweeps, one representative routing
threshold per mode is used for detailed error / coverage / Layer / per-Query
figures (`FOCUS_ROUTING_COSINE_THRESHOLD`, `FOCUS_ROUTING_L2_LAMBDA` in SH).
**All routing parameter values** still appear in the routing-sweep charts and
Routing × E–E heatmaps, and **all** numeric configurations are preserved in CSV.
The PCA illustration shows a representative actual group for each mode.
