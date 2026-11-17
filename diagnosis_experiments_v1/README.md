# CohereKV diagnosis experiments: IA-MIS + Count-aware IW-Mean

**Scope:** This release changes only `CohereKV/diagnosis_experiments/`. The eight previous native/sink-gated K-only MIS arms, their distance graph, node priority and Count-mean fusion are preserved. The main model, runtime patch, task evaluation and other project files are not changed.

## Method and reproducibility

Each current physical KV slot has an existing **current-round, count-decoupled** KeepKV-style recent-window importance `I_i` and a separately propagated token count `C_i`. Under the PPL compression code, this importance comes from captured base (unbiased-by-`log(C)`) attention scores; it is a current-slot attention estimate rather than a stored sum of historical per-original-token scores. The new geometry and fusion use **`W_i=C_i*I_i`**, without mutating the count or changing the old D-first node priority. For the one-round future-Query Replay, `C=1`, so `W=I`.

For each evicted token routed to retained anchor `a`, set `rho_i=W_i/W_a`. **Do not cap `rho` at 1.** A multi-round evicted state may have more accumulated weight than its retained anchor. To avoid division by zero, only for geometry compute `W_a_safe=max(W_a,max_k W_k * 1e-8,1e-30)` within the KV head. Then:

- **E–A gate:** `sqrt(2*rho_i/(1+rho_i))*||K_i-K_a|| <= lambda*S_ref`. Failed routed candidates are rejected before MIS.
- **E–E graph:** only E–A-eligible candidates build an edge when `sqrt(2*rho_i*rho_j/(rho_i+rho_j))*||K_i-K_j|| > lambda*S_ref` (zero when both ratios are zero).
- **Selection:** the **original**, unchanged D-first parallel MIS priority `I_i - sum_{j in conflict_neighbors} I_j` with existing degree/distance/hash tiebreaking.
- **Fusion:** retained anchor plus selected evicted nodes use `K'=sum W_i K_i / sum W_i`, likewise for V. A group with *all zero W* deterministically falls back to count-weighted mean. Zero-weight members otherwise carry no fusion vector contribution, but still update Count according to the mode.
- **Drop:** rejected routed tokens contribute neither KV nor Count. **AllCount:** rejected routed tokens contribute Count, but not KV; this includes tokens failing the E–A gate. All other routing failures are discarded. There is **no second routing**.

The E–A and E–E metrics are motivated by twice the minimum two-token importance-weighted Key SSE, not a guarantee of future attention output preservation. Both IA constraints are **pairwise**, not a group-wide drift bound. Retained anchor may move when accepted sources have large total `W`. The graph and IW mean use consistent `W=C*I` but D-first intentionally continues to use the **old `I`** so the graph/fusion ablations remain clean.

## Independent switches and ablations

All switches are ordinary variables at the top of `run_cosine_mis.sh`; **do not pass parameters through shell environment variables**. Each new method has an independent 0/1 switch. `full_cache` and `merge_all` controls are always present.

| Method(s) | Graph | Fusion | Notes |
| --- | --- | --- | --- |
| `merge_all` | none | Count mean, all routed | existing control |
| `importance_weighted_merge_all` | none | Count × Importance mean, all routed | new fusion-only control |
| `k_only_mis_drop`, `k_only_mis_drop_allcount` | old E–E distance | Count mean | existing methods |
| `k_only_importance_mis_drop`, `k_only_importance_mis_drop_allcount` | old E–E distance + original D-first | Count mean | existing methods |
| `k_only_importance_iwmean_drop`, `k_only_importance_iwmean_drop_allcount` | old E–E + D-first | Count × Importance mean | fusion-only MIS ablation |
| `ia_mis_countmean_drop`, `ia_mis_countmean_drop_allcount` | new E–A/E–E + D-first | Count mean | graph-only ablation |
| **`ia_mis_iwmean_drop`, `ia_mis_iwmean_drop_allcount`** | **new E–A/E–E + D-first** | **Count × Importance mean** | **full new proposal** |

All three new MIS families have independent `_sink` variants. A `_sink` variant uses its native MIS only in **train-calibrated sink KV heads**; otherwise it uses the unchanged count-mean `merge_all`, not `importance_weighted_merge_all`. Old eight method names are unchanged.

The shell runs **one shared `MIS_EE_LAMBDAS` sweep** for old and IA methods at frozen per-layer/head training Key reference `S_ref`, not multiple incomparable distances. Existing optional L2 routing and Cosine routing sweeps remain independent. The IA E–A gate is applied *after* the unchanged Top-1 cosine or L2 routing.

## Execute

Edit only this shell script for model, dataset, quick/full mode, per-method switches, sweeps and plot options:

```bash
cd /home/tlxr/CohereKV
python diagnosis_experiments/test_konly_importance_mis.py
python diagnosis_experiments/test_ia_mis.py
bash diagnosis_experiments/run_cosine_mis.sh
```

To smoke-test end to end first, set `QUICK_MODE=1`, select a handful of MIS methods and reduce the route/lambda/budget arrays **inside the shell script**. For the final run, set `QUICK_MODE=0` and select the required methods. The original `ROOT_DIR` is `/home/tlxr/CohereKV`; adjust it inside the script if needed. Requires model weights, dataset, CUDA model dependencies, Times New Roman and `pdftoppm` or `gs` for plotting.

## Results to upload for next analysis

At completion, the shell prints:

```text
[UPLOAD] /home/tlxr/CohereKV/diagnosis_experiments/results/<MODEL_BASENAME>/<EXPERIMENT_NAME>/journal_ia_mis_results.zip
```

**Upload this `journal_ia_mis_results.zip`**. It contains the frozen train calibration and sink mask, `run.log`, `run_config.json`, raw single-round Replay CSV (`cosine_mis_by_config.csv`, `cosine_mis_by_group.csv`, `cosine_mis_rejected.csv`, `cosine_mis_by_probe.csv`), multi-round `ppl_sweep/ppl_konly_importance_mis.csv`, comparison/summary CSV, examples and PNG/PDF figures (provided the relevant stages completed). `lambda` and the method/fusion identity are recorded in every row. The historical PPL CSV basename is retained for compatibility; it **also includes the new methods**.

**Interpretation limits:** replay uses real *future* Queries only for evaluation, never for compression. PPL re-evaluates current slot importance each compression round; it does not store a mathematically exact historical `importance_mass` for represented original tokens. In AllCount, inherited Count and merged K/V can be inconsistent by design. Changes in geometric SSE do not prove improvements in future-query Relative Attention-L2 or PPL; these must be measured. Times New Roman enforcement is strict: if the font is missing, numeric results are packaged before the plotting step fails with an explicit error; do not silently substitute a font.
