# CohereKV — Degree / E–A Priority Diagnosis

All modifications are contained within `diagnosis_experiments/`; the CohereKV model implementation is unchanged.

## Primary experiment: three parameter-free MIS priorities

With the same Importance Top-K retained selection, cosine routing, frozen TRAIN E–E scale, E–E conflicts, greedy MIS algorithm, Count Mean K/V, Reject Drop, and log(Count) attention compensation:

| Registry method | Priority (smaller is better) | K/V | Rejected Count |
|---|---|---|---|
| `k_only_mis_drop` | `(degree, E-A L2, stable token hash)` | Count Mean | Dropped |
| `k_only_degree_anchor_mis_drop` | `((degree+1)*E-A L2², degree, E-A L2, hash)` | Count Mean | Dropped |
| `k_only_anchor_first_mis_drop` | `(E-A L2², degree, hash)` | Count Mean | Dropped |

The Degree-first baseline is not modified. The two new methods share the same E–E conflict graph and do not introduce new trainable/tunable coefficients, alternative importance scores, or E–A gates. E–A first is lexicographic; degree matters only for exactly tied distances. An isolated E–E node has degree zero and is accepted regardless of priority.

### Other retained controls

`full_cache`, `merge_all`, `importance_weighted_merge_all`, legacy `k_only_mis_drop_allcount`, `k_only_importance_mis_drop` (Count×Importance D-first + Count Mean), and `k_only_importance_iwmean_drop` (Count×Importance D-first + Count×Importance Mean). The pure-Importance Mean `k_only_importance_imean_drop` variants and original Count/Importance sink controls remain independently switchable. Deprecated Anchor-β, High/Low-risk, historical-Query adaptive λ, and IA E–A/E–E methods and their scripts have been removed.

## Run

Edit model, budgets, methods and evaluation lengths in `diagnosis_experiments/run_cosine_mis.sh`. The defaults enable the Degree-first baseline and both new priorities, as well as `k_only_importance_iwmean_drop` and IW-MergeAll. No external environment-variable interface is required.

```bash
cd /home/tlxr/CohereKV
python diagnosis_experiments/test_konly_importance_mis.py
python diagnosis_experiments/test_degree_anchor_priorities.py
bash diagnosis_experiments/run_cosine_mis.sh
```

## Bulk-prefix continuation PPL (replaces streaming PPL)

The only PPL protocol is now **one-time bulk KV compression**. Edit the three
main controls in `run_cosine_mis.sh`:

```bash
PPL_CONTEXT_LEN=4096             # P: uncompressed prefix length
PPL_BUDGETS=("0.05" "0.1" "0.2" "0.4")  # retain fraction per KV head
PPL_CONTINUATION_LEN=256         # N: number of scored post-compression targets
```

Additional workload controls: `PPL_MAX_SEQUENCES=4` (0 = all complete test
windows) and `PPL_PREFILL_CHUNK_SIZE=128` (model-forward chunks **only**, never
extra compressions). `RUN_PPL=0` runs Replay without PPL.

For each nonoverlapping window the test protocol is:

1. Prefill **P context tokens** without any compression. Capture KeepKV
   historical importance from the final prefill chunk only.
2. At the **P-token boundary**, compress the original, uncompressed KV exactly
   once to `B = round(P * budget_ratio)` slots **per KV head**. No fixed token
   count is entered in the Shell; no future tokens influence selection.
3. Process **one unscored ground-truth continuation token** after compression
   to generate the first post-compression logit. This is necessary because the
   prefix's last logit predates compression and must **not** score a target.
4. Score the following **N ground-truth continuation targets**, using FP32 CE
   and FP64 accumulation. Every scored logit comes from a forward after the
   bulk compression. Continue with teacher forcing; **do not recompress** the
   newly appended continuation tokens. This is PPL, not free-text generation.

Thus each test window consumes `P + 1 + N` tokens; precisely **N** are scored.
All methods see identical prefix/warm-up/target windows and the same B for a
given budget. The reported `scored_tokens = N * number_of_windows`. Even the
full-cache control is evaluated on only these continuation targets, so absolute
PPL values are **not directly comparable** to the old whole-window GPTQ PPL.

The compressed continuation uses logical RoPE positions and a causal mask over
the physical KV layout, with Count-aware `log(C)` bias on merged prefix slots.
Because there is only one compression from uncompressed KVs, all input Counts
are initially one. The physical cache grows beyond B during continuation; it
is not repeatedly reduced back to B. This tests E–E interference from a large
**single** eviction, not streaming memory-budget enforcement.

The old `--seqlen` / `--step_size` streaming CLI and its repeated-compression
scoring loop are **removed**. Use `--context_len`, `--continuation_len`, and
`--prefill_chunk_size` instead. PPL CSV and metadata include the protocol,
context length, continuation length, effective budget slots and scored-token
counts. Prior results cannot be resumed under this new protocol.

Attention Replay is unaffected: it still evaluates held-out future Queries
without using them in MIS selection.

The script writes per-configuration Future-query Attention-L2 CSV, per-Head/group diagnostics, bulk-prefix continuation-only PPL CSV, and `.pdf`/`.png` figures (Times New Roman, vector PDF and 600-DPI PNG). After a successful run, upload the result archive at the final `[UPLOAD]` path:

`diagnosis_experiments/results/<MODEL>/<EXPERIMENT>/journal_ia_mis_results.zip`

Important columns include `mis_priority`, `accepted_mask_diff_vs_degree`, `n_mis_accepted`, `n_mis_rejected`, `mean_anchor_drift`, `output_rel_error` (Replay), and `ppl` (PPL). Paired comparisons should hold model, token windows, routing, KV budget, lambda, Count, and fusion mode fixed.
