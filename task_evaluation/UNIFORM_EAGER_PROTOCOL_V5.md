# Uniform eager protocol V5

## Formal protocol

Both `evaluate_ppl_kvmerge.py` and `evaluate_longbench_kvmerge.py` require
`--attn_implementation eager` for every method. A non-eager value is rejected
rather than silently converted.

This applies to:

- Full KV (`none` / `None`)
- GroupLogitKV
- mymethod
- SimCalKV
- KeepKV
- CaM
- CentroidKV

## LongBench memory handling

Full KV and CentroidKV do not consume attention scores. Their LongBench path
uses a neutral method-agnostic eager runtime that evaluates the query axis in
bounded blocks. For each query block it still uses the complete key/value
prefix, the same additive/causal mask, fp32 softmax, and eager value matmul.
No compression is performed between query blocks. Therefore the prompt has the
same single prefill/compression boundary as before; only the temporary
`[B,H,Q,K]` workspace is tiled.

Method-specific runtimes remain isolated:

- GroupLogitKV: own eager runtime and count-calibrated decode semantics.
- mymethod: own eager runtime and selector/count state.
- SimCalKV/CaM: neutral eager reduced-score capture runtime.
- KeepKV: own eager runtime and its existing one-shot chunked-prompt sufficient-statistic path.
- CentroidKV: neutral exact-eager runtime + CentroidKV K/V merge only.

## Runtime tile

LongBench exposes:

`--eager_attention_chunk_size 128`

This is a workspace tile, not a method hyperparameter and not a compression
boundary. `run_longbench_kvmerge_sweep.sh` exposes the matching editable top-of-file variable `EAGER_ATTENTION_CHUNK_SIZE`.

## Logging

Formal runs log:

`[ATTENTION_PROTOCOL] requested=eager effective=eager mode=eager ...`

PPL JSON records `attention_protocol=uniform_eager`. LongBench prediction rows
record `attn_implementation=eager`, `attention_protocol=uniform_eager`, and the
query tile size.
