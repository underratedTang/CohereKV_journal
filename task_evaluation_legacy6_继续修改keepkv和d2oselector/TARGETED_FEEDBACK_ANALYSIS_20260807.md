# Graph-MIS targeted feedback analysis (2026-08-07)

Protocol: Llama-3.2-3B, WikiText-2, 2 x 8192 tokens, chunk 32, compression ratio 0.4.

## Quality

- KeepKV public baseline: PPL 6.530204.
- CentroidKV public baseline: PPL 6.589800.
- MD-default KeepKV-style selector, eviction only: 6.585191.
- MD-default KeepKV-style selector, anchor-only lambda=0.5: 6.584929.
- Current-attention selector, anchor-only lambda=0.65: 6.547378.
- Current-attention selector, full Graph-MIS lambda=0.65: 6.543430.
- Current-attention selector, anchor-only lambda=0.75: 6.546109.
- Current-attention selector, full Graph-MIS lambda=0.75: 6.541927.

The default selector is the dominant quality bottleneck: even eviction-only with that split is ~6.585. With the controlled current-attention split, full Graph-MIS consistently improves over anchor-only by ~0.004 PPL while rejecting only about 1.6--2.0% of routed entries. At lambda=0.75, full Graph-MIS accepts 50.16% of evicted entries / 41.42% evicted count mass and maintains max post D/Delta < 1, whereas anchor-only reaches max D/Delta ~1.60.

## Engineering cost

At a 2048-token layer-0 snapshot:

- lambda=0.5: full plan 5.548 ms, anchor-only 4.252 ms, CentroidKV merge core 1.609 ms; full peak temporary allocation ~102.3 MiB.
- lambda=0.75: full plan 8.998 ms, anchor-only 5.447 ms, CentroidKV merge core 1.580 ms; full peak temporary allocation ~381.9 MiB.

The conflict-graph/MIS cost rises sharply with candidate-pool size (max pool 11 -> 36). This is now the primary engineering bottleneck. The next implementation optimization should target compact triangular/bitset conflict storage, size bucketing, and device-side/fused early-termination MIS; the mathematical grouping rule should not be changed based on these results.

## Next minimal experiment

Run `bash run_mymethod_lambda_refine.sh` to test lambda 0.85 and 0.95 under the controlled current-attention selector. Do not rerun public baselines. If PPL stops improving or Graph-MIS cost/coverage becomes excessive, use lambda=0.75 as the current operating point and proceed to LongBench.
