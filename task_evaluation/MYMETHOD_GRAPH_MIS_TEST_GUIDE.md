# Focused graph-MIS mymethod test guide

The first goal is to establish method correctness and whether the new graph-MIS
actually merges nontrivial token mass. Expensive LongBench sweeps should wait
until these checks pass.

## A. CPU/unit correctness (no model weights)

```bash
cd /home/tlxr/CohereKV/task_evaluation
PYTHONPATH=. python tests/test_mymethod.py -v
PYTHONPATH=. python tests/test_ablation_wrapper.py -v
PYTHONPATH=. python tests/test_centroidkv.py -v
```

Expected: all tests pass. They cover reference-scale equivalence, scale
invariance, conflict rejection, independent/maximal MIS, exact first-round
diameter, count-weighted K/V, frozen scale, LLaMA/Qwen parity, and randomized
cross-round certificate invariants.

## B. Real-KV geometry + grouping-cost diagnostic (run first)

Edit the manual variables in `run_mymethod_quick_validation.sh`, then:

```bash
bash run_mymethod_quick_validation.sh
```

Return the generated feedback zip. The important files are:

- `kv_snapshot.pt`
- `graph_mis_analysis.json` / `.csv`
- `grouping_benchmark.json`
- `ppl_*.json`
- `ppl_*_diagnostics.jsonl`

Questions answered:

1. Does pair-compatible routing find candidates?
2. What fraction survives graph MIS?
3. Does `full` stay strictly below the exact normalized-diameter threshold while
   `anchor_only` creates conflicting groups?
4. Is degree-priority maximal MIS close to the exact maximum independent-set
   cardinality on small candidate pools?
5. How much time is anchor routing versus conflict-graph + MIS?
6. Does PPL improve over `eviction_only`, proving the method is not merely
   dropping all evicted information?

## C. Lightweight comparison after choosing lambda

Set `LAMBDA_MYMETHOD` in `run_mymethod_light_compare.sh` from step B and run:

```bash
bash run_mymethod_light_compare.sh
```

This keeps all comparison methods. `GroupLogitKV` is labeled internal/unpublished;
it is useful only to isolate the grouping change. Public comparison methods are
CentroidKV, KeepKV, SimCalKV and CaM.

## D. Full evaluators

Once the method passes B/C, use the existing scripts:

```bash
bash run_ppl_kvmerge_sweep.sh
bash run_longbench_kvmerge_sweep.sh
bash run_system_kvmerge_sweep.sh
bash run_ablation_mymethod_sweep.sh
```

PPL remains time-limited to WikiText-2 with `MAX_EVAL_SEQUENCES=2` in the
provided shell defaults, per the current experimental constraint.
