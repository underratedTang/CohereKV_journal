# NPC 2026 First Diagnostic Results Review

## Evaluated configuration

The uploaded first-run results use Llama-3.2-3B, WikiText-2, two 8192-token evaluation windows, chunk size 32, compression every chunk, and compression ratio 0.4 for PPL diagnostics.

## Main finding: the current certificate is coverage-limited

At the currently tested thresholds, the certified method accepts almost no evicted count mass:

| tau | accepted count-mass ratio | PPL |
|---:|---:|---:|
| 0.6 | 0.0269% | 6.55094 |
| 0.7 | 0.00354% | 6.55583 |
| 0.8 | 0% | 6.55111 |
| 0.9 | 0% | 6.55111 |

The eviction-only PPL is 6.55111. Thus tau 0.8 and 0.9 are exactly eviction-only in this run, and tau 0.6/0.7 transfer negligible information. These settings cannot substantiate the claim that certified grouping preserves evicted information.

The offline real-key diagnostic confirms the cause. At compression ratio 0.4, mean certified acceptance is approximately:

| tau | certified accepted-entry ratio |
|---:|---:|
| 0.1 | 22.87% |
| 0.2 | 10.63% |
| 0.3 | 3.57% |
| 0.4 | 0.865% |
| 0.5 | 0.071% |
| 0.6 | 0% |

Therefore the urgent PPL sweep is tau in {0.1, 0.2, 0.3, 0.4}, not {0.6, 0.7, 0.8, 0.9}.

## What the results already support

- Ordinary star grouping exhibits real complete-group violations. The recorded maximum group violation rate reaches 16.9%, with diameter/delta up to 1.49.
- GroupLogitKV grouping can have very loose complete groups; the diagnostic recorded heads with all merged groups violating the method's Euclidean diameter budget and maximum diameter/delta above 10.
- The certified method has zero measured exact-diameter violations and zero radius/diameter certificate underestimates across the tested multiround trace.
- Unconstrained nearest merging accepts all evicted mass but worsens PPL to 6.57713 and reaches diameter/delta above 26.7. This supports the need for a quality gate.

These results support the safety/compactness motivation, but not yet the information-preservation advantage because coverage is too low.

## Claims that should not be emphasized yet

- Nearest-feasible-anchor rescue: the measured rescue ratio is zero in the current snapshot. Do not use it as a headline empirical contribution unless a setting with nonzero rescue is found.
- Sink absorption: disabling sink absorption produced identical PPL and coverage in the current run.
- End-to-end speed advantage: the current eager implementation is slower than GroupLogitKV and CentroidKV.

## Engineering result

For length 8192 and compression ratio 0.4, distance chunk size 256 substantially reduces the full method's peak temporary memory relative to an unchunked matrix and is faster in the uploaded benchmark. Use 256 for the next practical tests rather than 0.

## Time-limited next step

Run:

```bash
bash run_npc2026_fast_followup.sh
```

It runs only WikiText-2 and two evaluation sequences, adds direct PPL baselines, and scans tau 0.1-0.4 with the GroupLogitKV selector. The best tau should be selected by PPL together with accepted count-mass ratio, not by PPL alone.
