# D2O-style selector in mymethod

`MYMETHOD_SELECTOR="d2o"` only determines the retained/evicted partition.  Once
that partition is formed, all routing, complete-group certification, conflict
Graph-MIS selection, K/V merging, C/R/D propagation, and decode-time `+log(C)`
are mymethod operations.

## Importance state

Per physical KV head, D2O-style importance is post-softmax attention mass. GQA
query heads sharing a physical KV head are averaged before query rows are
summed.

`MYMETHOD_D2O_USE_LOG_COUNT=1` (default) uses the actual mymethod attention
computed from `qK/sqrt(d) + log(C)`.  Setting it to `0` recomputes selector-only
post-softmax attention without the count bias; this is an explicit ablation.

Temporal persistence is controlled independently:

```bash
MYMETHOD_D2O_SCORE_RESET_INTERVAL=0
```

- `0`: request-level cumulative importance (D2O-style default);
- `1`: reset importance after every compression;
- `N>1`: reset after every N compression rounds.

The compression round that triggers a reset first uses the accumulated score to
form retained/evicted sets; the reset happens only after compression.

## Structural protection

```bash
MYMETHOD_D2O_SINK_SIZE=4
MYMETHOD_D2O_RECENT_SIZE=128
```

Sink entries consume the heavy-hitter budget and recent entries are protected
separately.  These parameters are selector-only.  There is no
`SINK_CAN_ABSORB` switch: every retained entry, including sink and recent
entries, becomes an ordinary legal mymethod anchor.  Actual merging still
requires the mymethod pair certificate and Graph-MIS consistency checks.

After compression without a score reset, the D2O importance state is gathered
to retained anchors only.  Evicted-source historical importance is not added to
its merge destination.

## Relation to the attention selector

The current-attention selector uses the same actual post-softmax score but
accumulates only since the previous compression and resets after every
compression.  Therefore, under the same observation boundary,

```bash
MYMETHOD_SELECTOR="d2o"
MYMETHOD_D2O_SINK_SIZE=0
MYMETHOD_D2O_RECENT_SIZE=0
MYMETHOD_D2O_USE_LOG_COUNT=1
MYMETHOD_D2O_SCORE_RESET_INTERVAL=1
```

is intentionally implemented to use the same global retained/evicted partition
as `MYMETHOD_SELECTOR="attention"`, including tie handling.

Not imported from D2O: cosine nearest-neighbor routing, EMA merge threshold,
similarity-weighted K/V merging, or dynamic layer allocation.
