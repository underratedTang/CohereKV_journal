# D2O-style selector in mymethod

`MYMETHOD_SELECTOR="d2o"` changes only the retained/evicted partition. All
subsequent routing and merging remains mymethod.

Selection state is request-level cumulative post-softmax attention per physical
KV head. GQA query heads sharing one KV head are averaged before accumulation.
`MYMETHOD_D2O_USE_LOG_COUNT=1` accumulates the actual mymethod attention after
`+log(C)`; setting it to `0` is an explicit no-logC selector ablation.

The D2O-specific controls are independent of KeepKV controls:

```bash
MYMETHOD_D2O_SINK_SIZE=4
MYMETHOD_D2O_RECENT_SIZE=128
MYMETHOD_D2O_SINK_CAN_ABSORB=1
MYMETHOD_D2O_USE_LOG_COUNT=1
```

The recent region is protected directly. Sink entries are forced into the
heavy-hitter budget. Both sink and recent retained entries may act as mymethod
certified anchors when `MYMETHOD_D2O_SINK_CAN_ABSORB=1`; they are not guaranteed
to keep original K/V unchanged if certified candidates merge into them.

Compression does not clear cumulative importance. After compression, the state
is gathered to retained anchors; source history is not added to destinations.
A new request clears the cumulative state.

Not imported from D2O: cosine nearest-neighbor routing, EMA similarity threshold,
similarity-weighted merging, and dynamic layer cache allocation.
