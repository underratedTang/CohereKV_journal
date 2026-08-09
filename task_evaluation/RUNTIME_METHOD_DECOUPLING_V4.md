# Runtime method decoupling V4 — superseded

This note described the previous V4 package, which temporarily switched Full/CentroidKV LongBench execution from eager to SDPA to avoid long-prompt OOM. That policy is **superseded** by `UNIFORM_EAGER_PROTOCOL_V5.md`.

Current formal PPL and LongBench protocol: every method uses eager attention; long-context peak memory is handled by exact query-axis tiling inside eager attention rather than by a method-specific backend switch.
