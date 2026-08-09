# MyMethod selector log tags

Every `mymethod` evaluation emits a canonical selector tag to stdout/logs:

```text
[MYMETHOD_SELECTOR_TAG] <tag>
[MYMETHOD_SELECTOR_ACTIVE_CONFIG] <json>
```

Only hyperparameters that are active for the selected retained/evicted partition
are encoded in the tag. Inactive selector-family parameters are intentionally
omitted.

Examples:

```text
selector_attention
selector_keepkv__sink4__recent128__window128__beta1__logc0
selector_d2o__sink4__recent128__logc1__reset0
```

Meanings:

- `keepkv`: `sink`, `recent`, `window`, `beta`, `logc`.
- `d2o`: `sink`, `recent`, `logc`, `reset`.
- `attention`: no selector-specific hyperparameters.

The same canonical tag is used by `run_ppl_kvmerge_sweep.sh`,
`run_longbench_kvmerge_sweep.sh`, `run_system_kvmerge_sweep.sh`,
`run_mymethod_light_compare.sh`, and `run_ablation_mymethod_sweep.sh` when
constructing mymethod log/result filenames.

The tag describes only selector behavior. `lambda`, group mode, graph backend,
route backend, and other certified-grouping parameters remain separate mymethod
core configuration fields.
