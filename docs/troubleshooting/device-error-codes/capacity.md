# Chasing down a capacity code (1, 2, 4)

← [Device Error Codes](../device-error-codes.md)

SCOPE_DEADLOCK, HEAP_RING_DEADLOCK and DEP_POOL_OVERFLOW all say the same thing —
"some resource ran out" — without saying *which* resource or *which* scope. Do not
guess at the ring sizes. Turn on `scope_stats`, which records the high-water mark of
all four resources (task-window slots, heap bytes, dep-pool entries, tensormap
entries) per `PTO2_SCOPE`:

```python
cfg = CallConfig()
cfg.enable_scope_stats = True
cfg.output_prefix = "outputs/my_run"
worker.run(callable, args, cfg)
```

It works on a **failing** run: the metadata line is marked `"fatal": true` and
everything written before the fatal is kept. So point it straight at the workload
that trips the code.

| Bottleneck resource | Code | Fix |
| ------------------- | ---- | --- |
| task window | 1 | raise `ring_task_window` (`PTO2_RING_TASK_WINDOW`), or split the scope |
| heap | 2 | raise `ring_heap` (`PTO2_RING_HEAP`), or shrink per-task args / intermediate tensors |
| dep pool | 4 | raise `ring_dep_pool` (`PTO2_RING_DEP_POOL`), or cut the fanin count |

The runtime's own `error hint:` line already names the knob for the code it
latched; this table is for when you have `scope_stats` output and want to read the
peak back to a knob.

Report fields, the `Top Peaks` table and the plotting tool are documented in
[`../../dfx/scope-stats.md`](../../dfx/scope-stats.md).
