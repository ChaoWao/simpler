# paged_attention_ringbuffer — deliberately undersized rings

Not really a paged-attention example. **This directory contains no kernels at
all** — a single test file, which points its `CALLABLE` at the batch-paged-
attention kernels under `tests/st`:

```python
PA_KERNELS = "../../../../tests/st/a2a3/tensormap_and_ringbuffer/batch_paged_attention/kernels"
```

The attention math is borrowed, and correct output is only the pass criterion.
What is under test is the **ring buffers**: whether rotation and reclamation
survive a real workload when the rings are far too small for it.

## The knobs

One case, `ringbuffer_stress`, sized to hurt:

| `runtime_env` key | Value here | Note |
| ----------------- | ---------- | ---- |
| `ring_task_window` | 64 | Default is 16384 (`PTO2_TASK_WINDOW_SIZE`, `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2_types.h`) — **256× smaller** |
| `ring_heap` | 4 MiB | bytes per ring |
| `ring_dep_pool` | 256 | |

At a 64-task window, a workload that submits thousands of tasks must rotate the
window continuously — every task record is overwritten many times over during
one run. If reclamation is off by one, or a record is reused while still
referenced, the golden check fails or the orchestrator wedges.

Two things this example is the reference for:

- **Per-case ring sizing through `config.runtime_env`**, not the process-global
  `PTO2_RING_*` environment variables. The rings are sized per task, so one
  suite can mix a stress case with normally-sized ones.
- **Non-power-of-2 sizes are accepted.** 4 MiB is chosen to keep the stress
  intent compact, not because the runtime requires a round number.

It also exercises `INOUT` tensors, bfloat16, and mixed AIC+AIV execution
incidentally — the `UP` kernel takes four `INOUT` arguments.

## Run

```bash
pytest examples/a2a3/tensormap_and_ringbuffer/paged_attention_ringbuffer --platform a2a3 --device 0
```

Onboard only. Wrap in `task-submit` on a shared box.

A failure here is usually a ring bug, not an attention bug — the same kernels
pass in `tests/st/a2a3/tensormap_and_ringbuffer/batch_paged_attention/` at
default sizes, which is the first thing to check.

## See also

- [`../paged_attention/`](../paged_attention/) — the baseline implementation and its variants
- [`docs/troubleshooting/device-error-codes/capacity.md`](../../../../docs/troubleshooting/device-error-codes/capacity.md) — the sub-class codes each of these three rings raises when it overflows, and what to do about it
