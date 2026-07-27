# paged_attention — the baseline

Online-softmax paged attention in bfloat16, split across AIC and AIV. This is
the reference the three sibling variants are read against; start here.

## The four-task loop

Per (batch, head) and per KV block, the orchestration submits four tasks:

| Task | Core | Computation |
| ---- | ---- | ----------- |
| `QK` | AIC | `qi @ K^T` for the block |
| `SF` | AIV | softmax prepare — running `mi`, `li` |
| `PV` | AIC | `P @ V` |
| `UP` | AIV | online-softmax accumulation into the running output |

Ordering is **not written down**. The tasks sit inside a plain `PTO2_SCOPE()`
with a `PTO2_SCOPE_GUARD()`, and TensorMap derives `QK → SF → PV → UP` from the
tensors they share. That is the whole point of the baseline: the dependency
graph is a consequence of the data, not of the code.

## Cases

Seven, all for `platforms=["a2a3"]` with automatic AICPU thread selection:
`Case1`–`Case3` at production scale (up to batch 256, 16 heads, 8192 context),
`CaseSmall1` / `CaseSmall2`, and `CaseVarSeq2` / `CaseVarSeq4` for ragged sequence lengths.
The four kernels are registered as sub-callables named `QK`, `SF`, `PV`, `UP`.

## Run

```bash
pytest examples/a2a3/tensormap_and_ringbuffer/paged_attention --platform a2a3 --device 0
pytest examples/a2a3/tensormap_and_ringbuffer/paged_attention --platform a2a3 --device 0 -k Case1
```

Onboard only. Wrap in `task-submit` on a shared box.

## The variants

| Directory | Differs how |
| --------- | ----------- |
| [`../paged_attention_manual_scope/`](../paged_attention_manual_scope/) | **Orchestration only.** All four kernels are byte-identical to this directory's; 45 of 292 orchestration lines change to declare dependencies explicitly. |
| [`../paged_attention_unroll_manual_scope/`](../paged_attention_unroll_manual_scope/) | **A different implementation.** Kernels and orchestration are both substantially rewritten — it batches KV blocks into groups instead of one task set per block. |
| [`../paged_attention_ringbuffer/`](../paged_attention_ringbuffer/) | **No kernels of its own.** A ring-sizing stress harness that borrows the `tests/st` batch-paged-attention kernels. |
