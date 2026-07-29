# paged_attention_unroll_manual_scope — fewer, larger tasks

Not "the manual-scope variant plus a loop unroll". **Every file here differs
from the baseline**, kernels included — `aiv_softmax_prepare.cpp` is a full
rewrite (312 changed lines against a 156-line original). Read it as a second
implementation of paged attention, not as a patch on the first.

Changed lines against the baseline's copy of the same file, over that file's
original length:

| File | Changed / original |
| ---- | ------------------ |
| `kernels/aic/aic_qk_matmul.cpp` | 145 / 115 |
| `kernels/aic/aic_pv_matmul.cpp` | 164 / 114 |
| `kernels/aiv/aiv_softmax_prepare.cpp` | 312 / 156 |
| `kernels/aiv/aiv_online_update.cpp` | 27 / 256 |
| `kernels/orchestration/paged_attention_orch.cpp` | 226 / 292 |

## What it does differently

The baseline submits four tasks **per KV block**. This one batches up to
`N_UNROLL` blocks into a group and submits four tasks **per group**:

| Task | Computation over the whole group |
| ---- | -------------------------------- |
| `QK` | `qi @ K^T` for `n_blocks` → `sij_buf` shaped `(q_tile, n_blocks * block_size)` |
| `SF` | two-pass softmax over the full `sij_buf` → `pij_buf`, `mi`, `li` |
| `PV` | SplitK-accumulated `P @ V` → `oi_new` |
| `UP` | online-softmax accumulation with group-level `mi`, `li`, `oi_new` |

So the task count drops by roughly `N_UNROLL`, and each task carries
proportionally more work — the trade the whole directory exists to demonstrate.
Softmax becomes a two-pass over a wider buffer, which is why that kernel is
rewritten rather than adjusted.

`N_UNROLL` is a `#define` at the top of the orchestration source, currently
**64**.

Dependencies are declared explicitly inside
`PTO2_SCOPE(PTO2ScopeMode::MANUAL)`, using the primitive
`set_dependencies(deps, n)` form throughout —
[`../paged_attention_manual_scope/`](../paged_attention_manual_scope/) is the
directory to read for that API on its own, against unchanged kernels.

## Cases

Three — `Case1`–`Case3`, all `platforms=["a2a3"]` with
`aicpu_thread_num: 4`. The baseline's small and variable-sequence cases have no
counterpart here.

## Run

```bash
pytest examples/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_manual_scope --platform a2a3 --device 0
```

Onboard only. Wrap in `task-submit` on a shared box.
