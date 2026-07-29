# ffn_tp_parallel — tensor-parallel FFN in two dependent stages

**Two chip tasks per rank, chained by a dependency nothing in the Python
declares.** Stage 2 must not start until stage 1 has produced its output, and
the orch function contains no barrier, no event, and no ordering call — the
edge is discovered from the fact that both tasks name the same buffer.

Per rank:

| Stage | Core | Computation |
| ----- | ---- | ----------- |
| 1 | AIC | `partial_local = x_shard @ w_shard` |
| 2 | AIV | `y = sum over ranks of partial_local` |

## What this exercises

| Concept | How |
| ------- | --- |
| **Implicit producer/consumer edge** | `host_partial[i]` is `OUTPUT_EXISTING` on the stage-1 submit and `INPUT` on the stage-2 submit. Both carry the same `buffer.addr`, so TensorMap links the two tasks itself — there is no barrier, no event, and no ordering call in the orch function. |
| **Mixed core types in one DAG** | Stage 1 compiles with `core_type="aic"` and its orchestration calls `rt_submit_aic_task`; stage 2 uses `core_type="aiv"` and `rt_submit_aiv_task`. One `Worker`, one `run()`. |
| **`func_id`, not core id** | The integer in `children=[(0, core_callable)]` is the `func_id` the orchestration passes to `rt_submit_*_task(func_id, params)` — here `0` for the matmul and `1` for the reduce. It selects *which child kernel*, not which core type. |
| **Cross-rank exchange through a domain buffer** | The stage-2 kernel reduces over a `scratch` buffer in the communication window: a mailbox of `nranks * M * N` floats followed by a signal tail of `nranks` int32 slots. |
| **`config_name` on `ChipCallable.build`** | Both callables name an orchestration config symbol alongside the entry function. |

Read the stage-2 kernel as the four-phase collective it is: publish into the
peer's mailbox slot, notify, wait on the signal tail, accumulate.

## Run

```bash
# Simulation (2 ranks)
python examples/workers/l3/ffn_tp_parallel/main.py -p a2a3sim -d 0-1

# Hardware (2 ranks)
python examples/workers/l3/ffn_tp_parallel/main.py -p a2a3 -d 0-1
```

Or as a scene test — `a2a3sim`, `a2a3`, and `a5sim` are marked:

```bash
pytest examples/workers/l3/ffn_tp_parallel --platform a2a3sim
```

Exactly two devices are required — `parse_device_range` rejects any other
count. `M = K = N = 64` here and must match `TILE` / `kRows` / `kCols` in the
AIC and AIV kernels; changing one without the others silently corrupts the
result.

## Verification

Every rank's `y` is compared against `sum over r of x_shard[r] @ w_shard[r]`
computed in torch, with `rtol = atol = 1e-4` — the same tolerance
`scene_test` applies. All ranks must match: an allreduce that only converged
on rank 0 fails here.

## File structure

```text
ffn_tp_parallel/
├── kernels/
│   ├── aic/
│   │   └── kernel_local_linear.cpp
│   ├── aiv/
│   │   └── kernel_allreduce_sum.cpp
│   └── orchestration/
│       ├── allreduce_sum_orch.cpp
│       └── ffn_local_orch.cpp
├── main.py
├── test_ffn_tp_parallel.py
└── README.md
```
