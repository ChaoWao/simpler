# dual_domain_overlap — one chip in two communication domains

Three chips, two domains that share a member:

```text
left  = workers [0, 1]
right = workers [1, 2]        # chip 1 is in both
```

Chip 1 receives **an independent `ChipDomainContext` per domain**, each with
its own rank, its own scratch pointer, and its own peers. Nothing about being
in one domain constrains its role in the other.

The overlap is in **membership, not in time**: each context is created in its
own domain's run, the two are never live at the same time, and the two
domains' collectives never run concurrently. See
[Execution shape](#execution-shape).

Where `domain_rank_map` inspects the handles, this example runs real work
through both and then computes on the results.

## What this exercises

| Concept | How |
| ------- | --- |
| **Per-domain identity** | Chip 1 is rank 1 in `left` and rank 0 in `right`. The `workers` list order defines the dense rank, so the same chip legitimately holds a different rank in each domain. |
| **Domains allocated inside the orch function** | `with orch.allocate_domain(name=..., workers=..., window_size=..., buffers=[CommBufferSpec(...)])` — created and released within one orchestration, not configured on the `Worker`. |
| **`submit_next_level_group`** | Both the peer-waiting allreduce and the affine stage submit one `TaskArgs` per member in a single call with `workers=worker_indices`. |
| **Per-domain outputs on a shared chip** | `reduce_out` and `affine_out` are keyed by `(domain, chip)`, so chip 1 owns a separate buffer in each domain and writes a different value to each. `left`'s affine reads `reduce_out["left"][chip]`, so it cannot see `right`'s reduction. |

Ordering between the reduce and affine stages comes from the run boundary,
not from a producer/consumer edge — see [Execution shape](#execution-shape).
For an example where TensorMap's implicit same-`buffer.addr` dependency is
what orders two stages, see [`../ffn_tp_parallel/`](../ffn_tp_parallel/).

## Execution shape

Three synchronous `worker.run` calls — `run()` is `submit(...).wait()`, so
each one finishes before the next is submitted:

```text
run 1   allocate domain "left"   →  allreduce group over workers [0, 1]
run 2   allocate domain "right"  →  allreduce group over workers [1, 2]
run 3   affine group over [0, 1]  +  affine group over [1, 2]
```

Each run is its own DAG. A domain allocated inside a run stays live for that
whole run: `release()` on `with` exit is a non-blocking mark, and the backend
release happens after the run's completion wait, so tasks already submitted
with that `device_ctx` still see live memory.

The allreduce is one `submit_next_level_group` per domain because its Phase-2
device barrier makes every rank wait for its peers — a rank dispatched
without them cannot finish. The affine stage is grouped for symmetry; its
kernel does no cross-rank communication.

## Run

```bash
# Simulation (3 chips)
python examples/workers/l3/dual_domain_overlap/main.py -p a2a3sim -d 0-2

# Hardware (3 chips)
python examples/workers/l3/dual_domain_overlap/main.py -p a2a3 -d 0-2
```

Or as a scene test — `a2a3sim`, `a2a3`, and `a5sim` are marked (the pytest
path calls `run()` directly, so it is not limited by `main()`'s `--platform`
choices):

```bash
pytest examples/workers/l3/dual_domain_overlap --platform a2a3sim
```

Exactly three devices are required.

## Verification

Per domain and per member chip:

1. `reduce_out` equals the sum of that **domain's** inputs — chip 2's data must
   not reach `left`, nor chip 0's reach `right`.
2. `affine_out` equals `reduce_out * scale[chip] + bias[chip]`, with per-chip
   `scale` and `bias`, so a task that ran against the wrong chip's parameters
   is caught.

Both are checked to `1e-3`, and both tensors are additionally screened with
`torch.isfinite` — a non-finite value means the window was read before the
peer published, which a max-diff test alone can mask.

## File structure

```text
dual_domain_overlap/
├── kernels/
│   ├── aiv/
│   │   ├── affine_kernel.cpp
│   │   └── domain_allreduce_sum.cpp
│   └── orchestration/
│       ├── affine_orch.cpp
│       └── domain_allreduce_orch.cpp
├── main.py
├── test_dual_domain_overlap.py
└── README.md
```

`domain_allreduce_sum.cpp` and `domain_allreduce_orch.cpp` are also compiled by
`../domain_rank_map/`, which reaches into this directory rather than keeping a
copy.
