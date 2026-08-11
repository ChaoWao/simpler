# domain_rank_map — what a communication domain hands each chip

The reference for reading a `ChipDomainContext`. Three chips, two overlapping
domains, and a first pass that submits **no tasks at all** — it allocates both
domains and inspects the handles, so you can see exactly what
`orch.allocate_domain` returns before any kernel muddies the picture.

```text
even = workers [0, 2]
tail = workers [1, 2]        # chip 2 is in both
```

## What this exercises

| Property of the domain handle | Asserted here |
| ----------------------------- | ------------- |
| **`domain_rank` is dense and follows `workers` order** | `even` gives chip 0 → rank 0 and chip 2 → rank 1; `tail` gives chip 1 → rank 0 and chip 2 → rank 1. A chip's rank is per domain, not global — chip 2 is rank 1 in both, chip 0 is rank 0 in one and absent from the other. |
| **A non-member chip is absent, not zero-valued** | `tail[0]` raises `KeyError`. Indexing is the membership test. |
| **Overlapping domains carve separate buffers** | `even[2].buffers["scratch"].base != tail[2].buffers["scratch"].base` — chip 2's two memberships do not alias. |
| **`device_ctx` and buffer pointers are non-null** | Both must be set before a kernel can address the window. |
| **Two lifetime styles** | The inspection pass allocates both domains and releases them in a `finally` via `handle.release()`; the reduce pass uses `with orch.allocate_domain(...) as handle`. |

After the inspection pass, each domain runs its own small allreduce — in its
**own `worker.run()`**, so chip 2 never juggles two collectives at once. Both
domains are live simultaneously only during the inspection pass, which runs no
collective; `dual_domain_overlap` drives real work through two overlapping
domains and likewise gives each its own run. The ranks within each allreduce
are one `submit_next_level_group`, so every peer that participates in the
device barrier is dispatched as a complete set.

## Run

```bash
# Simulation (3 chips)
python examples/workers/l3/domain_rank_map/main.py -p a2a3sim -d 0-2

# Hardware (3 chips)
python examples/workers/l3/domain_rank_map/main.py -p a2a3 -d 0-2
```

Or as a scene test — `a2a3sim`, `a2a3`, and `a5sim` are marked (the pytest
path calls `run()` directly, so it is not limited by `main()`'s `--platform`
choices):

```bash
pytest examples/workers/l3/domain_rank_map --platform a2a3sim
```

Exactly three devices are required. The kernels are borrowed from
`../dual_domain_overlap/kernels/` rather than duplicated.

## Verification

Two independent checks, both required to pass:

1. **Structural** — the rank map, the `KeyError`, and the distinct scratch
   pointers described above.
2. **Numerical** — each domain's allreduce output equals the sum of its own
   members' inputs, within `1e-3`. This is what proves data moved *only*
   among a domain's participants: chip 1's contribution must not appear in
   `even`'s result.

## File structure

```text
domain_rank_map/
├── main.py
├── test_domain_rank_map.py
└── README.md
```
