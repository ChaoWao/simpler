# dual_domain_overlap — one chip in two communication domains

Three chips, two domains that share a member:

```text
left  = workers [0, 1]
right = workers [1, 2]        # chip 1 is in both
```

Chip 1 receives **two independent `ChipDomainContext` objects**, one per
domain, each with its own rank, its own scratch pointer, and its own peers.
Nothing about being in one domain constrains its role in the other.

Where `domain_rank_map` inspects the handles, this example runs real work
through both and then computes on the results.

## What this exercises

| Concept | How |
| ------- | --- |
| **Per-domain identity** | Chip 1 is rank 1 in `left` and rank 0 in `right`. The `workers` list order defines the dense rank, so the same chip legitimately holds two different ranks at once. |
| **Domains allocated inside the orch function** | `with orch.allocate_domain(name=..., workers=..., window_size=..., buffers=[CommBufferSpec(...)])` — created and released within one orchestration, not configured on the `Worker`. |
| **`submit_next_level_group`** | The affine stage submits one `TaskArgs` per member in a single call, with `workers=worker_indices`, instead of a loop of `submit_next_level`. |
| **Compute that depends only on its own domain's result** | Each affine task reads `reduce_out[domain][chip]`. The dependency is implicit — same `buffer.addr` as the reduce output — so `left`'s affine work can never consume `right`'s reduction. |

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
