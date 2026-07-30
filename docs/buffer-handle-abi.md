# BufferHandle / Tensor — the L3+ memory model

At L3 and above, tasks name their data with typed, self-describing **buffer
handles and views**, not raw pointers. This replaces the legacy "raw pointer +
`child_memory` bool" mechanism with an ABI that carries a canonical identity, a
backend descriptor, and a strided view — so a buffer can be resolved exactly
across the L3→L2 (and L4→L3) boundaries without a side table.

This page is the user-facing how-to. The byte layout itself is pinned by the
`static_assert`s in [`src/common/task_interface/buffer_handle.h`](../src/common/task_interface/buffer_handle.h)
— sizes, field offsets, and enum values all fail the build if they drift, and
[`tests/ut/cpp/types/test_buffer_handle.cpp`](../tests/ut/cpp/types/test_buffer_handle.cpp)
pins them again from the outside along with the blob codec.

## Three types

| Type | What it is | Where it lives |
| ---- | ---------- | -------------- |
| **`BufferHandle`** | An owned backing (POSIX shm / fork-COW / device malloc) with a canonical identity + lifecycle. Stays with the Worker that created it. | owner side (L3+) |
| **`Tensor`** | A **self-describing task argument**: the full handle descriptor embedded + a strided view `(byte_offset, shapes, strides, dtype)`. The wire element of `TaskArgs`. Carries no materialized address. | what you build and submit |
| **`ChipTensor`** | The materialized POD the device runtime ABI reads (address + strided view). Exists **only** at the L2 device-runtime boundary. | L2 leaf, internal |

**You never build a `ChipTensor`.** Name a `Tensor` over a handle and submit it;
the address is resolved on the consuming endpoint, and the C++ orchestration on
the chip receives the resolved form.

## Why `Tensor` and `ChipTensor` are two types

The two sit on opposite sides of one resolution step. A `Tensor` *cannot* carry an
address: at submit time none exists — a `POSIX_SHM` backing maps to a different VA
in every process, and a `DEVICE_MALLOC` one is only valid on its owner chip. A
`ChipTensor` *must* carry one; it is what the kernel dereferences. Two ways to
collapse that into a single type were considered and dropped.

### Rejected: merge `Tensor` into `ChipTensor`

Drop `buffer.addr`, add the handle descriptor, and have the H2D staging step
rewrite the backend tag and body (and mint a fresh identity for the device copy).
That is self-consistent, but it charges the device for host-side fields:

- **Device cost.** `sizeof(ChipTensor)` is pinned at **128 B** by
  `static_assert(… == 128, "Tensor must be 2 cache lines")` and by
  `PTO2_TASKPAYLOAD_TENSOR_STRIDE`, which the AICPU scheduler strides the payload
  with. A merged struct is ~192–216 B, taking `PTO2TaskPayload` from 4864 B to
  6912–7680 B — **+42% to +58% per task slot** — because the payload embeds
  `ChipTensor tensors[MAX_TENSOR_ARGS]` **by value** in the device task ring.
- **Zero return for that cost.** Most of what a merge would add has no reader on
  the far side. The field-by-field split is below.

#### What each type carries, and what the other has no use for

Only the middle block means the same thing in both — the geometry of the view.
Everything above it answers *"where does this live?"*, everything below it is
*"what has the L2 runtime decided about it?"*, and neither question is open on
the other side of materialization.

| `Tensor` (144 B, wire) | rel | `ChipTensor` (128 B, device) |
| ---------------------- | --- | ---------------------------- |
| `handle.magic` | ⟂ | — |
| `handle.identity` (32 B) | ⟂ | — |
| `handle.backend_kind` | ⟂ | — |
| `handle.body[32]` + `body_len` | ⟂ | — |
| `handle.nbytes` | ≈ | `buffer.size` |
| `handle.access` | ⟂ | — |
| `handle.owner_worker_path_id` | ⟂ | — |
| — | ⟂ | `buffer.addr` |
| `byte_offset` (bytes) | ≈ | `start_offset` (elements) |
| `shapes[5]` / `strides[5]` / `ndims` / `dtype` | = | `shapes[5]` / `strides[5]` / `ndims` / `dtype` |
| `handle.address_space` | = | `address_space` |
| — | ⟂ | `owner_task_id` |
| — | ⟂ | `version` |
| — | ⟂ | `manual_dep` |
| — | ⟂ | `is_contiguous`, `extent_elem_cache` |

**Dead on the device** (`Tensor`-only): `magic` discriminates untrusted bytes at
a decode boundary the device does not have. `identity` / `backend_kind` / `body`
/ `body_len` are the *recipe* for finding the backing — spent once at
materialization, never consulted again. `access` has no device enforcement point
(it is checked at submit). `owner_worker_path_id` is diagnostic.

**Meaningless before materialization** (`ChipTensor`-only): `buffer.addr` is the
resolved address, which by definition does not exist while the argument is still
crossing processes. `owner_task_id` / `version` / `manual_dep` are L2 OverlapMap
state — the producing task and its dependency treatment are decided by the chip
runtime, so an L3 builder has nothing to put there (`make_tensor_strided` fills
`PTO2TaskId::invalid()`, `0`, `false`). `is_contiguous` and `extent_elem_cache`
are derived from `shapes`/`strides`, cached for the AICore hot path so it can
skip cache line 2.

So a merged struct would carry ~70 B that the AICore never reads, in a type whose
size is pinned at two cache lines precisely because the scheduler walks it per
task.

> The one row with a plausible future device reader is `identity`: keying the L2
> OverlapMap by it rather than by `buffer.addr` would make two views of one
> backing bucket together by construction. That needs 32 B — which fits the
> existing `_pad_cl2[36]` at `sizeof == 128`, i.e. **without** merging anything
> else. If it is ever done, the H2D staging step must mint a *new* identity for
> each staged copy, because the device buffer is a distinct backing from the host
> one it was copied from.

### Rejected: keep the wire type transport-only, use `ChipTensor` in the L3 orch

This keeps one user-visible type but pays a conversion at every hop: the orch
builds a `ChipTensor`, the mailbox needs the self-describing form, and the
consumer needs a `ChipTensor` again. Each conversion has to **rewrite the
address** — and the sender cannot know the receiver's, so the receiver ends up
guessing. That is the pre-P1-B mechanism: `_rewrite_blob_host_addrs` rewrote
addresses by **numeric range** and mis-rewrote device pointers that happened to
fall inside a registered host range, patched by adding a `child_memory` skip
whose own comment records the hazard.

### Shipped instead: same name, same interface

The Python type is named `Tensor` and carries the C++ type's vocabulary —
`shapes` / `strides` / `dtype` / `ndims`, with `handle.tensor(...)` and
`args.add_tensor(...)` mirroring `Tensor.make(...)` / `add_tensor(...)`. The C++
binding is exported as `ChipTensor`.

**So the split costs the user nothing to know.** You name a `Tensor`, submit it,
and the chip's C++ orchestration receives it resolved; the type name does not
even appear in your code — you write `handle.tensor(shapes, dtype)` and
`args.add_tensor(t, tag)`. Resolving the address in between is the framework's
job, and keeping the two forms as separate types is what makes that boundary a
type change rather than a silently wrong address.

## Allocating buffers

```python
h  = worker.create_buffer(nbytes)                 # kind3: explicit shared host buffer (POSIX shm)
h  = worker.alloc_shared_tensor((M, N), dtype)    # kind3: shape-sized create_buffer
# inside an orch fn, for device (chip-private) memory:
d  = orch.alloc_child_tensor(worker, (M, N), dtype)  # kind4: DEVICE_MALLOC on that chip
```

`create_buffer` returns a `BufferHandle` backed by a POSIX shm the consumer maps
lazily on first receipt of a tensor over it (map-once, by identity);
`alloc_shared_tensor` returns a runtime-managed intermediate over a FORK_SHM ring
VA. `alloc_child_tensor` allocates device memory on a specific next-level worker
and wraps the pointer; its `.base` is the device pointer (the `orch.copy_to`
destination), and a tensor over it must be dispatched only to that worker.

## Naming a view

`handle.tensor(...)` names a view over the backing:

```python
v = h.tensor(shapes=(M, N), dtype)                        # contiguous (row-major strides)
v = h.tensor(shapes=(N, M), dtype, strides=(1, M))        # transposed
v = h.tensor(shapes=(M, K), dtype, byte_offset=off)       # sub-region
```

`strides` are **element** strides and are strictly > 0 — broadcast (stride 0) and
negative step are unsupported, and a singleton dimension's stride is never
normalized away. `byte_offset` is a **byte** offset and must be a multiple of the
dtype size.

## Submitting a task

```python
ta = TaskArgs()
ta.add_tensor(a_h.tensor((SIZE,), DataType.FLOAT32), TensorArgType.INPUT)
ta.add_tensor(out_h.tensor((SIZE,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
orch.submit_next_level(chip_handle, ta, cfg, worker=0)
```

`TaskArgs` carries `Tensor`s. Tags drive dependency inference, which keys on
the **canonical identity** (buffer granularity, the successor of the former
buffer-address key); byte-range overlap between same-buffer views is refined by
the L2 OverlapMap on the materialized tensors, not by the L3 key.

## Reading and writing data — torch only at the boundaries

The orch fn is a pure DAG builder: computing on data there would be invisible to
dependency inference. So **torch is used only outside `run()`** (fill inputs,
read outputs) or **inside a Python sub-worker** (a compute leaf):

```python
h = worker.create_buffer(n * 4)
torch.frombuffer(h.shm.buf, dtype=torch.float32, count=n).fill_(5.0)  # before run()
worker.run(my_orch, ...)                                             # orch names tensors only
result = torch.frombuffer(out_h.shm.buf, dtype=torch.float32, count=n)  # after run()
```

## How a tensor reaches its consumer (three-way split)

A `Tensor` on the wire is materialized differently by each consumer:

| Consumer | What it does |
| -------- | ------------ |
| **Chip leaf (L2 runtime)** | Materialize each one to a `ChipTensor` (map-once, keyed by identity), including **strided** views; hand the POD blob to `run_from_blob`. |
| **Python sub-worker** (compute) | Map each one into a `MappedArg`; the callable computes with `torch.frombuffer(arg.buffer, ...)`. No `ChipTensor`. |
| **Nested L4→L3 orch** (forwarding) | **Re-export** each backing to a handle `H'` that keeps the source's canonical identity — no pass-through, no map on the forwarding hop. |

**Re-export (no pass-through).** An upper-level tensor is forwarded on receipt as a
handle `H'` that keeps the source's **canonical identity unchanged** (invariant
across every edge — an L4 buffer forwarded L4→L3→L2 carries one identity at all
three layers; only role / materialized VA / view change), per-backing and without
mapping. A downstream compute leaf maps lazily, so pure forwarding carries no map
cost. Dependency inference keys on the invariant identity, so an alias /
retain-release does not split across layers.

## Canonical identity

Every backing carries a fixed-length 32-byte identity — an opaque per-incarnation
`owner_instance_id`, a `buffer_id` unique within that incarnation, and a `generation` that starts at
1 and increments whenever a `buffer_id` slot is reused (so a stale handle for a recycled slot is
rejected rather than silently resolving). It is **invariant across every edge**: an L4 buffer
forwarded L4→L3→L2 carries one identity at all three layers.

Identity is what dependency inference and the map-once import cache key on — never a materialized
address, which means nothing different in another process.

Nothing inside the identity bounds a read: it has no length field. Hashing and comparison are
therefore in-bounds for any bytes that arrive, structurally rather than by validation. The owning
worker's tree path is deliberately **not** part of it — a path is reused across restarts and
contributes no uniqueness, so it is interned to a diagnostic `owner_worker_path_id` whose table lives
only in the owning process. An id another process minted renders as `<path#N>`; nothing routes,
gates, or keys on it.

## Backends

| Backend | Materializes to | Used for |
| ------- | --------------- | -------- |
| `POSIX_SHM` | a named shm mapped into the consumer | `create_buffer` / `alloc_shared_tensor` |
| `FORK_SHM` | the same VA (copy-on-write inherited), no map | a pre-fork host buffer, zero-copy |
| `DEVICE_MALLOC` | the device pointer, no map (chip-local) | `alloc_child_tensor` |
| `REMOTE_SIDECAR` | (P2) resolved via the remote transport | an arg to a remote L3; the descriptor rides in the sidecar |

## Scope / status

Single-machine (host + device) L3→L2 and L4→L3→L2 dispatch is implemented and
verified in `a2a3sim` and onboard `a2a3`. The remote **receive** side and the
buffer lifecycle robustness (`release_buffer`, in-flight retain / deferred-free)
are later phases (P2).
