# Problems

Audit of `host_build_graph` for data structures inherited from
`tensormap_and_ringbuffer` whose actual hbg usage diverges from the tmr design
they were shaped for.

Original audit was against `upstream/main@761fdf8d`. **Updated 2026-08-25 against
`upstream/main@fc42cd1dd`**, after the L2-tensor split (#1974) and the ring removal
(#2004). #1980 retired the `PTO2` *identifiers* this audit named, so the spellings
below are the current ones; the brand prose survives in ~100 places, see
`KNOWN_ISSUES.md`.

---

## CLOSED

### 1. Single-ring shape carried as a multi-ring one — FIXED (#1965, #2004)

`PTO2_MAX_RING_DEPTH` was `1` against tmr's `4`, yet the tree carried 38
array declarations dimensioned by it and 21 loops that ran exactly once. #1965
collapsed each operation to one scalar form; #2004 went further and dropped the
ring itself — a task id is now its own table index, and `task_window_mask` /
`get_slot_by_task_id` / `TaskAllocResult::slot` are gone.

### 2. `PTO2RingSet` was a one-member wrapper named "Set" — FIXED (#1965)

### 3. `DEP_POOL_OVERFLOW` reported for fanin exhaustion — FIXED UPSTREAM (#1963)

Renamed to `SIMPLER_ERROR_FANIN_CAPACITY_EXCEEDED`, value 4 held.

### 4. `PTO2_DEP_POOL_SPIN_LIMIT` was a dead define — FIXED (#1965)

### 7. One byte layout shared by three unrelated structs — FIXED (#1974)

`ChipTensor`, `TensorMapEntry` and `TensorCreateInfo` shared a byte-level layout so
three copy paths could each be a single 64-byte `memcpy`, held by 21 hand-written
`offsetof` assertions across four trees. Two of the three were shaped backwards to
satisfy it, and the entry's `memcpy` wrote `ChipTensor::buffer.size` into a
`TensorMapEntry *`. Each struct now assigns its own fields;
`TensorMapEntry::copy_tensor_create_info` is deleted (it had no callers).

### 8. One tensor type served the boundary and both runtimes — FIXED (#1974)

`ChipTensor` carried `owner_task_id` / `version` / `manual_dep` and two derived
caches — what the *runtime* decides about an argument, on the type a *caller* hands
in. `create_from_chip_args` recorded the mismatch as
`debug_assert(!t.manual_dep && t.version == 0)`, and `docs/buffer-abi.md` called the
type "internal" while `ChipWorker.run` took a container of them.

Now: `ChipTensor` is 72 B of geometry plus a resolved address;
`simpler::{hbg,tmr}::Tensor` is 128 B and adds the runtime's own state;
`Runtime::set_orch_args` is the single adoption point and runs on the host in both
runtimes. `sizeof(ChipStorageTaskArgs)` fell from ~33.8 KB to 19464 B.

### 4b. A nonzero producer ring id becomes a ring-0 dependency — FIXED

`task_id.h` declared the encoding as `(ring_id << 32) | local_id` and named the
accessor `ring()`. **That was false for hbg**, which uses the high bits as an
id-space tag: `graph_execution.cpp` minted `TaskId::make(1, synthetic_local)` for a
Graph node. So every hbg guard reading `ring() != 0` looked like a bounds check on
a dimension that no longer exists, when it was really asking "is this a graph-node
id" — and `FaninBuilder::mark_seen` folded that question's answer into its dedup
return value, so `append_fanin_or_fail` read "not a ring task" as "not deduped
yet" and appended an edge to whatever `prod_slot` the foreign local id happened to
mask onto.

The bug survived #2004 unchanged: it moved `mark_seen` to key on the `TaskId` and
dropped the redundant ring/slot parameters, but left the space check folded into
the dedup return value.

`TaskId` is now an opaque 64-bit handle (`raw`, `invalid()`, `is_valid()`, `==`,
`sizeof == 8`); each runtime owns and names its own layout in
`src/common/{host_build_graph,tensormap_and_ringbuffer}/task_id_encoding.h` —
`TaskIdSpace{RING, GRAPH_NODE}` + `make_ring_task` / `make_graph_node` /
`is_ring_task` for hbg, a real `make_task_id(ring, local)` / `task_ring` for tmr.
`mark_seen` deduplicates and nothing else; `append_fanin_or_fail` rejects a
non-RING producer up front via `report_fatal(SIMPLER_ERROR_INVALID_ARGS, …)`.

Repro: `graph_node_dependency` in
`tests/st/host_build_graph_validation/`, verified to fail (`DID NOT RAISE` —
the bogus edge was built silently) with the new guard removed.

---

## STILL OPEN

### 5. `on_scope_end` is an empty stub — low value

`runtime/scheduler/scheduler.h`:
`void on_scope_end(TaskSlotState ** /*task_slot_states*/, int32_t /*count*/) {}`,
still called from `end_scope`, kept as a no-op so the orchestrator call site
matches tmr's. Inline empty function, so readability only — fold into the next
change that touches either file.

### 6. Docs name a field hbg does not have

`docs/RUNTIME_LOGIC.md` states hbg "never advances `last_task_alive`". That
identifier does not exist anywhere in the hbg tree except that sentence.

---

## Deliberate, do not "fix"

- `AsyncWaitList` — live in hbg (`scheduler_dispatch.cpp` reads `.count`).
- `SchedulerState` and its thirteen queue slot arrays — the scheduler runs on the
  AICPU, so the arena/offset form is required.
- `TensorMapEntry`'s two-cache-line mirror of the runtime `Tensor`'s hot fields — a
  shared performance design. Since #1974 it mirrors `simpler::{hbg,tmr}::Tensor`,
  not the boundary type.
- `ring_dep_pool` in the bind ABI — `[[maybe_unused]]` with "kept for ABI
  stability"; a deliberate cross-repo contract decision.
- `TaskTensor` — not a third type. A per-translation-unit alias for the runtime
  being built, so kernels (identical either way, and several compiled under both)
  name no runtime. The alternative was duplicating 13 MB of generated MoE kernels
  into the hbg tree, where the copies would drift.

## Open sizing question (not a vestige)

`TENSORMAP_POOL_SIZE` is `65536` in both runtimes, but tmr spends it per-ring
across 4 rings while hbg uses one ring's worth. Since #1962 that is an explicit
~8 MB allocation per bind. With `GRAPH_MAX_NODES = 1024` and a default task window
of 16384 it is not a hard bound on either side (a capacity check backs it), so
whether hbg wants its own number is a measurable question, not a guess.
