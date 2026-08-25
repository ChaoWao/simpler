# Problems

Audit of `host_build_graph` for data structures inherited from
`tensormap_and_ringbuffer` whose actual hbg usage diverges from the tmr design
they were shaped for.

Original audit was against `upstream/main@761fdf8d`. **Updated 2026-08-24 against
`upstream/main@3069f1aff`**, after which four of the six items are closed. Note
that #1963 renamed most hbg files (`pto_ring_buffer.h` → `ring_buffer.h`,
`pto_orchestrator.h` → `orchestrator.h`, `pto_runtime2.h` → `runtime_core.h`,
`pto_runtime2_init.cpp` → `runtime_init.cpp`, …), so paths below use the new names.

---

## CLOSED

### 1. Single-ring shape carried as a multi-ring one — FIXED (#1965)

`PTO2_MAX_RING_DEPTH` was `1` against tmr's `4`, yet the tree carried 38
`[PTO2_MAX_RING_DEPTH]` array declarations and 21 loops that ran exactly once.
Every shared-memory operation existed twice — a scalar wrapper that filled a
one-element array, and a `_per_ring` implementation reading `[0]` (one of which
cast the array to `void` without reading it). The macro is gone from this runtime;
each operation has one scalar form. `sum_ring_heap_sizes`, which summed one element
and checked it for overflow against itself, is deleted.

Boundaries deliberately held: the `PTO2_RING_*` knobs accept exactly what they did
before, the `RuntimeEnv` / `RUNTIME_ENV_RING_COUNT` ABI is untouched (shared with
tmr; this runtime reads slot 0), `bind_callable_to_runtime_impl`'s `extern "C"`
signature is unchanged, and the `[STALL]` grammar is byte-identical.

### 2. `PTO2RingSet` was a one-member wrapper named "Set" — FIXED (#1965)

Held a single `PTO2TaskAllocator`, with a doc comment claiming
"PTO2_MAX_RING_DEPTH instances exist, one per scope depth". The allocator now sits
on the orchestrator directly.

### 3. `DEP_POOL_OVERFLOW` reported for fanin exhaustion — FIXED UPSTREAM (#1963)

hbg has no dependency spill pool, yet latched a code named after tmr's. Renamed to
`SIMPLER_ERROR_FANIN_CAPACITY_EXCEEDED` — the mechanism both runtimes share, and
the words the log already printed. Value 4 held, so #1960's band contract and the
`-4` an existing caller sees are untouched. Not my change; landed independently
while #1965 was open.

### 4. `PTO2_DEP_POOL_SPIN_LIMIT` was a dead define — FIXED (#1965)

Defined in `ring_buffer.h`, referenced nowhere. Two struct summaries in that
file's header comment also described `FaninPool` / `DepListPool`, which the file
has not held since those pools were removed.

---

## STILL OPEN

### 5. `on_scope_end` is an empty stub — low value

- **Evidence**: `runtime/scheduler/scheduler.h`

  ```cpp
  void on_scope_end(PTO2TaskSlotState ** /*task_slot_states*/, int32_t /*count*/) {}
  ```

  still called from `end_scope` in `orchestrator_core/orchestrator.cpp`; its comment
  says it is kept as a no-op so the orchestrator call site matches tmr's.
- An inline empty function costs nothing at runtime, so this is readability only.
  Worth folding into the next change that touches either file, not worth its own PR.

### 6. Docs name a field hbg does not have

- **Evidence**: `docs/RUNTIME_LOGIC.md` states hbg "never advances
  `last_task_alive`". That identifier does not exist anywhere in the hbg tree
  except that sentence. Pre-existing, one line.

### 4b. A nonzero producer ring id becomes a ring-0 dependency

- **Found by**: CodeRabbit on PR #1965, verified against the code.
- **Evidence**: `PTO2FaninBuilder::mark_seen` returns `false` for a producer whose
  ring id is not 0, and `append_fanin_or_fail` reads `false` as "not deduped yet"
  and appends the producer — with `prod_slot` already resolved against the one
  shared-memory ring. A nonzero-ring producer id with a colliding local id would
  create an edge to an unrelated ring-0 task.
- **Not a regression**: before #1965 the guard read
  `prod_ring >= PTO2_MAX_RING_DEPTH` with that macro at `1`, which for a `uint8_t`
  is the same test as `prod_ring != 0`.
- **Reachability**: `prod_ring` is `producer_task_id.ring()`. hbg builds every task
  id on ring 0, and the Graph recorder already refuses an explicit dep whose
  `ring() != 0`. Reaching this needs a foreign or malformed task id handed to
  `set_dependencies` on the ordinary path.
- **Current state**: #1965 carries a `debug_assert(prod_ring == 0)` so UT and sim
  builds trap a violation, and a comment that no longer implies rejection. The
  fatal path CodeRabbit proposed (report `SIMPLER_ERROR_INVALID_ARGS` and refuse)
  wants a failing repro first per `discipline.md` §3, so it needs its own change.

---

## Already cleaned up by earlier work — not vestiges

Recorded so the next audit does not re-open them.

- `runtime/ring_buffer.h` is ~270 lines against tmr's 816: the reclaiming
  allocator, its 500 ms deadlock backstop and the reclaim bookkeeping are gone,
  with present-tense comments stating that nothing is reclaimed mid-run.
- The dep-pool arena region and the per-ring dep pools are gone; only comments
  explaining their absence remain.
- The orchestrator and its TensorMap are host-owned rather than arena regions
  (#1962), so the arena has two zones, both device-resident.

## Deliberate, do not "fix"

- `AsyncWaitList` — live in hbg (`scheduler_dispatch.cpp` reads
  `async_wait_list.count`).
- `PTO2SchedulerState` and its thirteen queue slot arrays — the scheduler really
  runs on the AICPU, so the arena/offset form is required.
- `PTO2TensorMapEntry`'s 128 B two-cache-line mirror of `ChipTensor` — a shared
  performance design, not a shape inherited by accident.
- `ring_dep_pool` in the bind ABI — `[[maybe_unused]]` with "kept for ABI
  stability"; a deliberate cross-repo contract decision.

## Open sizing question (not a vestige)

`PTO2_TENSORMAP_POOL_SIZE` is `65536` in both runtimes, but tmr spends it
per-ring across 4 rings while hbg uses one ring's worth. Since #1962 that is an
explicit ~8 MB allocation per bind. With `GRAPH_MAX_NODES = 1024` and a default
task window of 16384 it is not a hard bound on either side (a capacity check backs
it), so whether hbg wants its own number is a measurable question, not a guess.
