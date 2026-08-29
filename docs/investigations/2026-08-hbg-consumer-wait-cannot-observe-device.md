# hbg: `rt_set_tensor_data`'s consumer-wait cannot observe device progress

**Date**: 2026-08-28
**Verdict**: **not fixed** — identified while retiring host-orchestration leftovers
(#2068), left alone because the fix changes `rt_get_tensor_data` /
`rt_set_tensor_data` semantics and first needs a verdict on which behavior is
intended.

## The defect

`wait_for_tensor_ready()` in `src/common/host_build_graph/shared/runtime_core.cpp`
has two halves. Under `host_build_graph` they have different fates, and the
difference is not what a reader would guess.

`rt_set_tensor_data` calls it with `wait_for_consumers = true`, which spins on:

```cpp
SharedMemoryTaskHeader &cons_tasks = orch.sm_header->tasks;
while (cons_tasks.completed_watermark.load(std::memory_order_acquire) < slot.last_consumer_local_id) {
```

`orch.sm_header` is the **host mirror**. Three facts settle what that loop does:

| Fact | Where |
| ---- | ----- |
| The mirror's watermark is initialized to `-1` | `shared/shared_memory.cpp` (`init_header`) |
| `update_completed_watermark()` has exactly one caller, on the device | `{arch}/runtime/host_build_graph/runtime/scheduler/scheduler.h` (`on_mixed_task_complete`) |
| `last_consumer_local_id` is seeded to the task's **own** local id at submit, i.e. `>= 0` | `shared/orchestrator.cpp` (`prepare_task`, and the Graph outer shell) |

The device advances the watermark in *its* copy of the image; nothing writes the
host's after `init_header`. So the comparison is `-1 < own_id`, which is
**unconditionally true**, and the wait can only end at
`TENSOR_DATA_TIMEOUT_MS` = 15 s with `SIMPLER_ERROR_TENSOR_WAIT_TIMEOUT`.

**There is no subset that works.** In particular a producer completed inline on
the host is *not* an exception: the seed is its own id rather than `-1`, so a
producer with no consumers at all still fails the comparison. (An earlier draft
of #2068's description claimed that exception; it is wrong, and this entry
exists partly to keep that claim from being believed.)

The producer half is a different story. `rt_get_tensor_data` calls with
`wait_for_consumers = false`, which spins on `slot.task_state` instead — and
`alloc_tensors` does call `mark_completed()` on the host for a hidden-alloc task.
So **that** half does have a working case: a producer completed inline during
orchestration is observable, and only a producer left to the device is not.

## Why it looks intentional

The function's own comment already says the timeout is not a synchronization
mechanism:

> The host builds the complete graph before device scheduling starts, so a live
> device producer cannot complete during this call; the timeout remains a
> defensive failure backstop rather than a synchronization mechanism for
> orchestration code.

And `ChipTaskSlotState::last_consumer_local_id`'s comment calls itself
"inert-but-scaffolded for parity". Both readings are consistent with "hbg
orchestration is a bind-time pass, so nothing it waits on can be in flight".

What that reasoning does not cover is the seeded value. If the wait is meant to be
a no-op whenever no consumer is live, the mirror-side comparison should be
vacuous — and it would be, had the seed stayed `-1`. Seeding it to the task's own
id makes the loop always spin, which turns a "defensive backstop" into a
guaranteed 15 s stall for any caller that reaches it.

## Why it was not fixed in #2068

Everything else in that PR could be settled by grep and reachability: a field with
no writer, a branch with no reachable condition, a name for an event that cannot
happen. This one cannot. Two readings both fit the code, and they imply opposite
fixes:

1. **The degradation is intended.** hbg orchestration cannot observe device
   progress by construction, so `rt_set_tensor_data`'s consumer-wait is
   meaningless there and should be removed — along with the
   `last_consumer_local_id` maintenance that only feeds it — rather than made to
   work. `rt_set_tensor_data` then documents that it must not be called against a
   tensor whose consumers are device-side.
2. **A read is missing.** The wait is supposed to work, and the host should be
   reading the device's watermark (a D2H of the header field, or a mapped view)
   instead of its own mirror.

Choosing wrongly is worse than leaving it: (1) deletes a guard that some caller
may be relying on to fail loudly, and (2) adds a D2H to a path that currently
touches no device memory, on a runtime whose bind latency is actively being
optimized.

## What a fix needs first

- **Which callers reach the consumer-wait at all.** `rt_set_tensor_data` is
  orchestration-facing API, so the callers are outside this repo. Nothing in
  `examples/` or `tests/st/` exercises it against a device-side consumer today —
  which is also why the 15 s stall has never been reported.
- **Whether the same reading applies to the producer half.** It has a working case
  (inline-completed producers), so its timeout is not vacuous in the same way, and
  the two halves may not want the same treatment.

## Related

- [`2026-08-host-orch-phase-tail-is-page-faults.md`](2026-08-host-orch-phase-tail-is-page-faults.md)
  — why adding a D2H to the bind path is not free in the way it looks.
