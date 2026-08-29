# Investigations

Write-ups of work that was done to **answer a question** and where the
answer wasn't a code change that lives in the repo. Specifically:

- Optimizations that were considered, measured, and dropped (no signal /
  not worth the complexity).
- Designs that were prototyped and rejected.
- "I think we should do X" proposals where the analysis ruled out X
  (or scoped X to "later, when Y is true").

The point is to save the **next** person's time. If you find yourself
reaching for the same idea, the entry tells you:

1. It was already considered.
2. What measurement / argument shut it down.
3. Under what conditions it might become worth re-opening.

If you re-investigate anyway, update the entry with the new data —
don't open a parallel doc.

## What does NOT go here

- **Active bugs / unresolved problems** → `KNOWN_ISSUES.md` (local) or a
  GitHub issue.
- **Problems with a known fix or workaround** →
  `docs/troubleshooting/`.
- **Designs that were prototyped and shipped** → the design doc lives
  with the subsystem (e.g. `docs/dfx/<feature>.md`).
- **Architectural decisions that constrain future code** → if/when we
  adopt ADRs, those would live elsewhere; this folder is for things we
  *didn't* do.

## File naming

`YYYY-MM-<short-slug>.md` — e.g. `2026-06-chip-swimlane-defer-wmb.md`.
The date is the month the investigation was done so entries sort
chronologically and stale ones are easy to spot.

## Template

```markdown
# <Title — what was proposed, in one line>

**Date**: YYYY-MM-DD
**Verdict**: dropped / deferred-pending-X / superseded-by-Y

## Question

Brief statement of the proposal. Why it might be a good idea — the
intuition that would make a future contributor reach for the same
change.

## What was tried

Concrete actions. Commands, files touched, measurement setup. Enough
that someone can reproduce the measurement, not enough to retell the
whole codebase.

## Result

The numbers, the diff size, the bug found — whatever the actual output
of the investigation was.

## Why not (now)

The decision. Tie it to a specific signal in the result, not just
preference.

## When to reconsider

The condition under which this becomes worth re-opening. "If workload X
shows >Y µs in profile" / "after Z lands" / "if hardware changes such
that ...".

## References

- PRs, commits, issue links.
- Related rules (`.claude/rules/...`) or docs that informed the
  decision.
```

## Index

Newest first.

- [2026-08 — hbg: `rt_set_tensor_data`'s consumer-wait cannot observe device progress](2026-08-hbg-consumer-wait-cannot-observe-device.md) — **not fixed**, found while retiring host-orchestration leftovers (#2068). `wait_for_tensor_ready(wait_for_consumers=true)` spins on the **host mirror's** `completed_watermark`, which only the device advances — in its own copy — so the comparison is `-1 < last_consumer_local_id`, unconditionally true, and the wait can only end at the 15 s `TENSOR_DATA_TIMEOUT_MS`. **No subset works**: a producer completed inline on the host is not an exception, because the seed is the task's own id rather than `-1`, so even a producer with no consumers stalls (an earlier draft of #2068's description claimed that exception — it is wrong). The producer half is different and does have a working case, since it reads `task_state`, which `alloc_tensors` sets host-side. Left alone because two readings fit the code and imply opposite fixes — delete the wait as meaningless under host orchestration, or add the D2H it is missing — and picking wrong either drops a guard some out-of-repo caller may rely on or puts a device read on the bind path

- [2026-08 — The host-orchestration phase tail is page faults, not the code in the phase](2026-08-host-orch-phase-tail-is-page-faults.md) — root cause of the two shapes on every hbg host swimlane: 447 of the 449 `record_node` (now `record_in_graph_task`) calls above 10 µs took a minor fault, 19% of calls carry 79% of the phase, and a fault costs 14–33 µs here against 1.7 µs off-tree because the process's own `mmap`/`munmap` holds `mmap_lock` against every faulting thread — three 64 MiB unmaps in an off-tree reproducer recreate the whole distribution. The fault count is deterministic (1063/1065/1168 per two orchestrations) and drops to 29 when glibc keeps freed memory; the cost per fault varies 2.4× between runs of the same binary, which is the measurement noise that hid this for fourteen iterations. Refutes THP (`PR_SET_THP_DISABLE` leaves the count unchanged), preemption, and node shape; records why the tunables are not a fix (`args` regresses, and the probe build's −53% is the probe amplifying its own subject). Amended 2026-08-23: recording the Definition image into the retained upload staging (8 × ~126 KB per dsv4 bind, previously a vector per recording) moved `graph_upload`'s faults 38 → 1 per bind but left `host_orch`'s count unresolvable in both directions, because a freed 126 KB block is reused without re-faulting — size against glibc's mmap and trim thresholds, not byte count, decides what shows up in this tail. Amended 2026-08-25: retaining the 82 MB SM mirror on the runner (one buffer per pipeline slot instead of one per bind) removes an `mmap` + `munmap` of that size per bind — `hblkhd` stops returning to its pre-bind value on 6 of 6 binds, in both arms of two interleaved repetitions — and shows that a retained buffer must be handed over **uninitialized**: the first implementation used `std::vector::resize`, whose value-initialization faulted in all 20132 pages of the window on each rank's cold bind (~20k minflt against ~1100) and left the whole 82 MB resident. `host_orch`'s warm-bind fault count resolves in neither direction (base [1160, 1256] over eight binds, retained [181, 1268]), since the mirror is ~6 THP faults of a ~1200-fault bind; an earlier attribution of a warm-bind rise to glibc's dynamic mmap threshold is retracted there. That amendment also closes the entry's "Where a fix would go" list — items 1–3 shipped as #1981, item 4 as #1988 plus #2013, and item 3's flat-region form as #2015 — and records that none of them reached the ~1100 faults the submitting thread takes per bind, which is what is left. Amended again the same day: the claim in that amendment that pre-sizing the recorder's node and tensor storage would only help a cold bind is **retracted** — a slot-creation counter shows warm dsv4 binds still creating 1336 node slots of 1679, because #1981's retention is per thread while the pool hands bodies out through one shared FIFO. Reserving each node's own buffer to the cap makes it exactly one page and is *worse* than main (minflt 1070 → 2540); packing every body's tensors into one never-grown bump region is what helps (`record_node` warm min 1702/3423 → 1239/1563 µs). **Amended 2026-08-25 (last)**: the *count × price* framing this entry opened with is refuted by two arms pointing opposite ways — glibc keeping freed memory removes 86% of the faults and buys **no** time, while #2015 removes 6% and buys 29–43%. The count is not a lever; only the price is, and the in-tree `mmap_lock` writer that sets it is **`mprotect`**, which glibc uses to open a non-main arena (26 calls per bind in that band; those arenas only grow, `MADV_DONTNEED` there is 0). Every earlier strace here traced `madvise`/`mmap`/`munmap`/`brk` and not `mprotect`, which is why the in-tree source of the exclusion went unfound for three rounds. Consequence: the residual ~1100 resident-page re-faults per bind — survived #1981, #1988, #2013, #2015, mechanism undetermined — showed **no measurable latency change** when the tunable arm removed 86% of them, so they are **not currently established as a performance defect**; userspace tools are exhausted (`mincore` and pagemap both report presence, not writability), so pricing them at all needs `bpftrace` on `handle_mm_fault`. **Corrected 2026-08-26**: they are a **warm-up cost and they end** — at `f40cacf30`, `host_orch`'s per-bind `minflt` runs 989, 983 (cold), then 114, 173, 54, 3, 13, 13, 11, 8, and reaches 0 by the sixth bind on three independent runs. Every per-bind count in this entry came from three-to-six-round runs divided by the bind count, so each averaged two cold binds and three or four still-decaying ones under a steady-state label; the quantity being divided was never per-bind. Nothing "survived" the four changes, and the mechanism left undetermined for three rounds turned out not to need determining. Control plane at that commit: 0.529 ms min / 0.570 median over 8 warm binds (`host_orch` 0.341/0.364). The reusable half of this — that dropping the cold bind does not reach the steady state — is now a trap in [hbg-bind-phases.md](../dfx/hbg-bind-phases.md). Also carries the per-site fault table (two of its top three sites were deleted by #2019 and #2015 within days — keep the method, not the numbers) and the three tooling traps that each produced a wrong conclusion first
- [2026-08 — hbg: per-block Graph Definitions and cross-layer reuse](2026-08-hbg-graph-block-decomposition.md) — adopted at seven Definitions covering all 43 layers, after #1929 replaced the single recording slot the first attempt measured against (it demoted a Graph whose key differed from the in-flight recording's: 79 of 82 intended submissions recorded, host tasks rose 1131 → 1486). Now host submissions 1131 → 129, `host_orch` −44% and `sm_h2d` −85%, but `graph_upload` +207% for seven images instead of one, so the control plane nets −17% at the per-phase floor and **nothing at the median** — a predictable cost traded for a lower floor and a 133%-wide spread that depends on seven recording threads getting CPU. Keeps the structural map that made the reuse provable (367 kernels → 169 classes / 132 by code alone; which blocks can share a Definition and why the hash-routed MoE cannot), the two indices and the last layer's different `hc_post` destination, and the arithmetic that a recorded node costs about what a submitted task costs so break-even sits near three occurrences
- [2026-08 — hbg: uploading Graph Definitions once as shared device objects](2026-08-hbg-graph-definition-single-upload.md) — cut the per-replay 130 KB Definition re-serialization (image build 931→24 µs, orch total −54%), and confirmed the H2D stage is latency- not bandwidth-bound. **Amended 2026-08-18**: a `--rounds 3` split shows 12.19 of the residual 12.9 ms is the *one-time* `rtMalloc`+`memset` of 40 execution-storage blocks (~53 MB), not per-call latency — real per-call is ~17 µs, so batching the reference submissions is worth ≤0.6 ms, and 88% of this change's own cold-start gain (−1.879 ms in that split) came from execution storage shrinking rather than from the byte reduction it targeted. **Amended 2026-08-25**: the one-time verify gate this change introduced no longer hashes anything — `content_hash`, `verify_state` and the whole-image zero-fill are removed, since `graph_definition_array` plus `bind_graph_topology` already bound every device-side read, so the three ~1 MB passes per dsv4 bind bought no safety the structural checks did not
- [2026-07 — Why qwen3_14b_decode held a device for 406 s, and the four answers that were wrong](2026-07-qwen-scene-test-406s-decomposition.md) — root cause was torch thread oversubscription in goldens, not any of the big things: the reference walks 3584 tiny slice ops per layer and torch sizes its pool from the core count, so a 320-core host paid 6.35 s/layer against 1.05 s at 4. Capping to 8 (#1601) cut the golden 359 s → ~40 s and let the case rejoin the sweep. Records the measured decomposition (compile 59 s / fixture 13 s / golden 359 s / device tens of ms), how to split queue wait from card-held work via the `npu-lock` timestamps, and why golden caching, vectorisation, a nightly split and `skip_golden` were each dropped
- [2026-07 — `sync_start` drain retry ABA across reusable barrier state](2026-07-sync-start-drain-retry-aba.md) — reproduced #1455 deterministically by suspending an old-attempt scheduler between the ack barrier and election; retained a generation-tagged O(log N) tree with fixed thread-0 coordination after packed-atomic, no-root-broadcast, and rotating-coordinator experiments; removed the invasive runtime hook after hook-free UT/ST coverage was established
- [2026-07 — Host worker dispatch latency: where the remaining ~50 µs goes](2026-07-host-dispatch-latency-budget.md) — measured & dropped: after #1499 a `Worker.run()` costs **43.3 µs fixed + 8.08 µs per task**, so the "94% of latency is our runtime, not the IPC" headline is an artifact of benchmarking a *single* task — at 256 tasks/run the fixed cost is 2%. Also settles #1498 against itself for latency: a pipe wake measures 4.4 µs one way against a 3.5 µs bare fork+shm round trip, so blocking *adds* more than the whole IPC floor to every dispatch; only the CPU-while-idle argument survives
- [2026-07 — Containing A2/A3 SDMA stream teardown after an AICore fault](2026-07-a2a3-sdma-fault-teardown.md) — CANN exposes no remote-channel retirement fence: even device-confirmed stream frees followed by a soft-reset success that did not complete context teardown can add minutes to a later ordinary fault. Contained by making SDMA an explicit per-Worker opt-in (`enable_sdma`) so only opt-in Workers carry the slow teardown and ordinary Workers keep fast recovery; full closure needs CANN runtime + driver changes (#1425)
- [2026-07 — AICore-side arg fill for ALL dispatches (not just the gated path)](2026-07-aicore-fills-all-args-ready-path.md) — measured & rejected for the ready path: making the AICore fill its own `args[]` for every task adds ~1.0 µs to each task's `receive→start` setup (paged_attention_unroll: 349 ns → 1356 ns), because a ready task has no idle doorbell gate to hide the fill. Offload is a win only on `not_ready` (early-dispatch), where the AICore already spins at the gate; shipped design keeps the AICPU filling ready tasks (#1328)
- [2026-07 — chip swimlane AICore: switch-overhead source + FIN-early reorder & ACK-gate](2026-07-aicore-swimlane-switch-overhead-and-ack-gate.md) — measured: the ~0.8 µs inter-task switch is the record write-back `dcci(record,OUT)+dsb` (~0.5 µs) + payload setup (~0.28 µs), inherent and not reducible by moving FIN; the WAIT gap (p99 ~700 µs) dominates decode. Shipped: sample `end_time` after an early FIN, and an AICPU ACK-gate on buffer rotation (release the old buffer only when AICore ACKs the new buffer's first task) to close the FIN-before-record boundary race the reorder introduced
- [2026-07 — Removing LocalReadyBuffer exposed a missing dcci in EP dispatch](2026-07-local-buffer-removal-ep-combine-regression.md) — RESOLVED in #1245: local-buffer removal changed dispatch timing and unmasked a latent kernel bug (dispatch never dcci'd `recv_count_out` to HBM → local_expert read count=0 → all-zero output); fixed with a one-line dcci in the example kernel
- [2026-06 — Gating the two residual profiling enable() calls on the orch/scheduler hot path](2026-06-orch-profiling-enable-gates-hot-path.md) — gated under existing `SIMPLER_DFX`; magnitude unmeasured, no new macro
- [2026-06 — Replacing COND with GM+dcci for AICore→AICPU notification](2026-06-cond-vs-gm-notification.md)
- [2026-06 — Letting AICore directly read or write the SPR MMIO window](2026-06-aicore-mmio-to-spr.md)
- [2026-06 — PA-unroll 207001: an op-timeout-window issue fixed by #1035, not a launch-order bug](2026-06-pa-unroll-207001-optimeout-window.md)
- [2026-06 — Cross-task batched publish: hoist wmb across distinct tasks in one pop](2026-06-cross-task-batched-publish.md) — also carries the root cause + fix for the `spmd_sync_start_stress` 507018 drain-barrier hang
- [2026-06 — AICore first-task cold-start: pre-warm dispatch path](2026-06-aicore-cold-start-warmup.md)
- [2026-06 — a5 AICore op-timeout poisons the shared L2 worker (cascade)](2026-06-a5-aicore-op-timeout-cascade.md)
- [2026-06 — a5 AICPU filter gate: Scenario B fail-fast guard not added](2026-06-a5-aicpu-filter-gate-scenario-b-validation.md)
- [2026-06 — Sanitizer rollout scope: macOS, TSAN gating, LSan](2026-06-sanitizer-scope.md)
- [2026-06 — chip swimlane: defer per-task wmb to rotation](2026-06-chip-swimlane-defer-wmb.md)
