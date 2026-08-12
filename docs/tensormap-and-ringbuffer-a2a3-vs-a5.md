# `tensormap_and_ringbuffer`: A2/A3 vs. A5

This document describes the substantive differences in the current code under
`src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/`.

> **Maintenance baseline:** This document uses PR 1795 as its baseline. Update
> this document whenever the files or constants it describes change.

## Overview of Current Differences

The current differences fall into two categories:

- **Platform divergences that must be retained**: physical topology, cache
  coherence, the PMU collection protocol, system counter and DMB register
  layouts, the CANN launch ABI, A5 compiler-reserved symbols, and the optional
  URMA completion path currently unique to A5.
- **Software differences that can be converged or require further validation**:
  include guards, `block_num()`/`core_num()` naming, A2/A3 next-block
  prefetching, and the different fatal teardown strategies.

| Difference | Root cause | Must remain platform-specific? | Current decision |
| ---------- | ---------- | ------------------------------ | ---------------- |
| Compute topology | Physical hardware | Yes | Retain each platform's capacity constants and derived layouts |
| AICPU launch plan | Product thread limits, CANN launch ABI, and firmware topology | Yes | Retain the A5 dynamic topology query; do not equate thread limits with physical topology |
| Cache coherence | Hardware coherence model | Yes | Retain the required invalidate/flush operations on A2/A3; do not copy unnecessary maintenance operations to A5 |
| PMU collection | Hardware PMU and platform collection protocol | Yes | Retain the different counter counts, readers, and FIN submission paths |
| System counter and DMB | Hardware timing and register layout | Yes | Use the constants for each platform |
| `s_block_*` fields | Symbols reserved by the A5 CCEC compiler | Yes | Retain the platform-specific internal field names while preserving identical getter semantics |
| URMA completion | A5-specific implementation and product capability gate | Yes, for now | Retain the A5 path; do not claim that URMA is available in the default build |
| Include guards | Mechanical difference | No | Converge to `#pragma once` when the corresponding files are next modified |
| `block_num()`/`core_num()` | Historical interface naming | No | Retain both interfaces for now and do not confuse them with physical core counts |
| Next-block prefetch | A2/A3-only performance optimization | No | Retain on A2/A3; validate on A5 before considering a port |
| Fatal teardown | Software reliability strategy | No | Retain the current implementations; decide whether to converge after measuring the worst-case A5 teardown time |

Here, "platform divergences that must be retained" includes not only physical
hardware differences, but also implementation differences required by the
platform ABI, toolchain, and currently platform-specific backends. "Software
differences" means that the hardware and ABI do not require different solutions
on the two platforms; whether to converge immediately still depends on the
benefit and validation cost.

## Platform Divergences That Must Be Retained

### Compute Topology and AICPU Launch Plan

One A2/A3 runtime device corresponds to one die with 24 clusters, comprising
24 AICs and 48 AIVs. An A3 chip exposes its two dies as two device IDs. One A5
runtime device instead spans both dies of the chip and has 36 clusters,
comprising 36 AICs and 72 AIVs. Because the physical core and cluster counts
visible to one runtime device differ, `PLATFORM_MAX_CORES`,
`RUNTIME_MAX_WORKER`, and the capacities of per-core diagnostic buffers must
also differ.

In the current product configuration, A2/A3 uses at most four active AICPU
threads, while A5 uses at most five. `DeviceRuntimeLaunchDesc` retains
`aicpu_launch_count` on both platforms. A5 uses
`simpler_aicpu_query_topology` to determine the launch plan dynamically from
OCCUPY/FG/PG/SMT, whereas A2/A3 does not currently use the same dynamic query
path.

Physical topology, product thread limits, and firmware launch topology are
three related but distinct constraints. The physical core count determines the
runtime capacity ceiling, while the AICPU thread limit and topology query
determine how the current product launches and shards the scheduler. They are
not interchangeable.

| File | Role |
| ---- | ---- |
| `platform/include/common/platform_config.h` | Defines cluster/core counts, the product AICPU thread limit, and platform capacity constants |
| `runtime/runtime.h` | Maps `RUNTIME_MAX_WORKER` to 72/108 platform cores and defines the shared launch descriptor |
| `host/runtime_maker.cpp` | A5 registers and uses the dynamic topology query; A2/A3 does not use this path |

### Cache Coherence

On A2/A3, after Host DMA or SDMA writes to GM, the AICPU must invalidate its
cache before reading the data. On A5, DMA/HBM is coherent with the AICPU, so
these invalidations are unnecessary at the same points. The current SDMA
completion protocol publishes a monotonic completed post ID: both platforms
read it with acquire semantics and neither clears or retires the shared record.
AICore results are still published by the AICore with `dcci`.

This cache maintenance directly guarantees visibility. It cannot be
mechanically removed from A2/A3, nor should it be copied to A5 as unnecessary
overhead.

| File | Current difference |
| ---- | ------------------ |
| `aicpu/aicpu_executor.cpp` | A2/A3 invalidates `runtime->dev`, which Host DMA writes, before teardown; A5 does not require the corresponding operation |
| `runtime/backend/sdma/sdma_completion_scheduler.h` | A2/A3 invalidates the cache line before the acquire load of the completed post ID; A5 performs the acquire load directly; retirement is a no-op on both platforms |
| `runtime/pto_async_wait.h` | A2/A3 provides a cache-line invalidation helper for async COUNTER polling; A5 does not require it |
| `runtime/scheduler/pto_scheduler.h` | A2/A3 invalidates the cache line before async COUNTER polling; A5 polls directly |

### PMU, System Counter, and DMB

A2/A3 exposes eight PMU counters, which the AICPU reads from MMIO after FIN.
A5 exposes ten counters, which the AICore reads with the PTO-ISA `ld_dev`
operation and writes to a per-core staging slot before the AICPU submits the
record after FIN. The counter counts, MMIO readers, and available instructions
differ, so the record layout, FIN sequencing, and per-core ring must diverge.

The A2/A3 system counter runs at 50 MHz and uses DMB MMIO offset `0xA0`; the A5
values are 1 GHz and `0xD0`, respectively. These facts come from each
platform's `platform/include/common/platform_config.h`.
`docs/RUNTIME_LOGIC.md` only documents the corresponding mapping.

| File | Current difference |
| ---- | ------------------ |
| `platform/include/common/platform_config.h` | Defines the system counter frequency and DMB offset for each platform |
| `aicore/aicore_executor.cpp` | Both platforms bracket kernel execution with the PMU gate; A5 additionally writes the ten-counter staging record before FIN |
| `runtime/scheduler/scheduler_completion.cpp` | After FIN, A2/A3 invokes the AICPU MMIO reader for eight counters; A5 commits the ten-counter slot written by the AICore |
| `platform/shared/aicpu/pmu_collector_aicpu.cpp` | Implements the A2/A3 direct MMIO read and the A5 staging-slot consumption paths |

### Symbols Reserved by the A5 Compiler

The A5 CCEC compiler treats `block_idx` and `block_num` as built-in symbols, so
using the original field names causes a compilation conflict. A5 therefore
uses `s_block_idx` and `s_block_num` in `LocalContext` and the dispatch payload,
while A2/A3 continues to use `block_idx` and `block_num`. The getters and logical
semantics are identical on both platforms. This is a toolchain symbol
constraint, not a change to the runtime data model.

| File | Current difference |
| ---- | ------------------ |
| `common/intrinsic.h` | `LocalContext` uses the field names permitted by the corresponding platform |
| `runtime/pto2_dispatch_payload.h` | The dispatch payload follows the platform-specific field layout |
| `runtime/scheduler/scheduler_dispatch.cpp` | Writes the corresponding platform's payload fields |

### Optional A5-Specific URMA Backend

A5 contains the source path for issuing URMA completion requests, creating
deferred entries, forwarding FIN, and polling/retiring CQ entries. A2/A3
currently registers only the COUNTER and SDMA completion backends.

The repository does not currently define `PTO_URMA_SUPPORTED`. A5 therefore
compiles the shared ABI, mailbox, CQ polling/retirement, and related paths, but
the kernel path that successfully issues URMA PTO instructions is unreachable.
The current state is "implemented but disabled by default." It neither means
that the default A5 build supports URMA nor proves that the A2/A3 hardware does
not support URMA.

Both platforms already share `CompletionToken::backend_cookie`,
`ASYNC_ENGINE_URMA`, the 32-byte `DeferredCompletionEntry`, end-to-end cookie
propagation from the AICore slab into the 64-byte mailbox message, and the
generic completion-backend dispatch. The actual platform divergence is that
A2/A3 lacks the URMA completion type, request-issue implementation, registered
backend operations, and CQ polling/retirement path.

| Path stage | File | Current difference |
| ---------- | ---- | ------------------ |
| Request issue | `runtime/backend/urma/urma_completion_kernel.h` | Present only on A5; it invokes `TGET_ASYNC`/`TPUT_ASYNC` only when `PTO_URMA_SUPPORTED` is defined |
| Deferred entry | `runtime/aicore_completion_mailbox_types.h`, `runtime/pto_async_kernel_api.h` | Both platforms use the same 32-byte entry and propagate `backend_cookie`; A5 additionally defines the URMA completion type |
| FIN forwarding | `runtime/scheduler/scheduler_completion.cpp`, `runtime/aicore_completion_mailbox.h` | Both platforms carry `backend_cookie` into the same 64-byte mailbox message; A5 can populate it with URMA workspace metadata |
| CQ polling/retirement | `runtime/backend/urma/urma_completion_scheduler.h`, `runtime/pto_async_wait.h` | A5 registers URMA operations, polls CQE owner/status, advances the CQ/WQ tail, and updates the doorbell; the scheduler header itself is not guarded by the capability macro |

## Software Implementation Differences

### Include Guards

Some equivalent headers use path-based include guards on one platform and
`#pragma once` on the other. This does not change the ABI, concurrency
protocol, or runtime behavior, and there is no reason to retain it as a
long-term platform divergence. No standalone bulk cleanup is planned; each
file should converge to `#pragma once` when it is next modified, following the
repository convention.

The affected files are:

- `runtime/backend/sdma/sdma_completion_kernel.h`
- `runtime/backend/sdma/sdma_completion_scheduler.h`
- `runtime/pto_types.h`

### `block_num()` and `core_num()`

The A2/A3 launch accessor is named `block_num()`, while the A5 accessor is
named `core_num()`. Both represent the number of logical SPMD blocks in a task,
not the number of physical cores. This difference appears in
`runtime/pto_orchestrator.cpp` and `runtime/pto_submit_types.h`.

This is a historical interface naming difference, not a physical topology
difference. It is also distinct from the `s_block_num` payload field required
to avoid the A5 compiler-reserved symbol. Both existing interfaces are retained
for now.

### A2/A3 Next-Block Prefetch

The A2/A3 completion path calls `prefetch_block_dst()` in
`runtime/scheduler/scheduler_completion.cpp`; the corresponding helper is
declared in `runtime/scheduler/scheduler_context.h`. This prefetch reduces the
A2/A3 sync-start drain burst without changing the scheduling protocol.

Prefetch effectiveness depends strongly on platform characteristics such as
cache capacity. The repository does not contain an A5 measurement that
establishes a benefit, so the optimization should not be ported mechanically.
This is a platform-sensitive performance strategy, not a different scheduler
architecture.

### Fatal Teardown

The A2/A3 scheduler uses a dedicated fatal latch to elect an owner, broadcasts
EXIT to all AICores that completed the handshake, and then joins them in
parallel against a single deadline. A5 uses `completed_` to elect the owner and
deinitializes each core sequentially.

The A2/A3 implementation mitigates failure scenarios involving 48 SDMA remote
streams, but the hardware, PTO-ISA, and platform ABI do not mandate this
algorithm. The current A5 sequential deinitialization does not omit the
acknowledgement wait and is therefore not a known correctness defect. The risk
is that each unresponsive core may consume a one-second timeout, so the
aggregate teardown time can reach or exceed the AICPU operation-execution
timeout.

The current implementations are retained. The total teardown time for N
unresponsive cores should first be measured on A5. If a port is needed, the A5
handshake, EXIT MMIO, shared deadline, and core deinitialization semantics must
then be validated. See
[Issue #1710](https://github.com/hw-native-sys/simpler/issues/1710) for tracking.

| File | Current difference |
| ---- | ------------------ |
| `runtime/scheduler/scheduler_cold_path.cpp` | A2/A3 uses a fatal latch, broadcasts EXIT, and uses a shared deadline; A5 uses a `completed_` owner and deinitializes each core sequentially |
| `runtime/scheduler/scheduler_context.h` | A2/A3 declares fatal-owner election and broadcast/join helpers; A5 retains its existing emergency-shutdown interface |

## Files and Comparison Boundaries

### Byte-Identical Runtime Files

The following files at matching paths are byte-identical in the current code:

```text
build_config.py
docs/{MULTI_RING.md,SCALAR_DATA_ACCESS.md,SUBMIT_BY_CLUSTER.md,device_log_profiling.md,
      profiling_levels.md}
host/{dep_gen_replay.cpp,runtime_compile_info.cpp}
orchestration/{common.cpp,pto_arg_with_deps.h,pto_orchestration_api.h}
runtime/{common.h,pto_dep_compute.h,pto_orchestrator.h,pto_ring_buffer.cpp,
         pto_ring_buffer.h,pto_runtime2.cpp,pto_runtime2.h,pto_shared_memory.h,
         pto_tensormap.h,tensor_create_info.h}
runtime/scheduler/{pto_scheduler.cpp,scheduler_types.h}
runtime/shared/{pto_runtime2_init.cpp,pto_shared_memory.cpp,pto_tensormap.cpp,runtime.cpp}
```

This list only indicates that the file contents are identical. It does not
mean that every unlisted file has a substantive platform difference. Textual
differences not included in the evidence mapping above are primarily
mechanical, such as helper placement, comment detail, include ordering, and
blank lines.

### Examples and Tests

`examples/.../tensormap_and_ringbuffer` and
`tests/st/.../tensormap_and_ringbuffer` are outside the runtime implementation
comparison. Examples and tests unique to either platform primarily reflect
chip-feature validation and test-porting progress; they cannot be used to
infer whether the runtime supports a shared algorithm.

For example, A5 has `urma_deferred_completion_demo`, but this does not mean
that the current build defines `PTO_URMA_SUPPORTED`. Likewise, the absence of a
workload on one platform does not automatically mean that the corresponding
runtime capability is unavailable.
