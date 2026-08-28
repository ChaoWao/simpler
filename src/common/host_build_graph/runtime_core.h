/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * Runtime core interface
 *
 * This is the main header for the runtime.
 * It provides a unified API for task graph construction and execution.
 *
 * Key Features:
 * - Bump-allocated task table and graph heap (zero allocation overhead)
 * - Lazy invalidation TensorMap for dependency discovery
 * - Manual scopes that bypass TensorMap discovery for explicit dependencies
 * - Per-task spinlocks for concurrent fanout updates
 * - Orchestrator-Scheduler decoupling via shared memory
 *
 * Usage:
 *   1. Create runtime: RuntimeContext create methods
 *   2. Build task graph in orchestration function:
 *      - begin_scope() / end_scope()
 *      - submit_task()
 *   3. Mark orchestration complete: mark_done()
 *   4. Destroy runtime
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <type_traits>

#include "utils/device_arena.h"
#include "host_build_graph/runtime_types.h"
#include "graph_cache.h"
#include "host_build_graph/submit_types.h"
#include "host_build_graph/shared_memory.h"
#include "host_build_graph/task_allocator.h"
#include "host_build_graph/tensormap.h"
#include "scheduler/scheduler.h"
#include "host_build_graph/orchestrator.h"
#include "aicore_completion_mailbox.h"

// =============================================================================
// Runtime Context
// =============================================================================

/**
 * Runtime execution mode
 */
enum RuntimeMode {
    MODE_EXECUTE = 0,    // Execute tasks on workers
    MODE_SIMULATE = 1,   // Simulate task execution with cycle counting
    MODE_GRAPH_ONLY = 2  // Build graph only, no execution
};

/**
 * Function-pointer ops table for runtime operations.
 *
 * The orchestration .so calls runtime functions through this table
 * (via orchestration_api.h inline wrappers), so it has zero link
 * dependencies on runtime .cpp files.
 */
typedef struct RuntimeContext RuntimeContext;  // forward declare for ops signatures
class HostTensorAccessor;

struct RuntimeOps {
    TaskOutputTensors (*submit_task)(RuntimeContext *rt, const MixedKernels &mixed_kernels, const CoreTaskArgs &args);
    void (*scope_begin)(RuntimeContext *rt);
    void (*scope_end)(RuntimeContext *rt);
    void (*orchestration_done)(RuntimeContext *rt);
    bool (*is_fatal)(RuntimeContext *rt);
    void (*report_fatal)(RuntimeContext *rt, int32_t error_code, const char *func, const char *fmt, ...);

    // Logging (populated by runtime, called by orchestration)
    void (*log_error)(const char *func, const char *fmt, ...);
    void (*log_warn)(const char *func, const char *fmt, ...);
    void (*log_timing)(const char *func, const char *fmt, ...);
    void (*log_info)(const char *func, const char *fmt, ...);
    void (*log_debug)(const char *func, const char *fmt, ...);

    // Cross-layer data access (orchestration reads/writes tensor values via runtime)
    // Placed after logging to avoid shifting hot-path field offsets.
    uint64_t (*get_tensor_data)(
        RuntimeContext *rt, const simpler::hbg::Tensor &tensor, uint32_t ndims, const uint32_t indices[]
    );
    void (*set_tensor_data)(
        RuntimeContext *rt, const simpler::hbg::Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value
    );
    TaskOutputTensors (*alloc_tensors)(RuntimeContext *rt, const CoreTaskArgs &args);
    TaskOutputTensors (*submit_dummy_task)(RuntimeContext *rt, const CoreTaskArgs &args);

    // This-run core geometry latched by the host bind: MIX clusters
    // (one AIC each) and standalone AIV cores.
    int32_t (*available_cluster_count)(RuntimeContext *rt);
    int32_t (*available_aiv_count)(RuntimeContext *rt);
    GraphScopeResult (*graph_begin)(RuntimeContext *rt, uint64_t graph_key, const GraphTaskArgs &args);
    bool (*graph_prepare)(RuntimeContext *rt, void *recording_handle, const GraphTaskArgs &args);
    void (*graph_abort)(RuntimeContext *rt, void *recording_handle);
    bool (*graph_end)(RuntimeContext *rt);
    void (*graph_commit)(RuntimeContext *rt);
    // Record one orchestration-side phase on the calling thread. The submission
    // segments this carries are measured in the orchestration .so, which reaches the
    // runtime only through this table. Always present in the struct so the layout does
    // not move with SIMPLER_DFX; nullptr when off.
    void (*record_orch_phase)(uint32_t kind, uint64_t start_ns, uint64_t end_ns, uint64_t detail);
    // Queue one Graph body for asynchronous recording, and drain every queued one.
    // `job` is a `std::function<void(GraphTaskArgs &)> *` the pool moves out of --
    // whether or not it queues it, since start() takes the callable before it checks
    // capacity -- so the caller must not invoke it afterwards. Nothing is owned across
    // the boundary either way: the caller's std::function destructs normally, empty or
    // not, and rt_graph_submit's fallback re-runs its own copy of the body. The pool is
    // runtime-owned (host/graph_recorder_pool.h) and the AICPU build links a refusing
    // fallback, which is what makes the device path record synchronously.
    bool (*graph_record_start)(RuntimeContext *rt, const GraphTaskArgs &args, void *job);
    void (*graph_record_wait)(RuntimeContext *rt);
};

/**
 * Layout descriptor for the prebuilt runtime arena. Holds all sub-region
 * offsets (scheduler / sm_handle wrapper / runtime header / AICore mailbox)
 * plus the layout-defining capacities. Produced once on the host by
 * runtime_reserve_layout(); consumed by runtime_init_data_from_layout and
 * runtime_wire_arena_pointers.
 */
struct RuntimeArenaLayout {
    size_t off_sm_handle{0};
    SchedulerLayout sched;
    size_t off_scheduler{0};
    size_t off_runtime{0};
    size_t off_mailbox{0};
    // Every region in this arena is device-resident. The arena is reserved in two
    // zones, in this order, and the offsets above land in exactly one of them:
    //
    //   device-only  sm_handle, the AICore mailbox, the scheduler state and its
    //                queue slot arrays. Reachable storage whose content is a
    //                function of the layout, not of the run, so the device writes
    //                it at boot instead of receiving an initialization pattern
    //                over PCIe.
    //   copied       [off_copied_begin, off_copied_end). Every byte carries this
    //                run's own content, so bind ships the whole zone as one
    //                contiguous range.
    //
    // Past off_copied_end the device carries one further tail the host arena does
    // not: the shared-memory image, whose size is the submitted task count and
    // therefore not known when this layout is built. bind grows the device region
    // to cover it once orchestration ends.
    //
    // The copied zone is padded to a CHIP_ALIGN_SIZE boundary, so that tail begins
    // exactly at off_copied_end: the copied zone and the shared-memory image are
    // adjacent on the device and travel as one copy.
    //
    // The orchestrator is NOT here. It runs on the host, owns its own scratch, and
    // no device code reads any of it — see OrchestratorState.
    size_t off_copied_begin{0};
    size_t off_copied_end{0};

    // The task-table slot count this image was built for, resolved per bind from
    // runtime_env.ring_task_window. The device needs it to bound the pitch it
    // attaches with.
    uint64_t task_capacity{0};

    // Total arena byte size post-commit. Used by host to size the prebuilt
    // image buffer and as the rtMemcpy length, and requested of the device as
    // the region length before its shared-memory tail.
    size_t arena_size{0};
};

/**
 * Runtime context
 *
 * Contains all state for orchestration and scheduling.
 * In simulated mode, runs in single process with shared address space.
 */
struct RuntimeContext {
    // Ops table (first field — used by orchestration .so via function pointers)
    const RuntimeOps *ops;
    ScopeMode pending_scope_mode;

    // Components
    SharedMemoryHandle *sm_handle;
    // Host-only, and by pointer so that this header stays trivially copyable:
    // the orchestrator runs on the host and owns non-trivial scratch. Null on the
    // device — bind drops it before the copied zone is uploaded, so no device code
    // may dereference it.
    OrchestratorState *orchestrator;
    // Device-only zone: the scheduler state holds no per-run content, so it is
    // addressed through the arena rather than carried inside this header.
    SchedulerState *scheduler;
    AICoreCompletionMailbox *aicore_mailbox;

    // Mode
    RuntimeMode mode;

    // Statistics
    int64_t total_cycles;
    // Hidden alloc tasks the host orchestrator completed inline, published here by
    // rt_orchestration_done. The device-side executor folds this into its
    // completed_tasks_ progress counter so shutdown/profiling totals stay closed.
    int64_t inline_completed_tasks;
    // Graph definitions are process-local host cache entries. The callable
    // identity prevents two orchestration DSOs from sharing the same key.
    uint64_t active_callable_hash;

    // Host views of the tensors this run staged, owned by the run that
    // registered them. Null on the AICPU path, which loads device addresses
    // directly; get_tensor_data / set_tensor_data then fail closed rather than
    // dereferencing one. Lives past the first two fields, so the orchestration
    // .so's partial RuntimeContext definition neither sees nor needs it.
    HostTensorAccessor *tensor_access;

    // Prebuilt-arena fast path metadata. Carries every offset
    // wire_arena_pointers needs at AICPU boot so the AICPU can reconstruct
    // all arena-internal pointer fields without re-running init_data. The
    // device base of the runtime arena travels separately on the host-side
    // Runtime (Runtime::prebuilt_arena_base_), since the AICPU needs it
    // *before* dereferencing this image. Populated on host by
    // runtime_init_data_from_layout + runtime_wire_arena_pointers; read by
    // aicpu_executor.cpp.
    RuntimeArenaLayout prebuilt_layout;
};

// bind copies this header to the device as one contiguous range, so every byte
// of it has to survive a memcpy with no fix-up. That rules out an owning or
// otherwise non-trivial member — the orchestrator's scratch is reached through
// a pointer for exactly this reason.
static_assert(
    std::is_trivially_copyable_v<RuntimeContext> && std::is_standard_layout_v<RuntimeContext>,
    "RuntimeContext is copied to the device verbatim"
);

// =============================================================================
// Runtime Lifecycle API
// =============================================================================

/**
 * Phase 1 — declare every sub-region (sm_handle wrapper, scheduler / mailbox /
 * RuntimeContext header) on the supplied arena. Pure arithmetic; does not touch
 * device memory and may run on host. Returns the layout descriptor; caller
 * commits/attaches the arena before Phase 2/3.
 */
RuntimeArenaLayout runtime_reserve_layout(DeviceArena &arena, uint64_t task_capacity);

/**
 * Phase 2 — write the data half of the runtime arena: standalone fields,
 * memset'd arena regions, sub-structure initializers, and SM-side device
 * pointers. The arena must already be committed (or attached); writes go
 * into arena.base() + sub-region offsets.
 *
 * `sm_dev_base` is a device address; we only store it (never dereference).
 * Safe to run on a host arena that owns a host mirror of the runtime image —
 * the resulting buffer is rtMemcpy-ready.
 *
 * Returns the RuntimeContext* that sits at layout.off_runtime within the arena.
 * Caller must follow up with runtime_wire_arena_pointers; rt->ops and the
 * AICore-side count fields are left untouched and must be filled by the
 * AICPU at boot. Initializes the scheduler only: the orchestrator is a
 * host-owned object the host-orch path (run_host_orchestration) stands up and
 * points rt->orchestrator at, and it is never uploaded to the device.
 */
RuntimeContext *runtime_init_data_from_layout(
    DeviceArena &arena, const RuntimeArenaLayout &layout, RuntimeMode mode, void *sm_dev_base, uint64_t sm_size
);

/**
 * Phase 3 — wire the arena-internal pointer fields that exist on both sides
 * (rt->sm_handle, rt->aicore_mailbox, rt->scheduler and
 * scheduler.{ready_queues, ready_sync_queues, early_dispatch_queues}) so each
 * holds arena.base() + offset. Idempotent — runs on both host (writing
 * host-mirror addresses) and AICPU (writing device addresses) sides.
 */
void runtime_wire_arena_pointers(DeviceArena &arena, const RuntimeArenaLayout &layout, RuntimeContext *rt);

/**
 * AICPU-only Phase 4 — install the ops table, the one field the host could not
 * know at prebuilt-image build time (s_runtime_ops is a device-side file-local
 * global, so the host cannot resolve its device address). Call once per boot
 * after runtime_wire_arena_pointers.
 */
void runtime_bind_ops(RuntimeContext *rt);

// Backing the two graph_record_* ops. Weak fallbacks live in runtime_core.cpp so the
// AICPU build links a refusing start and a no-op wait; the host build overrides both in
// host/graph_recorder_pool.cpp, where the pool is.
bool graph_record_start_impl(RuntimeContext *rt, const GraphTaskArgs &args, void *job);
void graph_record_wait_impl(RuntimeContext *rt);

/**
 * Destroy runtime. With the prebuilt-arena fast path the arena buffer is
 * pooled across runs by DeviceRunner, so we never call arena.release()
 * here — the destructor only forgets sub-structure pointers (idempotent
 * cleanup).
 */
void runtime_destroy(RuntimeContext *rt, DeviceArena &arena);

/**
 * Set execution mode
 */
void runtime_set_mode(RuntimeContext *rt, RuntimeMode mode);

// =============================================================================
// Orchestration API (called by orchestration function)
// =============================================================================

/**
 * Begin a new scope
 *
 * submit_task requires at least one open scope. A MANUAL scope additionally makes
 * every submit inside it bypass TensorMap discovery and take its fanin from
 * CoreTaskArgs::set_dependencies() instead; an AUTO scope nested inside a MANUAL
 * one is rejected. The mode is read from RuntimeContext::pending_scope_mode, which
 * SIMPLER_SCOPE sets immediately before this call.
 */
void rt_scope_begin(RuntimeContext *rt);

/**
 * End current scope
 *
 * Closes the innermost scope, and when that is the outermost MANUAL one, returns
 * later submits to TensorMap discovery. A scope bounds no task or buffer lifetime
 * here: the task table is whole-graph-resident, so no task slot and no heap byte is
 * reclaimed before the run ends.
 */
void rt_scope_end(RuntimeContext *rt);

/**
 * Mark orchestration as complete
 *
 * Signals that no more tasks will be submitted.
 */
void rt_orchestration_done(RuntimeContext *rt);

/**
 * Enter fatal state explicitly from orchestration.
 */
void rt_report_fatal(RuntimeContext *rt, int32_t error_code, const char *func, const char *fmt, ...);

/**
 * Cross-layer data access: read a tensor value by waiting for its producer.
 */
uint64_t
get_tensor_data(RuntimeContext *rt, const simpler::hbg::Tensor &tensor, uint32_t ndims, const uint32_t indices[]);

/**
 * Cross-layer data access: write a value to a tensor at given indices.
 * Waits for producer completion (WAW) and all consumers (WAR) via TensorMap.
 * See set_tensor_data in orchestration_api.h for full documentation.
 */
void set_tensor_data(
    RuntimeContext *rt, const simpler::hbg::Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value
);

/**
 * Slim config struct exported by orchestration .so via aicpu_orchestration_config().
 * Shared definition with orchestration_api.h (same layout, guarded).
 */
#ifndef ORCHESTRATION_CONFIG_DEFINED
#define ORCHESTRATION_CONFIG_DEFINED
struct OrchestrationConfig {
    int expected_arg_count;
};
#endif
