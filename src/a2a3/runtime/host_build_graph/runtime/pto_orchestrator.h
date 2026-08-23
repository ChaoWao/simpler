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
 * PTO Runtime2 - Orchestrator Interface
 *
 * The Orchestrator is responsible for:
 * 1. Executing the orchestration function (Turing-complete control flow)
 * 2. Allocating intermediate buffers from the heap
 * 3. Submitting tasks via async InCore function calls
 * 4. Building the dependency graph using TensorMap
 * 5. Managing buffer scopes for lifecycle control
 *
 * The Orchestrator can run on either:
 * - Host CPU (lower latency for complex control, easier debugging)
 * - Device AI_CPU (lower latency for task submission)
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <memory>

#include "common/chip_swimlane_profiling.h"
#include "pto_ring_buffer.h"
#include "graph_cache.h"
#include "pto_runtime2_types.h"
#include "pto_submit_types.h"
#include "scheduler/pto_scheduler.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"

struct GraphHostState;

// =============================================================================
// Orchestrator State
// =============================================================================

/**
 * Orchestrator state structure (private to Orchestrator)
 *
 * Contains all state needed for task graph construction and buffer management.
 *
 * host_build_graph runs the orchestrator on the host and ships the shared-memory
 * image it produces, so this whole object is host-only: it owns its scratch
 * arrays outright and no device code reads any of them. PTO2Runtime therefore
 * holds it by pointer — a by-value member would put non-trivially-copyable state
 * inside the struct bind copies to the device.
 */
struct PTO2OrchestratorState {
    // === SHARED MEMORY ACCESS ===
    PTO2SharedMemoryHeader *sm_header;

    // === RING RESOURCES (single ring) ===
    PTO2RingSet ring;
    std::unique_ptr<uint32_t[]> fanin_seen_epoch;
    uint32_t fanin_seen_current_epoch{1};

    // === TENSOR MAP (Private) ===
    PTO2TensorMap tensor_map;  // Producer lookup

    // === SCOPE STACK (Private) ===
    // Single contiguous buffer of task IDs, partitioned by scope level.
    // scope_begins[i] is the index into scope_tasks where scope i starts.
    // Tasks for the top scope occupy [scope_begins[top], scope_tasks_size).
    std::unique_ptr<PTO2TaskSlotState *[]> scope_tasks;  // Flat buffer of taskSlotState (all scopes concatenated)
    int32_t scope_tasks_size{0};                         // Number of task IDs currently in the buffer
    int32_t scope_tasks_capacity{0};                     // Allocated capacity of scope_tasks
    std::unique_ptr<int32_t[]> scope_begins;             // scope_begins[i] = start index of scope i in scope_tasks
    int32_t scope_stack_top{-1};                         // Current top of stack (-1 = no scope open)
    uint64_t scope_stack_capacity{0};                    // Max nesting depth (PTO2_MAX_SCOPE_DEPTH)
    int32_t manual_begin_depth{PTO2_MAX_SCOPE_DEPTH};

    // === SCHEDULER REFERENCE ===
    // Note: In simulated mode, orchestrator and scheduler share address space
    // In real mode, they communicate via shared memory only
    PTO2SchedulerState *scheduler;  // For simulated mode only

    // Total core counts set once at executor init; used for submit-time deadlock detection.
    int32_t total_cluster_count{0};  // AIC cores = MIX clusters
    int32_t total_aiv_count{0};      // AIV cores (= 2 × clusters on standard hardware)

    // === GM HEAP (for output buffers) ===
    void *gm_heap_base;     // Base address of GM heap
    uint64_t gm_heap_size;  // Total size of GM heap (all rings)

    // === FATAL ERROR ===
    // Fatal error flag (single-thread access by orchestrator, no atomic needed)
    // Cross-thread notification uses shared memory orch_error_code (atomic)
    bool fatal;

    // Hidden alloc tasks complete synchronously inside the orchestrator and
    // therefore bypass the executor's normal worker-completion counter path.
    // rt_orchestration_done publishes this into PTO2Runtime, which is the copy
    // the device-side executor adds into its completed_tasks_ progress counter
    // so shutdown/profiling totals remain closed.
    int64_t inline_completed_tasks{0};

    // Host-only, like everything else here.
    GraphHostState *graph_host_state{nullptr};

    // === ARGUMENT POOLS (host-only) ===
    // The mirror's three argument pools and the next free element in each. A submit
    // binds its region at the cursor and advances it by what the task uses, so the
    // pools stay packed and the bind path reads the cursors as the image's pool
    // extents. Nothing is ever returned, and the pools are dimensioned for the worst
    // case (task_window tasks each at their full cap), so a bump cannot overflow.
    // The bases live here, not in the ring header: nothing on the device resolves one.
    //
    // The fanin cursor does not advance at bind time. PTO2FaninBuilder appends and
    // dedups producers afterwards, so the region's length is known only when the count
    // is published, and it advances there.
    int32_t *fanin_pool{nullptr};
    ChipTensor *tensor_pool{nullptr};
    uint64_t *scalar_pool{nullptr};
    int32_t fanin_pool_cursor{0};
    int32_t tensor_pool_cursor{0};
    int32_t scalar_pool_cursor{0};

    // === STATISTICS ===
#if SIMPLER_DFX
    int64_t tasks_submitted;
    int64_t buffers_allocated;
    int64_t bytes_allocated;

#endif

    bool in_manual_scope() const { return scope_stack_top >= manual_begin_depth; }

    // === Cold-path API (defined in pto_orchestrator.cpp) ===

    // Allocate the scratch arrays (fanin epoch table, scope arrays, tensor map)
    // and bind this orchestrator to one shared-memory mirror, GM heap and
    // scheduler. sm_base is the base of the mirror this orchestrator writes; it
    // is dereferenced, so a host-orch pass passes its host mirror rather than a
    // device address. task_window_size must be a power of two.
    //
    // Returns false when an allocation fails; the caller then has no hazard map
    // and must not orchestrate.
    bool
    init(void *sm_base, void *gm_heap, uint64_t heap_size, uint64_t task_window_size, PTO2SchedulerState *scheduler);

    void set_scheduler(PTO2SchedulerState *scheduler);
    void report_fatal(int32_t error_code, const char *func, const char *fmt, ...);
    void begin_scope(PTO2ScopeMode mode = PTO2ScopeMode::AUTO);
    void end_scope();
    TaskOutputTensors submit_task(const MixedKernels &mixed_kernels, const CoreTaskArgs &args);
    TaskOutputTensors submit_dummy_task(const CoreTaskArgs &args);
    TaskOutputTensors alloc_tensors(const CoreTaskArgs &args);
    GraphScopeResult graph_begin(uint64_t graph_key, const GraphTaskArgs &args, uint64_t callable_hash);
    bool graph_prepare(void *recording_handle, const GraphTaskArgs &args);
    void graph_abort(void *recording_handle);
    bool graph_end();
    void graph_commit();
    void mark_done();
};

// =============================================================================
// Orchestrator Profiling Data
// =============================================================================

#if SIMPLER_ORCH_PROFILING
struct PTO2OrchProfilingData {
    uint64_t alloc_cycle;  // Combined task slot + heap allocation
    uint64_t args_cycle;
    uint64_t lookup_cycle;
    uint64_t insert_cycle;
    uint64_t fanin_cycle;
    uint64_t scope_end_cycle;
    int64_t submit_count;
    // Wait time tracking for blocking phases
    uint64_t fanin_wait_cycle;  // Legacy (wiring): fanout_lock wait; polling has no such lock
    // Atomic operation counts per phase
    uint64_t args_atomic_count;
    uint64_t scope_end_atomic_count;
};

PTO2OrchProfilingData orchestrator_get_profiling();
#endif
