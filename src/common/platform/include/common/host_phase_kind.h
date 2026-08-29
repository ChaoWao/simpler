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
 * The kinds a HostPhaseRecord can be — the vocabulary alone, with no dependency
 * beyond <cstdint>.
 *
 * Separate from chip_swimlane_profiling.h, which owns the record, the pools and the
 * drain machinery, because those need <vector> and the platform config while three
 * translation units need only the kinds:
 *
 *   - the host trace and record store, which format and file the records;
 *   - host_build_graph/shared/orchestrator.cpp, compiled for the AICPU as well, where
 *     the platform's host headers are absent;
 *   - orchestration_api.h, compiled into the orchestration .so, likewise.
 *
 * So keep this header free of anything the AICPU or the orchestration .so cannot
 * take — no STL containers, no platform types. HostPhaseRecord stores a kind as a
 * plain uint32_t, so it does not depend on this header either way.
 */

#pragma once

#include <cstdint>

/**
 * What one HostPhaseRecord measured.
 *
 * The bind kinds partition the bind stage: their durations sum to the
 * `chip.run.bind` span. The orchestrator kinds are nested inside BindHostOrch
 * and do not partition it — some are sub-operations of others. Each orchestrator
 * kind is recorded at exactly one site, named beside it.
 */
enum class HostPhaseKind : uint32_t {
    BindArgs = 0,
    BindArenaBuild,
    BindStaticArena,
    BindGmHeap,
    BindSharedMem,
    BindRuntimeInit,
    BindHostOrch,
    BindGraphUpload,
    BindRelocate,
    BindSmH2d,
    BindArenaH2d,
    BindHostViewClose,
    // Recorded by the host orchestrator (host_build_graph/shared/orchestrator.cpp).
    OrchSubmitTask,         // submit_task_common: one ordinary task
    OrchAllocTensors,       // prepare_task: one alloc_tensors slot
    OrchRecordInGraphTask,  // graph_record_submit_in_graph_task: one recorded in-graph task
    OrchGraphSubmit,        // graph_submit_definition: one outer GRAPH task
    OrchBuildDefinition,    // graph_layout_definition + graph_fill_definition: one image
    OrchGraphBegin,         // graph_begin: the whole entry, OrchGraphSubmit nested inside
    OrchRecordingWait,      // graph_commit's wait for the last recorder to finish
    OrchGraphCommit,        // graph_commit: the wait plus back-patching every shell
    // Recorded in the orchestration .so (orchestration_api.h), which measures the
    // three submission segments the runtime cannot see.
    OrchSubmitAdmit,    // rt_submit_graph: entry until the Graph is admitted
    OrchRecordHandoff,  // rt_graph_begin's return until the recorder's first in-graph task
    OrchGeneratedArgs,  // between two submissions: the generated code's own arg setup
    Count
};

constexpr uint32_t kHostPhaseKindCount = static_cast<uint32_t>(HostPhaseKind::Count);
