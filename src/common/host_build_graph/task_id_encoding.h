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
 * How the `hbg` runtime encodes a `TaskId`. The layout is private to this runtime:
 * `TaskId` itself is an opaque 64-bit handle, and `tmr` encodes something else in
 * the same bits (see src/common/tensormap_and_ringbuffer/task_id_encoding.h).
 */

#pragma once

#include <cstdint>

#include "task_id.h"

namespace simpler::hbg {

/**
 * Which id space a task id belongs to, held in the high 32 bits.
 *
 * Everything this runtime schedules is a task. The high bits say whether a task
 * belongs to a Graph task or stands on its own, which is also what decides where
 * its id resolves:
 *
 *   GLOBAL   — a task of the run itself, holding a slot in the shared-memory task
 *              table. Its low bits are the task allocator's local id, resolvable
 *              via get_slot_by_task_id().
 *   IN_GRAPH — a task belonging to one Graph task's body. It lives in that Graph's
 *              own storage, not in the task table, so its low bits are the packed
 *              pair below and must never be resolved against a table slot.
 *
 * An IN_GRAPH id is minted twice for the same task, in two disjoint scopes. The
 * recorder mints one per task it records, with `graph_local_id` fixed at 0: a
 * Definition is shared by every shell that replays it, so record time has no
 * single Graph task to name, and the low field is then the task index alone —
 * which is what keeps it inside the task chains the recording's hazard map is
 * dimensioned for. Materialize mints the other, with the replaying shell's real
 * local id. The two never meet: a recorded id lives only in the recorder thread's
 * private map and the body's own locals, and a materialized task is addressed by
 * index rather than looked up by id.
 */
enum class TaskIdSpace : uint32_t { GLOBAL = 0, IN_GRAPH = 1 };

// An in-graph task's low bits pack its Graph task's local id above its index
// within that Graph. graph_execution.h asserts MAX_IN_GRAPH_TASKS fits the index.
inline constexpr uint32_t IN_GRAPH_TASK_INDEX_BITS = 10;

constexpr TaskId make_global_task(uint32_t local_id) {
    return TaskId{(static_cast<uint64_t>(TaskIdSpace::GLOBAL) << 32) | static_cast<uint64_t>(local_id)};
}

constexpr TaskId make_in_graph_task(uint32_t graph_local_id, uint32_t task_index) {
    const uint32_t packed = (graph_local_id << IN_GRAPH_TASK_INDEX_BITS) | task_index;
    return TaskId{(static_cast<uint64_t>(TaskIdSpace::IN_GRAPH) << 32) | static_cast<uint64_t>(packed)};
}

constexpr TaskIdSpace task_id_space(TaskId id) { return static_cast<TaskIdSpace>(static_cast<uint32_t>(id.raw >> 32)); }

// True exactly when this id names a slot in the shared-memory task table, which is
// what every caller that is about to resolve one is really asking. An IN_GRAPH id
// answers false and must not reach get_slot_by_task_id().
constexpr bool is_global_task(TaskId id) { return task_id_space(id) == TaskIdSpace::GLOBAL; }

// The low 32 bits. A task-table local id for a GLOBAL task; the packed pair above
// for an IN_GRAPH one, which is why callers that resolve a table slot must gate on
// is_global_task() first.
constexpr uint32_t task_local_id(TaskId id) { return static_cast<uint32_t>(id.raw & 0xFFFFFFFFu); }

}  // namespace simpler::hbg
