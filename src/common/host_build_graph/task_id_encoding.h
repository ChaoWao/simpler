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
 * This runtime has exactly one ring, so the high bits name the *storage* an id
 * resolves against rather than a ring index:
 *
 *   RING       — a task placed on the shared-memory ring. Its low bits are the
 *                task allocator's local id, resolvable via get_slot_by_task_id().
 *   GRAPH_NODE — a node materialized inside a Graph execution. It lives in that
 *                execution's GraphNodeStorage, not on the ring, so its low bits
 *                are the packed pair below and must never be resolved against a
 *                ring slot.
 */
enum class TaskIdSpace : uint32_t { RING = 0, GRAPH_NODE = 1 };

// A graph node's low bits pack its outer GRAPH task's local id above its index
// within that graph. graph_execution.h asserts GRAPH_MAX_NODES fits the index.
inline constexpr uint32_t GRAPH_NODE_INDEX_BITS = 10;

constexpr TaskId make_ring_task(uint32_t local_id) {
    return TaskId{(static_cast<uint64_t>(TaskIdSpace::RING) << 32) | static_cast<uint64_t>(local_id)};
}

constexpr TaskId make_graph_node(uint32_t outer_local_id, uint32_t node_index) {
    const uint32_t packed = (outer_local_id << GRAPH_NODE_INDEX_BITS) | node_index;
    return TaskId{(static_cast<uint64_t>(TaskIdSpace::GRAPH_NODE) << 32) | static_cast<uint64_t>(packed)};
}

constexpr TaskIdSpace task_id_space(TaskId id) { return static_cast<TaskIdSpace>(static_cast<uint32_t>(id.raw >> 32)); }

constexpr bool is_ring_task(TaskId id) { return task_id_space(id) == TaskIdSpace::RING; }

// The low 32 bits. A ring local id for a RING task; the packed pair above for a
// GRAPH_NODE, which is why callers that resolve a ring slot must gate on
// is_ring_task() first.
constexpr uint32_t task_local_id(TaskId id) { return static_cast<uint32_t>(id.raw & 0xFFFFFFFFu); }

}  // namespace simpler::hbg
