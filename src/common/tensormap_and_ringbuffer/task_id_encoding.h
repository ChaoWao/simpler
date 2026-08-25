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
 * How the `tmr` runtime encodes a `TaskId`. The layout is private to this runtime:
 * `TaskId` itself is an opaque 64-bit handle, and `hbg` encodes something else in
 * the same bits (see src/common/host_build_graph/task_id_encoding.h).
 */

#pragma once

#include <cstdint>

#include "task_id.h"

namespace simpler::tmr {

/**
 * raw = (ring_id << 32) | local_id
 *
 * ring_id:  which ring layer the task was placed on (0..CHIP_MAX_RING_DEPTH-1)
 * local_id: that ring's monotonic task counter
 *
 * Every task this runtime mints lives on a ring, so the pair fully identifies a
 * ring slot: `rings[ring_id].get_slot_by_task_id(local_id)`.
 */
constexpr TaskId make_task_id(uint8_t ring_id, uint32_t local_id) {
    return TaskId{(static_cast<uint64_t>(ring_id) << 32) | static_cast<uint64_t>(local_id)};
}

constexpr uint8_t task_ring(TaskId id) { return static_cast<uint8_t>(id.raw >> 32); }

constexpr uint32_t task_local_id(TaskId id) { return static_cast<uint32_t>(id.raw & 0xFFFFFFFFu); }

}  // namespace simpler::tmr
