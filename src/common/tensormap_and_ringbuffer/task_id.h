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
 * The `tmr` runtime's task handle and the layout it encodes into one.
 *
 * The layout is private to this runtime. `hbg` has its own TaskId, encoding an id
 * space in the same 64 bits (src/common/host_build_graph/task_id.h). The two are
 * distinct types in distinct namespaces, and a translation unit sees exactly one of
 * them. Nothing in the include path enforces that — src/common is on every target's
 * — so the trailing using-declaration does: reaching both headers from one scope is
 * a compile error, not a silent bind to whichever came first.
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace simpler::tmr {

/**
 * TaskId: a 64-bit task handle, `(ring_id << 32) | local_id`.
 *
 * ring_id:  which ring layer the task was placed on (0..CHIP_MAX_RING_DEPTH-1)
 * local_id: that ring's monotonic task counter
 *
 * Every task this runtime mints lives on a ring, so the pair fully identifies a
 * ring slot: `rings[id.ring()].get_slot_by_task_id(id.local_id())`.
 *
 * What every holder may rely on regardless of layout: the handle is 8 bytes,
 * copyable, comparable for identity, and has one reserved sentinel.
 *
 * Invalid sentinel: raw == UINT64_MAX. No valid task encodes it — a ring id is a
 * uint8_t, so bits 63-40 of a minted handle are always zero and raw stays below
 * 2^40 whatever the local counter reaches.
 */
struct TaskId {
    uint64_t raw;

    static constexpr TaskId invalid() { return TaskId{UINT64_MAX}; }

    static constexpr TaskId make(uint8_t ring_id, uint32_t local_id) {
        return TaskId{(static_cast<uint64_t>(ring_id) << 32) | static_cast<uint64_t>(local_id)};
    }

    constexpr bool is_valid() const { return raw != UINT64_MAX; }

    constexpr uint8_t ring() const { return static_cast<uint8_t>(raw >> 32); }

    constexpr uint32_t local_id() const { return static_cast<uint32_t>(raw & 0xFFFFFFFFu); }

    constexpr bool operator==(const TaskId &other) const { return raw == other.raw; }
    constexpr bool operator!=(const TaskId &other) const { return raw != other.raw; }
};

static_assert(
    std::is_trivially_copyable_v<TaskId> && std::is_standard_layout_v<TaskId>,
    "TaskId crosses the host-device boundary and must stay a POD wire type"
);
static_assert(sizeof(TaskId) == 8, "TaskId must stay 8 bytes (shared memory ABI)");

}  // namespace simpler::tmr

// A translation unit includes only its own runtime's task_id.h, so the unqualified
// name names this type. Two of these declarations in one scope are ill-formed, which
// is what makes a build that reaches both runtimes fail here rather than silently
// pick one.
//
// The Tensor next door carries no such declaration and is spelled simpler::tmr::Tensor
// at its call sites. TaskId differs because generated kernels and orchestration sources
// emit the bare name, and codegen has no runtime to qualify it with.
using simpler::tmr::TaskId;
