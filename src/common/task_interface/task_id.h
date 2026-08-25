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
 * TaskId — minimal standalone header.
 *
 * Factored out of runtime_types.h so that tensor.h can include it
 * without pulling in scheduler-internal constants (heap sizes, timeouts, etc.).
 */

#pragma once

#include <cstdint>

/**
 * TaskId: an opaque 64-bit task handle.
 *
 * The bit layout belongs to the runtime that mints the id, and the two runtimes
 * encode different things in the high bits — a ring index in `tmr`, an id space in
 * `hbg`. Neither is declared here; code that mints or decodes an id includes its
 * own runtime's encoding header:
 *   src/common/host_build_graph/task_id_encoding.h
 *   src/common/tensormap_and_ringbuffer/task_id_encoding.h
 *
 * What every holder may rely on: the handle is 8 bytes, copyable, comparable for
 * identity, and has one reserved sentinel.
 *
 * Invalid sentinel: raw == UINT64_MAX (no valid task has this encoding).
 */
struct TaskId {
    uint64_t raw;

    static constexpr TaskId invalid() { return TaskId{UINT64_MAX}; }

    constexpr bool is_valid() const { return raw != UINT64_MAX; }
    constexpr bool is_invalid() const { return raw == UINT64_MAX; }

    constexpr bool operator==(const TaskId &other) const { return raw == other.raw; }
    constexpr bool operator!=(const TaskId &other) const { return raw != other.raw; }
};

static_assert(sizeof(TaskId) == 8, "TaskId must stay 8 bytes (shared memory ABI)");
