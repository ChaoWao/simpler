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

#pragma once

#include <cstddef>
#include <cstdint>

inline constexpr const char *CHIP_SWIMLANE_ARCHITECTURE_NAME = "a5";

enum class ChipSwimlaneSchedPhaseKind : uint32_t {
    Complete = 0,
    Dispatch = 1,
    Release = 2,
    Dummy = 4,
    EarlyDispatch = 5,
    Resolve = 6,
    DummyTask = 7,
    Drain = 8,
    DrainPrepare = 9,
    DrainPublish = 10,
    AsyncPoll = 11,
    PredicatedSkip = 12,
    GraphPrepare = 13,
    ResolveStandalone = 14,
};

constexpr int CHIP_SWIMLANE_NUM_QUEUE_SHAPES = 3;

struct ChipSwimlaneAicpuSchedPhaseRecord {
    uint64_t start_time;
    uint64_t end_time;
    uint32_t loop_iter;
    ChipSwimlaneSchedPhaseKind kind;
    uint32_t tasks_processed;
    union {
        struct {
            uint32_t pop_hit;
            uint32_t pop_miss;
        } dispatch;
        struct {
            uint32_t local_id;
            uint32_t ring_id;
        } dummy_task;
        struct {
            uint32_t local_id;
            uint32_t ring_id;
        } graph_task;
    } phase_data;
    int16_t shared_depth_at_start[CHIP_SWIMLANE_NUM_QUEUE_SHAPES];
    int16_t shared_depth_at_end[CHIP_SWIMLANE_NUM_QUEUE_SHAPES];
    uint32_t _pad[4];
};

static_assert(sizeof(decltype(ChipSwimlaneAicpuSchedPhaseRecord::phase_data)) == 8);
static_assert(offsetof(ChipSwimlaneAicpuSchedPhaseRecord, phase_data) == 28);
static_assert(sizeof(ChipSwimlaneAicpuSchedPhaseRecord) == 64);
