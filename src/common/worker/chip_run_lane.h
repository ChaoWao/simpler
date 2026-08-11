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

#include <chrono>
#include <cstdint>
#include <memory>

#include "../task_interface/call_config.h"
#include "../task_interface/task_args.h"
#include "pipeline_slot_pool.h"

class ChipWorker;
struct ChipRunLaneState;
struct ChipRunState;

enum class ChipRunPreparationDisposition : int32_t {
    VALIDATED_ONLY = 1,
    NATIVE_PREPARED = 2,
};

class ChipRun {
public:
    using Clock = std::chrono::steady_clock;
    using Deadline = Clock::time_point;

    ChipRun() = default;

    bool done();
    bool wait_until(Deadline deadline);
    void activate();
    void abandon();

    bool launched() const;
    bool lane_poisoned() const;
    ChipRunPreparationDisposition preparation_disposition() const;

private:
    friend class ChipRunLane;
    ChipRun(std::shared_ptr<ChipRunLaneState> lane, std::shared_ptr<ChipRunState> run);

    std::shared_ptr<ChipRunLaneState> lane_;
    std::shared_ptr<ChipRunState> run_;
};

class ChipRunLane {
public:
    using Clock = ChipRun::Clock;
    using Deadline = ChipRun::Deadline;

    explicit ChipRunLane(ChipWorker &worker);
    ~ChipRunLane();

    ChipRunLane(const ChipRunLane &) = delete;
    ChipRunLane &operator=(const ChipRunLane &) = delete;

    ChipRun submit(
        int32_t callable_id, const ChipStorageTaskArgs &args, const CallConfig &config, const PipelineSlotLease &lease,
        uint64_t run_id, uint64_t dispatch_id, volatile int32_t *accepted_state = nullptr, int32_t accepted_value = 0,
        bool activated = true
    );
    ChipRun submit(
        int32_t callable_id, const ChipStorageTaskArgs &args, const CallConfig &config,
        volatile int32_t *accepted_state = nullptr, int32_t accepted_value = 0
    );

    void drain();
    void close();
    bool poisoned() const;

private:
    std::shared_ptr<ChipRunLaneState> state_;
};
