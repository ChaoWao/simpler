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

#include <atomic>
#include <cstdint>

#include "common/core_type.h"

class Runtime;
struct AicpuCoreLifecycleTrace;

class AicoreLifecycle {
public:
    int32_t pre_handshake_init(Runtime *runtime, int32_t aicpu_thread_num, uint64_t regs_base);
    void handshake_partition(Runtime *runtime, int32_t tidx, int32_t nthreads);
    int32_t post_handshake_init(Runtime *runtime);
    void publish_context_partition(Runtime *runtime, int32_t thread_idx);
    int32_t wait_bootstrap_complete(Runtime *runtime);
    int32_t release_partition(int32_t thread_idx, bool start_execution);
    void signal_shutdown_partition(int32_t thread_idx);
    int32_t finish_shutdown_partition(int32_t thread_idx, Runtime *runtime);
    void deinit();

private:
    static constexpr int32_t kMaxWorkers = 108;

    struct CoreState {
        uint64_t reg_addr;
        uint32_t physical_core_id;
        CoreType core_type;
        AicpuCoreLifecycleTrace *trace;
        uint64_t handshake_observed_cycles;
        uint64_t handshake_partition_complete_cycles;
    };

    CoreState cores_[kMaxWorkers]{};
    uint32_t physical_core_ids_[kMaxWorkers]{};
    std::atomic<bool> handshake_failed_{false};
    int32_t core_count_{0};
    int32_t aicpu_thread_num_{0};
    uint64_t regs_base_{0};
};
