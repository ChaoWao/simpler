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

#include <cstdint>
#include <limits>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

#include "host_build_graph/task_id_encoding.h"

#define FUNC_NOOP_AIC 0
#define FUNC_NOOP_AIV0 1
#define FUNC_NOOP_AIV1 2

namespace {

void submit_zero_block_task() {
    CoreTaskArgs args;
    args.launch_spec.set_block_num(0);
    rt_submit_aiv_task(FUNC_NOOP_AIV0, args);
}

void submit_overflowing_mix_task() {
    MixedKernels kernels;
    kernels.aic_kernel_id = FUNC_NOOP_AIC;
    kernels.aiv0_kernel_id = FUNC_NOOP_AIV0;
    kernels.aiv1_kernel_id = FUNC_NOOP_AIV1;

    CoreTaskArgs args;
    constexpr int16_t kOverflowingBlockCount = std::numeric_limits<int16_t>::max() / 3 + 1;
    args.launch_spec.set_block_num(kOverflowingBlockCount);
    rt_submit_task(kernels, args);
}

simpler::hbg::Tensor tensor_with_unbound_owner(const simpler::hbg::Tensor &external) {
    simpler::hbg::Tensor forged = external;
    forged.owner_task_id = simpler::hbg::make_global_task(17);
    return forged;
}

// An IN_GRAPH id names storage inside one Graph task's body, not a task-table slot,
// so it can never be a fanin producer. Declaring one as an explicit dependency is
// the caller error append_fanin_or_fail rejects.
void submit_task_depending_on_in_graph_task() {
    const TaskId deps[1] = {simpler::hbg::make_in_graph_task(/*graph_local_id=*/1, /*task_index=*/0)};
    CoreTaskArgs args;
    args.launch_spec.set_block_num(1);
    args.set_dependencies(deps, 1);
    rt_submit_aiv_task(FUNC_NOOP_AIV0, args);
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 2};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::hbg::Tensor &external = orch_args.tensor(0).ref();
    uint64_t case_id = orch_args.scalar(0);
    uint32_t index[1] = {0};

    switch (case_id) {
    case 0:
        submit_zero_block_task();
        return;
    case 1:
        submit_overflowing_mix_task();
        return;
    case 2:
        (void)get_tensor_data<int32_t>(tensor_with_unbound_owner(external), 1, index);
        return;
    case 3:
        set_tensor_data<int32_t>(tensor_with_unbound_owner(external), 1, index, 7);
        return;
    case 4:
        submit_task_depending_on_in_graph_task();
        return;
    default:
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "unknown validation case %llu", static_cast<unsigned long long>(case_id)
        );
        return;
    }
}

}  // extern "C"
