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

#include <stdint.h>

#include "orchestration_api.h"

namespace {

constexpr int32_t kAicFuncId = 0;
constexpr int32_t kAiv0FuncId = 1;
constexpr int32_t kAiv1FuncId = 2;
constexpr int32_t kSlotsPerBlock = 3;

int16_t half_capacity(int32_t capacity) {
    const int32_t half = capacity / 2;
    return static_cast<int16_t>(half < 2 ? 2 : half);
}

TaskId submit(
    const simpler::hbg::Tensor &output, const MixedKernels &kernels, int16_t block_num, int64_t base_cache_line,
    bool sync_start, const TaskId *dependency = nullptr
) {
    CoreTaskArgs args;
    args.add_inout(output);
    args.add_scalar(base_cache_line);
    args.launch_spec.set_block_num(block_num);
    args.launch_spec.set_require_sync_start(sync_start);
    if (dependency != nullptr) args.set_dependencies(dependency, 1);
    return rt_submit_task(kernels, args).task_id();
}

void report_layout(
    const simpler::hbg::Tensor &layout, int32_t task, int16_t block_num, int32_t base_cache_line, int32_t mask
) {
    const int32_t values[3] = {block_num, base_cache_line, mask};
    for (int32_t field = 0; field < 3; ++field) {
        uint32_t index[1] = {static_cast<uint32_t>(task * 3 + field)};
        set_tensor_data<int32_t>(layout, 1, index, values[field]);
    }
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 2};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const simpler::hbg::Tensor &output = args.tensor(0).ref();
    const simpler::hbg::Tensor &layout = args.tensor(1).ref();
    const int16_t aiv_blocks = half_capacity(rt_available_aiv_count());
    const int16_t mix_blocks = half_capacity(rt_available_cluster_count());
    const MixedKernels aic{kAicFuncId, INVALID_KERNEL_ID, INVALID_KERNEL_ID};
    const MixedKernels aiv{INVALID_KERNEL_ID, kAiv0FuncId, INVALID_KERNEL_ID};
    const MixedKernels mix{kAicFuncId, kAiv0FuncId, kAiv1FuncId};
    const int16_t block_nums[5] = {1, aiv_blocks, mix_blocks, aiv_blocks, mix_blocks};
    const int32_t masks[5] = {1, 2, 7, 2, 7};
    int32_t base_cache_line = 0;

    TaskId root = submit(output, aic, block_nums[0], base_cache_line, false);
    report_layout(layout, 0, block_nums[0], base_cache_line, masks[0]);
    base_cache_line += block_nums[0] * kSlotsPerBlock;
    (void)submit(output, aiv, block_nums[1], base_cache_line, false);
    report_layout(layout, 1, block_nums[1], base_cache_line, masks[1]);
    base_cache_line += block_nums[1] * kSlotsPerBlock;
    (void)submit(output, mix, block_nums[2], base_cache_line, false);
    report_layout(layout, 2, block_nums[2], base_cache_line, masks[2]);
    base_cache_line += block_nums[2] * kSlotsPerBlock;
    (void)submit(output, aiv, block_nums[3], base_cache_line, true);
    report_layout(layout, 3, block_nums[3], base_cache_line, masks[3]);
    base_cache_line += block_nums[3] * kSlotsPerBlock;
    (void)submit(output, mix, block_nums[4], base_cache_line, true, &root);
    report_layout(layout, 4, block_nums[4], base_cache_line, masks[4]);
}

}  // extern "C"
