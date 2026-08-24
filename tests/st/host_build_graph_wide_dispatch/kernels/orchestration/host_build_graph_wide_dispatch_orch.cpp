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

#include "arg_with_deps.h"      // NOLINT(build/include_subdir)
#include "orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_SPMD_MIX_AIC 0
#define FUNC_SPMD_MIX_AIV0 1
#define FUNC_SPMD_MIX_AIV1 2
#define FUNC_SPMD_WRITE_AIV 3

namespace {

MixedKernels mix_kernels() {
    MixedKernels kernels;
    kernels.aic_kernel_id = FUNC_SPMD_MIX_AIC;
    kernels.aiv0_kernel_id = FUNC_SPMD_MIX_AIV0;
    kernels.aiv1_kernel_id = FUNC_SPMD_MIX_AIV1;
    return kernels;
}

TaskId submit_aiv(const ChipTensor &output, int16_t block_num) {
    CoreTaskArgs args;
    args.add_inout(output);
    args.add_scalar(0);
    args.add_scalar(0);
    args.launch_spec.set_block_num(block_num);
    return rt_submit_aiv_task(FUNC_SPMD_WRITE_AIV, args).task_id();
}

TaskId submit_sync_mix(const ChipTensor &output, int16_t block_num, int64_t base, TaskId producer) {
    CoreTaskArgsWithDeps<1> args;
    args.add_inout(output);
    args.add_scalar(base);
    args.add_scalar(0);
    args.launch_spec.set_block_num(block_num);
    args.launch_spec.set_require_sync_start(true);
    args.add_dep(producer);
    return rt_submit_task(mix_kernels(), args).task_id();
}

void submit_normal_mix(const ChipTensor &output, int16_t block_num, int64_t base, TaskId producer) {
    CoreTaskArgsWithDeps<1> args;
    args.add_inout(output);
    args.add_scalar(base);
    args.add_scalar(0);
    args.launch_spec.set_block_num(block_num);
    args.add_dep(producer);
    rt_submit_task(mix_kernels(), args);
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 2};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const ChipTensor &output = orch_args.tensor(0).ref();
    const ChipTensor &layout = orch_args.tensor(1).ref();
    int16_t aiv_blocks = static_cast<int16_t>(rt_available_aiv_count());
    int16_t mix_blocks = static_cast<int16_t>(rt_available_cluster_count());

    TaskId aiv = submit_aiv(output, aiv_blocks);
    TaskId sync_mix = submit_sync_mix(output, mix_blocks, aiv_blocks, aiv);
    int64_t normal_base = aiv_blocks + 3 * mix_blocks;
    submit_normal_mix(output, mix_blocks, normal_base, sync_mix);

    uint32_t index[1] = {0};
    set_tensor_data<int32_t>(layout, 1, index, aiv_blocks);
    index[0] = 1;
    set_tensor_data<int32_t>(layout, 1, index, mix_blocks);
}

}  // extern "C"
