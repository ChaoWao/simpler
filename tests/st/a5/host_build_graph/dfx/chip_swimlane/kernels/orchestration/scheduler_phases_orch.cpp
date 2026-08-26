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

#include "orchestration_api.h"

namespace {

constexpr int kNoopKernel = 0;

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 1};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const simpler::hbg::Tensor &input = args.tensor(0).ref();

    CoreTaskArgs normal_args;
    normal_args.add_input(input);
    normal_args.launch_spec.set_block_num(1);
    const TaskId normal_task = rt_submit_aiv_task(kNoopKernel, normal_args).task_id();

    CoreTaskArgs dummy_args;
    dummy_args.set_dependencies(&normal_task, 1);
    rt_submit_dummy_task(dummy_args);
}

}  // extern "C"
