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
 * TensorCreateInfo::set_initial_value on the two paths that materialize a
 * runtime allocation: alloc_tensors and a submitted task's output.
 *
 * Both buffers live in the GM heap, which the host orchestrator reaches only
 * through the run's registered host view, so a value that arrives at the kernel
 * is evidence that view resolved. Each is observed by a kernel rather than by
 * get_tensor_data, which cannot read a runtime-created tensor here.
 */

#include <stdint.h>

#include <array>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_ADD 0
#define FUNC_ADD_SCALAR 1

namespace {

constexpr float kAllocFill = 7.0F;
constexpr float kDummyFill = 11.0F;

// out_alloc = kAllocFill + a, out_dummy = kDummyFill.
void initial_value_body(const CoreTaskArgs &args) {
    const ChipTensor &a = args.tensor(0).ref();
    const ChipTensor &out_alloc = args.tensor(1).ref();
    const ChipTensor &out_dummy = args.tensor(2).ref();

    const std::array<uint32_t, 1> shape{a.shapes[0]};

    // alloc_tensors path: the allocation is never written by a task, so the
    // adder's first operand is the initial value itself.
    TensorCreateInfo filled_info(shape.data(), static_cast<uint32_t>(shape.size()), DataType::FLOAT32);
    filled_info.set_initial_value(kAllocFill);
    TaskOutputTensors filled_outputs = alloc_tensors(filled_info);
    ChipTensor filled = filled_outputs.get_ref(0);

    CoreTaskArgs add_args;
    add_args.add_input(filled, a);
    add_args.add_output(out_alloc);
    rt_submit_aiv_task(FUNC_ADD, add_args);

    // Submitted-task path: a dummy task dispatches no kernel, so its output
    // also reaches its consumer holding nothing but the initial value.
    TensorCreateInfo dummy_info(shape.data(), static_cast<uint32_t>(shape.size()), DataType::FLOAT32);
    dummy_info.set_initial_value(kDummyFill);
    CoreTaskArgs dummy_args;
    dummy_args.add_output(dummy_info);
    TaskOutputTensors dummy_outputs = rt_submit_dummy_task(dummy_args);
    ChipTensor dummied = dummy_outputs.get_ref(0);

    CoreTaskArgs copy_args;
    copy_args.add_input(dummied);
    copy_args.add_output(out_dummy);
    copy_args.add_scalar(0.0F);
    rt_submit_aiv_task(FUNC_ADD_SCALAR, copy_args);
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    CoreTaskArgs body_args;
    body_args.add_input(args.tensor(0).ref());
    body_args.add_output(args.tensor(1).ref());
    body_args.add_output(args.tensor(2).ref());
    initial_value_body(body_args);
}

}  // extern "C"
