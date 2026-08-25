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

#include <array>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_LOG_SQRT 0
#define FUNC_MATMUL 1
#define FUNC_ADD_EXP 2

namespace {

// Qwen-style fixed decoder-layer topology:
//
//                 +--> matmul(weight_1) --+
// input -> norm --|                       +--> add + activation
//                 +--> matmul(weight_2) --+
//
// A cache miss records two AIC and two AIV tasks. Cache hits submit one outer
// Graph task and let Scheduler materialize this saved topology.
void decoder_layer(const GraphTaskArgs &args) {
    const simpler::hbg::Tensor &input = args.tensor(0).ref();
    const simpler::hbg::Tensor &weight_1 = args.tensor(1).ref();
    const simpler::hbg::Tensor &weight_2 = args.tensor(2).ref();
    const simpler::hbg::Tensor &output = args.tensor(3).ref();

    const std::array<uint32_t, 1> shape{input.shapes[0]};
    TensorCreateInfo normalized_info(shape.data(), static_cast<uint32_t>(shape.size()), DataType::FLOAT16);
    TensorCreateInfo projected_info(shape.data(), static_cast<uint32_t>(shape.size()), DataType::FLOAT32);

    CoreTaskArgs norm_args;
    norm_args.add_input(input);
    norm_args.add_output(normalized_info);
    TaskOutputTensors normalized_outputs = rt_submit_aiv_task(FUNC_LOG_SQRT, norm_args);
    simpler::hbg::Tensor normalized = normalized_outputs.get_ref(0);

    CoreTaskArgs left_args;
    left_args.add_input(normalized, weight_1);
    left_args.add_output(projected_info);
    TaskOutputTensors left_outputs = rt_submit_aic_task(FUNC_MATMUL, left_args);
    simpler::hbg::Tensor left = left_outputs.get_ref(0);

    CoreTaskArgs right_args;
    right_args.add_input(normalized, weight_2);
    right_args.add_output(projected_info);
    TaskOutputTensors right_outputs = rt_submit_aic_task(FUNC_MATMUL, right_args);
    simpler::hbg::Tensor right = right_outputs.get_ref(0);

    CoreTaskArgs activation_args;
    activation_args.add_input(left, right);
    activation_args.add_output(output);
    rt_submit_aiv_task(FUNC_ADD_EXP, activation_args);
}

void submit_layer(const GraphTaskArgs &args) { rt_submit_graph(&decoder_layer, args); }

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{
        .expected_arg_count = 6,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    for (int32_t output_index = 3; output_index < 6; ++output_index) {
        GraphTaskArgs layer_args;
        layer_args.add_input(args.tensor(0).ref(), args.tensor(1).ref(), args.tensor(2).ref());
        layer_args.add_output(args.tensor(output_index).ref());
        submit_layer(layer_args);
    }
}

}  // extern "C"
