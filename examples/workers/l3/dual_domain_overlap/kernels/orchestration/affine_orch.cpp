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

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig affine_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 4};
}

__attribute__((visibility("default"))) void affine_orchestration(const ChipTaskArgs &orch_args) {
    const simpler::tmr::Tensor &reduce_out = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &scale = orch_args.tensor(1).ref();
    const simpler::tmr::Tensor &bias = orch_args.tensor(2).ref();
    const simpler::tmr::Tensor &out = orch_args.tensor(3).ref();

    CoreTaskArgs params;
    params.add_input(reduce_out);
    params.add_input(scale);
    params.add_input(bias);
    params.add_output(out);
    rt_submit_aiv_task(0, params);
}

}  // extern "C"
