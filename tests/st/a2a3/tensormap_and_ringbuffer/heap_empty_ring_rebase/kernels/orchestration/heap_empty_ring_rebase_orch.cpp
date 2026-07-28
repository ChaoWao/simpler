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
 * Allocates one 1 MiB scratch in each of two consecutive scopes on ring 1.
 * Ring 1 has 1.5 MiB of heap, so the second allocation requires the drained
 * ring parked at offset 1 MiB to rebase to zero.
 */

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_FILL_CONST 0
#define FUNC_COPY_FIRST 1

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 2,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &ext_Y1 = orch_args.tensor(0).ref();
    const Tensor &ext_Y2 = orch_args.tensor(1).ref();

    uint32_t scratch_shapes[2] = {1024, 256};
    TensorCreateInfo scratch_ci(scratch_shapes, 2, DataType::FLOAT32);

    PTO2_SCOPE() {
        L0TaskArgs fill1;
        fill1.add_output(scratch_ci);
        TaskOutputTensors t1_outs = rt_submit_aic_task(FUNC_FILL_CONST, fill1);
        const Tensor &t1 = t1_outs.get_ref(0);

        L0TaskArgs copy1;
        copy1.add_input(t1);
        copy1.add_inout(ext_Y1);
        rt_submit_aic_task(FUNC_COPY_FIRST, copy1);
    }

    PTO2_SCOPE() {
        L0TaskArgs fill2;
        fill2.add_output(scratch_ci);
        TaskOutputTensors t2_outs = rt_submit_aic_task(FUNC_FILL_CONST, fill2);
        const Tensor &t2 = t2_outs.get_ref(0);

        L0TaskArgs copy2;
        copy2.add_input(t2);
        copy2.add_inout(ext_Y2);
        rt_submit_aic_task(FUNC_COPY_FIRST, copy2);
    }
}

}  // extern "C"
