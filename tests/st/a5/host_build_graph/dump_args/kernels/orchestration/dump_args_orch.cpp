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
 * Dump-args orchestration — submit_task / TensorMap form
 *
 * Builds: f = (a + b) + 1
 *   t0: f = a + b            (kernel_add, AIV)
 *   t1: f = f + 1 in place   (kernel_add_scalar_inplace, AIV, INOUT)
 *
 * The t0 -> t1 dependency is discovered by the TensorMap: t0 produces f
 * (add_output), t1 reads-and-writes f (add_inout). Per-task tensor metadata
 * for the dump subsystem is derived automatically from the Arg tensors.
 *
 * Arg layout: [a (IN), b (IN), f (OUT)].
 */

#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_ADD 0
#define FUNC_ADD_SCALAR_INPLACE 1

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const ChipTensor &a = orch_args.tensor(0).ref();
    const ChipTensor &b = orch_args.tensor(1).ref();
    const ChipTensor &f = orch_args.tensor(2).ref();

    // t0: f = a + b
    CoreTaskArgs p0;
    p0.add_input(a);
    p0.add_input(b);
    p0.add_output(f);
    rt_submit_aiv_task(FUNC_ADD, p0);

    // t1: f = f + 1 (in place); INOUT establishes the dependency on t0 via f
    float scale = 1.0F;
    CoreTaskArgs p1;
    p1.add_inout(f);
    p1.add_scalar(scale);
    p1.dump(f, scale);
    rt_submit_aiv_task(FUNC_ADD_SCALAR_INPLACE, p1);

    LOG_INFO("[dump_args_orch] Submitted f = (a + b) + 1");
}

}  // extern "C"
