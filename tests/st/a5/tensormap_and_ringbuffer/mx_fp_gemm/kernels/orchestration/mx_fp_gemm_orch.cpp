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
 * MXFP8 / MXFP4 ``TMATMUL_MX`` orchestration.
 *
 * L2 args: [A, As, B, Bs, C, mode]
 *   mode 0 = MXFP8 (FP8E4M3FN + E8M0), mode 1 = MXFP4 (FP4E2M1x2 + E8M0)
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_MATMUL_MX 0

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return OrchestrationConfig{
        .expected_arg_count = 6,  // A, As, B, Bs, C, mode
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const ChipTensor &ext_a = orch_args.tensor(0).ref();
    const ChipTensor &ext_as = orch_args.tensor(1).ref();
    const ChipTensor &ext_b = orch_args.tensor(2).ref();
    const ChipTensor &ext_bs = orch_args.tensor(3).ref();
    const ChipTensor &ext_c = orch_args.tensor(4).ref();
    const uint64_t mode = orch_args.scalar(0);

    LOG_INFO(
        "[mx_fp_gemm_orch] TMATMUL_MX mode=%llu A=%s As=%s B=%s Bs=%s C=%s", static_cast<unsigned long long>(mode),
        get_dtype_name(ext_a.dtype), get_dtype_name(ext_as.dtype), get_dtype_name(ext_b.dtype),
        get_dtype_name(ext_bs.dtype), get_dtype_name(ext_c.dtype)
    );

    SIMPLER_SCOPE() {
        CoreTaskArgs args;
        args.add_input(ext_a);
        args.add_input(ext_as);
        args.add_input(ext_b);
        args.add_input(ext_bs);
        args.add_output(ext_c);
        args.add_scalar(mode);
        rt_submit_aic_task(FUNC_MATMUL_MX, args);
    }
}

}  // extern "C"
