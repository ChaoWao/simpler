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
 * Negative ST orchestration: PTO2_ERROR_REQUIRE_SYNC_START_INVALID (code 7).
 *
 * Submits one SPMD AIV task that asks for require_sync_start with a block_num
 * far larger than the available cores. A sync-start launch needs every block
 * resident at once, so block_num > core limit is a guaranteed deadlock; the
 * orchestrator catches it proactively at submit time (before any dispatch) and
 * latches REQUIRE_SYNC_START_INVALID. core_num=1000 exceeds the AIV core count
 * on every supported platform. The noop kernel never runs.
 */

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_NOOP_KERNEL 0

template <typename Spec>
static inline auto set_spmd_count(Spec &spec, int16_t n) -> decltype(spec.set_block_num(n), void()) {
    spec.set_block_num(n);
}

template <typename Spec>
static inline auto set_spmd_count(Spec &spec, int16_t n) -> decltype(spec.set_core_num(n), void()) {
    spec.set_core_num(n);
}

static inline void set_block_count(CoreTaskArgs &args, int16_t n) { set_spmd_count(args.launch_spec, n); }

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 0,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    (void)orch_args;

    uint32_t shape[1] = {1};
    TensorCreateInfo ci(shape, 1, DataType::INT32);

    CoreTaskArgs args;
    args.add_output(ci);
    set_block_count(args, 1000);                    // >> available AIV cores
    args.launch_spec.set_require_sync_start(true);  // arm the sync-start deadlock guard
    rt_submit_aiv_task(FUNC_NOOP_KERNEL, args);
}

}  // extern "C"
