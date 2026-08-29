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
 * Many-to-one barrier via explicit set_dependencies — exercises the dep_gen
 * overflow chain wire format.
 *
 * Submits N producers each writing X[0] = 42.0, then a dummy_T whose only
 * dependency surface is set_dependencies_with_kinds({all N producer ids}, N), then a
 * consumer that ordering-depends on the barrier and copies X[0] -> Y[0].
 *
 * Picking N > DEP_GEN_MAX_EXPLICIT_DEPS (=64) forces the dep_gen capture to
 * spill into one or more DepGenOverflowRecord slots; picking N to span the
 * 64 + k*524 boundaries exercises both single- and multi-overflow chains.
 *
 * Args layout: [X, Y, scalar(N)]
 *   - X: every producer writes it (tensormap auto-deps the chain so the
 *        SENTINEL is preserved); consumer reads it. The value check is an
 *        execution sanity check; deps.json verifies the explicit barrier.
 *   - Y: consumer writes it; host checks Y[0] == SENTINEL.
 *
 * Scalar: N (1 .. MAX_PRODUCERS).
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_WRITE_CONST 0
#define FUNC_COPY_FIRST 1

// Stack room for producer_ids[] and producer_kinds[]. 600 covers the second
// overflow boundary (64 + 524 + 1);
// CHIP_DEP_LIST_POOL_SIZE (16384) is the real ceiling on a per-ring basis.
static constexpr int32_t MAX_PRODUCERS = 600;

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{
        .expected_arg_count = 3,  // X, Y, scalar(N)
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::tmr::Tensor &ext_X = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &ext_Y = orch_args.tensor(1).ref();

    uint64_t n_raw = orch_args.scalar(0);
    int32_t n = static_cast<int32_t>(n_raw);
    if (n < 1 || n > MAX_PRODUCERS) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "chain_barrier_orch: invalid n=%d", n);
        return;
    }

    TaskId producer_ids[MAX_PRODUCERS];
    DepFlags producer_kinds[MAX_PRODUCERS];

    // N producers each INOUT X. tensormap auto-deps them in a chain, so X[0]
    // stays at SENTINEL through all of them. The final value is an execution
    // sanity check; the deps.json assertions verify the explicit barrier.
    for (int32_t i = 0; i < n; i++) {
        CoreTaskArgs args;
        args.add_inout(ext_X);
        producer_ids[i] = rt_submit_aic_task(FUNC_WRITE_CONST, args).task_id();
        producer_kinds[i] = (i % 2 == 0) ? DEP_WAIT : (DEP_WAIT | DEP_RETAIN);
    }

    // Dummy barrier with explicit deps on ALL N producers. dc=n > 64 forces
    // the dep_gen writer to emit base + overflow chain.
    TaskId barrier_id;
    {
        CoreTaskArgs args;
        args.set_dependencies_with_kinds(producer_ids, producer_kinds, n);
        barrier_id = rt_submit_dummy_task(args).task_id();
    }

    // Consumer: ordering-only dep on barrier, reads X, writes Y.
    {
        CoreTaskArgsWithDeps<1> args;
        args.add_dep_wait(barrier_id);
        args.add_input(ext_X);
        args.add_inout(ext_Y);
        rt_submit_aic_task(FUNC_COPY_FIRST, args);
    }

    // The final pair reaches the same producer through an explicit ordering
    // dependency and runtime-output ownership. FaninBuilder folds both reasons
    // into one WAIT|RETAIN edge.
    uint32_t owned_shape[1] = {1};
    TensorCreateInfo owned_ci(owned_shape, 1, DataType::FLOAT32);
    CoreTaskArgs creator_args;
    creator_args.add_output(owned_ci);
    TaskOutputTensors creator_outputs = rt_submit_dummy_task(creator_args);
    const simpler::tmr::Tensor &owned = creator_outputs.get_ref(0);

    TaskId combined_dep = creator_outputs.task_id();
    DepFlags combined_kind = DEP_WAIT;
    CoreTaskArgs combined_args;
    combined_args.set_dependencies_with_kinds(&combined_dep, &combined_kind, 1);
    combined_args.add_input(owned);
    rt_submit_dummy_task(combined_args);
}

}  // extern "C"
