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
 * Chained Early-Dispatch Orchestration (host_build_graph)
 *
 * A two-level ED chain: the middle task is both a candidate and a tracked
 * producer. While the slow root still runs, P1 is pre-staged gated; its staged
 * blocks publish (publication counts placement, not launch), which makes C a
 * candidate while P1 itself still waits for its doorbell — a gated producer
 * with a gated consumer chained behind it. Release then cascades in staging
 * order: P0 completes -> P1's doorbells ring -> P1 completes -> C's ring.
 *
 * Tasks (each writes one cache line per block via the shared slow-write
 * kernel):
 *   P0: AIC blocks=8, base_cl=0,  long spin, allow_early_resolve=true
 *   P1: AIC blocks=8, base_cl=8,  short spin, allow_early_resolve=true, dep=[P0]
 *   C:  AIV blocks=8, base_cl=16, no spin, dep=[P1]
 *
 * Args layout: [output]
 */

#include <stddef.h>
#include <stdint.h>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)
#include "arg_with_deps.h"      // NOLINT(build/include_subdir)

#define FUNC_WRITE_AIC 0
#define FUNC_WRITE_AIV 1

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return OrchestrationConfig{
        .expected_arg_count = 1,
    };
}

// The root spins long enough for the scheduler to pre-stage P1 while P0 runs,
// and for C's detection to fire off P1's gated publication.
static constexpr int64_t ROOT_SPIN_ITERS = 10000000;
static constexpr int64_t MID_SPIN_ITERS = 100000;
static constexpr int32_t BLOCKS = 8;

static TaskId submit_slow(
    int32_t kernel_id, bool aic, const simpler::hbg::Tensor &out, int64_t base_cl, int64_t spin, TaskId dep,
    bool flagged
) {
    CoreTaskArgsWithDeps<2> args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.add_scalar(spin);
    args.launch_spec.set_block_num(BLOCKS);
    args.set_allow_early_resolve(flagged);
    if (dep.is_valid()) args.add_dep(dep);
    return aic ? rt_submit_aic_task(kernel_id, args).task_id() : rt_submit_aiv_task(kernel_id, args).task_id();
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::hbg::Tensor &out = orch_args.tensor(0).ref();

    rt_scope_begin(ScopeMode::MANUAL);
    TaskId p0 = submit_slow(FUNC_WRITE_AIC, true, out, 0, ROOT_SPIN_ITERS, TaskId::invalid(), /*flagged=*/true);
    TaskId p1 = submit_slow(FUNC_WRITE_AIC, true, out, BLOCKS, MID_SPIN_ITERS, p0, /*flagged=*/true);
    submit_slow(FUNC_WRITE_AIV, false, out, 2 * BLOCKS, 0, p1, /*flagged=*/false);
    rt_scope_end();

    LOG_INFO("[chained_early_dispatch] P0(%d) -> P1(%d, candidate+tracked) -> C(%d)", BLOCKS, BLOCKS, BLOCKS);
}

}  // extern "C"
