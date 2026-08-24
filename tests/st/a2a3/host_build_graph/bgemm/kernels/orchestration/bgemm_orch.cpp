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
 * BGEMM orchestration — submit_task / TensorMap form
 *
 * Tiled C = C_init + A @ B, tile 64x64, grid 4x4x4.  Per output tile (m,n),
 * for each k:
 *   P_k = A[m,k] @ B[k,n]   (gemm_tile, AIC)  — fresh runtime-allocated buffer
 *   C[m,n] += P_k           (tile_add, AIV, INOUT C tile)
 *
 * All dependencies are TensorMap-derived:
 *   - gemm_k -> add_k via the freshly-allocated P_k.
 *   - add_{k-1} -> add_k via the C[m,n] tile (add_inout overlap).
 * Allocating a fresh P per k (instead of reusing one P buffer per tile) removes
 * the write-after-read hazard the explicit-edge version needed an extra edge
 * for.
 *
 * Arg layout: [A (IN), B (IN), C (INOUT)] — flattened 1-D tile-first tensors.
 * C must be declared INOUT, not OUT: the k=0 tile_add reads C before any task
 * has written it, and the runtime stages only IN/INOUT tensors to the device
 * (a tensor declared OUT is handed to the kernel uninitialized).
 */

#include <stdint.h>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_GEMM 0
#define FUNC_TILE_ADD 1

extern "C" {

static constexpr int TILE = 64;
static constexpr int GRID_M = 4;
static constexpr int GRID_K = 4;
static constexpr int GRID_N = 4;
static constexpr int BATCH = 1;
static constexpr uint32_t TILE_ELEMS = TILE * TILE;

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::hbg::Tensor &a = orch_args.tensor(0).ref();
    const simpler::hbg::Tensor &b = orch_args.tensor(1).ref();
    const simpler::hbg::Tensor &c = orch_args.tensor(2).ref();

    uint32_t tile_shape[1] = {TILE_ELEMS};
    TensorCreateInfo p_ci(tile_shape, 1, DataType::FLOAT32);

    for (int batch = 0; batch < BATCH; batch++) {
        for (int m_idx = 0; m_idx < GRID_M; m_idx++) {
            for (int n_idx = 0; n_idx < GRID_N; n_idx++) {
                for (int k_idx = 0; k_idx < GRID_K; k_idx++) {
                    uint32_t a_off[1] = {
                        static_cast<uint32_t>((batch * GRID_M * GRID_K + m_idx * GRID_K + k_idx) * TILE_ELEMS)
                    };
                    uint32_t b_off[1] = {
                        static_cast<uint32_t>((batch * GRID_K * GRID_N + k_idx * GRID_N + n_idx) * TILE_ELEMS)
                    };
                    uint32_t c_off[1] = {
                        static_cast<uint32_t>((batch * GRID_M * GRID_N + m_idx * GRID_N + n_idx) * TILE_ELEMS)
                    };

                    simpler::hbg::Tensor a_view = a.view(tile_shape, a_off);
                    simpler::hbg::Tensor b_view = b.view(tile_shape, b_off);
                    simpler::hbg::Tensor c_view = c.view(tile_shape, c_off);

                    // P_k = A[m,k] @ B[k,n]
                    CoreTaskArgs p_gemm;
                    p_gemm.add_input(a_view);
                    p_gemm.add_input(b_view);
                    p_gemm.add_output(p_ci);
                    TaskOutputTensors p_out = rt_submit_aic_task(FUNC_GEMM, p_gemm);
                    simpler::hbg::Tensor p = p_out.get_ref(0);

                    // C[m,n] += P_k
                    CoreTaskArgs p_add;
                    p_add.add_inout(c_view);
                    p_add.add_input(p);
                    rt_submit_aiv_task(FUNC_TILE_ADD, p_add);
                }
            }
        }
    }

    LOG_INFO("[bgemm_orch] Submitted tiled C = A @ B");
}

}  // extern "C"
