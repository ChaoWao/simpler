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
// Kernel Function: kv_seed

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#if defined(__CPU_SIM)
#define __aicore__
#else
#define __aicore__ [aicore]
#endif
#endif

#include <pto/pto-inst.hpp>
#include "tensor.h"

using namespace pto;

// --- ptoas-generated code ---

enum class PTOAutoSyncTailMode : int {
    kBarrierAll = 0,
    kSetWaitMte3ToSEvent0 = 1,
};

static __aicore__ inline void ptoas_auto_sync_tail(PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
    switch (mode) {
    case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        break;
    case PTOAutoSyncTailMode::kBarrierAll:
    default:
        pipe_barrier(PIPE_ALL);
        break;
    }
}

static __aicore__ void kv_seed(__gm__ float *v1, __gm__ float *v2) {
    const float v3 = 0.0f;
    const int64_t v4 = 1;
    const int64_t v5 = 1024;
    const int64_t v6 = 16;
    const int64_t v7 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    Tile<
        TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v8 = Tile<
            TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v6, v5);
    uint64_t v9 = (uint64_t)v7;
    TASSIGN(v8, v9);
    TEXPANDS(v8, v3);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 1024> v10 = pto::Shape<1, 1, 1, 16, 1024>();
    pto::Stride<16384, 16384, 16384, 1024, 1> v11 = pto::Stride<16384, 16384, 16384, 1024, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<16384, 16384, 16384, 1024, 1>, pto::Layout::ND> v12 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<16384, 16384, 16384, 1024, 1>, pto::Layout::ND>(
            v1 + (v7 + v7 * v5 + v7 * v4), v10, v11
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v12, v8);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    Tile<
        TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v13 = Tile<
            TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v6, v5);
    uint64_t v14 = (uint64_t)v7;
    TASSIGN(v13, v14);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(v13, v3);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    pto::Shape<1, 1, 1, 16, 1024> v15 = pto::Shape<1, 1, 1, 16, 1024>();
    pto::Stride<16384, 16384, 16384, 1024, 1> v16 = pto::Stride<16384, 16384, 16384, 1024, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<16384, 16384, 16384, 1024, 1>, pto::Layout::ND> v17 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<16384, 16384, 16384, 1024, 1>, pto::Layout::ND>(
            v2 + (v7 + v7 * v5 + v7 * v4), v15, v16
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v17, v13);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Unpack tensor: k_proj_inline135__ssa_v0
    __gm__ Tensor *k_proj_inline135__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *k_proj_inline135__ssa_v0 =
        reinterpret_cast<__gm__ float *>(k_proj_inline135__ssa_v0_tensor->buffer.addr) +
        k_proj_inline135__ssa_v0_tensor->start_offset;

    // Unpack tensor: v_proj_inline255__ssa_v0
    __gm__ Tensor *v_proj_inline255__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *v_proj_inline255__ssa_v0 =
        reinterpret_cast<__gm__ float *>(v_proj_inline255__ssa_v0_tensor->buffer.addr) +
        v_proj_inline255__ssa_v0_tensor->start_offset;

    // Forward to ptoas-generated function
    kv_seed(k_proj_inline135__ssa_v0, v_proj_inline255__ssa_v0);
}
