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
// Kernel Function: x_gamma0

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
#include "intrinsic.h"

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

static __aicore__ void x_gamma0(__gm__ bfloat16_t *v1, __gm__ float *v2, __gm__ float *v3, int32_t v4, int32_t v5) {
    SaturationMode v6 = SaturationMode::OFF;
    RoundMode v7 = RoundMode::CAST_ROUND;
    const int64_t v8 = 256;
    const int64_t v9 = 2;
    const int64_t v10 = 4;
    const int64_t v11 = 1024;
    const int64_t v12 = 1;
    const int64_t v13 = 5120;
    const int64_t v14 = 16;
    const int64_t v15 = 33792;
    const int64_t v16 = 17408;
    const int64_t v17 = 16384;
    const int64_t v18 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    int64_t v19 = (int64_t)((uint64_t)((int64_t)v4) * (uint64_t)v11);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    for (size_t v20 = (size_t)v18; v20 < ((size_t)v10); v20 += (size_t)v9) {
        int64_t v21 = (int64_t)((uint64_t)((int64_t)v20) * (uint64_t)v8);
        int64_t v22 = (int64_t)((uint64_t)v19 + (uint64_t)v21);
        int64_t v23 = (int64_t)((uint64_t)v19 + (uint64_t)((int64_t)(uint64_t)v21 + (uint64_t)v8));
        Tile<
            TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v24 = Tile<
                TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v14, v8);
        uint64_t v25 = (uint64_t)v18;
        TASSIGN(v24, v25);
        pto::Shape<1, 1, 1, 16, 256> v26 = pto::Shape<1, 1, 1, 16, 256>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v27 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v28 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v2 + (v18 + v18 * v13 + v22 * v12), v26, v27
            );
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        TLOAD(v24, v28);
        Tile<
            TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v29 = Tile<
                TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v8);
        uint64_t v30 = (uint64_t)v17;
        TASSIGN(v29, v30);
        pto::Shape<1, 1, 1, 1, 256> v31 = pto::Shape<1, 1, 1, 1, 256>();
        pto::Stride<5120, 5120, 5120, 5120, 1> v32 = pto::Stride<5120, 5120, 5120, 5120, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND> v33 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND>(
                v3 + (v18 + v18 * v13 + v22 * v12), v31, v32
            );
        TLOAD(v29, v33);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        Tile<
            TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v34 = Tile<
                TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v14, v8);
        uint64_t v35 = (uint64_t)v16;
        TASSIGN(v34, v35);
        pto::Shape<1, 1, 1, 16, 256> v36 = pto::Shape<1, 1, 1, 16, 256>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v37 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v38 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v2 + (v18 + v18 * v13 + v23 * v12), v36, v37
            );
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        TLOAD(v34, v38);
        Tile<
            TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v39 = Tile<
                TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v8);
        uint64_t v40 = (uint64_t)v15;
        TASSIGN(v39, v40);
        pto::Shape<1, 1, 1, 1, 256> v41 = pto::Shape<1, 1, 1, 1, 256>();
        pto::Stride<5120, 5120, 5120, 5120, 1> v42 = pto::Stride<5120, 5120, 5120, 5120, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND> v43 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND>(
                v3 + (v18 + v18 * v13 + v23 * v12), v41, v42
            );
        TLOAD(v39, v43);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        Tile<
            TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v44 = Tile<
                TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v14, v8);
        uint64_t v45 = (uint64_t)v18;
        TASSIGN(v44, v45);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TCOLEXPANDMUL(v44, v24, v29);
        Tile<
            TileType::Vec, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v46 = Tile<
                TileType::Vec, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v14, v8);
        uint64_t v47 = (uint64_t)v18;
        TASSIGN(v46, v47);
        pipe_barrier(PIPE_V);
        TCVT(v46, v44, v7, v6);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pto::Shape<1, 1, 1, 16, 256> v48 = pto::Shape<1, 1, 1, 16, 256>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v49 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v50 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v1 + (v18 + v18 * v13 + v22 * v12), v48, v49
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v50, v46);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        Tile<
            TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v51 = Tile<
                TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v14, v8);
        uint64_t v52 = (uint64_t)v16;
        TASSIGN(v51, v52);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TCOLEXPANDMUL(v51, v34, v39);
        Tile<
            TileType::Vec, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v53 = Tile<
                TileType::Vec, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v14, v8);
        uint64_t v54 = (uint64_t)v16;
        TASSIGN(v53, v54);
        pipe_barrier(PIPE_V);
        TCVT(v53, v51, v7, v6);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        pto::Shape<1, 1, 1, 16, 256> v55 = pto::Shape<1, 1, 1, 16, 256>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v56 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v57 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v1 + (v18 + v18 * v13 + v23 * v12), v55, v56
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v57, v53);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: normed__ssa_v0
    __gm__ Tensor *normed__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ bfloat16_t *normed__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(normed__ssa_v0_tensor->buffer.addr) + normed__ssa_v0_tensor->start_offset;

    // Unpack tensor: cur__rv_v2
    __gm__ Tensor *cur__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *cur__rv_v2 =
        reinterpret_cast<__gm__ float *>(cur__rv_v2_tensor->buffer.addr) + cur__rv_v2_tensor->start_offset;

    // Unpack tensor: input_rms_weight__ssa_v0
    __gm__ Tensor *input_rms_weight__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *input_rms_weight__ssa_v0 =
        reinterpret_cast<__gm__ float *>(input_rms_weight__ssa_v0_tensor->buffer.addr) +
        input_rms_weight__ssa_v0_tensor->start_offset;

    // Forward to ptoas-generated function
    x_gamma0(normed__ssa_v0, cur__rv_v2, input_rms_weight__ssa_v0, __pypto_spmd_block_idx, __pypto_spmd_block_num);
}
