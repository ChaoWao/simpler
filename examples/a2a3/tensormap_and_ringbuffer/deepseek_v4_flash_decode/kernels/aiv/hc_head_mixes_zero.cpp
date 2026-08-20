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
// Kernel Function: hc_head_mixes_zero

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

static __aicore__ void hc_head_mixes_zero(__gm__ float *mixes_raw, int64_t total_rows, int32_t block_idx) {
    constexpr float zero = 0.0f;
    constexpr int64_t rows_per_block = 16;
    constexpr int64_t cols = 16;
    int64_t row_base = static_cast<int64_t>(block_idx) * rows_per_block;
    int64_t remaining_rows = total_rows - row_base;
    int64_t valid_rows = remaining_rows < rows_per_block ? remaining_rows : rows_per_block;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);

    Tile<
        TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        zero_tile = Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(valid_rows, cols);
    TASSIGN(zero_tile, 0);
    TEXPANDS(zero_tile, zero);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    pto::Shape<1, 1, 1, -1, 16> shape = pto::Shape<1, 1, 1, -1, 16>(1, 1, 1, valid_rows, cols);
    pto::Stride<256, 256, 256, 16, 1> stride = pto::Stride<256, 256, 256, 16, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> output =
        GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(
            mixes_raw + row_base * cols, shape, stride
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(output, zero_tile);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    int32_t block_idx = get_block_idx(args);
    __gm__ ChipTensor *mixes_raw_tensor = reinterpret_cast<__gm__ ChipTensor *>(args[0]);
    __gm__ float *mixes_raw =
        reinterpret_cast<__gm__ float *>(mixes_raw_tensor->buffer.addr) + mixes_raw_tensor->start_offset;
    int64_t total_rows = static_cast<int64_t>(mixes_raw_tensor->shapes[0]);
    hc_head_mixes_zero(mixes_raw, total_rows, block_idx);
}
