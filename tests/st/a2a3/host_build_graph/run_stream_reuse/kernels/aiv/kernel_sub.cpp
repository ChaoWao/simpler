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

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"

using namespace pto;

#include "pipe_sync.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ ChipTensor *src0_tensor = reinterpret_cast<__gm__ ChipTensor *>(args[0]);
    __gm__ ChipTensor *src1_tensor = reinterpret_cast<__gm__ ChipTensor *>(args[1]);
    __gm__ ChipTensor *out_tensor = reinterpret_cast<__gm__ ChipTensor *>(args[2]);

    __gm__ float *src0 = reinterpret_cast<__gm__ float *>(src0_tensor->buffer.addr) + src0_tensor->start_offset;
    __gm__ float *src1 = reinterpret_cast<__gm__ float *>(src1_tensor->buffer.addr) + src1_tensor->start_offset;
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    constexpr int kRows = 128;
    constexpr int kCols = 128;
    using Shape5D = Shape<1, 1, 1, kRows, kCols>;
    using Stride5D = Stride<1, 1, 1, kCols, 1>;
    using GlobalData = GlobalTensor<float, Shape5D, Stride5D>;
    using TileData = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;

    TileData src0_tile(kRows, kCols);
    TileData src1_tile(kRows, kCols);
    TileData dst_tile(kRows, kCols);
    TASSIGN(src0_tile, 0x0);
    TASSIGN(src1_tile, 0x10000);
    TASSIGN(dst_tile, 0x20000);

    GlobalData src0_global(src0);
    GlobalData src1_global(src1);
    GlobalData dst_global(out);
    TLOAD(src0_tile, src0_global);
    TLOAD(src1_tile, src1_global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TSUB(dst_tile, src0_tile, src1_tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst_global, dst_tile);
    pipe_sync();
}
