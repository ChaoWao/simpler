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
 * Tile-based Element-wise Addition Kernel (Vector Core) - INOUT Pattern
 *
 * Computes: C_tile = C_tile + P (tile_size x tile_size tile accumulation)
 *
 * Args (ChipTensor*):
 *   args[0] = C_tile (INOUT: read + write accumulator)
 *   args[1] = P      (INPUT: matmul result to accumulate)
 *
 * The A5 Case0 kernel uses 128 x 128 tiles and derives the tile count from the
 * ChipTensor view created by orchestration.
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

#include "tensor.h"

#include "pipe_sync.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

static __aicore__ inline int get_num_tiles(__gm__ ChipTensor *tensor, uint64_t tile_elems) {
    uint64_t total_elems = tensor->shapes[0];
    return static_cast<int>(total_elems / tile_elems);
}

template <int TILE>
static __aicore__ void tile_add_impl(__gm__ float *c_ptr, __gm__ float *p_ptr) {
    using DynShapeDim5 = pto::Shape<1, 1, 1, TILE, TILE>;
    using DynStridDim5 = pto::Stride<1, 1, 1, TILE, 1>;
    using GlobalData = pto::GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = pto::Tile<pto::TileType::Vec, float, TILE, TILE, pto::BLayout::RowMajor, -1, -1>;

    TileData cTile(TILE, TILE);
    TileData pTile(TILE, TILE);
    TileData outTile(TILE, TILE);
    TASSIGN(cTile, 0x0);
    TASSIGN(pTile, 0x10000);
    TASSIGN(outTile, 0x20000);

    GlobalData cGlobal(c_ptr);
    GlobalData pGlobal(p_ptr);
    GlobalData outGlobal(c_ptr);

    TLOAD(cTile, cGlobal);
    TLOAD(pTile, pGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(outTile, cTile, pTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(outGlobal, outTile);
    pipe_sync();
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ ChipTensor *c_tensor = reinterpret_cast<__gm__ ChipTensor *>(args[0]);
    __gm__ ChipTensor *p_tensor = reinterpret_cast<__gm__ ChipTensor *>(args[1]);
    constexpr uint64_t TILE_ELEMS = 128 * 128;
    int num_tiles = get_num_tiles(c_tensor, TILE_ELEMS);

    __gm__ float *base_c = reinterpret_cast<__gm__ float *>(c_tensor->buffer.addr) + c_tensor->start_offset;
    __gm__ float *base_p = reinterpret_cast<__gm__ float *>(p_tensor->buffer.addr) + p_tensor->start_offset;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        __gm__ float *c_ptr = base_c + (tile_idx * TILE_ELEMS);
        __gm__ float *p_ptr = base_p + (tile_idx * TILE_ELEMS);

        tile_add_impl<128>(c_ptr, p_ptr);
    }
}
