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
 * MXFP8 / MXFP4 matmul via pto-isa ``TMATMUL_MX`` (A5 Cube, onboard only).
 *
 * Fixed tile: M=128, K=64, N=64 (K aligned to 64; scale K = K/32 = 2).
 * mode (args[5]):
 *   0 — MXFP8: A/B float8_e4m3 [M,K]/[K,N]; As/Bs float8_e8m0 ZZ/NN
 *   1 — MXFP4: A/B float4_e2m1x2 packed GM (logical [M,K]/[K,N]); same scales
 *
 * Pattern mirrors pto-isa ``tmatmul_mx_kernel.cpp`` RunTMATMULMX (no bias).
 * a5sim is not supported: CPU stub TLOAD lacks MX_A_ZZ / MX_B_NN.
 *
 * Args: [A, As, B, Bs, C, mode]
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/npu/a5/utils.hpp>

#include "tensor.h"

using namespace pto;

#include "pipe_sync.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2) {
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

template <typename T>
AICORE constexpr inline T CeilDiv(T num_1, T num_2) {
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2;
}

template <typename AType, typename BType, int validM, int validK, int validN, bool isFp4>
static __aicore__ void matmul_mx_impl(
    __gm__ AType *src0, __gm__ float8_e8m0_t *src2, __gm__ BType *src1, __gm__ float8_e8m0_t *src3, __gm__ float *out
) {
    constexpr int blockAlign = isFp4 ? 64 : 32;
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int kAlign = CeilAlign<int>(validK, 64);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr uint8_t kMX = CeilDiv(kAlign, 32);

    using ScaleType = float8_e8m0_t;
    using OutType = float;

    using GlobalDataSrc0 = GlobalTensor<
        AType, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<
        BType, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;

    using MxShapeA = TileShape2D<ScaleType, M, kMX, Layout::MX_A_ZZ>;
    using MxStrideA = BaseShape2D<ScaleType, M, kMX, Layout::MX_A_ZZ>;
    using GlobalDataSrc2 = GlobalTensor<ScaleType, MxShapeA, MxStrideA, Layout::MX_A_ZZ>;

    using MxShapeB = TileShape2D<ScaleType, kMX, N, Layout::MX_B_NN>;
    using MxStrideB = BaseShape2D<ScaleType, kMX, N, Layout::MX_B_NN>;
    using GlobalDataSrc3 = GlobalTensor<ScaleType, MxShapeB, MxStrideB, Layout::MX_B_NN>;

    using GlobalDataOut = GlobalTensor<
        OutType, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataSrc3 src3Global(src3);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, AType, M, kAlign, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData =
        Tile<TileType::Mat, BType, kAlign, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TileScaleAData =
        Tile<TileType::Mat, ScaleType, M, kMX, BLayout::RowMajor, validM, kMX, SLayout::RowMajor, 32>;
    using TileScaleBData =
        Tile<TileType::Mat, ScaleType, kMX, N, BLayout::ColMajor, kMX, validN, SLayout::ColMajor, 32>;

    using LeftTile = TileLeft<AType, M, kAlign, validM, kAlign>;
    using RightTile = TileRight<BType, kAlign, N, kAlign, validN>;
    using LeftScaleTile = TileLeftScale<ScaleType, M, kMX, validM, kMX>;
    using RightScaleTile = TileRightScale<ScaleType, kMX, N, kMX, validN>;
    using AccTile = TileAcc<OutType, M, N, validM, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileScaleAData aScaleMatTile;
    TileScaleBData bScaleMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, M * kAlign);
    TASSIGN(aScaleMatTile, M * kAlign + kAlign * N);
    TASSIGN(bScaleMatTile, M * kAlign + kAlign * N + M * kMX);

    LeftTile aTile;
    RightTile bTile;
    LeftScaleTile aScaleTile;
    RightScaleTile bScaleTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    uint64_t scaleAAddr = pto::GetScaleAddr(aTile.data());
    uint64_t scaleBAddr = pto::GetScaleAddr(bTile.data());
    TASSIGN(aScaleTile, scaleAAddr);
    TASSIGN(bScaleTile, scaleBAddr);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    if constexpr (kAlign - validK >= blockAlign) {
        TFILLPAD(aMatTile, aMatTile);
    }
    TFILLPAD(bMatTile, bMatTile);

    TLOAD<TileScaleAData, GlobalDataSrc2>(aScaleMatTile, src2Global);
    TLOAD<TileScaleBData, GlobalDataSrc3>(bScaleMatTile, src3Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    TMOV(aScaleTile, aScaleMatTile);
    TMOV(bScaleTile, bScaleMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL_MX(cTile, aTile, aScaleTile, bTile, bScaleTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TSTORE(dstGlobal, cTile);
    pipe_sync();
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *a_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *as_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *b_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *bs_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ Tensor *c_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    const int mode = static_cast<int>(args[5]);

    __gm__ float8_e8m0_t *as =
        reinterpret_cast<__gm__ float8_e8m0_t *>(as_tensor->buffer.addr) + as_tensor->start_offset;
    __gm__ float8_e8m0_t *bs =
        reinterpret_cast<__gm__ float8_e8m0_t *>(bs_tensor->buffer.addr) + bs_tensor->start_offset;
    __gm__ float *c = reinterpret_cast<__gm__ float *>(c_tensor->buffer.addr) + c_tensor->start_offset;

    if (mode == 0) {
        __gm__ float8_e4m3_t *a =
            reinterpret_cast<__gm__ float8_e4m3_t *>(a_tensor->buffer.addr) + a_tensor->start_offset;
        __gm__ float8_e4m3_t *b =
            reinterpret_cast<__gm__ float8_e4m3_t *>(b_tensor->buffer.addr) + b_tensor->start_offset;
        matmul_mx_impl<float8_e4m3_t, float8_e4m3_t, 128, 64, 64, false>(a, as, b, bs, c);
    } else {
        __gm__ float4_e2m1x2_t *a =
            reinterpret_cast<__gm__ float4_e2m1x2_t *>(a_tensor->buffer.addr) + a_tensor->start_offset;
        __gm__ float4_e2m1x2_t *b =
            reinterpret_cast<__gm__ float4_e2m1x2_t *>(b_tensor->buffer.addr) + b_tensor->start_offset;
        matmul_mx_impl<float4_e2m1x2_t, float4_e2m1x2_t, 128, 64, 64, true>(a, as, b, bs, c);
    }
}
