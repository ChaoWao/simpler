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
// Kernel Function: out_proj_0

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

static __aicore__ void out_proj_0(
    __gm__ bfloat16_t *v1, __gm__ bfloat16_t *v2, __gm__ float *v3, int64_t v4, int64_t v5, int32_t v6, int32_t v7
) {
    const int64_t v8 = 960;
    const int64_t v9 = 2;
    const int64_t v10 = 15;
    const int64_t v11 = 32;
    const int64_t v12 = 64;
    const int64_t v13 = 512;
    const int64_t v14 = 5;
    const int64_t v15 = 1;
    const int64_t v16 = 5120;
    const int64_t v17 = 16;
    const int64_t v18 = 1024;
    const int64_t v19 = 32768;
    const int64_t v20 = 3072;
    const int64_t v21 = 2048;
    const int64_t v22 = 0;
    const int64_t v23 = 69632;
    const int64_t v24 = 4096;
    using T = float;

#if defined(__DAV_CUBE__)
    int64_t v25 = (int64_t)((uint64_t)v4 + (uint64_t)((int64_t)v6));
    int64_t v26 = (int64_t)((uint64_t)(v25 / v14) * (uint64_t)v13);
    int64_t v27 = (int64_t)((uint64_t)(v25 % v14) * (uint64_t)v18);
    Tile<
        TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v28 = Tile<
            TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v17, v12);
    uint64_t v29 = (uint64_t)v24;
    TASSIGN(v28, v29);
    pto::Shape<1, 1, 1, 16, 64> v30 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v31 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
        v32 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
            v1 + (v22 + v22 * v16 + v27 * v15), v30, v31
        );
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    TLOAD(v28, v32);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    Tile<
        TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v33 = Tile<
            TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v12, v13);
    uint64_t v34 = (uint64_t)v23;
    TASSIGN(v33, v34);
    int64_t v35 = (int64_t)((uint64_t)v5 + (uint64_t)v27);
    pto::Shape<1, 1, 1, 64, 512> v36 = pto::Shape<1, 1, 1, 64, 512>();
    pto::Stride<327680, 327680, 327680, 5120, 1> v37 = pto::Stride<327680, 327680, 327680, 5120, 1>();
    GlobalTensor<
        bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<327680, 327680, 327680, 5120, 1>, pto::Layout::ND>
        v38 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<327680, 327680, 327680, 5120, 1>, pto::Layout::ND>(
            v2 + (v22 + v35 * v16 + v26 * v15), v36, v37
        );
    TLOAD(v33, v38);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    Tile<
        TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v39 = Tile<
            TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v17, v11);
    uint64_t v40 = (uint64_t)v21;
    TASSIGN(v39, v40);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TEXTRACT(v39, v28, v22, v22);
    Tile<
        TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v41 = Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v11, v13);
    uint64_t v42 = (uint64_t)v22;
    TASSIGN(v41, v42);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TEXTRACT(v41, v33, v22, v22);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    Tile<
        TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v43 = Tile<
            TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v17, v11);
    uint64_t v44 = (uint64_t)v20;
    TASSIGN(v43, v44);
    TEXTRACT(v43, v28, v22, v11);
    Tile<
        TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v45 = Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v11, v13);
    uint64_t v46 = (uint64_t)v19;
    TASSIGN(v45, v46);
    TEXTRACT(v45, v33, v11, v22);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    Tile<
        TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v47 = Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v17, v13);
    uint64_t v48 = (uint64_t)v22;
    TASSIGN(v47, v48);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(v47, v39, v41);
    Tile<
        TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v49 = Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v17, v13);
    uint64_t v50 = (uint64_t)v22;
    TASSIGN(v49, v50);
    pipe_barrier(PIPE_M);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    TMATMUL_ACC(v49, v49, v43, v45);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    for (size_t v51 = (size_t)v15; v51 < ((size_t)v10); v51 += (size_t)v9) {
        int64_t v52 = (int64_t)((uint64_t)((int64_t)v51) * (uint64_t)v12);
        int64_t v53 = (int64_t)((uint64_t)v52 + (uint64_t)v12);
        Tile<
            TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v54 = Tile<
                TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v17, v12);
        uint64_t v55 = (uint64_t)v22;
        TASSIGN(v54, v55);
        pto::Shape<1, 1, 1, 16, 64> v56 = pto::Shape<1, 1, 1, 16, 64>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v57 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v58 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v1 + (v22 + v22 * v16 + (int64_t)((uint64_t)v27 + (uint64_t)v52) * v15), v56, v57
            );
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        TLOAD(v54, v58);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        Tile<
            TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v59 = Tile<
                TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v12, v13);
        uint64_t v60 = (uint64_t)v23;
        TASSIGN(v59, v60);
        pto::Shape<1, 1, 1, 64, 512> v61 = pto::Shape<1, 1, 1, 64, 512>();
        pto::Stride<327680, 327680, 327680, 5120, 1> v62 = pto::Stride<327680, 327680, 327680, 5120, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<327680, 327680, 327680, 5120, 1>, pto::Layout::ND>
            v63 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<327680, 327680, 327680, 5120, 1>,
                pto::Layout::ND>(v2 + (v22 + (int64_t)((uint64_t)v35 + (uint64_t)v52) * v16 + v26 * v15), v61, v62);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        TLOAD(v59, v63);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        Tile<
            TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v64 = Tile<
                TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v17, v12);
        uint64_t v65 = (uint64_t)v21;
        TASSIGN(v64, v65);
        pto::Shape<1, 1, 1, 16, 64> v66 = pto::Shape<1, 1, 1, 16, 64>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v67 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v68 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v1 + (v22 + v22 * v16 + (int64_t)((uint64_t)v27 + (uint64_t)v53) * v15), v66, v67
            );
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        TLOAD(v64, v68);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
        Tile<
            TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v69 = Tile<
                TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v12, v13);
        uint64_t v70 = (uint64_t)v24;
        TASSIGN(v69, v70);
        pto::Shape<1, 1, 1, 64, 512> v71 = pto::Shape<1, 1, 1, 64, 512>();
        pto::Stride<327680, 327680, 327680, 5120, 1> v72 = pto::Stride<327680, 327680, 327680, 5120, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<327680, 327680, 327680, 5120, 1>, pto::Layout::ND>
            v73 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<327680, 327680, 327680, 5120, 1>,
                pto::Layout::ND>(v2 + (v22 + (int64_t)((uint64_t)v35 + (uint64_t)v53) * v16 + v26 * v15), v71, v72);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
        TLOAD(v69, v73);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
        Tile<
            TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v74 = Tile<
                TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        uint64_t v75 = (uint64_t)v21;
        TASSIGN(v74, v75);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        TEXTRACT(v74, v54, v22, v22);
        Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v76 = Tile<
                TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v11, v13);
        uint64_t v77 = (uint64_t)v22;
        TASSIGN(v76, v77);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v76, v59, v22, v22);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        Tile<
            TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v78 = Tile<
                TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        uint64_t v79 = (uint64_t)v20;
        TASSIGN(v78, v79);
        TEXTRACT(v78, v54, v22, v11);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v80 = Tile<
                TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v11, v13);
        uint64_t v81 = (uint64_t)v19;
        TASSIGN(v80, v81);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        TEXTRACT(v80, v59, v11, v22);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v82 = Tile<
                TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v17, v13);
        uint64_t v83 = (uint64_t)v22;
        TASSIGN(v82, v83);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v82, v82, v74, v76);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v84 = Tile<
                TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v17, v13);
        uint64_t v85 = (uint64_t)v22;
        TASSIGN(v84, v85);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        TMATMUL_ACC(v84, v84, v78, v80);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        Tile<
            TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v86 = Tile<
                TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        uint64_t v87 = (uint64_t)v22;
        TASSIGN(v86, v87);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
        TEXTRACT(v86, v64, v22, v22);
        Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v88 = Tile<
                TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v11, v13);
        uint64_t v89 = (uint64_t)v22;
        TASSIGN(v88, v89);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
        TEXTRACT(v88, v69, v22, v22);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
        Tile<
            TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v90 = Tile<
                TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        uint64_t v91 = (uint64_t)v18;
        TASSIGN(v90, v91);
        TEXTRACT(v90, v64, v22, v11);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v92 = Tile<
                TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v11, v13);
        uint64_t v93 = (uint64_t)v19;
        TASSIGN(v92, v93);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        TEXTRACT(v92, v69, v11, v22);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
        Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v94 = Tile<
                TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v17, v13);
        uint64_t v95 = (uint64_t)v22;
        TASSIGN(v94, v95);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v94, v94, v86, v88);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v96 = Tile<
                TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v17, v13);
        uint64_t v97 = (uint64_t)v22;
        TASSIGN(v96, v97);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
        TMATMUL_ACC(v96, v96, v90, v92);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    }
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
    Tile<
        TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v98 = Tile<
            TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v17, v12);
    uint64_t v99 = (uint64_t)v24;
    TASSIGN(v98, v99);
    pto::Shape<1, 1, 1, 16, 64> v100 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v101 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
        v102 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
            v1 + (v22 + v22 * v16 + (int64_t)((uint64_t)v27 + (uint64_t)v8) * v15), v100, v101
        );
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
    TLOAD(v98, v102);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
    Tile<
        TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v103 = Tile<
            TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v12, v13);
    uint64_t v104 = (uint64_t)v23;
    TASSIGN(v103, v104);
    pto::Shape<1, 1, 1, 64, 512> v105 = pto::Shape<1, 1, 1, 64, 512>();
    pto::Stride<327680, 327680, 327680, 5120, 1> v106 = pto::Stride<327680, 327680, 327680, 5120, 1>();
    GlobalTensor<
        bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<327680, 327680, 327680, 5120, 1>, pto::Layout::ND>
        v107 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<327680, 327680, 327680, 5120, 1>, pto::Layout::ND>(
            v2 + (v22 + (int64_t)((uint64_t)v35 + (uint64_t)v8) * v16 + v26 * v15), v105, v106
        );
    TLOAD(v103, v107);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
    Tile<
        TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v108 = Tile<
            TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v17, v11);
    uint64_t v109 = (uint64_t)v21;
    TASSIGN(v108, v109);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
    TEXTRACT(v108, v98, v22, v22);
    Tile<
        TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v110 = Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v11, v13);
    uint64_t v111 = (uint64_t)v22;
    TASSIGN(v110, v111);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    TEXTRACT(v110, v103, v22, v22);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
    Tile<
        TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v112 = Tile<
            TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v17, v11);
    uint64_t v113 = (uint64_t)v20;
    TASSIGN(v112, v113);
    TEXTRACT(v112, v98, v22, v11);
    Tile<
        TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v114 = Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v11, v13);
    uint64_t v115 = (uint64_t)v19;
    TASSIGN(v114, v115);
    TEXTRACT(v114, v103, v11, v22);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
    Tile<
        TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v116 = Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v17, v13);
    uint64_t v117 = (uint64_t)v22;
    TASSIGN(v116, v117);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
    pipe_barrier(PIPE_M);
    TMATMUL_ACC(v116, v116, v108, v110);
    Tile<
        TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v118 = Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v17, v13);
    uint64_t v119 = (uint64_t)v22;
    TASSIGN(v118, v119);
    pipe_barrier(PIPE_M);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
    TMATMUL_ACC(v118, v118, v112, v114);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    Tile<
        TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v120 = Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v17, v13);
    uint64_t v121 = (uint64_t)v22;
    TASSIGN(v120, v121);
    pto::Shape<1, 1, 1, 16, 512> v122 = pto::Shape<1, 1, 1, 16, 512>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v123 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v124 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
            v3 + (v22 + v22 * v16 + v26 * v15), v122, v123
        );
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE<
        Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>,
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>,
        AtomicType::AtomicAdd>(v124, v120);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
#endif  // __DAV_CUBE__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: attn_out_inline282__ssa_v4
    __gm__ Tensor *attn_out_inline282__ssa_v4_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ bfloat16_t *attn_out_inline282__ssa_v4 =
        reinterpret_cast<__gm__ bfloat16_t *>(attn_out_inline282__ssa_v4_tensor->buffer.addr) +
        attn_out_inline282__ssa_v4_tensor->start_offset;

    // Unpack tensor: wo__ssa_v0
    __gm__ Tensor *wo__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ bfloat16_t *wo__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(wo__ssa_v0_tensor->buffer.addr) + wo__ssa_v0_tensor->start_offset;

    // Unpack tensor: attn_proj_fp32_inline220__rv_v5
    __gm__ Tensor *attn_proj_fp32_inline220__rv_v5_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *attn_proj_fp32_inline220__rv_v5 =
        reinterpret_cast<__gm__ float *>(attn_proj_fp32_inline220__rv_v5_tensor->buffer.addr) +
        attn_proj_fp32_inline220__rv_v5_tensor->start_offset;

    // Unpack scalar: N_OUT_DIRECT_inline61__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } N_OUT_DIRECT_inline61__ssa_v0_conv;
    N_OUT_DIRECT_inline61__ssa_v0_conv.u64 = args[3];
    int64_t N_OUT_DIRECT_inline61__ssa_v0 = N_OUT_DIRECT_inline61__ssa_v0_conv.val;

    // Unpack scalar: layer_hidden_base_inline151__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } layer_hidden_base_inline151__ssa_v0_conv;
    layer_hidden_base_inline151__ssa_v0_conv.u64 = args[4];
    int64_t layer_hidden_base_inline151__ssa_v0 = layer_hidden_base_inline151__ssa_v0_conv.val;

    // Forward to ptoas-generated function
    out_proj_0(
        attn_out_inline282__ssa_v4, wo__ssa_v0, attn_proj_fp32_inline220__rv_v5, N_OUT_DIRECT_inline61__ssa_v0,
        layer_hidden_base_inline151__ssa_v0, __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
