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
// Kernel Function: v_proj

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

static __aicore__ void
v_proj(__gm__ float *v1, __gm__ bfloat16_t *v2, __gm__ bfloat16_t *v3, int64_t v4, int32_t v5, int32_t v6) {
    const int64_t v7 = 768;
    const int64_t v8 = 64;
    const int64_t v9 = 128;
    const int64_t v10 = 256;
    const int64_t v11 = 2;
    const int64_t v12 = 512;
    const int64_t v13 = 5;
    const int64_t v14 = 5120;
    const int64_t v15 = 1;
    const int64_t v16 = 1024;
    const int64_t v17 = 16;
    const int64_t v18 = 2048;
    const int64_t v19 = 8192;
    const int64_t v20 = 32768;
    const int64_t v21 = 6144;
    const int64_t v22 = 4096;
    const int64_t v23 = 0;
    const int64_t v24 = 147456;
    const int64_t v25 = 16384;
    using T = float;

#if defined(__DAV_CUBE__)
    size_t v26 = (size_t)v23;
    size_t v27 = (size_t)v10;
    size_t v28 = (size_t)v9;
    int64_t v29 = (int64_t)v5;
    int64_t v30 = (int64_t)((uint64_t)(v29 % v13) * (uint64_t)v16);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    for (size_t v31 = v26; v31 < ((size_t)v11); v31 += (size_t)v15) {
        int64_t v32 = (int64_t)((uint64_t)((int64_t)(uint64_t)(v29 / v13) * (uint64_t)v12) +
                                (uint64_t)((int64_t)(uint64_t)((int64_t)v31) * (uint64_t)v10));
        Tile<
            TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v33 = Tile<
                TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v17, v10);
        uint64_t v34 = (uint64_t)v25;
        TASSIGN(v33, v34);
        pto::Shape<1, 1, 1, 16, 256> v35 = pto::Shape<1, 1, 1, 16, 256>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v36 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v37 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v2 + (v23 + v23 * v14 + v30 * v15), v35, v36
            );
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        TLOAD(v33, v37);
        Tile<
            TileType::Mat, bfloat16_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v38 = Tile<
                TileType::Mat, bfloat16_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v10, v10);
        uint64_t v39 = (uint64_t)v24;
        TASSIGN(v38, v39);
        pto::Shape<1, 1, 1, 256, 256> v40 = pto::Shape<1, 1, 1, 256, 256>();
        pto::Stride<262144, 262144, 262144, 1024, 1> v41 = pto::Stride<262144, 262144, 262144, 1024, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<262144, 262144, 262144, 1024, 1>, pto::Layout::ND>
            v42 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<262144, 262144, 262144, 1024, 1>,
                pto::Layout::ND>(v3 + (v23 + (int64_t)((uint64_t)v4 + (uint64_t)v30) * v16 + v32 * v15), v40, v41);
        TLOAD(v38, v42);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        for (size_t v43 = v26; v43 < v27; v43 += v28) {
            int64_t v44 = (int64_t)v43;
            Tile<
                TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v45 = Tile<
                    TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v17, v8);
            uint64_t v46 = (uint64_t)v22;
            TASSIGN(v45, v46);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            pipe_barrier(PIPE_MTE1);
            TEXTRACT(v45, v33, v23, v43);
            Tile<
                TileType::Right, bfloat16_t, 64, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v47 = Tile<
                    TileType::Right, bfloat16_t, 64, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v8, v10);
            uint64_t v48 = (uint64_t)v23;
            TASSIGN(v47, v48);
            TEXTRACT(v47, v38, v43, v23);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            Tile<
                TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v49 = Tile<
                    TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v17, v8);
            uint64_t v50 = (uint64_t)v21;
            TASSIGN(v49, v50);
            int64_t v51 = (int64_t)((uint64_t)v44 + (uint64_t)v8);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
            TEXTRACT(v49, v33, v23, v51);
            Tile<
                TileType::Right, bfloat16_t, 64, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v52 = Tile<
                    TileType::Right, bfloat16_t, 64, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v8, v10);
            uint64_t v53 = (uint64_t)v20;
            TASSIGN(v52, v53);
            TEXTRACT(v52, v38, v51, v23);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            if (v44 == v23) {
                Tile<
                    TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v54 = Tile<
                        TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v17, v10);
                uint64_t v55 = (uint64_t)v23;
                TASSIGN(v54, v55);
                pipe_barrier(PIPE_M);
                TMATMUL(v54, v45, v47);
            } else {
                Tile<
                    TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v56 = Tile<
                        TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v17, v10);
                uint64_t v57 = (uint64_t)v23;
                TASSIGN(v56, v57);
                pipe_barrier(PIPE_M);
                TMATMUL_ACC(v56, v56, v45, v47);
            };
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            Tile<
                TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v58 = Tile<
                    TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v17, v10);
            uint64_t v59 = (uint64_t)v23;
            TASSIGN(v58, v59);
            pipe_barrier(PIPE_M);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
            TMATMUL_ACC(v58, v58, v49, v52);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        };
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        int64_t v60 = (int64_t)((uint64_t)v30 + (uint64_t)v10);
        int64_t v61 = (int64_t)((uint64_t)v30 + (uint64_t)v12);
        Tile<
            TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v62 = Tile<
                TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v17, v10);
        uint64_t v63 = (uint64_t)v23;
        TASSIGN(v62, v63);
        pto::Shape<1, 1, 1, 16, 256> v64 = pto::Shape<1, 1, 1, 16, 256>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v65 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v66 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v2 + (v23 + v23 * v14 + v60 * v15), v64, v65
            );
        TLOAD(v62, v66);
        Tile<
            TileType::Mat, bfloat16_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v67 = Tile<
                TileType::Mat, bfloat16_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v10, v10);
        uint64_t v68 = (uint64_t)v24;
        TASSIGN(v67, v68);
        pto::Shape<1, 1, 1, 256, 256> v69 = pto::Shape<1, 1, 1, 256, 256>();
        pto::Stride<262144, 262144, 262144, 1024, 1> v70 = pto::Stride<262144, 262144, 262144, 1024, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<262144, 262144, 262144, 1024, 1>, pto::Layout::ND>
            v71 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<262144, 262144, 262144, 1024, 1>,
                pto::Layout::ND>(v3 + (v23 + (int64_t)((uint64_t)v4 + (uint64_t)v60) * v16 + v32 * v15), v69, v70);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        TLOAD(v67, v71);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        Tile<
            TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v72 = Tile<
                TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v17, v10);
        uint64_t v73 = (uint64_t)v19;
        TASSIGN(v72, v73);
        pto::Shape<1, 1, 1, 16, 256> v74 = pto::Shape<1, 1, 1, 16, 256>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v75 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v76 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v2 + (v23 + v23 * v14 + v61 * v15), v74, v75
            );
        TLOAD(v72, v76);
        Tile<
            TileType::Mat, bfloat16_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v77 = Tile<
                TileType::Mat, bfloat16_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v10, v10);
        uint64_t v78 = (uint64_t)v25;
        TASSIGN(v77, v78);
        pto::Shape<1, 1, 1, 256, 256> v79 = pto::Shape<1, 1, 1, 256, 256>();
        pto::Stride<262144, 262144, 262144, 1024, 1> v80 = pto::Stride<262144, 262144, 262144, 1024, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<262144, 262144, 262144, 1024, 1>, pto::Layout::ND>
            v81 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<262144, 262144, 262144, 1024, 1>,
                pto::Layout::ND>(v3 + (v23 + (int64_t)((uint64_t)v4 + (uint64_t)v61) * v16 + v32 * v15), v79, v80);
        TLOAD(v77, v81);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        for (size_t v82 = v26; v82 < v27; v82 += v28) {
            Tile<
                TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v83 = Tile<
                    TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v17, v8);
            uint64_t v84 = (uint64_t)v22;
            TASSIGN(v83, v84);
            pipe_barrier(PIPE_MTE1);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
            TEXTRACT(v83, v62, v23, v82);
            Tile<
                TileType::Right, bfloat16_t, 64, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v85 = Tile<
                    TileType::Right, bfloat16_t, 64, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v8, v10);
            uint64_t v86 = (uint64_t)v23;
            TASSIGN(v85, v86);
            TEXTRACT(v85, v67, v82, v23);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
            Tile<
                TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v87 = Tile<
                    TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v17, v8);
            uint64_t v88 = (uint64_t)v21;
            TASSIGN(v87, v88);
            int64_t v89 = (int64_t)((uint64_t)((int64_t)v82) + (uint64_t)v8);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
            TEXTRACT(v87, v62, v23, v89);
            Tile<
                TileType::Right, bfloat16_t, 64, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v90 = Tile<
                    TileType::Right, bfloat16_t, 64, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v8, v10);
            uint64_t v91 = (uint64_t)v20;
            TASSIGN(v90, v91);
            TEXTRACT(v90, v67, v89, v23);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
            Tile<
                TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v92 = Tile<
                    TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v17, v10);
            uint64_t v93 = (uint64_t)v23;
            TASSIGN(v92, v93);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v92, v92, v83, v85);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
            Tile<
                TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v94 = Tile<
                    TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v17, v10);
            uint64_t v95 = (uint64_t)v23;
            TASSIGN(v94, v95);
            pipe_barrier(PIPE_M);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
            TMATMUL_ACC(v94, v94, v87, v90);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
        };
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
        for (size_t v96 = v26; v96 < v27; v96 += v28) {
            Tile<
                TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v97 = Tile<
                    TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v17, v8);
            uint64_t v98 = (uint64_t)v23;
            TASSIGN(v97, v98);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
            TEXTRACT(v97, v72, v23, v96);
            Tile<
                TileType::Right, bfloat16_t, 64, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v99 = Tile<
                    TileType::Right, bfloat16_t, 64, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v8, v10);
            uint64_t v100 = (uint64_t)v23;
            TASSIGN(v99, v100);
            TEXTRACT(v99, v77, v96, v23);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
            Tile<
                TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v101 = Tile<
                    TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v17, v8);
            uint64_t v102 = (uint64_t)v18;
            TASSIGN(v101, v102);
            int64_t v103 = (int64_t)((uint64_t)((int64_t)v96) + (uint64_t)v8);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            TEXTRACT(v101, v72, v23, v103);
            Tile<
                TileType::Right, bfloat16_t, 64, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v104 = Tile<
                    TileType::Right, bfloat16_t, 64, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v8, v10);
            uint64_t v105 = (uint64_t)v20;
            TASSIGN(v104, v105);
            TEXTRACT(v104, v77, v103, v23);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
            Tile<
                TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v106 = Tile<
                    TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v17, v10);
            uint64_t v107 = (uint64_t)v23;
            TASSIGN(v106, v107);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v106, v106, v97, v99);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
            Tile<
                TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v108 = Tile<
                    TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v17, v10);
            uint64_t v109 = (uint64_t)v23;
            TASSIGN(v108, v109);
            pipe_barrier(PIPE_M);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
            TMATMUL_ACC(v108, v108, v101, v104);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        };
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        int64_t v110 = (int64_t)((uint64_t)v30 + (uint64_t)v7);
        Tile<
            TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v111 = Tile<
                TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v17, v10);
        uint64_t v112 = (uint64_t)v25;
        TASSIGN(v111, v112);
        pto::Shape<1, 1, 1, 16, 256> v113 = pto::Shape<1, 1, 1, 16, 256>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v114 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v115 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v2 + (v23 + v23 * v14 + v110 * v15), v113, v114
            );
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        TLOAD(v111, v115);
        Tile<
            TileType::Mat, bfloat16_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v116 = Tile<
                TileType::Mat, bfloat16_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v10, v10);
        uint64_t v117 = (uint64_t)v24;
        TASSIGN(v116, v117);
        pto::Shape<1, 1, 1, 256, 256> v118 = pto::Shape<1, 1, 1, 256, 256>();
        pto::Stride<262144, 262144, 262144, 1024, 1> v119 = pto::Stride<262144, 262144, 262144, 1024, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<262144, 262144, 262144, 1024, 1>, pto::Layout::ND>
            v120 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<262144, 262144, 262144, 1024, 1>,
                pto::Layout::ND>(v3 + (v23 + (int64_t)((uint64_t)v4 + (uint64_t)v110) * v16 + v32 * v15), v118, v119);
        TLOAD(v116, v120);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        for (size_t v121 = v26; v121 < v27; v121 += v28) {
            Tile<
                TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v122 = Tile<
                    TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v17, v8);
            uint64_t v123 = (uint64_t)v22;
            TASSIGN(v122, v123);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            TEXTRACT(v122, v111, v23, v121);
            Tile<
                TileType::Right, bfloat16_t, 64, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v124 = Tile<
                    TileType::Right, bfloat16_t, 64, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v8, v10);
            uint64_t v125 = (uint64_t)v23;
            TASSIGN(v124, v125);
            TEXTRACT(v124, v116, v121, v23);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
            Tile<
                TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v126 = Tile<
                    TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v17, v8);
            uint64_t v127 = (uint64_t)v21;
            TASSIGN(v126, v127);
            int64_t v128 = (int64_t)((uint64_t)((int64_t)v121) + (uint64_t)v8);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
            TEXTRACT(v126, v111, v23, v128);
            Tile<
                TileType::Right, bfloat16_t, 64, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v129 = Tile<
                    TileType::Right, bfloat16_t, 64, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v8, v10);
            uint64_t v130 = (uint64_t)v20;
            TASSIGN(v129, v130);
            TEXTRACT(v129, v116, v128, v23);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
            Tile<
                TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v131 = Tile<
                    TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v17, v10);
            uint64_t v132 = (uint64_t)v23;
            TASSIGN(v131, v132);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v131, v131, v122, v124);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            Tile<
                TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v133 = Tile<
                    TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v17, v10);
            uint64_t v134 = (uint64_t)v23;
            TASSIGN(v133, v134);
            pipe_barrier(PIPE_M);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
            TMATMUL_ACC(v133, v133, v126, v129);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        };
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        Tile<
            TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v135 = Tile<
                TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v17, v10);
        uint64_t v136 = (uint64_t)v23;
        TASSIGN(v135, v136);
        pto::Shape<1, 1, 1, 16, 256> v137 = pto::Shape<1, 1, 1, 16, 256>();
        pto::Stride<16384, 16384, 16384, 1024, 1> v138 = pto::Stride<16384, 16384, 16384, 1024, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<16384, 16384, 16384, 1024, 1>, pto::Layout::ND>
            v139 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<16384, 16384, 16384, 1024, 1>, pto::Layout::ND>(
                v1 + (v23 + v23 * v16 + v32 * v15), v137, v138
            );
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        pipe_barrier(PIPE_FIX);
        TSTORE<
            Tile<
                TileType::Acc, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>,
            GlobalTensor<
                float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<16384, 16384, 16384, 1024, 1>, pto::Layout::ND>,
            AtomicType::AtomicAdd>(v139, v135);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    }
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
#endif  // __DAV_CUBE__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: v_proj_inline255__ssa_v1
    __gm__ Tensor *v_proj_inline255__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *v_proj_inline255__ssa_v1 =
        reinterpret_cast<__gm__ float *>(v_proj_inline255__ssa_v1_tensor->buffer.addr) +
        v_proj_inline255__ssa_v1_tensor->start_offset;

    // Unpack tensor: normed__iter_v4
    __gm__ Tensor *normed__iter_v4_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ bfloat16_t *normed__iter_v4 = reinterpret_cast<__gm__ bfloat16_t *>(normed__iter_v4_tensor->buffer.addr) +
                                         normed__iter_v4_tensor->start_offset;

    // Unpack tensor: wv__ssa_v0
    __gm__ Tensor *wv__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ bfloat16_t *wv__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(wv__ssa_v0_tensor->buffer.addr) + wv__ssa_v0_tensor->start_offset;

    // Unpack scalar: layer_hidden_base_inline151__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } layer_hidden_base_inline151__ssa_v0_conv;
    layer_hidden_base_inline151__ssa_v0_conv.u64 = args[3];
    int64_t layer_hidden_base_inline151__ssa_v0 = layer_hidden_base_inline151__ssa_v0_conv.val;

    // Forward to ptoas-generated function
    v_proj(
        v_proj_inline255__ssa_v1, normed__iter_v4, wv__ssa_v0, layer_hidden_base_inline151__ssa_v0,
        __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
