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
// Kernel Function: lm_head_matmul

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

#if defined(__CPU_SIM)
// PTOAS v0.50+ emits cache_line_t::ENTIRE_DATA_CACHE / SINGLE_CACHE_LINE as
// scoped identifiers, but the pto-isa cpu_stub.hpp defines them as bare macros
// (#define ENTIRE_DATA_CACHE 0) — which breaks cache_line_t::ENTIRE_DATA_CACHE
// into cache_line_t::0. Undefine the macros and provide proper namespace-scoped
// constexpr constants. The same headers also #define dcci/dsb as macros that
// would expand our own inlines, so undefine + redefine all of them here.
#include <atomic>

// Forward-declare the overloads so the undefs below don't break chained includes.
namespace pypto_sim_detail {
template <typename... Args>
static inline void sim_dcci(Args...);  // defined after the undefs
static inline void sim_dsb(int kind);  // ditto
}  // namespace pypto_sim_detail

// Undefine conflicting macros from pto-isa cpu_stub.hpp / inner_kernel.h
// so our namespace-scoped constants and inline functions are used instead.
#undef ENTIRE_DATA_CACHE
#undef SINGLE_CACHE_LINE
#undef DSB_DDR
#undef dcci
#undef dsb
#undef CACHELINE_OUT

namespace cache_line_t {
constexpr int ENTIRE_DATA_CACHE = 0;
constexpr int SINGLE_CACHE_LINE = 0;
constexpr int CACHELINE_OUT = 0;
}  // namespace cache_line_t
typedef int mem_dsb_t;
#define DSB_DDR 0

static inline void dcci(...) { std::atomic_thread_fence(std::memory_order_seq_cst); }
static inline void dsb(mem_dsb_t) { std::atomic_thread_fence(std::memory_order_seq_cst); }
#endif  // __CPU_SIM

using namespace pto;

// --- ptoas-generated code ---

template <typename TensorT>
static __aicore__ inline auto PTOAS__GLOBAL_TENSOR_DATA(TensorT &tensor) -> decltype(tensor.data()) {
    return tensor.data();
}

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

template <typename Ptr>
static __aicore__ inline void PTOAS__DCCI_SINGLE_CACHE_LINE(Ptr ptr) {
    dcci((__gm__ void *)ptr, cache_line_t::SINGLE_CACHE_LINE);
}

static __aicore__ void
lm_head_matmul(__gm__ float *v1, __gm__ bfloat16_t *v2, __gm__ bfloat16_t *v3, int32_t v4, int32_t v5) {
    const int64_t v6 = 3840;
    unsigned v7 = 3840;
    const int64_t v8 = 2;
    const int64_t v9 = 15;
    const int64_t v10 = 256;
    const int64_t v11 = 128;
    const int64_t v12 = 24;
    const int64_t v13 = 505;
    const int64_t v14 = 1;
    const int64_t v15 = 16;
    const int64_t v16 = 4096;
    const int64_t v17 = 32768;
    const int64_t v18 = 12288;
    const int64_t v19 = 8192;
    const int64_t v20 = 0;
    const int64_t v21 = 81920;
    const int64_t v22 = 16384;
    using T = float;

#if defined(__DAV_CUBE__)
    // pto: %lm_core_inline36_inline13329__ssa_v0
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    for (int64_t i23 = (int64_t)v4; i23 < v13; i23 += v12) {
        // pto: %26
        int64_t v24 = (int64_t)((uint64_t)i23 * (uint64_t)v11);
        // pto: %mm_hidden0_inline37_inline13286__tile
        Tile<
            TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v25 = Tile<
                TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v10);
        // pto: %mm_hidden0_inline37_inline13286__tile
        uint64_t v26 = (uint64_t)v22;
        TASSIGN(v25, v26);
        // pto: %owner_hiddens_inline50_inline13292__ssa_v1_pview
        pto::Shape<1, 1, 1, 16, 256> v27 = pto::Shape<1, 1, 1, 16, 256>();
        // pto: %owner_hiddens_inline50_inline13292__ssa_v1_pview
        pto::Stride<65536, 65536, 65536, 4096, 1> v28 = pto::Stride<65536, 65536, 65536, 4096, 1>();
        // pto: %owner_hiddens_inline50_inline13292__ssa_v1_pview
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>
            v29 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(
                v2, v27, v28
            );
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        TLOAD(v25, v29);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        // pto: %mm_weight0_inline52_inline13279__tile
        Tile<
            TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v30 = Tile<
                TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v11, v10);
        // pto: %mm_weight0_inline52_inline13279__tile
        uint64_t v31 = (uint64_t)v21;
        TASSIGN(v30, v31);
        // pto: %27
        int64_t v32 = v24 < v20 ? v20 : v24;
        // pto: %lm_head_weight__ssa_v0_pview
        pto::Shape<1, 1, 1, 128, 256> v33 = pto::Shape<1, 1, 1, 128, 256>();
        // pto: %lm_head_weight__ssa_v0_pview
        pto::Stride<524288, 524288, 524288, 4096, 1> v34 = pto::Stride<524288, 524288, 524288, 4096, 1>();
        // pto: %lm_head_weight__ssa_v0_pview
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>, pto::Layout::ND>
            v35 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>,
                pto::Layout::ND>(v3 + (v20 + v32 * v16), v33, v34);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        TLOAD(v30, v35);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        // pto: %mm_weight0_inline52_inline13279__tile_t
        Tile<
            TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v36 = Tile<
                TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v10, v11);
        // pto: %mm_weight0_inline52_inline13279__tile_t
        uint64_t v37 = (uint64_t)v21;
        TASSIGN(v36, v37);
        // pto: %mm_acc_inline47_inline13315__tile_l0_init
        Tile<
            TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v38 = Tile<
                TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %mm_acc_inline47_inline13315__tile_l0_init
        uint64_t v39 = (uint64_t)v20;
        TASSIGN(v38, v39);
        // pto: %mm_acc_inline47_inline13315__tile_l0_a
        Tile<
            TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v40 = Tile<
                TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %mm_acc_inline47_inline13315__tile_l0_a
        uint64_t v41 = (uint64_t)v19;
        TASSIGN(v40, v41);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        TEXTRACT(v40, v25, v20, v20);
        // pto: %mm_acc_inline47_inline13315__tile_l0_b
        Tile<
            TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v42 = Tile<
                TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>(v11, v11);
        // pto: %mm_acc_inline47_inline13315__tile_l0_b
        uint64_t v43 = (uint64_t)v20;
        TASSIGN(v42, v43);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v42, v36, v20, v20);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        // pto: %0
        Tile<
            TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v44 = Tile<
                TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %0
        uint64_t v45 = (uint64_t)v18;
        TASSIGN(v44, v45);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v44, v25, v20, v11);
        // pto: %1
        Tile<
            TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v46 = Tile<
                TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>(v11, v11);
        // pto: %1
        uint64_t v47 = (uint64_t)v17;
        TASSIGN(v46, v47);
        TEXTRACT(v46, v36, v11, v20);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        // pto: %mm_acc_inline47_inline13315__tile_l0_c_first
        Tile<
            TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v48 = Tile<
                TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %mm_acc_inline47_inline13315__tile_l0_c_first
        uint64_t v49 = (uint64_t)v20;
        TASSIGN(v48, v49);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        TMATMUL(v48, v40, v42);
        // pto: %mm_acc_inline47_inline13315__tile_l0_c_acc
        Tile<
            TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v50 = Tile<
                TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %mm_acc_inline47_inline13315__tile_l0_c_acc
        uint64_t v51 = (uint64_t)v20;
        TASSIGN(v50, v51);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        TMATMUL_ACC(v50, v50, v44, v46);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        for (int64_t j52 = v14; j52 < v9; j52 += v8) {
            // pto: %28
            int64_t v53 = (int64_t)((uint64_t)j52 * (uint64_t)v10);
            // pto: %30
            int64_t v54 = (int64_t)((uint64_t)v53 + (uint64_t)v10);
            // pto: %mm_hidden_tile_inline54_inline13297__tile
            Tile<
                TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v55 = Tile<
                    TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v10);
            // pto: %mm_hidden_tile_inline54_inline13297__tile
            uint64_t v56 = (uint64_t)v20;
            TASSIGN(v55, v56);
            // pto: %31
            int64_t v57 = v53 < v20 ? v20 : v53;
            // pto: %32
            pto::Shape<1, 1, 1, 16, 256> v58 = pto::Shape<1, 1, 1, 16, 256>();
            // pto: %32
            pto::Stride<65536, 65536, 65536, 4096, 1> v59 = pto::Stride<65536, 65536, 65536, 4096, 1>();
            // pto: %32
            GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>
                v60 = GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<65536, 65536, 65536, 4096, 1>,
                    pto::Layout::ND>(v2 + (v20 + v57), v58, v59);
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
            TLOAD(v55, v60);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
            // pto: %mm_weight_tile_inline23_inline13282__tile
            Tile<
                TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v61 = Tile<
                    TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v11, v10);
            // pto: %mm_weight_tile_inline23_inline13282__tile
            uint64_t v62 = (uint64_t)v21;
            TASSIGN(v61, v62);
            // pto: %35
            pto::Shape<1, 1, 1, 128, 256> v63 = pto::Shape<1, 1, 1, 128, 256>();
            // pto: %35
            pto::Stride<524288, 524288, 524288, 4096, 1> v64 = pto::Stride<524288, 524288, 524288, 4096, 1>();
            // pto: %35
            GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>,
                pto::Layout::ND>
                v65 = GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>,
                    pto::Layout::ND>(v3 + ((v20 + v32 * v16) + v57), v63, v64);
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
            TLOAD(v61, v65);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
            // pto: %2
            Tile<
                TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v66 = Tile<
                    TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v10);
            // pto: %2
            uint64_t v67 = (uint64_t)v19;
            TASSIGN(v66, v67);
            // pto: %36
            int64_t v68 = v54 < v20 ? v20 : v54;
            // pto: %37
            pto::Shape<1, 1, 1, 16, 256> v69 = pto::Shape<1, 1, 1, 16, 256>();
            // pto: %37
            pto::Stride<65536, 65536, 65536, 4096, 1> v70 = pto::Stride<65536, 65536, 65536, 4096, 1>();
            // pto: %37
            GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>
                v71 = GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<65536, 65536, 65536, 4096, 1>,
                    pto::Layout::ND>(v2 + (v20 + v68), v69, v70);
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
            TLOAD(v66, v71);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
            // pto: %3
            Tile<
                TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v72 = Tile<
                    TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v11, v10);
            // pto: %3
            uint64_t v73 = (uint64_t)v22;
            TASSIGN(v72, v73);
            // pto: %40
            pto::Shape<1, 1, 1, 128, 256> v74 = pto::Shape<1, 1, 1, 128, 256>();
            // pto: %40
            pto::Stride<524288, 524288, 524288, 4096, 1> v75 = pto::Stride<524288, 524288, 524288, 4096, 1>();
            // pto: %40
            GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>,
                pto::Layout::ND>
                v76 = GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>,
                    pto::Layout::ND>(v3 + ((v20 + v32 * v16) + v68), v74, v75);
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
            TLOAD(v72, v76);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
            // pto: %mm_weight_tile_inline23_inline13282__tile_t
            Tile<
                TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v77 = Tile<
                    TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v10, v11);
            // pto: %mm_weight_tile_inline23_inline13282__tile_t
            uint64_t v78 = (uint64_t)v21;
            TASSIGN(v77, v78);
            // pto: %4
            Tile<
                TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v79 = Tile<
                    TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v11);
            // pto: %4
            uint64_t v80 = (uint64_t)v19;
            TASSIGN(v79, v80);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
            TEXTRACT(v79, v55, v20, v20);
            // pto: %5
            Tile<
                TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>
                v81 = Tile<
                    TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v11, v11);
            // pto: %5
            uint64_t v82 = (uint64_t)v20;
            TASSIGN(v81, v82);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
            TEXTRACT(v81, v77, v20, v20);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
            // pto: %6
            Tile<
                TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v83 = Tile<
                    TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v11);
            // pto: %6
            uint64_t v84 = (uint64_t)v18;
            TASSIGN(v83, v84);
            TEXTRACT(v83, v55, v20, v11);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
            // pto: %7
            Tile<
                TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>
                v85 = Tile<
                    TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v11, v11);
            // pto: %7
            uint64_t v86 = (uint64_t)v17;
            TASSIGN(v85, v86);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
            TEXTRACT(v85, v77, v11, v20);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
            // pto: %8
            Tile<
                TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v87 = Tile<
                    TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v15, v11);
            // pto: %8
            uint64_t v88 = (uint64_t)v20;
            TASSIGN(v87, v88);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v87, v87, v79, v81);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
            // pto: %9
            Tile<
                TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v89 = Tile<
                    TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v15, v11);
            // pto: %9
            uint64_t v90 = (uint64_t)v20;
            TASSIGN(v89, v90);
            pipe_barrier(PIPE_M);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
            TMATMUL_ACC(v89, v89, v83, v85);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
            // pto: %10
            Tile<
                TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v91 = Tile<
                    TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v10, v11);
            // pto: %10
            uint64_t v92 = (uint64_t)v22;
            TASSIGN(v91, v92);
            // pto: %11
            Tile<
                TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v93 = Tile<
                    TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v11);
            // pto: %11
            uint64_t v94 = (uint64_t)v20;
            TASSIGN(v93, v94);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
            TEXTRACT(v93, v66, v20, v20);
            // pto: %12
            Tile<
                TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>
                v95 = Tile<
                    TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v11, v11);
            // pto: %12
            uint64_t v96 = (uint64_t)v20;
            TASSIGN(v95, v96);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
            TEXTRACT(v95, v91, v20, v20);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
            // pto: %13
            Tile<
                TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v97 = Tile<
                    TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v11);
            // pto: %13
            uint64_t v98 = (uint64_t)v16;
            TASSIGN(v97, v98);
            TEXTRACT(v97, v66, v20, v11);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
            // pto: %14
            Tile<
                TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>
                v99 = Tile<
                    TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v11, v11);
            // pto: %14
            uint64_t v100 = (uint64_t)v17;
            TASSIGN(v99, v100);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
            TEXTRACT(v99, v91, v11, v20);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
            // pto: %15
            Tile<
                TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v101 = Tile<
                    TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v15, v11);
            // pto: %15
            uint64_t v102 = (uint64_t)v20;
            TASSIGN(v101, v102);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v101, v101, v93, v95);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
            // pto: %16
            Tile<
                TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v103 = Tile<
                    TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v15, v11);
            // pto: %16
            uint64_t v104 = (uint64_t)v20;
            TASSIGN(v103, v104);
            pipe_barrier(PIPE_M);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
            TMATMUL_ACC(v103, v103, v97, v99);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        }
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
        // pto: %17
        Tile<
            TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v105 = Tile<
                TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v10);
        // pto: %17
        uint64_t v106 = (uint64_t)v22;
        TASSIGN(v105, v106);
        // pto: %42
        pto::Shape<1, 1, 1, 16, 256> v107 = pto::Shape<1, 1, 1, 16, 256>();
        // pto: %42
        pto::Stride<65536, 65536, 65536, 4096, 1> v108 = pto::Stride<65536, 65536, 65536, 4096, 1>();
        // pto: %42
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>
            v109 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(
                v2 + v7, v107, v108
            );
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
        TLOAD(v105, v109);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
        // pto: %18
        Tile<
            TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v110 = Tile<
                TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v11, v10);
        // pto: %18
        uint64_t v111 = (uint64_t)v21;
        TASSIGN(v110, v111);
        // pto: %45
        pto::Shape<1, 1, 1, 128, 256> v112 = pto::Shape<1, 1, 1, 128, 256>();
        // pto: %45
        pto::Stride<524288, 524288, 524288, 4096, 1> v113 = pto::Stride<524288, 524288, 524288, 4096, 1>();
        // pto: %45
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>, pto::Layout::ND>
            v114 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>,
                pto::Layout::ND>(v3 + (v6 + v32 * v16), v112, v113);
        TLOAD(v110, v114);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
        // pto: %19
        Tile<
            TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v115 = Tile<
                TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v10, v11);
        // pto: %19
        uint64_t v116 = (uint64_t)v21;
        TASSIGN(v115, v116);
        // pto: %20
        Tile<
            TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v117 = Tile<
                TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %20
        uint64_t v118 = (uint64_t)v19;
        TASSIGN(v117, v118);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
        TEXTRACT(v117, v105, v20, v20);
        // pto: %21
        Tile<
            TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v119 = Tile<
                TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>(v11, v11);
        // pto: %21
        uint64_t v120 = (uint64_t)v20;
        TASSIGN(v119, v120);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
        TEXTRACT(v119, v115, v20, v20);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
        // pto: %22
        Tile<
            TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v121 = Tile<
                TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %22
        uint64_t v122 = (uint64_t)v18;
        TASSIGN(v121, v122);
        TEXTRACT(v121, v105, v20, v11);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        // pto: %23
        Tile<
            TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v123 = Tile<
                TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>(v11, v11);
        // pto: %23
        uint64_t v124 = (uint64_t)v17;
        TASSIGN(v123, v124);
        TEXTRACT(v123, v115, v11, v20);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
        // pto: %24
        Tile<
            TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v125 = Tile<
                TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %24
        uint64_t v126 = (uint64_t)v20;
        TASSIGN(v125, v126);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v125, v125, v117, v119);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        // pto: %25
        Tile<
            TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v127 = Tile<
                TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %25
        uint64_t v128 = (uint64_t)v20;
        TASSIGN(v127, v128);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
        TMATMUL_ACC(v127, v127, v121, v123);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        // pto: %mm_acc_inline47_inline13315__rv_v2
        Tile<
            TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v129 = Tile<
                TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %mm_acc_inline47_inline13315__rv_v2
        uint64_t v130 = (uint64_t)v20;
        TASSIGN(v129, v130);
        // pto: %logits_shards_inline40_inline13289__iter_v1_pview
        pto::Shape<1, 1, 1, 16, 128> v131 = pto::Shape<1, 1, 1, 16, 128>();
        // pto: %logits_shards_inline40_inline13289__iter_v1_pview
        pto::Stride<1034240, 1034240, 1034240, 64640, 1> v132 = pto::Stride<1034240, 1034240, 1034240, 64640, 1>();
        // pto: %logits_shards_inline40_inline13289__iter_v1_pview
        GlobalTensor<
            float, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<1034240, 1034240, 1034240, 64640, 1>, pto::Layout::ND>
            v133 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<1034240, 1034240, 1034240, 64640, 1>, pto::Layout::ND>(
                v1 + (v20 + v32), v131, v132
            );
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        TSTORE(v133, v129);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    }
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
#endif  // __DAV_CUBE__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: logits_shards_inline40_inline13289__ssa_v0
    __gm__ Tensor *logits_shards_inline40_inline13289__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *logits_shards_inline40_inline13289__ssa_v0 =
        reinterpret_cast<__gm__ float *>(logits_shards_inline40_inline13289__ssa_v0_tensor->buffer.addr) +
        logits_shards_inline40_inline13289__ssa_v0_tensor->start_offset;

    // Unpack tensor: owner_hiddens_inline50_inline13292__ssa_v1
    __gm__ Tensor *owner_hiddens_inline50_inline13292__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ bfloat16_t *owner_hiddens_inline50_inline13292__ssa_v1 =
        reinterpret_cast<__gm__ bfloat16_t *>(owner_hiddens_inline50_inline13292__ssa_v1_tensor->buffer.addr) +
        owner_hiddens_inline50_inline13292__ssa_v1_tensor->start_offset;

    // Unpack tensor: lm_head_weight__ssa_v0
    __gm__ Tensor *lm_head_weight__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ bfloat16_t *lm_head_weight__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(lm_head_weight__ssa_v0_tensor->buffer.addr) +
        lm_head_weight__ssa_v0_tensor->start_offset;

    // Forward to ptoas-generated function
    lm_head_matmul(
        logits_shards_inline40_inline13289__ssa_v0, owner_hiddens_inline50_inline13292__ssa_v1, lm_head_weight__ssa_v0,
        __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
