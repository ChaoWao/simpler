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
// Kernel Function: exp_w2_mm_3

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
exp_w2_mm_3(__gm__ int32_t *v1, __gm__ int8_t *v2, __gm__ int8_t *v3, int64_t v4, int32_t v5, int32_t v6) {
    const int64_t v7 = 128;
    const int64_t v8 = 512;
    const int64_t v9 = 256;
    const int64_t v10 = 4;
    const int64_t v11 = 1024;
    const int64_t v12 = 1;
    const int64_t v13 = 16;
    const int64_t v14 = 4096;
    const int64_t v15 = 2048;
    const int64_t v16 = 32768;
    const int64_t v17 = 6144;
    const int64_t v18 = 147456;
    const int64_t v19 = 139264;
    const int64_t v20 = 131072;
    const int64_t v21 = 0;
    using T = float;

#if defined(__DAV_CUBE__)
    // pto: %y_acc_inline2733_inline12985__phi_v5
    Tile<
        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v22 = Tile<
            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v13, v9);
    // pto: %y_acc_inline2733_inline12985__phi_v5
    uint64_t v23 = (uint64_t)v21;
    TASSIGN(v22, v23);
    // pto: %y_acc_inline2733_inline12985__tile_l0_c_phi
    Tile<
        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v24 = Tile<
            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v13, v9);
    // pto: %y_acc_inline2733_inline12985__tile_l0_c_phi
    uint64_t v25 = (uint64_t)v21;
    TASSIGN(v24, v25);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
    for (int64_t i26 = v21; i26 < v10; i26 += v12) {
        // pto: %wb_idx_inline2736_inline13078__ssa_v0, %16, %18, %17
        int64_t v27 = (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)v5) * (uint64_t)v11)) +
                                (uint64_t)((int64_t)((uint64_t)i26 * (uint64_t)v9)));
        // pto: %y_acc_inline2733_inline12985__tile
        Tile<
            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v28 = Tile<
                TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v13, v9);
        // pto: %y_acc_inline2733_inline12985__tile
        uint64_t v29 = (uint64_t)v21;
        TASSIGN(v28, v29);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        for (int64_t j30 = v21; j30 < v15; j30 += v11) {
            // pto: %h_k_inline2732_inline12919__tile
            Tile<
                TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v31 = Tile<
                    TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>(v13, v8);
            // pto: %h_k_inline2732_inline12919__tile
            uint64_t v32 = (uint64_t)v20;
            TASSIGN(v31, v32);
            // pto: %19
            int64_t v33 = j30 < v21 ? v21 : j30;
            // pto: %h_tile_i8_inline2806_inline13011__ssa_v4_pview
            pto::Shape<1, 1, 1, 16, 512> v34 = pto::Shape<1, 1, 1, 16, 512>();
            // pto: %h_tile_i8_inline2806_inline13011__ssa_v4_pview
            pto::Stride<32768, 32768, 32768, 2048, 1> v35 = pto::Stride<32768, 32768, 32768, 2048, 1>();
            // pto: %h_tile_i8_inline2806_inline13011__ssa_v4_pview
            GlobalTensor<
                int8_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND>
                v36 = GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND>(
                    v2 + (v21 + v33), v34, v35
                );
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
            TLOAD(v31, v36);
            // pto: %0
            Tile<
                TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v37 = Tile<
                    TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>(v13, v8);
            // pto: %0
            uint64_t v38 = (uint64_t)v19;
            TASSIGN(v37, v38);
            // pto: %20
            int64_t v39 = (int64_t)((uint64_t)j30 + (uint64_t)v8);
            // pto: %21
            int64_t v40 = v39 < v21 ? v21 : v39;
            // pto: %22
            pto::Shape<1, 1, 1, 16, 512> v41 = pto::Shape<1, 1, 1, 16, 512>();
            // pto: %22
            pto::Stride<32768, 32768, 32768, 2048, 1> v42 = pto::Stride<32768, 32768, 32768, 2048, 1>();
            // pto: %22
            GlobalTensor<
                int8_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND>
                v43 = GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND>(
                    v2 + (v21 + v40), v41, v42
                );
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
            TLOAD(v37, v43);
            // pto: %w2_k_inline2731_inline12882__tile
            Tile<
                TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v44 = Tile<
                    TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>(v9, v8);
            // pto: %w2_k_inline2731_inline12882__tile
            uint64_t v45 = (uint64_t)v18;
            TASSIGN(v44, v45);
            // pto: %23, %24
            int64_t v46 = (int64_t)((uint64_t)((int64_t)((uint64_t)v4 * (uint64_t)v14)) + (uint64_t)v27);
            // pto: %25
            int64_t v47 = v46 < v21 ? v21 : v46;
            // pto: %w2_k_inline2731_inline12882__tile_view2d_pview
            pto::Shape<1, 1, 1, 256, 512> v48 = pto::Shape<1, 1, 1, 256, 512>();
            // pto: %w2_k_inline2731_inline12882__tile_view2d_pview
            pto::Stride<524288, 524288, 524288, 2048, 1> v49 = pto::Stride<524288, 524288, 524288, 2048, 1>();
            // pto: %w2_k_inline2731_inline12882__tile_view2d_pview
            GlobalTensor<
                int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<524288, 524288, 524288, 2048, 1>, pto::Layout::ND>
                v50 = GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<524288, 524288, 524288, 2048, 1>,
                    pto::Layout::ND>(v3 + ((v21 + v47 * v15) + v33), v48, v49);
            TLOAD(v44, v50);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            // pto: %27
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            if (j30 == v21) {
                // pto: %w2_k_inline2731_inline12882__tile_t
                Tile<
                    TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v51 = Tile<
                        TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v8, v9);
                // pto: %w2_k_inline2731_inline12882__tile_t
                uint64_t v52 = (uint64_t)v18;
                TASSIGN(v51, v52);
                // pto: %y_acc_inline2733_inline12985__tile_l0_init
                Tile<
                    TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v53 = Tile<
                        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v13, v9);
                // pto: %y_acc_inline2733_inline12985__tile_l0_init
                uint64_t v54 = (uint64_t)v21;
                TASSIGN(v53, v54);
                for (int64_t k55 = v21; k55 < v8; k55 += v9) {
                    // pto: %y_acc_inline2733_inline12985__tile_l0_a
                    Tile<
                        TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v56 = Tile<
                            TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                            PadValue::Null, CompactMode::Null>(v13, v7);
                    // pto: %y_acc_inline2733_inline12985__tile_l0_a
                    uint64_t v57 = (uint64_t)v17;
                    TASSIGN(v56, v57);
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
                    pipe_barrier(PIPE_MTE1);
                    TEXTRACT(v56, v31, v21, k55);
                    // pto: %y_acc_inline2733_inline12985__tile_l0_b
                    Tile<
                        TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v58 = Tile<
                            TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                            PadValue::Null, CompactMode::Null>(v7, v9);
                    // pto: %y_acc_inline2733_inline12985__tile_l0_b
                    uint64_t v59 = (uint64_t)v16;
                    TASSIGN(v58, v59);
                    TEXTRACT(v58, v51, k55, v21);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                    // pto: %1
                    Tile<
                        TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v60 = Tile<
                            TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                            PadValue::Null, CompactMode::Null>(v13, v7);
                    // pto: %1
                    uint64_t v61 = (uint64_t)v21;
                    TASSIGN(v60, v61);
                    // pto: %28
                    int64_t v62 = (int64_t)((uint64_t)k55 + (uint64_t)v7);
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
                    TEXTRACT(v60, v31, v21, v62);
                    // pto: %2
                    Tile<
                        TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v63 = Tile<
                            TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                            PadValue::Null, CompactMode::Null>(v7, v9);
                    // pto: %2
                    uint64_t v64 = (uint64_t)v21;
                    TASSIGN(v63, v64);
                    TEXTRACT(v63, v51, v62, v21);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
                    // pto: %30
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                    if (k55 == v21) {
                        // pto: %y_acc_inline2733_inline12985__tile_l0_c_first
                        Tile<
                            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                            PadValue::Null, CompactMode::Null>
                            v65 = Tile<
                                TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                                PadValue::Null, CompactMode::Null>(v13, v9);
                        // pto: %y_acc_inline2733_inline12985__tile_l0_c_first
                        uint64_t v66 = (uint64_t)v21;
                        TASSIGN(v65, v66);
                        pipe_barrier(PIPE_M);
                        TMATMUL(v65, v56, v58);
                    } else {
                        // pto: %y_acc_inline2733_inline12985__tile_l0_c_acc
                        Tile<
                            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                            PadValue::Null, CompactMode::Null>
                            v67 = Tile<
                                TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                                PadValue::Null, CompactMode::Null>(v13, v9);
                        // pto: %y_acc_inline2733_inline12985__tile_l0_c_acc
                        uint64_t v68 = (uint64_t)v21;
                        TASSIGN(v67, v68);
                        pipe_barrier(PIPE_M);
                        TMATMUL_ACC(v67, v67, v56, v58);
                    }
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
                    // pto: %3
                    Tile<
                        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>
                        v69 = Tile<
                            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                            PadValue::Null, CompactMode::Null>(v13, v9);
                    // pto: %3
                    uint64_t v70 = (uint64_t)v21;
                    TASSIGN(v69, v70);
                    pipe_barrier(PIPE_M);
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
                    TMATMUL_ACC(v69, v69, v60, v63);
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
                }
            } else {
                // pto: %4
                Tile<
                    TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v71 = Tile<
                        TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v8, v9);
                // pto: %4
                uint64_t v72 = (uint64_t)v18;
                TASSIGN(v71, v72);
                for (int64_t k73 = v21; k73 < v8; k73 += v9) {
                    // pto: %y_acc_inline2733_inline12985__iter_v1_l0_a
                    Tile<
                        TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v74 = Tile<
                            TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                            PadValue::Null, CompactMode::Null>(v13, v7);
                    // pto: %y_acc_inline2733_inline12985__iter_v1_l0_a
                    uint64_t v75 = (uint64_t)v17;
                    TASSIGN(v74, v75);
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
                    TEXTRACT(v74, v31, v21, k73);
                    // pto: %y_acc_inline2733_inline12985__iter_v1_l0_b
                    Tile<
                        TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v76 = Tile<
                            TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                            PadValue::Null, CompactMode::Null>(v7, v9);
                    // pto: %y_acc_inline2733_inline12985__iter_v1_l0_b
                    uint64_t v77 = (uint64_t)v16;
                    TASSIGN(v76, v77);
                    TEXTRACT(v76, v71, k73, v21);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
                    // pto: %5
                    Tile<
                        TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v78 = Tile<
                            TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                            PadValue::Null, CompactMode::Null>(v13, v7);
                    // pto: %5
                    uint64_t v79 = (uint64_t)v21;
                    TASSIGN(v78, v79);
                    // pto: %31
                    int64_t v80 = (int64_t)((uint64_t)k73 + (uint64_t)v7);
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
                    TEXTRACT(v78, v31, v21, v80);
                    // pto: %6
                    Tile<
                        TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v81 = Tile<
                            TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                            PadValue::Null, CompactMode::Null>(v7, v9);
                    // pto: %6
                    uint64_t v82 = (uint64_t)v21;
                    TASSIGN(v81, v82);
                    TEXTRACT(v81, v71, v80, v21);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
                    // pto: %y_acc_inline2733_inline12985__iter_v1_l0_c_acc
                    Tile<
                        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>
                        v83 = Tile<
                            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                            PadValue::Null, CompactMode::Null>(v13, v9);
                    // pto: %y_acc_inline2733_inline12985__iter_v1_l0_c_acc
                    uint64_t v84 = (uint64_t)v21;
                    TASSIGN(v83, v84);
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
                    pipe_barrier(PIPE_M);
                    TMATMUL_ACC(v83, v83, v74, v76);
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
                    // pto: %7
                    Tile<
                        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>
                        v85 = Tile<
                            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                            PadValue::Null, CompactMode::Null>(v13, v9);
                    // pto: %7
                    uint64_t v86 = (uint64_t)v21;
                    TASSIGN(v85, v86);
                    pipe_barrier(PIPE_M);
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
                    TMATMUL_ACC(v85, v85, v78, v81);
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
                }
            }
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
            // pto: %8
            Tile<
                TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v87 = Tile<
                    TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>(v9, v8);
            // pto: %8
            uint64_t v88 = (uint64_t)v21;
            TASSIGN(v87, v88);
            // pto: %39
            pto::Shape<1, 1, 1, 256, 512> v89 = pto::Shape<1, 1, 1, 256, 512>();
            // pto: %39
            pto::Stride<524288, 524288, 524288, 2048, 1> v90 = pto::Stride<524288, 524288, 524288, 2048, 1>();
            // pto: %39
            GlobalTensor<
                int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<524288, 524288, 524288, 2048, 1>, pto::Layout::ND>
                v91 = GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<524288, 524288, 524288, 2048, 1>,
                    pto::Layout::ND>(v3 + ((v21 + v47 * v15) + v40), v89, v90);
            TLOAD(v87, v91);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
            // pto: %9
            Tile<
                TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v92 = Tile<
                    TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                    CompactMode::Null>(v8, v9);
            // pto: %9
            uint64_t v93 = (uint64_t)v21;
            TASSIGN(v92, v93);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
            for (int64_t k94 = v21; k94 < v8; k94 += v9) {
                // pto: %10
                Tile<
                    TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v95 = Tile<
                        TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>(v13, v7);
                // pto: %10
                uint64_t v96 = (uint64_t)v15;
                TASSIGN(v95, v96);
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
                TEXTRACT(v95, v37, v21, k94);
                // pto: %11
                Tile<
                    TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>
                    v97 = Tile<
                        TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v7, v9);
                // pto: %11
                uint64_t v98 = (uint64_t)v16;
                TASSIGN(v97, v98);
                pipe_barrier(PIPE_MTE1);
                TEXTRACT(v97, v92, k94, v21);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
                // pto: %12
                Tile<
                    TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v99 = Tile<
                        TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>(v13, v7);
                // pto: %12
                uint64_t v100 = (uint64_t)v14;
                TASSIGN(v99, v100);
                // pto: %41
                int64_t v101 = (int64_t)((uint64_t)k94 + (uint64_t)v7);
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
                TEXTRACT(v99, v37, v21, v101);
                // pto: %13
                Tile<
                    TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>
                    v102 = Tile<
                        TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v7, v9);
                // pto: %13
                uint64_t v103 = (uint64_t)v21;
                TASSIGN(v102, v103);
                TEXTRACT(v102, v92, v101, v21);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
                // pto: %14
                Tile<
                    TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v104 = Tile<
                        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v13, v9);
                // pto: %14
                uint64_t v105 = (uint64_t)v21;
                TASSIGN(v104, v105);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
                pipe_barrier(PIPE_M);
                TMATMUL_ACC(v104, v104, v95, v97);
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
                // pto: %15
                Tile<
                    TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v106 = Tile<
                        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v13, v9);
                // pto: %15
                uint64_t v107 = (uint64_t)v21;
                TASSIGN(v106, v107);
                pipe_barrier(PIPE_M);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
                TMATMUL_ACC(v106, v106, v99, v102);
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
            }
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        }
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        // pto: %t__tile
        Tile<
            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v108 = Tile<
                TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v13, v9);
        // pto: %t__tile
        uint64_t v109 = (uint64_t)v21;
        TASSIGN(v108, v109);
        // pto: %y_i32_inline2737_inline13010__iter_v1_pview
        pto::Shape<1, 1, 1, 16, 256> v110 = pto::Shape<1, 1, 1, 16, 256>();
        // pto: %y_i32_inline2737_inline13010__iter_v1_pview
        pto::Stride<65536, 65536, 65536, 4096, 1> v111 = pto::Stride<65536, 65536, 65536, 4096, 1>();
        // pto: %43, %y_i32_inline2737_inline13010__iter_v1_pview
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>
            v112 = GlobalTensor<
                int32_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(
                v1 + (v21 + (v27 < v21 ? v21 : v27)), v110, v111
            );
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        pipe_barrier(PIPE_FIX);
        TSTORE(v112, v108);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    }
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
#endif  // __DAV_CUBE__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: y_i32_inline2737_inline13010__ssa_v0
    __gm__ Tensor *y_i32_inline2737_inline13010__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int32_t *y_i32_inline2737_inline13010__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(y_i32_inline2737_inline13010__ssa_v0_tensor->buffer.addr) +
        y_i32_inline2737_inline13010__ssa_v0_tensor->start_offset;

    // Unpack tensor: h_tile_i8_inline2806_inline13011__ssa_v4
    __gm__ Tensor *h_tile_i8_inline2806_inline13011__ssa_v4_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int8_t *h_tile_i8_inline2806_inline13011__ssa_v4 =
        reinterpret_cast<__gm__ int8_t *>(h_tile_i8_inline2806_inline13011__ssa_v4_tensor->buffer.addr) +
        h_tile_i8_inline2806_inline13011__ssa_v4_tensor->start_offset;

    // Unpack tensor: routed_w2_last_inline489__ssa_v0
    __gm__ Tensor *routed_w2_last_inline489__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ int8_t *routed_w2_last_inline489__ssa_v0 =
        reinterpret_cast<__gm__ int8_t *>(routed_w2_last_inline489__ssa_v0_tensor->buffer.addr) +
        routed_w2_last_inline489__ssa_v0_tensor->start_offset;

    // Unpack scalar: local_e_inline2742_inline13113__idx_v0
    union {
        uint64_t u64;
        int64_t val;
    } local_e_inline2742_inline13113__idx_v0_conv;
    local_e_inline2742_inline13113__idx_v0_conv.u64 = args[3];
    int64_t local_e_inline2742_inline13113__idx_v0 = local_e_inline2742_inline13113__idx_v0_conv.val;

    // Forward to ptoas-generated function
    exp_w2_mm_3(
        y_i32_inline2737_inline13010__ssa_v0, h_tile_i8_inline2806_inline13011__ssa_v4,
        routed_w2_last_inline489__ssa_v0, local_e_inline2742_inline13113__idx_v0, __pypto_spmd_block_idx,
        __pypto_spmd_block_num
    );
}
