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
// Kernel Function: proj_b_mm_0

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

static __aicore__ void proj_b_mm_0(
    __gm__ int32_t *v1, __gm__ int8_t *v2, __gm__ int8_t *v3, int64_t v4, int64_t v5, int64_t v6, int32_t v7, int32_t v8
) {
    const int64_t v9 = 8192;
    const int64_t v10 = 128;
    const int64_t v11 = 4;
    const int64_t v12 = 256;
    const int64_t v13 = 2;
    const int64_t v14 = 512;
    const int64_t v15 = 1;
    const int64_t v16 = 16;
    const int64_t v17 = 2048;
    const int64_t v18 = 6144;
    const int64_t v19 = 32768;
    const int64_t v20 = 4096;
    const int64_t v21 = 73728;
    const int64_t v22 = 69632;
    const int64_t v23 = 0;
    using T = float;

#if defined(__DAV_CUBE__)
    // pto: %acc_b_inline1011_inline9749__phi_v5
    Tile<
        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v24 = Tile<
            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v16, v12);
    // pto: %acc_b_inline1011_inline9749__phi_v5
    uint64_t v25 = (uint64_t)v23;
    TASSIGN(v24, v25);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    for (int64_t i26 = v23; i26 < v13; i26 += v15) {
        // pto: %dc_inline953_inline9486__ssa_v0, %20, %22, %21
        int64_t v27 = (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)v7) * (uint64_t)v14)) +
                                (uint64_t)((int64_t)((uint64_t)i26 * (uint64_t)v12)));
        // pto: %acc_b_inline1011_inline9749__tile
        Tile<
            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v28 = Tile<
                TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v16, v12);
        // pto: %acc_b_inline1011_inline9749__tile
        uint64_t v29 = (uint64_t)v23;
        TASSIGN(v28, v29);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        for (int64_t j30 = v23; j30 < v11; j30 += v13) {
            // pto: %23
            int64_t v31 = (int64_t)((uint64_t)j30 * (uint64_t)v12);
            // pto: %24
            int64_t v32 = (int64_t)((uint64_t)v5 + (uint64_t)v31);
            // pto: %27, %26
            int64_t v33 = (int64_t)((uint64_t)v5 + (uint64_t)((int64_t)((uint64_t)v31 + (uint64_t)v12)));
            // pto: %28
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            if (j30 == v23) {
                // pto: %b_act_inline981_inline9483__tile
                Tile<
                    TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v34 = Tile<
                        TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>(v16, v12);
                // pto: %b_act_inline981_inline9483__tile
                uint64_t v35 = (uint64_t)v22;
                TASSIGN(v34, v35);
                // pto: %29
                int64_t v36 = v5 < v23 ? v23 : v5;
                // pto: %o_r_i8_pad_inline1095_inline9509__rv_v4_pview
                pto::Shape<1, 1, 1, 16, 256> v37 = pto::Shape<1, 1, 1, 16, 256>();
                // pto: %o_r_i8_pad_inline1095_inline9509__rv_v4_pview
                pto::Stride<131072, 131072, 131072, 8192, 1> v38 = pto::Stride<131072, 131072, 131072, 8192, 1>();
                // pto: %o_r_i8_pad_inline1095_inline9509__rv_v4_pview
                GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<131072, 131072, 131072, 8192, 1>, pto::Layout::ND>
                    v39 = GlobalTensor<
                        int8_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<131072, 131072, 131072, 8192, 1>,
                        pto::Layout::ND>(v2 + (v23 + v36), v37, v38);
                TLOAD(v34, v39);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                // pto: %b_weight_inline950_inline9862__tile
                Tile<
                    TileType::Mat, int8_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v40 = Tile<
                        TileType::Mat, int8_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>(v12, v12);
                // pto: %b_weight_inline950_inline9862__tile
                uint64_t v41 = (uint64_t)v21;
                TASSIGN(v40, v41);
                // pto: %wo_b_l1_inline736__ssa_v0_pview
                pto::Shape<1, 1, 1, 256, 256> v42 = pto::Shape<1, 1, 1, 256, 256>();
                // pto: %wo_b_l1_inline736__ssa_v0_pview
                pto::Stride<2097152, 2097152, 2097152, 8192, 1> v43 = pto::Stride<2097152, 2097152, 2097152, 8192, 1>();
                // pto: %30, %wo_b_l1_inline736__ssa_v0_pview
                GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>,
                    pto::Layout::ND>
                    v44 = GlobalTensor<
                        int8_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>,
                        pto::Layout::ND>(v3 + ((v23 + (v27 < v23 ? v23 : v27) * v9) + v36), v42, v43);
                TLOAD(v40, v44);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
                // pto: %b_weight_inline950_inline9862__tile_t
                Tile<
                    TileType::Mat, int8_t, 256, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v45 = Tile<
                        TileType::Mat, int8_t, 256, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v12, v12);
                // pto: %b_weight_inline950_inline9862__tile_t
                uint64_t v46 = (uint64_t)v21;
                TASSIGN(v45, v46);
                // pto: %acc_b_inline1011_inline9749__tile_l0_init
                Tile<
                    TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v47 = Tile<
                        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v16, v12);
                // pto: %acc_b_inline1011_inline9749__tile_l0_init
                uint64_t v48 = (uint64_t)v23;
                TASSIGN(v47, v48);
                // pto: %acc_b_inline1011_inline9749__tile_l0_a
                Tile<
                    TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v49 = Tile<
                        TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>(v16, v10);
                // pto: %acc_b_inline1011_inline9749__tile_l0_a
                uint64_t v50 = (uint64_t)v20;
                TASSIGN(v49, v50);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                TEXTRACT(v49, v34, v23, v23);
                // pto: %acc_b_inline1011_inline9749__tile_l0_b
                Tile<
                    TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>
                    v51 = Tile<
                        TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v10, v12);
                // pto: %acc_b_inline1011_inline9749__tile_l0_b
                uint64_t v52 = (uint64_t)v19;
                TASSIGN(v51, v52);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
                TEXTRACT(v51, v45, v23, v23);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                // pto: %0
                Tile<
                    TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v53 = Tile<
                        TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>(v16, v10);
                // pto: %0
                uint64_t v54 = (uint64_t)v18;
                TASSIGN(v53, v54);
                TEXTRACT(v53, v34, v23, v10);
                // pto: %1
                Tile<
                    TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>
                    v55 = Tile<
                        TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v10, v12);
                // pto: %1
                uint64_t v56 = (uint64_t)v23;
                TASSIGN(v55, v56);
                TEXTRACT(v55, v45, v10, v23);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
                // pto: %acc_b_inline1011_inline9749__tile_l0_c_first
                Tile<
                    TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v57 = Tile<
                        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v16, v12);
                // pto: %acc_b_inline1011_inline9749__tile_l0_c_first
                uint64_t v58 = (uint64_t)v23;
                TASSIGN(v57, v58);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                pipe_barrier(PIPE_M);
                TMATMUL(v57, v49, v51);
                // pto: %acc_b_inline1011_inline9749__tile_l0_c_acc
                Tile<
                    TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v59 = Tile<
                        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v16, v12);
                // pto: %acc_b_inline1011_inline9749__tile_l0_c_acc
                uint64_t v60 = (uint64_t)v23;
                TASSIGN(v59, v60);
                pipe_barrier(PIPE_M);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
                TMATMUL_ACC(v59, v59, v53, v55);
            } else {
                // pto: %2
                Tile<
                    TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v61 = Tile<
                        TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>(v16, v12);
                // pto: %2
                uint64_t v62 = (uint64_t)v22;
                TASSIGN(v61, v62);
                // pto: %32
                int64_t v63 = v32 < v23 ? v23 : v32;
                // pto: %33
                pto::Shape<1, 1, 1, 16, 256> v64 = pto::Shape<1, 1, 1, 16, 256>();
                // pto: %33
                pto::Stride<131072, 131072, 131072, 8192, 1> v65 = pto::Stride<131072, 131072, 131072, 8192, 1>();
                // pto: %33
                GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<131072, 131072, 131072, 8192, 1>, pto::Layout::ND>
                    v66 = GlobalTensor<
                        int8_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<131072, 131072, 131072, 8192, 1>,
                        pto::Layout::ND>(v2 + (v23 + v63), v64, v65);
                TLOAD(v61, v66);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
                // pto: %3
                Tile<
                    TileType::Mat, int8_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v67 = Tile<
                        TileType::Mat, int8_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>(v12, v12);
                // pto: %3
                uint64_t v68 = (uint64_t)v21;
                TASSIGN(v67, v68);
                // pto: %36
                pto::Shape<1, 1, 1, 256, 256> v69 = pto::Shape<1, 1, 1, 256, 256>();
                // pto: %36
                pto::Stride<2097152, 2097152, 2097152, 8192, 1> v70 = pto::Stride<2097152, 2097152, 2097152, 8192, 1>();
                // pto: %34, %36
                GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>,
                    pto::Layout::ND>
                    v71 = GlobalTensor<
                        int8_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>,
                        pto::Layout::ND>(v3 + ((v23 + (v27 < v23 ? v23 : v27) * v9) + v63), v69, v70);
                TLOAD(v67, v71);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
                // pto: %4
                Tile<
                    TileType::Mat, int8_t, 256, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v72 = Tile<
                        TileType::Mat, int8_t, 256, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v12, v12);
                // pto: %4
                uint64_t v73 = (uint64_t)v21;
                TASSIGN(v72, v73);
                // pto: %5
                Tile<
                    TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v74 = Tile<
                        TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>(v16, v10);
                // pto: %5
                uint64_t v75 = (uint64_t)v20;
                TASSIGN(v74, v75);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
                TEXTRACT(v74, v61, v23, v23);
                // pto: %6
                Tile<
                    TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>
                    v76 = Tile<
                        TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v10, v12);
                // pto: %6
                uint64_t v77 = (uint64_t)v19;
                TASSIGN(v76, v77);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
                TEXTRACT(v76, v72, v23, v23);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
                // pto: %7
                Tile<
                    TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v78 = Tile<
                        TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>(v16, v10);
                // pto: %7
                uint64_t v79 = (uint64_t)v18;
                TASSIGN(v78, v79);
                TEXTRACT(v78, v61, v23, v10);
                // pto: %8
                Tile<
                    TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>
                    v80 = Tile<
                        TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v10, v12);
                // pto: %8
                uint64_t v81 = (uint64_t)v23;
                TASSIGN(v80, v81);
                TEXTRACT(v80, v72, v10, v23);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
                // pto: %9
                Tile<
                    TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v82 = Tile<
                        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v16, v12);
                // pto: %9
                uint64_t v83 = (uint64_t)v23;
                TASSIGN(v82, v83);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
                pipe_barrier(PIPE_M);
                TMATMUL_ACC(v82, v82, v74, v76);
                // pto: %10
                Tile<
                    TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v84 = Tile<
                        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v16, v12);
                // pto: %10
                uint64_t v85 = (uint64_t)v23;
                TASSIGN(v84, v85);
                pipe_barrier(PIPE_M);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
                TMATMUL_ACC(v84, v84, v78, v80);
            }
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            // pto: %11
            Tile<
                TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v86 = Tile<
                    TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>(v16, v12);
            // pto: %11
            uint64_t v87 = (uint64_t)v23;
            TASSIGN(v86, v87);
            // pto: %37
            int64_t v88 = v33 < v23 ? v23 : v33;
            // pto: %38
            pto::Shape<1, 1, 1, 16, 256> v89 = pto::Shape<1, 1, 1, 16, 256>();
            // pto: %38
            pto::Stride<131072, 131072, 131072, 8192, 1> v90 = pto::Stride<131072, 131072, 131072, 8192, 1>();
            // pto: %38
            GlobalTensor<
                int8_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<131072, 131072, 131072, 8192, 1>, pto::Layout::ND>
                v91 = GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<131072, 131072, 131072, 8192, 1>,
                    pto::Layout::ND>(v2 + (v23 + v88), v89, v90);
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
            TLOAD(v86, v91);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
            // pto: %12
            Tile<
                TileType::Mat, int8_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v92 = Tile<
                    TileType::Mat, int8_t, 256, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>(v12, v12);
            // pto: %12
            uint64_t v93 = (uint64_t)v20;
            TASSIGN(v92, v93);
            // pto: %41
            pto::Shape<1, 1, 1, 256, 256> v94 = pto::Shape<1, 1, 1, 256, 256>();
            // pto: %41
            pto::Stride<2097152, 2097152, 2097152, 8192, 1> v95 = pto::Stride<2097152, 2097152, 2097152, 8192, 1>();
            // pto: %39, %41
            GlobalTensor<
                int8_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>, pto::Layout::ND>
                v96 = GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>,
                    pto::Layout::ND>(v3 + ((v23 + (v27 < v23 ? v23 : v27) * v9) + v88), v94, v95);
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
            TLOAD(v92, v96);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
            // pto: %13
            Tile<
                TileType::Mat, int8_t, 256, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v97 = Tile<
                    TileType::Mat, int8_t, 256, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                    CompactMode::Null>(v12, v12);
            // pto: %13
            uint64_t v98 = (uint64_t)v20;
            TASSIGN(v97, v98);
            // pto: %14
            Tile<
                TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v99 = Tile<
                    TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>(v16, v10);
            // pto: %14
            uint64_t v100 = (uint64_t)v23;
            TASSIGN(v99, v100);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
            TEXTRACT(v99, v86, v23, v23);
            // pto: %15
            Tile<
                TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v101 = Tile<
                    TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v10, v12);
            // pto: %15
            uint64_t v102 = (uint64_t)v19;
            TASSIGN(v101, v102);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            TEXTRACT(v101, v97, v23, v23);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
            // pto: %16
            Tile<
                TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v103 = Tile<
                    TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>(v16, v10);
            // pto: %16
            uint64_t v104 = (uint64_t)v17;
            TASSIGN(v103, v104);
            TEXTRACT(v103, v86, v23, v10);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
            // pto: %17
            Tile<
                TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v105 = Tile<
                    TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v10, v12);
            // pto: %17
            uint64_t v106 = (uint64_t)v23;
            TASSIGN(v105, v106);
            TEXTRACT(v105, v97, v10, v23);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
            // pto: %18
            Tile<
                TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v107 = Tile<
                    TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v16, v12);
            // pto: %18
            uint64_t v108 = (uint64_t)v23;
            TASSIGN(v107, v108);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v107, v107, v99, v101);
            // pto: %19
            Tile<
                TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v109 = Tile<
                    TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v16, v12);
            // pto: %19
            uint64_t v110 = (uint64_t)v23;
            TASSIGN(v109, v110);
            pipe_barrier(PIPE_M);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
            TMATMUL_ACC(v109, v109, v103, v105);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        }
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        // pto: %42, %43
        int64_t v111 = (int64_t)((uint64_t)((int64_t)((uint64_t)v6 * (uint64_t)v20)) + (uint64_t)v27);
        // pto: %partials_inline1006_inline9833__iter_v3_pview
        pto::Shape<1, 1, 1, 16, 256> v112 = pto::Shape<1, 1, 1, 16, 256>();
        // pto: %partials_inline1006_inline9833__iter_v3_pview
        pto::Stride<524288, 524288, 524288, 32768, 1> v113 = pto::Stride<524288, 524288, 524288, 32768, 1>();
        // pto: %44, %partials_inline1006_inline9833__iter_v3_pview
        GlobalTensor<
            int32_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<524288, 524288, 524288, 32768, 1>, pto::Layout::ND>
            v114 = GlobalTensor<
                int32_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<524288, 524288, 524288, 32768, 1>, pto::Layout::ND>(
                v1 + (v23 + (v111 < v23 ? v23 : v111)), v112, v113
            );
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        pipe_barrier(PIPE_FIX);
        TSTORE(v114, v28);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    }
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
#endif  // __DAV_CUBE__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: partials_inline1006_inline9833__iter_v1
    __gm__ Tensor *partials_inline1006_inline9833__iter_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int32_t *partials_inline1006_inline9833__iter_v1 =
        reinterpret_cast<__gm__ int32_t *>(partials_inline1006_inline9833__iter_v1_tensor->buffer.addr) +
        partials_inline1006_inline9833__iter_v1_tensor->start_offset;

    // Unpack tensor: o_r_i8_pad_inline1095_inline9509__rv_v4
    __gm__ Tensor *o_r_i8_pad_inline1095_inline9509__rv_v4_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int8_t *o_r_i8_pad_inline1095_inline9509__rv_v4 =
        reinterpret_cast<__gm__ int8_t *>(o_r_i8_pad_inline1095_inline9509__rv_v4_tensor->buffer.addr) +
        o_r_i8_pad_inline1095_inline9509__rv_v4_tensor->start_offset;

    // Unpack tensor: wo_b_l1_inline736__ssa_v0
    __gm__ Tensor *wo_b_l1_inline736__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ int8_t *wo_b_l1_inline736__ssa_v0 =
        reinterpret_cast<__gm__ int8_t *>(wo_b_l1_inline736__ssa_v0_tensor->buffer.addr) +
        wo_b_l1_inline736__ssa_v0_tensor->start_offset;

    // Unpack scalar: n0_inline1077_inline9504__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } n0_inline1077_inline9504__ssa_v0_conv;
    n0_inline1077_inline9504__ssa_v0_conv.u64 = args[3];
    int64_t n0_inline1077_inline9504__ssa_v0 = n0_inline1077_inline9504__ssa_v0_conv.val;

    // Unpack scalar: col_g_inline1089_inline9761__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } col_g_inline1089_inline9761__ssa_v0_conv;
    col_g_inline1089_inline9761__ssa_v0_conv.u64 = args[4];
    int64_t col_g_inline1089_inline9761__ssa_v0 = col_g_inline1089_inline9761__ssa_v0_conv.val;

    // Unpack scalar: g_inline978_inline9508__idx_v0
    union {
        uint64_t u64;
        int64_t val;
    } g_inline978_inline9508__idx_v0_conv;
    g_inline978_inline9508__idx_v0_conv.u64 = args[5];
    int64_t g_inline978_inline9508__idx_v0 = g_inline978_inline9508__idx_v0_conv.val;

    // Forward to ptoas-generated function
    proj_b_mm_0(
        partials_inline1006_inline9833__iter_v1, o_r_i8_pad_inline1095_inline9509__rv_v4, wo_b_l1_inline736__ssa_v0,
        n0_inline1077_inline9504__ssa_v0, col_g_inline1089_inline9761__ssa_v0, g_inline978_inline9508__idx_v0,
        __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
