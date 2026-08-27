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
// Kernel Function: qk_pv_2_aic

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

static __aicore__ void qk_pv_2_aic(
    __gm__ float *v1, __gm__ bfloat16_t *v2, __gm__ float *v3, __gm__ float *v4, __gm__ float *v5,
    __gm__ bfloat16_t *v6, __gm__ float *v7, int32_t v8, int32_t v9
) {
    const int32_t v10 = 128;
    const int32_t v11 = 32;
    const int64_t v12 = 96;
    const int64_t v13 = 16;
    const int64_t v14 = 32;
    const int64_t v15 = 64;
    const int64_t v16 = 128;
    const int64_t v17 = 2;
    const int64_t v18 = 512;
    const int64_t v19 = 256;
    const int64_t v20 = 32768;
    const int64_t v21 = 8192;
    const int64_t v22 = 0;
    const int64_t v23 = 196608;
    const int64_t v24 = 65536;
    const int32_t v25 = 0;
    const int64_t v26 = 1024;
    using T = float;

#if defined(__DAV_CUBE__)
    // pto: %qk_raw_inline1454_inline11337__tile_l0_c_phi
    Tile<
        TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v27 = Tile<
            TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v14, v16);
    // pto: %qk_raw_inline1454_inline11337__tile_l0_c_phi
    uint64_t v28 = (uint64_t)v22;
    TASSIGN(v27, v28);
    // pto: %48
    Tile<
        TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v29 = Tile<
            TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v14, v16);
    // pto: %48
    uint64_t v30 = (uint64_t)v22;
    TASSIGN(v29, v30);
    // pto: %qk_oi_inline1554_inline11355__tile_l0_c_phi
    Tile<
        TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v31 = Tile<
            TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v14, v18);
    // pto: %qk_oi_inline1554_inline11355__tile_l0_c_phi
    uint64_t v32 = (uint64_t)v22;
    TASSIGN(v31, v32);
    // pto: %66
    Tile<
        TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v33 = Tile<
            TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v14, v18);
    // pto: %66
    uint64_t v34 = (uint64_t)v22;
    TASSIGN(v33, v34);
    auto v35 = TPipe<0, Direction::DIR_BOTH, 16384, 4, 4, true>(v7, v25, v25);
    // pto: %qk_item_inline1457_inline11345__ssa_v0
    int64_t v36 = (int64_t)v8;
    // pto: %27
    int64_t v37 = v36 / v17;
    // pto: %29, %28
    int64_t v38 = (int64_t)((uint64_t)v36 - (uint64_t)((int64_t)((uint64_t)v37 * (uint64_t)v17)));
    // pto: %30
    int64_t v39 = (int64_t)((uint64_t)v37 * (uint64_t)v16);
    // pto: %32, %33, %31
    int64_t v40 = (int64_t)((uint64_t)((int64_t)((uint64_t)v37 * (uint64_t)v19)) +
                            (uint64_t)((int64_t)((uint64_t)v38 * (uint64_t)v16)));
    // pto: %qk_kv_inline1491_inline11532__tile
    Tile<
        TileType::Mat, bfloat16_t, 128, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v41 = Tile<
            TileType::Mat, bfloat16_t, 128, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qk_kv_inline1491_inline11532__tile
    uint64_t v42 = (uint64_t)v24;
    TASSIGN(v41, v42);
    // pto: %hca_kv_flat_inline1485_inline11368__rv_v11_pview
    pto::Shape<1, 1, 1, 128, 512> v43 = pto::Shape<1, 1, 1, 128, 512>();
    // pto: %hca_kv_flat_inline1485_inline11368__rv_v11_pview
    pto::Stride<65536, 65536, 65536, 512, 1> v44 = pto::Stride<65536, 65536, 65536, 512, 1>();
    // pto: %34, %hca_kv_flat_inline1485_inline11368__rv_v11_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<65536, 65536, 65536, 512, 1>, pto::Layout::ND>
        v45 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<65536, 65536, 65536, 512, 1>, pto::Layout::ND>(
            v2 + (v22 + (v40 < v22 ? v22 : v40) * v18), v43, v44
        );
    TLOAD(v41, v45);
    // pto: %35
    int64_t v46 = (int64_t)((uint64_t)v37 * (uint64_t)v15);
    // pto: %qk_q_tile_inline1514_inline11338__tile
    Tile<
        TileType::Mat, bfloat16_t, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v47 = Tile<
            TileType::Mat, bfloat16_t, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v14, v18);
    // pto: %qk_q_tile_inline1514_inline11338__tile
    uint64_t v48 = (uint64_t)v23;
    TASSIGN(v47, v48);
    // pto: %q_flat_inline1542_inline11350__ssa_v0_pview
    pto::Shape<1, 1, 1, 32, 512> v49 = pto::Shape<1, 1, 1, 32, 512>();
    // pto: %q_flat_inline1542_inline11350__ssa_v0_pview
    pto::Stride<16384, 16384, 16384, 512, 1> v50 = pto::Stride<16384, 16384, 16384, 512, 1>();
    // pto: %36, %q_flat_inline1542_inline11350__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 512>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND>
        v51 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 32, 512>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND>(
            v6 + (v22 + (v46 < v22 ? v22 : v46) * v18), v49, v50
        );
    TLOAD(v47, v51);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // pto: %qk_kv_inline1491_inline11532__tile_t
    Tile<
        TileType::Mat, bfloat16_t, 512, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v52 = Tile<
            TileType::Mat, bfloat16_t, 512, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v18, v16);
    // pto: %qk_kv_inline1491_inline11532__tile_t
    uint64_t v53 = (uint64_t)v24;
    TASSIGN(v52, v53);
    // pto: %qk_raw_inline1454_inline11337__tile_l0_init
    Tile<
        TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v54 = Tile<
            TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v14, v16);
    // pto: %qk_raw_inline1454_inline11337__tile_l0_init
    uint64_t v55 = (uint64_t)v22;
    TASSIGN(v54, v55);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    for (int64_t i56 = v22; i56 < v18; i56 += v19) {
        // pto: %qk_raw_inline1454_inline11337__tile_l0_a
        Tile<
            TileType::Left, bfloat16_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v57 = Tile<
                TileType::Left, bfloat16_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v14, v16);
        // pto: %qk_raw_inline1454_inline11337__tile_l0_a
        uint64_t v58 = (uint64_t)v22;
        TASSIGN(v57, v58);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        pipe_barrier(PIPE_MTE1);
        TEXTRACT(v57, v47, v22, i56);
        // pto: %qk_raw_inline1454_inline11337__tile_l0_b
        Tile<
            TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v59 = Tile<
                TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>(v16, v16);
        // pto: %qk_raw_inline1454_inline11337__tile_l0_b
        uint64_t v60 = (uint64_t)v22;
        TASSIGN(v59, v60);
        TEXTRACT(v59, v52, i56, v22);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        // pto: %0
        Tile<
            TileType::Left, bfloat16_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v61 = Tile<
                TileType::Left, bfloat16_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v14, v16);
        // pto: %0
        uint64_t v62 = (uint64_t)v21;
        TASSIGN(v61, v62);
        // pto: %37
        int64_t v63 = (int64_t)((uint64_t)i56 + (uint64_t)v16);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v61, v47, v22, v63);
        // pto: %1
        Tile<
            TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v64 = Tile<
                TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>(v16, v16);
        // pto: %1
        uint64_t v65 = (uint64_t)v20;
        TASSIGN(v64, v65);
        TEXTRACT(v64, v52, v63, v22);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        // pto: %39
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if (i56 == v22) {
            // pto: %qk_raw_inline1454_inline11337__tile_l0_c_first
            Tile<
                TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v66 = Tile<
                    TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v14, v16);
            // pto: %qk_raw_inline1454_inline11337__tile_l0_c_first
            uint64_t v67 = (uint64_t)v22;
            TASSIGN(v66, v67);
            pipe_barrier(PIPE_M);
            TMATMUL(v66, v57, v59);
        } else {
            // pto: %qk_raw_inline1454_inline11337__tile_l0_c_acc
            Tile<
                TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v68 = Tile<
                    TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v14, v16);
            // pto: %qk_raw_inline1454_inline11337__tile_l0_c_acc
            uint64_t v69 = (uint64_t)v22;
            TASSIGN(v68, v69);
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v68, v68, v57, v59);
        }
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        // pto: %2
        Tile<
            TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v70 = Tile<
                TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v14, v16);
        // pto: %2
        uint64_t v71 = (uint64_t)v22;
        TASSIGN(v70, v71);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        TMATMUL_ACC(v70, v70, v61, v64);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    }
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TPUSH<
        TPipe<0, Direction::DIR_BOTH, 16384, 4, 4, true>,
        Tile<
            TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>,
        TileSplitAxis::TILE_NO_SPLIT>(v35, v54);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    // pto: %41
    int64_t v72 = (int64_t)((uint64_t)v46 + (uint64_t)v14);
    // pto: %3
    Tile<
        TileType::Mat, bfloat16_t, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v73 = Tile<
            TileType::Mat, bfloat16_t, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v14, v18);
    // pto: %3
    uint64_t v74 = (uint64_t)v23;
    TASSIGN(v73, v74);
    // pto: %43
    pto::Shape<1, 1, 1, 32, 512> v75 = pto::Shape<1, 1, 1, 32, 512>();
    // pto: %43
    pto::Stride<16384, 16384, 16384, 512, 1> v76 = pto::Stride<16384, 16384, 16384, 512, 1>();
    // pto: %42, %43
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 512>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND>
        v77 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 32, 512>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND>(
            v6 + (v22 + (v72 < v22 ? v22 : v72) * v18), v75, v76
        );
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v73, v77);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    // pto: %4
    Tile<
        TileType::Mat, bfloat16_t, 512, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v78 = Tile<
            TileType::Mat, bfloat16_t, 512, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v18, v16);
    // pto: %4
    uint64_t v79 = (uint64_t)v24;
    TASSIGN(v78, v79);
    // pto: %5
    Tile<
        TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v80 = Tile<
            TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v14, v16);
    // pto: %5
    uint64_t v81 = (uint64_t)v22;
    TASSIGN(v80, v81);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    for (int64_t i82 = v22; i82 < v18; i82 += v19) {
        // pto: %6
        Tile<
            TileType::Left, bfloat16_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v83 = Tile<
                TileType::Left, bfloat16_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v14, v16);
        // pto: %6
        uint64_t v84 = (uint64_t)v22;
        TASSIGN(v83, v84);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        pipe_barrier(PIPE_MTE1);
        TEXTRACT(v83, v73, v22, i82);
        // pto: %7
        Tile<
            TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v85 = Tile<
                TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>(v16, v16);
        // pto: %7
        uint64_t v86 = (uint64_t)v22;
        TASSIGN(v85, v86);
        TEXTRACT(v85, v78, i82, v22);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        // pto: %8
        Tile<
            TileType::Left, bfloat16_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v87 = Tile<
                TileType::Left, bfloat16_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v14, v16);
        // pto: %8
        uint64_t v88 = (uint64_t)v21;
        TASSIGN(v87, v88);
        // pto: %45
        int64_t v89 = (int64_t)((uint64_t)i82 + (uint64_t)v16);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        TEXTRACT(v87, v73, v22, v89);
        // pto: %9
        Tile<
            TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v90 = Tile<
                TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>(v16, v16);
        // pto: %9
        uint64_t v91 = (uint64_t)v20;
        TASSIGN(v90, v91);
        TEXTRACT(v90, v78, v89, v22);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        // pto: %47
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        if (i82 == v22) {
            // pto: %10
            Tile<
                TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v92 = Tile<
                    TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v14, v16);
            // pto: %10
            uint64_t v93 = (uint64_t)v22;
            TASSIGN(v92, v93);
            pipe_barrier(PIPE_M);
            TMATMUL(v92, v83, v85);
        } else {
            // pto: %11
            Tile<
                TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v94 = Tile<
                    TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v14, v16);
            // pto: %11
            uint64_t v95 = (uint64_t)v22;
            TASSIGN(v94, v95);
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v94, v94, v83, v85);
        }
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        // pto: %12
        Tile<
            TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v96 = Tile<
                TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v14, v16);
        // pto: %12
        uint64_t v97 = (uint64_t)v22;
        TASSIGN(v96, v97);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        TMATMUL_ACC(v96, v96, v87, v90);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    }
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    TPUSH<
        TPipe<0, Direction::DIR_BOTH, 16384, 4, 4, true>,
        Tile<
            TileType::Acc, float, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>,
        TileSplitAxis::TILE_NO_SPLIT>(v35, v80);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    // pto: %qk_oi_inline1554_inline11355__tile_l0_init
    Tile<
        TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v98 = Tile<
            TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v14, v18);
    // pto: %qk_oi_inline1554_inline11355__tile_l0_init
    uint64_t v99 = (uint64_t)v22;
    TASSIGN(v98, v99);
    // pto: %qk_oi_inline1554_inline11355__tile_l0_lmat
    Tile<
        TileType::Mat, bfloat16_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v100 = Tile<
            TileType::Mat, bfloat16_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v11, v10);
    TPOP<
        TPipe<0, Direction::DIR_BOTH, 16384, 4, 4, true>,
        Tile<
            TileType::Mat, bfloat16_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>,
        TileSplitAxis::TILE_NO_SPLIT>(v35, v100);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
    for (int64_t i101 = v22; i101 < v16; i101 += v15) {
        // pto: %qk_oi_inline1554_inline11355__tile_l0_a
        Tile<
            TileType::Left, bfloat16_t, 32, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v102 = Tile<
                TileType::Left, bfloat16_t, 32, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v14, v14);
        // pto: %qk_oi_inline1554_inline11355__tile_l0_a
        uint64_t v103 = (uint64_t)v22;
        TASSIGN(v102, v103);
        pipe_barrier(PIPE_MTE1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        TEXTRACT(v102, v100, v22, i101);
        // pto: %qk_oi_inline1554_inline11355__tile_l0_b
        Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v104 = Tile<
                TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v14, v18);
        // pto: %qk_oi_inline1554_inline11355__tile_l0_b
        uint64_t v105 = (uint64_t)v22;
        TASSIGN(v104, v105);
        TEXTRACT(v104, v41, i101, v22);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
        // pto: %13
        Tile<
            TileType::Left, bfloat16_t, 32, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v106 = Tile<
                TileType::Left, bfloat16_t, 32, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v14, v14);
        // pto: %13
        uint64_t v107 = (uint64_t)v21;
        TASSIGN(v106, v107);
        // pto: %49
        int64_t v108 = (int64_t)((uint64_t)i101 + (uint64_t)v14);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
        TEXTRACT(v106, v100, v22, v108);
        // pto: %14
        Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v109 = Tile<
                TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v14, v18);
        // pto: %14
        uint64_t v110 = (uint64_t)v20;
        TASSIGN(v109, v110);
        TEXTRACT(v109, v41, v108, v22);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
        // pto: %51
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
        if (i101 == v22) {
            // pto: %qk_oi_inline1554_inline11355__tile_l0_c_first
            Tile<
                TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v111 = Tile<
                    TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v14, v18);
            // pto: %qk_oi_inline1554_inline11355__tile_l0_c_first
            uint64_t v112 = (uint64_t)v22;
            TASSIGN(v111, v112);
            pipe_barrier(PIPE_M);
            TMATMUL(v111, v102, v104);
        } else {
            // pto: %qk_oi_inline1554_inline11355__tile_l0_c_acc
            Tile<
                TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v113 = Tile<
                    TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v14, v18);
            // pto: %qk_oi_inline1554_inline11355__tile_l0_c_acc
            uint64_t v114 = (uint64_t)v22;
            TASSIGN(v113, v114);
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v113, v113, v102, v104);
        }
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        // pto: %15
        Tile<
            TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v115 = Tile<
                TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v14, v18);
        // pto: %15
        uint64_t v116 = (uint64_t)v22;
        TASSIGN(v115, v116);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
        TMATMUL_ACC(v115, v115, v106, v109);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
    }
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    TFREE<TPipe<0, Direction::DIR_BOTH, 16384, 4, 4, true>, TileSplitAxis::TILE_NO_SPLIT>(v35);
    // pto: %52
    int64_t v117 = (int64_t)((uint64_t)v38 * (uint64_t)v13);
    // pto: %53
    int64_t v118 = (int64_t)((uint64_t)v39 + (uint64_t)v117);
    // pto: %t__tile
    Tile<
        TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v119 = Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v13, v18);
    // pto: %t__tile
    uint64_t v120 = (uint64_t)v22;
    TASSIGN(v119, v120);
    // pto: %slice_view
    Tile<
        TileType::Acc, float, 32, 512, BLayout::ColMajor, 16, 512, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v121;
    // pto: %slice_view
    Tile<
        TileType::Acc, float, 32, 512, BLayout::ColMajor, 16, 512, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v122 = v121;
    // pto: %slice_view
    uint64_t v123 = (uint64_t)v22;
    TASSIGN(v122, v123);
    // pto: %sparse_blk_oi_inline1460_inline11347__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 512> v124 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %sparse_blk_oi_inline1460_inline11347__ssa_v0_pview
    pto::Stride<8192, 8192, 8192, 512, 1> v125 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %54, %sparse_blk_oi_inline1460_inline11347__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v126 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
            v5 + (v22 + (v118 < v22 ? v22 : v118) * v18), v124, v125
        );
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
    TSTORE(v126, v122);
    // pto: %55, %57
    int64_t v127 = (int64_t)((uint64_t)((int64_t)((uint64_t)v39 + (uint64_t)v14)) + (uint64_t)v117);
    // pto: %16
    Tile<
        TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v128 = Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v13, v18);
    // pto: %16
    uint64_t v129 = (uint64_t)v22;
    TASSIGN(v128, v129);
    // pto: %58
    Tile<
        TileType::Acc, float, 32, 512, BLayout::ColMajor, 16, 512, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v130;
    // pto: %58
    Tile<
        TileType::Acc, float, 32, 512, BLayout::ColMajor, 16, 512, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v131 = v130;
    // pto: %58
    uint64_t v132 = (uint64_t)v26;
    TASSIGN(v131, v132);
    // pto: %sparse_blk_oi_inline1460_inline11347__tile_pview
    pto::Shape<1, 1, 1, 16, 512> v133 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %sparse_blk_oi_inline1460_inline11347__tile_pview
    pto::Stride<8192, 8192, 8192, 512, 1> v134 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %60, %sparse_blk_oi_inline1460_inline11347__tile_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v135 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
            v5 + (v22 + (v127 < v22 ? v22 : v127) * v18), v133, v134
        );
    pipe_barrier(PIPE_FIX);
    TSTORE(v135, v131);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
    // pto: %17
    Tile<
        TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v136 = Tile<
            TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v14, v18);
    // pto: %17
    uint64_t v137 = (uint64_t)v22;
    TASSIGN(v136, v137);
    // pto: %61
    Tile<
        TileType::Mat, bfloat16_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v138 = Tile<
            TileType::Mat, bfloat16_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v11, v10);
    TPOP<
        TPipe<0, Direction::DIR_BOTH, 16384, 4, 4, true>,
        Tile<
            TileType::Mat, bfloat16_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>,
        TileSplitAxis::TILE_NO_SPLIT>(v35, v138);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    for (int64_t i139 = v22; i139 < v16; i139 += v15) {
        // pto: %18
        Tile<
            TileType::Left, bfloat16_t, 32, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v140 = Tile<
                TileType::Left, bfloat16_t, 32, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v14, v14);
        // pto: %18
        uint64_t v141 = (uint64_t)v22;
        TASSIGN(v140, v141);
        pipe_barrier(PIPE_MTE1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        TEXTRACT(v140, v138, v22, i139);
        // pto: %19
        Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v142 = Tile<
                TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v14, v18);
        // pto: %19
        uint64_t v143 = (uint64_t)v22;
        TASSIGN(v142, v143);
        TEXTRACT(v142, v41, i139, v22);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
        // pto: %20
        Tile<
            TileType::Left, bfloat16_t, 32, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v144 = Tile<
                TileType::Left, bfloat16_t, 32, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v14, v14);
        // pto: %20
        uint64_t v145 = (uint64_t)v21;
        TASSIGN(v144, v145);
        // pto: %63
        int64_t v146 = (int64_t)((uint64_t)i139 + (uint64_t)v14);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v144, v138, v22, v146);
        // pto: %21
        Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v147 = Tile<
                TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v14, v18);
        // pto: %21
        uint64_t v148 = (uint64_t)v20;
        TASSIGN(v147, v148);
        TEXTRACT(v147, v41, v146, v22);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
        // pto: %65
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
        if (i139 == v22) {
            // pto: %22
            Tile<
                TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v149 = Tile<
                    TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v14, v18);
            // pto: %22
            uint64_t v150 = (uint64_t)v22;
            TASSIGN(v149, v150);
            pipe_barrier(PIPE_M);
            TMATMUL(v149, v140, v142);
        } else {
            // pto: %23
            Tile<
                TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v151 = Tile<
                    TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v14, v18);
            // pto: %23
            uint64_t v152 = (uint64_t)v22;
            TASSIGN(v151, v152);
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v151, v151, v140, v142);
        }
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        // pto: %24
        Tile<
            TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v153 = Tile<
                TileType::Acc, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v14, v18);
        // pto: %24
        uint64_t v154 = (uint64_t)v22;
        TASSIGN(v153, v154);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
        TMATMUL_ACC(v153, v153, v144, v147);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    }
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID3);
    TFREE<TPipe<0, Direction::DIR_BOTH, 16384, 4, 4, true>, TileSplitAxis::TILE_NO_SPLIT>(v35);
    // pto: %67, %69
    int64_t v155 = (int64_t)((uint64_t)((int64_t)((uint64_t)v39 + (uint64_t)v15)) + (uint64_t)v117);
    // pto: %25
    Tile<
        TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v156 = Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v13, v18);
    // pto: %25
    uint64_t v157 = (uint64_t)v22;
    TASSIGN(v156, v157);
    // pto: %70
    Tile<
        TileType::Acc, float, 32, 512, BLayout::ColMajor, 16, 512, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v158;
    // pto: %70
    Tile<
        TileType::Acc, float, 32, 512, BLayout::ColMajor, 16, 512, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v159 = v158;
    // pto: %70
    uint64_t v160 = (uint64_t)v22;
    TASSIGN(v159, v160);
    // pto: %73
    pto::Shape<1, 1, 1, 16, 512> v161 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %73
    pto::Stride<8192, 8192, 8192, 512, 1> v162 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %72, %73
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v163 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
            v5 + (v22 + (v155 < v22 ? v22 : v155) * v18), v161, v162
        );
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID3);
    pipe_barrier(PIPE_FIX);
    TSTORE(v163, v159);
    // pto: %74, %76
    int64_t v164 = (int64_t)((uint64_t)((int64_t)((uint64_t)v39 + (uint64_t)v12)) + (uint64_t)v117);
    // pto: %26
    Tile<
        TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v165 = Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v13, v18);
    // pto: %26
    uint64_t v166 = (uint64_t)v22;
    TASSIGN(v165, v166);
    // pto: %77
    Tile<
        TileType::Acc, float, 32, 512, BLayout::ColMajor, 16, 512, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v167;
    // pto: %77
    Tile<
        TileType::Acc, float, 32, 512, BLayout::ColMajor, 16, 512, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v168 = v167;
    // pto: %77
    uint64_t v169 = (uint64_t)v26;
    TASSIGN(v168, v169);
    // pto: %80
    pto::Shape<1, 1, 1, 16, 512> v170 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %80
    pto::Stride<8192, 8192, 8192, 512, 1> v171 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %79, %80
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v172 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
            v5 + (v22 + (v164 < v22 ? v22 : v164) * v18), v170, v171
        );
    pipe_barrier(PIPE_FIX);
    TSTORE(v172, v168);
#endif  // __DAV_CUBE__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}

static __aicore__ void qk_pv_2_aiv(
    __gm__ float *v1, __gm__ bfloat16_t *v2, __gm__ float *v3, __gm__ float *v4, __gm__ float *v5,
    __gm__ bfloat16_t *v6, __gm__ float *v7, int32_t v8, int32_t v9, int32_t v10
) {
    SaturationMode v11 = SaturationMode::OFF;
    RoundMode v12 = RoundMode::CAST_RINT;
    const int32_t v13 = 128;
    const int32_t v14 = 32;
    const int64_t v15 = 256;
    const int64_t v16 = 16;
    const float v17 = 0.0f;
    const float v18 = 0.0441941731f;
    const int64_t v19 = 32;
    const int64_t v20 = 128;
    const int64_t v21 = 2;
    const int64_t v22 = 0;
    const int64_t v23 = 1;
    const int64_t v24 = 65664;
    const int64_t v25 = 65536;
    const int64_t v26 = 82688;
    const int64_t v27 = 66304;
    const int64_t v28 = 65792;
    const int32_t v29 = 0;
    const int64_t v30 = 64;
    const int64_t v31 = 65600;
    const int64_t v32 = 65728;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    auto v33 = TPipe<0, Direction::DIR_BOTH, 16384, 4, 4, true>(v7, v29, v29);
    // pto: %subblock_idx, %23
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    if ((int64_t)v10 == v22) {
        // pto: %qk_item_inline1457_inline11345__ssa_v0
        int64_t v34 = (int64_t)v8;
        // pto: %24
        int64_t v35 = v34 / v21;
        // pto: %26, %25
        int64_t v36 = (int64_t)((uint64_t)v34 - (uint64_t)((int64_t)((uint64_t)v35 * (uint64_t)v21)));
        // pto: %27
        int64_t v37 = (int64_t)((uint64_t)v35 * (uint64_t)v20);
        // pto: %28
        int64_t v38 = (int64_t)((uint64_t)v36 * (uint64_t)v20);
        // pto: %qk_bias_row_inline1537_inline11784__tile
        Tile<
            TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v39 = Tile<
                TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v23, v20);
        // pto: %qk_bias_row_inline1537_inline11784__tile
        uint64_t v40 = (uint64_t)v28;
        TASSIGN(v39, v40);
        // pto: %sparse_bias_inline1492_inline11373__ssa_v2_pview
        pto::Shape<1, 1, 1, 1, 128> v41 = pto::Shape<1, 1, 1, 1, 128>();
        // pto: %sparse_bias_inline1492_inline11373__ssa_v2_pview
        pto::Stride<256, 256, 256, 256, 1> v42 = pto::Stride<256, 256, 256, 256, 1>();
        // pto: %29, %sparse_bias_inline1492_inline11373__ssa_v2_pview, %30
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v43 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(
                v1 + ((v22 + (v35 < v22 ? v22 : v35) * v15) + (v38 < v22 ? v22 : v38)), v41, v42
            );
        TLOAD(v39, v43);
        for (int64_t i44 = v22; i44 < v21; i44 += v23) {
            // pto: %qk_raw_inline1454_inline11337__tile_Vec
            Tile<
                TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v45 = Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v14, v13);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            TPOP<
                TPipe<0, Direction::DIR_BOTH, 16384, 4, 4, true>,
                Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>,
                TileSplitAxis::TILE_NO_SPLIT>(v33, v45);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            // pto: %qk_scaled_inline1533_inline11519__tile
            Tile<
                TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v46 = Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v20);
            // pto: %qk_scaled_inline1533_inline11519__tile
            uint64_t v47 = (uint64_t)v27;
            TASSIGN(v46, v47);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            TMULS(v46, v45, v18);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            TFREE<TPipe<0, Direction::DIR_BOTH, 16384, 4, 4, true>, TileSplitAxis::TILE_NO_SPLIT>(v33);
            // pto: %t__tile
            Tile<
                TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v48 = Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v20);
            // pto: %t__tile
            uint64_t v49 = (uint64_t)v26;
            TASSIGN(v48, v49);
            TEXPANDS(v48, v17);
            // pto: %0
            Tile<
                TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v50 = Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v20);
            // pto: %0
            uint64_t v51 = (uint64_t)v26;
            TASSIGN(v50, v51);
            pipe_barrier(PIPE_V);
            TCOLEXPAND(v50, v39);
            // pto: %qk_scores_inline1538_inline11336__tile
            Tile<
                TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v52 = Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v20);
            // pto: %qk_scores_inline1538_inline11336__tile
            uint64_t v53 = (uint64_t)v27;
            TASSIGN(v52, v53);
            pipe_barrier(PIPE_V);
            TADD(v52, v46, v50);
            // pto: %tmp_tile
            Tile<
                TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v54 = Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v20);
            // pto: %tmp_tile
            uint64_t v55 = (uint64_t)v26;
            TASSIGN(v54, v55);
            // pto: %qk_mi_inline1558_inline11334__tile
            Tile<
                TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v56 = Tile<
                    TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v23);
            // pto: %qk_mi_inline1558_inline11334__tile
            uint64_t v57 = (uint64_t)v25;
            TASSIGN(v56, v57);
            pipe_barrier(PIPE_V);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            TROWMAX(v56, v52, v54);
            // pto: %1
            Tile<
                TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v58 = Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v20);
            // pto: %1
            uint64_t v59 = (uint64_t)v27;
            TASSIGN(v58, v59);
            pipe_barrier(PIPE_V);
            TROWEXPANDSUB(v58, v52, v56);
            // pto: %qk_exp_inline1479_inline11378__tile
            Tile<
                TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v60 = Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v20);
            // pto: %qk_exp_inline1479_inline11378__tile
            uint64_t v61 = (uint64_t)v27;
            TASSIGN(v60, v61);
            pipe_barrier(PIPE_V);
            TEXP(v60, v58);
            // pto: %2
            Tile<
                TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v62 = Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v20);
            // pto: %2
            uint64_t v63 = (uint64_t)v26;
            TASSIGN(v62, v63);
            // pto: %qk_li_inline1524_inline11332__tile
            Tile<
                TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v64 = Tile<
                    TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v23);
            // pto: %qk_li_inline1524_inline11332__tile
            uint64_t v65 = (uint64_t)v24;
            TASSIGN(v64, v65);
            pipe_barrier(PIPE_V);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
            TROWSUM(v64, v60, v62);
            // pto: %qk_exp_bf16_inline1600_inline11331__tile
            Tile<
                TileType::Vec, bfloat16_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v66 = Tile<
                    TileType::Vec, bfloat16_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                    PadValue::Null, CompactMode::Null>(v19, v20);
            // pto: %qk_exp_bf16_inline1600_inline11331__tile
            uint64_t v67 = (uint64_t)v27;
            TASSIGN(v66, v67);
            pipe_barrier(PIPE_V);
            TCVT(v66, v60, v12, v11);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TPUSH<
                TPipe<0, Direction::DIR_BOTH, 16384, 4, 4, true>,
                Tile<
                    TileType::Vec, bfloat16_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                    PadValue::Null, CompactMode::Null>,
                TileSplitAxis::TILE_NO_SPLIT>(v33, v66);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            // pto: %34
            int64_t v68 = (int64_t)((uint64_t)v36 * (uint64_t)v16);
            // pto: %33, %35
            int64_t v69 =
                (int64_t)((uint64_t)((int64_t)((uint64_t)v37 + (uint64_t)((int64_t)((uint64_t)i44 * (uint64_t)v30)))) +
                          (uint64_t)v68);
            // pto: %3
            Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v70 = Tile<
                    TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v16, v23);
            // pto: %3
            uint64_t v71 = (uint64_t)v25;
            TASSIGN(v70, v71);
            // pto: %slice_view
            Tile<
                TileType::Vec, float, 32, 1, BLayout::ColMajor, 16, 1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v72;
            // pto: %slice_view
            Tile<
                TileType::Vec, float, 32, 1, BLayout::ColMajor, 16, 1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v73 = v72;
            // pto: %slice_view
            uint64_t v74 = (uint64_t)v25;
            TASSIGN(v73, v74);
            // pto: %36
            int64_t v75 = v69 < v22 ? v22 : v69;
            // pto: %sparse_blk_mi_inline1555_inline11367__iter_v1_pview
            pto::Shape<1, 1, 1, 16, 1> v76 = pto::Shape<1, 1, 1, 16, 1>();
            // pto: %sparse_blk_mi_inline1555_inline11367__iter_v1_pview
            pto::Stride<16, 16, 16, 1, 1024> v77 = pto::Stride<16, 16, 16, 1, 1024>();
            // pto: %sparse_blk_mi_inline1555_inline11367__iter_v1_pview
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 1024>, pto::Layout::DN> v78 =
                GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 1024>, pto::Layout::DN>(
                    v4 + (v22 + v75), v76, v77
                );
            TSTORE(v78, v73);
            // pto: %4
            Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v79 = Tile<
                    TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v16, v23);
            // pto: %4
            uint64_t v80 = (uint64_t)v24;
            TASSIGN(v79, v80);
            // pto: %37
            Tile<
                TileType::Vec, float, 32, 1, BLayout::ColMajor, 16, 1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v81;
            // pto: %37
            Tile<
                TileType::Vec, float, 32, 1, BLayout::ColMajor, 16, 1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v82 = v81;
            // pto: %37
            uint64_t v83 = (uint64_t)v24;
            TASSIGN(v82, v83);
            // pto: %sparse_blk_li_inline1523_inline11348__iter_v1_pview
            pto::Shape<1, 1, 1, 16, 1> v84 = pto::Shape<1, 1, 1, 16, 1>();
            // pto: %sparse_blk_li_inline1523_inline11348__iter_v1_pview
            pto::Stride<16, 16, 16, 1, 1024> v85 = pto::Stride<16, 16, 16, 1, 1024>();
            // pto: %sparse_blk_li_inline1523_inline11348__iter_v1_pview
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 1024>, pto::Layout::DN> v86 =
                GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 1024>, pto::Layout::DN>(
                    v3 + (v22 + v75), v84, v85
                );
            TSTORE(v86, v82);
            // pto: %42, %31, %40, %41, %44
            int64_t v87 =
                (int64_t)((uint64_t)((int64_t)((uint64_t)v37 +
                                               (uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)
                                                                                                                  i44 *
                                                                                                              (uint64_t)
                                                                                                                  v21)) +
                                                                                         (uint64_t)v23)) *
                                                                    (uint64_t)v19)))) +
                          (uint64_t)v68);
            // pto: %5
            Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v88 = Tile<
                    TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v16, v23);
            // pto: %5
            uint64_t v89 = (uint64_t)v25;
            TASSIGN(v88, v89);
            // pto: %45
            Tile<
                TileType::Vec, float, 32, 1, BLayout::ColMajor, 16, 1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v90;
            // pto: %45
            Tile<
                TileType::Vec, float, 32, 1, BLayout::ColMajor, 16, 1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v91 = v90;
            // pto: %45
            uint64_t v92 = (uint64_t)v31;
            TASSIGN(v91, v92);
            // pto: %47
            int64_t v93 = v87 < v22 ? v22 : v87;
            // pto: %sparse_blk_mi_inline1555_inline11367__tile_pview
            pto::Shape<1, 1, 1, 16, 1> v94 = pto::Shape<1, 1, 1, 16, 1>();
            // pto: %sparse_blk_mi_inline1555_inline11367__tile_pview
            pto::Stride<16, 16, 16, 1, 1024> v95 = pto::Stride<16, 16, 16, 1, 1024>();
            // pto: %sparse_blk_mi_inline1555_inline11367__tile_pview
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 1024>, pto::Layout::DN> v96 =
                GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 1024>, pto::Layout::DN>(
                    v4 + (v22 + v93), v94, v95
                );
            pipe_barrier(PIPE_MTE3);
            TSTORE(v96, v91);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            // pto: %6
            Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v97 = Tile<
                    TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v16, v23);
            // pto: %6
            uint64_t v98 = (uint64_t)v24;
            TASSIGN(v97, v98);
            // pto: %48
            Tile<
                TileType::Vec, float, 32, 1, BLayout::ColMajor, 16, 1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v99;
            // pto: %48
            Tile<
                TileType::Vec, float, 32, 1, BLayout::ColMajor, 16, 1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v100 = v99;
            // pto: %48
            uint64_t v101 = (uint64_t)v32;
            TASSIGN(v100, v101);
            // pto: %sparse_blk_li_inline1523_inline11348__tile_pview
            pto::Shape<1, 1, 1, 16, 1> v102 = pto::Shape<1, 1, 1, 16, 1>();
            // pto: %sparse_blk_li_inline1523_inline11348__tile_pview
            pto::Stride<16, 16, 16, 1, 1024> v103 = pto::Stride<16, 16, 16, 1, 1024>();
            // pto: %sparse_blk_li_inline1523_inline11348__tile_pview
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 1024>, pto::Layout::DN> v104 =
                GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 1024>, pto::Layout::DN>(
                    v3 + (v22 + v93), v102, v103
                );
            TSTORE(v104, v100);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        }
    } else {
        // pto: %7
        Tile<
            TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v105 = Tile<
                TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v22, v22);
        // pto: %7
        uint64_t v106 = (uint64_t)v28;
        TASSIGN(v105, v106);
        for (int64_t i107 = v22; i107 < v21; i107 += v23) {
            // pto: %53
            Tile<
                TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v108 = Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v14, v13);
            v108.SetValidShape(v22, v22);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            TPOP<
                TPipe<0, Direction::DIR_BOTH, 16384, 4, 4, true>,
                Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>,
                TileSplitAxis::TILE_NO_SPLIT>(v33, v108);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            // pto: %8
            Tile<
                TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v109 = Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v22, v22);
            // pto: %8
            uint64_t v110 = (uint64_t)v27;
            TASSIGN(v109, v110);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
            TMULS(v109, v108, v18);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            TFREE<TPipe<0, Direction::DIR_BOTH, 16384, 4, 4, true>, TileSplitAxis::TILE_NO_SPLIT>(v33);
            // pto: %9
            Tile<
                TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v111 = Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v22, v22);
            // pto: %9
            uint64_t v112 = (uint64_t)v26;
            TASSIGN(v111, v112);
            TEXPANDS(v111, v17);
            // pto: %10
            Tile<
                TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v113 = Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v22, v22);
            // pto: %10
            uint64_t v114 = (uint64_t)v26;
            TASSIGN(v113, v114);
            pipe_barrier(PIPE_V);
            TCOLEXPAND(v113, v105);
            // pto: %11
            Tile<
                TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v115 = Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v22, v22);
            // pto: %11
            uint64_t v116 = (uint64_t)v27;
            TASSIGN(v115, v116);
            pipe_barrier(PIPE_V);
            TADD(v115, v109, v113);
            // pto: %12
            Tile<
                TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v117 = Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v22, v22);
            // pto: %12
            uint64_t v118 = (uint64_t)v26;
            TASSIGN(v117, v118);
            // pto: %13
            Tile<
                TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v119 = Tile<
                    TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v22, v22);
            // pto: %13
            uint64_t v120 = (uint64_t)v25;
            TASSIGN(v119, v120);
            pipe_barrier(PIPE_V);
            TROWMAX(v119, v115, v117);
            // pto: %14
            Tile<
                TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v121 = Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v22, v22);
            // pto: %14
            uint64_t v122 = (uint64_t)v27;
            TASSIGN(v121, v122);
            pipe_barrier(PIPE_V);
            TROWEXPANDSUB(v121, v115, v119);
            // pto: %15
            Tile<
                TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v123 = Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v22, v22);
            // pto: %15
            uint64_t v124 = (uint64_t)v27;
            TASSIGN(v123, v124);
            pipe_barrier(PIPE_V);
            TEXP(v123, v121);
            // pto: %16
            Tile<
                TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v125 = Tile<
                    TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v22, v22);
            // pto: %16
            uint64_t v126 = (uint64_t)v26;
            TASSIGN(v125, v126);
            // pto: %17
            Tile<
                TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v127 = Tile<
                    TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v22, v22);
            // pto: %17
            uint64_t v128 = (uint64_t)v25;
            TASSIGN(v127, v128);
            pipe_barrier(PIPE_V);
            TROWSUM(v127, v123, v125);
            // pto: %18
            Tile<
                TileType::Vec, bfloat16_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v129 = Tile<
                    TileType::Vec, bfloat16_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                    PadValue::Null, CompactMode::Null>(v22, v22);
            // pto: %18
            uint64_t v130 = (uint64_t)v27;
            TASSIGN(v129, v130);
            pipe_barrier(PIPE_V);
            TCVT(v129, v123, v12, v11);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            TPUSH<
                TPipe<0, Direction::DIR_BOTH, 16384, 4, 4, true>,
                Tile<
                    TileType::Vec, bfloat16_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                    PadValue::Null, CompactMode::Null>,
                TileSplitAxis::TILE_NO_SPLIT>(v33, v129);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
            // pto: %19
            Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v131 = Tile<
                    TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v22, v22);
            // pto: %19
            uint64_t v132 = (uint64_t)v25;
            TASSIGN(v131, v132);
            // pto: %20
            Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v133 = Tile<
                    TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v22, v22);
            // pto: %20
            uint64_t v134 = (uint64_t)v25;
            TASSIGN(v133, v134);
            // pto: %21
            Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v135 = Tile<
                    TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v22, v22);
            // pto: %21
            uint64_t v136 = (uint64_t)v25;
            TASSIGN(v135, v136);
            // pto: %22
            Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v137 = Tile<
                    TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v22, v22);
            // pto: %22
            uint64_t v138 = (uint64_t)v25;
            TASSIGN(v137, v138);
        }
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: sparse_bias_inline1492_inline11373__ssa_v2
    __gm__ Tensor *sparse_bias_inline1492_inline11373__ssa_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *sparse_bias_inline1492_inline11373__ssa_v2 =
        reinterpret_cast<__gm__ float *>(sparse_bias_inline1492_inline11373__ssa_v2_tensor->buffer.addr) +
        sparse_bias_inline1492_inline11373__ssa_v2_tensor->start_offset;

    // Unpack tensor: hca_kv_flat_inline1485_inline11368__rv_v11
    __gm__ Tensor *hca_kv_flat_inline1485_inline11368__rv_v11_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ bfloat16_t *hca_kv_flat_inline1485_inline11368__rv_v11 =
        reinterpret_cast<__gm__ bfloat16_t *>(hca_kv_flat_inline1485_inline11368__rv_v11_tensor->buffer.addr) +
        hca_kv_flat_inline1485_inline11368__rv_v11_tensor->start_offset;

    // Unpack tensor: sparse_blk_li_inline1523_inline11348__ssa_v0
    __gm__ Tensor *sparse_blk_li_inline1523_inline11348__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *sparse_blk_li_inline1523_inline11348__ssa_v0 =
        reinterpret_cast<__gm__ float *>(sparse_blk_li_inline1523_inline11348__ssa_v0_tensor->buffer.addr) +
        sparse_blk_li_inline1523_inline11348__ssa_v0_tensor->start_offset;

    // Unpack tensor: sparse_blk_mi_inline1555_inline11367__ssa_v0
    __gm__ Tensor *sparse_blk_mi_inline1555_inline11367__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ float *sparse_blk_mi_inline1555_inline11367__ssa_v0 =
        reinterpret_cast<__gm__ float *>(sparse_blk_mi_inline1555_inline11367__ssa_v0_tensor->buffer.addr) +
        sparse_blk_mi_inline1555_inline11367__ssa_v0_tensor->start_offset;

    // Unpack tensor: sparse_blk_oi_inline1460_inline11347__ssa_v0
    __gm__ Tensor *sparse_blk_oi_inline1460_inline11347__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ float *sparse_blk_oi_inline1460_inline11347__ssa_v0 =
        reinterpret_cast<__gm__ float *>(sparse_blk_oi_inline1460_inline11347__ssa_v0_tensor->buffer.addr) +
        sparse_blk_oi_inline1460_inline11347__ssa_v0_tensor->start_offset;

    // Unpack tensor: q_flat_inline1542_inline11350__ssa_v0
    __gm__ Tensor *q_flat_inline1542_inline11350__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[5]);
    __gm__ bfloat16_t *q_flat_inline1542_inline11350__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(q_flat_inline1542_inline11350__ssa_v0_tensor->buffer.addr) +
        q_flat_inline1542_inline11350__ssa_v0_tensor->start_offset;

    // Unpack tensor: __gm_pipe_buffer
    __gm__ Tensor *__gm_pipe_buffer_tensor = reinterpret_cast<__gm__ Tensor *>(args[6]);
    // SPMD: shard GM pipe workspace by logical block_idx to avoid overlap.
    int64_t __pypto_gm_block_num = static_cast<int64_t>(__pypto_spmd_block_num);
    if (__pypto_gm_block_num <= 0) __pypto_gm_block_num = 1;
    int64_t __pypto_gm_total_elems = static_cast<int64_t>(__gm_pipe_buffer_tensor->shapes[0]);
    int64_t __pypto_gm_elems_per_block = __pypto_gm_total_elems / __pypto_gm_block_num;
    int64_t __pypto_gm_block_offset = static_cast<int64_t>(__pypto_spmd_block_idx) * __pypto_gm_elems_per_block;
    __gm__ float *__gm_pipe_buffer = reinterpret_cast<__gm__ float *>(__gm_pipe_buffer_tensor->buffer.addr) +
                                     __gm_pipe_buffer_tensor->start_offset + __pypto_gm_block_offset;

    // Forward to ptoas-generated function
    qk_pv_2_aic(
        sparse_bias_inline1492_inline11373__ssa_v2, hca_kv_flat_inline1485_inline11368__rv_v11,
        sparse_blk_li_inline1523_inline11348__ssa_v0, sparse_blk_mi_inline1555_inline11367__ssa_v0,
        sparse_blk_oi_inline1460_inline11347__ssa_v0, q_flat_inline1542_inline11350__ssa_v0, __gm_pipe_buffer,
        __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
