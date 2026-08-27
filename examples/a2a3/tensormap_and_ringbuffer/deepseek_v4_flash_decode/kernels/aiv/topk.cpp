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
// Kernel Function: topk

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
topk(__gm__ int32_t *v1, __gm__ int32_t *v2, __gm__ int32_t *v3, __gm__ float *v4, int32_t v5, int32_t v6) {
    const int32_t v7 = 0;
    const float v8 = -3.40282347E+38f;
    const int32_t v9 = -1;
    const int64_t v10 = 2;
    const int64_t v11 = 4;
    const int64_t v12 = 1;
    const int64_t v13 = 4096;
    const int64_t v14 = 8;
    const int64_t v15 = 49152;
    const int64_t v16 = 65536;
    const int64_t v17 = 32768;
    const int64_t v18 = 0;
    const int64_t v19 = 8192;
    const int32_t v20 = 64;
    const int32_t v21 = 256;
    const int32_t v22 = 1024;
    const int64_t v23 = 1024;
    const int64_t v24 = 2048;
    const int64_t v25 = 512;
    pto::MrgSortExecutedNumList v26 = pto::MrgSortExecutedNumList{0, 0, 0, 0};
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %c0_ui32
    uint32_t v27 = (uint32_t)v7;
    // pto: %topk_idxs_flat_inline1930_inline10737__ssa_v0_view
    int64_t v28 = v14 * v13;
    // pto: %topk_idxs_flat_inline1930_inline10737__ssa_v0_view
    int64_t v29 = v12 * v28;
    // pto: %topk_idxs_flat_inline1930_inline10737__ssa_v0_view
    pto::Shape<1, 1, 1, -1, -1> v30 = pto::Shape<1, 1, 1, -1, -1>(v12, v12, v12, v14, v13);
    // pto: %topk_idxs_flat_inline1930_inline10737__ssa_v0_view
    pto::Stride<-1, -1, -1, -1, -1> v31 = pto::Stride<-1, -1, -1, -1, -1>(v12 * v29, v29, v28, v13, v12);
    // pto: %topk_idxs_flat_inline1930_inline10737__ssa_v0_view
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v32 =
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(
            v1, v30, v31
        );
    // pto: %t_inline1992_inline10416__ssa_v0
    int64_t v33 = (int64_t)v5;
    // pto: %invalid_idxs_inline1947_inline10420__tile
    Tile<
        TileType::Vec, int32_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v34 = Tile<
            TileType::Vec, int32_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v12, v13);
    // pto: %invalid_idxs_inline1947_inline10420__tile
    uint64_t v35 = (uint64_t)v18;
    TASSIGN(v34, v35);
    TEXPANDS(v34, v9);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %2
    int64_t v36 = v33 < v18 ? v18 : v33;
    // pto: %topk_idxs_flat_inline1930_inline10737__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 4096> v37 = pto::Shape<1, 1, 1, 1, 4096>();
    // pto: %topk_idxs_flat_inline1930_inline10737__ssa_v0_pview
    pto::Stride<4096, 4096, 4096, 4096, 1> v38 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %topk_idxs_flat_inline1930_inline10737__ssa_v0_pview
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v39 =
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v1 + (v18 + v36 * v13), v37, v38
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v39, v34);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    // pto: %3, %6
    int32_t v40 = (v2)[v33 / v10];
    // pto: %7, %8
    int64_t v41 = (int64_t)v40 / v11;
    // pto: %pos_t_inline2003_inline10827__tile
    int32_t v42 = (v3)[v33];
    // pto: %9, %10, %11
    int64_t v43 = (int64_t)((uint64_t)((int64_t)v42) + (uint64_t)v12) / v11;
    // pto: %12
    int64_t v44 = v41 < v43 ? v41 : v43;
    // pto: %13
    int64_t v45 = v44 < v13 ? v44 : v13;
    // pto: %14
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    if (v45 > v18) {
        // pto: %score_full_raw_inline1943_inline10847__tile
        Tile<
            TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v46 = Tile<
                TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v13);
        // pto: %score_full_raw_inline1943_inline10847__tile
        uint64_t v47 = (uint64_t)v18;
        TASSIGN(v46, v47);
        // pto: %score_flat_inline1948_inline10368__rv_v2_pview
        pto::Shape<1, 1, 1, 1, 4096> v48 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %score_flat_inline1948_inline10368__rv_v2_pview
        pto::Stride<4096, 4096, 4096, 4096, 1> v49 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %score_flat_inline1948_inline10368__rv_v2_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v50 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
                v4 + (v18 + v36 * v13), v48, v49
            );
        TLOAD(v46, v50);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        v46.SetValidShape(v12, v45);
        // pto: %score_full_inline1927_inline10776__tile
        Tile<
            TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>
            v51 = Tile<
                TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
                CompactMode::Null>(v12, v13);
        // pto: %score_full_inline1927_inline10776__tile
        uint64_t v52 = (uint64_t)v18;
        TASSIGN(v51, v52);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TFILLPAD(v51, v46);
        // pto: %0
        Tile<
            TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v53 = Tile<
                TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v13);
        // pto: %0
        uint64_t v54 = (uint64_t)v17;
        TASSIGN(v53, v54);
        TEXPANDS(v53, v8);
        // pto: %score_full_v1_inline1926_inline10785__tile
        Tile<
            TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>
            v55 = Tile<
                TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
                CompactMode::Null>(v12, v13);
        // pto: %score_full_v1_inline1926_inline10785__tile
        uint64_t v56 = (uint64_t)v17;
        TASSIGN(v55, v56);
        pipe_barrier(PIPE_V);
        TMAX(v55, v51, v53);
        // pto: %idx_init_inline1925_inline10789__tile
        Tile<
            TileType::Vec, uint32_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v57 = Tile<
                TileType::Vec, uint32_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v13);
        // pto: %idx_init_inline1925_inline10789__tile
        uint64_t v58 = (uint64_t)v16;
        TASSIGN(v57, v58);
        TCI<Tile<
                TileType::Vec, uint32_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>,
            uint32_t, 0>(v57, v27);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        // pto: %sorted_full_inline1987_inline10739__tile
        Tile<
            TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>
            v59 = Tile<
                TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
                CompactMode::Null>(v12, v19);
        // pto: %sorted_full_inline1987_inline10739__tile
        uint64_t v60 = (uint64_t)v18;
        TASSIGN(v59, v60);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        TSORT32(v59, v55, v57);
        // pto: %sorted_full_v1_inline1924_inline10850__tile
        Tile<
            TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>
            v61 = Tile<
                TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
                CompactMode::Null>(v12, v19);
        // pto: %sorted_full_v1_inline1924_inline10850__tile
        uint64_t v62 = (uint64_t)v17;
        TASSIGN(v61, v62);
        pipe_barrier(PIPE_V);
        TMRGSORT(v61, v59, v20);
        // pto: %sorted_full_v2_inline1923_inline10810__tile
        Tile<
            TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>
            v63 = Tile<
                TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
                CompactMode::Null>(v12, v19);
        // pto: %sorted_full_v2_inline1923_inline10810__tile
        uint64_t v64 = (uint64_t)v18;
        TASSIGN(v63, v64);
        pipe_barrier(PIPE_V);
        TMRGSORT(v63, v61, v21);
        // pto: %sorted_full_v3_inline2007_inline10371__tile
        Tile<
            TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>
            v65 = Tile<
                TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
                CompactMode::Null>(v12, v19);
        // pto: %sorted_full_v3_inline2007_inline10371__tile
        uint64_t v66 = (uint64_t)v17;
        TASSIGN(v65, v66);
        pipe_barrier(PIPE_V);
        TMRGSORT(v65, v63, v22);
        // pto: %half0_candidates_inline1959_inline10851__tile
        Tile<
            TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>
            v67 = Tile<
                TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
                CompactMode::Null>(v12, v23);
        // pto: %half0_candidates_inline1959_inline10851__tile
        uint64_t v68 = (uint64_t)v17;
        TASSIGN(v67, v68);
        // pto: %slice_view
        Tile<
            TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>
            v69;
        // pto: %slice_view
        Tile<
            TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>
            v70 = v69;
        // pto: %slice_view
        uint64_t v71 = (uint64_t)v17;
        TASSIGN(v70, v71);
        // pto: %half1_candidates_inline1922_inline10854__tile
        Tile<
            TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>
            v72 = Tile<
                TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
                CompactMode::Null>(v12, v23);
        // pto: %half1_candidates_inline1922_inline10854__tile
        uint64_t v73 = (uint64_t)v15;
        TASSIGN(v72, v73);
        // pto: %18
        Tile<
            TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>
            v74;
        // pto: %18
        Tile<
            TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>
            v75 = v74;
        // pto: %18
        uint64_t v76 = (uint64_t)v15;
        TASSIGN(v75, v76);
        // pto: %mrgsort2_tmp
        Tile<
            TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v77 = Tile<
                TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v24);
        // pto: %mrgsort2_tmp
        uint64_t v78 = (uint64_t)v18;
        TASSIGN(v77, v78);
        // pto: %merged_candidates_inline1921_inline10551__tile
        Tile<
            TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>
            v79 = Tile<
                TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
                CompactMode::Null>(v12, v24);
        // pto: %merged_candidates_inline1921_inline10551__tile
        uint64_t v80 = (uint64_t)v16;
        TASSIGN(v79, v80);
        pipe_barrier(PIPE_V);
        TMRGSORT<
            Tile<
                TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
                CompactMode::Null>,
            Tile<
                TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>,
            Tile<
                TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min,
                CompactMode::Null>,
            Tile<
                TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min,
                CompactMode::Null>,
            false>(v79, v26, v77, v70, v75);
        // pto: %topk_pairs_inline2021_inline10765__tile
        Tile<
            TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>
            v81 = Tile<
                TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
                CompactMode::Null>(v12, v23);
        // pto: %topk_pairs_inline2021_inline10765__tile
        uint64_t v82 = (uint64_t)v16;
        TASSIGN(v81, v82);
        // pto: %19
        Tile<
            TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>
            v83;
        // pto: %19
        Tile<
            TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>
            v84 = v83;
        // pto: %19
        uint64_t v85 = (uint64_t)v16;
        TASSIGN(v84, v85);
        // pto: %topk_idxs_tile_inline1920_inline10508__tile
        Tile<
            TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>
            v86 = Tile<
                TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
                CompactMode::Null>(v12, v25);
        // pto: %topk_idxs_tile_inline1920_inline10508__tile
        uint64_t v87 = (uint64_t)v18;
        TASSIGN(v86, v87);
        pipe_barrier(PIPE_V);
        TGATHER<
            Tile<
                TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
                CompactMode::Null>,
            Tile<
                TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min,
                CompactMode::Null>,
            MaskPattern::P1010>(v86, v84);
        // pto: %20
        int64_t v88 = v45 < v25 ? v45 : v25;
        v86.SetValidShape(v12, v88);
        // pto: %1
        Tile<
            TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>
            v89 = Tile<
                TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
                CompactMode::Null>(v12, v88);
        // pto: %1
        uint64_t v90 = (uint64_t)v18;
        TASSIGN(v89, v90);
        pipe_barrier(PIPE_V);
        TADDS(v89, v86, v7);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        // pto: %topk_idxs_flat_inline1930_inline10737__tile_pview
        __gm__ int32_t *v91 = PTOAS__GLOBAL_TENSOR_DATA(v32);
        // pto: %topk_idxs_flat_inline1930_inline10737__tile_pview
        int64_t v92 = v12 * v13;
        // pto: %topk_idxs_flat_inline1930_inline10737__tile_pview
        int64_t v93 = v12 * v92;
        // pto: %topk_idxs_flat_inline1930_inline10737__tile_pview
        pto::Shape<1, 1, 1, 1, -1> v94 = pto::Shape<1, 1, 1, 1, -1>(v12, v12, v12, v12, v88);
        // pto: %topk_idxs_flat_inline1930_inline10737__tile_pview
        pto::Stride<-1, -1, -1, -1, -1> v95 = pto::Stride<-1, -1, -1, -1, -1>(v12 * v93, v93, v92, v13, v12);
        // pto: %topk_idxs_flat_inline1930_inline10737__tile_pview
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v96 =
            GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(
                v91 + ((v18 + v36 * v13) + v18 * v12), v94, v95
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSTORE(v96, v89);
    }
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: topk_idxs_flat_inline1930_inline10737__ssa_v0
    __gm__ Tensor *topk_idxs_flat_inline1930_inline10737__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int32_t *topk_idxs_flat_inline1930_inline10737__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(topk_idxs_flat_inline1930_inline10737__ssa_v0_tensor->buffer.addr) +
        topk_idxs_flat_inline1930_inline10737__ssa_v0_tensor->start_offset;

    // Unpack tensor: kv_seq_lens__ssa_v0
    __gm__ Tensor *kv_seq_lens__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int32_t *kv_seq_lens__ssa_v0 = reinterpret_cast<__gm__ int32_t *>(kv_seq_lens__ssa_v0_tensor->buffer.addr) +
                                          kv_seq_lens__ssa_v0_tensor->start_offset;

    // Unpack tensor: position_ids_bsd_inline10328__ssa_v0
    __gm__ Tensor *position_ids_bsd_inline10328__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ int32_t *position_ids_bsd_inline10328__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(position_ids_bsd_inline10328__ssa_v0_tensor->buffer.addr) +
        position_ids_bsd_inline10328__ssa_v0_tensor->start_offset;

    // Unpack tensor: score_flat_inline1948_inline10368__rv_v2
    __gm__ Tensor *score_flat_inline1948_inline10368__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ float *score_flat_inline1948_inline10368__rv_v2 =
        reinterpret_cast<__gm__ float *>(score_flat_inline1948_inline10368__rv_v2_tensor->buffer.addr) +
        score_flat_inline1948_inline10368__rv_v2_tensor->start_offset;

    // Forward to ptoas-generated function
    topk(
        topk_idxs_flat_inline1930_inline10737__ssa_v0, kv_seq_lens__ssa_v0, position_ids_bsd_inline10328__ssa_v0,
        score_flat_inline1948_inline10368__rv_v2, __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
