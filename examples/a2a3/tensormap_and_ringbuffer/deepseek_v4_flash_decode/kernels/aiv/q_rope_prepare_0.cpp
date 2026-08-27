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
// Kernel Function: q_rope_prepare_0

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

static __aicore__ void q_rope_prepare_0(
    __gm__ bfloat16_t *v1, __gm__ bfloat16_t *v2, __gm__ float *v3, __gm__ float *v4, __gm__ int32_t *v5, int64_t v6,
    int32_t v7, int32_t v8
) {
    RoundMode v9 = RoundMode::CAST_TRUNC;
    SaturationMode v10 = SaturationMode::OFF;
    RoundMode v11 = RoundMode::CAST_ROUND;
    const float v12 = 2.0f;
    const float v13 = 0.5f;
    const int32_t v14 = 0;
    const float v15 = 1.0f;
    const int64_t v16 = 8;
    const int64_t v17 = 1;
    const int64_t v18 = 64;
    const int64_t v19 = 8448;
    const int64_t v20 = 8192;
    const int64_t v21 = 6144;
    const int64_t v22 = 4096;
    const int64_t v23 = 2048;
    const int64_t v24 = 0;
    const int64_t v25 = 10752;
    const int64_t v26 = 8704;
    const int64_t v27 = 256;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %qrp_idx_inline838_inline9691__ssa_v0, %7
    int64_t v28 = (int64_t)((uint64_t)((int64_t)v7) * (uint64_t)v16);
    // pto: %qrp_ones_inline847_inline9863__tile
    Tile<
        TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v29 = Tile<
            TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qrp_ones_inline847_inline9863__tile
    uint64_t v30 = (uint64_t)v26;
    TASSIGN(v29, v30);
    TEXPANDS(v29, v15);
    // pto: %qrp_idx_i32_inline857_inline9726__tile
    Tile<
        TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v31 = Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v18);
    // pto: %qrp_idx_i32_inline857_inline9726__tile
    uint64_t v32 = (uint64_t)v25;
    TASSIGN(v31, v32);
    TCI<Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>,
        int32_t, 0>(v31, v14);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    // pto: %qrp_idx_fp32_inline885_inline9746__tile
    Tile<
        TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v33 = Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v18);
    // pto: %qrp_idx_fp32_inline885_inline9746__tile
    uint64_t v34 = (uint64_t)v25;
    TASSIGN(v33, v34);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    TCVT(v33, v31, v11, v10);
    // pto: %qrp_col_inline802_inline9636__tile
    Tile<
        TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v35 = Tile<
            TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qrp_col_inline802_inline9636__tile
    uint64_t v36 = (uint64_t)v26;
    TASSIGN(v35, v36);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v35, v29, v33);
    // pto: %qrp_half_inline851_inline9663__tile
    Tile<
        TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v37 = Tile<
            TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qrp_half_inline851_inline9663__tile
    uint64_t v38 = (uint64_t)v25;
    TASSIGN(v37, v38);
    pipe_barrier(PIPE_V);
    TMULS(v37, v35, v13);
    // pto: %qrp_dup_i32_inline895_inline9672__tile
    Tile<
        TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v39 = Tile<
            TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qrp_dup_i32_inline895_inline9672__tile
    uint64_t v40 = (uint64_t)v25;
    TASSIGN(v39, v40);
    pipe_barrier(PIPE_V);
    TCVT(v39, v37, v9, v10);
    // pto: %qrp_dup_f_inline855_inline9647__tile
    Tile<
        TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v41 = Tile<
            TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qrp_dup_f_inline855_inline9647__tile
    uint64_t v42 = (uint64_t)v25;
    TASSIGN(v41, v42);
    pipe_barrier(PIPE_V);
    TCVT(v41, v39, v11, v10);
    // pto: %qrp_dup_idx_inline852_inline9683__tile
    Tile<
        TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v43 = Tile<
            TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qrp_dup_idx_inline852_inline9683__tile
    uint64_t v44 = (uint64_t)v24;
    TASSIGN(v43, v44);
    pipe_barrier(PIPE_V);
    TCVT(v43, v41, v11, v10);
    // pto: %t__tile
    Tile<
        TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v45 = Tile<
            TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %t__tile
    uint64_t v46 = (uint64_t)v25;
    TASSIGN(v45, v46);
    pipe_barrier(PIPE_V);
    TMULS(v45, v41, v12);
    // pto: %qrp_lane_inline888_inline9665__tile
    Tile<
        TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v47 = Tile<
            TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qrp_lane_inline888_inline9665__tile
    uint64_t v48 = (uint64_t)v25;
    TASSIGN(v47, v48);
    pipe_barrier(PIPE_V);
    TSUB(v47, v35, v45);
    // pto: %qrp_next_col_inline925_inline9649__tile
    Tile<
        TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v49 = Tile<
            TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qrp_next_col_inline925_inline9649__tile
    uint64_t v50 = (uint64_t)v26;
    TASSIGN(v49, v50);
    pipe_barrier(PIPE_V);
    TADDS(v49, v35, v15);
    // pto: %qrp_lane_offset_inline860_inline9747__tile
    Tile<
        TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v51 = Tile<
            TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qrp_lane_offset_inline860_inline9747__tile
    uint64_t v52 = (uint64_t)v23;
    TASSIGN(v51, v52);
    TMULS(v51, v47, v12);
    // pto: %qrp_swap_f_inline861_inline9748__tile
    Tile<
        TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v53 = Tile<
            TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qrp_swap_f_inline861_inline9748__tile
    uint64_t v54 = (uint64_t)v26;
    TASSIGN(v53, v54);
    pipe_barrier(PIPE_V);
    TSUB(v53, v49, v51);
    // pto: %qrp_swap_idx_inline800_inline9581__tile
    Tile<
        TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v55 = Tile<
            TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qrp_swap_idx_inline800_inline9581__tile
    uint64_t v56 = (uint64_t)v26;
    TASSIGN(v55, v56);
    pipe_barrier(PIPE_V);
    TCVT(v55, v53, v11, v10);
    // pto: %0
    Tile<
        TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v57 = Tile<
            TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %0
    uint64_t v58 = (uint64_t)v25;
    TASSIGN(v57, v58);
    TMULS(v57, v47, v12);
    // pto: %qrp_sign_inline845_inline9618__tile
    Tile<
        TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v59 = Tile<
            TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qrp_sign_inline845_inline9618__tile
    uint64_t v60 = (uint64_t)v25;
    TASSIGN(v59, v60);
    pipe_barrier(PIPE_V);
    TSUBS(v59, v57, v15);
    // pto: %qrp_cos_rows_inline866_inline9643__tile
    Tile<
        TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v61 = Tile<
            TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qrp_cos_rows_inline866_inline9643__tile
    uint64_t v62 = (uint64_t)v22;
    TASSIGN(v61, v62);
    // pto: %8
    int64_t v63 = v28 < v24 ? v24 : v28;
    // pto: %rope_cos_view_inline825_inline9697__ssa_v0_pview
    pto::Shape<1, 1, 1, 8, 64> v64 = pto::Shape<1, 1, 1, 8, 64>();
    // pto: %rope_cos_view_inline825_inline9697__ssa_v0_pview
    pto::Stride<512, 512, 512, 64, 1> v65 = pto::Stride<512, 512, 512, 64, 1>();
    // pto: %rope_cos_view_inline825_inline9697__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v66 =
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(
            v1 + (v24 + v63 * v18), v64, v65
        );
    TLOAD(v61, v66);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %qrp_sin_rows_inline826_inline9617__tile
    Tile<
        TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v67 = Tile<
            TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qrp_sin_rows_inline826_inline9617__tile
    uint64_t v68 = (uint64_t)v21;
    TASSIGN(v67, v68);
    // pto: %rope_sin_view_inline813_inline9668__ssa_v0_pview
    pto::Shape<1, 1, 1, 8, 64> v69 = pto::Shape<1, 1, 1, 8, 64>();
    // pto: %rope_sin_view_inline813_inline9668__ssa_v0_pview
    pto::Stride<512, 512, 512, 64, 1> v70 = pto::Stride<512, 512, 512, 64, 1>();
    // pto: %rope_sin_view_inline813_inline9668__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v71 =
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(
            v2 + (v24 + v63 * v18), v69, v70
        );
    TLOAD(v67, v71);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %qrp_cos_inline817_inline9654__tile
    Tile<
        TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v72 = Tile<
            TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qrp_cos_inline817_inline9654__tile
    uint64_t v73 = (uint64_t)v23;
    TASSIGN(v72, v73);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(v72, v61, v11, v10);
    // pto: %qrp_sin_inline815_inline9688__tile
    Tile<
        TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v74 = Tile<
            TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qrp_sin_inline815_inline9688__tile
    uint64_t v75 = (uint64_t)v22;
    TASSIGN(v74, v75);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TCVT(v74, v67, v11, v10);
    // pto: %gather_acc_init
    Tile<
        TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v76 = Tile<
            TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %gather_acc_init
    uint64_t v77 = (uint64_t)v21;
    TASSIGN(v76, v77);
    for (int64_t i78 = v24; i78 < v16; i78 += v17) {
        // pto: %gather_inp_row
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v79 = Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v18);
        // pto: %gather_inp_row
        uint64_t v80 = (uint64_t)v23;
        TASSIGN(v79, v80);
        // pto: %slice_view
        int64_t v81 = (int64_t)((uint64_t)i78 * (uint64_t)v27);
        // pto: %slice_view
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v82;
        // pto: %slice_view
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v83 = v82;
        // pto: %slice_view
        uint64_t v84 = (uint64_t)((int64_t)((uint64_t)v81 + (uint64_t)v23));
        TASSIGN(v83, v84);
        // pto: %gather_idx_row
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v85 = Tile<
                TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v18);
        // pto: %gather_idx_row
        uint64_t v86 = (uint64_t)v24;
        TASSIGN(v85, v86);
        // pto: %10
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v87;
        // pto: %10
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v88 = v87;
        // pto: %10
        uint64_t v89 = (uint64_t)v81;
        TASSIGN(v88, v89);
        // pto: %gather_row_tmp
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v90 = Tile<
                TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v18);
        // pto: %gather_row_tmp
        uint64_t v91 = (uint64_t)v20;
        TASSIGN(v90, v91);
        // pto: %gather_row
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v92 = Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v18);
        // pto: %gather_row
        uint64_t v93 = (uint64_t)v19;
        TASSIGN(v92, v93);
        pipe_barrier(PIPE_V);
        TGATHER(v92, v83, v88, v90);
        // pto: %assemble_view
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v94;
        // pto: %assemble_view
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v95 = v94;
        // pto: %assemble_view
        uint64_t v96 = (uint64_t)((int64_t)((uint64_t)v81 + (uint64_t)v21));
        TASSIGN(v95, v96);
        pipe_barrier(PIPE_V);
        TMOV(v95, v92);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %qrp_cos_il_inline843_inline9719__tile
    Tile<
        TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v97 = Tile<
            TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qrp_cos_il_inline843_inline9719__tile
    uint64_t v98 = (uint64_t)v21;
    TASSIGN(v97, v98);
    // pto: %1
    Tile<
        TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v99 = Tile<
            TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %1
    uint64_t v100 = (uint64_t)v23;
    TASSIGN(v99, v100);
    for (int64_t i101 = v24; i101 < v16; i101 += v17) {
        // pto: %2
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v102 = Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v18);
        // pto: %2
        uint64_t v103 = (uint64_t)v22;
        TASSIGN(v102, v103);
        // pto: %12
        int64_t v104 = (int64_t)((uint64_t)i101 * (uint64_t)v27);
        // pto: %12
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v105;
        // pto: %12
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v106 = v105;
        // pto: %12
        uint64_t v107 = (uint64_t)((int64_t)((uint64_t)v104 + (uint64_t)v22));
        TASSIGN(v106, v107);
        // pto: %3
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v108 = Tile<
                TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v18);
        // pto: %3
        uint64_t v109 = (uint64_t)v24;
        TASSIGN(v108, v109);
        // pto: %13
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v110;
        // pto: %13
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v111 = v110;
        // pto: %13
        uint64_t v112 = (uint64_t)v104;
        TASSIGN(v111, v112);
        // pto: %4
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v113 = Tile<
                TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v18);
        // pto: %4
        uint64_t v114 = (uint64_t)v20;
        TASSIGN(v113, v114);
        // pto: %5
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v115 = Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v18);
        // pto: %5
        uint64_t v116 = (uint64_t)v19;
        TASSIGN(v115, v116);
        pipe_barrier(PIPE_V);
        TGATHER(v115, v106, v111, v113);
        // pto: %14
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v117;
        // pto: %14
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v118 = v117;
        // pto: %14
        uint64_t v119 = (uint64_t)((int64_t)((uint64_t)v104 + (uint64_t)v23));
        TASSIGN(v118, v119);
        pipe_barrier(PIPE_V);
        TMOV(v118, v115);
    }
    // pto: %qrp_sin_il_inline850_inline9614__tile
    Tile<
        TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v120 = Tile<
            TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qrp_sin_il_inline850_inline9614__tile
    uint64_t v121 = (uint64_t)v23;
    TASSIGN(v120, v121);
    // pto: %qrp_sin_signed_inline810_inline9736__tile
    Tile<
        TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v122 = Tile<
            TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v18);
    // pto: %qrp_sin_signed_inline810_inline9736__tile
    uint64_t v123 = (uint64_t)v25;
    TASSIGN(v122, v123);
    pipe_barrier(PIPE_V);
    TMUL(v122, v120, v59);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %q_rope_cos_il_inline862_inline9639__ssa_v0_pview
    pto::Shape<1, 1, 1, 8, 64> v124 = pto::Shape<1, 1, 1, 8, 64>();
    // pto: %q_rope_cos_il_inline862_inline9639__ssa_v0_pview
    pto::Stride<512, 512, 512, 64, 1> v125 = pto::Stride<512, 512, 512, 64, 1>();
    // pto: %q_rope_cos_il_inline862_inline9639__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v126 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(
            v3 + (v24 + v63 * v18), v124, v125
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v126, v97);
    // pto: %q_rope_sin_signed_inline898_inline9722__ssa_v0_pview
    pto::Shape<1, 1, 1, 8, 64> v127 = pto::Shape<1, 1, 1, 8, 64>();
    // pto: %q_rope_sin_signed_inline898_inline9722__ssa_v0_pview
    pto::Stride<512, 512, 512, 64, 1> v128 = pto::Stride<512, 512, 512, 64, 1>();
    // pto: %q_rope_sin_signed_inline898_inline9722__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v129 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(
            v4 + (v24 + v63 * v18), v127, v128
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v129, v122);
    // pto: %q_rope_swap_idx_inline864_inline9696__ssa_v0_pview
    pto::Shape<1, 1, 1, 8, 64> v130 = pto::Shape<1, 1, 1, 8, 64>();
    // pto: %q_rope_swap_idx_inline864_inline9696__ssa_v0_pview
    pto::Stride<512, 512, 512, 64, 1> v131 = pto::Stride<512, 512, 512, 64, 1>();
    // pto: %q_rope_swap_idx_inline864_inline9696__ssa_v0_pview
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v132 =
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(
            v5 + (v24 + v63 * v18), v130, v131
        );
    TSTORE(v132, v55);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: rope_cos_view_inline825_inline9697__ssa_v0
    __gm__ Tensor *rope_cos_view_inline825_inline9697__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ bfloat16_t *rope_cos_view_inline825_inline9697__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(rope_cos_view_inline825_inline9697__ssa_v0_tensor->buffer.addr) +
        rope_cos_view_inline825_inline9697__ssa_v0_tensor->start_offset;

    // Unpack tensor: rope_sin_view_inline813_inline9668__ssa_v0
    __gm__ Tensor *rope_sin_view_inline813_inline9668__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ bfloat16_t *rope_sin_view_inline813_inline9668__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(rope_sin_view_inline813_inline9668__ssa_v0_tensor->buffer.addr) +
        rope_sin_view_inline813_inline9668__ssa_v0_tensor->start_offset;

    // Unpack tensor: q_rope_cos_il_inline862_inline9639__ssa_v0
    __gm__ Tensor *q_rope_cos_il_inline862_inline9639__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *q_rope_cos_il_inline862_inline9639__ssa_v0 =
        reinterpret_cast<__gm__ float *>(q_rope_cos_il_inline862_inline9639__ssa_v0_tensor->buffer.addr) +
        q_rope_cos_il_inline862_inline9639__ssa_v0_tensor->start_offset;

    // Unpack tensor: q_rope_sin_signed_inline898_inline9722__ssa_v0
    __gm__ Tensor *q_rope_sin_signed_inline898_inline9722__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ float *q_rope_sin_signed_inline898_inline9722__ssa_v0 =
        reinterpret_cast<__gm__ float *>(q_rope_sin_signed_inline898_inline9722__ssa_v0_tensor->buffer.addr) +
        q_rope_sin_signed_inline898_inline9722__ssa_v0_tensor->start_offset;

    // Unpack tensor: q_rope_swap_idx_inline864_inline9696__ssa_v0
    __gm__ Tensor *q_rope_swap_idx_inline864_inline9696__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ int32_t *q_rope_swap_idx_inline864_inline9696__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(q_rope_swap_idx_inline864_inline9696__ssa_v0_tensor->buffer.addr) +
        q_rope_swap_idx_inline864_inline9696__ssa_v0_tensor->start_offset;

    // Extract dynamic dim: t_dim_inline889_inline9734__ssa_v0
    int64_t t_dim_inline889_inline9734__ssa_v0 =
        static_cast<int64_t>(rope_cos_view_inline825_inline9697__ssa_v0_tensor->shapes[0]);

    // Forward to ptoas-generated function
    q_rope_prepare_0(
        rope_cos_view_inline825_inline9697__ssa_v0, rope_sin_view_inline813_inline9668__ssa_v0,
        q_rope_cos_il_inline862_inline9639__ssa_v0, q_rope_sin_signed_inline898_inline9722__ssa_v0,
        q_rope_swap_idx_inline864_inline9696__ssa_v0, t_dim_inline889_inline9734__ssa_v0, __pypto_spmd_block_idx,
        __pypto_spmd_block_num
    );
}
