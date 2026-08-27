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
// Kernel Function: rope_cs_2

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
rope_cs_2(__gm__ bfloat16_t *v1, __gm__ bfloat16_t *v2, __gm__ float *v3, __gm__ float *v4, int32_t v5, int32_t v6) {
    RoundMode v7 = RoundMode::CAST_TRUNC;
    SaturationMode v8 = SaturationMode::OFF;
    RoundMode v9 = RoundMode::CAST_ROUND;
    const float v10 = 2.0f;
    const float v11 = 0.5f;
    const int32_t v12 = 0;
    const float v13 = 1.0f;
    const int64_t v14 = 32;
    const int64_t v15 = 16;
    const int64_t v16 = 1;
    const int64_t v17 = 8;
    const int64_t v18 = 2176;
    const int64_t v19 = 2048;
    const int64_t v20 = 1536;
    const int64_t v21 = 1024;
    const int64_t v22 = 0;
    const int64_t v23 = 3328;
    const int64_t v24 = 2304;
    const int64_t v25 = 64;
    const int64_t v26 = 128;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %cp_inline1571_inline11328__ssa_v0
    int64_t v27 = (int64_t)v5;
    // pto: %18
    int64_t v28 = (int64_t)((uint64_t)v27 * (uint64_t)v15);
    int64_t v29 = (int64_t)((uint64_t)v27 * (uint64_t)v14);
    // pto: %t__tile
    Tile<
        TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v30 = Tile<
            TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v14);
    // pto: %t__tile
    uint64_t v31 = (uint64_t)v24;
    TASSIGN(v30, v31);
    TEXPANDS(v30, v13);
    // pto: %0
    Tile<
        TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v32 = Tile<
            TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v14);
    // pto: %0
    uint64_t v33 = (uint64_t)v23;
    TASSIGN(v32, v33);
    TCI<Tile<
            TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>,
        int32_t, 0>(v32, v12);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    // pto: %1
    Tile<
        TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v34 = Tile<
            TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v14);
    // pto: %1
    uint64_t v35 = (uint64_t)v23;
    TASSIGN(v34, v35);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    TCVT(v34, v32, v9, v8);
    // pto: %cs_col_inline1569_inline11326__tile
    Tile<
        TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v36 = Tile<
            TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v14);
    // pto: %cs_col_inline1569_inline11326__tile
    uint64_t v37 = (uint64_t)v24;
    TASSIGN(v36, v37);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v36, v30, v34);
    // pto: %2
    Tile<
        TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v38 = Tile<
            TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v14);
    // pto: %2
    uint64_t v39 = (uint64_t)v23;
    TASSIGN(v38, v39);
    pipe_barrier(PIPE_V);
    TMULS(v38, v36, v11);
    // pto: %3
    Tile<
        TileType::Vec, int32_t, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v40 = Tile<
            TileType::Vec, int32_t, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v14);
    // pto: %3
    uint64_t v41 = (uint64_t)v23;
    TASSIGN(v40, v41);
    pipe_barrier(PIPE_V);
    TCVT(v40, v38, v7, v8);
    // pto: %cs_dup_f_inline1455_inline11646__tile
    Tile<
        TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v42 = Tile<
            TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v14);
    // pto: %cs_dup_f_inline1455_inline11646__tile
    uint64_t v43 = (uint64_t)v23;
    TASSIGN(v42, v43);
    pipe_barrier(PIPE_V);
    TCVT(v42, v40, v9, v8);
    // pto: %cs_dup_idx_inline1465_inline11484__tile
    Tile<
        TileType::Vec, int32_t, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v44 = Tile<
            TileType::Vec, int32_t, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v14);
    // pto: %cs_dup_idx_inline1465_inline11484__tile
    uint64_t v45 = (uint64_t)v22;
    TASSIGN(v44, v45);
    pipe_barrier(PIPE_V);
    TCVT(v44, v42, v9, v8);
    // pto: %4
    Tile<
        TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v46 = Tile<
            TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v14);
    // pto: %4
    uint64_t v47 = (uint64_t)v23;
    TASSIGN(v46, v47);
    pipe_barrier(PIPE_V);
    TMULS(v46, v42, v10);
    // pto: %cs_lane_inline1502_inline11325__tile
    Tile<
        TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v48 = Tile<
            TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v14);
    // pto: %cs_lane_inline1502_inline11325__tile
    uint64_t v49 = (uint64_t)v24;
    TASSIGN(v48, v49);
    pipe_barrier(PIPE_V);
    TSUB(v48, v36, v46);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    // pto: %5
    Tile<
        TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v50 = Tile<
            TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v14);
    // pto: %5
    uint64_t v51 = (uint64_t)v24;
    TASSIGN(v50, v51);
    pipe_barrier(PIPE_V);
    TMULS(v50, v48, v10);
    // pto: %6
    Tile<
        TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v52 = Tile<
            TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v14);
    // pto: %6
    uint64_t v53 = (uint64_t)v24;
    TASSIGN(v52, v53);
    pipe_barrier(PIPE_V);
    TSUBS(v52, v50, v13);
    // pto: %cs_sign_inline1574_inline11514__tile
    Tile<
        TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v54 = Tile<
            TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v14);
    // pto: %cs_sign_inline1574_inline11514__tile
    uint64_t v55 = (uint64_t)v24;
    TASSIGN(v54, v55);
    pipe_barrier(PIPE_V);
    TNEG(v54, v52);
    // pto: %7
    Tile<
        TileType::Vec, bfloat16_t, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v56 = Tile<
            TileType::Vec, bfloat16_t, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v15);
    // pto: %7
    uint64_t v57 = (uint64_t)v23;
    TASSIGN(v56, v57);
    // pto: %20
    int64_t v58 = v28 < v22 ? v22 : v28;
    // pto: %rope_cos_t_inline11615__rv_v2_pview
    pto::Shape<1, 1, 1, 8, 16> v59 = pto::Shape<1, 1, 1, 8, 16>();
    // pto: %rope_cos_t_inline11615__rv_v2_pview
    pto::Stride<512, 512, 512, 64, 1> v60 = pto::Stride<512, 512, 512, 64, 1>();
    // pto: %rope_cos_t_inline11615__rv_v2_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 16>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v61 =
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 16>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(
            v1 + (v22 + v58), v59, v60
        );
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v56, v61);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %cs_cos_inline1597_inline11339__tile
    Tile<
        TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v62 = Tile<
            TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v15);
    // pto: %cs_cos_inline1597_inline11339__tile
    uint64_t v63 = (uint64_t)v21;
    TASSIGN(v62, v63);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(v62, v56, v9, v8);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    // pto: %8
    Tile<
        TileType::Vec, bfloat16_t, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v64 = Tile<
            TileType::Vec, bfloat16_t, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v15);
    // pto: %8
    uint64_t v65 = (uint64_t)v23;
    TASSIGN(v64, v65);
    // pto: %rope_sin_t_inline11554__rv_v2_pview
    pto::Shape<1, 1, 1, 8, 16> v66 = pto::Shape<1, 1, 1, 8, 16>();
    // pto: %rope_sin_t_inline11554__rv_v2_pview
    pto::Stride<512, 512, 512, 64, 1> v67 = pto::Stride<512, 512, 512, 64, 1>();
    // pto: %rope_sin_t_inline11554__rv_v2_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 16>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v68 =
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 16>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(
            v2 + (v22 + v58), v66, v67
        );
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    TLOAD(v64, v68);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %cs_sin_inline1513_inline11324__tile
    Tile<
        TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v69 = Tile<
            TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v15);
    // pto: %cs_sin_inline1513_inline11324__tile
    uint64_t v70 = (uint64_t)v20;
    TASSIGN(v69, v70);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TCVT(v69, v64, v9, v8);
    // pto: %gather_acc_init
    Tile<
        TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v71 = Tile<
            TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v14);
    // pto: %gather_acc_init
    uint64_t v72 = (uint64_t)v23;
    TASSIGN(v71, v72);
    for (int64_t i73 = v22; i73 < v17; i73 += v16) {
        // pto: %gather_inp_row
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v74 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v15);
        // pto: %gather_inp_row
        uint64_t v75 = (uint64_t)v21;
        TASSIGN(v74, v75);
        // pto: %slice_view
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v76;
        // pto: %slice_view
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v77 = v76;
        // pto: %slice_view
        uint64_t v78 = (uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)i73 * (uint64_t)v25)) + (uint64_t)v21));
        TASSIGN(v77, v78);
        // pto: %gather_idx_row
        Tile<
            TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v79 = Tile<
                TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v14);
        // pto: %gather_idx_row
        uint64_t v80 = (uint64_t)v22;
        TASSIGN(v79, v80);
        // pto: %22
        int64_t v81 = (int64_t)((uint64_t)i73 * (uint64_t)v26);
        // pto: %22
        Tile<
            TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v82;
        // pto: %22
        Tile<
            TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v83 = v82;
        // pto: %22
        uint64_t v84 = (uint64_t)v81;
        TASSIGN(v83, v84);
        // pto: %gather_row_tmp
        Tile<
            TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v85 = Tile<
                TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v14);
        // pto: %gather_row_tmp
        uint64_t v86 = (uint64_t)v19;
        TASSIGN(v85, v86);
        // pto: %gather_row
        Tile<
            TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v87 = Tile<
                TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v14);
        // pto: %gather_row
        uint64_t v88 = (uint64_t)v18;
        TASSIGN(v87, v88);
        pipe_barrier(PIPE_V);
        TGATHER(v87, v77, v83, v85);
        // pto: %assemble_view
        Tile<
            TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v89;
        // pto: %assemble_view
        Tile<
            TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v90 = v89;
        // pto: %assemble_view
        uint64_t v91 = (uint64_t)((int64_t)((uint64_t)v81 + (uint64_t)v23));
        TASSIGN(v90, v91);
        pipe_barrier(PIPE_V);
        TMOV(v90, v87);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %9
    Tile<
        TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v92 = Tile<
            TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v14);
    // pto: %9
    uint64_t v93 = (uint64_t)v23;
    TASSIGN(v92, v93);
    // pto: %23
    int64_t v94 = v29 < v22 ? v22 : v29;
    // pto: %rope_cos_il_inline1559_inline11376__ssa_v0_pview
    pto::Shape<1, 1, 1, 8, 32> v95 = pto::Shape<1, 1, 1, 8, 32>();
    // pto: %rope_cos_il_inline1559_inline11376__ssa_v0_pview
    pto::Stride<512, 512, 512, 64, 1> v96 = pto::Stride<512, 512, 512, 64, 1>();
    // pto: %rope_cos_il_inline1559_inline11376__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 32>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v97 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 32>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(
            v3 + (v22 + v94), v95, v96
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v97, v92);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    // pto: %10
    Tile<
        TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v98 = Tile<
            TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v14);
    // pto: %10
    uint64_t v99 = (uint64_t)v23;
    TASSIGN(v98, v99);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    for (int64_t i100 = v22; i100 < v17; i100 += v16) {
        // pto: %11
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v101 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v15);
        // pto: %11
        uint64_t v102 = (uint64_t)v20;
        TASSIGN(v101, v102);
        // pto: %25
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v103;
        // pto: %25
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v104 = v103;
        // pto: %25
        uint64_t v105 = (uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)i100 * (uint64_t)v25)) + (uint64_t)v20));
        TASSIGN(v104, v105);
        // pto: %12
        Tile<
            TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v106 = Tile<
                TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v14);
        // pto: %12
        uint64_t v107 = (uint64_t)v22;
        TASSIGN(v106, v107);
        // pto: %26
        int64_t v108 = (int64_t)((uint64_t)i100 * (uint64_t)v26);
        // pto: %26
        Tile<
            TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v109;
        // pto: %26
        Tile<
            TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v110 = v109;
        // pto: %26
        uint64_t v111 = (uint64_t)v108;
        TASSIGN(v110, v111);
        // pto: %13
        Tile<
            TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v112 = Tile<
                TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v14);
        // pto: %13
        uint64_t v113 = (uint64_t)v21;
        TASSIGN(v112, v113);
        // pto: %14
        Tile<
            TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v114 = Tile<
                TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v14);
        // pto: %14
        uint64_t v115 = (uint64_t)v19;
        TASSIGN(v114, v115);
        pipe_barrier(PIPE_V);
        TGATHER(v114, v104, v110, v112);
        // pto: %27
        Tile<
            TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v116;
        // pto: %27
        Tile<
            TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v117 = v116;
        // pto: %27
        uint64_t v118 = (uint64_t)((int64_t)((uint64_t)v108 + (uint64_t)v23));
        TASSIGN(v117, v118);
        pipe_barrier(PIPE_V);
        TMOV(v117, v114);
    }
    // pto: %16
    Tile<
        TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v119 = Tile<
            TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v14);
    // pto: %16
    uint64_t v120 = (uint64_t)v23;
    TASSIGN(v119, v120);
    // pto: %17
    Tile<
        TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v121 = Tile<
            TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v14);
    // pto: %17
    uint64_t v122 = (uint64_t)v24;
    TASSIGN(v121, v122);
    pipe_barrier(PIPE_V);
    TMUL(v121, v119, v54);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %rope_sin_signed_inline1560_inline11790__ssa_v0_pview
    pto::Shape<1, 1, 1, 8, 32> v123 = pto::Shape<1, 1, 1, 8, 32>();
    // pto: %rope_sin_signed_inline1560_inline11790__ssa_v0_pview
    pto::Stride<512, 512, 512, 64, 1> v124 = pto::Stride<512, 512, 512, 64, 1>();
    // pto: %rope_sin_signed_inline1560_inline11790__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 32>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v125 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 32>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(
            v4 + (v22 + v94), v123, v124
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v125, v121);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: rope_cos_t_inline11615__rv_v2
    __gm__ Tensor *rope_cos_t_inline11615__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ bfloat16_t *rope_cos_t_inline11615__rv_v2 =
        reinterpret_cast<__gm__ bfloat16_t *>(rope_cos_t_inline11615__rv_v2_tensor->buffer.addr) +
        rope_cos_t_inline11615__rv_v2_tensor->start_offset;

    // Unpack tensor: rope_sin_t_inline11554__rv_v2
    __gm__ Tensor *rope_sin_t_inline11554__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ bfloat16_t *rope_sin_t_inline11554__rv_v2 =
        reinterpret_cast<__gm__ bfloat16_t *>(rope_sin_t_inline11554__rv_v2_tensor->buffer.addr) +
        rope_sin_t_inline11554__rv_v2_tensor->start_offset;

    // Unpack tensor: rope_cos_il_inline1559_inline11376__ssa_v0
    __gm__ Tensor *rope_cos_il_inline1559_inline11376__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *rope_cos_il_inline1559_inline11376__ssa_v0 =
        reinterpret_cast<__gm__ float *>(rope_cos_il_inline1559_inline11376__ssa_v0_tensor->buffer.addr) +
        rope_cos_il_inline1559_inline11376__ssa_v0_tensor->start_offset;

    // Unpack tensor: rope_sin_signed_inline1560_inline11790__ssa_v0
    __gm__ Tensor *rope_sin_signed_inline1560_inline11790__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ float *rope_sin_signed_inline1560_inline11790__ssa_v0 =
        reinterpret_cast<__gm__ float *>(rope_sin_signed_inline1560_inline11790__ssa_v0_tensor->buffer.addr) +
        rope_sin_signed_inline1560_inline11790__ssa_v0_tensor->start_offset;

    // Forward to ptoas-generated function
    rope_cs_2(
        rope_cos_t_inline11615__rv_v2, rope_sin_t_inline11554__rv_v2, rope_cos_il_inline1559_inline11376__ssa_v0,
        rope_sin_signed_inline1560_inline11790__ssa_v0, __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
