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
// Kernel Function: ffn_norm_0

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

static __aicore__ void ffn_norm_0(
    __gm__ bfloat16_t *v1, __gm__ bfloat16_t *v2, __gm__ float *v3, __gm__ float *v4, __gm__ float *v5,
    __gm__ float *v6, int32_t v7, int32_t v8
) {
    SaturationMode v9 = SaturationMode::OFF;
    RoundMode v10 = RoundMode::CAST_ROUND;
    const float v11 = 0.00787401571f;
    const float v12 = 127.0f;
    const float v13 = 9.99999974E-5f;
    const float v14 = 9.99999997E-7f;
    const float v15 = 2.44140625E-4f;
    const int64_t v16 = 512;
    const int64_t v17 = 1;
    const int64_t v18 = 4096;
    const int64_t v19 = 8;
    const int64_t v20 = 16384;
    const int64_t v21 = 0;
    const int64_t v22 = 16416;
    const int64_t v23 = 32800;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %tok_inline2569_inline10010__ssa_v0
    int64_t v24 = (int64_t)v7;
    // pto: %t__tmp_v512
    Tile<
        TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v25 = Tile<
            TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v18);
    // pto: %t__tmp_v512
    uint64_t v26 = (uint64_t)v23;
    TASSIGN(v25, v26);
    // pto: %0
    int64_t v27 = v24 < v21 ? v21 : v24;
    // pto: %x_mixed_inline10004__rv_v2_pview
    pto::Shape<1, 1, 1, 1, 4096> v28 = pto::Shape<1, 1, 1, 1, 4096>();
    // pto: %x_mixed_inline10004__rv_v2_pview
    pto::Stride<4096, 4096, 4096, 4096, 1> v29 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %x_mixed_inline10004__rv_v2_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v30 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v1 + (v21 + v27 * v18), v28, v29
        );
    TLOAD(v25, v30);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %rms_x_inline2565_inline10025__ssa_v0
    Tile<
        TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v31 = Tile<
            TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v18);
    // pto: %rms_x_inline2565_inline10025__ssa_v0
    uint64_t v32 = (uint64_t)v22;
    TASSIGN(v31, v32);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(v31, v25, v10, v9);
    // pto: %t__tmp_v513
    Tile<
        TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v33 = Tile<
            TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v18);
    // pto: %t__tmp_v513
    uint64_t v34 = (uint64_t)v21;
    TASSIGN(v33, v34);
    // pto: %norm_w_2d_inline2589_inline10171__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 4096> v35 = pto::Shape<1, 1, 1, 1, 4096>();
    // pto: %norm_w_2d_inline2589_inline10171__ssa_v0_pview
    pto::Stride<4096, 4096, 4096, 4096, 1> v36 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %norm_w_2d_inline2589_inline10171__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v37 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v2, v35, v36
        );
    TLOAD(v33, v37);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %rms_w_inline2587_inline10163__ssa_v0
    Tile<
        TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v38 = Tile<
            TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v18);
    // pto: %rms_w_inline2587_inline10163__ssa_v0
    uint64_t v39 = (uint64_t)v23;
    TASSIGN(v38, v39);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    pipe_barrier(PIPE_V);
    TCVT(v38, v33, v10, v9);
    // pto: %xg_inline2581_inline10002__ssa_v0
    Tile<
        TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v40 = Tile<
            TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v18);
    // pto: %xg_inline2581_inline10002__ssa_v0
    uint64_t v41 = (uint64_t)v23;
    TASSIGN(v40, v41);
    pipe_barrier(PIPE_V);
    TMUL(v40, v31, v38);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %xg_buf_inline2582_inline10158__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 4096> v42 = pto::Shape<1, 1, 1, 1, 4096>();
    // pto: %xg_buf_inline2582_inline10158__ssa_v0_pview
    pto::Stride<4096, 4096, 4096, 4096, 1> v43 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %xg_buf_inline2582_inline10158__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v44 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v3 + (v21 + v27 * v18), v42, v43
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v44, v40);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    // pto: %t__tmp_v514
    Tile<
        TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v45 = Tile<
            TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v18);
    // pto: %t__tmp_v514
    uint64_t v46 = (uint64_t)v22;
    TASSIGN(v45, v46);
    pipe_barrier(PIPE_V);
    TMUL(v45, v31, v31);
    // pto: %sq_rows_inline2563_inline10077__ssa_v0
    Tile<
        TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v47 = Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v19, v16);
    // pto: %sq_rows_inline2563_inline10077__ssa_v0
    uint64_t v48 = (uint64_t)v22;
    TASSIGN(v47, v48);
    // pto: %sq_partial_tmp_inline2560_inline10061__ssa_v0
    Tile<
        TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v49 = Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v19, v16);
    // pto: %sq_partial_tmp_inline2560_inline10061__ssa_v0
    uint64_t v50 = (uint64_t)v21;
    TASSIGN(v49, v50);
    // pto: %sq_partial_inline2555_inline9993__ssa_v0
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v51 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v19, v17);
    // pto: %sq_partial_inline2555_inline9993__ssa_v0
    uint64_t v52 = (uint64_t)v20;
    TASSIGN(v51, v52);
    pipe_barrier(PIPE_V);
    TROWSUM(v51, v47, v49);
    // pto: %sq_reduce_inline2556_inline10174__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v53 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v19, v19);
    // pto: %sq_reduce_inline2556_inline10174__ssa_v0
    uint64_t v54 = (uint64_t)v22;
    TASSIGN(v53, v54);
    // pto: %t__tmp_v515
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v55 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v19);
    // pto: %t__tmp_v515
    uint64_t v56 = (uint64_t)v20;
    TASSIGN(v55, v56);
    // pto: %assemble_view
    Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v57;
    // pto: %assemble_view
    Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v58 = v57;
    // pto: %assemble_view
    uint64_t v59 = (uint64_t)v22;
    TASSIGN(v58, v59);
    pipe_barrier(PIPE_V);
    TMOV(v58, v55);
    v53.SetValidShape(v17, v19);
    // pto: %sq_sum_tmp_inline2617_inline10009__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v60 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v19, v19);
    // pto: %sq_sum_tmp_inline2617_inline10009__ssa_v0
    uint64_t v61 = (uint64_t)v21;
    TASSIGN(v60, v61);
    // pto: %sq_sum_inline2574_inline9989__ssa_v0
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v62 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %sq_sum_inline2574_inline9989__ssa_v0
    uint64_t v63 = (uint64_t)v20;
    TASSIGN(v62, v63);
    pipe_barrier(PIPE_V);
    TROWSUM(v62, v53, v60);
    // pto: %t__tmp_v516
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v64 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %t__tmp_v516
    uint64_t v65 = (uint64_t)v20;
    TASSIGN(v64, v65);
    v64.SetValidShape(v17, v17);
    // pto: %t__tmp_v517
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v66 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %t__tmp_v517
    uint64_t v67 = (uint64_t)v22;
    TASSIGN(v66, v67);
    pipe_barrier(PIPE_V);
    TMULS(v66, v64, v15);
    // pto: %t__tmp_v518
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v68 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %t__tmp_v518
    uint64_t v69 = (uint64_t)v22;
    TASSIGN(v68, v69);
    pipe_barrier(PIPE_V);
    TADDS(v68, v66, v14);
    // pto: %t__tmp_v519
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v70 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %t__tmp_v519
    uint64_t v71 = (uint64_t)v22;
    TASSIGN(v70, v71);
    pipe_barrier(PIPE_V);
    TSQRT(v70, v68);
    // pto: %inv_rms_inline2616_inline9996__ssa_v0
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v72 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %inv_rms_inline2616_inline9996__ssa_v0
    uint64_t v73 = (uint64_t)v21;
    TASSIGN(v72, v73);
    pipe_barrier(PIPE_V);
    TRECIP(v72, v70);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %inv_rms_buf_inline2585_inline10044__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 1> v74 = pto::Shape<1, 1, 1, 1, 1>();
    // pto: %inv_rms_buf_inline2585_inline10044__ssa_v0_pview
    pto::Stride<1, 1, 1, 1, 16> v75 = pto::Stride<1, 1, 1, 1, 16>();
    // pto: %inv_rms_buf_inline2585_inline10044__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 16>, pto::Layout::DN> v76 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 16>, pto::Layout::DN>(
            v4 + (v21 + v27), v74, v75
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v76, v72);
    // pto: %t__tmp_v520
    Tile<
        TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v77 = Tile<
            TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v18);
    // pto: %t__tmp_v520
    uint64_t v78 = (uint64_t)v22;
    TASSIGN(v77, v78);
    pipe_barrier(PIPE_V);
    TABS(v77, v40);
    // pto: %xg_abs_rows_inline2551_inline9973__ssa_v0
    Tile<
        TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v79 = Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v19, v16);
    // pto: %xg_abs_rows_inline2551_inline9973__ssa_v0
    uint64_t v80 = (uint64_t)v22;
    TASSIGN(v79, v80);
    // pto: %amax_partial_tmp_inline2548_inline10053__ssa_v0
    Tile<
        TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v81 = Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v19, v16);
    // pto: %amax_partial_tmp_inline2548_inline10053__ssa_v0
    uint64_t v82 = (uint64_t)v23;
    TASSIGN(v81, v82);
    // pto: %amax_partial_inline2557_inline10008__ssa_v0
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v83 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v19, v17);
    // pto: %amax_partial_inline2557_inline10008__ssa_v0
    uint64_t v84 = (uint64_t)v20;
    TASSIGN(v83, v84);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TROWMAX(v83, v79, v81);
    // pto: %amax_reduce_inline2552_inline10058__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v85 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v19, v19);
    // pto: %amax_reduce_inline2552_inline10058__ssa_v0
    uint64_t v86 = (uint64_t)v22;
    TASSIGN(v85, v86);
    // pto: %t__tmp_v521
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v87 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v19);
    // pto: %t__tmp_v521
    uint64_t v88 = (uint64_t)v20;
    TASSIGN(v87, v88);
    // pto: %3
    Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v89;
    // pto: %3
    Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v90 = v89;
    // pto: %3
    uint64_t v91 = (uint64_t)v22;
    TASSIGN(v90, v91);
    pipe_barrier(PIPE_V);
    TMOV(v90, v87);
    v85.SetValidShape(v17, v19);
    // pto: %amax_tmp_inline2546_inline10072__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v92 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v19, v19);
    // pto: %amax_tmp_inline2546_inline10072__ssa_v0
    uint64_t v93 = (uint64_t)v23;
    TASSIGN(v92, v93);
    // pto: %xg_amax_inline2612_inline10000__ssa_v0
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v94 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %xg_amax_inline2612_inline10000__ssa_v0
    uint64_t v95 = (uint64_t)v20;
    TASSIGN(v94, v95);
    pipe_barrier(PIPE_V);
    TROWMAX(v94, v85, v92);
    // pto: %t__tmp_v522
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v96 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %t__tmp_v522
    uint64_t v97 = (uint64_t)v20;
    TASSIGN(v96, v97);
    v96.SetValidShape(v17, v17);
    // pto: %amax_eps_inline2595_inline10070__ssa_v0
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v98 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v19);
    // pto: %amax_eps_inline2595_inline10070__ssa_v0
    uint64_t v99 = (uint64_t)v22;
    TASSIGN(v98, v99);
    pipe_barrier(PIPE_V);
    TEXPANDS(v98, v13);
    v98.SetValidShape(v17, v17);
    // pto: %xg_amax_v2_inline2597_inline10074__ssa_v0
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v100 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %xg_amax_v2_inline2597_inline10074__ssa_v0
    uint64_t v101 = (uint64_t)v22;
    TASSIGN(v100, v101);
    pipe_barrier(PIPE_V);
    TMAX(v100, v96, v98);
    // pto: %scale_max_inline2599_inline10197__ssa_v0
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v102 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v19);
    // pto: %scale_max_inline2599_inline10197__ssa_v0
    uint64_t v103 = (uint64_t)v23;
    TASSIGN(v102, v103);
    TEXPANDS(v102, v12);
    v102.SetValidShape(v17, v17);
    // pto: %xg_sq_inline2580_inline10078__ssa_v0
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v104 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %xg_sq_inline2580_inline10078__ssa_v0
    uint64_t v105 = (uint64_t)v23;
    TASSIGN(v104, v105);
    pipe_barrier(PIPE_V);
    TDIV(v104, v102, v100);
    // pto: %xg_dequant_scale_inline2588_inline10116__ssa_v0
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v106 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %xg_dequant_scale_inline2588_inline10116__ssa_v0
    uint64_t v107 = (uint64_t)v22;
    TASSIGN(v106, v107);
    pipe_barrier(PIPE_V);
    TMULS(v106, v100, v11);
    // pto: %x_norm_dequant_scale_inline2600_inline10081__ssa_v0
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v108 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %x_norm_dequant_scale_inline2600_inline10081__ssa_v0
    uint64_t v109 = (uint64_t)v22;
    TASSIGN(v108, v109);
    pipe_barrier(PIPE_V);
    TMUL(v108, v106, v72);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    // pto: %x_norm_scale_inline10007__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 1> v110 = pto::Shape<1, 1, 1, 1, 1>();
    // pto: %x_norm_scale_inline10007__ssa_v0_pview
    pto::Stride<1, 1, 1, 1, 8> v111 = pto::Stride<1, 1, 1, 1, 8>();
    // pto: %x_norm_scale_inline10007__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 8>, pto::Layout::DN> v112 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 8>, pto::Layout::DN>(
            v5 + (v21 + v27), v110, v111
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    TSTORE(v112, v108);
    // pto: %xn_scale_buf_inline2572_inline10120__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 1> v113 = pto::Shape<1, 1, 1, 1, 1>();
    // pto: %xn_scale_buf_inline2572_inline10120__ssa_v0_pview
    pto::Stride<1, 1, 1, 1, 16> v114 = pto::Stride<1, 1, 1, 1, 16>();
    // pto: %xn_scale_buf_inline2572_inline10120__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 16>, pto::Layout::DN> v115 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 16>, pto::Layout::DN>(
            v6 + (v21 + v27), v113, v114
        );
    TSTORE(v115, v104);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: x_mixed_inline10004__rv_v2
    __gm__ Tensor *x_mixed_inline10004__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ bfloat16_t *x_mixed_inline10004__rv_v2 =
        reinterpret_cast<__gm__ bfloat16_t *>(x_mixed_inline10004__rv_v2_tensor->buffer.addr) +
        x_mixed_inline10004__rv_v2_tensor->start_offset;

    // Unpack tensor: norm_w_2d_inline2589_inline10171__ssa_v0
    __gm__ Tensor *norm_w_2d_inline2589_inline10171__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ bfloat16_t *norm_w_2d_inline2589_inline10171__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(norm_w_2d_inline2589_inline10171__ssa_v0_tensor->buffer.addr) +
        norm_w_2d_inline2589_inline10171__ssa_v0_tensor->start_offset;

    // Unpack tensor: xg_buf_inline2582_inline10158__ssa_v0
    __gm__ Tensor *xg_buf_inline2582_inline10158__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *xg_buf_inline2582_inline10158__ssa_v0 =
        reinterpret_cast<__gm__ float *>(xg_buf_inline2582_inline10158__ssa_v0_tensor->buffer.addr) +
        xg_buf_inline2582_inline10158__ssa_v0_tensor->start_offset;

    // Unpack tensor: inv_rms_buf_inline2585_inline10044__ssa_v0
    __gm__ Tensor *inv_rms_buf_inline2585_inline10044__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ float *inv_rms_buf_inline2585_inline10044__ssa_v0 =
        reinterpret_cast<__gm__ float *>(inv_rms_buf_inline2585_inline10044__ssa_v0_tensor->buffer.addr) +
        inv_rms_buf_inline2585_inline10044__ssa_v0_tensor->start_offset;

    // Unpack tensor: x_norm_scale_inline10007__ssa_v0
    __gm__ Tensor *x_norm_scale_inline10007__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ float *x_norm_scale_inline10007__ssa_v0 =
        reinterpret_cast<__gm__ float *>(x_norm_scale_inline10007__ssa_v0_tensor->buffer.addr) +
        x_norm_scale_inline10007__ssa_v0_tensor->start_offset;

    // Unpack tensor: xn_scale_buf_inline2572_inline10120__ssa_v0
    __gm__ Tensor *xn_scale_buf_inline2572_inline10120__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[5]);
    __gm__ float *xn_scale_buf_inline2572_inline10120__ssa_v0 =
        reinterpret_cast<__gm__ float *>(xn_scale_buf_inline2572_inline10120__ssa_v0_tensor->buffer.addr) +
        xn_scale_buf_inline2572_inline10120__ssa_v0_tensor->start_offset;

    // Forward to ptoas-generated function
    ffn_norm_0(
        x_mixed_inline10004__rv_v2, norm_w_2d_inline2589_inline10171__ssa_v0, xg_buf_inline2582_inline10158__ssa_v0,
        inv_rms_buf_inline2585_inline10044__ssa_v0, x_norm_scale_inline10007__ssa_v0,
        xn_scale_buf_inline2572_inline10120__ssa_v0, __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
