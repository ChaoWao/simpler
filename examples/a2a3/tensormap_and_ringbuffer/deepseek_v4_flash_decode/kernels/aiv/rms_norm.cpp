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
// Kernel Function: rms_norm

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
rms_norm(__gm__ bfloat16_t *v1, __gm__ bfloat16_t *v2, __gm__ bfloat16_t *v3, int32_t v4, int32_t v5) {
    RoundMode v6 = RoundMode::CAST_RINT;
    SaturationMode v7 = SaturationMode::OFF;
    RoundMode v8 = RoundMode::CAST_ROUND;
    const float v9 = 9.99999997E-7f;
    const float v10 = 2.44140625E-4f;
    const int64_t v11 = 128;
    const int64_t v12 = 2;
    const int64_t v13 = 32;
    const float v14 = 0.0f;
    const int64_t v15 = 1;
    const int64_t v16 = 8;
    const int64_t v17 = 8512;
    const int64_t v18 = 8224;
    const int64_t v19 = 8192;
    const int64_t v20 = 0;
    const int64_t v21 = 16960;
    const int64_t v22 = 8768;
    const int64_t v23 = 4096;
    const int64_t v24 = 12864;
    const int64_t v25 = 8256;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %tg_idx_inline762_inline8891__ssa_v0, %26
    int64_t v26 = (int64_t)((uint64_t)((int64_t)v4) * (uint64_t)v16);
    // pto: %x_sq_sum_inline761_inline8941__tile
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v27 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v16);
    // pto: %x_sq_sum_inline761_inline8941__tile
    uint64_t v28 = (uint64_t)v25;
    TASSIGN(v27, v28);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    TEXPANDS(v27, v14);
    for (int64_t i29 = v20; i29 < v13; i29 += v12) {
        // pto: %27
        int64_t v30 = (int64_t)((uint64_t)i29 * (uint64_t)v11);
        // pto: %29
        int64_t v31 = (int64_t)((uint64_t)v30 + (uint64_t)v11);
        // pto: %t__tile
        Tile<
            TileType::Vec, bfloat16_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v32 = Tile<
                TileType::Vec, bfloat16_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v11);
        // pto: %t__tile
        uint64_t v33 = (uint64_t)v24;
        TASSIGN(v32, v33);
        // pto: %30
        int64_t v34 = v26 < v20 ? v20 : v26;
        // pto: %x_mixed_inline8888__rv_v2_pview
        pto::Shape<1, 1, 1, 8, 128> v35 = pto::Shape<1, 1, 1, 8, 128>();
        // pto: %x_mixed_inline8888__rv_v2_pview
        pto::Stride<32768, 32768, 32768, 4096, 1> v36 = pto::Stride<32768, 32768, 32768, 4096, 1>();
        // pto: %x_mixed_inline8888__rv_v2_pview, %31
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>
            v37 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(
                v1 + ((v20 + v34 * v23) + (v30 < v20 ? v20 : v30)), v35, v36
            );
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v32, v37);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %0
        Tile<
            TileType::Vec, bfloat16_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v38 = Tile<
                TileType::Vec, bfloat16_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v11);
        // pto: %0
        uint64_t v39 = (uint64_t)v23;
        TASSIGN(v38, v39);
        // pto: %34
        pto::Shape<1, 1, 1, 8, 128> v40 = pto::Shape<1, 1, 1, 8, 128>();
        // pto: %34
        pto::Stride<32768, 32768, 32768, 4096, 1> v41 = pto::Stride<32768, 32768, 32768, 4096, 1>();
        // pto: %34, %33
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>
            v42 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(
                v1 + ((v20 + v34 * v23) + (v31 < v20 ? v20 : v31)), v40, v41
            );
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(v38, v42);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %rms_x_chunk_inline759_inline8846__tile
        Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v43 = Tile<
                TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v11);
        // pto: %rms_x_chunk_inline759_inline8846__tile
        uint64_t v44 = (uint64_t)v22;
        TASSIGN(v43, v44);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        TCVT(v43, v32, v8, v7);
        // pto: %1
        Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v45 = Tile<
                TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v11);
        // pto: %1
        uint64_t v46 = (uint64_t)v22;
        TASSIGN(v45, v46);
        pipe_barrier(PIPE_V);
        TMUL(v45, v43, v43);
        // pto: %tmp_tile
        Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v47 = Tile<
                TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v11);
        // pto: %tmp_tile
        uint64_t v48 = (uint64_t)v24;
        TASSIGN(v47, v48);
        // pto: %2
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v49 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v15);
        // pto: %2
        uint64_t v50 = (uint64_t)v21;
        TASSIGN(v49, v50);
        pipe_barrier(PIPE_V);
        TROWSUM(v49, v45, v47);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %3
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v51 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v16);
        // pto: %3
        uint64_t v52 = (uint64_t)v21;
        TASSIGN(v51, v52);
        // pto: %4
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v53 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v16);
        // pto: %4
        uint64_t v54 = (uint64_t)v22;
        TASSIGN(v53, v54);
        pipe_barrier(PIPE_V);
        TADD(v53, v27, v51);
        // pto: %5
        Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v55 = Tile<
                TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v11);
        // pto: %5
        uint64_t v56 = (uint64_t)v20;
        TASSIGN(v55, v56);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TCVT(v55, v38, v8, v7);
        // pto: %6
        Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v57 = Tile<
                TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v11);
        // pto: %6
        uint64_t v58 = (uint64_t)v20;
        TASSIGN(v57, v58);
        pipe_barrier(PIPE_V);
        TMUL(v57, v55, v55);
        // pto: %7
        Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v59 = Tile<
                TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v11);
        // pto: %7
        uint64_t v60 = (uint64_t)v23;
        TASSIGN(v59, v60);
        // pto: %8
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v61 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v15);
        // pto: %8
        uint64_t v62 = (uint64_t)v19;
        TASSIGN(v61, v62);
        pipe_barrier(PIPE_V);
        TROWSUM(v61, v57, v59);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        // pto: %9
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v63 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v16);
        // pto: %9
        uint64_t v64 = (uint64_t)v19;
        TASSIGN(v63, v64);
        // pto: %10
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v65 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v16);
        // pto: %10
        uint64_t v66 = (uint64_t)v25;
        TASSIGN(v65, v66);
        pipe_barrier(PIPE_V);
        TADD(v65, v53, v63);
    }
    // pto: %11
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v67 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v16);
    // pto: %11
    uint64_t v68 = (uint64_t)v22;
    TASSIGN(v67, v68);
    pipe_barrier(PIPE_V);
    TMULS(v67, v27, v10);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    // pto: %12
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v69 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v16);
    // pto: %12
    uint64_t v70 = (uint64_t)v22;
    TASSIGN(v69, v70);
    pipe_barrier(PIPE_V);
    TADDS(v69, v67, v9);
    // pto: %rsqrt_tmp
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v71 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v16);
    // pto: %rsqrt_tmp
    uint64_t v72 = (uint64_t)v24;
    TASSIGN(v71, v72);
    // pto: %x_inv_rms_inline766_inline8915__tile
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v73 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v16);
    // pto: %x_inv_rms_inline766_inline8915__tile
    uint64_t v74 = (uint64_t)v18;
    TASSIGN(v73, v74);
    pipe_barrier(PIPE_V);
    TRSQRT(v73, v69, v71);
    // pto: %x_inv_rms_t_inline764_inline8872__tile
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v75 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v15);
    // pto: %x_inv_rms_t_inline764_inline8872__tile
    uint64_t v76 = (uint64_t)v18;
    TASSIGN(v75, v76);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    for (int64_t i77 = v20; i77 < v13; i77 += v12) {
        // pto: %35
        int64_t v78 = (int64_t)((uint64_t)i77 * (uint64_t)v11);
        // pto: %37
        int64_t v79 = (int64_t)((uint64_t)v78 + (uint64_t)v11);
        // pto: %13
        Tile<
            TileType::Vec, bfloat16_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v80 = Tile<
                TileType::Vec, bfloat16_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v11);
        // pto: %13
        uint64_t v81 = (uint64_t)v20;
        TASSIGN(v80, v81);
        // pto: %38
        int64_t v82 = v26 < v20 ? v20 : v26;
        // pto: %39
        int64_t v83 = v78 < v20 ? v20 : v78;
        // pto: %40
        pto::Shape<1, 1, 1, 8, 128> v84 = pto::Shape<1, 1, 1, 8, 128>();
        // pto: %40
        pto::Stride<32768, 32768, 32768, 4096, 1> v85 = pto::Stride<32768, 32768, 32768, 4096, 1>();
        // pto: %40
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>
            v86 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(
                v1 + ((v20 + v82 * v23) + v83), v84, v85
            );
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        TLOAD(v80, v86);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %14
        Tile<
            TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v87 = Tile<
                TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %14
        uint64_t v88 = (uint64_t)v25;
        TASSIGN(v87, v88);
        // pto: %attn_norm_w_l0_inline616__ssa_v0_pview
        pto::Shape<1, 1, 1, 1, 128> v89 = pto::Shape<1, 1, 1, 1, 128>();
        // pto: %attn_norm_w_l0_inline616__ssa_v0_pview
        pto::Stride<128, 128, 128, 128, 1> v90 = pto::Stride<128, 128, 128, 128, 1>();
        // pto: %attn_norm_w_l0_inline616__ssa_v0_pview
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v91 =
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(
                v3 + (v20 + v83), v89, v90
            );
        TLOAD(v87, v91);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %15
        Tile<
            TileType::Vec, bfloat16_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v92 = Tile<
                TileType::Vec, bfloat16_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v11);
        // pto: %15
        uint64_t v93 = (uint64_t)v23;
        TASSIGN(v92, v93);
        // pto: %43
        int64_t v94 = v79 < v20 ? v20 : v79;
        // pto: %44
        pto::Shape<1, 1, 1, 8, 128> v95 = pto::Shape<1, 1, 1, 8, 128>();
        // pto: %44
        pto::Stride<32768, 32768, 32768, 4096, 1> v96 = pto::Stride<32768, 32768, 32768, 4096, 1>();
        // pto: %44
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>
            v97 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(
                v1 + ((v20 + v82 * v23) + v94), v95, v96
            );
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        TLOAD(v92, v97);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %16
        Tile<
            TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v98 = Tile<
                TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %16
        uint64_t v99 = (uint64_t)v17;
        TASSIGN(v98, v99);
        // pto: %46
        pto::Shape<1, 1, 1, 1, 128> v100 = pto::Shape<1, 1, 1, 1, 128>();
        // pto: %46
        pto::Stride<128, 128, 128, 128, 1> v101 = pto::Stride<128, 128, 128, 128, 1>();
        // pto: %46
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>
            v102 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(
                v3 + (v20 + v94), v100, v101
            );
        TLOAD(v98, v102);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %apply_x_chunk_inline769_inline9094__tile
        Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v103 = Tile<
                TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v11);
        // pto: %apply_x_chunk_inline769_inline9094__tile
        uint64_t v104 = (uint64_t)v22;
        TASSIGN(v103, v104);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        TCVT(v103, v80, v8, v7);
        // pto: %17
        Tile<
            TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v105 = Tile<
                TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %17
        uint64_t v106 = (uint64_t)v25;
        TASSIGN(v105, v106);
        // pto: %norm_w_chunk_inline756_inline8864__tile
        Tile<
            TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v107 = Tile<
                TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %norm_w_chunk_inline756_inline8864__tile
        uint64_t v108 = (uint64_t)v20;
        TASSIGN(v107, v108);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        TCVT(v107, v105, v8, v7);
        // pto: %18
        Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v109 = Tile<
                TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v11);
        // pto: %18
        uint64_t v110 = (uint64_t)v22;
        TASSIGN(v109, v110);
        TROWEXPANDMUL(v109, v103, v75);
        // pto: %x_normed_chunk_inline755_inline8833__tile
        Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v111 = Tile<
                TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v11);
        // pto: %x_normed_chunk_inline755_inline8833__tile
        uint64_t v112 = (uint64_t)v22;
        TASSIGN(v111, v112);
        pipe_barrier(PIPE_V);
        TCOLEXPANDMUL(v111, v109, v107);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        // pto: %19
        Tile<
            TileType::Vec, bfloat16_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v113 = Tile<
                TileType::Vec, bfloat16_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v11);
        // pto: %19
        uint64_t v114 = (uint64_t)v22;
        TASSIGN(v113, v114);
        pipe_barrier(PIPE_V);
        TCVT(v113, v111, v6, v7);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %x_normed_t_inline9040__iter_v1_pview
        pto::Shape<1, 1, 1, 8, 128> v115 = pto::Shape<1, 1, 1, 8, 128>();
        // pto: %x_normed_t_inline9040__iter_v1_pview
        pto::Stride<32768, 32768, 32768, 4096, 1> v116 = pto::Stride<32768, 32768, 32768, 4096, 1>();
        // pto: %x_normed_t_inline9040__iter_v1_pview
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>
            v117 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(
                v2 + ((v20 + v82 * v23) + v83), v115, v116
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v117, v113);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        // pto: %20
        Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v118 = Tile<
                TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v11);
        // pto: %20
        uint64_t v119 = (uint64_t)v24;
        TASSIGN(v118, v119);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        TCVT(v118, v92, v8, v7);
        // pto: %21
        Tile<
            TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v120 = Tile<
                TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %21
        uint64_t v121 = (uint64_t)v17;
        TASSIGN(v120, v121);
        // pto: %22
        Tile<
            TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v122 = Tile<
                TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %22
        uint64_t v123 = (uint64_t)v23;
        TASSIGN(v122, v123);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        TCVT(v122, v120, v8, v7);
        // pto: %23
        Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v124 = Tile<
                TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v11);
        // pto: %23
        uint64_t v125 = (uint64_t)v24;
        TASSIGN(v124, v125);
        TROWEXPANDMUL(v124, v118, v75);
        // pto: %24
        Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v126 = Tile<
                TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v11);
        // pto: %24
        uint64_t v127 = (uint64_t)v24;
        TASSIGN(v126, v127);
        pipe_barrier(PIPE_V);
        TCOLEXPANDMUL(v126, v124, v122);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        // pto: %25
        Tile<
            TileType::Vec, bfloat16_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v128 = Tile<
                TileType::Vec, bfloat16_t, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v11);
        // pto: %25
        uint64_t v129 = (uint64_t)v24;
        TASSIGN(v128, v129);
        pipe_barrier(PIPE_V);
        TCVT(v128, v126, v6, v7);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        // pto: %x_normed_t_inline9040__tile_pview
        pto::Shape<1, 1, 1, 8, 128> v130 = pto::Shape<1, 1, 1, 8, 128>();
        // pto: %x_normed_t_inline9040__tile_pview
        pto::Stride<32768, 32768, 32768, 4096, 1> v131 = pto::Stride<32768, 32768, 32768, 4096, 1>();
        // pto: %x_normed_t_inline9040__tile_pview
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>
            v132 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(
                v2 + ((v20 + v82 * v23) + v94), v130, v131
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v132, v128);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: x_mixed_inline8888__rv_v2
    __gm__ Tensor *x_mixed_inline8888__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ bfloat16_t *x_mixed_inline8888__rv_v2 =
        reinterpret_cast<__gm__ bfloat16_t *>(x_mixed_inline8888__rv_v2_tensor->buffer.addr) +
        x_mixed_inline8888__rv_v2_tensor->start_offset;

    // Unpack tensor: x_normed_t_inline9040__ssa_v0
    __gm__ Tensor *x_normed_t_inline9040__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ bfloat16_t *x_normed_t_inline9040__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(x_normed_t_inline9040__ssa_v0_tensor->buffer.addr) +
        x_normed_t_inline9040__ssa_v0_tensor->start_offset;

    // Unpack tensor: attn_norm_w_l0_inline616__ssa_v0
    __gm__ Tensor *attn_norm_w_l0_inline616__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ bfloat16_t *attn_norm_w_l0_inline616__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(attn_norm_w_l0_inline616__ssa_v0_tensor->buffer.addr) +
        attn_norm_w_l0_inline616__ssa_v0_tensor->start_offset;

    // Forward to ptoas-generated function
    rms_norm(
        x_mixed_inline8888__rv_v2, x_normed_t_inline9040__ssa_v0, attn_norm_w_l0_inline616__ssa_v0,
        __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
