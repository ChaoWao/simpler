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
// Kernel Function: rmsnorm_rope_cache_write_1

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

static __aicore__ void rmsnorm_rope_cache_write_1(
    __gm__ float *v1, __gm__ float *v2, __gm__ float *v3, __gm__ float *v4, __gm__ bfloat16_t *v5,
    __gm__ bfloat16_t *v6, __gm__ float *v7, __gm__ int32_t *v8, __gm__ int64_t *v9, int64_t v10
) {
    RoundMode v11 = RoundMode::CAST_RINT;
    RoundMode v12 = RoundMode::CAST_TRUNC;
    unsigned v13 = 448;
    SaturationMode v14 = SaturationMode::OFF;
    RoundMode v15 = RoundMode::CAST_ROUND;
    const int64_t v16 = 3;
    const float v17 = 2.0f;
    const float v18 = 0.5f;
    const int32_t v19 = 0;
    const float v20 = 1.0f;
    const int64_t v21 = 448;
    const float v22 = 9.99999997E-7f;
    const float v23 = 0.001953125f;
    const int64_t v24 = 128;
    const float v25 = 0.0f;
    const int64_t v26 = 2;
    const int64_t v27 = 512;
    const int64_t v28 = 16;
    const int64_t v29 = 1;
    const int64_t v30 = 64;
    const int64_t v31 = 4;
    const int64_t v32 = 20800;
    const int64_t v33 = 20544;
    const int64_t v34 = 16384;
    const int64_t v35 = 4096;
    const int64_t v36 = 16448;
    const int64_t v37 = 8192;
    const int64_t v38 = 0;
    const int64_t v39 = 21056;
    const int64_t v40 = 256;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %cos_b_inline1833_inline12612__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v41 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %cos_b_inline1833_inline12612__tile
    uint64_t v42 = (uint64_t)v39;
    TASSIGN(v41, v42);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    TEXPANDS(v41, v25);
    // pto: %sin_b_inline1832_inline12566__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v43 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %sin_b_inline1832_inline12566__tile
    uint64_t v44 = (uint64_t)v38;
    TASSIGN(v43, v44);
    TEXPANDS(v43, v25);
    // pto: %t__tile
    Tile<
        TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v45 = Tile<
            TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v31, v30);
    // pto: %t__tile
    uint64_t v46 = (uint64_t)v37;
    TASSIGN(v45, v46);
    // pto: %cmp_cos_il_inline12470__ssa_v1_pview
    pto::Shape<1, 1, 1, 4, 64> v47 = pto::Shape<1, 1, 1, 4, 64>();
    // pto: %cmp_cos_il_inline12470__ssa_v1_pview
    pto::Stride<256, 256, 256, 64, 1> v48 = pto::Stride<256, 256, 256, 64, 1>();
    // pto: %cmp_cos_il_inline12470__ssa_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 4, 64>, pto::Stride<256, 256, 256, 64, 1>, pto::Layout::ND> v49 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 4, 64>, pto::Stride<256, 256, 256, 64, 1>, pto::Layout::ND>(
            v1, v47, v48
        );
    TLOAD(v45, v49);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %assemble_view
    Tile<
        TileType::Vec, float, 4, 64, BLayout::RowMajor, 4, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v50;
    // pto: %assemble_view
    Tile<
        TileType::Vec, float, 4, 64, BLayout::RowMajor, 4, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v51 = v50;
    // pto: %assemble_view
    uint64_t v52 = (uint64_t)v39;
    TASSIGN(v51, v52);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    TMOV(v51, v45);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    // pto: %1
    Tile<
        TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v53 = Tile<
            TileType::Vec, float, 4, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v31, v30);
    // pto: %1
    uint64_t v54 = (uint64_t)v37;
    TASSIGN(v53, v54);
    // pto: %cmp_sin_signed_inline12417__ssa_v1_pview
    pto::Shape<1, 1, 1, 4, 64> v55 = pto::Shape<1, 1, 1, 4, 64>();
    // pto: %cmp_sin_signed_inline12417__ssa_v1_pview
    pto::Stride<256, 256, 256, 64, 1> v56 = pto::Stride<256, 256, 256, 64, 1>();
    // pto: %cmp_sin_signed_inline12417__ssa_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 4, 64>, pto::Stride<256, 256, 256, 64, 1>, pto::Layout::ND> v57 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 4, 64>, pto::Stride<256, 256, 256, 64, 1>, pto::Layout::ND>(
            v2, v55, v56
        );
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v53, v57);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %23
    Tile<
        TileType::Vec, float, 4, 64, BLayout::RowMajor, 4, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v58;
    // pto: %23
    Tile<
        TileType::Vec, float, 4, 64, BLayout::RowMajor, 4, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v59 = v58;
    // pto: %23
    uint64_t v60 = (uint64_t)v38;
    TASSIGN(v59, v60);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TMOV(v59, v53);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    // pto: %partial_sq_inline1856_inline12342__tile
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v61 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v29, v28);
    // pto: %partial_sq_inline1856_inline12342__tile
    uint64_t v62 = (uint64_t)v36;
    TASSIGN(v61, v62);
    TEXPANDS(v61, v25);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    for (int64_t i63 = v38; i63 < v27; i63 += v30) {
        // pto: %kv_rms_chunk_inline1827_inline12431__tile
        Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v64 = Tile<
                TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v30);
        // pto: %kv_rms_chunk_inline1827_inline12431__tile
        uint64_t v65 = (uint64_t)v37;
        TASSIGN(v64, v65);
        // pto: %pooled_kv_inline1855_inline12392__rv_v2_pview
        pto::Shape<1, 1, 1, 16, 64> v66 = pto::Shape<1, 1, 1, 16, 64>();
        // pto: %pooled_kv_inline1855_inline12392__rv_v2_pview
        pto::Stride<8192, 8192, 8192, 512, 1> v67 = pto::Stride<8192, 8192, 8192, 512, 1>();
        // pto: %24, %pooled_kv_inline1855_inline12392__rv_v2_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v68 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
                v3 + (v38 + (i63 < v38 ? v38 : i63)), v66, v67
            );
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        TLOAD(v64, v68);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %kv_rms_sq_inline1826_inline12844__tile
        Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v69 = Tile<
                TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v30);
        // pto: %kv_rms_sq_inline1826_inline12844__tile
        uint64_t v70 = (uint64_t)v35;
        TASSIGN(v69, v70);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        TMUL(v69, v64, v64);
        // pto: %tmp_tile
        Tile<
            TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v71 = Tile<
                TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v24);
        // pto: %tmp_tile
        uint64_t v72 = (uint64_t)v37;
        TASSIGN(v71, v72);
        // pto: %3
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v73 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v29);
        // pto: %3
        uint64_t v74 = (uint64_t)v34;
        TASSIGN(v73, v74);
        pipe_barrier(PIPE_V);
        TROWSUM(v73, v69, v71);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        // pto: %kv_rms_rowsum_inline1825_inline12668__tile
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v75 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v29, v28);
        // pto: %kv_rms_rowsum_inline1825_inline12668__tile
        uint64_t v76 = (uint64_t)v34;
        TASSIGN(v75, v76);
        // pto: %4
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v77 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v29, v28);
        // pto: %4
        uint64_t v78 = (uint64_t)v36;
        TASSIGN(v77, v78);
        pipe_barrier(PIPE_V);
        TADD(v77, v61, v75);
    }
    // pto: %5
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v79 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v29, v28);
    // pto: %5
    uint64_t v80 = (uint64_t)v37;
    TASSIGN(v79, v80);
    pipe_barrier(PIPE_V);
    TMULS(v79, v61, v23);
    // pto: %6
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v81 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v29, v28);
    // pto: %6
    uint64_t v82 = (uint64_t)v37;
    TASSIGN(v81, v82);
    pipe_barrier(PIPE_V);
    TADDS(v81, v79, v22);
    // pto: %variance_inline1824_inline12671__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v83 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v29);
    // pto: %variance_inline1824_inline12671__tile
    uint64_t v84 = (uint64_t)v37;
    TASSIGN(v83, v84);
    // pto: %t__rm_a0_tmp_v0
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v85 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v29, v28);
    // pto: %t__rm_a0_tmp_v0
    uint64_t v86 = (uint64_t)v37;
    TASSIGN(v85, v86);
    // pto: %t__row_major_tmp_v1
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v87 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v29, v28);
    // pto: %t__row_major_tmp_v1
    uint64_t v88 = (uint64_t)v37;
    TASSIGN(v87, v88);
    pipe_barrier(PIPE_V);
    TSQRT(v87, v85);
    // pto: %7
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v89 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v29);
    // pto: %7
    uint64_t v90 = (uint64_t)v37;
    TASSIGN(v89, v90);
    // pto: %inv_rms_inline1841_inline12285__rm_a0_tmp_v2
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v91 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v29, v28);
    // pto: %inv_rms_inline1841_inline12285__rm_a0_tmp_v2
    uint64_t v92 = (uint64_t)v37;
    TASSIGN(v91, v92);
    // pto: %inv_rms_inline1841_inline12285__row_major_tmp_v3
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v93 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v29, v28);
    // pto: %inv_rms_inline1841_inline12285__row_major_tmp_v3
    uint64_t v94 = (uint64_t)v33;
    TASSIGN(v93, v94);
    pipe_barrier(PIPE_V);
    TRECIP(v93, v91);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    // pto: %inv_rms_inline1841_inline12285__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v95 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v29);
    // pto: %inv_rms_inline1841_inline12285__tile
    uint64_t v96 = (uint64_t)v33;
    TASSIGN(v95, v96);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    for (int64_t i97 = v38; i97 < v21; i97 += v30) {
        // pto: %kv_norm_chunk_inline1882_inline12376__tile
        Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v98 = Tile<
                TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v30);
        // pto: %kv_norm_chunk_inline1882_inline12376__tile
        uint64_t v99 = (uint64_t)v37;
        TASSIGN(v98, v99);
        // pto: %25
        int64_t v100 = i97 < v38 ? v38 : i97;
        // pto: %26
        pto::Shape<1, 1, 1, 16, 64> v101 = pto::Shape<1, 1, 1, 16, 64>();
        // pto: %26
        pto::Stride<8192, 8192, 8192, 512, 1> v102 = pto::Stride<8192, 8192, 8192, 512, 1>();
        // pto: %26
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v103 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
                v3 + (v38 + v100), v101, v102
            );
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        TLOAD(v98, v103);
        // pto: %8
        Tile<
            TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v104 = Tile<
                TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v29, v30);
        // pto: %8
        uint64_t v105 = (uint64_t)v36;
        TASSIGN(v104, v105);
        // pto: %norm_w_2d_inline1842_inline12329__ssa_v0_pview
        pto::Shape<1, 1, 1, 1, 64> v106 = pto::Shape<1, 1, 1, 1, 64>();
        // pto: %norm_w_2d_inline1842_inline12329__ssa_v0_pview
        pto::Stride<512, 512, 512, 512, 1> v107 = pto::Stride<512, 512, 512, 512, 1>();
        // pto: %norm_w_2d_inline1842_inline12329__ssa_v0_pview
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v108 =
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(
                v5 + (v38 + v100), v106, v107
            );
        TLOAD(v104, v108);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %gamma_inline1823_inline12673__tile
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v109 = Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v29, v30);
        // pto: %gamma_inline1823_inline12673__tile
        uint64_t v110 = (uint64_t)v35;
        TASSIGN(v109, v110);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        TCVT(v109, v104, v15, v14);
        // pto: %9
        Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v111 = Tile<
                TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v30);
        // pto: %9
        uint64_t v112 = (uint64_t)v37;
        TASSIGN(v111, v112);
        TROWEXPANDMUL(v111, v98, v95);
        // pto: %normed_chunk_inline1849_inline12674__tile
        Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v113 = Tile<
                TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v30);
        // pto: %normed_chunk_inline1849_inline12674__tile
        uint64_t v114 = (uint64_t)v37;
        TASSIGN(v113, v114);
        pipe_barrier(PIPE_V);
        TCOLEXPANDMUL(v113, v111, v109);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %normed_kv_inline1835_inline12626__iter_v1_pview
        pto::Shape<1, 1, 1, 16, 64> v115 = pto::Shape<1, 1, 1, 16, 64>();
        // pto: %normed_kv_inline1835_inline12626__iter_v1_pview
        pto::Stride<8192, 8192, 8192, 512, 1> v116 = pto::Stride<8192, 8192, 8192, 512, 1>();
        // pto: %normed_kv_inline1835_inline12626__iter_v1_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v117 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
                v4 + (v38 + v100), v115, v116
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v117, v113);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    // pto: %kv_rope_norm_inline1914_inline12484__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v118 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %kv_rope_norm_inline1914_inline12484__tile
    uint64_t v119 = (uint64_t)v37;
    TASSIGN(v118, v119);
    // pto: %29
    pto::Shape<1, 1, 1, 16, 64> v120 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %29
    pto::Stride<8192, 8192, 8192, 512, 1> v121 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %29
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v122 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
            v3 + v13, v120, v121
        );
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v118, v122);
    // pto: %10
    Tile<
        TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v123 = Tile<
            TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v29, v30);
    // pto: %10
    uint64_t v124 = (uint64_t)v36;
    TASSIGN(v123, v124);
    // pto: %30
    pto::Shape<1, 1, 1, 1, 64> v125 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %30
    pto::Stride<512, 512, 512, 512, 1> v126 = pto::Stride<512, 512, 512, 512, 1>();
    // pto: %30
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v127 =
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(
            v5 + v13, v125, v126
        );
    TLOAD(v123, v127);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    // pto: %gamma_rope_inline1822_inline12676__tile
    Tile<
        TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v128 = Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v29, v30);
    // pto: %gamma_rope_inline1822_inline12676__tile
    uint64_t v129 = (uint64_t)v35;
    TASSIGN(v128, v129);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    TCVT(v128, v123, v15, v14);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    // pto: %11
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v130 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %11
    uint64_t v131 = (uint64_t)v37;
    TASSIGN(v130, v131);
    TROWEXPANDMUL(v130, v118, v95);
    // pto: %rope_normed_inline1848_inline12679__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v132 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %rope_normed_inline1848_inline12679__tile
    uint64_t v133 = (uint64_t)v37;
    TASSIGN(v132, v133);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v132, v130, v128);
    // pto: %rope_ones_inline1845_inline12681__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v134 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %rope_ones_inline1845_inline12681__tile
    uint64_t v135 = (uint64_t)v35;
    TASSIGN(v134, v135);
    pipe_barrier(PIPE_V);
    TEXPANDS(v134, v20);
    // pto: %12
    Tile<
        TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v136 = Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v29, v30);
    // pto: %12
    uint64_t v137 = (uint64_t)v36;
    TASSIGN(v136, v137);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TCI<Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>,
        int32_t, 0>(v136, v19);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    // pto: %13
    Tile<
        TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v138 = Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v29, v30);
    // pto: %13
    uint64_t v139 = (uint64_t)v36;
    TASSIGN(v138, v139);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    TCVT(v138, v136, v15, v14);
    // pto: %rope_col_inline1831_inline12245__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v140 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %rope_col_inline1831_inline12245__tile
    uint64_t v141 = (uint64_t)v35;
    TASSIGN(v140, v141);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v140, v134, v138);
    // pto: %14
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v142 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %14
    uint64_t v143 = (uint64_t)v36;
    TASSIGN(v142, v143);
    pipe_barrier(PIPE_V);
    TMULS(v142, v140, v18);
    // pto: %15
    Tile<
        TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v144 = Tile<
            TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %15
    uint64_t v145 = (uint64_t)v36;
    TASSIGN(v144, v145);
    pipe_barrier(PIPE_V);
    TCVT(v144, v142, v12, v14);
    // pto: %rope_dup_f_inline1893_inline12710__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v146 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %rope_dup_f_inline1893_inline12710__tile
    uint64_t v147 = (uint64_t)v36;
    TASSIGN(v146, v147);
    pipe_barrier(PIPE_V);
    TCVT(v146, v144, v15, v14);
    // pto: %16
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v148 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %16
    uint64_t v149 = (uint64_t)v36;
    TASSIGN(v148, v149);
    pipe_barrier(PIPE_V);
    TMULS(v148, v146, v17);
    // pto: %rope_lane_inline1829_inline12683__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v150 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %rope_lane_inline1829_inline12683__tile
    uint64_t v151 = (uint64_t)v36;
    TASSIGN(v150, v151);
    pipe_barrier(PIPE_V);
    TSUB(v150, v140, v148);
    // pto: %17
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v152 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %17
    uint64_t v153 = (uint64_t)v35;
    TASSIGN(v152, v153);
    pipe_barrier(PIPE_V);
    TADDS(v152, v140, v20);
    // pto: %18
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v154 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %18
    uint64_t v155 = (uint64_t)v36;
    TASSIGN(v154, v155);
    TMULS(v154, v150, v17);
    // pto: %19
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v156 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %19
    uint64_t v157 = (uint64_t)v35;
    TASSIGN(v156, v157);
    pipe_barrier(PIPE_V);
    TSUB(v156, v152, v154);
    // pto: %rope_swap_idx_inline1828_inline12686__tile
    Tile<
        TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v158 = Tile<
            TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %rope_swap_idx_inline1828_inline12686__tile
    uint64_t v159 = (uint64_t)v35;
    TASSIGN(v158, v159);
    pipe_barrier(PIPE_V);
    TCVT(v158, v156, v15, v14);
    // pto: %gather_acc_init
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v160 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %gather_acc_init
    uint64_t v161 = (uint64_t)v36;
    TASSIGN(v160, v161);
    for (int64_t i162 = v38; i162 < v28; i162 += v29) {
        // pto: %gather_inp_row
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v163 = Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v29, v30);
        // pto: %gather_inp_row
        uint64_t v164 = (uint64_t)v37;
        TASSIGN(v163, v164);
        // pto: %slice_view
        int64_t v165 = (int64_t)((uint64_t)i162 * (uint64_t)v40);
        // pto: %slice_view
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v166;
        // pto: %slice_view
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v167 = v166;
        // pto: %slice_view
        uint64_t v168 = (uint64_t)((int64_t)((uint64_t)v165 + (uint64_t)v37));
        TASSIGN(v167, v168);
        // pto: %gather_idx_row
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v169 = Tile<
                TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v29, v30);
        // pto: %gather_idx_row
        uint64_t v170 = (uint64_t)v35;
        TASSIGN(v169, v170);
        // pto: %31
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v171;
        // pto: %31
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v172 = v171;
        // pto: %31
        uint64_t v173 = (uint64_t)((int64_t)((uint64_t)v165 + (uint64_t)v35));
        TASSIGN(v172, v173);
        // pto: %gather_row_tmp
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v174 = Tile<
                TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v29, v30);
        // pto: %gather_row_tmp
        uint64_t v175 = (uint64_t)v33;
        TASSIGN(v174, v175);
        // pto: %gather_row
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v176 = Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v29, v30);
        // pto: %gather_row
        uint64_t v177 = (uint64_t)v32;
        TASSIGN(v176, v177);
        pipe_barrier(PIPE_V);
        TGATHER(v176, v167, v172, v174);
        // pto: %32
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v178;
        // pto: %32
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v179 = v178;
        // pto: %32
        uint64_t v180 = (uint64_t)((int64_t)((uint64_t)v165 + (uint64_t)v36));
        TASSIGN(v179, v180);
        pipe_barrier(PIPE_V);
        TMOV(v179, v176);
    }
    // pto: %swapped_inline1898_inline12531__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v181 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %swapped_inline1898_inline12531__tile
    uint64_t v182 = (uint64_t)v36;
    TASSIGN(v181, v182);
    // pto: %20
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v183 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %20
    uint64_t v184 = (uint64_t)v37;
    TASSIGN(v183, v184);
    TMUL(v183, v132, v41);
    // pto: %21
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v185 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %21
    uint64_t v186 = (uint64_t)v39;
    TASSIGN(v185, v186);
    pipe_barrier(PIPE_V);
    TMUL(v185, v181, v43);
    // pto: %rope_rot_inline1858_inline12688__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v187 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %rope_rot_inline1858_inline12688__tile
    uint64_t v188 = (uint64_t)v37;
    TASSIGN(v187, v188);
    pipe_barrier(PIPE_V);
    TADD(v187, v183, v185);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %normed_kv_inline1835_inline12626__rv_v2_pview
    pto::Shape<1, 1, 1, 16, 64> v189 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %normed_kv_inline1835_inline12626__rv_v2_pview
    pto::Stride<8192, 8192, 8192, 512, 1> v190 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %normed_kv_inline1835_inline12626__rv_v2_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v191 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
            v4 + v13, v189, v190
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v191, v187);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    for (int64_t i192 = v38; i192 < v31; i192 += v29) {
        // pto: %flat_offset_mul
        int64_t v193 = (int64_t)((uint64_t)i192 * (uint64_t)v26);
        // pto: %first_pos_b_inline1905_inline12623__tile
        int32_t v194 = (v8)[v193];
        // pto: %34, %35
        int64_t v195 = (int64_t)v194 % v31;
        // pto: %36
        if (v195 >= v26) {
            // pto: %kv_row_fp32_inline1834_inline12541__tile
            Tile<
                TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v196 = Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v29, v27);
            // pto: %kv_row_fp32_inline1834_inline12541__tile
            uint64_t v197 = (uint64_t)v37;
            TASSIGN(v196, v197);
            // pto: %normed_kv_inline1835_inline12626__tile_pview
            pto::Shape<1, 1, 1, 1, 512> v198 = pto::Shape<1, 1, 1, 1, 512>();
            // pto: %normed_kv_inline1835_inline12626__tile_pview
            pto::Stride<512, 512, 512, 512, 1> v199 = pto::Stride<512, 512, 512, 512, 1>();
            // pto: %38, %normed_kv_inline1835_inline12626__tile_pview
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v200 =
                GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(
                    v4 + (v38 + (i192 < v38 ? v38 : i192) * v27), v198, v199
                );
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
            TLOAD(v196, v200);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            // pto: %40, %37, %cache_row_i64_inline1851_inline12577__tile
            int64_t v201 = (v9)[(int64_t)((uint64_t)v193 + (uint64_t)((int64_t)((uint64_t)v16 - (uint64_t)v195)))];
            // pto: %42
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            if (v201 >= v38) {
                // pto: %kv_flat_inline1901_inline12341__iter_v1_pview
                pto::Shape<1, 1, 1, 1, 512> v202 = pto::Shape<1, 1, 1, 1, 512>();
                // pto: %kv_flat_inline1901_inline12341__iter_v1_pview
                pto::Stride<512, 512, 512, 512, 1> v203 = pto::Stride<512, 512, 512, 512, 1>();
                // pto: %45, %kv_flat_inline1901_inline12341__iter_v1_pview
                GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>
                    v204 = GlobalTensor<
                        float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(
                        v7 + (v38 + (v193 < v38 ? v38 : v193) * v27), v202, v203
                    );
                TSTORE(v204, v196);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                // pto: %22
                Tile<
                    TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v205 = Tile<
                        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                        PadValue::Null, CompactMode::Null>(v29, v27);
                // pto: %22
                uint64_t v206 = (uint64_t)v37;
                TASSIGN(v205, v206);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                TCVT(v205, v196, v11, v14);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
                // pto: %cmp_kv_cache_flat_inline1870_inline12447__iter_v1_pview
                pto::Shape<1, 1, 1, 1, 512> v207 = pto::Shape<1, 1, 1, 1, 512>();
                // pto: %cmp_kv_cache_flat_inline1870_inline12447__iter_v1_pview
                pto::Stride<512, 512, 512, 512, 1> v208 = pto::Stride<512, 512, 512, 512, 1>();
                // pto: %46, %cmp_kv_cache_flat_inline1870_inline12447__iter_v1_pview
                GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>
                    v209 = GlobalTensor<
                        bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(
                        v6 + (v38 + (v201 < v38 ? v38 : v201) * v27), v207, v208
                    );
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
                TSTORE(v209, v205);
            }
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        }
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Unpack tensor: cmp_cos_il_inline12470__ssa_v1
    __gm__ Tensor *cmp_cos_il_inline12470__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *cmp_cos_il_inline12470__ssa_v1 =
        reinterpret_cast<__gm__ float *>(cmp_cos_il_inline12470__ssa_v1_tensor->buffer.addr) +
        cmp_cos_il_inline12470__ssa_v1_tensor->start_offset;

    // Unpack tensor: cmp_sin_signed_inline12417__ssa_v1
    __gm__ Tensor *cmp_sin_signed_inline12417__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *cmp_sin_signed_inline12417__ssa_v1 =
        reinterpret_cast<__gm__ float *>(cmp_sin_signed_inline12417__ssa_v1_tensor->buffer.addr) +
        cmp_sin_signed_inline12417__ssa_v1_tensor->start_offset;

    // Unpack tensor: pooled_kv_inline1855_inline12392__rv_v2
    __gm__ Tensor *pooled_kv_inline1855_inline12392__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *pooled_kv_inline1855_inline12392__rv_v2 =
        reinterpret_cast<__gm__ float *>(pooled_kv_inline1855_inline12392__rv_v2_tensor->buffer.addr) +
        pooled_kv_inline1855_inline12392__rv_v2_tensor->start_offset;

    // Unpack tensor: normed_kv_inline1835_inline12626__ssa_v0
    __gm__ Tensor *normed_kv_inline1835_inline12626__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ float *normed_kv_inline1835_inline12626__ssa_v0 =
        reinterpret_cast<__gm__ float *>(normed_kv_inline1835_inline12626__ssa_v0_tensor->buffer.addr) +
        normed_kv_inline1835_inline12626__ssa_v0_tensor->start_offset;

    // Unpack tensor: norm_w_2d_inline1842_inline12329__ssa_v0
    __gm__ Tensor *norm_w_2d_inline1842_inline12329__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ bfloat16_t *norm_w_2d_inline1842_inline12329__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(norm_w_2d_inline1842_inline12329__ssa_v0_tensor->buffer.addr) +
        norm_w_2d_inline1842_inline12329__ssa_v0_tensor->start_offset;

    // Unpack tensor: cmp_kv_cache_flat_inline1870_inline12447__ssa_v0
    __gm__ Tensor *cmp_kv_cache_flat_inline1870_inline12447__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[5]);
    __gm__ bfloat16_t *cmp_kv_cache_flat_inline1870_inline12447__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(cmp_kv_cache_flat_inline1870_inline12447__ssa_v0_tensor->buffer.addr) +
        cmp_kv_cache_flat_inline1870_inline12447__ssa_v0_tensor->start_offset;

    // Unpack tensor: kv_flat_inline1901_inline12341__ssa_v0
    __gm__ Tensor *kv_flat_inline1901_inline12341__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[6]);
    __gm__ float *kv_flat_inline1901_inline12341__ssa_v0 =
        reinterpret_cast<__gm__ float *>(kv_flat_inline1901_inline12341__ssa_v0_tensor->buffer.addr) +
        kv_flat_inline1901_inline12341__ssa_v0_tensor->start_offset;

    // Unpack tensor: position_ids_bsd_inline12252__ssa_v0
    __gm__ Tensor *position_ids_bsd_inline12252__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[7]);
    __gm__ int32_t *position_ids_bsd_inline12252__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(position_ids_bsd_inline12252__ssa_v0_tensor->buffer.addr) +
        position_ids_bsd_inline12252__ssa_v0_tensor->start_offset;

    // Unpack tensor: cmp_slot_mapping_bsd_inline12251__ssa_v0
    __gm__ Tensor *cmp_slot_mapping_bsd_inline12251__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[8]);
    __gm__ int64_t *cmp_slot_mapping_bsd_inline12251__ssa_v0 =
        reinterpret_cast<__gm__ int64_t *>(cmp_slot_mapping_bsd_inline12251__ssa_v0_tensor->buffer.addr) +
        cmp_slot_mapping_bsd_inline12251__ssa_v0_tensor->start_offset;

    // Extract dynamic dim: cmp_block_num_inline1877_inline12248__ssa_v0
    int64_t cmp_block_num_inline1877_inline12248__ssa_v0 =
        (static_cast<int64_t>(cmp_kv_cache_flat_inline1870_inline12447__ssa_v0_tensor->shapes[0]) / 128);

    // Forward to ptoas-generated function
    rmsnorm_rope_cache_write_1(
        cmp_cos_il_inline12470__ssa_v1, cmp_sin_signed_inline12417__ssa_v1, pooled_kv_inline1855_inline12392__rv_v2,
        normed_kv_inline1835_inline12626__ssa_v0, norm_w_2d_inline1842_inline12329__ssa_v0,
        cmp_kv_cache_flat_inline1870_inline12447__ssa_v0, kv_flat_inline1901_inline12341__ssa_v0,
        position_ids_bsd_inline12252__ssa_v0, cmp_slot_mapping_bsd_inline12251__ssa_v0,
        cmp_block_num_inline1877_inline12248__ssa_v0
    );
}
