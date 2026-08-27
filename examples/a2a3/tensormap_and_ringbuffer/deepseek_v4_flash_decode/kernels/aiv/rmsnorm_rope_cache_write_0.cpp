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
// Kernel Function: rmsnorm_rope_cache_write_0

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

static __aicore__ void rmsnorm_rope_cache_write_0(
    __gm__ float *v1, __gm__ float *v2, __gm__ float *v3, __gm__ float *v4, __gm__ bfloat16_t *v5,
    __gm__ bfloat16_t *v6, __gm__ float *v7, __gm__ int32_t *v8, __gm__ int64_t *v9, int64_t v10, int64_t v11,
    int64_t v12, int64_t v13
) {
    RoundMode v14 = RoundMode::CAST_RINT;
    RoundMode v15 = RoundMode::CAST_TRUNC;
    unsigned v16 = 448;
    unsigned v17 = 384;
    SaturationMode v18 = SaturationMode::OFF;
    RoundMode v19 = RoundMode::CAST_ROUND;
    const int64_t v20 = 127;
    const float v21 = 2.0f;
    const float v22 = 0.5f;
    const int32_t v23 = 0;
    const float v24 = 1.0f;
    const int64_t v25 = 6;
    const float v26 = 9.99999997E-7f;
    const float v27 = 0.001953125f;
    const int64_t v28 = 128;
    const int64_t v29 = 8;
    const float v30 = 0.0f;
    const int64_t v31 = 2;
    const int64_t v32 = 512;
    const int64_t v33 = 16;
    const int64_t v34 = 1;
    const int64_t v35 = 64;
    const int64_t v36 = 28864;
    const int64_t v37 = 28800;
    const int64_t v38 = 28736;
    const int64_t v39 = 16448;
    const int64_t v40 = 16384;
    const int64_t v41 = 4096;
    const int64_t v42 = 20544;
    const int64_t v43 = 28992;
    const int64_t v44 = 8192;
    const int64_t v45 = 0;
    const int64_t v46 = 29248;
    const int64_t v47 = 256;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %cos_b_inline1377_inline11647__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v48 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %cos_b_inline1377_inline11647__tile
    uint64_t v49 = (uint64_t)v46;
    TASSIGN(v48, v49);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
    TEXPANDS(v48, v30);
    // pto: %sin_b_inline1353_inline11594__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v50 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %sin_b_inline1353_inline11594__tile
    uint64_t v51 = (uint64_t)v45;
    TASSIGN(v50, v51);
    TEXPANDS(v50, v30);
    for (int64_t i52 = v45; i52 < v10; i52 += v34) {
        // pto: %t__tile
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v53 = Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v34, v35);
        // pto: %t__tile
        uint64_t v54 = (uint64_t)v44;
        TASSIGN(v53, v54);
        // pto: %39
        int64_t v55 = i52 < v45 ? v45 : i52;
        // pto: %cmp_cos_il_inline11581__ssa_v1_pview
        pto::Shape<1, 1, 1, 1, 64> v56 = pto::Shape<1, 1, 1, 1, 64>();
        // pto: %cmp_cos_il_inline11581__ssa_v1_pview
        pto::Stride<64, 64, 64, 64, 1> v57 = pto::Stride<64, 64, 64, 64, 1>();
        // pto: %cmp_cos_il_inline11581__ssa_v1_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v58 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(
                v1 + (v45 + v55 * v35), v56, v57
            );
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v53, v58);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %assemble_view
        int64_t v59 = (int64_t)((uint64_t)i52 * (uint64_t)v47);
        // pto: %assemble_view
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v60;
        // pto: %assemble_view
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v61 = v60;
        // pto: %assemble_view
        uint64_t v62 = (uint64_t)((int64_t)((uint64_t)v59 + (uint64_t)v46));
        TASSIGN(v61, v62);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        TMOV(v61, v53);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        // pto: %1
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v63 = Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v34, v35);
        // pto: %1
        uint64_t v64 = (uint64_t)v44;
        TASSIGN(v63, v64);
        // pto: %cmp_sin_signed_inline11505__ssa_v1_pview
        pto::Shape<1, 1, 1, 1, 64> v65 = pto::Shape<1, 1, 1, 1, 64>();
        // pto: %cmp_sin_signed_inline11505__ssa_v1_pview
        pto::Stride<64, 64, 64, 64, 1> v66 = pto::Stride<64, 64, 64, 64, 1>();
        // pto: %cmp_sin_signed_inline11505__ssa_v1_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v67 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(
                v2 + (v45 + v55 * v35), v65, v66
            );
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(v63, v67);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %41
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v68;
        // pto: %41
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v69 = v68;
        // pto: %41
        uint64_t v70 = (uint64_t)v59;
        TASSIGN(v69, v70);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TMOV(v69, v63);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    // pto: %partial_sq_inline1352_inline11773__tile
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v71 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v34, v33);
    // pto: %partial_sq_inline1352_inline11773__tile
    uint64_t v72 = (uint64_t)v43;
    TASSIGN(v71, v72);
    TEXPANDS(v71, v30);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    for (int64_t i73 = v45; i73 < v29; i73 += v31) {
        // pto: %42
        int64_t v74 = (int64_t)((uint64_t)i73 * (uint64_t)v35);
        // pto: %44
        int64_t v75 = (int64_t)((uint64_t)v74 + (uint64_t)v35);
        // pto: %kv_rms_chunk_inline1350_inline11419__tile
        Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v76 = Tile<
                TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v33, v35);
        // pto: %kv_rms_chunk_inline1350_inline11419__tile
        uint64_t v77 = (uint64_t)v44;
        TASSIGN(v76, v77);
        // pto: %pooled_kv_inline1357_inline11800__rv_v2_pview
        pto::Shape<1, 1, 1, 16, 64> v78 = pto::Shape<1, 1, 1, 16, 64>();
        // pto: %pooled_kv_inline1357_inline11800__rv_v2_pview
        pto::Stride<8192, 8192, 8192, 512, 1> v79 = pto::Stride<8192, 8192, 8192, 512, 1>();
        // pto: %45, %pooled_kv_inline1357_inline11800__rv_v2_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v80 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
                v3 + (v45 + (v74 < v45 ? v45 : v74)), v78, v79
            );
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        TLOAD(v76, v80);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %3
        Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v81 = Tile<
                TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v33, v35);
        // pto: %3
        uint64_t v82 = (uint64_t)v42;
        TASSIGN(v81, v82);
        // pto: %47
        pto::Shape<1, 1, 1, 16, 64> v83 = pto::Shape<1, 1, 1, 16, 64>();
        // pto: %47
        pto::Stride<8192, 8192, 8192, 512, 1> v84 = pto::Stride<8192, 8192, 8192, 512, 1>();
        // pto: %46, %47
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v85 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
                v3 + (v45 + (v75 < v45 ? v45 : v75)), v83, v84
            );
        TLOAD(v81, v85);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %kv_rms_sq_inline1374_inline11634__tile
        Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v86 = Tile<
                TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v33, v35);
        // pto: %kv_rms_sq_inline1374_inline11634__tile
        uint64_t v87 = (uint64_t)v41;
        TASSIGN(v86, v87);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        TMUL(v86, v76, v76);
        // pto: %tmp_tile
        Tile<
            TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v88 = Tile<
                TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v33, v28);
        // pto: %tmp_tile
        uint64_t v89 = (uint64_t)v44;
        TASSIGN(v88, v89);
        // pto: %4
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v90 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v33, v34);
        // pto: %4
        uint64_t v91 = (uint64_t)v40;
        TASSIGN(v90, v91);
        pipe_barrier(PIPE_V);
        TROWSUM(v90, v86, v88);
        // pto: %kv_rms_rowsum_inline1349_inline11417__tile
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v92 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v34, v33);
        // pto: %kv_rms_rowsum_inline1349_inline11417__tile
        uint64_t v93 = (uint64_t)v40;
        TASSIGN(v92, v93);
        // pto: %5
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v94 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v34, v33);
        // pto: %5
        uint64_t v95 = (uint64_t)v44;
        TASSIGN(v94, v95);
        pipe_barrier(PIPE_V);
        TADD(v94, v71, v92);
        // pto: %6
        Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v96 = Tile<
                TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v33, v35);
        // pto: %6
        uint64_t v97 = (uint64_t)v39;
        TASSIGN(v96, v97);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        TMUL(v96, v81, v81);
        // pto: %7
        Tile<
            TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v98 = Tile<
                TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v33, v28);
        // pto: %7
        uint64_t v99 = (uint64_t)v42;
        TASSIGN(v98, v99);
        // pto: %8
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v100 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v33, v34);
        // pto: %8
        uint64_t v101 = (uint64_t)v38;
        TASSIGN(v100, v101);
        pipe_barrier(PIPE_V);
        TROWSUM(v100, v96, v98);
        // pto: %9
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v102 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v34, v33);
        // pto: %9
        uint64_t v103 = (uint64_t)v38;
        TASSIGN(v102, v103);
        // pto: %10
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v104 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v34, v33);
        // pto: %10
        uint64_t v105 = (uint64_t)v43;
        TASSIGN(v104, v105);
        pipe_barrier(PIPE_V);
        TADD(v104, v94, v102);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    }
    // pto: %11
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v106 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v34, v33);
    // pto: %11
    uint64_t v107 = (uint64_t)v44;
    TASSIGN(v106, v107);
    pipe_barrier(PIPE_V);
    TMULS(v106, v71, v27);
    // pto: %12
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v108 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v34, v33);
    // pto: %12
    uint64_t v109 = (uint64_t)v44;
    TASSIGN(v108, v109);
    pipe_barrier(PIPE_V);
    TADDS(v108, v106, v26);
    // pto: %variance_inline1418_inline11415__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v110 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v34);
    // pto: %variance_inline1418_inline11415__tile
    uint64_t v111 = (uint64_t)v44;
    TASSIGN(v110, v111);
    // pto: %t__rm_a0_tmp_v0
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v112 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v34, v33);
    // pto: %t__rm_a0_tmp_v0
    uint64_t v113 = (uint64_t)v44;
    TASSIGN(v112, v113);
    // pto: %t__row_major_tmp_v1
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v114 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v34, v33);
    // pto: %t__row_major_tmp_v1
    uint64_t v115 = (uint64_t)v44;
    TASSIGN(v114, v115);
    pipe_barrier(PIPE_V);
    TSQRT(v114, v112);
    // pto: %13
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v116 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v34);
    // pto: %13
    uint64_t v117 = (uint64_t)v44;
    TASSIGN(v116, v117);
    // pto: %inv_rms_inline1347_inline11413__rm_a0_tmp_v2
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v118 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v34, v33);
    // pto: %inv_rms_inline1347_inline11413__rm_a0_tmp_v2
    uint64_t v119 = (uint64_t)v44;
    TASSIGN(v118, v119);
    // pto: %inv_rms_inline1347_inline11413__row_major_tmp_v3
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v120 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v34, v33);
    // pto: %inv_rms_inline1347_inline11413__row_major_tmp_v3
    uint64_t v121 = (uint64_t)v37;
    TASSIGN(v120, v121);
    pipe_barrier(PIPE_V);
    TRECIP(v120, v118);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    // pto: %inv_rms_inline1347_inline11413__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v122 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v34);
    // pto: %inv_rms_inline1347_inline11413__tile
    uint64_t v123 = (uint64_t)v37;
    TASSIGN(v122, v123);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    for (int64_t i124 = v45; i124 < v25; i124 += v31) {
        // pto: %48
        int64_t v125 = (int64_t)((uint64_t)i124 * (uint64_t)v35);
        // pto: %50
        int64_t v126 = (int64_t)((uint64_t)v125 + (uint64_t)v35);
        // pto: %kv_norm_chunk_inline1342_inline11628__tile
        Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v127 = Tile<
                TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v33, v35);
        // pto: %kv_norm_chunk_inline1342_inline11628__tile
        uint64_t v128 = (uint64_t)v44;
        TASSIGN(v127, v128);
        // pto: %51
        int64_t v129 = v125 < v45 ? v45 : v125;
        // pto: %52
        pto::Shape<1, 1, 1, 16, 64> v130 = pto::Shape<1, 1, 1, 16, 64>();
        // pto: %52
        pto::Stride<8192, 8192, 8192, 512, 1> v131 = pto::Stride<8192, 8192, 8192, 512, 1>();
        // pto: %52
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v132 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
                v3 + (v45 + v129), v130, v131
            );
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        TLOAD(v127, v132);
        // pto: %14
        Tile<
            TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v133 = Tile<
                TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v34, v35);
        // pto: %14
        uint64_t v134 = (uint64_t)v43;
        TASSIGN(v133, v134);
        // pto: %norm_w_2d_inline1362_inline11652__ssa_v0_pview
        pto::Shape<1, 1, 1, 1, 64> v135 = pto::Shape<1, 1, 1, 1, 64>();
        // pto: %norm_w_2d_inline1362_inline11652__ssa_v0_pview
        pto::Stride<512, 512, 512, 512, 1> v136 = pto::Stride<512, 512, 512, 512, 1>();
        // pto: %norm_w_2d_inline1362_inline11652__ssa_v0_pview
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v137 =
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(
                v5 + (v45 + v129), v135, v136
            );
        TLOAD(v133, v137);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %15
        Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v138 = Tile<
                TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v33, v35);
        // pto: %15
        uint64_t v139 = (uint64_t)v42;
        TASSIGN(v138, v139);
        // pto: %54
        int64_t v140 = v126 < v45 ? v45 : v126;
        // pto: %55
        pto::Shape<1, 1, 1, 16, 64> v141 = pto::Shape<1, 1, 1, 16, 64>();
        // pto: %55
        pto::Stride<8192, 8192, 8192, 512, 1> v142 = pto::Stride<8192, 8192, 8192, 512, 1>();
        // pto: %55
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v143 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
                v3 + (v45 + v140), v141, v142
            );
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        TLOAD(v138, v143);
        // pto: %16
        Tile<
            TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v144 = Tile<
                TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v34, v35);
        // pto: %16
        uint64_t v145 = (uint64_t)v36;
        TASSIGN(v144, v145);
        // pto: %57
        pto::Shape<1, 1, 1, 1, 64> v146 = pto::Shape<1, 1, 1, 1, 64>();
        // pto: %57
        pto::Stride<512, 512, 512, 512, 1> v147 = pto::Stride<512, 512, 512, 512, 1>();
        // pto: %57
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v148 =
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(
                v5 + (v45 + v140), v146, v147
            );
        TLOAD(v144, v148);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %gamma_inline1341_inline11576__tile
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v149 = Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v34, v35);
        // pto: %gamma_inline1341_inline11576__tile
        uint64_t v150 = (uint64_t)v41;
        TASSIGN(v149, v150);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        TCVT(v149, v133, v19, v18);
        // pto: %17
        Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v151 = Tile<
                TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v33, v35);
        // pto: %17
        uint64_t v152 = (uint64_t)v44;
        TASSIGN(v151, v152);
        TROWEXPANDMUL(v151, v127, v122);
        // pto: %normed_chunk_inline1383_inline11410__tile
        Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v153 = Tile<
                TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v33, v35);
        // pto: %normed_chunk_inline1383_inline11410__tile
        uint64_t v154 = (uint64_t)v44;
        TASSIGN(v153, v154);
        pipe_barrier(PIPE_V);
        TCOLEXPANDMUL(v153, v151, v149);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %normed_kv_inline1420_inline11506__iter_v1_pview
        pto::Shape<1, 1, 1, 16, 64> v155 = pto::Shape<1, 1, 1, 16, 64>();
        // pto: %normed_kv_inline1420_inline11506__iter_v1_pview
        pto::Stride<8192, 8192, 8192, 512, 1> v156 = pto::Stride<8192, 8192, 8192, 512, 1>();
        // pto: %normed_kv_inline1420_inline11506__iter_v1_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v157 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
                v4 + (v45 + v129), v155, v156
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v157, v153);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        // pto: %18
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v158 = Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v34, v35);
        // pto: %18
        uint64_t v159 = (uint64_t)v39;
        TASSIGN(v158, v159);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        TCVT(v158, v144, v19, v18);
        // pto: %19
        Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v160 = Tile<
                TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v33, v35);
        // pto: %19
        uint64_t v161 = (uint64_t)v42;
        TASSIGN(v160, v161);
        TROWEXPANDMUL(v160, v138, v122);
        // pto: %20
        Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v162 = Tile<
                TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v33, v35);
        // pto: %20
        uint64_t v163 = (uint64_t)v42;
        TASSIGN(v162, v163);
        pipe_barrier(PIPE_V);
        TCOLEXPANDMUL(v162, v160, v158);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        // pto: %normed_kv_inline1420_inline11506__tile_pview
        pto::Shape<1, 1, 1, 16, 64> v164 = pto::Shape<1, 1, 1, 16, 64>();
        // pto: %normed_kv_inline1420_inline11506__tile_pview
        pto::Stride<8192, 8192, 8192, 512, 1> v165 = pto::Stride<8192, 8192, 8192, 512, 1>();
        // pto: %normed_kv_inline1420_inline11506__tile_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v166 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
                v4 + (v45 + v140), v164, v165
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v166, v162);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    // pto: %21
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v167 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %21
    uint64_t v168 = (uint64_t)v44;
    TASSIGN(v167, v168);
    // pto: %61
    pto::Shape<1, 1, 1, 16, 64> v169 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %61
    pto::Stride<8192, 8192, 8192, 512, 1> v170 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %61
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v171 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
            v3 + v17, v169, v170
        );
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    TLOAD(v167, v171);
    // pto: %22
    Tile<
        TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v172 = Tile<
            TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v34, v35);
    // pto: %22
    uint64_t v173 = (uint64_t)v41;
    TASSIGN(v172, v173);
    // pto: %62
    pto::Shape<1, 1, 1, 1, 64> v174 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %62
    pto::Stride<512, 512, 512, 512, 1> v175 = pto::Stride<512, 512, 512, 512, 1>();
    // pto: %62
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v176 =
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(
            v5 + v17, v174, v175
        );
    TLOAD(v172, v176);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
    // pto: %23
    Tile<
        TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v177 = Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v34, v35);
    // pto: %23
    uint64_t v178 = (uint64_t)v42;
    TASSIGN(v177, v178);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TCVT(v177, v172, v19, v18);
    // pto: %24
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v179 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %24
    uint64_t v180 = (uint64_t)v44;
    TASSIGN(v179, v180);
    TROWEXPANDMUL(v179, v167, v122);
    // pto: %25
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v181 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %25
    uint64_t v182 = (uint64_t)v44;
    TASSIGN(v181, v182);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v181, v179, v177);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    // pto: %normed_kv_inline1420_inline11506__rv_v2_main_pview
    pto::Shape<1, 1, 1, 16, 64> v183 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %normed_kv_inline1420_inline11506__rv_v2_main_pview
    pto::Stride<8192, 8192, 8192, 512, 1> v184 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %normed_kv_inline1420_inline11506__rv_v2_main_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v185 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
            v4 + v17, v183, v184
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    TSTORE(v185, v181);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    // pto: %kv_rope_norm_inline1340_inline11707__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v186 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %kv_rope_norm_inline1340_inline11707__tile
    uint64_t v187 = (uint64_t)v44;
    TASSIGN(v186, v187);
    // pto: %64
    pto::Shape<1, 1, 1, 16, 64> v188 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %64
    pto::Stride<8192, 8192, 8192, 512, 1> v189 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %64
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v190 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
            v3 + v16, v188, v189
        );
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    TLOAD(v186, v190);
    // pto: %26
    Tile<
        TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v191 = Tile<
            TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v34, v35);
    // pto: %26
    uint64_t v192 = (uint64_t)v41;
    TASSIGN(v191, v192);
    // pto: %65
    pto::Shape<1, 1, 1, 1, 64> v193 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %65
    pto::Stride<512, 512, 512, 512, 1> v194 = pto::Stride<512, 512, 512, 512, 1>();
    // pto: %65
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v195 =
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(
            v5 + v16, v193, v194
        );
    TLOAD(v191, v195);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    // pto: %gamma_rope_inline1339_inline11409__tile
    Tile<
        TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v196 = Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v34, v35);
    // pto: %gamma_rope_inline1339_inline11409__tile
    uint64_t v197 = (uint64_t)v42;
    TASSIGN(v196, v197);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    TCVT(v196, v191, v19, v18);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    // pto: %27
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v198 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %27
    uint64_t v199 = (uint64_t)v44;
    TASSIGN(v198, v199);
    TROWEXPANDMUL(v198, v186, v122);
    // pto: %rope_normed_inline1338_inline11407__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v200 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %rope_normed_inline1338_inline11407__tile
    uint64_t v201 = (uint64_t)v44;
    TASSIGN(v200, v201);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v200, v198, v196);
    // pto: %rope_ones_inline1337_inline11406__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v202 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %rope_ones_inline1337_inline11406__tile
    uint64_t v203 = (uint64_t)v42;
    TASSIGN(v202, v203);
    pipe_barrier(PIPE_V);
    TEXPANDS(v202, v24);
    // pto: %28
    Tile<
        TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v204 = Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v34, v35);
    // pto: %28
    uint64_t v205 = (uint64_t)v41;
    TASSIGN(v204, v205);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TCI<Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>,
        int32_t, 0>(v204, v23);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    // pto: %29
    Tile<
        TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v206 = Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v34, v35);
    // pto: %29
    uint64_t v207 = (uint64_t)v41;
    TASSIGN(v206, v207);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    TCVT(v206, v204, v19, v18);
    // pto: %rope_col_inline1336_inline11405__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v208 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %rope_col_inline1336_inline11405__tile
    uint64_t v209 = (uint64_t)v42;
    TASSIGN(v208, v209);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v208, v202, v206);
    // pto: %30
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v210 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %30
    uint64_t v211 = (uint64_t)v41;
    TASSIGN(v210, v211);
    pipe_barrier(PIPE_V);
    TMULS(v210, v208, v22);
    // pto: %31
    Tile<
        TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v212 = Tile<
            TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %31
    uint64_t v213 = (uint64_t)v41;
    TASSIGN(v212, v213);
    pipe_barrier(PIPE_V);
    TCVT(v212, v210, v15, v18);
    // pto: %rope_dup_f_inline1335_inline11402__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v214 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %rope_dup_f_inline1335_inline11402__tile
    uint64_t v215 = (uint64_t)v41;
    TASSIGN(v214, v215);
    pipe_barrier(PIPE_V);
    TCVT(v214, v212, v19, v18);
    // pto: %32
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v216 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %32
    uint64_t v217 = (uint64_t)v41;
    TASSIGN(v216, v217);
    pipe_barrier(PIPE_V);
    TMULS(v216, v214, v21);
    // pto: %rope_lane_inline1345_inline11401__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v218 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %rope_lane_inline1345_inline11401__tile
    uint64_t v219 = (uint64_t)v41;
    TASSIGN(v218, v219);
    pipe_barrier(PIPE_V);
    TSUB(v218, v208, v216);
    // pto: %33
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v220 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %33
    uint64_t v221 = (uint64_t)v42;
    TASSIGN(v220, v221);
    pipe_barrier(PIPE_V);
    TADDS(v220, v208, v24);
    // pto: %34
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v222 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %34
    uint64_t v223 = (uint64_t)v41;
    TASSIGN(v222, v223);
    TMULS(v222, v218, v21);
    // pto: %35
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v224 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %35
    uint64_t v225 = (uint64_t)v42;
    TASSIGN(v224, v225);
    pipe_barrier(PIPE_V);
    TSUB(v224, v220, v222);
    // pto: %rope_swap_idx_inline1344_inline11400__tile
    Tile<
        TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v226 = Tile<
            TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %rope_swap_idx_inline1344_inline11400__tile
    uint64_t v227 = (uint64_t)v42;
    TASSIGN(v226, v227);
    pipe_barrier(PIPE_V);
    TCVT(v226, v224, v19, v18);
    // pto: %gather_acc_init
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v228 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %gather_acc_init
    uint64_t v229 = (uint64_t)v41;
    TASSIGN(v228, v229);
    for (int64_t i230 = v45; i230 < v33; i230 += v34) {
        // pto: %gather_inp_row
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v231 = Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v34, v35);
        // pto: %gather_inp_row
        uint64_t v232 = (uint64_t)v44;
        TASSIGN(v231, v232);
        // pto: %slice_view
        int64_t v233 = (int64_t)((uint64_t)i230 * (uint64_t)v47);
        // pto: %slice_view
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v234;
        // pto: %slice_view
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v235 = v234;
        // pto: %slice_view
        uint64_t v236 = (uint64_t)((int64_t)((uint64_t)v233 + (uint64_t)v44));
        TASSIGN(v235, v236);
        // pto: %gather_idx_row
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v237 = Tile<
                TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v34, v35);
        // pto: %gather_idx_row
        uint64_t v238 = (uint64_t)v42;
        TASSIGN(v237, v238);
        // pto: %66
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v239;
        // pto: %66
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v240 = v239;
        // pto: %66
        uint64_t v241 = (uint64_t)((int64_t)((uint64_t)v233 + (uint64_t)v42));
        TASSIGN(v240, v241);
        // pto: %gather_row_tmp
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v242 = Tile<
                TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v34, v35);
        // pto: %gather_row_tmp
        uint64_t v243 = (uint64_t)v39;
        TASSIGN(v242, v243);
        // pto: %gather_row
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v244 = Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v34, v35);
        // pto: %gather_row
        uint64_t v245 = (uint64_t)v43;
        TASSIGN(v244, v245);
        pipe_barrier(PIPE_V);
        TGATHER(v244, v235, v240, v242);
        // pto: %67
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v246;
        // pto: %67
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v247 = v246;
        // pto: %67
        uint64_t v248 = (uint64_t)((int64_t)((uint64_t)v233 + (uint64_t)v41));
        TASSIGN(v247, v248);
        pipe_barrier(PIPE_V);
        TMOV(v247, v244);
    }
    // pto: %swapped_inline1397_inline11398__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v249 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %swapped_inline1397_inline11398__tile
    uint64_t v250 = (uint64_t)v41;
    TASSIGN(v249, v250);
    // pto: %36
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v251 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %36
    uint64_t v252 = (uint64_t)v44;
    TASSIGN(v251, v252);
    TMUL(v251, v200, v48);
    // pto: %37
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v253 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %37
    uint64_t v254 = (uint64_t)v42;
    TASSIGN(v253, v254);
    pipe_barrier(PIPE_V);
    TMUL(v253, v249, v50);
    // pto: %rope_rot_inline1334_inline11396__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v255 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v33, v35);
    // pto: %rope_rot_inline1334_inline11396__tile
    uint64_t v256 = (uint64_t)v44;
    TASSIGN(v255, v256);
    pipe_barrier(PIPE_V);
    TADD(v255, v251, v253);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    // pto: %normed_kv_inline1420_inline11506__rv_v2_pview
    pto::Shape<1, 1, 1, 16, 64> v257 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %normed_kv_inline1420_inline11506__rv_v2_pview
    pto::Stride<8192, 8192, 8192, 512, 1> v258 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %normed_kv_inline1420_inline11506__rv_v2_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v259 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
            v4 + v16, v257, v258
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    TSTORE(v259, v255);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    for (int64_t i260 = v45; i260 < v10; i260 += v34) {
        // pto: %flat_offset_mul
        int64_t v261 = (int64_t)((uint64_t)i260 * (uint64_t)v31);
        // pto: %first_pos_b_inline1391_inline11394__tile
        int32_t v262 = (v8)[v261];
        // pto: %69, %70
        int64_t v263 = (int64_t)v262 % v28;
        // pto: %71, %72
        if ((int64_t)((uint64_t)v263 + (uint64_t)v11) >= v28) {
            // pto: %kv_row_inline1373_inline11448__tile
            Tile<
                TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v264 = Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v34, v32);
            // pto: %kv_row_inline1373_inline11448__tile
            uint64_t v265 = (uint64_t)v44;
            TASSIGN(v264, v265);
            // pto: %75
            pto::Shape<1, 1, 1, 1, 512> v266 = pto::Shape<1, 1, 1, 1, 512>();
            // pto: %75
            pto::Stride<512, 512, 512, 512, 1> v267 = pto::Stride<512, 512, 512, 512, 1>();
            // pto: %74, %75
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v268 =
                GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(
                    v4 + (v45 + (i260 < v45 ? v45 : i260) * v32), v266, v267
                );
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
            TLOAD(v264, v268);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            // pto: %77, %73, %cmp_row_i64_inline1331_inline11545__tile
            int64_t v269 = (v9)[(int64_t)((uint64_t)v261 + (uint64_t)((int64_t)((uint64_t)v20 - (uint64_t)v263)))];
            // pto: %79
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            if (v269 >= v45) {
                // pto: %81
                int64_t v270 = (int64_t)((uint64_t)i260 * (uint64_t)v11);
                // pto: %kv_flat_inline1421_inline11424__iter_v1_pview
                pto::Shape<1, 1, 1, 1, 512> v271 = pto::Shape<1, 1, 1, 1, 512>();
                // pto: %kv_flat_inline1421_inline11424__iter_v1_pview
                pto::Stride<512, 512, 512, 512, 1> v272 = pto::Stride<512, 512, 512, 512, 1>();
                // pto: %82, %kv_flat_inline1421_inline11424__iter_v1_pview
                GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>
                    v273 = GlobalTensor<
                        float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(
                        v7 + (v45 + (v270 < v45 ? v45 : v270) * v32), v271, v272
                    );
                TSTORE(v273, v264);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
                // pto: %38
                Tile<
                    TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v274 = Tile<
                        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                        PadValue::Null, CompactMode::Null>(v34, v32);
                // pto: %38
                uint64_t v275 = (uint64_t)v44;
                TASSIGN(v274, v275);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
                TCVT(v274, v264, v14, v18);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
                // pto: %cmp_kv_cache_flat_inline1354_inline11453__iter_v1_pview
                pto::Shape<1, 1, 1, 1, 512> v276 = pto::Shape<1, 1, 1, 1, 512>();
                // pto: %cmp_kv_cache_flat_inline1354_inline11453__iter_v1_pview
                pto::Stride<512, 512, 512, 512, 1> v277 = pto::Stride<512, 512, 512, 512, 1>();
                // pto: %83, %cmp_kv_cache_flat_inline1354_inline11453__iter_v1_pview
                GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>
                    v278 = GlobalTensor<
                        bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(
                        v6 + (v45 + (v269 < v45 ? v45 : v269) * v32), v276, v277
                    );
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
                TSTORE(v278, v274);
            }
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
        }
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Unpack tensor: cmp_cos_il_inline11581__ssa_v1
    __gm__ Tensor *cmp_cos_il_inline11581__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *cmp_cos_il_inline11581__ssa_v1 =
        reinterpret_cast<__gm__ float *>(cmp_cos_il_inline11581__ssa_v1_tensor->buffer.addr) +
        cmp_cos_il_inline11581__ssa_v1_tensor->start_offset;

    // Unpack tensor: cmp_sin_signed_inline11505__ssa_v1
    __gm__ Tensor *cmp_sin_signed_inline11505__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *cmp_sin_signed_inline11505__ssa_v1 =
        reinterpret_cast<__gm__ float *>(cmp_sin_signed_inline11505__ssa_v1_tensor->buffer.addr) +
        cmp_sin_signed_inline11505__ssa_v1_tensor->start_offset;

    // Unpack tensor: pooled_kv_inline1357_inline11800__rv_v2
    __gm__ Tensor *pooled_kv_inline1357_inline11800__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *pooled_kv_inline1357_inline11800__rv_v2 =
        reinterpret_cast<__gm__ float *>(pooled_kv_inline1357_inline11800__rv_v2_tensor->buffer.addr) +
        pooled_kv_inline1357_inline11800__rv_v2_tensor->start_offset;

    // Unpack tensor: normed_kv_inline1420_inline11506__ssa_v0
    __gm__ Tensor *normed_kv_inline1420_inline11506__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ float *normed_kv_inline1420_inline11506__ssa_v0 =
        reinterpret_cast<__gm__ float *>(normed_kv_inline1420_inline11506__ssa_v0_tensor->buffer.addr) +
        normed_kv_inline1420_inline11506__ssa_v0_tensor->start_offset;

    // Unpack tensor: norm_w_2d_inline1362_inline11652__ssa_v0
    __gm__ Tensor *norm_w_2d_inline1362_inline11652__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ bfloat16_t *norm_w_2d_inline1362_inline11652__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(norm_w_2d_inline1362_inline11652__ssa_v0_tensor->buffer.addr) +
        norm_w_2d_inline1362_inline11652__ssa_v0_tensor->start_offset;

    // Unpack tensor: cmp_kv_cache_flat_inline1354_inline11453__ssa_v0
    __gm__ Tensor *cmp_kv_cache_flat_inline1354_inline11453__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[5]);
    __gm__ bfloat16_t *cmp_kv_cache_flat_inline1354_inline11453__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(cmp_kv_cache_flat_inline1354_inline11453__ssa_v0_tensor->buffer.addr) +
        cmp_kv_cache_flat_inline1354_inline11453__ssa_v0_tensor->start_offset;

    // Unpack tensor: kv_flat_inline1421_inline11424__ssa_v0
    __gm__ Tensor *kv_flat_inline1421_inline11424__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[6]);
    __gm__ float *kv_flat_inline1421_inline11424__ssa_v0 =
        reinterpret_cast<__gm__ float *>(kv_flat_inline1421_inline11424__ssa_v0_tensor->buffer.addr) +
        kv_flat_inline1421_inline11424__ssa_v0_tensor->start_offset;

    // Unpack tensor: position_ids_bsd_inline11737__ssa_v0
    __gm__ Tensor *position_ids_bsd_inline11737__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[7]);
    __gm__ int32_t *position_ids_bsd_inline11737__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(position_ids_bsd_inline11737__ssa_v0_tensor->buffer.addr) +
        position_ids_bsd_inline11737__ssa_v0_tensor->start_offset;

    // Unpack tensor: cmp_slot_mapping_bsd_inline11721__ssa_v0
    __gm__ Tensor *cmp_slot_mapping_bsd_inline11721__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[8]);
    __gm__ int64_t *cmp_slot_mapping_bsd_inline11721__ssa_v0 =
        reinterpret_cast<__gm__ int64_t *>(cmp_slot_mapping_bsd_inline11721__ssa_v0_tensor->buffer.addr) +
        cmp_slot_mapping_bsd_inline11721__ssa_v0_tensor->start_offset;

    // Unpack scalar: b_dim_inline1393_inline11785__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } b_dim_inline1393_inline11785__ssa_v0_conv;
    b_dim_inline1393_inline11785__ssa_v0_conv.u64 = args[9];
    int64_t b_dim_inline1393_inline11785__ssa_v0 = b_dim_inline1393_inline11785__ssa_v0_conv.val;

    // Unpack scalar: s_dim_inline1364_inline11456__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } s_dim_inline1364_inline11456__ssa_v0_conv;
    s_dim_inline1364_inline11456__ssa_v0_conv.u64 = args[10];
    int64_t s_dim_inline1364_inline11456__ssa_v0 = s_dim_inline1364_inline11456__ssa_v0_conv.val;

    // Extract dynamic dim: cmp_flat_rows_inline1395_inline11423__ssa_v0
    int64_t cmp_flat_rows_inline1395_inline11423__ssa_v0 =
        static_cast<int64_t>(cmp_kv_cache_flat_inline1354_inline11453__ssa_v0_tensor->shapes[0]);

    // Extract dynamic dim: bs_inline1375_inline11667__ssa_v0
    int64_t bs_inline1375_inline11667__ssa_v0 =
        static_cast<int64_t>(kv_flat_inline1421_inline11424__ssa_v0_tensor->shapes[0]);

    // Forward to ptoas-generated function
    rmsnorm_rope_cache_write_0(
        cmp_cos_il_inline11581__ssa_v1, cmp_sin_signed_inline11505__ssa_v1, pooled_kv_inline1357_inline11800__rv_v2,
        normed_kv_inline1420_inline11506__ssa_v0, norm_w_2d_inline1362_inline11652__ssa_v0,
        cmp_kv_cache_flat_inline1354_inline11453__ssa_v0, kv_flat_inline1421_inline11424__ssa_v0,
        position_ids_bsd_inline11737__ssa_v0, cmp_slot_mapping_bsd_inline11721__ssa_v0,
        b_dim_inline1393_inline11785__ssa_v0, s_dim_inline1364_inline11456__ssa_v0,
        cmp_flat_rows_inline1395_inline11423__ssa_v0, bs_inline1375_inline11667__ssa_v0
    );
}
