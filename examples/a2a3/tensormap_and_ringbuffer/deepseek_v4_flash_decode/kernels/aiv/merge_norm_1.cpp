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
// Kernel Function: merge_norm_1

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

static __aicore__ void merge_norm_1(
    __gm__ float *v1, __gm__ float *v2, __gm__ float *v3, __gm__ float *v4, __gm__ float *v5, __gm__ float *v6,
    __gm__ int32_t *v7, __gm__ bfloat16_t *v8, int32_t v9, int32_t v10
) {
    SaturationMode v11 = SaturationMode::OFF;
    RoundMode v12 = RoundMode::CAST_RINT;
    const int64_t v13 = 15;
    const int64_t v14 = 14;
    const int64_t v15 = 13;
    const int64_t v16 = 12;
    const int64_t v17 = 11;
    const int64_t v18 = 10;
    const int64_t v19 = 9;
    const int64_t v20 = 7;
    const int64_t v21 = 6;
    const int64_t v22 = 3;
    const int64_t v23 = 448;
    const int64_t v24 = 2;
    const int64_t v25 = 5;
    const int64_t v26 = 80;
    const int64_t v27 = 4;
    const int64_t v28 = 16;
    const int64_t v29 = 8;
    const int64_t v30 = 64;
    const int64_t v31 = 512;
    const int64_t v32 = 1;
    const int64_t v33 = 15360;
    const int64_t v34 = 14336;
    const int64_t v35 = 13312;
    const int64_t v36 = 12288;
    const int64_t v37 = 11264;
    const int64_t v38 = 10240;
    const int64_t v39 = 9216;
    const int64_t v40 = 8192;
    const int64_t v41 = 7168;
    const int64_t v42 = 6144;
    const int64_t v43 = 5120;
    const int64_t v44 = 4096;
    const int64_t v45 = 3072;
    const int64_t v46 = 2048;
    const int64_t v47 = 1024;
    const int64_t v48 = 1792;
    const int64_t v49 = 131584;
    const int64_t v50 = 131520;
    const int64_t v51 = 131456;
    const int64_t v52 = 131648;
    const int64_t v53 = 131392;
    const int64_t v54 = 98560;
    const int64_t v55 = 164928;
    const int64_t v56 = 164672;
    const int64_t v57 = 98624;
    const int64_t v58 = 164416;
    const int64_t v59 = 65792;
    const int64_t v60 = 65728;
    const int64_t v61 = 65664;
    const int64_t v62 = 32896;
    const int64_t v63 = 32832;
    const int64_t v64 = 32768;
    const int64_t v65 = 0;
    const int64_t v66 = 165248;
    const int64_t v67 = 165184;
    const int64_t v68 = 256;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %m_idx_inline2243_inline10401__ssa_v0
    int64_t v69 = (int64_t)v9;
    // pto: %69
    int64_t v70 = v69 / v27;
    // pto: %71, %70, %72
    int64_t v71 =
        (int64_t)((uint64_t)((int64_t)((uint64_t)v69 - (uint64_t)((int64_t)((uint64_t)v70 * (uint64_t)v27)))) *
                  (uint64_t)v28);
    // pto: %73
    int64_t v72 = (int64_t)((uint64_t)v69 * (uint64_t)v26);
    // pto: %m_mi_inline2137_inline10408__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v73 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %m_mi_inline2137_inline10408__tile
    uint64_t v74 = (uint64_t)v67;
    TASSIGN(v73, v74);
    // pto: %74
    int64_t v75 = v72 < v65 ? v65 : v72;
    // pto: %sparse_blk_mi_inline2120_inline10805__rv_v2_pview
    pto::Shape<1, 1, 1, 16, 1> v76 = pto::Shape<1, 1, 1, 16, 1>();
    // pto: %sparse_blk_mi_inline2120_inline10805__rv_v2_pview
    pto::Stride<16, 16, 16, 1, 2560> v77 = pto::Stride<16, 16, 16, 1, 2560>();
    // pto: %sparse_blk_mi_inline2120_inline10805__rv_v2_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 2560>, pto::Layout::DN> v78 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 2560>, pto::Layout::DN>(
            v1 + (v65 + v75), v76, v77
        );
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    TLOAD(v73, v78);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
    // pto: %m_li_inline2213_inline10285__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v79 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %m_li_inline2213_inline10285__tile
    uint64_t v80 = (uint64_t)v66;
    TASSIGN(v79, v80);
    // pto: %sparse_blk_li_inline2118_inline10891__rv_v2_pview
    pto::Shape<1, 1, 1, 16, 1> v81 = pto::Shape<1, 1, 1, 16, 1>();
    // pto: %sparse_blk_li_inline2118_inline10891__rv_v2_pview
    pto::Stride<16, 16, 16, 1, 2560> v82 = pto::Stride<16, 16, 16, 1, 2560>();
    // pto: %sparse_blk_li_inline2118_inline10891__rv_v2_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 2560>, pto::Layout::DN> v83 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 2560>, pto::Layout::DN>(
            v2 + (v65 + v75), v81, v82
        );
    TLOAD(v79, v83);
    // pto: %m_oi_inline2163_inline10841__tile
    Tile<
        TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v84 = Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v31);
    // pto: %m_oi_inline2163_inline10841__tile
    uint64_t v85 = (uint64_t)v65;
    TASSIGN(v84, v85);
    // pto: %sparse_blk_oi_inline2116_inline10892__rv_v2_pview
    pto::Shape<1, 1, 1, 16, 512> v86 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %sparse_blk_oi_inline2116_inline10892__rv_v2_pview
    pto::Stride<8192, 8192, 8192, 512, 1> v87 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %sparse_blk_oi_inline2116_inline10892__rv_v2_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v88 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
            v3 + (v65 + v75 * v31), v86, v87
        );
    TLOAD(v84, v88);
    for (int64_t i89 = v32; i89 < v25; i89 += v24) {
        // pto: %77
        int64_t v90 = (int64_t)((uint64_t)i89 * (uint64_t)v28);
        // pto: %78
        int64_t v91 = (int64_t)((uint64_t)v72 + (uint64_t)v90);
        // pto: %81, %80
        int64_t v92 = (int64_t)((uint64_t)v72 + (uint64_t)((int64_t)((uint64_t)v90 + (uint64_t)v28)));
        // pto: %m_cur_mi_inline2237_inline10284__tile
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v93 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %m_cur_mi_inline2237_inline10284__tile
        uint64_t v94 = (uint64_t)v64;
        TASSIGN(v93, v94);
        // pto: %82
        int64_t v95 = v91 < v65 ? v65 : v91;
        // pto: %83
        pto::Shape<1, 1, 1, 16, 1> v96 = pto::Shape<1, 1, 1, 16, 1>();
        // pto: %83
        pto::Stride<16, 16, 16, 1, 2560> v97 = pto::Stride<16, 16, 16, 1, 2560>();
        // pto: %83
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 2560>, pto::Layout::DN> v98 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 2560>, pto::Layout::DN>(
                v1 + (v65 + v95), v96, v97
            );
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v93, v98);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %m_cur_li_inline2254_inline10282__tile
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v99 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %m_cur_li_inline2254_inline10282__tile
        uint64_t v100 = (uint64_t)v63;
        TASSIGN(v99, v100);
        // pto: %85
        pto::Shape<1, 1, 1, 16, 1> v101 = pto::Shape<1, 1, 1, 16, 1>();
        // pto: %85
        pto::Stride<16, 16, 16, 1, 2560> v102 = pto::Stride<16, 16, 16, 1, 2560>();
        // pto: %85
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 2560>, pto::Layout::DN> v103 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 2560>, pto::Layout::DN>(
                v2 + (v65 + v95), v101, v102
            );
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(v99, v103);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %m_cur_oi_inline2255_inline10281__tile
        Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v104 = Tile<
                TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v31);
        // pto: %m_cur_oi_inline2255_inline10281__tile
        uint64_t v105 = (uint64_t)v62;
        TASSIGN(v104, v105);
        // pto: %87
        pto::Shape<1, 1, 1, 16, 512> v106 = pto::Shape<1, 1, 1, 16, 512>();
        // pto: %87
        pto::Stride<8192, 8192, 8192, 512, 1> v107 = pto::Stride<8192, 8192, 8192, 512, 1>();
        // pto: %87
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v108 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
                v3 + (v65 + v95 * v31), v106, v107
            );
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        TLOAD(v104, v108);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %0
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v109 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %0
        uint64_t v110 = (uint64_t)v61;
        TASSIGN(v109, v110);
        // pto: %88
        int64_t v111 = v92 < v65 ? v65 : v92;
        // pto: %89
        pto::Shape<1, 1, 1, 16, 1> v112 = pto::Shape<1, 1, 1, 16, 1>();
        // pto: %89
        pto::Stride<16, 16, 16, 1, 2560> v113 = pto::Stride<16, 16, 16, 1, 2560>();
        // pto: %89
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 2560>, pto::Layout::DN> v114 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 2560>, pto::Layout::DN>(
                v1 + (v65 + v111), v112, v113
            );
        TLOAD(v109, v114);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %1
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v115 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %1
        uint64_t v116 = (uint64_t)v60;
        TASSIGN(v115, v116);
        // pto: %91
        pto::Shape<1, 1, 1, 16, 1> v117 = pto::Shape<1, 1, 1, 16, 1>();
        // pto: %91
        pto::Stride<16, 16, 16, 1, 2560> v118 = pto::Stride<16, 16, 16, 1, 2560>();
        // pto: %91
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 2560>, pto::Layout::DN> v119 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 2560>, pto::Layout::DN>(
                v2 + (v65 + v111), v117, v118
            );
        TLOAD(v115, v119);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %2
        Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v120 = Tile<
                TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v31);
        // pto: %2
        uint64_t v121 = (uint64_t)v59;
        TASSIGN(v120, v121);
        // pto: %93
        pto::Shape<1, 1, 1, 16, 512> v122 = pto::Shape<1, 1, 1, 16, 512>();
        // pto: %93
        pto::Stride<8192, 8192, 8192, 512, 1> v123 = pto::Stride<8192, 8192, 8192, 512, 1>();
        // pto: %93
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v124 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
                v3 + (v65 + v111 * v31), v122, v123
            );
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        TLOAD(v120, v124);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %m_mi_new_inline2256_inline10280__rm_a0_tmp_v0
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v125 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %m_mi_new_inline2256_inline10280__rm_a0_tmp_v0
        uint64_t v126 = (uint64_t)v67;
        TASSIGN(v125, v126);
        // pto: %m_mi_new_inline2256_inline10280__rm_a1_tmp_v1
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v127 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %m_mi_new_inline2256_inline10280__rm_a1_tmp_v1
        uint64_t v128 = (uint64_t)v64;
        TASSIGN(v127, v128);
        // pto: %m_mi_new_inline2256_inline10280__row_major_tmp_v2
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v129 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %m_mi_new_inline2256_inline10280__row_major_tmp_v2
        uint64_t v130 = (uint64_t)v58;
        TASSIGN(v129, v130);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        TMAX(v129, v125, v127);
        // pto: %m_mi_new_inline2256_inline10280__tile
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v131 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %m_mi_new_inline2256_inline10280__tile
        uint64_t v132 = (uint64_t)v58;
        TASSIGN(v131, v132);
        // pto: %m_mi_inline2137_inline10408__ssa_v3
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v133 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %m_mi_inline2137_inline10408__ssa_v3
        uint64_t v134 = (uint64_t)v58;
        TASSIGN(v133, v134);
        // pto: %t__rm_a0_tmp_v3
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v135 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %t__rm_a0_tmp_v3
        uint64_t v136 = (uint64_t)v67;
        TASSIGN(v135, v136);
        // pto: %t__rm_a1_tmp_v4
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v137 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %t__rm_a1_tmp_v4
        uint64_t v138 = (uint64_t)v58;
        TASSIGN(v137, v138);
        // pto: %t__row_major_tmp_v5
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v139 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %t__row_major_tmp_v5
        uint64_t v140 = (uint64_t)v57;
        TASSIGN(v139, v140);
        pipe_barrier(PIPE_V);
        TSUB(v139, v135, v137);
        // pto: %t__tile
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v141 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %t__tile
        uint64_t v142 = (uint64_t)v57;
        TASSIGN(v141, v142);
        // pto: %m_alpha_inline2098_inline10385__rm_a0_tmp_v6
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v143 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %m_alpha_inline2098_inline10385__rm_a0_tmp_v6
        uint64_t v144 = (uint64_t)v57;
        TASSIGN(v143, v144);
        // pto: %m_alpha_inline2098_inline10385__row_major_tmp_v7
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v145 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %m_alpha_inline2098_inline10385__row_major_tmp_v7
        uint64_t v146 = (uint64_t)v56;
        TASSIGN(v145, v146);
        pipe_barrier(PIPE_V);
        TEXP(v145, v143);
        // pto: %m_alpha_inline2098_inline10385__tile
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v147 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %m_alpha_inline2098_inline10385__tile
        uint64_t v148 = (uint64_t)v56;
        TASSIGN(v147, v148);
        // pto: %t__rm_a0_tmp_v8
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v149 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %t__rm_a0_tmp_v8
        uint64_t v150 = (uint64_t)v64;
        TASSIGN(v149, v150);
        // pto: %t__rm_a1_tmp_v9
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v151 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %t__rm_a1_tmp_v9
        uint64_t v152 = (uint64_t)v58;
        TASSIGN(v151, v152);
        // pto: %t__row_major_tmp_v10
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v153 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %t__row_major_tmp_v10
        uint64_t v154 = (uint64_t)v57;
        TASSIGN(v153, v154);
        pipe_barrier(PIPE_V);
        TSUB(v153, v149, v151);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %3
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v155 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %3
        uint64_t v156 = (uint64_t)v57;
        TASSIGN(v155, v156);
        // pto: %m_beta_inline2097_inline10279__rm_a0_tmp_v11
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v157 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %m_beta_inline2097_inline10279__rm_a0_tmp_v11
        uint64_t v158 = (uint64_t)v57;
        TASSIGN(v157, v158);
        // pto: %m_beta_inline2097_inline10279__row_major_tmp_v12
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v159 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %m_beta_inline2097_inline10279__row_major_tmp_v12
        uint64_t v160 = (uint64_t)v55;
        TASSIGN(v159, v160);
        pipe_barrier(PIPE_V);
        TEXP(v159, v157);
        // pto: %m_beta_inline2097_inline10279__tile
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v161 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %m_beta_inline2097_inline10279__tile
        uint64_t v162 = (uint64_t)v55;
        TASSIGN(v161, v162);
        // pto: %t__rm_a0_tmp_v13
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v163 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %t__rm_a0_tmp_v13
        uint64_t v164 = (uint64_t)v56;
        TASSIGN(v163, v164);
        // pto: %t__rm_a1_tmp_v14
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v165 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %t__rm_a1_tmp_v14
        uint64_t v166 = (uint64_t)v66;
        TASSIGN(v165, v166);
        // pto: %t__row_major_tmp_v15
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v167 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %t__row_major_tmp_v15
        uint64_t v168 = (uint64_t)v57;
        TASSIGN(v167, v168);
        pipe_barrier(PIPE_V);
        TMUL(v167, v163, v165);
        // pto: %4
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v169 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %4
        uint64_t v170 = (uint64_t)v57;
        TASSIGN(v169, v170);
        // pto: %t__rm_a0_tmp_v16
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v171 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %t__rm_a0_tmp_v16
        uint64_t v172 = (uint64_t)v55;
        TASSIGN(v171, v172);
        // pto: %t__rm_a1_tmp_v17
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v173 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %t__rm_a1_tmp_v17
        uint64_t v174 = (uint64_t)v63;
        TASSIGN(v173, v174);
        // pto: %t__row_major_tmp_v18
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v175 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %t__row_major_tmp_v18
        uint64_t v176 = (uint64_t)v54;
        TASSIGN(v175, v176);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TMUL(v175, v171, v173);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        // pto: %5
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v177 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %5
        uint64_t v178 = (uint64_t)v54;
        TASSIGN(v177, v178);
        // pto: %m_li_inline2213_inline10285__rm_a0_tmp_v19
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v179 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %m_li_inline2213_inline10285__rm_a0_tmp_v19
        uint64_t v180 = (uint64_t)v57;
        TASSIGN(v179, v180);
        // pto: %m_li_inline2213_inline10285__rm_a1_tmp_v20
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v181 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %m_li_inline2213_inline10285__rm_a1_tmp_v20
        uint64_t v182 = (uint64_t)v54;
        TASSIGN(v181, v182);
        // pto: %m_li_inline2213_inline10285__row_major_tmp_v21
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v183 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %m_li_inline2213_inline10285__row_major_tmp_v21
        uint64_t v184 = (uint64_t)v54;
        TASSIGN(v183, v184);
        pipe_barrier(PIPE_V);
        TADD(v183, v179, v181);
        // pto: %6
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v185 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %6
        uint64_t v186 = (uint64_t)v54;
        TASSIGN(v185, v186);
        // pto: %7
        Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v187 = Tile<
                TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v31);
        // pto: %7
        uint64_t v188 = (uint64_t)v57;
        TASSIGN(v187, v188);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v187, v84, v147);
        // pto: %8
        Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v189 = Tile<
                TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v31);
        // pto: %8
        uint64_t v190 = (uint64_t)v62;
        TASSIGN(v189, v190);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        TROWEXPANDMUL(v189, v104, v161);
        // pto: %9
        Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v191 = Tile<
                TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v31);
        // pto: %9
        uint64_t v192 = (uint64_t)v62;
        TASSIGN(v191, v192);
        pipe_barrier(PIPE_V);
        TADD(v191, v187, v189);
        // pto: %10
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v193 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %10
        uint64_t v194 = (uint64_t)v58;
        TASSIGN(v193, v194);
        // pto: %11
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v195 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %11
        uint64_t v196 = (uint64_t)v61;
        TASSIGN(v195, v196);
        // pto: %12
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v197 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %12
        uint64_t v198 = (uint64_t)v53;
        TASSIGN(v197, v198);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        TMAX(v197, v193, v195);
        // pto: %13
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v199 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %13
        uint64_t v200 = (uint64_t)v53;
        TASSIGN(v199, v200);
        // pto: %14
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v201 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %14
        uint64_t v202 = (uint64_t)v53;
        TASSIGN(v201, v202);
        // pto: %15
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v203 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %15
        uint64_t v204 = (uint64_t)v58;
        TASSIGN(v203, v204);
        // pto: %16
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v205 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %16
        uint64_t v206 = (uint64_t)v53;
        TASSIGN(v205, v206);
        // pto: %17
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v207 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %17
        uint64_t v208 = (uint64_t)v52;
        TASSIGN(v207, v208);
        pipe_barrier(PIPE_V);
        TSUB(v207, v203, v205);
        // pto: %18
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v209 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %18
        uint64_t v210 = (uint64_t)v52;
        TASSIGN(v209, v210);
        // pto: %19
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v211 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %19
        uint64_t v212 = (uint64_t)v52;
        TASSIGN(v211, v212);
        // pto: %20
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v213 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %20
        uint64_t v214 = (uint64_t)v51;
        TASSIGN(v213, v214);
        pipe_barrier(PIPE_V);
        TEXP(v213, v211);
        // pto: %21
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v215 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %21
        uint64_t v216 = (uint64_t)v51;
        TASSIGN(v215, v216);
        // pto: %22
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v217 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %22
        uint64_t v218 = (uint64_t)v61;
        TASSIGN(v217, v218);
        // pto: %23
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v219 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %23
        uint64_t v220 = (uint64_t)v53;
        TASSIGN(v219, v220);
        // pto: %24
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v221 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %24
        uint64_t v222 = (uint64_t)v52;
        TASSIGN(v221, v222);
        pipe_barrier(PIPE_V);
        TSUB(v221, v217, v219);
        // pto: %25
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v223 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %25
        uint64_t v224 = (uint64_t)v52;
        TASSIGN(v223, v224);
        // pto: %26
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v225 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %26
        uint64_t v226 = (uint64_t)v52;
        TASSIGN(v225, v226);
        // pto: %27
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v227 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %27
        uint64_t v228 = (uint64_t)v50;
        TASSIGN(v227, v228);
        pipe_barrier(PIPE_V);
        TEXP(v227, v225);
        // pto: %28
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v229 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %28
        uint64_t v230 = (uint64_t)v50;
        TASSIGN(v229, v230);
        // pto: %29
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v231 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %29
        uint64_t v232 = (uint64_t)v51;
        TASSIGN(v231, v232);
        // pto: %30
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v233 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %30
        uint64_t v234 = (uint64_t)v54;
        TASSIGN(v233, v234);
        // pto: %31
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v235 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %31
        uint64_t v236 = (uint64_t)v52;
        TASSIGN(v235, v236);
        pipe_barrier(PIPE_V);
        TMUL(v235, v231, v233);
        // pto: %32
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v237 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %32
        uint64_t v238 = (uint64_t)v52;
        TASSIGN(v237, v238);
        // pto: %33
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v239 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %33
        uint64_t v240 = (uint64_t)v50;
        TASSIGN(v239, v240);
        // pto: %34
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v241 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %34
        uint64_t v242 = (uint64_t)v60;
        TASSIGN(v241, v242);
        // pto: %35
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v243 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %35
        uint64_t v244 = (uint64_t)v49;
        TASSIGN(v243, v244);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        TMUL(v243, v239, v241);
        // pto: %36
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v245 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %36
        uint64_t v246 = (uint64_t)v49;
        TASSIGN(v245, v246);
        // pto: %37
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v247 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %37
        uint64_t v248 = (uint64_t)v52;
        TASSIGN(v247, v248);
        // pto: %38
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v249 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %38
        uint64_t v250 = (uint64_t)v49;
        TASSIGN(v249, v250);
        // pto: %39
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v251 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v28);
        // pto: %39
        uint64_t v252 = (uint64_t)v49;
        TASSIGN(v251, v252);
        pipe_barrier(PIPE_V);
        TADD(v251, v247, v249);
        // pto: %40
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v253 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %40
        uint64_t v254 = (uint64_t)v49;
        TASSIGN(v253, v254);
        // pto: %41
        Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v255 = Tile<
                TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v31);
        // pto: %41
        uint64_t v256 = (uint64_t)v52;
        TASSIGN(v255, v256);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v255, v191, v215);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        // pto: %42
        Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v257 = Tile<
                TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v31);
        // pto: %42
        uint64_t v258 = (uint64_t)v59;
        TASSIGN(v257, v258);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        TROWEXPANDMUL(v257, v120, v229);
        // pto: %43
        Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v259 = Tile<
                TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v31);
        // pto: %43
        uint64_t v260 = (uint64_t)v65;
        TASSIGN(v259, v260);
        pipe_barrier(PIPE_V);
        TADD(v259, v255, v257);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        // pto: %m_li_inline2213_inline10285__tile_mv
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v261 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %m_li_inline2213_inline10285__tile_mv
        uint64_t v262 = (uint64_t)v66;
        TASSIGN(v261, v262);
        TMOV(v261, v253);
        // pto: %m_mi_inline2137_inline10408__ssa_v3_mv
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v263 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v28, v32);
        // pto: %m_mi_inline2137_inline10408__ssa_v3_mv
        uint64_t v264 = (uint64_t)v67;
        TASSIGN(v263, v264);
        TMOV(v263, v201);
    }
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    // pto: %44
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v265 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %44
    uint64_t v266 = (uint64_t)v62;
    TASSIGN(v265, v266);
    // pto: %attn_sink_csa_inline727__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 16> v267 = pto::Shape<1, 1, 1, 1, 16>();
    // pto: %attn_sink_csa_inline727__ssa_v0_pview
    pto::Stride<16, 16, 16, 16, 1> v268 = pto::Stride<16, 16, 16, 16, 1>();
    // pto: %94, %attn_sink_csa_inline727__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v269 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(
            v4 + (v65 + (v71 < v65 ? v65 : v71)), v267, v268
        );
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    TLOAD(v265, v269);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    // pto: %n_sink_bias_inline2248_inline10461__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v270 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %n_sink_bias_inline2248_inline10461__tile
    uint64_t v271 = (uint64_t)v62;
    TASSIGN(v270, v271);
    // pto: %t__rm_a0_tmp_v22
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v272 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__rm_a0_tmp_v22
    uint64_t v273 = (uint64_t)v67;
    TASSIGN(v272, v273);
    // pto: %t__rm_a1_tmp_v23
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v274 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__rm_a1_tmp_v23
    uint64_t v275 = (uint64_t)v67;
    TASSIGN(v274, v275);
    // pto: %t__row_major_tmp_v24
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v276 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__row_major_tmp_v24
    uint64_t v277 = (uint64_t)v59;
    TASSIGN(v276, v277);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
    TSUB(v276, v272, v274);
    // pto: %45
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v278 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %45
    uint64_t v279 = (uint64_t)v59;
    TASSIGN(v278, v279);
    // pto: %n_sink_tile_inline2152_inline10278__rm_a0_tmp_v25
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v280 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %n_sink_tile_inline2152_inline10278__rm_a0_tmp_v25
    uint64_t v281 = (uint64_t)v59;
    TASSIGN(v280, v281);
    // pto: %n_sink_tile_inline2152_inline10278__rm_a1_tmp_v26
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v282 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %n_sink_tile_inline2152_inline10278__rm_a1_tmp_v26
    uint64_t v283 = (uint64_t)v62;
    TASSIGN(v282, v283);
    // pto: %n_sink_tile_inline2152_inline10278__row_major_tmp_v27
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v284 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %n_sink_tile_inline2152_inline10278__row_major_tmp_v27
    uint64_t v285 = (uint64_t)v62;
    TASSIGN(v284, v285);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    TADD(v284, v280, v282);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    // pto: %n_sink_tile_inline2152_inline10278__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v286 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %n_sink_tile_inline2152_inline10278__tile
    uint64_t v287 = (uint64_t)v62;
    TASSIGN(v286, v287);
    // pto: %t__rm_a0_tmp_v28
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v288 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__rm_a0_tmp_v28
    uint64_t v289 = (uint64_t)v62;
    TASSIGN(v288, v289);
    // pto: %t__rm_a1_tmp_v29
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v290 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__rm_a1_tmp_v29
    uint64_t v291 = (uint64_t)v67;
    TASSIGN(v290, v291);
    // pto: %t__row_major_tmp_v30
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v292 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__row_major_tmp_v30
    uint64_t v293 = (uint64_t)v62;
    TASSIGN(v292, v293);
    pipe_barrier(PIPE_V);
    TSUB(v292, v288, v290);
    // pto: %46
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v294 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %46
    uint64_t v295 = (uint64_t)v62;
    TASSIGN(v294, v295);
    // pto: %t__rm_a0_tmp_v31
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v296 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__rm_a0_tmp_v31
    uint64_t v297 = (uint64_t)v62;
    TASSIGN(v296, v297);
    // pto: %t__row_major_tmp_v32
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v298 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__row_major_tmp_v32
    uint64_t v299 = (uint64_t)v62;
    TASSIGN(v298, v299);
    pipe_barrier(PIPE_V);
    TEXP(v298, v296);
    // pto: %47
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v300 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %47
    uint64_t v301 = (uint64_t)v62;
    TASSIGN(v300, v301);
    // pto: %n_denom_inline2094_inline10726__rm_a0_tmp_v33
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v302 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %n_denom_inline2094_inline10726__rm_a0_tmp_v33
    uint64_t v303 = (uint64_t)v66;
    TASSIGN(v302, v303);
    // pto: %n_denom_inline2094_inline10726__rm_a1_tmp_v34
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v304 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %n_denom_inline2094_inline10726__rm_a1_tmp_v34
    uint64_t v305 = (uint64_t)v62;
    TASSIGN(v304, v305);
    // pto: %n_denom_inline2094_inline10726__row_major_tmp_v35
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v306 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %n_denom_inline2094_inline10726__row_major_tmp_v35
    uint64_t v307 = (uint64_t)v62;
    TASSIGN(v306, v307);
    pipe_barrier(PIPE_V);
    TADD(v306, v302, v304);
    // pto: %n_denom_inline2094_inline10726__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v308 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %n_denom_inline2094_inline10726__tile
    uint64_t v309 = (uint64_t)v62;
    TASSIGN(v308, v309);
    // pto: %48
    Tile<
        TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v310 = Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v31);
    // pto: %48
    uint64_t v311 = (uint64_t)v65;
    TASSIGN(v310, v311);
    pipe_barrier(PIPE_V);
    TROWEXPANDDIV(v310, v84, v308);
    // pto: %n_full_inline2093_inline10400__tile
    Tile<
        TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v312 = Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v31);
    // pto: %n_full_inline2093_inline10400__tile
    uint64_t v313 = (uint64_t)v65;
    TASSIGN(v312, v313);
    // pto: %slice_view
    Tile<
        TileType::Vec, float, 16, 512, BLayout::RowMajor, 16, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v314;
    // pto: %slice_view
    Tile<
        TileType::Vec, float, 16, 512, BLayout::RowMajor, 16, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v315 = v314;
    // pto: %slice_view
    uint64_t v316 = (uint64_t)v65;
    TASSIGN(v315, v316);
    // pto: %n_bf16_inline2092_inline10293__tile
    Tile<
        TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v317 = Tile<
            TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v31);
    // pto: %n_bf16_inline2092_inline10293__tile
    uint64_t v318 = (uint64_t)v62;
    TASSIGN(v317, v318);
    pipe_barrier(PIPE_V);
    TCVT(v317, v315, v12, v11);
    // pto: %m_rope_inline2091_inline10295__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v319 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %m_rope_inline2091_inline10295__tile
    uint64_t v320 = (uint64_t)v48;
    TASSIGN(v319, v320);
    // pto: %95
    Tile<
        TileType::Vec, float, 16, 512, BLayout::RowMajor, 16, 64, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v321;
    // pto: %95
    Tile<
        TileType::Vec, float, 16, 512, BLayout::RowMajor, 16, 64, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v322 = v321;
    // pto: %95
    uint64_t v323 = (uint64_t)v48;
    TASSIGN(v322, v323);
    // pto: %m_cos_il_inline2123_inline10276__tile
    Tile<
        TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v324 = Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v30);
    // pto: %m_cos_il_inline2123_inline10276__tile
    uint64_t v325 = (uint64_t)v52;
    TASSIGN(v324, v325);
    // pto: %96
    int64_t v326 = v70 < v65 ? v65 : v70;
    // pto: %rope_cos_il_inline2100_inline10619__ssa_v1_pview
    pto::Shape<1, 1, 1, 1, 64> v327 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %rope_cos_il_inline2100_inline10619__ssa_v1_pview
    pto::Stride<64, 64, 64, 64, 1> v328 = pto::Stride<64, 64, 64, 64, 1>();
    // pto: %rope_cos_il_inline2100_inline10619__ssa_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v329 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(
            v5 + (v65 + v326 * v30), v327, v328
        );
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    TLOAD(v324, v329);
    // pto: %m_sin_signed_inline2090_inline10298__tile
    Tile<
        TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v330 = Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v30);
    // pto: %m_sin_signed_inline2090_inline10298__tile
    uint64_t v331 = (uint64_t)v58;
    TASSIGN(v330, v331);
    // pto: %rope_sin_signed_inline2192_inline10396__ssa_v1_pview
    pto::Shape<1, 1, 1, 1, 64> v332 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %rope_sin_signed_inline2192_inline10396__ssa_v1_pview
    pto::Stride<64, 64, 64, 64, 1> v333 = pto::Stride<64, 64, 64, 64, 1>();
    // pto: %rope_sin_signed_inline2192_inline10396__ssa_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v334 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(
            v6 + (v65 + v326 * v30), v332, v333
        );
    TLOAD(v330, v334);
    // pto: %49
    Tile<
        TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v335 = Tile<
            TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %49
    uint64_t v336 = (uint64_t)v59;
    TASSIGN(v335, v336);
    // pto: %rope_swap_idx_inline2169_inline10828__ssa_v1_pview
    pto::Shape<1, 1, 1, 16, 64> v337 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %rope_swap_idx_inline2169_inline10828__ssa_v1_pview
    pto::Stride<1024, 1024, 1024, 64, 1> v338 = pto::Stride<1024, 1024, 1024, 64, 1>();
    // pto: %rope_swap_idx_inline2169_inline10828__ssa_v1_pview
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v339 =
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(
            v7, v337, v338
        );
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    TLOAD(v335, v339);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %gather_acc_init
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v340 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %gather_acc_init
    uint64_t v341 = (uint64_t)v57;
    TASSIGN(v340, v341);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    for (int64_t i342 = v65; i342 < v28; i342 += v32) {
        // pto: %gather_inp_row
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v343 = Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v30);
        // pto: %gather_inp_row
        uint64_t v344 = (uint64_t)v65;
        TASSIGN(v343, v344);
        // pto: %98
        Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v345;
        // pto: %98
        Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v346 = v345;
        // pto: %98
        uint64_t v347 = (uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)i342 * (uint64_t)v46)) + (uint64_t)v48));
        TASSIGN(v346, v347);
        // pto: %gather_idx_row
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v348 = Tile<
                TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v30);
        // pto: %gather_idx_row
        uint64_t v349 = (uint64_t)v59;
        TASSIGN(v348, v349);
        // pto: %99
        int64_t v350 = (int64_t)((uint64_t)i342 * (uint64_t)v68);
        // pto: %99
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v351;
        // pto: %99
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v352 = v351;
        // pto: %99
        uint64_t v353 = (uint64_t)((int64_t)((uint64_t)v350 + (uint64_t)v59));
        TASSIGN(v352, v353);
        // pto: %gather_row_tmp
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v354 = Tile<
                TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v30);
        // pto: %gather_row_tmp
        uint64_t v355 = (uint64_t)v56;
        TASSIGN(v354, v355);
        // pto: %gather_row
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v356 = Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v30);
        // pto: %gather_row
        uint64_t v357 = (uint64_t)v55;
        TASSIGN(v356, v357);
        pipe_barrier(PIPE_V);
        TGATHER(v356, v346, v352, v354);
        // pto: %assemble_view
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v358;
        // pto: %assemble_view
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v359 = v358;
        // pto: %assemble_view
        uint64_t v360 = (uint64_t)((int64_t)((uint64_t)v350 + (uint64_t)v57));
        TASSIGN(v359, v360);
        pipe_barrier(PIPE_V);
        TMOV(v359, v356);
    }
    // pto: %m_swapped_inline2088_inline10489__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v361 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %m_swapped_inline2088_inline10489__tile
    uint64_t v362 = (uint64_t)v57;
    TASSIGN(v361, v362);
    // pto: %m_rope_inline2091_inline10295__tile_textract
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v363 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %m_rope_inline2091_inline10295__tile_textract
    uint64_t v364 = (uint64_t)v59;
    TASSIGN(v363, v364);
    TEXTRACT(v363, v310, v65, v23);
    // pto: %50
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v365 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %50
    uint64_t v366 = (uint64_t)v65;
    TASSIGN(v365, v366);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v365, v363, v324);
    // pto: %51
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v367 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %51
    uint64_t v368 = (uint64_t)v59;
    TASSIGN(v367, v368);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v367, v361, v330);
    // pto: %m_rot_inline2087_inline10283__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v369 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %m_rot_inline2087_inline10283__tile
    uint64_t v370 = (uint64_t)v65;
    TASSIGN(v369, v370);
    pipe_barrier(PIPE_V);
    TADD(v369, v365, v367);
    // pto: %n_rope_bf16_inline2086_inline10275__tile
    Tile<
        TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v371 = Tile<
            TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %n_rope_bf16_inline2086_inline10275__tile
    uint64_t v372 = (uint64_t)v59;
    TASSIGN(v371, v372);
    pipe_barrier(PIPE_V);
    TCVT(v371, v369, v12, v11);
    // pto: %52
    Tile<
        TileType::Vec, bfloat16_t, 16, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v373 = Tile<
            TileType::Vec, bfloat16_t, 16, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v23);
    // pto: %52
    uint64_t v374 = (uint64_t)v62;
    TASSIGN(v373, v374);
    // pto: %100
    Tile<
        TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, 16, 448, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v375;
    // pto: %100
    Tile<
        TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, 16, 448, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v376 = v375;
    // pto: %100
    uint64_t v377 = (uint64_t)v62;
    TASSIGN(v376, v377);
    // pto: %n_full_bf16_inline2085_inline10274__tile
    Tile<
        TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v378 = Tile<
            TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v31);
    // pto: %n_full_bf16_inline2085_inline10274__tile
    uint64_t v379 = (uint64_t)v65;
    TASSIGN(v378, v379);
    pipe_barrier(PIPE_V);
    TCONCAT(v378, v376, v371);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %101, %102, %103
    int64_t v380 = (int64_t)((uint64_t)((int64_t)((uint64_t)(v71 / v29) * (uint64_t)v29)) + (uint64_t)v70);
    // pto: %104, %105
    int64_t v381 = (int64_t)((uint64_t)(v71 % v29) * (uint64_t)v31);
    // pto: %53
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v382 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %53
    uint64_t v383 = (uint64_t)v65;
    TASSIGN(v382, v383);
    // pto: %106
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v384;
    // pto: %106
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v385 = v384;
    // pto: %106
    uint64_t v386 = (uint64_t)v65;
    TASSIGN(v385, v386);
    // pto: %108
    int64_t v387 = v381 < v65 ? v65 : v381;
    // pto: %o_packed_inline2242_inline10294__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 512> v388 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %o_packed_inline2242_inline10294__ssa_v0_pview
    pto::Stride<4096, 4096, 4096, 4096, 1> v389 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %107, %o_packed_inline2242_inline10294__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v390 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v65 + (v380 < v65 ? v65 : v380) * v44) + v387), v388, v389
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v390, v385);
    // pto: %109
    int64_t v391 = (int64_t)((uint64_t)v71 + (uint64_t)v32);
    // pto: %110, %111, %112
    int64_t v392 = (int64_t)((uint64_t)((int64_t)((uint64_t)(v391 / v29) * (uint64_t)v29)) + (uint64_t)v70);
    // pto: %114, %115
    int64_t v393 = (int64_t)((uint64_t)(v391 % v29) * (uint64_t)v31);
    // pto: %54
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v394 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %54
    uint64_t v395 = (uint64_t)v47;
    TASSIGN(v394, v395);
    // pto: %116
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v396;
    // pto: %116
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v397 = v396;
    // pto: %116
    uint64_t v398 = (uint64_t)v47;
    TASSIGN(v397, v398);
    // pto: %119
    int64_t v399 = v393 < v65 ? v65 : v393;
    // pto: %o_packed_inline2242_inline10294__tile_pview
    pto::Shape<1, 1, 1, 1, 512> v400 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %o_packed_inline2242_inline10294__tile_pview
    pto::Stride<4096, 4096, 4096, 4096, 1> v401 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %118, %o_packed_inline2242_inline10294__tile_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v402 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v65 + (v392 < v65 ? v65 : v392) * v44) + v399), v400, v401
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v402, v397);
    // pto: %120
    int64_t v403 = (int64_t)((uint64_t)v71 + (uint64_t)v24);
    // pto: %121, %122, %123
    int64_t v404 = (int64_t)((uint64_t)((int64_t)((uint64_t)(v403 / v29) * (uint64_t)v29)) + (uint64_t)v70);
    // pto: %125, %126
    int64_t v405 = (int64_t)((uint64_t)(v403 % v29) * (uint64_t)v31);
    // pto: %55
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v406 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %55
    uint64_t v407 = (uint64_t)v46;
    TASSIGN(v406, v407);
    // pto: %127
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v408;
    // pto: %127
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v409 = v408;
    // pto: %127
    uint64_t v410 = (uint64_t)v46;
    TASSIGN(v409, v410);
    // pto: %130
    int64_t v411 = v405 < v65 ? v65 : v405;
    // pto: %131
    pto::Shape<1, 1, 1, 1, 512> v412 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %131
    pto::Stride<4096, 4096, 4096, 4096, 1> v413 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %129, %131
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v414 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v65 + (v404 < v65 ? v65 : v404) * v44) + v411), v412, v413
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v414, v409);
    // pto: %132
    int64_t v415 = (int64_t)((uint64_t)v71 + (uint64_t)v22);
    // pto: %133, %134, %135
    int64_t v416 = (int64_t)((uint64_t)((int64_t)((uint64_t)(v415 / v29) * (uint64_t)v29)) + (uint64_t)v70);
    // pto: %137, %138
    int64_t v417 = (int64_t)((uint64_t)(v415 % v29) * (uint64_t)v31);
    // pto: %56
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v418 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %56
    uint64_t v419 = (uint64_t)v45;
    TASSIGN(v418, v419);
    // pto: %139
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v420;
    // pto: %139
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v421 = v420;
    // pto: %139
    uint64_t v422 = (uint64_t)v45;
    TASSIGN(v421, v422);
    // pto: %142
    int64_t v423 = v417 < v65 ? v65 : v417;
    // pto: %143
    pto::Shape<1, 1, 1, 1, 512> v424 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %143
    pto::Stride<4096, 4096, 4096, 4096, 1> v425 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %141, %143
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v426 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v65 + (v416 < v65 ? v65 : v416) * v44) + v423), v424, v425
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v426, v421);
    // pto: %144
    int64_t v427 = (int64_t)((uint64_t)v71 + (uint64_t)v27);
    // pto: %145, %146, %147
    int64_t v428 = (int64_t)((uint64_t)((int64_t)((uint64_t)(v427 / v29) * (uint64_t)v29)) + (uint64_t)v70);
    // pto: %149, %150
    int64_t v429 = (int64_t)((uint64_t)(v427 % v29) * (uint64_t)v31);
    // pto: %57
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v430 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %57
    uint64_t v431 = (uint64_t)v44;
    TASSIGN(v430, v431);
    // pto: %151
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v432;
    // pto: %151
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v433 = v432;
    // pto: %151
    uint64_t v434 = (uint64_t)v44;
    TASSIGN(v433, v434);
    // pto: %154
    int64_t v435 = v429 < v65 ? v65 : v429;
    // pto: %155
    pto::Shape<1, 1, 1, 1, 512> v436 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %155
    pto::Stride<4096, 4096, 4096, 4096, 1> v437 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %153, %155
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v438 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v65 + (v428 < v65 ? v65 : v428) * v44) + v435), v436, v437
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v438, v433);
    // pto: %156
    int64_t v439 = (int64_t)((uint64_t)v71 + (uint64_t)v25);
    // pto: %157, %158, %159
    int64_t v440 = (int64_t)((uint64_t)((int64_t)((uint64_t)(v439 / v29) * (uint64_t)v29)) + (uint64_t)v70);
    // pto: %161, %162
    int64_t v441 = (int64_t)((uint64_t)(v439 % v29) * (uint64_t)v31);
    // pto: %58
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v442 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %58
    uint64_t v443 = (uint64_t)v43;
    TASSIGN(v442, v443);
    // pto: %163
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v444;
    // pto: %163
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v445 = v444;
    // pto: %163
    uint64_t v446 = (uint64_t)v43;
    TASSIGN(v445, v446);
    // pto: %166
    int64_t v447 = v441 < v65 ? v65 : v441;
    // pto: %167
    pto::Shape<1, 1, 1, 1, 512> v448 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %167
    pto::Stride<4096, 4096, 4096, 4096, 1> v449 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %165, %167
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v450 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v65 + (v440 < v65 ? v65 : v440) * v44) + v447), v448, v449
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v450, v445);
    // pto: %168
    int64_t v451 = (int64_t)((uint64_t)v71 + (uint64_t)v21);
    // pto: %169, %170, %171
    int64_t v452 = (int64_t)((uint64_t)((int64_t)((uint64_t)(v451 / v29) * (uint64_t)v29)) + (uint64_t)v70);
    // pto: %173, %174
    int64_t v453 = (int64_t)((uint64_t)(v451 % v29) * (uint64_t)v31);
    // pto: %59
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v454 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %59
    uint64_t v455 = (uint64_t)v42;
    TASSIGN(v454, v455);
    // pto: %175
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v456;
    // pto: %175
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v457 = v456;
    // pto: %175
    uint64_t v458 = (uint64_t)v42;
    TASSIGN(v457, v458);
    // pto: %178
    int64_t v459 = v453 < v65 ? v65 : v453;
    // pto: %179
    pto::Shape<1, 1, 1, 1, 512> v460 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %179
    pto::Stride<4096, 4096, 4096, 4096, 1> v461 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %177, %179
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v462 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v65 + (v452 < v65 ? v65 : v452) * v44) + v459), v460, v461
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v462, v457);
    // pto: %180
    int64_t v463 = (int64_t)((uint64_t)v71 + (uint64_t)v20);
    // pto: %181, %182, %183
    int64_t v464 = (int64_t)((uint64_t)((int64_t)((uint64_t)(v463 / v29) * (uint64_t)v29)) + (uint64_t)v70);
    // pto: %185, %186
    int64_t v465 = (int64_t)((uint64_t)(v463 % v29) * (uint64_t)v31);
    // pto: %60
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v466 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %60
    uint64_t v467 = (uint64_t)v41;
    TASSIGN(v466, v467);
    // pto: %187
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v468;
    // pto: %187
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v469 = v468;
    // pto: %187
    uint64_t v470 = (uint64_t)v41;
    TASSIGN(v469, v470);
    // pto: %190
    int64_t v471 = v465 < v65 ? v65 : v465;
    // pto: %191
    pto::Shape<1, 1, 1, 1, 512> v472 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %191
    pto::Stride<4096, 4096, 4096, 4096, 1> v473 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %189, %191
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v474 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v65 + (v464 < v65 ? v65 : v464) * v44) + v471), v472, v473
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v474, v469);
    // pto: %195
    int64_t v475 = (int64_t)((uint64_t)v380 + (uint64_t)v29);
    // pto: %61
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v476 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %61
    uint64_t v477 = (uint64_t)v40;
    TASSIGN(v476, v477);
    // pto: %198
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v478;
    // pto: %198
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v479 = v478;
    // pto: %198
    uint64_t v480 = (uint64_t)v40;
    TASSIGN(v479, v480);
    // pto: %202
    pto::Shape<1, 1, 1, 1, 512> v481 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %202
    pto::Stride<4096, 4096, 4096, 4096, 1> v482 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %200, %202
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v483 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v65 + (v475 < v65 ? v65 : v475) * v44) + v387), v481, v482
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v483, v479);
    // pto: %203, %204, %205, %206
    int64_t v484 =
        (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)v71 + (uint64_t)v19) / v29) * (uint64_t)v29)) +
                  (uint64_t)v70);
    // pto: %62
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v485 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %62
    uint64_t v486 = (uint64_t)v39;
    TASSIGN(v485, v486);
    // pto: %210
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v487;
    // pto: %210
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v488 = v487;
    // pto: %210
    uint64_t v489 = (uint64_t)v39;
    TASSIGN(v488, v489);
    // pto: %214
    pto::Shape<1, 1, 1, 1, 512> v490 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %214
    pto::Stride<4096, 4096, 4096, 4096, 1> v491 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %212, %214
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v492 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v65 + (v484 < v65 ? v65 : v484) * v44) + v399), v490, v491
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v492, v488);
    // pto: %215, %216, %217, %218
    int64_t v493 =
        (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)v71 + (uint64_t)v18) / v29) * (uint64_t)v29)) +
                  (uint64_t)v70);
    // pto: %63
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v494 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %63
    uint64_t v495 = (uint64_t)v38;
    TASSIGN(v494, v495);
    // pto: %222
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v496;
    // pto: %222
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v497 = v496;
    // pto: %222
    uint64_t v498 = (uint64_t)v38;
    TASSIGN(v497, v498);
    // pto: %226
    pto::Shape<1, 1, 1, 1, 512> v499 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %226
    pto::Stride<4096, 4096, 4096, 4096, 1> v500 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %224, %226
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v501 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v65 + (v493 < v65 ? v65 : v493) * v44) + v411), v499, v500
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v501, v497);
    // pto: %227, %228, %229, %230
    int64_t v502 =
        (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)v71 + (uint64_t)v17) / v29) * (uint64_t)v29)) +
                  (uint64_t)v70);
    // pto: %64
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v503 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %64
    uint64_t v504 = (uint64_t)v37;
    TASSIGN(v503, v504);
    // pto: %234
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v505;
    // pto: %234
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v506 = v505;
    // pto: %234
    uint64_t v507 = (uint64_t)v37;
    TASSIGN(v506, v507);
    // pto: %238
    pto::Shape<1, 1, 1, 1, 512> v508 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %238
    pto::Stride<4096, 4096, 4096, 4096, 1> v509 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %236, %238
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v510 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v65 + (v502 < v65 ? v65 : v502) * v44) + v423), v508, v509
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v510, v506);
    // pto: %239, %240, %241, %242
    int64_t v511 =
        (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)v71 + (uint64_t)v16) / v29) * (uint64_t)v29)) +
                  (uint64_t)v70);
    // pto: %65
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v512 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %65
    uint64_t v513 = (uint64_t)v36;
    TASSIGN(v512, v513);
    // pto: %246
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v514;
    // pto: %246
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v515 = v514;
    // pto: %246
    uint64_t v516 = (uint64_t)v36;
    TASSIGN(v515, v516);
    // pto: %250
    pto::Shape<1, 1, 1, 1, 512> v517 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %250
    pto::Stride<4096, 4096, 4096, 4096, 1> v518 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %248, %250
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v519 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v65 + (v511 < v65 ? v65 : v511) * v44) + v435), v517, v518
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v519, v515);
    // pto: %251, %252, %253, %254
    int64_t v520 =
        (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)v71 + (uint64_t)v15) / v29) * (uint64_t)v29)) +
                  (uint64_t)v70);
    // pto: %66
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v521 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %66
    uint64_t v522 = (uint64_t)v35;
    TASSIGN(v521, v522);
    // pto: %258
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v523;
    // pto: %258
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v524 = v523;
    // pto: %258
    uint64_t v525 = (uint64_t)v35;
    TASSIGN(v524, v525);
    // pto: %262
    pto::Shape<1, 1, 1, 1, 512> v526 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %262
    pto::Stride<4096, 4096, 4096, 4096, 1> v527 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %260, %262
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v528 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v65 + (v520 < v65 ? v65 : v520) * v44) + v447), v526, v527
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v528, v524);
    // pto: %263, %264, %265, %266
    int64_t v529 =
        (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)v71 + (uint64_t)v14) / v29) * (uint64_t)v29)) +
                  (uint64_t)v70);
    // pto: %67
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v530 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %67
    uint64_t v531 = (uint64_t)v34;
    TASSIGN(v530, v531);
    // pto: %270
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v532;
    // pto: %270
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v533 = v532;
    // pto: %270
    uint64_t v534 = (uint64_t)v34;
    TASSIGN(v533, v534);
    // pto: %274
    pto::Shape<1, 1, 1, 1, 512> v535 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %274
    pto::Stride<4096, 4096, 4096, 4096, 1> v536 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %272, %274
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v537 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v65 + (v529 < v65 ? v65 : v529) * v44) + v459), v535, v536
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v537, v533);
    // pto: %275, %276, %277, %278
    int64_t v538 =
        (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)v71 + (uint64_t)v13) / v29) * (uint64_t)v29)) +
                  (uint64_t)v70);
    // pto: %68
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v539 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %68
    uint64_t v540 = (uint64_t)v33;
    TASSIGN(v539, v540);
    // pto: %282
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v541;
    // pto: %282
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v542 = v541;
    // pto: %282
    uint64_t v543 = (uint64_t)v33;
    TASSIGN(v542, v543);
    // pto: %286
    pto::Shape<1, 1, 1, 1, 512> v544 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %286
    pto::Stride<4096, 4096, 4096, 4096, 1> v545 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %284, %286
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v546 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v65 + (v538 < v65 ? v65 : v538) * v44) + v471), v544, v545
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v546, v542);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: sparse_blk_mi_inline2120_inline10805__rv_v2
    __gm__ Tensor *sparse_blk_mi_inline2120_inline10805__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *sparse_blk_mi_inline2120_inline10805__rv_v2 =
        reinterpret_cast<__gm__ float *>(sparse_blk_mi_inline2120_inline10805__rv_v2_tensor->buffer.addr) +
        sparse_blk_mi_inline2120_inline10805__rv_v2_tensor->start_offset;

    // Unpack tensor: sparse_blk_li_inline2118_inline10891__rv_v2
    __gm__ Tensor *sparse_blk_li_inline2118_inline10891__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *sparse_blk_li_inline2118_inline10891__rv_v2 =
        reinterpret_cast<__gm__ float *>(sparse_blk_li_inline2118_inline10891__rv_v2_tensor->buffer.addr) +
        sparse_blk_li_inline2118_inline10891__rv_v2_tensor->start_offset;

    // Unpack tensor: sparse_blk_oi_inline2116_inline10892__rv_v2
    __gm__ Tensor *sparse_blk_oi_inline2116_inline10892__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *sparse_blk_oi_inline2116_inline10892__rv_v2 =
        reinterpret_cast<__gm__ float *>(sparse_blk_oi_inline2116_inline10892__rv_v2_tensor->buffer.addr) +
        sparse_blk_oi_inline2116_inline10892__rv_v2_tensor->start_offset;

    // Unpack tensor: attn_sink_csa_inline727__ssa_v0
    __gm__ Tensor *attn_sink_csa_inline727__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ float *attn_sink_csa_inline727__ssa_v0 =
        reinterpret_cast<__gm__ float *>(attn_sink_csa_inline727__ssa_v0_tensor->buffer.addr) +
        attn_sink_csa_inline727__ssa_v0_tensor->start_offset;

    // Unpack tensor: rope_cos_il_inline2100_inline10619__ssa_v1
    __gm__ Tensor *rope_cos_il_inline2100_inline10619__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ float *rope_cos_il_inline2100_inline10619__ssa_v1 =
        reinterpret_cast<__gm__ float *>(rope_cos_il_inline2100_inline10619__ssa_v1_tensor->buffer.addr) +
        rope_cos_il_inline2100_inline10619__ssa_v1_tensor->start_offset;

    // Unpack tensor: rope_sin_signed_inline2192_inline10396__ssa_v1
    __gm__ Tensor *rope_sin_signed_inline2192_inline10396__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[5]);
    __gm__ float *rope_sin_signed_inline2192_inline10396__ssa_v1 =
        reinterpret_cast<__gm__ float *>(rope_sin_signed_inline2192_inline10396__ssa_v1_tensor->buffer.addr) +
        rope_sin_signed_inline2192_inline10396__ssa_v1_tensor->start_offset;

    // Unpack tensor: rope_swap_idx_inline2169_inline10828__ssa_v1
    __gm__ Tensor *rope_swap_idx_inline2169_inline10828__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[6]);
    __gm__ int32_t *rope_swap_idx_inline2169_inline10828__ssa_v1 =
        reinterpret_cast<__gm__ int32_t *>(rope_swap_idx_inline2169_inline10828__ssa_v1_tensor->buffer.addr) +
        rope_swap_idx_inline2169_inline10828__ssa_v1_tensor->start_offset;

    // Unpack tensor: o_packed_inline2242_inline10294__ssa_v0
    __gm__ Tensor *o_packed_inline2242_inline10294__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[7]);
    __gm__ bfloat16_t *o_packed_inline2242_inline10294__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(o_packed_inline2242_inline10294__ssa_v0_tensor->buffer.addr) +
        o_packed_inline2242_inline10294__ssa_v0_tensor->start_offset;

    // Forward to ptoas-generated function
    merge_norm_1(
        sparse_blk_mi_inline2120_inline10805__rv_v2, sparse_blk_li_inline2118_inline10891__rv_v2,
        sparse_blk_oi_inline2116_inline10892__rv_v2, attn_sink_csa_inline727__ssa_v0,
        rope_cos_il_inline2100_inline10619__ssa_v1, rope_sin_signed_inline2192_inline10396__ssa_v1,
        rope_swap_idx_inline2169_inline10828__ssa_v1, o_packed_inline2242_inline10294__ssa_v0, __pypto_spmd_block_idx,
        __pypto_spmd_block_num
    );
}
