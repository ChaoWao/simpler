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
// Kernel Function: merge_norm_2

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

static __aicore__ void merge_norm_2(
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
    const int64_t v22 = 5;
    const int64_t v23 = 3;
    const int64_t v24 = 2;
    const int64_t v25 = 448;
    const int64_t v26 = 32;
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
    const int64_t v48 = 74624;
    const int64_t v49 = 1792;
    const int64_t v50 = 74368;
    const int64_t v51 = 65920;
    const int64_t v52 = 65664;
    const int64_t v53 = 70272;
    const int64_t v54 = 66176;
    const int64_t v55 = 32896;
    const int64_t v56 = 32832;
    const int64_t v57 = 32768;
    const int64_t v58 = 0;
    const int64_t v59 = 74944;
    const int64_t v60 = 74880;
    const int64_t v61 = 256;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %m_idx_inline1578_inline11437__ssa_v0
    int64_t v62 = (int64_t)v9;
    // pto: %32
    int64_t v63 = v62 / v27;
    // pto: %34, %33, %35
    int64_t v64 =
        (int64_t)((uint64_t)((int64_t)((uint64_t)v62 - (uint64_t)((int64_t)((uint64_t)v63 * (uint64_t)v27)))) *
                  (uint64_t)v28);
    // pto: %36
    int64_t v65 = (int64_t)((uint64_t)v62 * (uint64_t)v26);
    // pto: %m_mi_inline1517_inline11318__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v66 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %m_mi_inline1517_inline11318__tile
    uint64_t v67 = (uint64_t)v60;
    TASSIGN(v66, v67);
    // pto: %37
    int64_t v68 = v65 < v58 ? v58 : v65;
    // pto: %sparse_blk_mi_inline1555_inline11367__rv_v2_pview
    pto::Shape<1, 1, 1, 16, 1> v69 = pto::Shape<1, 1, 1, 16, 1>();
    // pto: %sparse_blk_mi_inline1555_inline11367__rv_v2_pview
    pto::Stride<16, 16, 16, 1, 1024> v70 = pto::Stride<16, 16, 16, 1, 1024>();
    // pto: %sparse_blk_mi_inline1555_inline11367__rv_v2_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 1024>, pto::Layout::DN> v71 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 1024>, pto::Layout::DN>(
            v1 + (v58 + v68), v69, v70
        );
    TLOAD(v66, v71);
    // pto: %m_li_inline1575_inline11317__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v72 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %m_li_inline1575_inline11317__tile
    uint64_t v73 = (uint64_t)v59;
    TASSIGN(v72, v73);
    // pto: %sparse_blk_li_inline1523_inline11348__rv_v2_pview
    pto::Shape<1, 1, 1, 16, 1> v74 = pto::Shape<1, 1, 1, 16, 1>();
    // pto: %sparse_blk_li_inline1523_inline11348__rv_v2_pview
    pto::Stride<16, 16, 16, 1, 1024> v75 = pto::Stride<16, 16, 16, 1, 1024>();
    // pto: %sparse_blk_li_inline1523_inline11348__rv_v2_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 1024>, pto::Layout::DN> v76 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 1024>, pto::Layout::DN>(
            v2 + (v58 + v68), v74, v75
        );
    TLOAD(v72, v76);
    // pto: %m_oi_inline1581_inline11316__tile
    Tile<
        TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v77 = Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v31);
    // pto: %m_oi_inline1581_inline11316__tile
    uint64_t v78 = (uint64_t)v58;
    TASSIGN(v77, v78);
    // pto: %sparse_blk_oi_inline1460_inline11347__rv_v2_pview
    pto::Shape<1, 1, 1, 16, 512> v79 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %sparse_blk_oi_inline1460_inline11347__rv_v2_pview
    pto::Stride<8192, 8192, 8192, 512, 1> v80 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %sparse_blk_oi_inline1460_inline11347__rv_v2_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v81 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
            v3 + (v58 + v68 * v31), v79, v80
        );
    TLOAD(v77, v81);
    // pto: %40
    int64_t v82 = (int64_t)((uint64_t)v65 + (uint64_t)v28);
    // pto: %m_cur_mi_inline1583_inline11555__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v83 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %m_cur_mi_inline1583_inline11555__tile
    uint64_t v84 = (uint64_t)v57;
    TASSIGN(v83, v84);
    // pto: %41
    int64_t v85 = v82 < v58 ? v58 : v82;
    // pto: %42
    pto::Shape<1, 1, 1, 16, 1> v86 = pto::Shape<1, 1, 1, 16, 1>();
    // pto: %42
    pto::Stride<16, 16, 16, 1, 1024> v87 = pto::Stride<16, 16, 16, 1, 1024>();
    // pto: %42
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 1024>, pto::Layout::DN> v88 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 1024>, pto::Layout::DN>(
            v1 + (v58 + v85), v86, v87
        );
    TLOAD(v83, v88);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %m_cur_li_inline1584_inline11315__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v89 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %m_cur_li_inline1584_inline11315__tile
    uint64_t v90 = (uint64_t)v56;
    TASSIGN(v89, v90);
    // pto: %44
    pto::Shape<1, 1, 1, 16, 1> v91 = pto::Shape<1, 1, 1, 16, 1>();
    // pto: %44
    pto::Stride<16, 16, 16, 1, 1024> v92 = pto::Stride<16, 16, 16, 1, 1024>();
    // pto: %44
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 1024>, pto::Layout::DN> v93 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 1024>, pto::Layout::DN>(
            v2 + (v58 + v85), v91, v92
        );
    TLOAD(v89, v93);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %m_cur_oi_inline1588_inline11314__tile
    Tile<
        TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v94 = Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v31);
    // pto: %m_cur_oi_inline1588_inline11314__tile
    uint64_t v95 = (uint64_t)v55;
    TASSIGN(v94, v95);
    // pto: %46
    pto::Shape<1, 1, 1, 16, 512> v96 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %46
    pto::Stride<8192, 8192, 8192, 512, 1> v97 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %46
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v98 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
            v3 + (v58 + v85 * v31), v96, v97
        );
    TLOAD(v94, v98);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    // pto: %m_mi_new_inline1589_inline11313__rm_a0_tmp_v0
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v99 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %m_mi_new_inline1589_inline11313__rm_a0_tmp_v0
    uint64_t v100 = (uint64_t)v60;
    TASSIGN(v99, v100);
    // pto: %m_mi_new_inline1589_inline11313__rm_a1_tmp_v1
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v101 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %m_mi_new_inline1589_inline11313__rm_a1_tmp_v1
    uint64_t v102 = (uint64_t)v57;
    TASSIGN(v101, v102);
    // pto: %m_mi_new_inline1589_inline11313__row_major_tmp_v2
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v103 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %m_mi_new_inline1589_inline11313__row_major_tmp_v2
    uint64_t v104 = (uint64_t)v54;
    TASSIGN(v103, v104);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMAX(v103, v99, v101);
    // pto: %m_mi_new_inline1589_inline11313__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v105 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %m_mi_new_inline1589_inline11313__tile
    uint64_t v106 = (uint64_t)v54;
    TASSIGN(v105, v106);
    // pto: %t__rm_a0_tmp_v3
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v107 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__rm_a0_tmp_v3
    uint64_t v108 = (uint64_t)v60;
    TASSIGN(v107, v108);
    // pto: %t__rm_a1_tmp_v4
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v109 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__rm_a1_tmp_v4
    uint64_t v110 = (uint64_t)v54;
    TASSIGN(v109, v110);
    // pto: %t__row_major_tmp_v5
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v111 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__row_major_tmp_v5
    uint64_t v112 = (uint64_t)v53;
    TASSIGN(v111, v112);
    pipe_barrier(PIPE_V);
    TSUB(v111, v107, v109);
    // pto: %t__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v113 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %t__tile
    uint64_t v114 = (uint64_t)v53;
    TASSIGN(v113, v114);
    // pto: %m_alpha_inline1508_inline11509__rm_a0_tmp_v6
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v115 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %m_alpha_inline1508_inline11509__rm_a0_tmp_v6
    uint64_t v116 = (uint64_t)v53;
    TASSIGN(v115, v116);
    // pto: %m_alpha_inline1508_inline11509__row_major_tmp_v7
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v117 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %m_alpha_inline1508_inline11509__row_major_tmp_v7
    uint64_t v118 = (uint64_t)v53;
    TASSIGN(v117, v118);
    pipe_barrier(PIPE_V);
    TEXP(v117, v115);
    // pto: %m_alpha_inline1508_inline11509__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v119 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %m_alpha_inline1508_inline11509__tile
    uint64_t v120 = (uint64_t)v53;
    TASSIGN(v119, v120);
    // pto: %t__rm_a0_tmp_v8
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v121 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__rm_a0_tmp_v8
    uint64_t v122 = (uint64_t)v57;
    TASSIGN(v121, v122);
    // pto: %t__rm_a1_tmp_v9
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v123 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__rm_a1_tmp_v9
    uint64_t v124 = (uint64_t)v54;
    TASSIGN(v123, v124);
    // pto: %t__row_major_tmp_v10
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v125 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__row_major_tmp_v10
    uint64_t v126 = (uint64_t)v52;
    TASSIGN(v125, v126);
    TSUB(v125, v121, v123);
    // pto: %0
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v127 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %0
    uint64_t v128 = (uint64_t)v52;
    TASSIGN(v127, v128);
    // pto: %m_beta_inline1561_inline11531__rm_a0_tmp_v11
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v129 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %m_beta_inline1561_inline11531__rm_a0_tmp_v11
    uint64_t v130 = (uint64_t)v52;
    TASSIGN(v129, v130);
    // pto: %m_beta_inline1561_inline11531__row_major_tmp_v12
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v131 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %m_beta_inline1561_inline11531__row_major_tmp_v12
    uint64_t v132 = (uint64_t)v52;
    TASSIGN(v131, v132);
    pipe_barrier(PIPE_V);
    TEXP(v131, v129);
    // pto: %m_beta_inline1561_inline11531__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v133 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %m_beta_inline1561_inline11531__tile
    uint64_t v134 = (uint64_t)v52;
    TASSIGN(v133, v134);
    // pto: %t__rm_a0_tmp_v13
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v135 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__rm_a0_tmp_v13
    uint64_t v136 = (uint64_t)v53;
    TASSIGN(v135, v136);
    // pto: %t__rm_a1_tmp_v14
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v137 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__rm_a1_tmp_v14
    uint64_t v138 = (uint64_t)v59;
    TASSIGN(v137, v138);
    // pto: %t__row_major_tmp_v15
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v139 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__row_major_tmp_v15
    uint64_t v140 = (uint64_t)v51;
    TASSIGN(v139, v140);
    TMUL(v139, v135, v137);
    // pto: %1
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v141 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %1
    uint64_t v142 = (uint64_t)v51;
    TASSIGN(v141, v142);
    // pto: %t__rm_a0_tmp_v16
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v143 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__rm_a0_tmp_v16
    uint64_t v144 = (uint64_t)v52;
    TASSIGN(v143, v144);
    // pto: %t__rm_a1_tmp_v17
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v145 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__rm_a1_tmp_v17
    uint64_t v146 = (uint64_t)v56;
    TASSIGN(v145, v146);
    // pto: %t__row_major_tmp_v18
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v147 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__row_major_tmp_v18
    uint64_t v148 = (uint64_t)v50;
    TASSIGN(v147, v148);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TMUL(v147, v143, v145);
    // pto: %2
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v149 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %2
    uint64_t v150 = (uint64_t)v50;
    TASSIGN(v149, v150);
    // pto: %m_li_inline1575_inline11317__rm_a0_tmp_v19
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v151 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %m_li_inline1575_inline11317__rm_a0_tmp_v19
    uint64_t v152 = (uint64_t)v51;
    TASSIGN(v151, v152);
    // pto: %m_li_inline1575_inline11317__rm_a1_tmp_v20
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v153 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %m_li_inline1575_inline11317__rm_a1_tmp_v20
    uint64_t v154 = (uint64_t)v50;
    TASSIGN(v153, v154);
    // pto: %m_li_inline1575_inline11317__row_major_tmp_v21
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v155 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %m_li_inline1575_inline11317__row_major_tmp_v21
    uint64_t v156 = (uint64_t)v51;
    TASSIGN(v155, v156);
    pipe_barrier(PIPE_V);
    TADD(v155, v151, v153);
    // pto: %3
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v157 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %3
    uint64_t v158 = (uint64_t)v51;
    TASSIGN(v157, v158);
    // pto: %4
    Tile<
        TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v159 = Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v31);
    // pto: %4
    uint64_t v160 = (uint64_t)v58;
    TASSIGN(v159, v160);
    TROWEXPANDMUL(v159, v77, v119);
    // pto: %5
    Tile<
        TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v161 = Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v31);
    // pto: %5
    uint64_t v162 = (uint64_t)v55;
    TASSIGN(v161, v162);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    TROWEXPANDMUL(v161, v94, v133);
    // pto: %6
    Tile<
        TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v163 = Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v31);
    // pto: %6
    uint64_t v164 = (uint64_t)v58;
    TASSIGN(v163, v164);
    pipe_barrier(PIPE_V);
    TADD(v163, v159, v161);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    // pto: %m_mi_inline1517_inline11318__ssa_v3
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v165 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %m_mi_inline1517_inline11318__ssa_v3
    uint64_t v166 = (uint64_t)v54;
    TASSIGN(v165, v166);
    // pto: %m_li_inline1575_inline11317__rv_v2
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v167 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %m_li_inline1575_inline11317__rv_v2
    uint64_t v168 = (uint64_t)v51;
    TASSIGN(v167, v168);
    // pto: %m_mi_inline1517_inline11318__rv_v2
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v169 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %m_mi_inline1517_inline11318__rv_v2
    uint64_t v170 = (uint64_t)v54;
    TASSIGN(v169, v170);
    // pto: %m_oi_inline1581_inline11316__rv_v2
    Tile<
        TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v171 = Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v31);
    // pto: %m_oi_inline1581_inline11316__rv_v2
    uint64_t v172 = (uint64_t)v58;
    TASSIGN(v171, v172);
    // pto: %7
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v173 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %7
    uint64_t v174 = (uint64_t)v55;
    TASSIGN(v173, v174);
    // pto: %attn_sink_hca_inline717__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 16> v175 = pto::Shape<1, 1, 1, 1, 16>();
    // pto: %attn_sink_hca_inline717__ssa_v0_pview
    pto::Stride<16, 16, 16, 16, 1> v176 = pto::Stride<16, 16, 16, 16, 1>();
    // pto: %47, %attn_sink_hca_inline717__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v177 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(
            v4 + (v58 + (v64 < v58 ? v58 : v64)), v175, v176
        );
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v173, v177);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    // pto: %n_sink_bias_inline1594_inline11684__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v178 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %n_sink_bias_inline1594_inline11684__tile
    uint64_t v179 = (uint64_t)v55;
    TASSIGN(v178, v179);
    // pto: %t__rm_a0_tmp_v22
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v180 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__rm_a0_tmp_v22
    uint64_t v181 = (uint64_t)v54;
    TASSIGN(v180, v181);
    // pto: %t__rm_a1_tmp_v23
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v182 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__rm_a1_tmp_v23
    uint64_t v183 = (uint64_t)v54;
    TASSIGN(v182, v183);
    // pto: %t__row_major_tmp_v24
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v184 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__row_major_tmp_v24
    uint64_t v185 = (uint64_t)v53;
    TASSIGN(v184, v185);
    TSUB(v184, v180, v182);
    // pto: %8
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v186 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %8
    uint64_t v187 = (uint64_t)v53;
    TASSIGN(v186, v187);
    // pto: %n_sink_tile_inline1527_inline11769__rm_a0_tmp_v25
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v188 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %n_sink_tile_inline1527_inline11769__rm_a0_tmp_v25
    uint64_t v189 = (uint64_t)v53;
    TASSIGN(v188, v189);
    // pto: %n_sink_tile_inline1527_inline11769__rm_a1_tmp_v26
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v190 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %n_sink_tile_inline1527_inline11769__rm_a1_tmp_v26
    uint64_t v191 = (uint64_t)v55;
    TASSIGN(v190, v191);
    // pto: %n_sink_tile_inline1527_inline11769__row_major_tmp_v27
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v192 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %n_sink_tile_inline1527_inline11769__row_major_tmp_v27
    uint64_t v193 = (uint64_t)v55;
    TASSIGN(v192, v193);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    TADD(v192, v188, v190);
    // pto: %n_sink_tile_inline1527_inline11769__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v194 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %n_sink_tile_inline1527_inline11769__tile
    uint64_t v195 = (uint64_t)v55;
    TASSIGN(v194, v195);
    // pto: %t__rm_a0_tmp_v28
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v196 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__rm_a0_tmp_v28
    uint64_t v197 = (uint64_t)v55;
    TASSIGN(v196, v197);
    // pto: %t__rm_a1_tmp_v29
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v198 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__rm_a1_tmp_v29
    uint64_t v199 = (uint64_t)v54;
    TASSIGN(v198, v199);
    // pto: %t__row_major_tmp_v30
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v200 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__row_major_tmp_v30
    uint64_t v201 = (uint64_t)v55;
    TASSIGN(v200, v201);
    pipe_barrier(PIPE_V);
    TSUB(v200, v196, v198);
    // pto: %9
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v202 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %9
    uint64_t v203 = (uint64_t)v55;
    TASSIGN(v202, v203);
    // pto: %t__rm_a0_tmp_v31
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v204 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__rm_a0_tmp_v31
    uint64_t v205 = (uint64_t)v55;
    TASSIGN(v204, v205);
    // pto: %t__row_major_tmp_v32
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v206 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %t__row_major_tmp_v32
    uint64_t v207 = (uint64_t)v55;
    TASSIGN(v206, v207);
    pipe_barrier(PIPE_V);
    TEXP(v206, v204);
    // pto: %10
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v208 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %10
    uint64_t v209 = (uint64_t)v55;
    TASSIGN(v208, v209);
    // pto: %n_denom_inline1598_inline11605__rm_a0_tmp_v33
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v210 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %n_denom_inline1598_inline11605__rm_a0_tmp_v33
    uint64_t v211 = (uint64_t)v51;
    TASSIGN(v210, v211);
    // pto: %n_denom_inline1598_inline11605__rm_a1_tmp_v34
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v212 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %n_denom_inline1598_inline11605__rm_a1_tmp_v34
    uint64_t v213 = (uint64_t)v55;
    TASSIGN(v212, v213);
    // pto: %n_denom_inline1598_inline11605__row_major_tmp_v35
    Tile<
        TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v214 = Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v28);
    // pto: %n_denom_inline1598_inline11605__row_major_tmp_v35
    uint64_t v215 = (uint64_t)v55;
    TASSIGN(v214, v215);
    pipe_barrier(PIPE_V);
    TADD(v214, v210, v212);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    // pto: %n_denom_inline1598_inline11605__tile
    Tile<
        TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v216 = Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v32);
    // pto: %n_denom_inline1598_inline11605__tile
    uint64_t v217 = (uint64_t)v55;
    TASSIGN(v216, v217);
    // pto: %11
    Tile<
        TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v218 = Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v31);
    // pto: %11
    uint64_t v219 = (uint64_t)v58;
    TASSIGN(v218, v219);
    pipe_barrier(PIPE_V);
    TROWEXPANDDIV(v218, v171, v216);
    // pto: %n_full_inline1595_inline11704__tile
    Tile<
        TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v220 = Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v31);
    // pto: %n_full_inline1595_inline11704__tile
    uint64_t v221 = (uint64_t)v58;
    TASSIGN(v220, v221);
    // pto: %slice_view
    Tile<
        TileType::Vec, float, 16, 512, BLayout::RowMajor, 16, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v222;
    // pto: %slice_view
    Tile<
        TileType::Vec, float, 16, 512, BLayout::RowMajor, 16, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v223 = v222;
    // pto: %slice_view
    uint64_t v224 = (uint64_t)v58;
    TASSIGN(v223, v224);
    // pto: %n_bf16_inline1601_inline11312__tile
    Tile<
        TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v225 = Tile<
            TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v31);
    // pto: %n_bf16_inline1601_inline11312__tile
    uint64_t v226 = (uint64_t)v55;
    TASSIGN(v225, v226);
    pipe_barrier(PIPE_V);
    TCVT(v225, v223, v12, v11);
    // pto: %m_rope_inline1602_inline11440__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v227 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %m_rope_inline1602_inline11440__tile
    uint64_t v228 = (uint64_t)v49;
    TASSIGN(v227, v228);
    // pto: %48
    Tile<
        TileType::Vec, float, 16, 512, BLayout::RowMajor, 16, 64, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v229;
    // pto: %48
    Tile<
        TileType::Vec, float, 16, 512, BLayout::RowMajor, 16, 64, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v230 = v229;
    // pto: %48
    uint64_t v231 = (uint64_t)v49;
    TASSIGN(v230, v231);
    // pto: %m_cos_il_inline1603_inline11706__tile
    Tile<
        TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v232 = Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v30);
    // pto: %m_cos_il_inline1603_inline11706__tile
    uint64_t v233 = (uint64_t)v52;
    TASSIGN(v232, v233);
    // pto: %49
    int64_t v234 = v63 < v58 ? v58 : v63;
    // pto: %rope_cos_il_inline1559_inline11376__ssa_v1_pview
    pto::Shape<1, 1, 1, 1, 64> v235 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %rope_cos_il_inline1559_inline11376__ssa_v1_pview
    pto::Stride<64, 64, 64, 64, 1> v236 = pto::Stride<64, 64, 64, 64, 1>();
    // pto: %rope_cos_il_inline1559_inline11376__ssa_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v237 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(
            v5 + (v58 + v234 * v30), v235, v236
        );
    TLOAD(v232, v237);
    // pto: %m_sin_signed_inline1592_inline11311__tile
    Tile<
        TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v238 = Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v30);
    // pto: %m_sin_signed_inline1592_inline11311__tile
    uint64_t v239 = (uint64_t)v51;
    TASSIGN(v238, v239);
    // pto: %rope_sin_signed_inline1560_inline11790__ssa_v1_pview
    pto::Shape<1, 1, 1, 1, 64> v240 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %rope_sin_signed_inline1560_inline11790__ssa_v1_pview
    pto::Stride<64, 64, 64, 64, 1> v241 = pto::Stride<64, 64, 64, 64, 1>();
    // pto: %rope_sin_signed_inline1560_inline11790__ssa_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v242 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(
            v6 + (v58 + v234 * v30), v240, v241
        );
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    TLOAD(v238, v242);
    // pto: %12
    Tile<
        TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v243 = Tile<
            TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %12
    uint64_t v244 = (uint64_t)v54;
    TASSIGN(v243, v244);
    // pto: %rope_swap_idx_inline1490_inline11789__ssa_v1_pview
    pto::Shape<1, 1, 1, 16, 64> v245 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %rope_swap_idx_inline1490_inline11789__ssa_v1_pview
    pto::Stride<1024, 1024, 1024, 64, 1> v246 = pto::Stride<1024, 1024, 1024, 64, 1>();
    // pto: %rope_swap_idx_inline1490_inline11789__ssa_v1_pview
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v247 =
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(
            v7, v245, v246
        );
    TLOAD(v243, v247);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    // pto: %gather_acc_init
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v248 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %gather_acc_init
    uint64_t v249 = (uint64_t)v53;
    TASSIGN(v248, v249);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    for (int64_t i250 = v58; i250 < v28; i250 += v32) {
        // pto: %gather_inp_row
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v251 = Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v30);
        // pto: %gather_inp_row
        uint64_t v252 = (uint64_t)v58;
        TASSIGN(v251, v252);
        // pto: %51
        Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v253;
        // pto: %51
        Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v254 = v253;
        // pto: %51
        uint64_t v255 = (uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)i250 * (uint64_t)v46)) + (uint64_t)v49));
        TASSIGN(v254, v255);
        // pto: %gather_idx_row
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v256 = Tile<
                TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v30);
        // pto: %gather_idx_row
        uint64_t v257 = (uint64_t)v54;
        TASSIGN(v256, v257);
        // pto: %52
        int64_t v258 = (int64_t)((uint64_t)i250 * (uint64_t)v61);
        // pto: %52
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v259;
        // pto: %52
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v260 = v259;
        // pto: %52
        uint64_t v261 = (uint64_t)((int64_t)((uint64_t)v258 + (uint64_t)v54));
        TASSIGN(v260, v261);
        // pto: %gather_row_tmp
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v262 = Tile<
                TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v30);
        // pto: %gather_row_tmp
        uint64_t v263 = (uint64_t)v50;
        TASSIGN(v262, v263);
        // pto: %gather_row
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v264 = Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v32, v30);
        // pto: %gather_row
        uint64_t v265 = (uint64_t)v48;
        TASSIGN(v264, v265);
        pipe_barrier(PIPE_V);
        TGATHER(v264, v254, v260, v262);
        // pto: %assemble_view
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v266;
        // pto: %assemble_view
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v267 = v266;
        // pto: %assemble_view
        uint64_t v268 = (uint64_t)((int64_t)((uint64_t)v258 + (uint64_t)v53));
        TASSIGN(v267, v268);
        pipe_barrier(PIPE_V);
        TMOV(v267, v264);
    }
    // pto: %m_swapped_inline1604_inline11442__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v269 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %m_swapped_inline1604_inline11442__tile
    uint64_t v270 = (uint64_t)v53;
    TASSIGN(v269, v270);
    // pto: %m_rope_inline1602_inline11440__tile_textract
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v271 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %m_rope_inline1602_inline11440__tile_textract
    uint64_t v272 = (uint64_t)v54;
    TASSIGN(v271, v272);
    TEXTRACT(v271, v218, v58, v25);
    // pto: %13
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v273 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %13
    uint64_t v274 = (uint64_t)v58;
    TASSIGN(v273, v274);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v273, v271, v232);
    // pto: %14
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v275 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %14
    uint64_t v276 = (uint64_t)v54;
    TASSIGN(v275, v276);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v275, v269, v238);
    // pto: %m_rot_inline1550_inline11310__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v277 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %m_rot_inline1550_inline11310__tile
    uint64_t v278 = (uint64_t)v58;
    TASSIGN(v277, v278);
    pipe_barrier(PIPE_V);
    TADD(v277, v273, v275);
    // pto: %n_rope_bf16_inline1509_inline11381__tile
    Tile<
        TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v279 = Tile<
            TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v30);
    // pto: %n_rope_bf16_inline1509_inline11381__tile
    uint64_t v280 = (uint64_t)v54;
    TASSIGN(v279, v280);
    pipe_barrier(PIPE_V);
    TCVT(v279, v277, v12, v11);
    // pto: %15
    Tile<
        TileType::Vec, bfloat16_t, 16, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v281 = Tile<
            TileType::Vec, bfloat16_t, 16, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v25);
    // pto: %15
    uint64_t v282 = (uint64_t)v55;
    TASSIGN(v281, v282);
    // pto: %53
    Tile<
        TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, 16, 448, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v283;
    // pto: %53
    Tile<
        TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, 16, 448, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v284 = v283;
    // pto: %53
    uint64_t v285 = (uint64_t)v55;
    TASSIGN(v284, v285);
    // pto: %n_full_bf16_inline1585_inline11309__tile
    Tile<
        TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v286 = Tile<
            TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v28, v31);
    // pto: %n_full_bf16_inline1585_inline11309__tile
    uint64_t v287 = (uint64_t)v58;
    TASSIGN(v286, v287);
    pipe_barrier(PIPE_V);
    TCONCAT(v286, v284, v279);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %54, %55, %56
    int64_t v288 = (int64_t)((uint64_t)((int64_t)((uint64_t)(v64 / v29) * (uint64_t)v29)) + (uint64_t)v63);
    // pto: %57, %58
    int64_t v289 = (int64_t)((uint64_t)(v64 % v29) * (uint64_t)v31);
    // pto: %16
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v290 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %16
    uint64_t v291 = (uint64_t)v58;
    TASSIGN(v290, v291);
    // pto: %59
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v292;
    // pto: %59
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v293 = v292;
    // pto: %59
    uint64_t v294 = (uint64_t)v58;
    TASSIGN(v293, v294);
    // pto: %61
    int64_t v295 = v289 < v58 ? v58 : v289;
    // pto: %o_packed_inline1463_inline11349__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 512> v296 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %o_packed_inline1463_inline11349__ssa_v0_pview
    pto::Stride<4096, 4096, 4096, 4096, 1> v297 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %60, %o_packed_inline1463_inline11349__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v298 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v58 + (v288 < v58 ? v58 : v288) * v44) + v295), v296, v297
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v298, v293);
    // pto: %62
    int64_t v299 = (int64_t)((uint64_t)v64 + (uint64_t)v32);
    // pto: %63, %64, %65
    int64_t v300 = (int64_t)((uint64_t)((int64_t)((uint64_t)(v299 / v29) * (uint64_t)v29)) + (uint64_t)v63);
    // pto: %67, %68
    int64_t v301 = (int64_t)((uint64_t)(v299 % v29) * (uint64_t)v31);
    // pto: %17
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v302 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %17
    uint64_t v303 = (uint64_t)v47;
    TASSIGN(v302, v303);
    // pto: %69
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v304;
    // pto: %69
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v305 = v304;
    // pto: %69
    uint64_t v306 = (uint64_t)v47;
    TASSIGN(v305, v306);
    // pto: %72
    int64_t v307 = v301 < v58 ? v58 : v301;
    // pto: %o_packed_inline1463_inline11349__tile_pview
    pto::Shape<1, 1, 1, 1, 512> v308 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %o_packed_inline1463_inline11349__tile_pview
    pto::Stride<4096, 4096, 4096, 4096, 1> v309 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %71, %o_packed_inline1463_inline11349__tile_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v310 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v58 + (v300 < v58 ? v58 : v300) * v44) + v307), v308, v309
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v310, v305);
    // pto: %73
    int64_t v311 = (int64_t)((uint64_t)v64 + (uint64_t)v24);
    // pto: %74, %75, %76
    int64_t v312 = (int64_t)((uint64_t)((int64_t)((uint64_t)(v311 / v29) * (uint64_t)v29)) + (uint64_t)v63);
    // pto: %78, %79
    int64_t v313 = (int64_t)((uint64_t)(v311 % v29) * (uint64_t)v31);
    // pto: %18
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v314 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %18
    uint64_t v315 = (uint64_t)v46;
    TASSIGN(v314, v315);
    // pto: %80
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v316;
    // pto: %80
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v317 = v316;
    // pto: %80
    uint64_t v318 = (uint64_t)v46;
    TASSIGN(v317, v318);
    // pto: %83
    int64_t v319 = v313 < v58 ? v58 : v313;
    // pto: %84
    pto::Shape<1, 1, 1, 1, 512> v320 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %84
    pto::Stride<4096, 4096, 4096, 4096, 1> v321 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %82, %84
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v322 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v58 + (v312 < v58 ? v58 : v312) * v44) + v319), v320, v321
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v322, v317);
    // pto: %85
    int64_t v323 = (int64_t)((uint64_t)v64 + (uint64_t)v23);
    // pto: %86, %87, %88
    int64_t v324 = (int64_t)((uint64_t)((int64_t)((uint64_t)(v323 / v29) * (uint64_t)v29)) + (uint64_t)v63);
    // pto: %90, %91
    int64_t v325 = (int64_t)((uint64_t)(v323 % v29) * (uint64_t)v31);
    // pto: %19
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v326 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %19
    uint64_t v327 = (uint64_t)v45;
    TASSIGN(v326, v327);
    // pto: %92
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v328;
    // pto: %92
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v329 = v328;
    // pto: %92
    uint64_t v330 = (uint64_t)v45;
    TASSIGN(v329, v330);
    // pto: %95
    int64_t v331 = v325 < v58 ? v58 : v325;
    // pto: %96
    pto::Shape<1, 1, 1, 1, 512> v332 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %96
    pto::Stride<4096, 4096, 4096, 4096, 1> v333 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %94, %96
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v334 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v58 + (v324 < v58 ? v58 : v324) * v44) + v331), v332, v333
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v334, v329);
    // pto: %97
    int64_t v335 = (int64_t)((uint64_t)v64 + (uint64_t)v27);
    // pto: %98, %99, %100
    int64_t v336 = (int64_t)((uint64_t)((int64_t)((uint64_t)(v335 / v29) * (uint64_t)v29)) + (uint64_t)v63);
    // pto: %102, %103
    int64_t v337 = (int64_t)((uint64_t)(v335 % v29) * (uint64_t)v31);
    // pto: %20
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v338 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %20
    uint64_t v339 = (uint64_t)v44;
    TASSIGN(v338, v339);
    // pto: %104
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v340;
    // pto: %104
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v341 = v340;
    // pto: %104
    uint64_t v342 = (uint64_t)v44;
    TASSIGN(v341, v342);
    // pto: %107
    int64_t v343 = v337 < v58 ? v58 : v337;
    // pto: %108
    pto::Shape<1, 1, 1, 1, 512> v344 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %108
    pto::Stride<4096, 4096, 4096, 4096, 1> v345 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %106, %108
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v346 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v58 + (v336 < v58 ? v58 : v336) * v44) + v343), v344, v345
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v346, v341);
    // pto: %109
    int64_t v347 = (int64_t)((uint64_t)v64 + (uint64_t)v22);
    // pto: %110, %111, %112
    int64_t v348 = (int64_t)((uint64_t)((int64_t)((uint64_t)(v347 / v29) * (uint64_t)v29)) + (uint64_t)v63);
    // pto: %114, %115
    int64_t v349 = (int64_t)((uint64_t)(v347 % v29) * (uint64_t)v31);
    // pto: %21
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v350 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %21
    uint64_t v351 = (uint64_t)v43;
    TASSIGN(v350, v351);
    // pto: %116
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v352;
    // pto: %116
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v353 = v352;
    // pto: %116
    uint64_t v354 = (uint64_t)v43;
    TASSIGN(v353, v354);
    // pto: %119
    int64_t v355 = v349 < v58 ? v58 : v349;
    // pto: %120
    pto::Shape<1, 1, 1, 1, 512> v356 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %120
    pto::Stride<4096, 4096, 4096, 4096, 1> v357 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %118, %120
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v358 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v58 + (v348 < v58 ? v58 : v348) * v44) + v355), v356, v357
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v358, v353);
    // pto: %121
    int64_t v359 = (int64_t)((uint64_t)v64 + (uint64_t)v21);
    // pto: %122, %123, %124
    int64_t v360 = (int64_t)((uint64_t)((int64_t)((uint64_t)(v359 / v29) * (uint64_t)v29)) + (uint64_t)v63);
    // pto: %126, %127
    int64_t v361 = (int64_t)((uint64_t)(v359 % v29) * (uint64_t)v31);
    // pto: %22
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v362 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %22
    uint64_t v363 = (uint64_t)v42;
    TASSIGN(v362, v363);
    // pto: %128
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v364;
    // pto: %128
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v365 = v364;
    // pto: %128
    uint64_t v366 = (uint64_t)v42;
    TASSIGN(v365, v366);
    // pto: %131
    int64_t v367 = v361 < v58 ? v58 : v361;
    // pto: %132
    pto::Shape<1, 1, 1, 1, 512> v368 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %132
    pto::Stride<4096, 4096, 4096, 4096, 1> v369 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %130, %132
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v370 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v58 + (v360 < v58 ? v58 : v360) * v44) + v367), v368, v369
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v370, v365);
    // pto: %133
    int64_t v371 = (int64_t)((uint64_t)v64 + (uint64_t)v20);
    // pto: %134, %135, %136
    int64_t v372 = (int64_t)((uint64_t)((int64_t)((uint64_t)(v371 / v29) * (uint64_t)v29)) + (uint64_t)v63);
    // pto: %138, %139
    int64_t v373 = (int64_t)((uint64_t)(v371 % v29) * (uint64_t)v31);
    // pto: %23
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v374 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %23
    uint64_t v375 = (uint64_t)v41;
    TASSIGN(v374, v375);
    // pto: %140
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v376;
    // pto: %140
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v377 = v376;
    // pto: %140
    uint64_t v378 = (uint64_t)v41;
    TASSIGN(v377, v378);
    // pto: %143
    int64_t v379 = v373 < v58 ? v58 : v373;
    // pto: %144
    pto::Shape<1, 1, 1, 1, 512> v380 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %144
    pto::Stride<4096, 4096, 4096, 4096, 1> v381 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %142, %144
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v382 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v58 + (v372 < v58 ? v58 : v372) * v44) + v379), v380, v381
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v382, v377);
    // pto: %148
    int64_t v383 = (int64_t)((uint64_t)v288 + (uint64_t)v29);
    // pto: %24
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v384 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %24
    uint64_t v385 = (uint64_t)v40;
    TASSIGN(v384, v385);
    // pto: %151
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v386;
    // pto: %151
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v387 = v386;
    // pto: %151
    uint64_t v388 = (uint64_t)v40;
    TASSIGN(v387, v388);
    // pto: %155
    pto::Shape<1, 1, 1, 1, 512> v389 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %155
    pto::Stride<4096, 4096, 4096, 4096, 1> v390 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %153, %155
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v391 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v58 + (v383 < v58 ? v58 : v383) * v44) + v295), v389, v390
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v391, v387);
    // pto: %156, %157, %158, %159
    int64_t v392 =
        (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)v64 + (uint64_t)v19) / v29) * (uint64_t)v29)) +
                  (uint64_t)v63);
    // pto: %25
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v393 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %25
    uint64_t v394 = (uint64_t)v39;
    TASSIGN(v393, v394);
    // pto: %163
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v395;
    // pto: %163
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v396 = v395;
    // pto: %163
    uint64_t v397 = (uint64_t)v39;
    TASSIGN(v396, v397);
    // pto: %167
    pto::Shape<1, 1, 1, 1, 512> v398 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %167
    pto::Stride<4096, 4096, 4096, 4096, 1> v399 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %165, %167
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v400 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v58 + (v392 < v58 ? v58 : v392) * v44) + v307), v398, v399
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v400, v396);
    // pto: %168, %169, %170, %171
    int64_t v401 =
        (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)v64 + (uint64_t)v18) / v29) * (uint64_t)v29)) +
                  (uint64_t)v63);
    // pto: %26
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v402 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %26
    uint64_t v403 = (uint64_t)v38;
    TASSIGN(v402, v403);
    // pto: %175
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v404;
    // pto: %175
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v405 = v404;
    // pto: %175
    uint64_t v406 = (uint64_t)v38;
    TASSIGN(v405, v406);
    // pto: %179
    pto::Shape<1, 1, 1, 1, 512> v407 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %179
    pto::Stride<4096, 4096, 4096, 4096, 1> v408 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %177, %179
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v409 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v58 + (v401 < v58 ? v58 : v401) * v44) + v319), v407, v408
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v409, v405);
    // pto: %180, %181, %182, %183
    int64_t v410 =
        (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)v64 + (uint64_t)v17) / v29) * (uint64_t)v29)) +
                  (uint64_t)v63);
    // pto: %27
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v411 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %27
    uint64_t v412 = (uint64_t)v37;
    TASSIGN(v411, v412);
    // pto: %187
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v413;
    // pto: %187
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v414 = v413;
    // pto: %187
    uint64_t v415 = (uint64_t)v37;
    TASSIGN(v414, v415);
    // pto: %191
    pto::Shape<1, 1, 1, 1, 512> v416 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %191
    pto::Stride<4096, 4096, 4096, 4096, 1> v417 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %189, %191
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v418 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v58 + (v410 < v58 ? v58 : v410) * v44) + v331), v416, v417
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v418, v414);
    // pto: %192, %193, %194, %195
    int64_t v419 =
        (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)v64 + (uint64_t)v16) / v29) * (uint64_t)v29)) +
                  (uint64_t)v63);
    // pto: %28
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v420 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %28
    uint64_t v421 = (uint64_t)v36;
    TASSIGN(v420, v421);
    // pto: %199
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v422;
    // pto: %199
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v423 = v422;
    // pto: %199
    uint64_t v424 = (uint64_t)v36;
    TASSIGN(v423, v424);
    // pto: %203
    pto::Shape<1, 1, 1, 1, 512> v425 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %203
    pto::Stride<4096, 4096, 4096, 4096, 1> v426 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %201, %203
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v427 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v58 + (v419 < v58 ? v58 : v419) * v44) + v343), v425, v426
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v427, v423);
    // pto: %204, %205, %206, %207
    int64_t v428 =
        (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)v64 + (uint64_t)v15) / v29) * (uint64_t)v29)) +
                  (uint64_t)v63);
    // pto: %29
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v429 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %29
    uint64_t v430 = (uint64_t)v35;
    TASSIGN(v429, v430);
    // pto: %211
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v431;
    // pto: %211
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v432 = v431;
    // pto: %211
    uint64_t v433 = (uint64_t)v35;
    TASSIGN(v432, v433);
    // pto: %215
    pto::Shape<1, 1, 1, 1, 512> v434 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %215
    pto::Stride<4096, 4096, 4096, 4096, 1> v435 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %213, %215
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v436 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v58 + (v428 < v58 ? v58 : v428) * v44) + v355), v434, v435
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v436, v432);
    // pto: %216, %217, %218, %219
    int64_t v437 =
        (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)v64 + (uint64_t)v14) / v29) * (uint64_t)v29)) +
                  (uint64_t)v63);
    // pto: %30
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v438 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %30
    uint64_t v439 = (uint64_t)v34;
    TASSIGN(v438, v439);
    // pto: %223
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v440;
    // pto: %223
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v441 = v440;
    // pto: %223
    uint64_t v442 = (uint64_t)v34;
    TASSIGN(v441, v442);
    // pto: %227
    pto::Shape<1, 1, 1, 1, 512> v443 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %227
    pto::Stride<4096, 4096, 4096, 4096, 1> v444 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %225, %227
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v445 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v58 + (v437 < v58 ? v58 : v437) * v44) + v367), v443, v444
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v445, v441);
    // pto: %228, %229, %230, %231
    int64_t v446 =
        (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)v64 + (uint64_t)v13) / v29) * (uint64_t)v29)) +
                  (uint64_t)v63);
    // pto: %31
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v447 = Tile<
            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v32, v31);
    // pto: %31
    uint64_t v448 = (uint64_t)v33;
    TASSIGN(v447, v448);
    // pto: %235
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v449;
    // pto: %235
    Tile<
        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v450 = v449;
    // pto: %235
    uint64_t v451 = (uint64_t)v33;
    TASSIGN(v450, v451);
    // pto: %239
    pto::Shape<1, 1, 1, 1, 512> v452 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %239
    pto::Stride<4096, 4096, 4096, 4096, 1> v453 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %237, %239
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v454 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v8 + ((v58 + (v446 < v58 ? v58 : v446) * v44) + v379), v452, v453
        );
    pipe_barrier(PIPE_MTE3);
    TSTORE(v454, v450);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: sparse_blk_mi_inline1555_inline11367__rv_v2
    __gm__ Tensor *sparse_blk_mi_inline1555_inline11367__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *sparse_blk_mi_inline1555_inline11367__rv_v2 =
        reinterpret_cast<__gm__ float *>(sparse_blk_mi_inline1555_inline11367__rv_v2_tensor->buffer.addr) +
        sparse_blk_mi_inline1555_inline11367__rv_v2_tensor->start_offset;

    // Unpack tensor: sparse_blk_li_inline1523_inline11348__rv_v2
    __gm__ Tensor *sparse_blk_li_inline1523_inline11348__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *sparse_blk_li_inline1523_inline11348__rv_v2 =
        reinterpret_cast<__gm__ float *>(sparse_blk_li_inline1523_inline11348__rv_v2_tensor->buffer.addr) +
        sparse_blk_li_inline1523_inline11348__rv_v2_tensor->start_offset;

    // Unpack tensor: sparse_blk_oi_inline1460_inline11347__rv_v2
    __gm__ Tensor *sparse_blk_oi_inline1460_inline11347__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *sparse_blk_oi_inline1460_inline11347__rv_v2 =
        reinterpret_cast<__gm__ float *>(sparse_blk_oi_inline1460_inline11347__rv_v2_tensor->buffer.addr) +
        sparse_blk_oi_inline1460_inline11347__rv_v2_tensor->start_offset;

    // Unpack tensor: attn_sink_hca_inline717__ssa_v0
    __gm__ Tensor *attn_sink_hca_inline717__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ float *attn_sink_hca_inline717__ssa_v0 =
        reinterpret_cast<__gm__ float *>(attn_sink_hca_inline717__ssa_v0_tensor->buffer.addr) +
        attn_sink_hca_inline717__ssa_v0_tensor->start_offset;

    // Unpack tensor: rope_cos_il_inline1559_inline11376__ssa_v1
    __gm__ Tensor *rope_cos_il_inline1559_inline11376__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ float *rope_cos_il_inline1559_inline11376__ssa_v1 =
        reinterpret_cast<__gm__ float *>(rope_cos_il_inline1559_inline11376__ssa_v1_tensor->buffer.addr) +
        rope_cos_il_inline1559_inline11376__ssa_v1_tensor->start_offset;

    // Unpack tensor: rope_sin_signed_inline1560_inline11790__ssa_v1
    __gm__ Tensor *rope_sin_signed_inline1560_inline11790__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[5]);
    __gm__ float *rope_sin_signed_inline1560_inline11790__ssa_v1 =
        reinterpret_cast<__gm__ float *>(rope_sin_signed_inline1560_inline11790__ssa_v1_tensor->buffer.addr) +
        rope_sin_signed_inline1560_inline11790__ssa_v1_tensor->start_offset;

    // Unpack tensor: rope_swap_idx_inline1490_inline11789__ssa_v1
    __gm__ Tensor *rope_swap_idx_inline1490_inline11789__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[6]);
    __gm__ int32_t *rope_swap_idx_inline1490_inline11789__ssa_v1 =
        reinterpret_cast<__gm__ int32_t *>(rope_swap_idx_inline1490_inline11789__ssa_v1_tensor->buffer.addr) +
        rope_swap_idx_inline1490_inline11789__ssa_v1_tensor->start_offset;

    // Unpack tensor: o_packed_inline1463_inline11349__ssa_v0
    __gm__ Tensor *o_packed_inline1463_inline11349__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[7]);
    __gm__ bfloat16_t *o_packed_inline1463_inline11349__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(o_packed_inline1463_inline11349__ssa_v0_tensor->buffer.addr) +
        o_packed_inline1463_inline11349__ssa_v0_tensor->start_offset;

    // Forward to ptoas-generated function
    merge_norm_2(
        sparse_blk_mi_inline1555_inline11367__rv_v2, sparse_blk_li_inline1523_inline11348__rv_v2,
        sparse_blk_oi_inline1460_inline11347__rv_v2, attn_sink_hca_inline717__ssa_v0,
        rope_cos_il_inline1559_inline11376__ssa_v1, rope_sin_signed_inline1560_inline11790__ssa_v1,
        rope_swap_idx_inline1490_inline11789__ssa_v1, o_packed_inline1463_inline11349__ssa_v0, __pypto_spmd_block_idx,
        __pypto_spmd_block_num
    );
}
