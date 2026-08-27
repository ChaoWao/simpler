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
// Kernel Function: hc_pre_rms_1

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
hc_pre_rms_1(__gm__ float *v1, __gm__ float *v2, int64_t v3, int64_t v4, int32_t v5, int32_t v6) {
    const float v7 = 9.99999997E-7f;
    const float v8 = 6.10351563E-5f;
    const int64_t v9 = 1536;
    const int64_t v10 = 1024;
    const int64_t v11 = 512;
    const int64_t v12 = 4;
    const int64_t v13 = 32;
    const float v14 = 0.0f;
    const int64_t v15 = 8;
    const int64_t v16 = 1;
    const int64_t v17 = 49248;
    const int64_t v18 = 32864;
    const int64_t v19 = 32800;
    const int64_t v20 = 16416;
    const int64_t v21 = 16384;
    const int64_t v22 = 0;
    const int64_t v23 = 131200;
    const int64_t v24 = 114816;
    const int64_t v25 = 98432;
    const int64_t v26 = 82048;
    const int64_t v27 = 65664;
    const int64_t v28 = 49280;
    const int64_t v29 = 32832;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %inv_rms_inline14231__ssa_v0_view
    int64_t v30 = v4 * v16;
    // pto: %inv_rms_inline14231__ssa_v0_view
    int64_t v31 = v16 * v30;
    // pto: %inv_rms_inline14231__ssa_v0_view
    pto::Shape<1, 1, 1, -1, -1> v32 = pto::Shape<1, 1, 1, -1, -1>(v16, v16, v16, v4, v16);
    // pto: %inv_rms_inline14231__ssa_v0_view
    pto::Stride<-1, -1, -1, -1, -1> v33 = pto::Stride<-1, -1, -1, -1, -1>(v16 * v31, v31, v30, v16, v4);
    // pto: %inv_rms_inline14231__ssa_v0_view
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v34 =
        GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(
            v2, v32, v33
        );
    // pto: %t_inline14336__ssa_v0, %24
    int64_t v35 = (int64_t)((uint64_t)((int64_t)v5) * (uint64_t)v15);
    // pto: %sq_sum_inline14279__tile
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v36 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v15);
    // pto: %sq_sum_inline14279__tile
    uint64_t v37 = (uint64_t)v29;
    TASSIGN(v36, v37);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    TEXPANDS(v36, v14);
    for (int64_t i38 = v22; i38 < v13; i38 += v12) {
        // pto: %25
        int64_t v39 = (int64_t)((uint64_t)i38 * (uint64_t)v11);
        // pto: %27
        int64_t v40 = (int64_t)((uint64_t)v39 + (uint64_t)v11);
        // pto: %29
        int64_t v41 = (int64_t)((uint64_t)v39 + (uint64_t)v10);
        // pto: %31
        int64_t v42 = (int64_t)((uint64_t)v39 + (uint64_t)v9);
        // pto: %x_chunk_inline14282__tile
        Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v43 = Tile<
                TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %x_chunk_inline14282__tile
        uint64_t v44 = (uint64_t)v28;
        TASSIGN(v43, v44);
        // pto: %32
        int64_t v45 = v35 < v22 ? v22 : v35;
        // pto: %x_flat_inline14276__ssa_v0_pview
        pto::Shape<1, 1, 1, 8, 512> v46 = pto::Shape<1, 1, 1, 8, 512>();
        // pto: %x_flat_inline14276__ssa_v0_pview
        pto::Stride<131072, 131072, 131072, 16384, 1> v47 = pto::Stride<131072, 131072, 131072, 16384, 1>();
        // pto: %x_flat_inline14276__ssa_v0_pview, %33
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>
            v48 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(
                v1 + ((v22 + v45 * v21) + (v39 < v22 ? v22 : v39)), v46, v47
            );
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v43, v48);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %0
        Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v49 = Tile<
                TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %0
        uint64_t v50 = (uint64_t)v27;
        TASSIGN(v49, v50);
        // pto: %36
        pto::Shape<1, 1, 1, 8, 512> v51 = pto::Shape<1, 1, 1, 8, 512>();
        // pto: %36
        pto::Stride<131072, 131072, 131072, 16384, 1> v52 = pto::Stride<131072, 131072, 131072, 16384, 1>();
        // pto: %36, %35
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>
            v53 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(
                v1 + ((v22 + v45 * v21) + (v40 < v22 ? v22 : v40)), v51, v52
            );
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(v49, v53);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %1
        Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v54 = Tile<
                TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %1
        uint64_t v55 = (uint64_t)v26;
        TASSIGN(v54, v55);
        // pto: %39
        pto::Shape<1, 1, 1, 8, 512> v56 = pto::Shape<1, 1, 1, 8, 512>();
        // pto: %39
        pto::Stride<131072, 131072, 131072, 16384, 1> v57 = pto::Stride<131072, 131072, 131072, 16384, 1>();
        // pto: %39, %38
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>
            v58 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(
                v1 + ((v22 + v45 * v21) + (v41 < v22 ? v22 : v41)), v56, v57
            );
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        TLOAD(v54, v58);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %2
        Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v59 = Tile<
                TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %2
        uint64_t v60 = (uint64_t)v25;
        TASSIGN(v59, v60);
        // pto: %42
        pto::Shape<1, 1, 1, 8, 512> v61 = pto::Shape<1, 1, 1, 8, 512>();
        // pto: %42
        pto::Stride<131072, 131072, 131072, 16384, 1> v62 = pto::Stride<131072, 131072, 131072, 16384, 1>();
        // pto: %42, %41
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>
            v63 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(
                v1 + ((v22 + v45 * v21) + (v42 < v22 ? v22 : v42)), v61, v62
            );
        TLOAD(v59, v63);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %t__tile
        Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v64 = Tile<
                TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %t__tile
        uint64_t v65 = (uint64_t)v28;
        TASSIGN(v64, v65);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TMUL(v64, v43, v43);
        // pto: %tmp_tile
        Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v66 = Tile<
                TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %tmp_tile
        uint64_t v67 = (uint64_t)v24;
        TASSIGN(v66, v67);
        // pto: %3
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v68 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v16);
        // pto: %3
        uint64_t v69 = (uint64_t)v23;
        TASSIGN(v68, v69);
        pipe_barrier(PIPE_V);
        TROWSUM(v68, v64, v66);
        // pto: %4
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v70 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v15);
        // pto: %4
        uint64_t v71 = (uint64_t)v23;
        TASSIGN(v70, v71);
        // pto: %5
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v72 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v15);
        // pto: %5
        uint64_t v73 = (uint64_t)v28;
        TASSIGN(v72, v73);
        pipe_barrier(PIPE_V);
        TADD(v72, v36, v70);
        // pto: %6
        Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v74 = Tile<
                TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %6
        uint64_t v75 = (uint64_t)v27;
        TASSIGN(v74, v75);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TMUL(v74, v49, v49);
        // pto: %7
        Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v76 = Tile<
                TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %7
        uint64_t v77 = (uint64_t)v22;
        TASSIGN(v76, v77);
        // pto: %8
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v78 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v16);
        // pto: %8
        uint64_t v79 = (uint64_t)v21;
        TASSIGN(v78, v79);
        pipe_barrier(PIPE_V);
        TROWSUM(v78, v74, v76);
        // pto: %9
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v80 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v15);
        // pto: %9
        uint64_t v81 = (uint64_t)v21;
        TASSIGN(v80, v81);
        // pto: %10
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v82 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v15);
        // pto: %10
        uint64_t v83 = (uint64_t)v27;
        TASSIGN(v82, v83);
        pipe_barrier(PIPE_V);
        TADD(v82, v72, v80);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %11
        Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v84 = Tile<
                TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %11
        uint64_t v85 = (uint64_t)v26;
        TASSIGN(v84, v85);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        TMUL(v84, v54, v54);
        // pto: %12
        Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v86 = Tile<
                TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %12
        uint64_t v87 = (uint64_t)v20;
        TASSIGN(v86, v87);
        // pto: %13
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v88 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v16);
        // pto: %13
        uint64_t v89 = (uint64_t)v19;
        TASSIGN(v88, v89);
        pipe_barrier(PIPE_V);
        TROWSUM(v88, v84, v86);
        // pto: %14
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v90 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v15);
        // pto: %14
        uint64_t v91 = (uint64_t)v19;
        TASSIGN(v90, v91);
        // pto: %15
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v92 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v15);
        // pto: %15
        uint64_t v93 = (uint64_t)v26;
        TASSIGN(v92, v93);
        pipe_barrier(PIPE_V);
        TADD(v92, v82, v90);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        // pto: %16
        Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v94 = Tile<
                TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %16
        uint64_t v95 = (uint64_t)v25;
        TASSIGN(v94, v95);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        TMUL(v94, v59, v59);
        // pto: %17
        Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v96 = Tile<
                TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        // pto: %17
        uint64_t v97 = (uint64_t)v18;
        TASSIGN(v96, v97);
        // pto: %18
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v98 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v16);
        // pto: %18
        uint64_t v99 = (uint64_t)v17;
        TASSIGN(v98, v99);
        pipe_barrier(PIPE_V);
        TROWSUM(v98, v94, v96);
        // pto: %19
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v100 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v15);
        // pto: %19
        uint64_t v101 = (uint64_t)v17;
        TASSIGN(v100, v101);
        // pto: %20
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v102 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v15);
        // pto: %20
        uint64_t v103 = (uint64_t)v29;
        TASSIGN(v102, v103);
        pipe_barrier(PIPE_V);
        TADD(v102, v92, v100);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    }
    // pto: %21
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v104 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v15);
    // pto: %21
    uint64_t v105 = (uint64_t)v28;
    TASSIGN(v104, v105);
    pipe_barrier(PIPE_V);
    TMULS(v104, v36, v8);
    // pto: %22
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v106 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v15);
    // pto: %22
    uint64_t v107 = (uint64_t)v28;
    TASSIGN(v106, v107);
    pipe_barrier(PIPE_V);
    TADDS(v106, v104, v7);
    // pto: %rsqrt_tmp
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v108 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v15);
    // pto: %rsqrt_tmp
    uint64_t v109 = (uint64_t)v27;
    TASSIGN(v108, v109);
    // pto: %23
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v110 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v15);
    // pto: %23
    uint64_t v111 = (uint64_t)v26;
    TASSIGN(v110, v111);
    pipe_barrier(PIPE_V);
    TRSQRT(v110, v106, v108);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %inv_inline14237__tile
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v112 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v16);
    // pto: %inv_inline14237__tile
    uint64_t v113 = (uint64_t)v26;
    TASSIGN(v112, v113);
    // pto: %inv_rms_inline14231__ssa_v0_pview
    __gm__ float *v114 = PTOAS__GLOBAL_TENSOR_DATA(v34);
    // pto: %inv_rms_inline14231__ssa_v0_pview
    int64_t v115 = v15 * v16;
    // pto: %inv_rms_inline14231__ssa_v0_pview
    int64_t v116 = v16 * v115;
    // pto: %inv_rms_inline14231__ssa_v0_pview
    pto::Shape<1, 1, 1, 8, 1> v117 = pto::Shape<1, 1, 1, 8, 1>(v16, v16, v16, v15, v16);
    // pto: %inv_rms_inline14231__ssa_v0_pview
    pto::Stride<-1, -1, -1, -1, -1> v118 = pto::Stride<-1, -1, -1, -1, -1>(v16 * v116, v116, v115, v16, v4);
    // pto: %43, %inv_rms_inline14231__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v119 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(
            v114 + ((v22 + (v35 < v22 ? v22 : v35) * v16) + v22 * v4), v117, v118
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v119, v112);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: x_flat_inline14276__ssa_v0
    __gm__ Tensor *x_flat_inline14276__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *x_flat_inline14276__ssa_v0 =
        reinterpret_cast<__gm__ float *>(x_flat_inline14276__ssa_v0_tensor->buffer.addr) +
        x_flat_inline14276__ssa_v0_tensor->start_offset;

    // Unpack tensor: inv_rms_inline14231__ssa_v0
    __gm__ Tensor *inv_rms_inline14231__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *inv_rms_inline14231__ssa_v0 =
        reinterpret_cast<__gm__ float *>(inv_rms_inline14231__ssa_v0_tensor->buffer.addr) +
        inv_rms_inline14231__ssa_v0_tensor->start_offset;

    // Extract dynamic dim: t_dim_inline14263__ssa_v0
    int64_t t_dim_inline14263__ssa_v0 = static_cast<int64_t>(x_flat_inline14276__ssa_v0_tensor->shapes[0]);

    // Extract dynamic dim: t_linear_inline14281__ssa_v0
    int64_t t_linear_inline14281__ssa_v0 = static_cast<int64_t>(inv_rms_inline14231__ssa_v0_tensor->shapes[0]);

    // Forward to ptoas-generated function
    hc_pre_rms_1(
        x_flat_inline14276__ssa_v0, inv_rms_inline14231__ssa_v0, t_dim_inline14263__ssa_v0,
        t_linear_inline14281__ssa_v0, __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
