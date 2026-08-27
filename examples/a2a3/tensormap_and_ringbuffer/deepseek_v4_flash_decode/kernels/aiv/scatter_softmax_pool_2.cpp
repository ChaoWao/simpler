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
// Kernel Function: scatter_softmax_pool_2

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

static __aicore__ void scatter_softmax_pool_2(
    __gm__ float *v1, __gm__ float *v2, __gm__ int32_t *v3, __gm__ int64_t *v4, __gm__ float *v5, __gm__ float *v6,
    __gm__ float *v7, __gm__ int32_t *v8, int64_t v9
) {
    const int64_t v10 = 1536;
    const int64_t v11 = 2048;
    const float v12 = 0.0f;
    const float v13 = -3.40282347E+38f;
    const int64_t v14 = 3;
    const int64_t v15 = 1024;
    const int64_t v16 = 2;
    const int64_t v17 = 4;
    const int64_t v18 = 512;
    const int64_t v19 = 1;
    const int64_t v20 = 18432;
    const int64_t v21 = 16384;
    const int64_t v22 = 12288;
    const int64_t v23 = 8192;
    const int64_t v24 = 4096;
    const int64_t v25 = 0;
    const int64_t v26 = 24576;
    const int64_t v27 = 20480;
    const int64_t v28 = -4;
    const int64_t v29 = -8;
    const int64_t v30 = -1;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %front_kv_inline1862_inline12642__phi_v2
    Tile<
        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v31 = Tile<
            TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v19, v18);
    // pto: %front_kv_inline1862_inline12642__phi_v2
    uint64_t v32 = (uint64_t)v21;
    TASSIGN(v31, v32);
    // pto: %front_score_inline1911_inline12604__phi_v2
    Tile<
        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v33 = Tile<
            TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v19, v18);
    // pto: %front_score_inline1911_inline12604__phi_v2
    uint64_t v34 = (uint64_t)v22;
    TASSIGN(v33, v34);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    for (int64_t i35 = v25; i35 < v17; i35 += v19) {
        // pto: %flat_offset_mul
        int64_t v36 = (int64_t)((uint64_t)i35 * (uint64_t)v16);
        // pto: %token_pos_inline1853_inline12386__tile
        int32_t v37 = (v3)[v36];
        // pto: %state_row_i64_inline1850_inline12546__tile
        int64_t v38 = (v4)[v36];
        // pto: %23, %24
        int64_t v39 = (int64_t)v37 % v17;
        // pto: %28
        int64_t v40 = (int64_t)((uint64_t)v36 + (uint64_t)v19);
        // pto: %26
        int32_t v41 = (v3)[v40];
        // pto: %29
        int64_t v42 = (v4)[v40];
        // pto: %34, %35
        int64_t v43 = (int64_t)v41 % v17;
        // pto: %38
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        if (v38 >= v25) {
            // pto: %kv_tile_inline1869_inline12657__tile
            Tile<
                TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v44 = Tile<
                    TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v15);
            // pto: %kv_tile_inline1869_inline12657__tile
            uint64_t v45 = (uint64_t)v27;
            TASSIGN(v44, v45);
            // pto: %40
            int64_t v46 = v36 < v25 ? v25 : v36;
            // pto: %cmp4_kv_proj_pad_inline1900_inline12391__ssa_v1_pview
            pto::Shape<1, 1, 1, 1, 1024> v47 = pto::Shape<1, 1, 1, 1, 1024>();
            // pto: %cmp4_kv_proj_pad_inline1900_inline12391__ssa_v1_pview
            pto::Stride<1024, 1024, 1024, 1024, 1> v48 = pto::Stride<1024, 1024, 1024, 1024, 1>();
            // pto: %cmp4_kv_proj_pad_inline1900_inline12391__ssa_v1_pview
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>
                v49 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(
                    v5 + (v25 + v46 * v15), v47, v48
                );
            TLOAD(v44, v49);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            // pto: %score_tile_inline1897_inline12617__tile
            Tile<
                TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v50 = Tile<
                    TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v15);
            // pto: %score_tile_inline1897_inline12617__tile
            uint64_t v51 = (uint64_t)v26;
            TASSIGN(v50, v51);
            // pto: %cmp4_score_proj_pad_inline1878_inline12521__ssa_v1_pview
            pto::Shape<1, 1, 1, 1, 1024> v52 = pto::Shape<1, 1, 1, 1, 1024>();
            // pto: %cmp4_score_proj_pad_inline1878_inline12521__ssa_v1_pview
            pto::Stride<1024, 1024, 1024, 1024, 1> v53 = pto::Stride<1024, 1024, 1024, 1024, 1>();
            // pto: %cmp4_score_proj_pad_inline1878_inline12521__ssa_v1_pview
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>
                v54 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(
                    v6 + (v25 + v46 * v15), v52, v53
                );
            TLOAD(v50, v54);
            // pto: %ape_tile_inline1907_inline12641__tile
            Tile<
                TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v55 = Tile<
                    TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v15);
            // pto: %ape_tile_inline1907_inline12641__tile
            uint64_t v56 = (uint64_t)v25;
            TASSIGN(v55, v56);
            // pto: %csa_cmp_ape_last_inline679__ssa_v0_pview
            pto::Shape<1, 1, 1, 1, 1024> v57 = pto::Shape<1, 1, 1, 1, 1024>();
            // pto: %csa_cmp_ape_last_inline679__ssa_v0_pview
            pto::Stride<1024, 1024, 1024, 1024, 1> v58 = pto::Stride<1024, 1024, 1024, 1024, 1>();
            // pto: %42, %csa_cmp_ape_last_inline679__ssa_v0_pview
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>
                v59 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(
                    v7 + (v25 + (v39 < v25 ? v25 : v39) * v15), v57, v58
                );
            TLOAD(v55, v59);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            // pto: %score_tile_v1_inline1902_inline12518__tile
            Tile<
                TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v60 = Tile<
                    TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v15);
            // pto: %score_tile_v1_inline1902_inline12518__tile
            uint64_t v61 = (uint64_t)v26;
            TASSIGN(v60, v61);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TADD(v60, v50, v55);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            // pto: %43
            int64_t v62 = v38 < v25 ? v25 : v38;
            // pto: %compress_state_flat_inline1881_inline12478__iter_v1_pview
            pto::Shape<1, 1, 1, 1, 1024> v63 = pto::Shape<1, 1, 1, 1, 1024>();
            // pto: %compress_state_flat_inline1881_inline12478__iter_v1_pview
            pto::Stride<2048, 2048, 2048, 2048, 1> v64 = pto::Stride<2048, 2048, 2048, 2048, 1>();
            // pto: %compress_state_flat_inline1881_inline12478__iter_v1_pview
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>
                v65 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>(
                    v1 + (v25 + v62 * v11), v63, v64
                );
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_MTE3);
            TSTORE(v65, v44);
            // pto: %compress_state_flat_inline1881_inline12478__tile_pview
            pto::Shape<1, 1, 1, 1, 1024> v66 = pto::Shape<1, 1, 1, 1, 1024>();
            // pto: %compress_state_flat_inline1881_inline12478__tile_pview
            pto::Stride<2048, 2048, 2048, 2048, 1> v67 = pto::Stride<2048, 2048, 2048, 2048, 1>();
            // pto: %compress_state_flat_inline1881_inline12478__tile_pview
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>
                v68 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>(
                    v1 + (v15 + v62 * v11), v66, v67
                );
            pipe_barrier(PIPE_MTE3);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(v68, v60);
        }
        // pto: %47
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        if (v42 >= v25) {
            // pto: %0
            Tile<
                TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v69 = Tile<
                    TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v15);
            // pto: %0
            uint64_t v70 = (uint64_t)v24;
            TASSIGN(v69, v70);
            // pto: %49
            int64_t v71 = v40 < v25 ? v25 : v40;
            // pto: %50
            pto::Shape<1, 1, 1, 1, 1024> v72 = pto::Shape<1, 1, 1, 1, 1024>();
            // pto: %50
            pto::Stride<1024, 1024, 1024, 1024, 1> v73 = pto::Stride<1024, 1024, 1024, 1024, 1>();
            // pto: %50
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>
                v74 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(
                    v5 + (v25 + v71 * v15), v72, v73
                );
            TLOAD(v69, v74);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
            // pto: %1
            Tile<
                TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v75 = Tile<
                    TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v15);
            // pto: %1
            uint64_t v76 = (uint64_t)v23;
            TASSIGN(v75, v76);
            // pto: %52
            pto::Shape<1, 1, 1, 1, 1024> v77 = pto::Shape<1, 1, 1, 1, 1024>();
            // pto: %52
            pto::Stride<1024, 1024, 1024, 1024, 1> v78 = pto::Stride<1024, 1024, 1024, 1024, 1>();
            // pto: %52
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>
                v79 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(
                    v6 + (v25 + v71 * v15), v77, v78
                );
            TLOAD(v75, v79);
            // pto: %2
            Tile<
                TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v80 = Tile<
                    TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v15);
            // pto: %2
            uint64_t v81 = (uint64_t)v22;
            TASSIGN(v80, v81);
            // pto: %54
            pto::Shape<1, 1, 1, 1, 1024> v82 = pto::Shape<1, 1, 1, 1, 1024>();
            // pto: %54
            pto::Stride<1024, 1024, 1024, 1024, 1> v83 = pto::Stride<1024, 1024, 1024, 1024, 1>();
            // pto: %53, %54
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>
                v84 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(
                    v7 + (v25 + (v43 < v25 ? v25 : v43) * v15), v82, v83
                );
            TLOAD(v80, v84);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            // pto: %3
            Tile<
                TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v85 = Tile<
                    TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v15);
            // pto: %3
            uint64_t v86 = (uint64_t)v23;
            TASSIGN(v85, v86);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            TADD(v85, v75, v80);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            // pto: %56
            int64_t v87 = v42 < v25 ? v25 : v42;
            // pto: %compress_state_flat_inline1881_inline12478__phi_v7_pview
            pto::Shape<1, 1, 1, 1, 1024> v88 = pto::Shape<1, 1, 1, 1, 1024>();
            // pto: %compress_state_flat_inline1881_inline12478__phi_v7_pview
            pto::Stride<2048, 2048, 2048, 2048, 1> v89 = pto::Stride<2048, 2048, 2048, 2048, 1>();
            // pto: %compress_state_flat_inline1881_inline12478__phi_v7_pview
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>
                v90 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>(
                    v1 + (v25 + v87 * v11), v88, v89
                );
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
            pipe_barrier(PIPE_MTE3);
            TSTORE(v90, v69);
            // pto: %59
            pto::Shape<1, 1, 1, 1, 1024> v91 = pto::Shape<1, 1, 1, 1, 1024>();
            // pto: %59
            pto::Stride<2048, 2048, 2048, 2048, 1> v92 = pto::Stride<2048, 2048, 2048, 2048, 1>();
            // pto: %59
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>
                v93 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>(
                    v1 + (v15 + v87 * v11), v91, v92
                );
            pipe_barrier(PIPE_MTE3);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            TSTORE(v93, v85);
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        // pto: %first_pos_b_inline1905_inline12623__tile
        int32_t v94 = (v3)[v36];
        // pto: %62
        int64_t v95 = (int64_t)v94;
        // pto: %63
        int64_t v96 = v95 % v17;
        // pto: %66, %64
        int64_t v97 = (int64_t)((uint64_t)v95 + (uint64_t)((int64_t)((uint64_t)v17 - (uint64_t)v96)));
        // pto: %71
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        if (v96 >= v16) {
            int64_t v98 = (int64_t)((uint64_t)v97 + (uint64_t)v30);
            // pto: %76
            int64_t v99 = (int64_t)((uint64_t)i35 * (uint64_t)v24);
            // pto: %77, %73, %75
            int32_t v100 = (v8)[(int64_t)((uint64_t)v99 + (uint64_t)(v98 / v17))];
            // pto: %78, %79, %80, %74
            int64_t v101 =
                (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)v100) * (uint64_t)v17)) + (uint64_t)(v98 % v17));
            // pto: %mi_inline1891_inline12637__tile
            Tile<
                TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v102 = Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v18);
            // pto: %mi_inline1891_inline12637__tile
            uint64_t v103 = (uint64_t)v27;
            TASSIGN(v102, v103);
            // pto: %81
            int64_t v104 = v101 < v25 ? v25 : v101;
            // pto: %82
            pto::Shape<1, 1, 1, 1, 512> v105 = pto::Shape<1, 1, 1, 1, 512>();
            // pto: %82
            pto::Stride<2048, 2048, 2048, 2048, 1> v106 = pto::Stride<2048, 2048, 2048, 2048, 1>();
            // pto: %82
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>
                v107 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>(
                    v1 + (v10 + v104 * v11), v105, v106
                );
            TLOAD(v102, v107);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
            // pto: %t__tile
            Tile<
                TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v108 = Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v18);
            // pto: %t__tile
            uint64_t v109 = (uint64_t)v26;
            TASSIGN(v108, v109);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
            TSUB(v108, v102, v102);
            // pto: %li_inline1910_inline12638__tile
            Tile<
                TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v110 = Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v18);
            // pto: %li_inline1910_inline12638__tile
            uint64_t v111 = (uint64_t)v26;
            TASSIGN(v110, v111);
            pipe_barrier(PIPE_V);
            TEXP(v110, v108);
            // pto: %oi_inline1899_inline12477__tile
            Tile<
                TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v112 = Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v18);
            // pto: %oi_inline1899_inline12477__tile
            uint64_t v113 = (uint64_t)v25;
            TASSIGN(v112, v113);
            // pto: %84
            pto::Shape<1, 1, 1, 1, 512> v114 = pto::Shape<1, 1, 1, 1, 512>();
            // pto: %84
            pto::Stride<2048, 2048, 2048, 2048, 1> v115 = pto::Stride<2048, 2048, 2048, 2048, 1>();
            // pto: %84
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>
                v116 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>(
                    v1 + (v18 + v104 * v11), v114, v115
                );
            TLOAD(v112, v116);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
            for (int64_t j117 = v25; j117 < v17; j117 += v19) {
                // pto: %85
                int64_t v118 = (int64_t)((uint64_t)((int64_t)((uint64_t)v97 + (uint64_t)v29)) + (uint64_t)j117);
                // pto: %front_score_inline1911_inline12604__tile
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v119 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %front_score_inline1911_inline12604__tile
                uint64_t v120 = (uint64_t)v24;
                TASSIGN(v119, v120);
                pipe_barrier(PIPE_V);
                TEXPANDS(v119, v13);
                // pto: %front_kv_inline1862_inline12642__tile
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v121 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %front_kv_inline1862_inline12642__tile
                uint64_t v122 = (uint64_t)v23;
                TASSIGN(v121, v122);
                TEXPANDS(v121, v12);
                // pto: %87
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                if (v95 >= v17) {
                    // pto: %92, %88, %90
                    int32_t v123 = (v8)[(int64_t)((uint64_t)v99 + (uint64_t)(v118 / v17))];
                    // pto: %93, %94, %95, %89
                    int64_t v124 = (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)v123) * (uint64_t)v17)) +
                                             (uint64_t)(v118 % v17));
                    // pto: %4
                    Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>
                        v125 = Tile<
                            TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                            PadValue::Null, CompactMode::Null>(v19, v18);
                    // pto: %4
                    uint64_t v126 = (uint64_t)v22;
                    TASSIGN(v125, v126);
                    // pto: %96
                    int64_t v127 = v124 < v25 ? v25 : v124;
                    // pto: %97
                    pto::Shape<1, 1, 1, 1, 512> v128 = pto::Shape<1, 1, 1, 1, 512>();
                    // pto: %97
                    pto::Stride<2048, 2048, 2048, 2048, 1> v129 = pto::Stride<2048, 2048, 2048, 2048, 1>();
                    // pto: %97
                    GlobalTensor<
                        float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>
                        v130 = GlobalTensor<
                            float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<2048, 2048, 2048, 2048, 1>,
                            pto::Layout::ND>(v1 + (v15 + v127 * v11), v128, v129);
                    TLOAD(v125, v130);
                    // pto: %5
                    Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>
                        v131 = Tile<
                            TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                            PadValue::Null, CompactMode::Null>(v19, v18);
                    // pto: %5
                    uint64_t v132 = (uint64_t)v21;
                    TASSIGN(v131, v132);
                    // pto: %99
                    pto::Shape<1, 1, 1, 1, 512> v133 = pto::Shape<1, 1, 1, 1, 512>();
                    // pto: %99
                    pto::Stride<2048, 2048, 2048, 2048, 1> v134 = pto::Stride<2048, 2048, 2048, 2048, 1>();
                    // pto: %99
                    GlobalTensor<
                        float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>
                        v135 = GlobalTensor<
                            float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<2048, 2048, 2048, 2048, 1>,
                            pto::Layout::ND>(v1 + (v25 + v127 * v11), v133, v134);
                    TLOAD(v131, v135);
                } else {
                    // pto: %front_kv_inline1862_inline12642__tile_mv
                    Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>
                        v136 = Tile<
                            TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                            PadValue::Null, CompactMode::Null>(v19, v18);
                    // pto: %front_kv_inline1862_inline12642__tile_mv
                    uint64_t v137 = (uint64_t)v21;
                    TASSIGN(v136, v137);
                    pipe_barrier(PIPE_V);
                    TMOV(v136, v121);
                    // pto: %front_score_inline1911_inline12604__tile_mv
                    Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>
                        v138 = Tile<
                            TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                            PadValue::Null, CompactMode::Null>(v19, v18);
                    // pto: %front_score_inline1911_inline12604__tile_mv
                    uint64_t v139 = (uint64_t)v22;
                    TASSIGN(v138, v139);
                    TMOV(v138, v119);
                }
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
                // pto: %mi_next_front_inline1880_inline12403__tile
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v140 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %mi_next_front_inline1880_inline12403__tile
                uint64_t v141 = (uint64_t)v24;
                TASSIGN(v140, v141);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
                pipe_barrier(PIPE_V);
                TMAX(v140, v102, v33);
                // pto: %6
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v142 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %6
                uint64_t v143 = (uint64_t)v23;
                TASSIGN(v142, v143);
                pipe_barrier(PIPE_V);
                TSUB(v142, v102, v140);
                // pto: %alpha_front_inline1918_inline12695__tile
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v144 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %alpha_front_inline1918_inline12695__tile
                uint64_t v145 = (uint64_t)v23;
                TASSIGN(v144, v145);
                pipe_barrier(PIPE_V);
                TEXP(v144, v142);
                // pto: %7
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v146 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %7
                uint64_t v147 = (uint64_t)v22;
                TASSIGN(v146, v147);
                TSUB(v146, v33, v140);
                // pto: %beta_front_inline1885_inline12646__tile
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v148 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %beta_front_inline1885_inline12646__tile
                uint64_t v149 = (uint64_t)v22;
                TASSIGN(v148, v149);
                pipe_barrier(PIPE_V);
                TEXP(v148, v146);
                // pto: %8
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v150 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %8
                uint64_t v151 = (uint64_t)v20;
                TASSIGN(v150, v151);
                TMUL(v150, v144, v110);
                // pto: %9
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v152 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %9
                uint64_t v153 = (uint64_t)v26;
                TASSIGN(v152, v153);
                pipe_barrier(PIPE_V);
                TADD(v152, v150, v148);
                // pto: %10
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v154 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %10
                uint64_t v155 = (uint64_t)v23;
                TASSIGN(v154, v155);
                TMUL(v154, v112, v144);
                // pto: %11
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v156 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %11
                uint64_t v157 = (uint64_t)v22;
                TASSIGN(v156, v157);
                pipe_barrier(PIPE_V);
                TMUL(v156, v31, v148);
                // pto: %12
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v158 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %12
                uint64_t v159 = (uint64_t)v25;
                TASSIGN(v158, v159);
                pipe_barrier(PIPE_V);
                TADD(v158, v154, v156);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                // pto: %mi_inline1891_inline12637__ssa_v3
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v160 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %mi_inline1891_inline12637__ssa_v3
                uint64_t v161 = (uint64_t)v24;
                TASSIGN(v160, v161);
                // pto: %mi_inline1891_inline12637__ssa_v3_mv
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v162 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %mi_inline1891_inline12637__ssa_v3_mv
                uint64_t v163 = (uint64_t)v27;
                TASSIGN(v162, v163);
                TMOV(v162, v160);
            }
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            for (int64_t j164 = v25; j164 < v14; j164 += v19) {
                // pto: %100
                int64_t v165 = (int64_t)((uint64_t)((int64_t)((uint64_t)v97 + (uint64_t)v28)) + (uint64_t)j164);
                // pto: %105, %101, %103
                int32_t v166 = (v8)[(int64_t)((uint64_t)v99 + (uint64_t)(v165 / v17))];
                // pto: %106, %107, %108, %102
                int64_t v167 = (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)v166) * (uint64_t)v17)) +
                                         (uint64_t)(v165 % v17));
                // pto: %back_score_inline1861_inline12655__tile
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v168 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %back_score_inline1861_inline12655__tile
                uint64_t v169 = (uint64_t)v24;
                TASSIGN(v168, v169);
                // pto: %109
                int64_t v170 = v167 < v25 ? v25 : v167;
                // pto: %110
                pto::Shape<1, 1, 1, 1, 512> v171 = pto::Shape<1, 1, 1, 1, 512>();
                // pto: %110
                pto::Stride<2048, 2048, 2048, 2048, 1> v172 = pto::Stride<2048, 2048, 2048, 2048, 1>();
                // pto: %110
                GlobalTensor<
                    float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>
                    v173 = GlobalTensor<
                        float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>(
                        v1 + (v10 + v170 * v11), v171, v172
                    );
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
                TLOAD(v168, v173);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
                // pto: %back_kv_inline1838_inline12656__tile
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v174 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %back_kv_inline1838_inline12656__tile
                uint64_t v175 = (uint64_t)v23;
                TASSIGN(v174, v175);
                // pto: %112
                pto::Shape<1, 1, 1, 1, 512> v176 = pto::Shape<1, 1, 1, 1, 512>();
                // pto: %112
                pto::Stride<2048, 2048, 2048, 2048, 1> v177 = pto::Stride<2048, 2048, 2048, 2048, 1>();
                // pto: %112
                GlobalTensor<
                    float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>
                    v178 = GlobalTensor<
                        float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>(
                        v1 + (v18 + v170 * v11), v176, v177
                    );
                TLOAD(v174, v178);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
                // pto: %mi_next_back_inline1837_inline12658__tile
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v179 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %mi_next_back_inline1837_inline12658__tile
                uint64_t v180 = (uint64_t)v22;
                TASSIGN(v179, v180);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
                pipe_barrier(PIPE_V);
                TMAX(v179, v102, v168);
                // pto: %13
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v181 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %13
                uint64_t v182 = (uint64_t)v21;
                TASSIGN(v181, v182);
                pipe_barrier(PIPE_V);
                TSUB(v181, v102, v179);
                // pto: %alpha_back_inline1864_inline12568__tile
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v183 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %alpha_back_inline1864_inline12568__tile
                uint64_t v184 = (uint64_t)v21;
                TASSIGN(v183, v184);
                pipe_barrier(PIPE_V);
                TEXP(v183, v181);
                // pto: %14
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v185 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %14
                uint64_t v186 = (uint64_t)v24;
                TASSIGN(v185, v186);
                TSUB(v185, v168, v179);
                // pto: %beta_back_inline1883_inline12662__tile
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v187 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %beta_back_inline1883_inline12662__tile
                uint64_t v188 = (uint64_t)v24;
                TASSIGN(v187, v188);
                pipe_barrier(PIPE_V);
                TEXP(v187, v185);
                // pto: %15
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v189 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %15
                uint64_t v190 = (uint64_t)v20;
                TASSIGN(v189, v190);
                TMUL(v189, v183, v110);
                // pto: %16
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v191 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %16
                uint64_t v192 = (uint64_t)v26;
                TASSIGN(v191, v192);
                pipe_barrier(PIPE_V);
                TADD(v191, v189, v187);
                // pto: %17
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v193 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %17
                uint64_t v194 = (uint64_t)v21;
                TASSIGN(v193, v194);
                TMUL(v193, v112, v183);
                // pto: %18
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v195 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %18
                uint64_t v196 = (uint64_t)v24;
                TASSIGN(v195, v196);
                pipe_barrier(PIPE_V);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
                TMUL(v195, v174, v187);
                // pto: %19
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v197 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %19
                uint64_t v198 = (uint64_t)v25;
                TASSIGN(v197, v198);
                pipe_barrier(PIPE_V);
                TADD(v197, v193, v195);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
                // pto: %mi_inline1891_inline12637__ssa_v6
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v199 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %mi_inline1891_inline12637__ssa_v6
                uint64_t v200 = (uint64_t)v22;
                TASSIGN(v199, v200);
                // pto: %mi_inline1891_inline12637__ssa_v6_mv
                Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v201 = Tile<
                        TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                        CompactMode::Null>(v19, v18);
                // pto: %mi_inline1891_inline12637__ssa_v6_mv
                uint64_t v202 = (uint64_t)v27;
                TASSIGN(v201, v202);
                TMOV(v201, v199);
            }
            // pto: %pooled_chunk_inline1836_inline12667__tile
            Tile<
                TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v203 = Tile<
                    TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v18);
            // pto: %pooled_chunk_inline1836_inline12667__tile
            uint64_t v204 = (uint64_t)v27;
            TASSIGN(v203, v204);
            pipe_barrier(PIPE_V);
            TDIV(v203, v112, v110);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
            // pto: %pooled_kv_inline1855_inline12392__iter_v1_pview
            pto::Shape<1, 1, 1, 1, 512> v205 = pto::Shape<1, 1, 1, 1, 512>();
            // pto: %pooled_kv_inline1855_inline12392__iter_v1_pview
            pto::Stride<512, 512, 512, 512, 1> v206 = pto::Stride<512, 512, 512, 512, 1>();
            // pto: %113, %pooled_kv_inline1855_inline12392__iter_v1_pview
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v207 =
                GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(
                    v2 + (v25 + (i35 < v25 ? v25 : i35) * v18), v205, v206
                );
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
            TSTORE(v207, v203);
        }
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Unpack tensor: compress_state_flat_inline1881_inline12478__ssa_v0
    __gm__ Tensor *compress_state_flat_inline1881_inline12478__ssa_v0_tensor =
        reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *compress_state_flat_inline1881_inline12478__ssa_v0 =
        reinterpret_cast<__gm__ float *>(compress_state_flat_inline1881_inline12478__ssa_v0_tensor->buffer.addr) +
        compress_state_flat_inline1881_inline12478__ssa_v0_tensor->start_offset;

    // Unpack tensor: pooled_kv_inline1855_inline12392__ssa_v0
    __gm__ Tensor *pooled_kv_inline1855_inline12392__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *pooled_kv_inline1855_inline12392__ssa_v0 =
        reinterpret_cast<__gm__ float *>(pooled_kv_inline1855_inline12392__ssa_v0_tensor->buffer.addr) +
        pooled_kv_inline1855_inline12392__ssa_v0_tensor->start_offset;

    // Unpack tensor: position_ids_bsd_inline12252__ssa_v0
    __gm__ Tensor *position_ids_bsd_inline12252__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ int32_t *position_ids_bsd_inline12252__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(position_ids_bsd_inline12252__ssa_v0_tensor->buffer.addr) +
        position_ids_bsd_inline12252__ssa_v0_tensor->start_offset;

    // Unpack tensor: state_slot_mapping_bsd_inline12635__ssa_v0
    __gm__ Tensor *state_slot_mapping_bsd_inline12635__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ int64_t *state_slot_mapping_bsd_inline12635__ssa_v0 =
        reinterpret_cast<__gm__ int64_t *>(state_slot_mapping_bsd_inline12635__ssa_v0_tensor->buffer.addr) +
        state_slot_mapping_bsd_inline12635__ssa_v0_tensor->start_offset;

    // Unpack tensor: cmp4_kv_proj_pad_inline1900_inline12391__ssa_v1
    __gm__ Tensor *cmp4_kv_proj_pad_inline1900_inline12391__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ float *cmp4_kv_proj_pad_inline1900_inline12391__ssa_v1 =
        reinterpret_cast<__gm__ float *>(cmp4_kv_proj_pad_inline1900_inline12391__ssa_v1_tensor->buffer.addr) +
        cmp4_kv_proj_pad_inline1900_inline12391__ssa_v1_tensor->start_offset;

    // Unpack tensor: cmp4_score_proj_pad_inline1878_inline12521__ssa_v1
    __gm__ Tensor *cmp4_score_proj_pad_inline1878_inline12521__ssa_v1_tensor =
        reinterpret_cast<__gm__ Tensor *>(args[5]);
    __gm__ float *cmp4_score_proj_pad_inline1878_inline12521__ssa_v1 =
        reinterpret_cast<__gm__ float *>(cmp4_score_proj_pad_inline1878_inline12521__ssa_v1_tensor->buffer.addr) +
        cmp4_score_proj_pad_inline1878_inline12521__ssa_v1_tensor->start_offset;

    // Unpack tensor: csa_cmp_ape_last_inline679__ssa_v0
    __gm__ Tensor *csa_cmp_ape_last_inline679__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[6]);
    __gm__ float *csa_cmp_ape_last_inline679__ssa_v0 =
        reinterpret_cast<__gm__ float *>(csa_cmp_ape_last_inline679__ssa_v0_tensor->buffer.addr) +
        csa_cmp_ape_last_inline679__ssa_v0_tensor->start_offset;

    // Unpack tensor: csa_compress_state_block_table__ssa_v0
    __gm__ Tensor *csa_compress_state_block_table__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[7]);
    __gm__ int32_t *csa_compress_state_block_table__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(csa_compress_state_block_table__ssa_v0_tensor->buffer.addr) +
        csa_compress_state_block_table__ssa_v0_tensor->start_offset;

    // Extract dynamic dim: compress_state_block_num_inline1857_inline12519__ssa_v0
    int64_t compress_state_block_num_inline1857_inline12519__ssa_v0 =
        (static_cast<int64_t>(compress_state_flat_inline1881_inline12478__ssa_v0_tensor->shapes[0]) / 4);

    // Forward to ptoas-generated function
    scatter_softmax_pool_2(
        compress_state_flat_inline1881_inline12478__ssa_v0, pooled_kv_inline1855_inline12392__ssa_v0,
        position_ids_bsd_inline12252__ssa_v0, state_slot_mapping_bsd_inline12635__ssa_v0,
        cmp4_kv_proj_pad_inline1900_inline12391__ssa_v1, cmp4_score_proj_pad_inline1878_inline12521__ssa_v1,
        csa_cmp_ape_last_inline679__ssa_v0, csa_compress_state_block_table__ssa_v0,
        compress_state_block_num_inline1857_inline12519__ssa_v0
    );
}
