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
// Kernel Function: gate_1_aiv

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

#if !defined(__CPU_SIM)
#include "intrinsic.h"

// A2A3 mixed tasks run the same AIV kernel on two vector cores.
// Bridge the runtime-provided lane id into PTO-ISA get_subblockid().
[[block_local]] static int32_t pypto_runtime_subblock_id;
#define get_subblockid() pypto_runtime_subblock_id
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

static __aicore__ void gate_1_aic(
    __gm__ float *v1, __gm__ float *v2, __gm__ float *v3, __gm__ float *v4, __gm__ float *v5, __gm__ float *v6,
    __gm__ float *v7, int64_t v8, int32_t v9, int32_t v10, int32_t v11
) {
    const int64_t v12 = 4096;
    const int64_t v13 = 512;
    const int64_t v14 = 1024;
    const int64_t v15 = 2048;
    const int64_t v16 = 16;
    const int64_t v17 = 32768;
    const int64_t v18 = 131072;
    const int64_t v19 = 393216;
    const int64_t v20 = 262144;
    const int64_t v21 = 0;
    const int32_t v22 = 0;
    using T = float;

#if defined(__DAV_CUBE__)
    // pto: %gate_logits_tile_inline2592_inline11143__tile_l0_c_phi
    Tile<
        TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v23 = Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v16, v16);
    // pto: %gate_logits_tile_inline2592_inline11143__tile_l0_c_phi
    uint64_t v24 = (uint64_t)v21;
    TASSIGN(v23, v24);
    auto v25 = TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>(v7, v22, v22);
    // pto: %gb_idx_inline2614_inline11134__ssa_v0
    int64_t v26 = (int64_t)v10;
    // pto: %12, %14
    int64_t v27 = (int64_t)((uint64_t)(v26 / v8) * (uint64_t)v16);
    // pto: %13, %15
    int64_t v28 = (int64_t)((uint64_t)(v26 % v8) * (uint64_t)v16);
    // pto: %gate_logits_tile_inline2592_inline11143__tile
    Tile<
        TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v29 = Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v16, v16);
    // pto: %gate_logits_tile_inline2592_inline11143__tile
    uint64_t v30 = (uint64_t)v21;
    TASSIGN(v29, v30);
    // pto: %gd_x_inline2620_inline11017__tile
    Tile<
        TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v31 = Tile<
            TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v16, v15);
    // pto: %gd_x_inline2620_inline11017__tile
    uint64_t v32 = (uint64_t)v20;
    TASSIGN(v31, v32);
    // pto: %16
    int64_t v33 = v27 < v21 ? v21 : v27;
    // pto: %xg_buf_inline2582_inline11203__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 2048> v34 = pto::Shape<1, 1, 1, 16, 2048>();
    // pto: %xg_buf_inline2582_inline11203__ssa_v0_pview
    pto::Stride<65536, 65536, 65536, 4096, 1> v35 = pto::Stride<65536, 65536, 65536, 4096, 1>();
    // pto: %xg_buf_inline2582_inline11203__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v36 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(
            v2 + (v21 + v33 * v12), v34, v35
        );
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    TLOAD(v31, v36);
    // pto: %gd_w_inline2571_inline11010__tile
    Tile<
        TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v37 = Tile<
            TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v16, v15);
    // pto: %gd_w_inline2571_inline11010__tile
    uint64_t v38 = (uint64_t)v19;
    TASSIGN(v37, v38);
    // pto: %17
    int64_t v39 = v28 < v21 ? v21 : v28;
    // pto: %gate_w_csa_inline540__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 2048> v40 = pto::Shape<1, 1, 1, 16, 2048>();
    // pto: %gate_w_csa_inline540__ssa_v0_pview
    pto::Stride<65536, 65536, 65536, 4096, 1> v41 = pto::Stride<65536, 65536, 65536, 4096, 1>();
    // pto: %gate_w_csa_inline540__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v42 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(
            v3 + (v21 + v39 * v12), v40, v41
        );
    TLOAD(v37, v42);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // pto: %0
    Tile<
        TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v43 = Tile<
            TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v16, v15);
    // pto: %0
    uint64_t v44 = (uint64_t)v21;
    TASSIGN(v43, v44);
    // pto: %19
    pto::Shape<1, 1, 1, 16, 2048> v45 = pto::Shape<1, 1, 1, 16, 2048>();
    // pto: %19
    pto::Stride<65536, 65536, 65536, 4096, 1> v46 = pto::Stride<65536, 65536, 65536, 4096, 1>();
    // pto: %19
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v47 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(
            v2 + (v15 + v33 * v12), v45, v46
        );
    TLOAD(v43, v47);
    // pto: %1
    Tile<
        TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v48 = Tile<
            TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v16, v15);
    // pto: %1
    uint64_t v49 = (uint64_t)v18;
    TASSIGN(v48, v49);
    // pto: %21
    pto::Shape<1, 1, 1, 16, 2048> v50 = pto::Shape<1, 1, 1, 16, 2048>();
    // pto: %21
    pto::Stride<65536, 65536, 65536, 4096, 1> v51 = pto::Stride<65536, 65536, 65536, 4096, 1>();
    // pto: %21
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v52 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(
            v3 + (v15 + v39 * v12), v50, v51
        );
    TLOAD(v48, v52);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    // pto: %gd_w_inline2571_inline11010__tile_t
    Tile<
        TileType::Mat, float, 2048, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v53 = Tile<
            TileType::Mat, float, 2048, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v15, v16);
    // pto: %gd_w_inline2571_inline11010__tile_t
    uint64_t v54 = (uint64_t)v19;
    TASSIGN(v53, v54);
    // pto: %gate_logits_tile_inline2592_inline11143__tile_l0_init
    Tile<
        TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v55 = Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v16, v16);
    // pto: %gate_logits_tile_inline2592_inline11143__tile_l0_init
    uint64_t v56 = (uint64_t)v21;
    TASSIGN(v55, v56);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    for (int64_t i57 = v21; i57 < v15; i57 += v14) {
        // pto: %gate_logits_tile_inline2592_inline11143__tile_l0_a
        Tile<
            TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v58 = Tile<
                TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v16, v13);
        // pto: %gate_logits_tile_inline2592_inline11143__tile_l0_a
        uint64_t v59 = (uint64_t)v21;
        TASSIGN(v58, v59);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        pipe_barrier(PIPE_MTE1);
        TEXTRACT(v58, v31, v21, i57);
        // pto: %gate_logits_tile_inline2592_inline11143__tile_l0_b
        Tile<
            TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v60 = Tile<
                TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v13, v16);
        // pto: %gate_logits_tile_inline2592_inline11143__tile_l0_b
        uint64_t v61 = (uint64_t)v21;
        TASSIGN(v60, v61);
        TEXTRACT(v60, v53, i57, v21);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        // pto: %2
        Tile<
            TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v62 = Tile<
                TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v16, v13);
        // pto: %2
        uint64_t v63 = (uint64_t)v17;
        TASSIGN(v62, v63);
        // pto: %22
        int64_t v64 = (int64_t)((uint64_t)i57 + (uint64_t)v13);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v62, v31, v21, v64);
        // pto: %3
        Tile<
            TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v65 = Tile<
                TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v13, v16);
        // pto: %3
        uint64_t v66 = (uint64_t)v17;
        TASSIGN(v65, v66);
        TEXTRACT(v65, v53, v64, v21);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        // pto: %24
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if (i57 == v21) {
            // pto: %gate_logits_tile_inline2592_inline11143__tile_l0_c_first
            Tile<
                TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v67 = Tile<
                    TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v16, v16);
            // pto: %gate_logits_tile_inline2592_inline11143__tile_l0_c_first
            uint64_t v68 = (uint64_t)v21;
            TASSIGN(v67, v68);
            pipe_barrier(PIPE_M);
            TMATMUL(v67, v58, v60);
        } else {
            // pto: %gate_logits_tile_inline2592_inline11143__tile_l0_c_acc
            Tile<
                TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v69 = Tile<
                    TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v16, v16);
            // pto: %gate_logits_tile_inline2592_inline11143__tile_l0_c_acc
            uint64_t v70 = (uint64_t)v21;
            TASSIGN(v69, v70);
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v69, v69, v58, v60);
        }
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        // pto: %4
        Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v71 = Tile<
                TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %4
        uint64_t v72 = (uint64_t)v21;
        TASSIGN(v71, v72);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        TMATMUL_ACC(v71, v71, v62, v65);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    }
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    // pto: %5
    Tile<
        TileType::Mat, float, 2048, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v73 = Tile<
            TileType::Mat, float, 2048, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v15, v16);
    // pto: %5
    uint64_t v74 = (uint64_t)v18;
    TASSIGN(v73, v74);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    for (int64_t i75 = v21; i75 < v15; i75 += v14) {
        // pto: %6
        Tile<
            TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v76 = Tile<
                TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v16, v13);
        // pto: %6
        uint64_t v77 = (uint64_t)v21;
        TASSIGN(v76, v77);
        pipe_barrier(PIPE_MTE1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        TEXTRACT(v76, v43, v21, i75);
        // pto: %7
        Tile<
            TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v78 = Tile<
                TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v13, v16);
        // pto: %7
        uint64_t v79 = (uint64_t)v21;
        TASSIGN(v78, v79);
        TEXTRACT(v78, v73, i75, v21);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        // pto: %8
        Tile<
            TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v80 = Tile<
                TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v16, v13);
        // pto: %8
        uint64_t v81 = (uint64_t)v17;
        TASSIGN(v80, v81);
        // pto: %26
        int64_t v82 = (int64_t)((uint64_t)i75 + (uint64_t)v13);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        TEXTRACT(v80, v43, v21, v82);
        // pto: %9
        Tile<
            TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v83 = Tile<
                TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v13, v16);
        // pto: %9
        uint64_t v84 = (uint64_t)v17;
        TASSIGN(v83, v84);
        TEXTRACT(v83, v73, v82, v21);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        // pto: %10
        Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v85 = Tile<
                TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %10
        uint64_t v86 = (uint64_t)v21;
        TASSIGN(v85, v86);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v85, v85, v76, v78);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        // pto: %11
        Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v87 = Tile<
                TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %11
        uint64_t v88 = (uint64_t)v21;
        TASSIGN(v87, v88);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        TMATMUL_ACC(v87, v87, v80, v83);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TPUSH<
        TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>,
        Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>,
        TileSplitAxis::TILE_NO_SPLIT>(v25, v55);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
#endif  // __DAV_CUBE__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}

static __aicore__ void gate_1_aiv(
    __gm__ float *v1, __gm__ float *v2, __gm__ float *v3, __gm__ float *v4, __gm__ float *v5, __gm__ float *v6,
    __gm__ float *v7, int64_t v8, int32_t v9, int32_t v10, int32_t v11, int32_t v12
) {
    const int64_t v13 = 256;
    const int32_t v14 = 16;
    const int64_t v15 = 3;
    const float v16 = 10.0f;
    const float v17 = 1.0f;
    const float v18 = 0.0f;
    const int64_t v19 = 0;
    const int64_t v20 = 16;
    const int64_t v21 = 1;
    const int64_t v22 = 9216;
    const int64_t v23 = 8192;
    const int64_t v24 = 10368;
    const int64_t v25 = 10304;
    const int64_t v26 = 10240;
    const int32_t v27 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    auto v28 = TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>(v7, v27, v27);
    // pto: %subblock_idx, %36
    if ((int64_t)v12 == v19) {
        // pto: %gb_idx_inline2614_inline11134__ssa_v0
        int64_t v29 = (int64_t)v10;
        // pto: %37, %39
        int64_t v30 = (int64_t)((uint64_t)(v29 / v8) * (uint64_t)v20);
        // pto: %38, %40
        int64_t v31 = (int64_t)((uint64_t)(v29 % v8) * (uint64_t)v20);
        // pto: %t__tile
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v32 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v21, v20);
        // pto: %t__tile
        uint64_t v33 = (uint64_t)v26;
        TASSIGN(v32, v33);
        // pto: %41
        int64_t v34 = v31 < v19 ? v19 : v31;
        // pto: %gate_bias_csa_inline538__ssa_v0_pview
        pto::Shape<1, 1, 1, 1, 16> v35 = pto::Shape<1, 1, 1, 1, 16>();
        // pto: %gate_bias_csa_inline538__ssa_v0_pview
        pto::Stride<16, 16, 16, 16, 1> v36 = pto::Stride<16, 16, 16, 16, 1>();
        // pto: %gate_bias_csa_inline538__ssa_v0_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v37 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(
                v1 + (v19 + v34), v35, v36
            );
        TLOAD(v32, v37);
        // pto: %gp_bias_row_inline2559_inline11008__tile
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v38 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v21, v20);
        // pto: %gp_bias_row_inline2559_inline11008__tile
        uint64_t v39 = (uint64_t)v26;
        TASSIGN(v38, v39);
        // pto: %0
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v40 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v20, v21);
        // pto: %0
        uint64_t v41 = (uint64_t)v25;
        TASSIGN(v40, v41);
        // pto: %42
        int64_t v42 = v30 < v19 ? v19 : v30;
        // pto: %inv_rms_buf_inline2585_inline11089__ssa_v0_pview
        pto::Shape<1, 1, 1, 16, 1> v43 = pto::Shape<1, 1, 1, 16, 1>();
        // pto: %inv_rms_buf_inline2585_inline11089__ssa_v0_pview
        pto::Stride<16, 16, 16, 1, 16> v44 = pto::Stride<16, 16, 16, 1, 16>();
        // pto: %inv_rms_buf_inline2585_inline11089__ssa_v0_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 16>, pto::Layout::DN> v45 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 16>, pto::Layout::DN>(
                v4 + (v19 + v42), v43, v44
            );
        TLOAD(v40, v45);
        // pto: %gate_logits_tile_inline2592_inline11143__rv_v2_Vec
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v46 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v14, v14);
        TPOP<
            TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>,
            Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>,
            TileSplitAxis::TILE_NO_SPLIT>(v28, v46);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %gate_logits_tile_v1_inline2568_inline11187__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v47 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v20, v20);
        // pto: %gate_logits_tile_v1_inline2568_inline11187__tile
        uint64_t v48 = (uint64_t)v24;
        TASSIGN(v47, v48);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TROWEXPANDMUL(v47, v46, v40);
        TFREE<TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>, TileSplitAxis::TILE_NO_SPLIT>(v28);
        // pto: %gp_relu_inline2577_inline11090__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v49 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v20, v20);
        // pto: %gp_relu_inline2577_inline11090__tile
        uint64_t v50 = (uint64_t)v23;
        TASSIGN(v49, v50);
        pipe_barrier(PIPE_V);
        TMAXS(v49, v47, v18);
        // pto: %1
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v51 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v20, v20);
        // pto: %1
        uint64_t v52 = (uint64_t)v22;
        TASSIGN(v51, v52);
        TNEG(v51, v47);
        // pto: %gp_abs_inline2545_inline11116__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v53 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v20, v20);
        // pto: %gp_abs_inline2545_inline11116__tile
        uint64_t v54 = (uint64_t)v22;
        TASSIGN(v53, v54);
        pipe_barrier(PIPE_V);
        TMAX(v53, v47, v51);
        // pto: %2
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v55 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v20, v20);
        // pto: %2
        uint64_t v56 = (uint64_t)v22;
        TASSIGN(v55, v56);
        pipe_barrier(PIPE_V);
        TNEG(v55, v53);
        // pto: %3
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v57 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v20, v20);
        // pto: %3
        uint64_t v58 = (uint64_t)v22;
        TASSIGN(v57, v58);
        pipe_barrier(PIPE_V);
        TEXP(v57, v55);
        // pto: %4
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v59 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v20, v20);
        // pto: %4
        uint64_t v60 = (uint64_t)v22;
        TASSIGN(v59, v60);
        pipe_barrier(PIPE_V);
        TADDS(v59, v57, v17);
        // pto: %5
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v61 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v20, v20);
        // pto: %5
        uint64_t v62 = (uint64_t)v22;
        TASSIGN(v61, v62);
        pipe_barrier(PIPE_V);
        TLOG(v61, v59);
        // pto: %gp_softplus_log_inline2547_inline11073__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v63 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v20, v20);
        // pto: %gp_softplus_log_inline2547_inline11073__tile
        uint64_t v64 = (uint64_t)v23;
        TASSIGN(v63, v64);
        pipe_barrier(PIPE_V);
        TADD(v63, v49, v61);
        // pto: %6
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v65 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v20, v20);
        // pto: %6
        uint64_t v66 = (uint64_t)v22;
        TASSIGN(v65, v66);
        pipe_barrier(PIPE_V);
        TNEG(v65, v47);
        // pto: %7
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v67 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v20, v20);
        // pto: %7
        uint64_t v68 = (uint64_t)v22;
        TASSIGN(v67, v68);
        pipe_barrier(PIPE_V);
        TSUBS(v67, v65, v16);
        // pto: %8
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v69 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v20, v20);
        // pto: %8
        uint64_t v70 = (uint64_t)v22;
        TASSIGN(v69, v70);
        pipe_barrier(PIPE_V);
        TMAXS(v69, v67, v18);
        // pto: %gp_neg_floor_mask_inline2544_inline11114__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v71 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v20, v20);
        // pto: %gp_neg_floor_mask_inline2544_inline11114__tile
        uint64_t v72 = (uint64_t)v22;
        TASSIGN(v71, v72);
        pipe_barrier(PIPE_V);
        TMINS(v71, v69, v17);
        // pto: %9
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v73 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v20, v20);
        // pto: %9
        uint64_t v74 = (uint64_t)v24;
        TASSIGN(v73, v74);
        TMINS(v73, v47, v18);
        // pto: %10
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v75 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v20, v20);
        // pto: %10
        uint64_t v76 = (uint64_t)v24;
        TASSIGN(v75, v76);
        pipe_barrier(PIPE_V);
        TEXP(v75, v73);
        // pto: %gp_neg_floor_inline2550_inline11014__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v77 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v20, v20);
        // pto: %gp_neg_floor_inline2550_inline11014__tile
        uint64_t v78 = (uint64_t)v24;
        TASSIGN(v77, v78);
        pipe_barrier(PIPE_V);
        TMUL(v77, v71, v75);
        // pto: %gp_softplus_inline2543_inline11104__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v79 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v20, v20);
        // pto: %gp_softplus_inline2543_inline11104__tile
        uint64_t v80 = (uint64_t)v24;
        TASSIGN(v79, v80);
        pipe_barrier(PIPE_V);
        TMAX(v79, v63, v77);
        // pto: %gp_score_inline2542_inline11025__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v81 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v20, v20);
        // pto: %gp_score_inline2542_inline11025__tile
        uint64_t v82 = (uint64_t)v24;
        TASSIGN(v81, v82);
        pipe_barrier(PIPE_V);
        TSQRT(v81, v79);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %route_scores_buf_inline2607_inline11080__ssa_v0_pview
        pto::Shape<1, 1, 1, 16, 16> v83 = pto::Shape<1, 1, 1, 16, 16>();
        // pto: %route_scores_buf_inline2607_inline11080__ssa_v0_pview
        pto::Stride<4096, 4096, 4096, 256, 1> v84 = pto::Stride<4096, 4096, 4096, 256, 1>();
        // pto: %route_scores_buf_inline2607_inline11080__ssa_v0_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v85 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(
                v5 + ((v19 + v42 * v13) + v34), v83, v84
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v85, v81);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        // pto: %45, %46
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        if ((int64_t)v9 >= v15) {
            // pto: %11
            Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v86 = Tile<
                    TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v20, v20);
            // pto: %11
            uint64_t v87 = (uint64_t)v23;
            TASSIGN(v86, v87);
            TEXPANDS(v86, v17);
            // pto: %gp_bias_inline2541_inline11138__tile
            Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v88 = Tile<
                    TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v20, v20);
            // pto: %gp_bias_inline2541_inline11138__tile
            uint64_t v89 = (uint64_t)v23;
            TASSIGN(v88, v89);
            pipe_barrier(PIPE_V);
            TCOLEXPANDMUL(v88, v86, v38);
            // pto: %gp_biased_inline2570_inline11074__tile
            Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v90 = Tile<
                    TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v20, v20);
            // pto: %gp_biased_inline2570_inline11074__tile
            uint64_t v91 = (uint64_t)v24;
            TASSIGN(v90, v91);
            pipe_barrier(PIPE_V);
            TADD(v90, v81, v88);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            // pto: %biased_scores_buf_inline2564_inline11169__ssa_v1_pview
            pto::Shape<1, 1, 1, 16, 16> v92 = pto::Shape<1, 1, 1, 16, 16>();
            // pto: %biased_scores_buf_inline2564_inline11169__ssa_v1_pview
            pto::Stride<4096, 4096, 4096, 256, 1> v93 = pto::Stride<4096, 4096, 4096, 256, 1>();
            // pto: %biased_scores_buf_inline2564_inline11169__ssa_v1_pview
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>
                v94 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(
                    v6 + ((v19 + v42 * v13) + v34), v92, v93
                );
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            TSTORE(v94, v90);
        }
    } else {
        // pto: %12
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v95 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v21, v19);
        // pto: %12
        uint64_t v96 = (uint64_t)v26;
        TASSIGN(v95, v96);
        // pto: %13
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v97 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %13
        uint64_t v98 = (uint64_t)v26;
        TASSIGN(v97, v98);
        // pto: %14
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v99 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %14
        uint64_t v100 = (uint64_t)v25;
        TASSIGN(v99, v100);
        // pto: %50
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v101 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v14, v14);
        v101.SetValidShape(v19, v19);
        TPOP<
            TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>,
            Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>,
            TileSplitAxis::TILE_NO_SPLIT>(v28, v101);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %15
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v102 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %15
        uint64_t v103 = (uint64_t)v24;
        TASSIGN(v102, v103);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TROWEXPANDMUL(v102, v101, v99);
        TFREE<TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>, TileSplitAxis::TILE_NO_SPLIT>(v28);
        // pto: %16
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v104 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %16
        uint64_t v105 = (uint64_t)v23;
        TASSIGN(v104, v105);
        pipe_barrier(PIPE_V);
        TMAXS(v104, v102, v18);
        // pto: %17
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v106 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %17
        uint64_t v107 = (uint64_t)v22;
        TASSIGN(v106, v107);
        TNEG(v106, v102);
        // pto: %18
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v108 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %18
        uint64_t v109 = (uint64_t)v22;
        TASSIGN(v108, v109);
        pipe_barrier(PIPE_V);
        TMAX(v108, v102, v106);
        // pto: %19
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v110 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %19
        uint64_t v111 = (uint64_t)v22;
        TASSIGN(v110, v111);
        pipe_barrier(PIPE_V);
        TNEG(v110, v108);
        // pto: %20
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v112 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %20
        uint64_t v113 = (uint64_t)v22;
        TASSIGN(v112, v113);
        pipe_barrier(PIPE_V);
        TEXP(v112, v110);
        // pto: %21
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v114 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %21
        uint64_t v115 = (uint64_t)v22;
        TASSIGN(v114, v115);
        pipe_barrier(PIPE_V);
        TADDS(v114, v112, v17);
        // pto: %22
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v116 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %22
        uint64_t v117 = (uint64_t)v22;
        TASSIGN(v116, v117);
        pipe_barrier(PIPE_V);
        TLOG(v116, v114);
        // pto: %23
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v118 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %23
        uint64_t v119 = (uint64_t)v23;
        TASSIGN(v118, v119);
        pipe_barrier(PIPE_V);
        TADD(v118, v104, v116);
        // pto: %24
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v120 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %24
        uint64_t v121 = (uint64_t)v22;
        TASSIGN(v120, v121);
        pipe_barrier(PIPE_V);
        TNEG(v120, v102);
        // pto: %25
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v122 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %25
        uint64_t v123 = (uint64_t)v22;
        TASSIGN(v122, v123);
        pipe_barrier(PIPE_V);
        TSUBS(v122, v120, v16);
        // pto: %26
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v124 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %26
        uint64_t v125 = (uint64_t)v22;
        TASSIGN(v124, v125);
        pipe_barrier(PIPE_V);
        TMAXS(v124, v122, v18);
        // pto: %27
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v126 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %27
        uint64_t v127 = (uint64_t)v22;
        TASSIGN(v126, v127);
        pipe_barrier(PIPE_V);
        TMINS(v126, v124, v17);
        // pto: %28
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v128 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %28
        uint64_t v129 = (uint64_t)v24;
        TASSIGN(v128, v129);
        TMINS(v128, v102, v18);
        // pto: %29
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v130 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %29
        uint64_t v131 = (uint64_t)v24;
        TASSIGN(v130, v131);
        pipe_barrier(PIPE_V);
        TEXP(v130, v128);
        // pto: %30
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v132 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %30
        uint64_t v133 = (uint64_t)v24;
        TASSIGN(v132, v133);
        pipe_barrier(PIPE_V);
        TMUL(v132, v126, v130);
        // pto: %31
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v134 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %31
        uint64_t v135 = (uint64_t)v24;
        TASSIGN(v134, v135);
        pipe_barrier(PIPE_V);
        TMAX(v134, v118, v132);
        // pto: %32
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v136 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v19);
        // pto: %32
        uint64_t v137 = (uint64_t)v24;
        TASSIGN(v136, v137);
        pipe_barrier(PIPE_V);
        TSQRT(v136, v134);
        // pto: %51, %52
        if ((int64_t)v9 >= v15) {
            // pto: %33
            Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v138 = Tile<
                    TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v19);
            // pto: %33
            uint64_t v139 = (uint64_t)v23;
            TASSIGN(v138, v139);
            TEXPANDS(v138, v17);
            // pto: %34
            Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v140 = Tile<
                    TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v19);
            // pto: %34
            uint64_t v141 = (uint64_t)v23;
            TASSIGN(v140, v141);
            pipe_barrier(PIPE_V);
            TCOLEXPANDMUL(v140, v138, v97);
            // pto: %35
            Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v142 = Tile<
                    TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v19);
            // pto: %35
            uint64_t v143 = (uint64_t)v24;
            TASSIGN(v142, v143);
            pipe_barrier(PIPE_V);
            TADD(v142, v136, v140);
        }
    }
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
#if !defined(__CPU_SIM)
    // Read A2A3 mixed-task subblock id from runtime dispatch context
    pypto_runtime_subblock_id = get_sub_block_id(args);
#endif

    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Read SPMD subblock (AIV lane) id from runtime dispatch payload
    int32_t __pypto_spmd_subblock_idx = get_sub_block_id(args);

    // Unpack tensor: gate_bias_csa_inline538__ssa_v0
    __gm__ Tensor *gate_bias_csa_inline538__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *gate_bias_csa_inline538__ssa_v0 =
        reinterpret_cast<__gm__ float *>(gate_bias_csa_inline538__ssa_v0_tensor->buffer.addr) +
        gate_bias_csa_inline538__ssa_v0_tensor->start_offset;

    // Unpack tensor: xg_buf_inline2582_inline11203__ssa_v0
    __gm__ Tensor *xg_buf_inline2582_inline11203__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *xg_buf_inline2582_inline11203__ssa_v0 =
        reinterpret_cast<__gm__ float *>(xg_buf_inline2582_inline11203__ssa_v0_tensor->buffer.addr) +
        xg_buf_inline2582_inline11203__ssa_v0_tensor->start_offset;

    // Unpack tensor: gate_w_csa_inline540__ssa_v0
    __gm__ Tensor *gate_w_csa_inline540__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *gate_w_csa_inline540__ssa_v0 =
        reinterpret_cast<__gm__ float *>(gate_w_csa_inline540__ssa_v0_tensor->buffer.addr) +
        gate_w_csa_inline540__ssa_v0_tensor->start_offset;

    // Unpack tensor: inv_rms_buf_inline2585_inline11089__ssa_v0
    __gm__ Tensor *inv_rms_buf_inline2585_inline11089__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ float *inv_rms_buf_inline2585_inline11089__ssa_v0 =
        reinterpret_cast<__gm__ float *>(inv_rms_buf_inline2585_inline11089__ssa_v0_tensor->buffer.addr) +
        inv_rms_buf_inline2585_inline11089__ssa_v0_tensor->start_offset;

    // Unpack tensor: route_scores_buf_inline2607_inline11080__ssa_v0
    __gm__ Tensor *route_scores_buf_inline2607_inline11080__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ float *route_scores_buf_inline2607_inline11080__ssa_v0 =
        reinterpret_cast<__gm__ float *>(route_scores_buf_inline2607_inline11080__ssa_v0_tensor->buffer.addr) +
        route_scores_buf_inline2607_inline11080__ssa_v0_tensor->start_offset;

    // Unpack tensor: biased_scores_buf_inline2564_inline11169__ssa_v1
    __gm__ Tensor *biased_scores_buf_inline2564_inline11169__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[5]);
    __gm__ float *biased_scores_buf_inline2564_inline11169__ssa_v1 =
        reinterpret_cast<__gm__ float *>(biased_scores_buf_inline2564_inline11169__ssa_v1_tensor->buffer.addr) +
        biased_scores_buf_inline2564_inline11169__ssa_v1_tensor->start_offset;

    // Unpack tensor: __gm_pipe_buffer
    __gm__ Tensor *__gm_pipe_buffer_tensor = reinterpret_cast<__gm__ Tensor *>(args[6]);
    // SPMD: shard GM pipe workspace by logical block_idx to avoid overlap.
    int64_t __pypto_gm_block_num = static_cast<int64_t>(__pypto_spmd_block_num);
    if (__pypto_gm_block_num <= 0) __pypto_gm_block_num = 1;
    int64_t __pypto_gm_total_elems = static_cast<int64_t>(__gm_pipe_buffer_tensor->shapes[0]);
    int64_t __pypto_gm_elems_per_block = __pypto_gm_total_elems / __pypto_gm_block_num;
    int64_t __pypto_gm_block_offset = static_cast<int64_t>(__pypto_spmd_block_idx) * __pypto_gm_elems_per_block;
    __gm__ float *__gm_pipe_buffer = reinterpret_cast<__gm__ float *>(__gm_pipe_buffer_tensor->buffer.addr) +
                                     __gm_pipe_buffer_tensor->start_offset + __pypto_gm_block_offset;

    // Unpack scalar: GATE_N_BLOCKS_inline2586_inline11158__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } GATE_N_BLOCKS_inline2586_inline11158__ssa_v0_conv;
    GATE_N_BLOCKS_inline2586_inline11158__ssa_v0_conv.u64 = args[7];
    int64_t GATE_N_BLOCKS_inline2586_inline11158__ssa_v0 = GATE_N_BLOCKS_inline2586_inline11158__ssa_v0_conv.val;

    // Unpack scalar: csa_layer_inline714__ssa_v0
    union {
        uint64_t u64;
        int32_t val;
    } csa_layer_inline714__ssa_v0_conv;
    csa_layer_inline714__ssa_v0_conv.u64 = args[8];
    int32_t csa_layer_inline714__ssa_v0 = csa_layer_inline714__ssa_v0_conv.val;

    // Forward to ptoas-generated function
    gate_1_aiv(
        gate_bias_csa_inline538__ssa_v0, xg_buf_inline2582_inline11203__ssa_v0, gate_w_csa_inline540__ssa_v0,
        inv_rms_buf_inline2585_inline11089__ssa_v0, route_scores_buf_inline2607_inline11080__ssa_v0,
        biased_scores_buf_inline2564_inline11169__ssa_v1, __gm_pipe_buffer,
        GATE_N_BLOCKS_inline2586_inline11158__ssa_v0, csa_layer_inline714__ssa_v0, __pypto_spmd_block_idx,
        __pypto_spmd_block_num, __pypto_spmd_subblock_idx
    );
}
