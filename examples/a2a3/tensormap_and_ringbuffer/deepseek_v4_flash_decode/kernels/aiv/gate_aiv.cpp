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
// Kernel Function: gate_aiv

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

static __aicore__ void gate_aic(
    __gm__ float *v1, __gm__ float *v2, __gm__ float *v3, __gm__ float *v4, __gm__ float *v5, __gm__ float *v6,
    int32_t v7, int32_t v8
) {
    const int64_t v9 = 4096;
    const int64_t v10 = 512;
    const int64_t v11 = 1024;
    const int64_t v12 = 2048;
    const int64_t v13 = 4;
    const int64_t v14 = 16;
    const int64_t v15 = 32768;
    const int64_t v16 = 393216;
    const int64_t v17 = 262144;
    const int64_t v18 = 131072;
    const int64_t v19 = 0;
    const int32_t v20 = 0;
    using T = float;

#if defined(__DAV_CUBE__)
    // pto: %gate_logits_tile_inline2592_inline9336__tile_l0_c_phi
    Tile<
        TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v21 = Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v14, v14);
    // pto: %gate_logits_tile_inline2592_inline9336__tile_l0_c_phi
    uint64_t v22 = (uint64_t)v19;
    TASSIGN(v21, v22);
    auto v23 = TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>(v6, v20, v20);
    // pto: %gb_idx_inline2614_inline9327__ssa_v0
    int64_t v24 = (int64_t)v7;
    // pto: %12, %14
    int64_t v25 = (int64_t)((uint64_t)(v24 / v13) * (uint64_t)v14);
    // pto: %13, %15
    int64_t v26 = (int64_t)((uint64_t)(v24 % v13) * (uint64_t)v14);
    // pto: %gate_logits_tile_inline2592_inline9336__tile
    Tile<
        TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v27 = Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v14, v14);
    // pto: %gate_logits_tile_inline2592_inline9336__tile
    uint64_t v28 = (uint64_t)v19;
    TASSIGN(v27, v28);
    // pto: %gd_x_inline2620_inline9210__tile
    Tile<
        TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v29 = Tile<
            TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v14, v12);
    // pto: %gd_x_inline2620_inline9210__tile
    uint64_t v30 = (uint64_t)v18;
    TASSIGN(v29, v30);
    // pto: %16
    int64_t v31 = v25 < v19 ? v19 : v25;
    // pto: %xg_buf_inline2582_inline9396__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 2048> v32 = pto::Shape<1, 1, 1, 16, 2048>();
    // pto: %xg_buf_inline2582_inline9396__ssa_v0_pview
    pto::Stride<65536, 65536, 65536, 4096, 1> v33 = pto::Stride<65536, 65536, 65536, 4096, 1>();
    // pto: %xg_buf_inline2582_inline9396__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v34 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(
            v2 + (v19 + v31 * v9), v32, v33
        );
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    TLOAD(v29, v34);
    // pto: %gd_w_inline2571_inline9203__tile
    Tile<
        TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v35 = Tile<
            TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v14, v12);
    // pto: %gd_w_inline2571_inline9203__tile
    uint64_t v36 = (uint64_t)v17;
    TASSIGN(v35, v36);
    // pto: %17
    int64_t v37 = v26 < v19 ? v19 : v26;
    // pto: %gate_w_l0_inline573__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 2048> v38 = pto::Shape<1, 1, 1, 16, 2048>();
    // pto: %gate_w_l0_inline573__ssa_v0_pview
    pto::Stride<65536, 65536, 65536, 4096, 1> v39 = pto::Stride<65536, 65536, 65536, 4096, 1>();
    // pto: %gate_w_l0_inline573__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v40 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(
            v3 + (v19 + v37 * v9), v38, v39
        );
    TLOAD(v35, v40);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // pto: %0
    Tile<
        TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v41 = Tile<
            TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v14, v12);
    // pto: %0
    uint64_t v42 = (uint64_t)v16;
    TASSIGN(v41, v42);
    // pto: %19
    pto::Shape<1, 1, 1, 16, 2048> v43 = pto::Shape<1, 1, 1, 16, 2048>();
    // pto: %19
    pto::Stride<65536, 65536, 65536, 4096, 1> v44 = pto::Stride<65536, 65536, 65536, 4096, 1>();
    // pto: %19
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v45 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(
            v2 + (v12 + v31 * v9), v43, v44
        );
    TLOAD(v41, v45);
    // pto: %1
    Tile<
        TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v46 = Tile<
            TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v14, v12);
    // pto: %1
    uint64_t v47 = (uint64_t)v19;
    TASSIGN(v46, v47);
    // pto: %21
    pto::Shape<1, 1, 1, 16, 2048> v48 = pto::Shape<1, 1, 1, 16, 2048>();
    // pto: %21
    pto::Stride<65536, 65536, 65536, 4096, 1> v49 = pto::Stride<65536, 65536, 65536, 4096, 1>();
    // pto: %21
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v50 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(
            v3 + (v12 + v37 * v9), v48, v49
        );
    TLOAD(v46, v50);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    // pto: %gd_w_inline2571_inline9203__tile_t
    Tile<
        TileType::Mat, float, 2048, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v51 = Tile<
            TileType::Mat, float, 2048, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v12, v14);
    // pto: %gd_w_inline2571_inline9203__tile_t
    uint64_t v52 = (uint64_t)v17;
    TASSIGN(v51, v52);
    // pto: %gate_logits_tile_inline2592_inline9336__tile_l0_init
    Tile<
        TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v53 = Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v14, v14);
    // pto: %gate_logits_tile_inline2592_inline9336__tile_l0_init
    uint64_t v54 = (uint64_t)v19;
    TASSIGN(v53, v54);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    for (int64_t i55 = v19; i55 < v12; i55 += v11) {
        // pto: %gate_logits_tile_inline2592_inline9336__tile_l0_a
        Tile<
            TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v56 = Tile<
                TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v14, v10);
        // pto: %gate_logits_tile_inline2592_inline9336__tile_l0_a
        uint64_t v57 = (uint64_t)v19;
        TASSIGN(v56, v57);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        pipe_barrier(PIPE_MTE1);
        TEXTRACT(v56, v29, v19, i55);
        // pto: %gate_logits_tile_inline2592_inline9336__tile_l0_b
        Tile<
            TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v58 = Tile<
                TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v10, v14);
        // pto: %gate_logits_tile_inline2592_inline9336__tile_l0_b
        uint64_t v59 = (uint64_t)v19;
        TASSIGN(v58, v59);
        TEXTRACT(v58, v51, i55, v19);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        // pto: %2
        Tile<
            TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v60 = Tile<
                TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v14, v10);
        // pto: %2
        uint64_t v61 = (uint64_t)v15;
        TASSIGN(v60, v61);
        // pto: %22
        int64_t v62 = (int64_t)((uint64_t)i55 + (uint64_t)v10);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v60, v29, v19, v62);
        // pto: %3
        Tile<
            TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v63 = Tile<
                TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v10, v14);
        // pto: %3
        uint64_t v64 = (uint64_t)v15;
        TASSIGN(v63, v64);
        TEXTRACT(v63, v51, v62, v19);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        // pto: %24
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if (i55 == v19) {
            // pto: %gate_logits_tile_inline2592_inline9336__tile_l0_c_first
            Tile<
                TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v65 = Tile<
                    TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v14, v14);
            // pto: %gate_logits_tile_inline2592_inline9336__tile_l0_c_first
            uint64_t v66 = (uint64_t)v19;
            TASSIGN(v65, v66);
            pipe_barrier(PIPE_M);
            TMATMUL(v65, v56, v58);
        } else {
            // pto: %gate_logits_tile_inline2592_inline9336__tile_l0_c_acc
            Tile<
                TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v67 = Tile<
                    TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v14, v14);
            // pto: %gate_logits_tile_inline2592_inline9336__tile_l0_c_acc
            uint64_t v68 = (uint64_t)v19;
            TASSIGN(v67, v68);
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v67, v67, v56, v58);
        }
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        // pto: %4
        Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v69 = Tile<
                TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v14, v14);
        // pto: %4
        uint64_t v70 = (uint64_t)v19;
        TASSIGN(v69, v70);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        TMATMUL_ACC(v69, v69, v60, v63);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    }
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    // pto: %5
    Tile<
        TileType::Mat, float, 2048, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v71 = Tile<
            TileType::Mat, float, 2048, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v12, v14);
    // pto: %5
    uint64_t v72 = (uint64_t)v19;
    TASSIGN(v71, v72);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    for (int64_t i73 = v19; i73 < v12; i73 += v11) {
        // pto: %6
        Tile<
            TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v74 = Tile<
                TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v14, v10);
        // pto: %6
        uint64_t v75 = (uint64_t)v19;
        TASSIGN(v74, v75);
        pipe_barrier(PIPE_MTE1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        TEXTRACT(v74, v41, v19, i73);
        // pto: %7
        Tile<
            TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v76 = Tile<
                TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v10, v14);
        // pto: %7
        uint64_t v77 = (uint64_t)v19;
        TASSIGN(v76, v77);
        TEXTRACT(v76, v71, i73, v19);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        // pto: %8
        Tile<
            TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v78 = Tile<
                TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v14, v10);
        // pto: %8
        uint64_t v79 = (uint64_t)v15;
        TASSIGN(v78, v79);
        // pto: %26
        int64_t v80 = (int64_t)((uint64_t)i73 + (uint64_t)v10);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        TEXTRACT(v78, v41, v19, v80);
        // pto: %9
        Tile<
            TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v81 = Tile<
                TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v10, v14);
        // pto: %9
        uint64_t v82 = (uint64_t)v15;
        TASSIGN(v81, v82);
        TEXTRACT(v81, v71, v80, v19);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        // pto: %10
        Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v83 = Tile<
                TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v14, v14);
        // pto: %10
        uint64_t v84 = (uint64_t)v19;
        TASSIGN(v83, v84);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v83, v83, v74, v76);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        // pto: %11
        Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v85 = Tile<
                TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v14, v14);
        // pto: %11
        uint64_t v86 = (uint64_t)v19;
        TASSIGN(v85, v86);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        TMATMUL_ACC(v85, v85, v78, v81);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TPUSH<
        TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>,
        Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>,
        TileSplitAxis::TILE_NO_SPLIT>(v23, v53);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
#endif  // __DAV_CUBE__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}

static __aicore__ void gate_aiv(
    __gm__ float *v1, __gm__ float *v2, __gm__ float *v3, __gm__ float *v4, __gm__ float *v5, __gm__ float *v6,
    int32_t v7, int32_t v8, int32_t v9
) {
    const int64_t v10 = 256;
    const int32_t v11 = 16;
    const float v12 = 10.0f;
    const float v13 = 1.0f;
    const float v14 = 0.0f;
    const int64_t v15 = 4;
    const int64_t v16 = 0;
    const int64_t v17 = 16;
    const int64_t v18 = 1;
    const int64_t v19 = 10304;
    const int64_t v20 = 9280;
    const int64_t v21 = 8256;
    const int64_t v22 = 8192;
    const int32_t v23 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    auto v24 = TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>(v6, v23, v23);
    // pto: %subblock_idx, %29
    if ((int64_t)v9 == v16) {
        // pto: %gb_idx_inline2614_inline9327__ssa_v0
        int64_t v25 = (int64_t)v7;
        // pto: %30, %32
        int64_t v26 = (int64_t)((uint64_t)(v25 / v15) * (uint64_t)v17);
        // pto: %31, %33
        int64_t v27 = (int64_t)((uint64_t)(v25 % v15) * (uint64_t)v17);
        // pto: %t__tile
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v28 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v18);
        // pto: %t__tile
        uint64_t v29 = (uint64_t)v22;
        TASSIGN(v28, v29);
        // pto: %34
        int64_t v30 = v26 < v16 ? v16 : v26;
        // pto: %inv_rms_buf_inline2585_inline9282__ssa_v0_pview
        pto::Shape<1, 1, 1, 16, 1> v31 = pto::Shape<1, 1, 1, 16, 1>();
        // pto: %inv_rms_buf_inline2585_inline9282__ssa_v0_pview
        pto::Stride<16, 16, 16, 1, 16> v32 = pto::Stride<16, 16, 16, 1, 16>();
        // pto: %inv_rms_buf_inline2585_inline9282__ssa_v0_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 16>, pto::Layout::DN> v33 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 16>, pto::Layout::DN>(
                v4 + (v16 + v30), v31, v32
            );
        TLOAD(v28, v33);
        // pto: %gate_logits_tile_inline2592_inline9336__rv_v2_Vec
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v34 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v11, v11);
        TPOP<
            TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>,
            Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>,
            TileSplitAxis::TILE_NO_SPLIT>(v24, v34);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %gate_logits_tile_v1_inline2568_inline9380__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v35 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %gate_logits_tile_v1_inline2568_inline9380__tile
        uint64_t v36 = (uint64_t)v21;
        TASSIGN(v35, v36);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TROWEXPANDMUL(v35, v34, v28);
        TFREE<TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>, TileSplitAxis::TILE_NO_SPLIT>(v24);
        // pto: %gp_relu_inline2577_inline9283__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v37 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %gp_relu_inline2577_inline9283__tile
        uint64_t v38 = (uint64_t)v20;
        TASSIGN(v37, v38);
        pipe_barrier(PIPE_V);
        TMAXS(v37, v35, v14);
        // pto: %0
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v39 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %0
        uint64_t v40 = (uint64_t)v19;
        TASSIGN(v39, v40);
        TNEG(v39, v35);
        // pto: %gp_abs_inline2545_inline9309__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v41 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %gp_abs_inline2545_inline9309__tile
        uint64_t v42 = (uint64_t)v19;
        TASSIGN(v41, v42);
        pipe_barrier(PIPE_V);
        TMAX(v41, v35, v39);
        // pto: %1
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v43 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %1
        uint64_t v44 = (uint64_t)v19;
        TASSIGN(v43, v44);
        pipe_barrier(PIPE_V);
        TNEG(v43, v41);
        // pto: %2
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v45 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %2
        uint64_t v46 = (uint64_t)v19;
        TASSIGN(v45, v46);
        pipe_barrier(PIPE_V);
        TEXP(v45, v43);
        // pto: %3
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v47 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %3
        uint64_t v48 = (uint64_t)v19;
        TASSIGN(v47, v48);
        pipe_barrier(PIPE_V);
        TADDS(v47, v45, v13);
        // pto: %4
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v49 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %4
        uint64_t v50 = (uint64_t)v19;
        TASSIGN(v49, v50);
        pipe_barrier(PIPE_V);
        TLOG(v49, v47);
        // pto: %gp_softplus_log_inline2547_inline9266__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v51 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %gp_softplus_log_inline2547_inline9266__tile
        uint64_t v52 = (uint64_t)v20;
        TASSIGN(v51, v52);
        pipe_barrier(PIPE_V);
        TADD(v51, v37, v49);
        // pto: %5
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v53 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %5
        uint64_t v54 = (uint64_t)v19;
        TASSIGN(v53, v54);
        pipe_barrier(PIPE_V);
        TNEG(v53, v35);
        // pto: %6
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v55 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %6
        uint64_t v56 = (uint64_t)v19;
        TASSIGN(v55, v56);
        pipe_barrier(PIPE_V);
        TSUBS(v55, v53, v12);
        // pto: %7
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v57 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %7
        uint64_t v58 = (uint64_t)v19;
        TASSIGN(v57, v58);
        pipe_barrier(PIPE_V);
        TMAXS(v57, v55, v14);
        // pto: %gp_neg_floor_mask_inline2544_inline9307__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v59 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %gp_neg_floor_mask_inline2544_inline9307__tile
        uint64_t v60 = (uint64_t)v19;
        TASSIGN(v59, v60);
        pipe_barrier(PIPE_V);
        TMINS(v59, v57, v13);
        // pto: %8
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v61 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %8
        uint64_t v62 = (uint64_t)v21;
        TASSIGN(v61, v62);
        TMINS(v61, v35, v14);
        // pto: %9
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v63 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %9
        uint64_t v64 = (uint64_t)v21;
        TASSIGN(v63, v64);
        pipe_barrier(PIPE_V);
        TEXP(v63, v61);
        // pto: %gp_neg_floor_inline2550_inline9207__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v65 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %gp_neg_floor_inline2550_inline9207__tile
        uint64_t v66 = (uint64_t)v21;
        TASSIGN(v65, v66);
        pipe_barrier(PIPE_V);
        TMUL(v65, v59, v63);
        // pto: %gp_softplus_inline2543_inline9297__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v67 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %gp_softplus_inline2543_inline9297__tile
        uint64_t v68 = (uint64_t)v21;
        TASSIGN(v67, v68);
        pipe_barrier(PIPE_V);
        TMAX(v67, v51, v65);
        // pto: %gp_score_inline2542_inline9218__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v69 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %gp_score_inline2542_inline9218__tile
        uint64_t v70 = (uint64_t)v21;
        TASSIGN(v69, v70);
        pipe_barrier(PIPE_V);
        TSQRT(v69, v67);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %route_scores_buf_inline2607_inline9273__ssa_v0_pview
        pto::Shape<1, 1, 1, 16, 16> v71 = pto::Shape<1, 1, 1, 16, 16>();
        // pto: %route_scores_buf_inline2607_inline9273__ssa_v0_pview
        pto::Stride<4096, 4096, 4096, 256, 1> v72 = pto::Stride<4096, 4096, 4096, 256, 1>();
        // pto: %route_scores_buf_inline2607_inline9273__ssa_v0_pview, %36
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v73 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(
                v5 + ((v16 + v30 * v10) + (v27 < v16 ? v16 : v27)), v71, v72
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v73, v69);
    } else {
        // pto: %10
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v74 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %10
        uint64_t v75 = (uint64_t)v22;
        TASSIGN(v74, v75);
        // pto: %38
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v76 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v11, v11);
        v76.SetValidShape(v16, v16);
        TPOP<
            TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>,
            Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>,
            TileSplitAxis::TILE_NO_SPLIT>(v24, v76);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %11
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v77 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %11
        uint64_t v78 = (uint64_t)v21;
        TASSIGN(v77, v78);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TROWEXPANDMUL(v77, v76, v74);
        TFREE<TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>, TileSplitAxis::TILE_NO_SPLIT>(v24);
        // pto: %12
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v79 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %12
        uint64_t v80 = (uint64_t)v20;
        TASSIGN(v79, v80);
        pipe_barrier(PIPE_V);
        TMAXS(v79, v77, v14);
        // pto: %13
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v81 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %13
        uint64_t v82 = (uint64_t)v19;
        TASSIGN(v81, v82);
        TNEG(v81, v77);
        // pto: %14
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v83 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %14
        uint64_t v84 = (uint64_t)v19;
        TASSIGN(v83, v84);
        pipe_barrier(PIPE_V);
        TMAX(v83, v77, v81);
        // pto: %15
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v85 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %15
        uint64_t v86 = (uint64_t)v19;
        TASSIGN(v85, v86);
        pipe_barrier(PIPE_V);
        TNEG(v85, v83);
        // pto: %16
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v87 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %16
        uint64_t v88 = (uint64_t)v19;
        TASSIGN(v87, v88);
        pipe_barrier(PIPE_V);
        TEXP(v87, v85);
        // pto: %17
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v89 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %17
        uint64_t v90 = (uint64_t)v19;
        TASSIGN(v89, v90);
        pipe_barrier(PIPE_V);
        TADDS(v89, v87, v13);
        // pto: %18
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v91 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %18
        uint64_t v92 = (uint64_t)v19;
        TASSIGN(v91, v92);
        pipe_barrier(PIPE_V);
        TLOG(v91, v89);
        // pto: %19
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v93 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %19
        uint64_t v94 = (uint64_t)v20;
        TASSIGN(v93, v94);
        pipe_barrier(PIPE_V);
        TADD(v93, v79, v91);
        // pto: %20
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v95 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %20
        uint64_t v96 = (uint64_t)v19;
        TASSIGN(v95, v96);
        pipe_barrier(PIPE_V);
        TNEG(v95, v77);
        // pto: %21
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v97 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %21
        uint64_t v98 = (uint64_t)v19;
        TASSIGN(v97, v98);
        pipe_barrier(PIPE_V);
        TSUBS(v97, v95, v12);
        // pto: %22
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v99 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %22
        uint64_t v100 = (uint64_t)v19;
        TASSIGN(v99, v100);
        pipe_barrier(PIPE_V);
        TMAXS(v99, v97, v14);
        // pto: %23
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v101 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %23
        uint64_t v102 = (uint64_t)v19;
        TASSIGN(v101, v102);
        pipe_barrier(PIPE_V);
        TMINS(v101, v99, v13);
        // pto: %24
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v103 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %24
        uint64_t v104 = (uint64_t)v21;
        TASSIGN(v103, v104);
        TMINS(v103, v77, v14);
        // pto: %25
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v105 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %25
        uint64_t v106 = (uint64_t)v21;
        TASSIGN(v105, v106);
        pipe_barrier(PIPE_V);
        TEXP(v105, v103);
        // pto: %26
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v107 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %26
        uint64_t v108 = (uint64_t)v21;
        TASSIGN(v107, v108);
        pipe_barrier(PIPE_V);
        TMUL(v107, v101, v105);
        // pto: %27
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v109 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %27
        uint64_t v110 = (uint64_t)v21;
        TASSIGN(v109, v110);
        pipe_barrier(PIPE_V);
        TMAX(v109, v93, v107);
        // pto: %28
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v111 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v16, v16);
        // pto: %28
        uint64_t v112 = (uint64_t)v21;
        TASSIGN(v111, v112);
        pipe_barrier(PIPE_V);
        TSQRT(v111, v109);
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

    // Unpack tensor: gate_bias_l0_inline566__ssa_v0
    __gm__ Tensor *gate_bias_l0_inline566__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *gate_bias_l0_inline566__ssa_v0 =
        reinterpret_cast<__gm__ float *>(gate_bias_l0_inline566__ssa_v0_tensor->buffer.addr) +
        gate_bias_l0_inline566__ssa_v0_tensor->start_offset;

    // Unpack tensor: xg_buf_inline2582_inline9396__ssa_v0
    __gm__ Tensor *xg_buf_inline2582_inline9396__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *xg_buf_inline2582_inline9396__ssa_v0 =
        reinterpret_cast<__gm__ float *>(xg_buf_inline2582_inline9396__ssa_v0_tensor->buffer.addr) +
        xg_buf_inline2582_inline9396__ssa_v0_tensor->start_offset;

    // Unpack tensor: gate_w_l0_inline573__ssa_v0
    __gm__ Tensor *gate_w_l0_inline573__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *gate_w_l0_inline573__ssa_v0 =
        reinterpret_cast<__gm__ float *>(gate_w_l0_inline573__ssa_v0_tensor->buffer.addr) +
        gate_w_l0_inline573__ssa_v0_tensor->start_offset;

    // Unpack tensor: inv_rms_buf_inline2585_inline9282__ssa_v0
    __gm__ Tensor *inv_rms_buf_inline2585_inline9282__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ float *inv_rms_buf_inline2585_inline9282__ssa_v0 =
        reinterpret_cast<__gm__ float *>(inv_rms_buf_inline2585_inline9282__ssa_v0_tensor->buffer.addr) +
        inv_rms_buf_inline2585_inline9282__ssa_v0_tensor->start_offset;

    // Unpack tensor: route_scores_buf_inline2607_inline9273__ssa_v0
    __gm__ Tensor *route_scores_buf_inline2607_inline9273__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ float *route_scores_buf_inline2607_inline9273__ssa_v0 =
        reinterpret_cast<__gm__ float *>(route_scores_buf_inline2607_inline9273__ssa_v0_tensor->buffer.addr) +
        route_scores_buf_inline2607_inline9273__ssa_v0_tensor->start_offset;

    // Unpack tensor: __gm_pipe_buffer
    __gm__ Tensor *__gm_pipe_buffer_tensor = reinterpret_cast<__gm__ Tensor *>(args[5]);
    // SPMD: shard GM pipe workspace by logical block_idx to avoid overlap.
    int64_t __pypto_gm_block_num = static_cast<int64_t>(__pypto_spmd_block_num);
    if (__pypto_gm_block_num <= 0) __pypto_gm_block_num = 1;
    int64_t __pypto_gm_total_elems = static_cast<int64_t>(__gm_pipe_buffer_tensor->shapes[0]);
    int64_t __pypto_gm_elems_per_block = __pypto_gm_total_elems / __pypto_gm_block_num;
    int64_t __pypto_gm_block_offset = static_cast<int64_t>(__pypto_spmd_block_idx) * __pypto_gm_elems_per_block;
    __gm__ float *__gm_pipe_buffer = reinterpret_cast<__gm__ float *>(__gm_pipe_buffer_tensor->buffer.addr) +
                                     __gm_pipe_buffer_tensor->start_offset + __pypto_gm_block_offset;

    // Forward to ptoas-generated function
    gate_aiv(
        gate_bias_l0_inline566__ssa_v0, xg_buf_inline2582_inline9396__ssa_v0, gate_w_l0_inline573__ssa_v0,
        inv_rms_buf_inline2585_inline9282__ssa_v0, route_scores_buf_inline2607_inline9273__ssa_v0, __gm_pipe_buffer,
        __pypto_spmd_block_idx, __pypto_spmd_block_num, __pypto_spmd_subblock_idx
    );
}
