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
// Kernel Function: gate_2_aiv

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

static __aicore__ void gate_2_aic(
    __gm__ float *v1, __gm__ float *v2, __gm__ float *v3, __gm__ float *v4, __gm__ float *v5, __gm__ float *v6,
    __gm__ float *v7, int64_t v8, int32_t v9, int32_t v10
) {
    const int64_t v11 = 4096;
    const int64_t v12 = 512;
    const int64_t v13 = 1024;
    const int64_t v14 = 2048;
    const int64_t v15 = 16;
    const int64_t v16 = 32768;
    const int64_t v17 = 131072;
    const int64_t v18 = 393216;
    const int64_t v19 = 262144;
    const int64_t v20 = 0;
    const int32_t v21 = 0;
    using T = float;

#if defined(__DAV_CUBE__)
    // pto: %gate_logits_tile_inline2592_inline12022__tile_l0_c_phi
    Tile<
        TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v22 = Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v15, v15);
    // pto: %gate_logits_tile_inline2592_inline12022__tile_l0_c_phi
    uint64_t v23 = (uint64_t)v20;
    TASSIGN(v22, v23);
    auto v24 = TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>(v7, v21, v21);
    // pto: %gb_idx_inline2614_inline12013__ssa_v0
    int64_t v25 = (int64_t)v9;
    // pto: %12, %14
    int64_t v26 = (int64_t)((uint64_t)(v25 / v8) * (uint64_t)v15);
    // pto: %13, %15
    int64_t v27 = (int64_t)((uint64_t)(v25 % v8) * (uint64_t)v15);
    // pto: %gate_logits_tile_inline2592_inline12022__tile
    Tile<
        TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v28 = Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v15, v15);
    // pto: %gate_logits_tile_inline2592_inline12022__tile
    uint64_t v29 = (uint64_t)v20;
    TASSIGN(v28, v29);
    // pto: %gd_x_inline2620_inline11896__tile
    Tile<
        TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v30 = Tile<
            TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v15, v14);
    // pto: %gd_x_inline2620_inline11896__tile
    uint64_t v31 = (uint64_t)v19;
    TASSIGN(v30, v31);
    // pto: %16
    int64_t v32 = v26 < v20 ? v20 : v26;
    // pto: %xg_buf_inline2582_inline12082__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 2048> v33 = pto::Shape<1, 1, 1, 16, 2048>();
    // pto: %xg_buf_inline2582_inline12082__ssa_v0_pview
    pto::Stride<65536, 65536, 65536, 4096, 1> v34 = pto::Stride<65536, 65536, 65536, 4096, 1>();
    // pto: %xg_buf_inline2582_inline12082__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v35 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(
            v2 + (v20 + v32 * v11), v33, v34
        );
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    TLOAD(v30, v35);
    // pto: %gd_w_inline2571_inline11889__tile
    Tile<
        TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v36 = Tile<
            TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v15, v14);
    // pto: %gd_w_inline2571_inline11889__tile
    uint64_t v37 = (uint64_t)v18;
    TASSIGN(v36, v37);
    // pto: %17
    int64_t v38 = v27 < v20 ? v20 : v27;
    // pto: %gate_w_hca_inline680__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 2048> v39 = pto::Shape<1, 1, 1, 16, 2048>();
    // pto: %gate_w_hca_inline680__ssa_v0_pview
    pto::Stride<65536, 65536, 65536, 4096, 1> v40 = pto::Stride<65536, 65536, 65536, 4096, 1>();
    // pto: %gate_w_hca_inline680__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v41 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(
            v3 + (v20 + v38 * v11), v39, v40
        );
    TLOAD(v36, v41);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // pto: %0
    Tile<
        TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v42 = Tile<
            TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v15, v14);
    // pto: %0
    uint64_t v43 = (uint64_t)v20;
    TASSIGN(v42, v43);
    // pto: %19
    pto::Shape<1, 1, 1, 16, 2048> v44 = pto::Shape<1, 1, 1, 16, 2048>();
    // pto: %19
    pto::Stride<65536, 65536, 65536, 4096, 1> v45 = pto::Stride<65536, 65536, 65536, 4096, 1>();
    // pto: %19
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v46 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(
            v2 + (v14 + v32 * v11), v44, v45
        );
    TLOAD(v42, v46);
    // pto: %1
    Tile<
        TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v47 = Tile<
            TileType::Mat, float, 16, 2048, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v15, v14);
    // pto: %1
    uint64_t v48 = (uint64_t)v17;
    TASSIGN(v47, v48);
    // pto: %21
    pto::Shape<1, 1, 1, 16, 2048> v49 = pto::Shape<1, 1, 1, 16, 2048>();
    // pto: %21
    pto::Stride<65536, 65536, 65536, 4096, 1> v50 = pto::Stride<65536, 65536, 65536, 4096, 1>();
    // pto: %21
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v51 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 2048>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(
            v3 + (v14 + v38 * v11), v49, v50
        );
    TLOAD(v47, v51);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    // pto: %gd_w_inline2571_inline11889__tile_t
    Tile<
        TileType::Mat, float, 2048, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v52 = Tile<
            TileType::Mat, float, 2048, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v14, v15);
    // pto: %gd_w_inline2571_inline11889__tile_t
    uint64_t v53 = (uint64_t)v18;
    TASSIGN(v52, v53);
    // pto: %gate_logits_tile_inline2592_inline12022__tile_l0_init
    Tile<
        TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v54 = Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v15, v15);
    // pto: %gate_logits_tile_inline2592_inline12022__tile_l0_init
    uint64_t v55 = (uint64_t)v20;
    TASSIGN(v54, v55);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    for (int64_t i56 = v20; i56 < v14; i56 += v13) {
        // pto: %gate_logits_tile_inline2592_inline12022__tile_l0_a
        Tile<
            TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v57 = Tile<
                TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v12);
        // pto: %gate_logits_tile_inline2592_inline12022__tile_l0_a
        uint64_t v58 = (uint64_t)v20;
        TASSIGN(v57, v58);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        pipe_barrier(PIPE_MTE1);
        TEXTRACT(v57, v30, v20, i56);
        // pto: %gate_logits_tile_inline2592_inline12022__tile_l0_b
        Tile<
            TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v59 = Tile<
                TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v12, v15);
        // pto: %gate_logits_tile_inline2592_inline12022__tile_l0_b
        uint64_t v60 = (uint64_t)v20;
        TASSIGN(v59, v60);
        TEXTRACT(v59, v52, i56, v20);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        // pto: %2
        Tile<
            TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v61 = Tile<
                TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v12);
        // pto: %2
        uint64_t v62 = (uint64_t)v16;
        TASSIGN(v61, v62);
        // pto: %22
        int64_t v63 = (int64_t)((uint64_t)i56 + (uint64_t)v12);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v61, v30, v20, v63);
        // pto: %3
        Tile<
            TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v64 = Tile<
                TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v12, v15);
        // pto: %3
        uint64_t v65 = (uint64_t)v16;
        TASSIGN(v64, v65);
        TEXTRACT(v64, v52, v63, v20);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        // pto: %24
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if (i56 == v20) {
            // pto: %gate_logits_tile_inline2592_inline12022__tile_l0_c_first
            Tile<
                TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v66 = Tile<
                    TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v15, v15);
            // pto: %gate_logits_tile_inline2592_inline12022__tile_l0_c_first
            uint64_t v67 = (uint64_t)v20;
            TASSIGN(v66, v67);
            pipe_barrier(PIPE_M);
            TMATMUL(v66, v57, v59);
        } else {
            // pto: %gate_logits_tile_inline2592_inline12022__tile_l0_c_acc
            Tile<
                TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v68 = Tile<
                    TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v15, v15);
            // pto: %gate_logits_tile_inline2592_inline12022__tile_l0_c_acc
            uint64_t v69 = (uint64_t)v20;
            TASSIGN(v68, v69);
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v68, v68, v57, v59);
        }
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        // pto: %4
        Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v70 = Tile<
                TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v15, v15);
        // pto: %4
        uint64_t v71 = (uint64_t)v20;
        TASSIGN(v70, v71);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        TMATMUL_ACC(v70, v70, v61, v64);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    }
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    // pto: %5
    Tile<
        TileType::Mat, float, 2048, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v72 = Tile<
            TileType::Mat, float, 2048, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v14, v15);
    // pto: %5
    uint64_t v73 = (uint64_t)v17;
    TASSIGN(v72, v73);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    for (int64_t i74 = v20; i74 < v14; i74 += v13) {
        // pto: %6
        Tile<
            TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v75 = Tile<
                TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v12);
        // pto: %6
        uint64_t v76 = (uint64_t)v20;
        TASSIGN(v75, v76);
        pipe_barrier(PIPE_MTE1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        TEXTRACT(v75, v42, v20, i74);
        // pto: %7
        Tile<
            TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v77 = Tile<
                TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v12, v15);
        // pto: %7
        uint64_t v78 = (uint64_t)v20;
        TASSIGN(v77, v78);
        TEXTRACT(v77, v72, i74, v20);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        // pto: %8
        Tile<
            TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v79 = Tile<
                TileType::Left, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v12);
        // pto: %8
        uint64_t v80 = (uint64_t)v16;
        TASSIGN(v79, v80);
        // pto: %26
        int64_t v81 = (int64_t)((uint64_t)i74 + (uint64_t)v12);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        TEXTRACT(v79, v42, v20, v81);
        // pto: %9
        Tile<
            TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v82 = Tile<
                TileType::Right, float, 512, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v12, v15);
        // pto: %9
        uint64_t v83 = (uint64_t)v16;
        TASSIGN(v82, v83);
        TEXTRACT(v82, v72, v81, v20);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        // pto: %10
        Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v84 = Tile<
                TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v15, v15);
        // pto: %10
        uint64_t v85 = (uint64_t)v20;
        TASSIGN(v84, v85);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v84, v84, v75, v77);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        // pto: %11
        Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v86 = Tile<
                TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v15, v15);
        // pto: %11
        uint64_t v87 = (uint64_t)v20;
        TASSIGN(v86, v87);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        TMATMUL_ACC(v86, v86, v79, v82);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TPUSH<
        TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>,
        Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>,
        TileSplitAxis::TILE_NO_SPLIT>(v24, v54);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
#endif  // __DAV_CUBE__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}

static __aicore__ void gate_2_aiv(
    __gm__ float *v1, __gm__ float *v2, __gm__ float *v3, __gm__ float *v4, __gm__ float *v5, __gm__ float *v6,
    __gm__ float *v7, int64_t v8, int32_t v9, int32_t v10, int32_t v11
) {
    const int64_t v12 = 256;
    const int32_t v13 = 16;
    const float v14 = 10.0f;
    const float v15 = 1.0f;
    const float v16 = 0.0f;
    const int64_t v17 = 0;
    const int64_t v18 = 16;
    const int64_t v19 = 1;
    const int64_t v20 = 9216;
    const int64_t v21 = 8192;
    const int64_t v22 = 10368;
    const int64_t v23 = 10304;
    const int64_t v24 = 10240;
    const int32_t v25 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    auto v26 = TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>(v7, v25, v25);
    // pto: %subblock_idx, %36
    if ((int64_t)v11 == v17) {
        // pto: %gb_idx_inline2614_inline12013__ssa_v0
        int64_t v27 = (int64_t)v9;
        // pto: %37, %39
        int64_t v28 = (int64_t)((uint64_t)(v27 / v8) * (uint64_t)v18);
        // pto: %38, %40
        int64_t v29 = (int64_t)((uint64_t)(v27 % v8) * (uint64_t)v18);
        // pto: %t__tile
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v30 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v18);
        // pto: %t__tile
        uint64_t v31 = (uint64_t)v24;
        TASSIGN(v30, v31);
        // pto: %41
        int64_t v32 = v29 < v17 ? v17 : v29;
        // pto: %gate_bias_hca_inline699__ssa_v0_pview
        pto::Shape<1, 1, 1, 1, 16> v33 = pto::Shape<1, 1, 1, 1, 16>();
        // pto: %gate_bias_hca_inline699__ssa_v0_pview
        pto::Stride<16, 16, 16, 16, 1> v34 = pto::Stride<16, 16, 16, 16, 1>();
        // pto: %gate_bias_hca_inline699__ssa_v0_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v35 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(
                v1 + (v17 + v32), v33, v34
            );
        TLOAD(v30, v35);
        // pto: %gp_bias_row_inline2559_inline11887__tile
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v36 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v18);
        // pto: %gp_bias_row_inline2559_inline11887__tile
        uint64_t v37 = (uint64_t)v24;
        TASSIGN(v36, v37);
        // pto: %0
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v38 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v19);
        // pto: %0
        uint64_t v39 = (uint64_t)v23;
        TASSIGN(v38, v39);
        // pto: %42
        int64_t v40 = v28 < v17 ? v17 : v28;
        // pto: %inv_rms_buf_inline2585_inline11968__ssa_v0_pview
        pto::Shape<1, 1, 1, 16, 1> v41 = pto::Shape<1, 1, 1, 16, 1>();
        // pto: %inv_rms_buf_inline2585_inline11968__ssa_v0_pview
        pto::Stride<16, 16, 16, 1, 16> v42 = pto::Stride<16, 16, 16, 1, 16>();
        // pto: %inv_rms_buf_inline2585_inline11968__ssa_v0_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 16>, pto::Layout::DN> v43 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 16>, pto::Layout::DN>(
                v4 + (v17 + v40), v41, v42
            );
        TLOAD(v38, v43);
        // pto: %gate_logits_tile_inline2592_inline12022__rv_v2_Vec
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v44 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v13, v13);
        TPOP<
            TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>,
            Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>,
            TileSplitAxis::TILE_NO_SPLIT>(v26, v44);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %gate_logits_tile_v1_inline2568_inline12066__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v45 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %gate_logits_tile_v1_inline2568_inline12066__tile
        uint64_t v46 = (uint64_t)v22;
        TASSIGN(v45, v46);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TROWEXPANDMUL(v45, v44, v38);
        TFREE<TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>, TileSplitAxis::TILE_NO_SPLIT>(v26);
        // pto: %gp_relu_inline2577_inline11969__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v47 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %gp_relu_inline2577_inline11969__tile
        uint64_t v48 = (uint64_t)v21;
        TASSIGN(v47, v48);
        pipe_barrier(PIPE_V);
        TMAXS(v47, v45, v16);
        // pto: %1
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v49 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %1
        uint64_t v50 = (uint64_t)v20;
        TASSIGN(v49, v50);
        TNEG(v49, v45);
        // pto: %gp_abs_inline2545_inline11995__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v51 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %gp_abs_inline2545_inline11995__tile
        uint64_t v52 = (uint64_t)v20;
        TASSIGN(v51, v52);
        pipe_barrier(PIPE_V);
        TMAX(v51, v45, v49);
        // pto: %2
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v53 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %2
        uint64_t v54 = (uint64_t)v20;
        TASSIGN(v53, v54);
        pipe_barrier(PIPE_V);
        TNEG(v53, v51);
        // pto: %3
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v55 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %3
        uint64_t v56 = (uint64_t)v20;
        TASSIGN(v55, v56);
        pipe_barrier(PIPE_V);
        TEXP(v55, v53);
        // pto: %4
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v57 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %4
        uint64_t v58 = (uint64_t)v20;
        TASSIGN(v57, v58);
        pipe_barrier(PIPE_V);
        TADDS(v57, v55, v15);
        // pto: %5
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v59 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %5
        uint64_t v60 = (uint64_t)v20;
        TASSIGN(v59, v60);
        pipe_barrier(PIPE_V);
        TLOG(v59, v57);
        // pto: %gp_softplus_log_inline2547_inline11952__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v61 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %gp_softplus_log_inline2547_inline11952__tile
        uint64_t v62 = (uint64_t)v21;
        TASSIGN(v61, v62);
        pipe_barrier(PIPE_V);
        TADD(v61, v47, v59);
        // pto: %6
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v63 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %6
        uint64_t v64 = (uint64_t)v20;
        TASSIGN(v63, v64);
        pipe_barrier(PIPE_V);
        TNEG(v63, v45);
        // pto: %7
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v65 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %7
        uint64_t v66 = (uint64_t)v20;
        TASSIGN(v65, v66);
        pipe_barrier(PIPE_V);
        TSUBS(v65, v63, v14);
        // pto: %8
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v67 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %8
        uint64_t v68 = (uint64_t)v20;
        TASSIGN(v67, v68);
        pipe_barrier(PIPE_V);
        TMAXS(v67, v65, v16);
        // pto: %gp_neg_floor_mask_inline2544_inline11993__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v69 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %gp_neg_floor_mask_inline2544_inline11993__tile
        uint64_t v70 = (uint64_t)v20;
        TASSIGN(v69, v70);
        pipe_barrier(PIPE_V);
        TMINS(v69, v67, v15);
        // pto: %9
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v71 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %9
        uint64_t v72 = (uint64_t)v22;
        TASSIGN(v71, v72);
        TMINS(v71, v45, v16);
        // pto: %10
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v73 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %10
        uint64_t v74 = (uint64_t)v22;
        TASSIGN(v73, v74);
        pipe_barrier(PIPE_V);
        TEXP(v73, v71);
        // pto: %gp_neg_floor_inline2550_inline11893__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v75 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %gp_neg_floor_inline2550_inline11893__tile
        uint64_t v76 = (uint64_t)v22;
        TASSIGN(v75, v76);
        pipe_barrier(PIPE_V);
        TMUL(v75, v69, v73);
        // pto: %gp_softplus_inline2543_inline11983__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v77 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %gp_softplus_inline2543_inline11983__tile
        uint64_t v78 = (uint64_t)v22;
        TASSIGN(v77, v78);
        pipe_barrier(PIPE_V);
        TMAX(v77, v61, v75);
        // pto: %gp_score_inline2542_inline11904__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v79 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %gp_score_inline2542_inline11904__tile
        uint64_t v80 = (uint64_t)v22;
        TASSIGN(v79, v80);
        pipe_barrier(PIPE_V);
        TSQRT(v79, v77);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %route_scores_buf_inline2607_inline11959__ssa_v0_pview
        pto::Shape<1, 1, 1, 16, 16> v81 = pto::Shape<1, 1, 1, 16, 16>();
        // pto: %route_scores_buf_inline2607_inline11959__ssa_v0_pview
        pto::Stride<4096, 4096, 4096, 256, 1> v82 = pto::Stride<4096, 4096, 4096, 256, 1>();
        // pto: %route_scores_buf_inline2607_inline11959__ssa_v0_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v83 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(
                v5 + ((v17 + v40 * v12) + v32), v81, v82
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v83, v79);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        // pto: %11
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v84 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %11
        uint64_t v85 = (uint64_t)v21;
        TASSIGN(v84, v85);
        TEXPANDS(v84, v15);
        // pto: %gp_bias_inline2541_inline12017__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v86 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %gp_bias_inline2541_inline12017__tile
        uint64_t v87 = (uint64_t)v21;
        TASSIGN(v86, v87);
        pipe_barrier(PIPE_V);
        TCOLEXPANDMUL(v86, v84, v36);
        // pto: %gp_biased_inline2570_inline11953__tile
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v88 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v18);
        // pto: %gp_biased_inline2570_inline11953__tile
        uint64_t v89 = (uint64_t)v22;
        TASSIGN(v88, v89);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        TADD(v88, v79, v86);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        // pto: %biased_scores_buf_inline2564_inline12048__ssa_v1_pview
        pto::Shape<1, 1, 1, 16, 16> v90 = pto::Shape<1, 1, 1, 16, 16>();
        // pto: %biased_scores_buf_inline2564_inline12048__ssa_v1_pview
        pto::Stride<4096, 4096, 4096, 256, 1> v91 = pto::Stride<4096, 4096, 4096, 256, 1>();
        // pto: %biased_scores_buf_inline2564_inline12048__ssa_v1_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v92 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(
                v6 + ((v17 + v40 * v12) + v32), v90, v91
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSTORE(v92, v88);
    } else {
        // pto: %12
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v93 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v19, v17);
        // pto: %12
        uint64_t v94 = (uint64_t)v24;
        TASSIGN(v93, v94);
        // pto: %13
        Tile<
            TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v95 = Tile<
                TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %13
        uint64_t v96 = (uint64_t)v24;
        TASSIGN(v95, v96);
        // pto: %14
        Tile<
            TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v97 = Tile<
                TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %14
        uint64_t v98 = (uint64_t)v23;
        TASSIGN(v97, v98);
        // pto: %48
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v99 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v13, v13);
        v99.SetValidShape(v17, v17);
        TPOP<
            TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>,
            Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>,
            TileSplitAxis::TILE_NO_SPLIT>(v26, v99);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %15
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v100 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %15
        uint64_t v101 = (uint64_t)v22;
        TASSIGN(v100, v101);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TROWEXPANDMUL(v100, v99, v97);
        TFREE<TPipe<0, Direction::DIR_C2V, 1024, 8, 8, true>, TileSplitAxis::TILE_NO_SPLIT>(v26);
        // pto: %16
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v102 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %16
        uint64_t v103 = (uint64_t)v21;
        TASSIGN(v102, v103);
        pipe_barrier(PIPE_V);
        TMAXS(v102, v100, v16);
        // pto: %17
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v104 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %17
        uint64_t v105 = (uint64_t)v20;
        TASSIGN(v104, v105);
        TNEG(v104, v100);
        // pto: %18
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v106 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %18
        uint64_t v107 = (uint64_t)v20;
        TASSIGN(v106, v107);
        pipe_barrier(PIPE_V);
        TMAX(v106, v100, v104);
        // pto: %19
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v108 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %19
        uint64_t v109 = (uint64_t)v20;
        TASSIGN(v108, v109);
        pipe_barrier(PIPE_V);
        TNEG(v108, v106);
        // pto: %20
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v110 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %20
        uint64_t v111 = (uint64_t)v20;
        TASSIGN(v110, v111);
        pipe_barrier(PIPE_V);
        TEXP(v110, v108);
        // pto: %21
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v112 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %21
        uint64_t v113 = (uint64_t)v20;
        TASSIGN(v112, v113);
        pipe_barrier(PIPE_V);
        TADDS(v112, v110, v15);
        // pto: %22
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v114 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %22
        uint64_t v115 = (uint64_t)v20;
        TASSIGN(v114, v115);
        pipe_barrier(PIPE_V);
        TLOG(v114, v112);
        // pto: %23
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v116 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %23
        uint64_t v117 = (uint64_t)v21;
        TASSIGN(v116, v117);
        pipe_barrier(PIPE_V);
        TADD(v116, v102, v114);
        // pto: %24
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v118 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %24
        uint64_t v119 = (uint64_t)v20;
        TASSIGN(v118, v119);
        pipe_barrier(PIPE_V);
        TNEG(v118, v100);
        // pto: %25
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v120 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %25
        uint64_t v121 = (uint64_t)v20;
        TASSIGN(v120, v121);
        pipe_barrier(PIPE_V);
        TSUBS(v120, v118, v14);
        // pto: %26
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v122 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %26
        uint64_t v123 = (uint64_t)v20;
        TASSIGN(v122, v123);
        pipe_barrier(PIPE_V);
        TMAXS(v122, v120, v16);
        // pto: %27
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v124 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %27
        uint64_t v125 = (uint64_t)v20;
        TASSIGN(v124, v125);
        pipe_barrier(PIPE_V);
        TMINS(v124, v122, v15);
        // pto: %28
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v126 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %28
        uint64_t v127 = (uint64_t)v22;
        TASSIGN(v126, v127);
        TMINS(v126, v100, v16);
        // pto: %29
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v128 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %29
        uint64_t v129 = (uint64_t)v22;
        TASSIGN(v128, v129);
        pipe_barrier(PIPE_V);
        TEXP(v128, v126);
        // pto: %30
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v130 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %30
        uint64_t v131 = (uint64_t)v22;
        TASSIGN(v130, v131);
        pipe_barrier(PIPE_V);
        TMUL(v130, v124, v128);
        // pto: %31
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v132 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %31
        uint64_t v133 = (uint64_t)v22;
        TASSIGN(v132, v133);
        pipe_barrier(PIPE_V);
        TMAX(v132, v116, v130);
        // pto: %32
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v134 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %32
        uint64_t v135 = (uint64_t)v22;
        TASSIGN(v134, v135);
        pipe_barrier(PIPE_V);
        TSQRT(v134, v132);
        // pto: %33
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v136 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %33
        uint64_t v137 = (uint64_t)v21;
        TASSIGN(v136, v137);
        TEXPANDS(v136, v15);
        // pto: %34
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v138 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %34
        uint64_t v139 = (uint64_t)v21;
        TASSIGN(v138, v139);
        pipe_barrier(PIPE_V);
        TCOLEXPANDMUL(v138, v136, v95);
        // pto: %35
        Tile<
            TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v140 = Tile<
                TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v17);
        // pto: %35
        uint64_t v141 = (uint64_t)v22;
        TASSIGN(v140, v141);
        pipe_barrier(PIPE_V);
        TADD(v140, v134, v138);
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

    // Unpack tensor: gate_bias_hca_inline699__ssa_v0
    __gm__ Tensor *gate_bias_hca_inline699__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *gate_bias_hca_inline699__ssa_v0 =
        reinterpret_cast<__gm__ float *>(gate_bias_hca_inline699__ssa_v0_tensor->buffer.addr) +
        gate_bias_hca_inline699__ssa_v0_tensor->start_offset;

    // Unpack tensor: xg_buf_inline2582_inline12082__ssa_v0
    __gm__ Tensor *xg_buf_inline2582_inline12082__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *xg_buf_inline2582_inline12082__ssa_v0 =
        reinterpret_cast<__gm__ float *>(xg_buf_inline2582_inline12082__ssa_v0_tensor->buffer.addr) +
        xg_buf_inline2582_inline12082__ssa_v0_tensor->start_offset;

    // Unpack tensor: gate_w_hca_inline680__ssa_v0
    __gm__ Tensor *gate_w_hca_inline680__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *gate_w_hca_inline680__ssa_v0 =
        reinterpret_cast<__gm__ float *>(gate_w_hca_inline680__ssa_v0_tensor->buffer.addr) +
        gate_w_hca_inline680__ssa_v0_tensor->start_offset;

    // Unpack tensor: inv_rms_buf_inline2585_inline11968__ssa_v0
    __gm__ Tensor *inv_rms_buf_inline2585_inline11968__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ float *inv_rms_buf_inline2585_inline11968__ssa_v0 =
        reinterpret_cast<__gm__ float *>(inv_rms_buf_inline2585_inline11968__ssa_v0_tensor->buffer.addr) +
        inv_rms_buf_inline2585_inline11968__ssa_v0_tensor->start_offset;

    // Unpack tensor: route_scores_buf_inline2607_inline11959__ssa_v0
    __gm__ Tensor *route_scores_buf_inline2607_inline11959__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ float *route_scores_buf_inline2607_inline11959__ssa_v0 =
        reinterpret_cast<__gm__ float *>(route_scores_buf_inline2607_inline11959__ssa_v0_tensor->buffer.addr) +
        route_scores_buf_inline2607_inline11959__ssa_v0_tensor->start_offset;

    // Unpack tensor: biased_scores_buf_inline2564_inline12048__ssa_v1
    __gm__ Tensor *biased_scores_buf_inline2564_inline12048__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[5]);
    __gm__ float *biased_scores_buf_inline2564_inline12048__ssa_v1 =
        reinterpret_cast<__gm__ float *>(biased_scores_buf_inline2564_inline12048__ssa_v1_tensor->buffer.addr) +
        biased_scores_buf_inline2564_inline12048__ssa_v1_tensor->start_offset;

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

    // Unpack scalar: GATE_N_BLOCKS_inline2586_inline12037__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } GATE_N_BLOCKS_inline2586_inline12037__ssa_v0_conv;
    GATE_N_BLOCKS_inline2586_inline12037__ssa_v0_conv.u64 = args[7];
    int64_t GATE_N_BLOCKS_inline2586_inline12037__ssa_v0 = GATE_N_BLOCKS_inline2586_inline12037__ssa_v0_conv.val;

    // Forward to ptoas-generated function
    gate_2_aiv(
        gate_bias_hca_inline699__ssa_v0, xg_buf_inline2582_inline12082__ssa_v0, gate_w_hca_inline680__ssa_v0,
        inv_rms_buf_inline2585_inline11968__ssa_v0, route_scores_buf_inline2607_inline11959__ssa_v0,
        biased_scores_buf_inline2564_inline12048__ssa_v1, __gm_pipe_buffer,
        GATE_N_BLOCKS_inline2586_inline12037__ssa_v0, __pypto_spmd_block_idx, __pypto_spmd_block_num,
        __pypto_spmd_subblock_idx
    );
}
