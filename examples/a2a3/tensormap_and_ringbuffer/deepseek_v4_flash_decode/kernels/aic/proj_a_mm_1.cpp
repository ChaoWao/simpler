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
// Kernel Function: proj_a_mm_1

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

static __aicore__ void proj_a_mm_1(
    __gm__ bfloat16_t *v1, __gm__ bfloat16_t *v2, __gm__ float *v3, int64_t v4, int64_t v5, int64_t v6, int32_t v7,
    int32_t v8
) {
    const int64_t v9 = 3840;
    const int64_t v10 = 2;
    const int64_t v11 = 15;
    const int64_t v12 = 256;
    const int64_t v13 = 128;
    const int64_t v14 = 16;
    const int64_t v15 = 1024;
    const int64_t v16 = 8;
    const int64_t v17 = 1;
    const int64_t v18 = 4096;
    const int64_t v19 = 32768;
    const int64_t v20 = 12288;
    const int64_t v21 = 8192;
    const int64_t v22 = 0;
    const int64_t v23 = 81920;
    const int64_t v24 = 16384;
    using T = float;

#if defined(__DAV_CUBE__)
    // pto: %nf_inline2131_inline10897__ssa_v0, %23
    int64_t v25 = (int64_t)((uint64_t)((int64_t)v7) * (uint64_t)v13);
    // pto: %xa0_chunk_inline2076_inline10268__tile
    Tile<
        TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v26 = Tile<
            TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v16, v12);
    // pto: %xa0_chunk_inline2076_inline10268__tile
    uint64_t v27 = (uint64_t)v24;
    TASSIGN(v26, v27);
    // pto: %24
    int64_t v28 = v4 < v22 ? v22 : v4;
    // pto: %o_packed_inline2242_inline10294__ssa_v16_pview
    pto::Shape<1, 1, 1, 8, 256> v29 = pto::Shape<1, 1, 1, 8, 256>();
    // pto: %o_packed_inline2242_inline10294__ssa_v16_pview
    pto::Stride<32768, 32768, 32768, 4096, 1> v30 = pto::Stride<32768, 32768, 32768, 4096, 1>();
    // pto: %o_packed_inline2242_inline10294__ssa_v16_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>
        v31 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(
            v1 + (v22 + v28 * v18), v29, v30
        );
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    TLOAD(v26, v31);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // pto: %wa0_chunk_inline2075_inline10326__tile
    Tile<
        TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v32 = Tile<
            TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v13, v12);
    // pto: %wa0_chunk_inline2075_inline10326__tile
    uint64_t v33 = (uint64_t)v23;
    TASSIGN(v32, v33);
    // pto: %25, %26
    int64_t v34 = (int64_t)((uint64_t)((int64_t)((uint64_t)v5 * (uint64_t)v15)) + (uint64_t)v25);
    // pto: %27
    int64_t v35 = v34 < v22 ? v22 : v34;
    // pto: %wa0_chunk_inline2075_inline10326__tile_view2d_pview
    pto::Shape<1, 1, 1, 128, 256> v36 = pto::Shape<1, 1, 1, 128, 256>();
    // pto: %wa0_chunk_inline2075_inline10326__tile_view2d_pview
    pto::Stride<524288, 524288, 524288, 4096, 1> v37 = pto::Stride<524288, 524288, 524288, 4096, 1>();
    // pto: %wa0_chunk_inline2075_inline10326__tile_view2d_pview
    GlobalTensor<
        bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>, pto::Layout::ND>
        v38 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>, pto::Layout::ND>(
            v2 + (v22 + v35 * v18), v36, v37
        );
    TLOAD(v32, v38);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    // pto: %wa0_chunk_inline2075_inline10326__tile_t
    Tile<
        TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v39 = Tile<
            TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v12, v13);
    // pto: %wa0_chunk_inline2075_inline10326__tile_t
    uint64_t v40 = (uint64_t)v23;
    TASSIGN(v39, v40);
    // pto: %acc_a_inline2080_inline10267__tile_l0_init_storage
    Tile<
        TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v41 = Tile<
            TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v14, v13);
    // pto: %acc_a_inline2080_inline10267__tile_l0_init_storage
    uint64_t v42 = (uint64_t)v22;
    TASSIGN(v41, v42);
    v41.SetValidShape(v16, v13);
    // pto: %acc_a_inline2080_inline10267__tile_l0_a
    Tile<
        TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Normal>
        v43 = Tile<
            TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Normal>(v16, v13);
    // pto: %acc_a_inline2080_inline10267__tile_l0_a
    uint64_t v44 = (uint64_t)v21;
    TASSIGN(v43, v44);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TEXTRACT(v43, v26, v22, v22);
    // pto: %acc_a_inline2080_inline10267__tile_l0_b
    Tile<
        TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v45 = Tile<
            TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v13, v13);
    // pto: %acc_a_inline2080_inline10267__tile_l0_b
    uint64_t v46 = (uint64_t)v22;
    TASSIGN(v45, v46);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TEXTRACT(v45, v39, v22, v22);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    // pto: %0
    Tile<
        TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Normal>
        v47 = Tile<
            TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Normal>(v16, v13);
    // pto: %0
    uint64_t v48 = (uint64_t)v20;
    TASSIGN(v47, v48);
    TEXTRACT(v47, v26, v22, v13);
    // pto: %1
    Tile<
        TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v49 = Tile<
            TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v13, v13);
    // pto: %1
    uint64_t v50 = (uint64_t)v19;
    TASSIGN(v49, v50);
    TEXTRACT(v49, v39, v13, v22);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    // pto: %acc_a_inline2080_inline10267__tile_l0_c_first
    Tile<
        TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v51 = Tile<
            TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v16, v13);
    // pto: %acc_a_inline2080_inline10267__tile_l0_c_first
    uint64_t v52 = (uint64_t)v22;
    TASSIGN(v51, v52);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(v51, v43, v45);
    // pto: %acc_a_inline2080_inline10267__tile_l0_c_acc
    Tile<
        TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v53 = Tile<
            TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v16, v13);
    // pto: %acc_a_inline2080_inline10267__tile_l0_c_acc
    uint64_t v54 = (uint64_t)v22;
    TASSIGN(v53, v54);
    pipe_barrier(PIPE_M);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    TMATMUL_ACC(v53, v53, v47, v49);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    for (int64_t i55 = v17; i55 < v11; i55 += v10) {
        // pto: %28
        int64_t v56 = (int64_t)((uint64_t)i55 * (uint64_t)v12);
        // pto: %30
        int64_t v57 = (int64_t)((uint64_t)v56 + (uint64_t)v12);
        // pto: %xa_k_chunk_inline2252_inline10265__tile
        Tile<
            TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v58 = Tile<
                TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v16, v12);
        // pto: %xa_k_chunk_inline2252_inline10265__tile
        uint64_t v59 = (uint64_t)v22;
        TASSIGN(v58, v59);
        // pto: %32
        int64_t v60 = v56 < v22 ? v22 : v56;
        // pto: %33
        pto::Shape<1, 1, 1, 8, 256> v61 = pto::Shape<1, 1, 1, 8, 256>();
        // pto: %33
        pto::Stride<32768, 32768, 32768, 4096, 1> v62 = pto::Stride<32768, 32768, 32768, 4096, 1>();
        // pto: %33
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>
            v63 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(
                v1 + ((v22 + v28 * v18) + v60), v61, v62
            );
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        TLOAD(v58, v63);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        // pto: %2
        Tile<
            TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v64 = Tile<
                TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v16, v12);
        // pto: %2
        uint64_t v65 = (uint64_t)v21;
        TASSIGN(v64, v65);
        // pto: %35
        int64_t v66 = v57 < v22 ? v22 : v57;
        // pto: %36
        pto::Shape<1, 1, 1, 8, 256> v67 = pto::Shape<1, 1, 1, 8, 256>();
        // pto: %36
        pto::Stride<32768, 32768, 32768, 4096, 1> v68 = pto::Stride<32768, 32768, 32768, 4096, 1>();
        // pto: %36
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>
            v69 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(
                v1 + ((v22 + v28 * v18) + v66), v67, v68
            );
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        TLOAD(v64, v69);
        // pto: %wa_k_chunk_inline2072_inline10908__tile
        Tile<
            TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v70 = Tile<
                TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v13, v12);
        // pto: %wa_k_chunk_inline2072_inline10908__tile
        uint64_t v71 = (uint64_t)v23;
        TASSIGN(v70, v71);
        // pto: %wa_k_chunk_inline2072_inline10908__tile_view2d_pview
        pto::Shape<1, 1, 1, 128, 256> v72 = pto::Shape<1, 1, 1, 128, 256>();
        // pto: %wa_k_chunk_inline2072_inline10908__tile_view2d_pview
        pto::Stride<524288, 524288, 524288, 4096, 1> v73 = pto::Stride<524288, 524288, 524288, 4096, 1>();
        // pto: %wa_k_chunk_inline2072_inline10908__tile_view2d_pview
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>, pto::Layout::ND>
            v74 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>,
                pto::Layout::ND>(v2 + ((v22 + v35 * v18) + v60), v72, v73);
        TLOAD(v70, v74);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        // pto: %3
        Tile<
            TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v75 = Tile<
                TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v13, v12);
        // pto: %3
        uint64_t v76 = (uint64_t)v24;
        TASSIGN(v75, v76);
        // pto: %46
        pto::Shape<1, 1, 1, 128, 256> v77 = pto::Shape<1, 1, 1, 128, 256>();
        // pto: %46
        pto::Stride<524288, 524288, 524288, 4096, 1> v78 = pto::Stride<524288, 524288, 524288, 4096, 1>();
        // pto: %46
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>, pto::Layout::ND>
            v79 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>,
                pto::Layout::ND>(v2 + ((v22 + v35 * v18) + v66), v77, v78);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        TLOAD(v75, v79);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
        // pto: %wa_k_chunk_inline2072_inline10908__tile_t
        Tile<
            TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v80 = Tile<
                TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v12, v13);
        // pto: %wa_k_chunk_inline2072_inline10908__tile_t
        uint64_t v81 = (uint64_t)v23;
        TASSIGN(v80, v81);
        // pto: %acc_a_inline2080_inline10267__iter_v1_l0_a
        Tile<
            TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Normal>
            v82 = Tile<
                TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Normal>(v16, v13);
        // pto: %acc_a_inline2080_inline10267__iter_v1_l0_a
        uint64_t v83 = (uint64_t)v21;
        TASSIGN(v82, v83);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        TEXTRACT(v82, v58, v22, v22);
        // pto: %acc_a_inline2080_inline10267__iter_v1_l0_b
        Tile<
            TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v84 = Tile<
                TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>(v13, v13);
        // pto: %acc_a_inline2080_inline10267__iter_v1_l0_b
        uint64_t v85 = (uint64_t)v22;
        TASSIGN(v84, v85);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v84, v80, v22, v22);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        // pto: %4
        Tile<
            TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Normal>
            v86 = Tile<
                TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Normal>(v16, v13);
        // pto: %4
        uint64_t v87 = (uint64_t)v20;
        TASSIGN(v86, v87);
        TEXTRACT(v86, v58, v22, v13);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        // pto: %5
        Tile<
            TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v88 = Tile<
                TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>(v13, v13);
        // pto: %5
        uint64_t v89 = (uint64_t)v19;
        TASSIGN(v88, v89);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        TEXTRACT(v88, v80, v13, v22);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        // pto: %acc_a_inline2080_inline10267__iter_v1_l0_c_acc
        Tile<
            TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v90 = Tile<
                TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v16, v13);
        // pto: %acc_a_inline2080_inline10267__iter_v1_l0_c_acc
        uint64_t v91 = (uint64_t)v22;
        TASSIGN(v90, v91);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v90, v90, v82, v84);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        // pto: %6
        Tile<
            TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v92 = Tile<
                TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v16, v13);
        // pto: %6
        uint64_t v93 = (uint64_t)v22;
        TASSIGN(v92, v93);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        TMATMUL_ACC(v92, v92, v86, v88);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        // pto: %7
        Tile<
            TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v94 = Tile<
                TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v12, v13);
        // pto: %7
        uint64_t v95 = (uint64_t)v24;
        TASSIGN(v94, v95);
        // pto: %8
        Tile<
            TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Normal>
            v96 = Tile<
                TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Normal>(v16, v13);
        // pto: %8
        uint64_t v97 = (uint64_t)v22;
        TASSIGN(v96, v97);
        TEXTRACT(v96, v64, v22, v22);
        // pto: %9
        Tile<
            TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v98 = Tile<
                TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>(v13, v13);
        // pto: %9
        uint64_t v99 = (uint64_t)v22;
        TASSIGN(v98, v99);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
        TEXTRACT(v98, v94, v22, v22);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
        // pto: %10
        Tile<
            TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Normal>
            v100 = Tile<
                TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Normal>(v16, v13);
        // pto: %10
        uint64_t v101 = (uint64_t)v18;
        TASSIGN(v100, v101);
        TEXTRACT(v100, v64, v22, v13);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        // pto: %11
        Tile<
            TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v102 = Tile<
                TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>(v13, v13);
        // pto: %11
        uint64_t v103 = (uint64_t)v19;
        TASSIGN(v102, v103);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        TEXTRACT(v102, v94, v13, v22);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
        // pto: %12
        Tile<
            TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v104 = Tile<
                TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v16, v13);
        // pto: %12
        uint64_t v105 = (uint64_t)v22;
        TASSIGN(v104, v105);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v104, v104, v96, v98);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        // pto: %13
        Tile<
            TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v106 = Tile<
                TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v16, v13);
        // pto: %13
        uint64_t v107 = (uint64_t)v22;
        TASSIGN(v106, v107);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
        TMATMUL_ACC(v106, v106, v100, v102);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    }
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    // pto: %14
    Tile<
        TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v108 = Tile<
            TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v16, v12);
    // pto: %14
    uint64_t v109 = (uint64_t)v24;
    TASSIGN(v108, v109);
    // pto: %48
    pto::Shape<1, 1, 1, 8, 256> v110 = pto::Shape<1, 1, 1, 8, 256>();
    // pto: %48
    pto::Stride<32768, 32768, 32768, 4096, 1> v111 = pto::Stride<32768, 32768, 32768, 4096, 1>();
    // pto: %48
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>
        v112 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(
            v1 + (v9 + v28 * v18), v110, v111
        );
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    TLOAD(v108, v112);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
    // pto: %15
    Tile<
        TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v113 = Tile<
            TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v13, v12);
    // pto: %15
    uint64_t v114 = (uint64_t)v23;
    TASSIGN(v113, v114);
    // pto: %53
    pto::Shape<1, 1, 1, 128, 256> v115 = pto::Shape<1, 1, 1, 128, 256>();
    // pto: %53
    pto::Stride<524288, 524288, 524288, 4096, 1> v116 = pto::Stride<524288, 524288, 524288, 4096, 1>();
    // pto: %53
    GlobalTensor<
        bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>, pto::Layout::ND>
        v117 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>, pto::Layout::ND>(
            v2 + (v9 + v35 * v18), v115, v116
        );
    TLOAD(v113, v117);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
    // pto: %16
    Tile<
        TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v118 = Tile<
            TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v12, v13);
    // pto: %16
    uint64_t v119 = (uint64_t)v23;
    TASSIGN(v118, v119);
    // pto: %17
    Tile<
        TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Normal>
        v120 = Tile<
            TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Normal>(v16, v13);
    // pto: %17
    uint64_t v121 = (uint64_t)v21;
    TASSIGN(v120, v121);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
    TEXTRACT(v120, v108, v22, v22);
    // pto: %18
    Tile<
        TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v122 = Tile<
            TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v13, v13);
    // pto: %18
    uint64_t v123 = (uint64_t)v22;
    TASSIGN(v122, v123);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    TEXTRACT(v122, v118, v22, v22);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
    // pto: %19
    Tile<
        TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Normal>
        v124 = Tile<
            TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Normal>(v16, v13);
    // pto: %19
    uint64_t v125 = (uint64_t)v20;
    TASSIGN(v124, v125);
    TEXTRACT(v124, v108, v22, v13);
    // pto: %20
    Tile<
        TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v126 = Tile<
            TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v13, v13);
    // pto: %20
    uint64_t v127 = (uint64_t)v19;
    TASSIGN(v126, v127);
    TEXTRACT(v126, v118, v13, v22);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
    // pto: %21
    Tile<
        TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v128 = Tile<
            TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v16, v13);
    // pto: %21
    uint64_t v129 = (uint64_t)v22;
    TASSIGN(v128, v129);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
    pipe_barrier(PIPE_M);
    TMATMUL_ACC(v128, v128, v120, v122);
    // pto: %22
    Tile<
        TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v130 = Tile<
            TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v16, v13);
    // pto: %22
    uint64_t v131 = (uint64_t)v22;
    TASSIGN(v130, v131);
    pipe_barrier(PIPE_M);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
    TMATMUL_ACC(v130, v130, v124, v126);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    // pto: %acc_a_inline2080_inline10267__rv_v2
    Tile<
        TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v132 = Tile<
            TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v16, v13);
    // pto: %acc_a_inline2080_inline10267__rv_v2
    uint64_t v133 = (uint64_t)v22;
    TASSIGN(v132, v133);
    // pto: %54
    int64_t v134 = (int64_t)((uint64_t)v6 + (uint64_t)v25);
    // pto: %o_r_pad_inline2225_inline10832__iter_v1_pview
    pto::Shape<1, 1, 1, 8, 128> v135 = pto::Shape<1, 1, 1, 8, 128>();
    // pto: %o_r_pad_inline2225_inline10832__iter_v1_pview
    pto::Stride<65536, 65536, 65536, 8192, 1> v136 = pto::Stride<65536, 65536, 65536, 8192, 1>();
    // pto: %55, %o_r_pad_inline2225_inline10832__iter_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND> v137 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND>(
            v3 + (v22 + (v134 < v22 ? v22 : v134)), v135, v136
        );
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(v137, v132);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
#endif  // __DAV_CUBE__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: o_packed_inline2242_inline10294__ssa_v16
    __gm__ Tensor *o_packed_inline2242_inline10294__ssa_v16_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ bfloat16_t *o_packed_inline2242_inline10294__ssa_v16 =
        reinterpret_cast<__gm__ bfloat16_t *>(o_packed_inline2242_inline10294__ssa_v16_tensor->buffer.addr) +
        o_packed_inline2242_inline10294__ssa_v16_tensor->start_offset;

    // Unpack tensor: wo_a_csa_inline672__ssa_v0
    __gm__ Tensor *wo_a_csa_inline672__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ bfloat16_t *wo_a_csa_inline672__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(wo_a_csa_inline672__ssa_v0_tensor->buffer.addr) +
        wo_a_csa_inline672__ssa_v0_tensor->start_offset;

    // Unpack tensor: o_r_pad_inline2225_inline10832__iter_v1
    __gm__ Tensor *o_r_pad_inline2225_inline10832__iter_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *o_r_pad_inline2225_inline10832__iter_v1 =
        reinterpret_cast<__gm__ float *>(o_r_pad_inline2225_inline10832__iter_v1_tensor->buffer.addr) +
        o_r_pad_inline2225_inline10832__iter_v1_tensor->start_offset;

    // Unpack scalar: row_base_o_inline2078_inline10745__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } row_base_o_inline2078_inline10745__ssa_v0_conv;
    row_base_o_inline2078_inline10745__ssa_v0_conv.u64 = args[3];
    int64_t row_base_o_inline2078_inline10745__ssa_v0 = row_base_o_inline2078_inline10745__ssa_v0_conv.val;

    // Unpack scalar: g_inline2162_inline10271__idx_v0
    union {
        uint64_t u64;
        int64_t val;
    } g_inline2162_inline10271__idx_v0_conv;
    g_inline2162_inline10271__idx_v0_conv.u64 = args[4];
    int64_t g_inline2162_inline10271__idx_v0 = g_inline2162_inline10271__idx_v0_conv.val;

    // Unpack scalar: out_col_g_inline2181_inline10439__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } out_col_g_inline2181_inline10439__ssa_v0_conv;
    out_col_g_inline2181_inline10439__ssa_v0_conv.u64 = args[5];
    int64_t out_col_g_inline2181_inline10439__ssa_v0 = out_col_g_inline2181_inline10439__ssa_v0_conv.val;

    // Forward to ptoas-generated function
    proj_a_mm_1(
        o_packed_inline2242_inline10294__ssa_v16, wo_a_csa_inline672__ssa_v0, o_r_pad_inline2225_inline10832__iter_v1,
        row_base_o_inline2078_inline10745__ssa_v0, g_inline2162_inline10271__idx_v0,
        out_col_g_inline2181_inline10439__ssa_v0, __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
