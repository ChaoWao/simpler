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
// Kernel Function: route_hash

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

static __aicore__ void route_hash(
    __gm__ int64_t *v1, __gm__ int32_t *v2, __gm__ float *v3, __gm__ int32_t *v4, __gm__ float *v5, int64_t v6,
    int32_t v7, int32_t v8
) {
    const float v9 = 1.5f;
    const int64_t v10 = 128;
    const int64_t v11 = 256;
    const int64_t v12 = 6;
    const int64_t v13 = 1;
    const int64_t v14 = 8;
    const int64_t v15 = 288;
    const int64_t v16 = 0;
    const int64_t v17 = 32;
    const int64_t v18 = 8768;
    const int64_t v19 = 576;
    const int64_t v20 = 320;
    const int32_t v21 = 0;
    const int64_t v22 = 1024;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %th_idx_inline2540_inline9222__ssa_v0, %5
    int64_t v23 = (int64_t)((uint64_t)((int64_t)v7) * (uint64_t)v14);
    // pto: %hs_idx_tile_inline2593_inline9244__tile
    Tile<
        TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v24 = Tile<
            TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v14);
    // pto: %hs_idx_tile_inline2593_inline9244__tile
    uint64_t v25 = (uint64_t)v20;
    TASSIGN(v24, v25);
    for (int64_t i26 = v16; i26 < v14; i26 += v13) {
        // pto: %7, %6
        int64_t v27 = (v1)[(int64_t)((uint64_t)v23 + (uint64_t)i26)];
        for (int64_t j28 = v16; j28 < v12; j28 += v13) {
            // pto: %flat_offset_mul, %flat_offset, %9
            int32_t v29 = (v2)[(int64_t)((uint64_t)((int64_t)((uint64_t)v27 * (uint64_t)v12)) + (uint64_t)j28)];
            // pto: %10, %11
            int64_t v30 = (int64_t)((uint64_t)((int64_t)((uint64_t)i26 * (uint64_t)v14)) + (uint64_t)j28);
            v24.SetValue(v30, v29);
        }
        for (int64_t j31 = v12; j31 < v14; j31 += v13) {
            // pto: %13, %14
            int64_t v32 = (int64_t)((uint64_t)((int64_t)((uint64_t)i26 * (uint64_t)v14)) + (uint64_t)j31);
            v24.SetValue(v32, v21);
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    set_flag(PIPE_S, PIPE_V, EVENT_ID1);
    // pto: %local_scores_inline2591_inline9221__tile
    Tile<
        TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v33 = Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v11);
    // pto: %local_scores_inline2591_inline9221__tile
    uint64_t v34 = (uint64_t)v19;
    TASSIGN(v33, v34);
    // pto: %t__tile
    Tile<
        TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v35 = Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v11);
    // pto: %t__tile
    uint64_t v36 = (uint64_t)v18;
    TASSIGN(v35, v36);
    // pto: %route_scores_buf_inline2607_inline9273__ssa_v1_pview
    pto::Shape<1, 1, 1, 8, 256> v37 = pto::Shape<1, 1, 1, 8, 256>();
    // pto: %route_scores_buf_inline2607_inline9273__ssa_v1_pview
    pto::Stride<2048, 2048, 2048, 256, 1> v38 = pto::Stride<2048, 2048, 2048, 256, 1>();
    // pto: %15, %route_scores_buf_inline2607_inline9273__ssa_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<2048, 2048, 2048, 256, 1>, pto::Layout::ND> v39 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<2048, 2048, 2048, 256, 1>, pto::Layout::ND>(
            v3 + (v16 + (v23 < v16 ? v16 : v23) * v11), v37, v38
        );
    TLOAD(v35, v39);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %assemble_view
    Tile<
        TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v40;
    // pto: %assemble_view
    Tile<
        TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v41 = v40;
    // pto: %assemble_view
    uint64_t v42 = (uint64_t)v19;
    TASSIGN(v41, v42);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMOV(v41, v35);
    // pto: %gather_acc_init
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v43 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v14);
    // pto: %gather_acc_init
    uint64_t v44 = (uint64_t)v18;
    TASSIGN(v43, v44);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
    for (int64_t i45 = v16; i45 < v14; i45 += v13) {
        // pto: %gather_inp_row
        Tile<
            TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v46 = Tile<
                TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v13, v11);
        // pto: %gather_inp_row
        uint64_t v47 = (uint64_t)v19;
        TASSIGN(v46, v47);
        // pto: %slice_view
        Tile<
            TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v48;
        // pto: %slice_view
        Tile<
            TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v49 = v48;
        // pto: %slice_view
        uint64_t v50 = (uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)i45 * (uint64_t)v22)) + (uint64_t)v19));
        TASSIGN(v49, v50);
        // pto: %gather_idx_row
        Tile<
            TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v51 = Tile<
                TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v13, v14);
        // pto: %gather_idx_row
        uint64_t v52 = (uint64_t)v20;
        TASSIGN(v51, v52);
        // pto: %16
        int64_t v53 = (int64_t)((uint64_t)i45 * (uint64_t)v17);
        // pto: %16
        Tile<
            TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v54;
        // pto: %16
        Tile<
            TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v55 = v54;
        // pto: %16
        uint64_t v56 = (uint64_t)((int64_t)((uint64_t)v53 + (uint64_t)v20));
        TASSIGN(v55, v56);
        // pto: %gather_row_tmp
        Tile<
            TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v57 = Tile<
                TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v13, v14);
        // pto: %gather_row_tmp
        uint64_t v58 = (uint64_t)v17;
        TASSIGN(v57, v58);
        // pto: %gather_row
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v59 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v13, v14);
        // pto: %gather_row
        uint64_t v60 = (uint64_t)v16;
        TASSIGN(v59, v60);
        pipe_barrier(PIPE_V);
        TGATHER(v59, v49, v55, v57);
        // pto: %17
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v61;
        // pto: %17
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v62 = v61;
        // pto: %17
        uint64_t v63 = (uint64_t)((int64_t)((uint64_t)v53 + (uint64_t)v18));
        TASSIGN(v62, v63);
        pipe_barrier(PIPE_V);
        TMOV(v62, v59);
    }
    // pto: %gather_all_inline2535_inline9251__tile
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v64 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v14);
    // pto: %gather_all_inline2535_inline9251__tile
    uint64_t v65 = (uint64_t)v18;
    TASSIGN(v64, v65);
    v64.SetValidShape(v14, v12);
    // pto: %hs_vals_pad_inline2532_inline9453__tile
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v66 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v14, v14);
    // pto: %hs_vals_pad_inline2532_inline9453__tile
    uint64_t v67 = (uint64_t)v18;
    TASSIGN(v66, v67);
    pipe_barrier(PIPE_V);
    TFILLPAD(v66, v64);
    // pto: %hs_idx_read_inline2531_inline9200__tile
    Tile<
        TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v68 = Tile<
            TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v14);
    // pto: %hs_idx_read_inline2531_inline9200__tile
    uint64_t v69 = (uint64_t)v17;
    TASSIGN(v68, v69);
    // pto: %1
    Tile<
        TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v70 = Tile<
            TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v14);
    // pto: %1
    uint64_t v71 = (uint64_t)v20;
    TASSIGN(v70, v71);
    // pto: %18
    Tile<
        TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v72;
    // pto: %18
    Tile<
        TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v73 = v72;
    // pto: %18
    uint64_t v74 = (uint64_t)v20;
    TASSIGN(v73, v74);
    // pto: %19
    Tile<
        TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v75;
    // pto: %19
    Tile<
        TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v76 = v75;
    // pto: %19
    uint64_t v77 = (uint64_t)v17;
    TASSIGN(v76, v77);
    TMOV(v76, v73);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    // pto: %tmp_tile
    Tile<
        TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v78 = Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v10);
    // pto: %tmp_tile
    uint64_t v79 = (uint64_t)v19;
    TASSIGN(v78, v79);
    // pto: %3
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v80 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v13);
    // pto: %3
    uint64_t v81 = (uint64_t)v15;
    TASSIGN(v80, v81);
    pipe_barrier(PIPE_V);
    TROWSUM(v80, v66, v78);
    // pto: %hs_denom_inline2530_inline9449__tile
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v82 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v13);
    // pto: %hs_denom_inline2530_inline9449__tile
    uint64_t v83 = (uint64_t)v15;
    TASSIGN(v82, v83);
    // pto: %4
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v84 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v14, v14);
    // pto: %4
    uint64_t v85 = (uint64_t)v19;
    TASSIGN(v84, v85);
    pipe_barrier(PIPE_V);
    TROWEXPANDDIV(v84, v66, v82);
    // pto: %hs_weights_pad_inline2529_inline9198__tile
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v86 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v14, v14);
    // pto: %hs_weights_pad_inline2529_inline9198__tile
    uint64_t v87 = (uint64_t)v19;
    TASSIGN(v86, v87);
    pipe_barrier(PIPE_V);
    TMULS(v86, v84, v9);
    set_flag(PIPE_V, PIPE_S, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
    for (int64_t i88 = v16; i88 < v14; i88 += v13) {
        // pto: %20
        int64_t v89 = (int64_t)((uint64_t)v23 + (uint64_t)i88);
        // pto: %21
        if (v89 < v6) {
            for (int64_t j90 = v16; j90 < v12; j90 += v13) {
                // pto: %23, %24
                int64_t v91 = (int64_t)((uint64_t)((int64_t)((uint64_t)i88 * (uint64_t)v14)) + (uint64_t)j90);
                // pto: %22
                int32_t v92 = v68.GetValue(v91);
                // pto: %26, %27
                int64_t v93 = (int64_t)((uint64_t)((int64_t)((uint64_t)v89 * (uint64_t)v12)) + (uint64_t)j90);
                (v4)[v93] = v92;
                // pto: %28
                float v94 = v86.GetValue(v91);
                (v5)[v93] = v94;
            }
        }
    }
    pipe_barrier(PIPE_ALL);
    dcci((__gm__ void *)0, cache_line_t::ENTIRE_DATA_CACHE);
    dsb((mem_dsb_t)0);
#endif  // __DAV_VEC__

    pipe_barrier(PIPE_ALL);
    dcci((__gm__ void *)0, cache_line_t::ENTIRE_DATA_CACHE);
    dsb((mem_dsb_t)0);
    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: input_ids__ssa_v0
    __gm__ Tensor *input_ids__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int64_t *input_ids__ssa_v0 = reinterpret_cast<__gm__ int64_t *>(input_ids__ssa_v0_tensor->buffer.addr) +
                                        input_ids__ssa_v0_tensor->start_offset;

    // Unpack tensor: tid2eid_l0_inline607__ssa_v0
    __gm__ Tensor *tid2eid_l0_inline607__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int32_t *tid2eid_l0_inline607__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(tid2eid_l0_inline607__ssa_v0_tensor->buffer.addr) +
        tid2eid_l0_inline607__ssa_v0_tensor->start_offset;

    // Unpack tensor: route_scores_buf_inline2607_inline9273__ssa_v1
    __gm__ Tensor *route_scores_buf_inline2607_inline9273__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *route_scores_buf_inline2607_inline9273__ssa_v1 =
        reinterpret_cast<__gm__ float *>(route_scores_buf_inline2607_inline9273__ssa_v1_tensor->buffer.addr) +
        route_scores_buf_inline2607_inline9273__ssa_v1_tensor->start_offset;

    // Unpack tensor: indices_inline9286__ssa_v0
    __gm__ Tensor *indices_inline9286__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ int32_t *indices_inline9286__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(indices_inline9286__ssa_v0_tensor->buffer.addr) +
        indices_inline9286__ssa_v0_tensor->start_offset;

    // Unpack tensor: weights_inline9276__ssa_v0
    __gm__ Tensor *weights_inline9276__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ float *weights_inline9276__ssa_v0 =
        reinterpret_cast<__gm__ float *>(weights_inline9276__ssa_v0_tensor->buffer.addr) +
        weights_inline9276__ssa_v0_tensor->start_offset;

    // Unpack scalar: active_tokens_inline2578_inline9235__phi_v4
    union {
        uint64_t u64;
        int64_t val;
    } active_tokens_inline2578_inline9235__phi_v4_conv;
    active_tokens_inline2578_inline9235__phi_v4_conv.u64 = args[5];
    int64_t active_tokens_inline2578_inline9235__phi_v4 = active_tokens_inline2578_inline9235__phi_v4_conv.val;

    // Forward to ptoas-generated function
    route_hash(
        input_ids__ssa_v0, tid2eid_l0_inline607__ssa_v0, route_scores_buf_inline2607_inline9273__ssa_v1,
        indices_inline9286__ssa_v0, weights_inline9276__ssa_v0, active_tokens_inline2578_inline9235__phi_v4,
        __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
