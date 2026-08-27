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
// Kernel Function: hc_head_rms

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

static __aicore__ void hc_head_rms(__gm__ float *v1, __gm__ float *v2, int64_t v3, int32_t v4, int32_t v5) {
    const int64_t v6 = 512;
    const float v7 = 0.0f;
    const int64_t v8 = 1024;
    const int64_t v9 = 8;
    const int64_t v10 = 16;
    const int64_t v11 = 1;
    const int64_t v12 = 16384;
    const int64_t v13 = 0;
    const int64_t v14 = 49184;
    const int64_t v15 = 32800;
    const int64_t v16 = 16416;
    const int64_t v17 = 49216;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %sq_part_inline13219__ssa_v0_view
    int64_t v18 = v10 * v3;
    // pto: %sq_part_inline13219__ssa_v0_view
    int64_t v19 = v11 * v18;
    // pto: %sq_part_inline13219__ssa_v0_view
    pto::Shape<1, 1, 1, -1, -1> v20 = pto::Shape<1, 1, 1, -1, -1>(v11, v11, v11, v10, v3);
    // pto: %sq_part_inline13219__ssa_v0_view
    pto::Stride<-1, -1, -1, -1, -1> v21 = pto::Stride<-1, -1, -1, -1, -1>(v11 * v19, v19, v18, v3, v11);
    // pto: %sq_part_inline13219__ssa_v0_view
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v22 =
        GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(
            v2, v20, v21
        );
    // pto: %task_inline13217__ssa_v0
    int64_t v23 = (int64_t)v4;
    // pto: %8, %9
    int64_t v24 = (int64_t)((uint64_t)(v23 / v10) * (uint64_t)v9);
    // pto: %10
    int64_t v25 = v23 % v10;
    // pto: %11
    int64_t v26 = (int64_t)((uint64_t)v25 * (uint64_t)v8);
    // pto: %sq_sum_inline13213__tile
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v27 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v11, v9);
    // pto: %sq_sum_inline13213__tile
    uint64_t v28 = (uint64_t)v17;
    TASSIGN(v27, v28);
    TEXPANDS(v27, v7);
    // pto: %x_rms_inline13224__tile
    Tile<
        TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v29 = Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v9, v6);
    // pto: %x_rms_inline13224__tile
    uint64_t v30 = (uint64_t)v16;
    TASSIGN(v29, v30);
    // pto: %12
    int64_t v31 = v24 < v13 ? v13 : v24;
    // pto: %x_flat_inline13229__ssa_v0_pview
    pto::Shape<1, 1, 1, 8, 512> v32 = pto::Shape<1, 1, 1, 8, 512>();
    // pto: %x_flat_inline13229__ssa_v0_pview
    pto::Stride<131072, 131072, 131072, 16384, 1> v33 = pto::Stride<131072, 131072, 131072, 16384, 1>();
    // pto: %x_flat_inline13229__ssa_v0_pview, %13
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>
        v34 = GlobalTensor<
            float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(
            v1 + ((v13 + v31 * v12) + (v26 < v13 ? v13 : v26)), v32, v33
        );
    TLOAD(v29, v34);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %t__tile
    Tile<
        TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v35 = Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v9, v6);
    // pto: %t__tile
    uint64_t v36 = (uint64_t)v16;
    TASSIGN(v35, v36);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMUL(v35, v29, v29);
    // pto: %tmp_tile
    Tile<
        TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v37 = Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v9, v6);
    // pto: %tmp_tile
    uint64_t v38 = (uint64_t)v15;
    TASSIGN(v37, v38);
    // pto: %sq_col_inline13232__tile
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v39 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v9, v11);
    // pto: %sq_col_inline13232__tile
    uint64_t v40 = (uint64_t)v14;
    TASSIGN(v39, v40);
    pipe_barrier(PIPE_V);
    TROWSUM(v39, v35, v37);
    // pto: %0
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v41 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v11, v9);
    // pto: %0
    uint64_t v42 = (uint64_t)v14;
    TASSIGN(v41, v42);
    // pto: %1
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v43 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v11, v9);
    // pto: %1
    uint64_t v44 = (uint64_t)v16;
    TASSIGN(v43, v44);
    pipe_barrier(PIPE_V);
    TADD(v43, v27, v41);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    // pto: %14
    int64_t v45 = (int64_t)((uint64_t)v26 + (uint64_t)v6);
    // pto: %2
    Tile<
        TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v46 = Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v9, v6);
    // pto: %2
    uint64_t v47 = (uint64_t)v17;
    TASSIGN(v46, v47);
    // pto: %17
    pto::Shape<1, 1, 1, 8, 512> v48 = pto::Shape<1, 1, 1, 8, 512>();
    // pto: %17
    pto::Stride<131072, 131072, 131072, 16384, 1> v49 = pto::Stride<131072, 131072, 131072, 16384, 1>();
    // pto: %17, %16
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>
        v50 = GlobalTensor<
            float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(
            v1 + ((v13 + v31 * v12) + (v45 < v13 ? v13 : v45)), v48, v49
        );
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v46, v50);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %3
    Tile<
        TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v51 = Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v9, v6);
    // pto: %3
    uint64_t v52 = (uint64_t)v17;
    TASSIGN(v51, v52);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TMUL(v51, v46, v46);
    // pto: %4
    Tile<
        TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v53 = Tile<
            TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v9, v6);
    // pto: %4
    uint64_t v54 = (uint64_t)v13;
    TASSIGN(v53, v54);
    // pto: %5
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v55 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v9, v11);
    // pto: %5
    uint64_t v56 = (uint64_t)v12;
    TASSIGN(v55, v56);
    pipe_barrier(PIPE_V);
    TROWSUM(v55, v51, v53);
    // pto: %6
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v57 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v11, v9);
    // pto: %6
    uint64_t v58 = (uint64_t)v12;
    TASSIGN(v57, v58);
    // pto: %7
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v59 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v11, v9);
    // pto: %7
    uint64_t v60 = (uint64_t)v17;
    TASSIGN(v59, v60);
    pipe_barrier(PIPE_V);
    TADD(v59, v43, v57);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %sq_sum_inline13213__rv_v2
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v61 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v11, v9);
    // pto: %sq_sum_inline13213__rv_v2
    uint64_t v62 = (uint64_t)v17;
    TASSIGN(v61, v62);
    // pto: %sq_part_inline13219__ssa_v0_pview
    __gm__ float *v63 = PTOAS__GLOBAL_TENSOR_DATA(v22);
    // pto: %sq_part_inline13219__ssa_v0_pview
    int64_t v64 = v11 * v3;
    // pto: %sq_part_inline13219__ssa_v0_pview
    int64_t v65 = v11 * v64;
    // pto: %sq_part_inline13219__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 8> v66 = pto::Shape<1, 1, 1, 1, 8>(v11, v11, v11, v11, v9);
    // pto: %sq_part_inline13219__ssa_v0_pview
    pto::Stride<-1, -1, -1, -1, -1> v67 = pto::Stride<-1, -1, -1, -1, -1>(v11 * v65, v65, v64, v3, v11);
    // pto: %18, %sq_part_inline13219__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v68 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(
            v63 + ((v13 + (v25 < v13 ? v13 : v25) * v3) + v31 * v11), v66, v67
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v68, v61);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: x_flat_inline13229__ssa_v0
    __gm__ Tensor *x_flat_inline13229__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *x_flat_inline13229__ssa_v0 =
        reinterpret_cast<__gm__ float *>(x_flat_inline13229__ssa_v0_tensor->buffer.addr) +
        x_flat_inline13229__ssa_v0_tensor->start_offset;

    // Unpack tensor: sq_part_inline13219__ssa_v0
    __gm__ Tensor *sq_part_inline13219__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *sq_part_inline13219__ssa_v0 =
        reinterpret_cast<__gm__ float *>(sq_part_inline13219__ssa_v0_tensor->buffer.addr) +
        sq_part_inline13219__ssa_v0_tensor->start_offset;

    // Extract dynamic dim: t_dim_inline13231__ssa_v0
    int64_t t_dim_inline13231__ssa_v0 = static_cast<int64_t>(x_flat_inline13229__ssa_v0_tensor->shapes[0]);

    // Forward to ptoas-generated function
    hc_head_rms(
        x_flat_inline13229__ssa_v0, sq_part_inline13219__ssa_v0, t_dim_inline13231__ssa_v0, __pypto_spmd_block_idx,
        __pypto_spmd_block_num
    );
}
