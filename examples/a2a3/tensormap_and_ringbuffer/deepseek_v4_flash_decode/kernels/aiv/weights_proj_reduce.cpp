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
// Kernel Function: weights_proj_reduce

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

static __aicore__ void weights_proj_reduce(__gm__ float *v1, __gm__ float *v2) {
    unsigned v3 = 3072;
    unsigned v4 = 2048;
    unsigned v5 = 1024;
    const float v6 = 0.0110485433f;
    const int64_t v7 = 16;
    const int64_t v8 = 64;
    const int64_t v9 = 4096;
    const int64_t v10 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %w_sum_inline1952_inline10819__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v11 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v7, v8);
    // pto: %w_sum_inline1952_inline10819__tile
    uint64_t v12 = (uint64_t)v10;
    TASSIGN(v11, v12);
    // pto: %weights_partial_inline2014_inline10666__ssa_v1_pview
    pto::Shape<1, 1, 1, 16, 64> v13 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %weights_partial_inline2014_inline10666__ssa_v1_pview
    pto::Stride<1024, 1024, 1024, 64, 1> v14 = pto::Stride<1024, 1024, 1024, 64, 1>();
    // pto: %weights_partial_inline2014_inline10666__ssa_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v15 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(
            v1, v13, v14
        );
    TLOAD(v11, v15);
    // pto: %t__tile
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v16 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v7, v8);
    // pto: %t__tile
    uint64_t v17 = (uint64_t)v9;
    TASSIGN(v16, v17);
    // pto: %6
    pto::Shape<1, 1, 1, 16, 64> v18 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %6
    pto::Stride<1024, 1024, 1024, 64, 1> v19 = pto::Stride<1024, 1024, 1024, 64, 1>();
    // pto: %6
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v20 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(
            v1 + v5, v18, v19
        );
    TLOAD(v16, v20);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %0
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v21 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v7, v8);
    // pto: %0
    uint64_t v22 = (uint64_t)v10;
    TASSIGN(v21, v22);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(v21, v11, v16);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    // pto: %1
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v23 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v7, v8);
    // pto: %1
    uint64_t v24 = (uint64_t)v9;
    TASSIGN(v23, v24);
    // pto: %7
    pto::Shape<1, 1, 1, 16, 64> v25 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %7
    pto::Stride<1024, 1024, 1024, 64, 1> v26 = pto::Stride<1024, 1024, 1024, 64, 1>();
    // pto: %7
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v27 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(
            v1 + v4, v25, v26
        );
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v23, v27);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %2
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v28 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v7, v8);
    // pto: %2
    uint64_t v29 = (uint64_t)v10;
    TASSIGN(v28, v29);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TADD(v28, v21, v23);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    // pto: %3
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v30 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v7, v8);
    // pto: %3
    uint64_t v31 = (uint64_t)v9;
    TASSIGN(v30, v31);
    // pto: %8
    pto::Shape<1, 1, 1, 16, 64> v32 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %8
    pto::Stride<1024, 1024, 1024, 64, 1> v33 = pto::Stride<1024, 1024, 1024, 64, 1>();
    // pto: %8
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v34 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(
            v1 + v3, v32, v33
        );
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    TLOAD(v30, v34);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    // pto: %4
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v35 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v7, v8);
    // pto: %4
    uint64_t v36 = (uint64_t)v10;
    TASSIGN(v35, v36);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    TADD(v35, v28, v30);
    // pto: %5
    Tile<
        TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v37 = Tile<
            TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v7, v8);
    // pto: %5
    uint64_t v38 = (uint64_t)v10;
    TASSIGN(v37, v38);
    pipe_barrier(PIPE_V);
    TMULS(v37, v35, v6);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %weights_inline2015_inline10621__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 64> v39 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %weights_inline2015_inline10621__ssa_v0_pview
    pto::Stride<1024, 1024, 1024, 64, 1> v40 = pto::Stride<1024, 1024, 1024, 64, 1>();
    // pto: %weights_inline2015_inline10621__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v41 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(
            v2, v39, v40
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v41, v37);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Unpack tensor: weights_partial_inline2014_inline10666__ssa_v1
    __gm__ Tensor *weights_partial_inline2014_inline10666__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *weights_partial_inline2014_inline10666__ssa_v1 =
        reinterpret_cast<__gm__ float *>(weights_partial_inline2014_inline10666__ssa_v1_tensor->buffer.addr) +
        weights_partial_inline2014_inline10666__ssa_v1_tensor->start_offset;

    // Unpack tensor: weights_inline2015_inline10621__ssa_v0
    __gm__ Tensor *weights_inline2015_inline10621__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *weights_inline2015_inline10621__ssa_v0 =
        reinterpret_cast<__gm__ float *>(weights_inline2015_inline10621__ssa_v0_tensor->buffer.addr) +
        weights_inline2015_inline10621__ssa_v0_tensor->start_offset;

    // Forward to ptoas-generated function
    weights_proj_reduce(weights_partial_inline2014_inline10666__ssa_v1, weights_inline2015_inline10621__ssa_v0);
}
