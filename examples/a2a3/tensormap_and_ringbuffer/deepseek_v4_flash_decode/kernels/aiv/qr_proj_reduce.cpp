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
// Kernel Function: qr_proj_reduce

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
qr_proj_reduce(__gm__ float *v1, __gm__ float *v2, int64_t v3, int64_t v4, int64_t v5, int32_t v6, int32_t v7) {
    const int64_t v8 = 1024;
    const int64_t v9 = 128;
    const int64_t v10 = 16;
    const int64_t v11 = 8;
    const int64_t v12 = 8192;
    const int64_t v13 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %qr_reduce_idx_inline818_inline8895__ssa_v0
    int64_t v14 = (int64_t)v6;
    // pto: %1, %2
    int64_t v15 = (int64_t)((uint64_t)(v14 / v11) * (uint64_t)v10);
    // pto: %3, %4
    int64_t v16 = (int64_t)((uint64_t)(v14 % v11) * (uint64_t)v9);
    // pto: %qr_total_inline876_inline8909__tile
    Tile<
        TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v17 = Tile<
            TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v10, v9);
    // pto: %qr_total_inline876_inline8909__tile
    uint64_t v18 = (uint64_t)v13;
    TASSIGN(v17, v18);
    // pto: %5
    int64_t v19 = v15 < v13 ? v13 : v15;
    // pto: %6
    int64_t v20 = v16 < v13 ? v13 : v16;
    // pto: %qr_partials_inline803_inline8977__rv_v2_pview
    pto::Shape<1, 1, 1, 16, 128> v21 = pto::Shape<1, 1, 1, 16, 128>();
    // pto: %qr_partials_inline803_inline8977__rv_v2_pview
    pto::Stride<16384, 16384, 16384, 1024, 1> v22 = pto::Stride<16384, 16384, 16384, 1024, 1>();
    // pto: %qr_partials_inline803_inline8977__rv_v2_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<16384, 16384, 16384, 1024, 1>, pto::Layout::ND> v23 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<16384, 16384, 16384, 1024, 1>, pto::Layout::ND>(
            v1 + ((v13 + v19 * v8) + v20), v21, v22
        );
    TLOAD(v17, v23);
    // pto: %7
    int64_t v24 = (int64_t)((uint64_t)v5 + (uint64_t)v15);
    // pto: %t__tile
    Tile<
        TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v25 = Tile<
            TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v10, v9);
    // pto: %t__tile
    uint64_t v26 = (uint64_t)v12;
    TASSIGN(v25, v26);
    // pto: %10
    pto::Shape<1, 1, 1, 16, 128> v27 = pto::Shape<1, 1, 1, 16, 128>();
    // pto: %10
    pto::Stride<16384, 16384, 16384, 1024, 1> v28 = pto::Stride<16384, 16384, 16384, 1024, 1>();
    // pto: %8, %10
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<16384, 16384, 16384, 1024, 1>, pto::Layout::ND> v29 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<16384, 16384, 16384, 1024, 1>, pto::Layout::ND>(
            v1 + ((v13 + (v24 < v13 ? v13 : v24) * v8) + v20), v27, v28
        );
    TLOAD(v25, v29);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %0
    Tile<
        TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v30 = Tile<
            TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v10, v9);
    // pto: %0
    uint64_t v31 = (uint64_t)v13;
    TASSIGN(v30, v31);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(v30, v17, v25);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %qr_fp32_inline806_inline8876__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 128> v32 = pto::Shape<1, 1, 1, 16, 128>();
    // pto: %qr_fp32_inline806_inline8876__ssa_v0_pview
    pto::Stride<16384, 16384, 16384, 1024, 1> v33 = pto::Stride<16384, 16384, 16384, 1024, 1>();
    // pto: %qr_fp32_inline806_inline8876__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<16384, 16384, 16384, 1024, 1>, pto::Layout::ND> v34 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<16384, 16384, 16384, 1024, 1>, pto::Layout::ND>(
            v2 + ((v13 + v19 * v8) + v20), v32, v33
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v34, v30);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: qr_partials_inline803_inline8977__rv_v2
    __gm__ Tensor *qr_partials_inline803_inline8977__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *qr_partials_inline803_inline8977__rv_v2 =
        reinterpret_cast<__gm__ float *>(qr_partials_inline803_inline8977__rv_v2_tensor->buffer.addr) +
        qr_partials_inline803_inline8977__rv_v2_tensor->start_offset;

    // Unpack tensor: qr_fp32_inline806_inline8876__ssa_v0
    __gm__ Tensor *qr_fp32_inline806_inline8876__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *qr_fp32_inline806_inline8876__ssa_v0 =
        reinterpret_cast<__gm__ float *>(qr_fp32_inline806_inline8876__ssa_v0_tensor->buffer.addr) +
        qr_fp32_inline806_inline8876__ssa_v0_tensor->start_offset;

    // Unpack scalar: t_matmul_inline856_inline8847__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } t_matmul_inline856_inline8847__ssa_v0_conv;
    t_matmul_inline856_inline8847__ssa_v0_conv.u64 = args[2];
    int64_t t_matmul_inline856_inline8847__ssa_v0 = t_matmul_inline856_inline8847__ssa_v0_conv.val;

    // Extract dynamic dim: qr_partial_rows_inline846_inline8854__ssa_v0
    int64_t qr_partial_rows_inline846_inline8854__ssa_v0 =
        static_cast<int64_t>(qr_partials_inline803_inline8977__rv_v2_tensor->shapes[0]);

    // Extract dynamic dim: t_matmul_inline856_inline8847__ssa_v0_1
    int64_t t_matmul_inline856_inline8847__ssa_v0_1 =
        static_cast<int64_t>(qr_fp32_inline806_inline8876__ssa_v0_tensor->shapes[0]);

    // Forward to ptoas-generated function
    qr_proj_reduce(
        qr_partials_inline803_inline8977__rv_v2, qr_fp32_inline806_inline8876__ssa_v0,
        t_matmul_inline856_inline8847__ssa_v0, qr_partial_rows_inline846_inline8854__ssa_v0,
        t_matmul_inline856_inline8847__ssa_v0_1, __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
