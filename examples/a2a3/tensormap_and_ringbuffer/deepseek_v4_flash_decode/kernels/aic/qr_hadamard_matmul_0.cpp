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
// Kernel Function: qr_hadamard_matmul_0

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
qr_hadamard_matmul_0(__gm__ bfloat16_t *v1, __gm__ bfloat16_t *v2, __gm__ float *v3, int32_t v4, int32_t v5) {
    const int64_t v6 = 64;
    const int64_t v7 = 128;
    const int64_t v8 = 16384;
    const int64_t v9 = 0;
    using T = float;

#if defined(__DAV_CUBE__)
    // pto: %idx_inline2009_inline12466__ssa_v0, %0
    int64_t v10 = (int64_t)((uint64_t)((int64_t)v4) * (uint64_t)v6);
    // pto: %t__tile
    Tile<
        TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v11 = Tile<
            TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v6, v7);
    // pto: %t__tile
    uint64_t v12 = (uint64_t)v9;
    TASSIGN(v11, v12);
    // pto: %1
    int64_t v13 = v10 < v9 ? v9 : v10;
    // pto: %qr_bf16_inline1969_inline12702__ssa_v1_pview
    pto::Shape<1, 1, 1, 64, 128> v14 = pto::Shape<1, 1, 1, 64, 128>();
    // pto: %qr_bf16_inline1969_inline12702__ssa_v1_pview
    pto::Stride<8192, 8192, 8192, 128, 1> v15 = pto::Stride<8192, 8192, 8192, 128, 1>();
    // pto: %qr_bf16_inline1969_inline12702__ssa_v1_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v16 =
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(
            v1 + (v9 + v13 * v7), v14, v15
        );
    TLOAD(v11, v16);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // pto: %csa_hadamard_idx_last_inline669__ssa_v0_mat
    Tile<
        TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v17 = Tile<
            TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v7, v7);
    // pto: %csa_hadamard_idx_last_inline669__ssa_v0_mat
    uint64_t v18 = (uint64_t)v8;
    TASSIGN(v17, v18);
    // pto: %csa_hadamard_idx_last_inline669__ssa_v0_pview
    pto::Shape<1, 1, 1, 128, 128> v19 = pto::Shape<1, 1, 1, 128, 128>();
    // pto: %csa_hadamard_idx_last_inline669__ssa_v0_pview
    pto::Stride<16384, 16384, 16384, 128, 1> v20 = pto::Stride<16384, 16384, 16384, 128, 1>();
    // pto: %csa_hadamard_idx_last_inline669__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>
        v21 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(
            v2, v19, v20
        );
    TLOAD(v17, v21);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    // pto: %t__tile_Left
    Tile<
        TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v22 = Tile<
            TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v6, v7);
    // pto: %t__tile_Left
    uint64_t v23 = (uint64_t)v9;
    TASSIGN(v22, v23);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(v22, v11);
    // pto: %csa_hadamard_idx_last_inline669__ssa_v0_mat_Right
    Tile<
        TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v24 = Tile<
            TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v7, v7);
    // pto: %csa_hadamard_idx_last_inline669__ssa_v0_mat_Right
    uint64_t v25 = (uint64_t)v9;
    TASSIGN(v24, v25);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TMOV(v24, v17);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    // pto: %qh_acc_inline2010_inline12639__tile
    Tile<
        TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v26 = Tile<
            TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v6, v7);
    // pto: %qh_acc_inline2010_inline12639__tile
    uint64_t v27 = (uint64_t)v9;
    TASSIGN(v26, v27);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(v26, v22, v24);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    // pto: %qh_acc_gm_inline2006_inline12354__ssa_v0_pview
    pto::Shape<1, 1, 1, 64, 128> v28 = pto::Shape<1, 1, 1, 64, 128>();
    // pto: %qh_acc_gm_inline2006_inline12354__ssa_v0_pview
    pto::Stride<8192, 8192, 8192, 128, 1> v29 = pto::Stride<8192, 8192, 8192, 128, 1>();
    // pto: %qh_acc_gm_inline2006_inline12354__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v30 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(
            v3 + (v9 + v13 * v7), v28, v29
        );
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(v30, v26);
#endif  // __DAV_CUBE__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: qr_bf16_inline1969_inline12702__ssa_v1
    __gm__ Tensor *qr_bf16_inline1969_inline12702__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ bfloat16_t *qr_bf16_inline1969_inline12702__ssa_v1 =
        reinterpret_cast<__gm__ bfloat16_t *>(qr_bf16_inline1969_inline12702__ssa_v1_tensor->buffer.addr) +
        qr_bf16_inline1969_inline12702__ssa_v1_tensor->start_offset;

    // Unpack tensor: csa_hadamard_idx_last_inline669__ssa_v0
    __gm__ Tensor *csa_hadamard_idx_last_inline669__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ bfloat16_t *csa_hadamard_idx_last_inline669__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(csa_hadamard_idx_last_inline669__ssa_v0_tensor->buffer.addr) +
        csa_hadamard_idx_last_inline669__ssa_v0_tensor->start_offset;

    // Unpack tensor: qh_acc_gm_inline2006_inline12354__ssa_v0
    __gm__ Tensor *qh_acc_gm_inline2006_inline12354__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *qh_acc_gm_inline2006_inline12354__ssa_v0 =
        reinterpret_cast<__gm__ float *>(qh_acc_gm_inline2006_inline12354__ssa_v0_tensor->buffer.addr) +
        qh_acc_gm_inline2006_inline12354__ssa_v0_tensor->start_offset;

    // Forward to ptoas-generated function
    qr_hadamard_matmul_0(
        qr_bf16_inline1969_inline12702__ssa_v1, csa_hadamard_idx_last_inline669__ssa_v0,
        qh_acc_gm_inline2006_inline12354__ssa_v0, __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
