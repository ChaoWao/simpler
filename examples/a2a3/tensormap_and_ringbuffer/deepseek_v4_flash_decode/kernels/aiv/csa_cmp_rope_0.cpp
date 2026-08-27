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
// Kernel Function: csa_cmp_rope_0

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

static __aicore__ void
csa_cmp_rope_0(__gm__ float *v1, __gm__ float *v2, __gm__ int32_t *v3, __gm__ bfloat16_t *v4, __gm__ bfloat16_t *v5) {
    SaturationMode v6 = SaturationMode::OFF;
    RoundMode v7 = RoundMode::CAST_ROUND;
    const int64_t v8 = 2;
    const int64_t v9 = 1;
    const int64_t v10 = 32;
    const int64_t v11 = 4;
    const int64_t v12 = 64;
    const int64_t v13 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    for (int64_t i14 = v13; i14 < v11; i14 += v9) {
        // pto: %3, %first_pos_b_inline12445__tile
        int32_t v15 = (v3)[(int64_t)((uint64_t)i14 * (uint64_t)v8)];
        // pto: %4
        int64_t v16 = (int64_t)v15;
        // pto: %8, %6, %5, %9
        int64_t v17 = (int64_t)((uint64_t)((int64_t)((uint64_t)v16 +
                                                     (uint64_t)((int64_t)((uint64_t)v11 - (uint64_t)(v16 % v11))))) -
                                (uint64_t)v11);
        // pto: %t__tile
        Tile<
            TileType::Vec, bfloat16_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v18 = Tile<
                TileType::Vec, bfloat16_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v9, v10);
        // pto: %t__tile
        uint64_t v19 = (uint64_t)v13;
        TASSIGN(v18, v19);
        // pto: %11
        int64_t v20 = v17 < v13 ? v13 : v17;
        // pto: %compressed_freqs_cos_inline559__ssa_v0_pview
        pto::Shape<1, 1, 1, 1, 32> v21 = pto::Shape<1, 1, 1, 1, 32>();
        // pto: %compressed_freqs_cos_inline559__ssa_v0_pview
        pto::Stride<64, 64, 64, 64, 1> v22 = pto::Stride<64, 64, 64, 64, 1>();
        // pto: %compressed_freqs_cos_inline559__ssa_v0_pview
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v23 =
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(
                v4 + (v13 + v20 * v12), v21, v22
            );
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v18, v23);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %0
        Tile<
            TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v24 = Tile<
                TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v9, v10);
        // pto: %0
        uint64_t v25 = (uint64_t)v12;
        TASSIGN(v24, v25);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        TCVT(v24, v18, v7, v6);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %12
        int64_t v26 = i14 < v13 ? v13 : i14;
        // pto: %cmp_cos_inline12490__iter_v1_pview
        pto::Shape<1, 1, 1, 1, 32> v27 = pto::Shape<1, 1, 1, 1, 32>();
        // pto: %cmp_cos_inline12490__iter_v1_pview
        pto::Stride<32, 32, 32, 32, 1> v28 = pto::Stride<32, 32, 32, 32, 1>();
        // pto: %cmp_cos_inline12490__iter_v1_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v29 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(
                v1 + (v13 + v26 * v10), v27, v28
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v29, v24);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        // pto: %1
        Tile<
            TileType::Vec, bfloat16_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v30 = Tile<
                TileType::Vec, bfloat16_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v9, v10);
        // pto: %1
        uint64_t v31 = (uint64_t)v13;
        TASSIGN(v30, v31);
        // pto: %compressed_freqs_sin_inline692__ssa_v0_pview
        pto::Shape<1, 1, 1, 1, 32> v32 = pto::Shape<1, 1, 1, 1, 32>();
        // pto: %compressed_freqs_sin_inline692__ssa_v0_pview
        pto::Stride<64, 64, 64, 64, 1> v33 = pto::Stride<64, 64, 64, 64, 1>();
        // pto: %compressed_freqs_sin_inline692__ssa_v0_pview
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v34 =
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(
                v5 + (v13 + v20 * v12), v32, v33
            );
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(v30, v34);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %2
        Tile<
            TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v35 = Tile<
                TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v9, v10);
        // pto: %2
        uint64_t v36 = (uint64_t)v12;
        TASSIGN(v35, v36);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        TCVT(v35, v30, v7, v6);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        // pto: %cmp_sin_inline12261__iter_v1_pview
        pto::Shape<1, 1, 1, 1, 32> v37 = pto::Shape<1, 1, 1, 1, 32>();
        // pto: %cmp_sin_inline12261__iter_v1_pview
        pto::Stride<32, 32, 32, 32, 1> v38 = pto::Stride<32, 32, 32, 32, 1>();
        // pto: %cmp_sin_inline12261__iter_v1_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v39 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(
                v2 + (v13 + v26 * v10), v37, v38
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSTORE(v39, v35);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Unpack tensor: cmp_cos_inline12490__ssa_v0
    __gm__ Tensor *cmp_cos_inline12490__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *cmp_cos_inline12490__ssa_v0 =
        reinterpret_cast<__gm__ float *>(cmp_cos_inline12490__ssa_v0_tensor->buffer.addr) +
        cmp_cos_inline12490__ssa_v0_tensor->start_offset;

    // Unpack tensor: cmp_sin_inline12261__ssa_v0
    __gm__ Tensor *cmp_sin_inline12261__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *cmp_sin_inline12261__ssa_v0 =
        reinterpret_cast<__gm__ float *>(cmp_sin_inline12261__ssa_v0_tensor->buffer.addr) +
        cmp_sin_inline12261__ssa_v0_tensor->start_offset;

    // Unpack tensor: position_ids__ssa_v0
    __gm__ Tensor *position_ids__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ int32_t *position_ids__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(position_ids__ssa_v0_tensor->buffer.addr) +
        position_ids__ssa_v0_tensor->start_offset;

    // Unpack tensor: compressed_freqs_cos_inline559__ssa_v0
    __gm__ Tensor *compressed_freqs_cos_inline559__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ bfloat16_t *compressed_freqs_cos_inline559__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(compressed_freqs_cos_inline559__ssa_v0_tensor->buffer.addr) +
        compressed_freqs_cos_inline559__ssa_v0_tensor->start_offset;

    // Unpack tensor: compressed_freqs_sin_inline692__ssa_v0
    __gm__ Tensor *compressed_freqs_sin_inline692__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ bfloat16_t *compressed_freqs_sin_inline692__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(compressed_freqs_sin_inline692__ssa_v0_tensor->buffer.addr) +
        compressed_freqs_sin_inline692__ssa_v0_tensor->start_offset;

    // Forward to ptoas-generated function
    csa_cmp_rope_0(
        cmp_cos_inline12490__ssa_v0, cmp_sin_inline12261__ssa_v0, position_ids__ssa_v0,
        compressed_freqs_cos_inline559__ssa_v0, compressed_freqs_sin_inline692__ssa_v0
    );
}
