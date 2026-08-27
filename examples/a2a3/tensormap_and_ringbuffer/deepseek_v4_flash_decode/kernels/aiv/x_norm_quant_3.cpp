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
// Kernel Function: x_norm_quant_3

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

static __aicore__ void x_norm_quant_3(__gm__ float *v1, __gm__ int8_t *v2, __gm__ float *v3, int64_t v4) {
    RoundMode v5 = RoundMode::CAST_TRUNC;
    RoundMode v6 = RoundMode::CAST_ROUND;
    SaturationMode v7 = SaturationMode::OFF;
    RoundMode v8 = RoundMode::CAST_RINT;
    const int64_t v9 = 256;
    const int64_t v10 = 512;
    const int64_t v11 = 4096;
    const int64_t v12 = 8;
    const int64_t v13 = 1;
    const int64_t v14 = 8224;
    const int64_t v15 = 32;
    const int64_t v16 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %xn_sq_col_inline2609_inline13061__tile
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v17 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v12, v13);
    // pto: %xn_sq_col_inline2609_inline13061__tile
    uint64_t v18 = (uint64_t)v16;
    TASSIGN(v17, v18);
    // pto: %6
    int64_t v19 = v4 < v16 ? v16 : v4;
    // pto: %xn_scale_buf_inline2572_inline13089__ssa_v0_pview
    pto::Shape<1, 1, 1, 8, 1> v20 = pto::Shape<1, 1, 1, 8, 1>();
    // pto: %xn_scale_buf_inline2572_inline13089__ssa_v0_pview
    pto::Stride<8, 8, 8, 1, 16> v21 = pto::Stride<8, 8, 8, 1, 16>();
    // pto: %xn_scale_buf_inline2572_inline13089__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<8, 8, 8, 1, 16>, pto::Layout::DN> v22 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<8, 8, 8, 1, 16>, pto::Layout::DN>(
            v1 + (v16 + v19), v20, v21
        );
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v17, v22);
    for (int64_t i23 = v16; i23 < v11; i23 += v10) {
        // pto: %t__tile
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v24 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v9);
        // pto: %t__tile
        uint64_t v25 = (uint64_t)v15;
        TASSIGN(v24, v25);
        // pto: %8
        int64_t v26 = i23 < v16 ? v16 : i23;
        // pto: %xg_buf_inline2582_inline13127__ssa_v0_pview
        pto::Shape<1, 1, 1, 8, 256> v27 = pto::Shape<1, 1, 1, 8, 256>();
        // pto: %xg_buf_inline2582_inline13127__ssa_v0_pview
        pto::Stride<32768, 32768, 32768, 4096, 1> v28 = pto::Stride<32768, 32768, 32768, 4096, 1>();
        // pto: %xg_buf_inline2582_inline13127__ssa_v0_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>
            v29 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(
                v3 + ((v16 + v19 * v11) + v26), v27, v28
            );
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        TLOAD(v24, v29);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %0
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v30 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v9);
        // pto: %0
        uint64_t v31 = (uint64_t)v14;
        TASSIGN(v30, v31);
        // pto: %10
        int64_t v32 = (int64_t)((uint64_t)i23 + (uint64_t)v9);
        // pto: %11
        int64_t v33 = v32 < v16 ? v16 : v32;
        // pto: %12
        pto::Shape<1, 1, 1, 8, 256> v34 = pto::Shape<1, 1, 1, 8, 256>();
        // pto: %12
        pto::Stride<32768, 32768, 32768, 4096, 1> v35 = pto::Stride<32768, 32768, 32768, 4096, 1>();
        // pto: %12
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>
            v36 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(
                v3 + ((v16 + v19 * v11) + v33), v34, v35
            );
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        TLOAD(v30, v36);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %xn_q_scaled_inline2608_inline13191__tile
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v37 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v9);
        // pto: %xn_q_scaled_inline2608_inline13191__tile
        uint64_t v38 = (uint64_t)v15;
        TASSIGN(v37, v38);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TROWEXPANDMUL(v37, v24, v17);
        // pto: %xn_q_i32_inline2575_inline13071__tile
        Tile<
            TileType::Vec, int32_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v39 = Tile<
                TileType::Vec, int32_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v9);
        // pto: %xn_q_i32_inline2575_inline13071__tile
        uint64_t v40 = (uint64_t)v15;
        TASSIGN(v39, v40);
        pipe_barrier(PIPE_V);
        TCVT(v39, v37, v8, v7);
        // pto: %xn_q_half_inline2562_inline12989__tile
        Tile<
            TileType::Vec, half, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v41 = Tile<
                TileType::Vec, half, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v9);
        // pto: %xn_q_half_inline2562_inline12989__tile
        uint64_t v42 = (uint64_t)v15;
        TASSIGN(v41, v42);
        pipe_barrier(PIPE_V);
        TCVT(v41, v39, v6, v7);
        // pto: %1
        Tile<
            TileType::Vec, int8_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v43 = Tile<
                TileType::Vec, int8_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v9);
        // pto: %1
        uint64_t v44 = (uint64_t)v15;
        TASSIGN(v43, v44);
        pipe_barrier(PIPE_V);
        TCVT(v43, v41, v5, v7);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %x_norm_i8_inline12996__iter_v3_pview
        pto::Shape<1, 1, 1, 8, 256> v45 = pto::Shape<1, 1, 1, 8, 256>();
        // pto: %x_norm_i8_inline12996__iter_v3_pview
        pto::Stride<32768, 32768, 32768, 4096, 1> v46 = pto::Stride<32768, 32768, 32768, 4096, 1>();
        // pto: %x_norm_i8_inline12996__iter_v3_pview
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>
            v47 = GlobalTensor<
                int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(
                v2 + ((v16 + v19 * v11) + v26), v45, v46
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v47, v43);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        // pto: %2
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v48 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v9);
        // pto: %2
        uint64_t v49 = (uint64_t)v14;
        TASSIGN(v48, v49);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TROWEXPANDMUL(v48, v30, v17);
        // pto: %3
        Tile<
            TileType::Vec, int32_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v50 = Tile<
                TileType::Vec, int32_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v9);
        // pto: %3
        uint64_t v51 = (uint64_t)v14;
        TASSIGN(v50, v51);
        pipe_barrier(PIPE_V);
        TCVT(v50, v48, v8, v7);
        // pto: %4
        Tile<
            TileType::Vec, half, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v52 = Tile<
                TileType::Vec, half, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v9);
        // pto: %4
        uint64_t v53 = (uint64_t)v14;
        TASSIGN(v52, v53);
        pipe_barrier(PIPE_V);
        TCVT(v52, v50, v6, v7);
        // pto: %5
        Tile<
            TileType::Vec, int8_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v54 = Tile<
                TileType::Vec, int8_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v9);
        // pto: %5
        uint64_t v55 = (uint64_t)v14;
        TASSIGN(v54, v55);
        pipe_barrier(PIPE_V);
        TCVT(v54, v52, v5, v7);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        // pto: %x_norm_i8_inline12996__tile_pview
        pto::Shape<1, 1, 1, 8, 256> v56 = pto::Shape<1, 1, 1, 8, 256>();
        // pto: %x_norm_i8_inline12996__tile_pview
        pto::Stride<32768, 32768, 32768, 4096, 1> v57 = pto::Stride<32768, 32768, 32768, 4096, 1>();
        // pto: %x_norm_i8_inline12996__tile_pview
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>
            v58 = GlobalTensor<
                int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(
                v2 + ((v16 + v19 * v11) + v33), v56, v57
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v58, v54);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Unpack tensor: xn_scale_buf_inline2572_inline13089__ssa_v0
    __gm__ Tensor *xn_scale_buf_inline2572_inline13089__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *xn_scale_buf_inline2572_inline13089__ssa_v0 =
        reinterpret_cast<__gm__ float *>(xn_scale_buf_inline2572_inline13089__ssa_v0_tensor->buffer.addr) +
        xn_scale_buf_inline2572_inline13089__ssa_v0_tensor->start_offset;

    // Unpack tensor: x_norm_i8_inline12996__iter_v1
    __gm__ Tensor *x_norm_i8_inline12996__iter_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int8_t *x_norm_i8_inline12996__iter_v1 =
        reinterpret_cast<__gm__ int8_t *>(x_norm_i8_inline12996__iter_v1_tensor->buffer.addr) +
        x_norm_i8_inline12996__iter_v1_tensor->start_offset;

    // Unpack tensor: xg_buf_inline2582_inline13127__ssa_v0
    __gm__ Tensor *xg_buf_inline2582_inline13127__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *xg_buf_inline2582_inline13127__ssa_v0 =
        reinterpret_cast<__gm__ float *>(xg_buf_inline2582_inline13127__ssa_v0_tensor->buffer.addr) +
        xg_buf_inline2582_inline13127__ssa_v0_tensor->start_offset;

    // Unpack scalar: t0_inline2605_inline12987__idx_v0
    union {
        uint64_t u64;
        int64_t val;
    } t0_inline2605_inline12987__idx_v0_conv;
    t0_inline2605_inline12987__idx_v0_conv.u64 = args[3];
    int64_t t0_inline2605_inline12987__idx_v0 = t0_inline2605_inline12987__idx_v0_conv.val;

    // Forward to ptoas-generated function
    x_norm_quant_3(
        xn_scale_buf_inline2572_inline13089__ssa_v0, x_norm_i8_inline12996__iter_v1,
        xg_buf_inline2582_inline13127__ssa_v0, t0_inline2605_inline12987__idx_v0
    );
}
