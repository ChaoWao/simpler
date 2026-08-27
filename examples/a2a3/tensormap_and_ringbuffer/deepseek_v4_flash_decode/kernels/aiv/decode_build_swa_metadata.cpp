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
// Kernel Function: decode_build_swa_metadata

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
decode_build_swa_metadata(__gm__ int32_t *v1, __gm__ int32_t *v2, __gm__ int32_t *v3, int32_t v4, int32_t v5) {
    const int32_t v6 = -1;
    const int64_t v7 = 2;
    const int64_t v8 = 128;
    const int64_t v9 = 1;
    const int64_t v10 = 512;
    const int64_t v11 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %token_inline742_inline8662__ssa_v0
    int64_t v12 = (int64_t)v4;
    // pto: %position_inline749_inline8673__tile
    int32_t v13 = (v1)[v12];
    // pto: %2
    int64_t v14 = (int64_t)v13;
    // pto: %3
    int64_t v15 = (int64_t)((uint64_t)v14 + (uint64_t)v9);
    // pto: %4
    int64_t v16 = v15 < v8 ? v15 : v8;
    // pto: %index_row_inline743_inline8668__tile
    Tile<
        TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v17 = Tile<
            TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v9, v8);
    // pto: %index_row_inline743_inline8668__tile
    uint64_t v18 = (uint64_t)v11;
    TASSIGN(v17, v18);
    // pto: %t__tile
    Tile<
        TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v19 = Tile<
            TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v9, v8);
    // pto: %t__tile
    uint64_t v20 = (uint64_t)v10;
    TASSIGN(v19, v20);
    TEXPANDS(v19, v6);
    // pto: %assemble_view
    Tile<
        TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v21;
    // pto: %assemble_view
    Tile<
        TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v22 = v21;
    // pto: %assemble_view
    uint64_t v23 = (uint64_t)v11;
    TASSIGN(v22, v23);
    pipe_barrier(PIPE_V);
    TMOV(v22, v19);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    for (int64_t i24 = v11; i24 < v8; i24 += v9) {
        // pto: %8
        if (i24 < v16) {
            // pto: %6, %7, %9
            int64_t v25 =
                (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)v14 - (uint64_t)v16)) + (uint64_t)v9)) +
                          (uint64_t)i24);
            // pto: %1, %flat_offset_mul, %flat_offset, %10, %visible_physical_block_inline754_inline8685__tile
            int32_t v26 =
                (v2)[(int64_t)((uint64_t)((int64_t)((uint64_t)(v12 / v7) * (uint64_t)v8)) + (uint64_t)(v25 / v8))];
            // pto: %13, %14, %15, %11, %16
            int32_t v27 = (int32_t)((int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)v26) * (uint64_t)v8)) +
                                              (uint64_t)(v25 % v8)));
            v17.SetValue(i24, v27);
        }
    }
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    // pto: %swa_indices_inline636__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 128> v28 = pto::Shape<1, 1, 1, 1, 128>();
    // pto: %swa_indices_inline636__ssa_v0_pview
    pto::Stride<128, 128, 128, 128, 1> v29 = pto::Stride<128, 128, 128, 128, 1>();
    // pto: %19, %swa_indices_inline636__ssa_v0_pview
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v30 =
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(
            v3 + (v11 + (v12 < v11 ? v11 : v12) * v8), v28, v29
        );
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v30, v17);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: position_ids__ssa_v0
    __gm__ Tensor *position_ids__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int32_t *position_ids__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(position_ids__ssa_v0_tensor->buffer.addr) +
        position_ids__ssa_v0_tensor->start_offset;

    // Unpack tensor: block_table__ssa_v0
    __gm__ Tensor *block_table__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int32_t *block_table__ssa_v0 = reinterpret_cast<__gm__ int32_t *>(block_table__ssa_v0_tensor->buffer.addr) +
                                          block_table__ssa_v0_tensor->start_offset;

    // Unpack tensor: swa_indices_inline636__ssa_v0
    __gm__ Tensor *swa_indices_inline636__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ int32_t *swa_indices_inline636__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(swa_indices_inline636__ssa_v0_tensor->buffer.addr) +
        swa_indices_inline636__ssa_v0_tensor->start_offset;

    // Forward to ptoas-generated function
    decode_build_swa_metadata(
        position_ids__ssa_v0, block_table__ssa_v0, swa_indices_inline636__ssa_v0, __pypto_spmd_block_idx,
        __pypto_spmd_block_num
    );
}
