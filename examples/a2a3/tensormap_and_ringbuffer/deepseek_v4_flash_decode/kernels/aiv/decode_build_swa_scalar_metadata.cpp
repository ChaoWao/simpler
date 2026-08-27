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
// Kernel Function: decode_build_swa_scalar_metadata

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

static __aicore__ void decode_build_swa_scalar_metadata(
    __gm__ int32_t *v1, __gm__ int32_t *v2, __gm__ int64_t *v3, __gm__ int32_t *v4, int32_t v5, int64_t v6, int32_t v7,
    int32_t v8
) {
    const int64_t v9 = 2;
    const int64_t v10 = 128;
    const int64_t v11 = 1;
    const int64_t v12 = 8;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %metadata_core_inline746_inline8661__ssa_v0
    for (int64_t i13 = (int64_t)v7; i13 < v12; i13 += v11) {
        // pto: %position_inline749_inline8673__tile
        int32_t v14 = (v1)[i13];
        // pto: %1
        int64_t v15 = (int64_t)v14;
        // pto: %0, %flat_offset_mul, %flat_offset, %2, %physical_block_inline740_inline8682__tile
        int32_t v16 =
            (v2)[(int64_t)((uint64_t)((int64_t)((uint64_t)(i13 / v9) * (uint64_t)v10)) + (uint64_t)(v15 / v10))];
        // pto: %6, %7, %8, %4
        int64_t v17 =
            (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)v16) * (uint64_t)v10)) + (uint64_t)(v15 % v10));
        (v3)[i13] = v17;
        // pto: %11
        int64_t v18 = (int64_t)((uint64_t)v15 + (uint64_t)v11);
        // pto: %12, %13
        int32_t v19 = (int32_t)(v18 < v10 ? v18 : v10);
        (v4)[i13] = v19;
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

    // Unpack tensor: position_ids__ssa_v0
    __gm__ Tensor *position_ids__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int32_t *position_ids__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(position_ids__ssa_v0_tensor->buffer.addr) +
        position_ids__ssa_v0_tensor->start_offset;

    // Unpack tensor: block_table__ssa_v0
    __gm__ Tensor *block_table__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int32_t *block_table__ssa_v0 = reinterpret_cast<__gm__ int32_t *>(block_table__ssa_v0_tensor->buffer.addr) +
                                          block_table__ssa_v0_tensor->start_offset;

    // Unpack tensor: swa_slot_mapping_inline576__ssa_v0
    __gm__ Tensor *swa_slot_mapping_inline576__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ int64_t *swa_slot_mapping_inline576__ssa_v0 =
        reinterpret_cast<__gm__ int64_t *>(swa_slot_mapping_inline576__ssa_v0_tensor->buffer.addr) +
        swa_slot_mapping_inline576__ssa_v0_tensor->start_offset;

    // Unpack tensor: swa_lens_inline628__ssa_v0
    __gm__ Tensor *swa_lens_inline628__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ int32_t *swa_lens_inline628__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(swa_lens_inline628__ssa_v0_tensor->buffer.addr) +
        swa_lens_inline628__ssa_v0_tensor->start_offset;

    // Unpack scalar: position_inline749_inline8673__ssa_v0
    union {
        uint64_t u64;
        int32_t val;
    } position_inline749_inline8673__ssa_v0_conv;
    position_inline749_inline8673__ssa_v0_conv.u64 = args[4];
    int32_t position_inline749_inline8673__ssa_v0 = position_inline749_inline8673__ssa_v0_conv.val;

    // Unpack scalar: request_inline748_inline8666__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } request_inline748_inline8666__ssa_v0_conv;
    request_inline748_inline8666__ssa_v0_conv.u64 = args[5];
    int64_t request_inline748_inline8666__ssa_v0 = request_inline748_inline8666__ssa_v0_conv.val;

    // Forward to ptoas-generated function
    decode_build_swa_scalar_metadata(
        position_ids__ssa_v0, block_table__ssa_v0, swa_slot_mapping_inline576__ssa_v0, swa_lens_inline628__ssa_v0,
        position_inline749_inline8673__ssa_v0, request_inline748_inline8666__ssa_v0, __pypto_spmd_block_idx,
        __pypto_spmd_block_num
    );
}
