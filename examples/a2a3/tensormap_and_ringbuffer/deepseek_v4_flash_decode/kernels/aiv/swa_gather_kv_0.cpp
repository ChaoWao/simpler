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
// Kernel Function: swa_gather_kv_0

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
swa_gather_kv_0(__gm__ bfloat16_t *v1, __gm__ int32_t *v2, __gm__ bfloat16_t *v3, int64_t v4, int32_t v5, int32_t v6) {
    const bfloat16_t v7 = 0.0f;
    const int64_t v8 = 15;
    const int64_t v9 = 16;
    const int64_t v10 = 2;
    const int64_t v11 = 32;
    const int64_t v12 = 4;
    const int64_t v13 = 128;
    const int64_t v14 = 1;
    const int64_t v15 = 512;
    const int64_t v16 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %g_task_inline1012_inline9854__ssa_v0
    int64_t v17 = (int64_t)v5;
    // pto: %2
    int64_t v18 = v17 / v12;
    // pto: %6
    int64_t v19 = (int64_t)((uint64_t)v18 * (uint64_t)v13);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    for (int64_t i20 = v16; i20 < v10; i20 += v14) {
        // pto: %4, %3, %5, %8, %7
        int64_t v21 =
            (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)v17 -
                                                                (uint64_t)((int64_t)((uint64_t)v18 * (uint64_t)v12)))) *
                                           (uint64_t)v11)) +
                      (uint64_t)((int64_t)((uint64_t)i20 * (uint64_t)v9)));
        // pto: %9
        int64_t v22 = (int64_t)((uint64_t)v19 + (uint64_t)v21);
        // pto: %g_first_inline1057_inline9858__tile
        int32_t v23 = (v2)[v22];
        // pto: %12, %10, %g_last_inline1058_inline9859__tile
        int32_t v24 = (v2)[(int64_t)((uint64_t)v19 + (uint64_t)((int64_t)((uint64_t)v21 + (uint64_t)v8)))];
        // pto: %15
        int64_t v25 = (int64_t)v23;
        // pto: %13, %14, %18, %16, %17, %19
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        if ((int64_t)((uint64_t)((int64_t)((int32_t)((uint32_t)v24 - (uint32_t)v23))) +
                      (uint64_t)((int64_t)((uint64_t)(v25 < v16 ? v25 : v16) * (uint64_t)v9))) == v8) {
            // pto: %t__tile
            Tile<
                TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v26 = Tile<
                    TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                    PadValue::Null, CompactMode::Null>(v9, v15);
            // pto: %t__tile
            uint64_t v27 = (uint64_t)v16;
            TASSIGN(v26, v27);
            // pto: %ori_kv_flat_inline1033_inline9732__ssa_v0_pview
            pto::Shape<1, 1, 1, 16, 512> v28 = pto::Shape<1, 1, 1, 16, 512>();
            // pto: %ori_kv_flat_inline1033_inline9732__ssa_v0_pview
            pto::Stride<8192, 8192, 8192, 512, 1> v29 = pto::Stride<8192, 8192, 8192, 512, 1>();
            // pto: %21, %ori_kv_flat_inline1033_inline9732__ssa_v0_pview
            GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>
                v30 = GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
                    v3 + (v16 + (v25 < v16 ? v16 : v25) * v15), v28, v29
                );
            TLOAD(v26, v30);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            // pto: %swa_kv_flat_inline1000_inline9849__iter_v1_pview
            pto::Shape<1, 1, 1, 16, 512> v31 = pto::Shape<1, 1, 1, 16, 512>();
            // pto: %swa_kv_flat_inline1000_inline9849__iter_v1_pview
            pto::Stride<8192, 8192, 8192, 512, 1> v32 = pto::Stride<8192, 8192, 8192, 512, 1>();
            // pto: %22, %swa_kv_flat_inline1000_inline9849__iter_v1_pview
            GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>
                v33 = GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
                    v1 + (v16 + (v22 < v16 ? v16 : v22) * v15), v31, v32
                );
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(v33, v26);
        } else {
            for (int64_t j34 = v16; j34 < v9; j34 += v14) {
                // pto: %23
                int64_t v35 = (int64_t)((uint64_t)v22 + (uint64_t)j34);
                // pto: %26, %24, %g_slot_i32_inline1035_inline9867__tile
                int32_t v36 = (v2)[(int64_t)((uint64_t)v19 + (uint64_t)((int64_t)((uint64_t)v21 + (uint64_t)j34)))];
                // pto: %27
                int64_t v37 = (int64_t)v36;
                // pto: %28
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
                if (v37 >= v16) {
                    // pto: %0
                    Tile<
                        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                        PadValue::Null, CompactMode::Null>
                        v38 = Tile<
                            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                            PadValue::Null, CompactMode::Null>(v14, v15);
                    // pto: %0
                    uint64_t v39 = (uint64_t)v16;
                    TASSIGN(v38, v39);
                    // pto: %31
                    pto::Shape<1, 1, 1, 1, 512> v40 = pto::Shape<1, 1, 1, 1, 512>();
                    // pto: %31
                    pto::Stride<512, 512, 512, 512, 1> v41 = pto::Stride<512, 512, 512, 512, 1>();
                    // pto: %30, %31
                    GlobalTensor<
                        bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>
                        v42 = GlobalTensor<
                            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>,
                            pto::Layout::ND>(v3 + (v16 + (v37 < v16 ? v16 : v37) * v15), v40, v41);
                    TLOAD(v38, v42);
                    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
                    // pto: %swa_kv_flat_inline1000_inline9849__iter_v4_pview
                    pto::Shape<1, 1, 1, 1, 512> v43 = pto::Shape<1, 1, 1, 1, 512>();
                    // pto: %swa_kv_flat_inline1000_inline9849__iter_v4_pview
                    pto::Stride<512, 512, 512, 512, 1> v44 = pto::Stride<512, 512, 512, 512, 1>();
                    // pto: %33, %swa_kv_flat_inline1000_inline9849__iter_v4_pview
                    GlobalTensor<
                        bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>
                        v45 = GlobalTensor<
                            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>,
                            pto::Layout::ND>(v1 + (v16 + (v35 < v16 ? v16 : v35) * v15), v43, v44);
                    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
                    TSTORE(v45, v38);
                } else {
                    // pto: %1
                    Tile<
                        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                        PadValue::Null, CompactMode::Null>
                        v46 = Tile<
                            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                            PadValue::Null, CompactMode::Null>(v14, v15);
                    // pto: %1
                    uint64_t v47 = (uint64_t)v16;
                    TASSIGN(v46, v47);
                    TEXPANDS(v46, v7);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    // pto: %36
                    pto::Shape<1, 1, 1, 1, 512> v48 = pto::Shape<1, 1, 1, 1, 512>();
                    // pto: %36
                    pto::Stride<512, 512, 512, 512, 1> v49 = pto::Stride<512, 512, 512, 512, 1>();
                    // pto: %35, %36
                    GlobalTensor<
                        bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>
                        v50 = GlobalTensor<
                            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>,
                            pto::Layout::ND>(v1 + (v16 + (v35 < v16 ? v16 : v35) * v15), v48, v49);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    TSTORE(v50, v46);
                }
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            }
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: swa_kv_flat_inline1000_inline9849__ssa_v0
    __gm__ Tensor *swa_kv_flat_inline1000_inline9849__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ bfloat16_t *swa_kv_flat_inline1000_inline9849__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(swa_kv_flat_inline1000_inline9849__ssa_v0_tensor->buffer.addr) +
        swa_kv_flat_inline1000_inline9849__ssa_v0_tensor->start_offset;

    // Unpack tensor: swa_indices_inline636__ssa_v1
    __gm__ Tensor *swa_indices_inline636__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int32_t *swa_indices_inline636__ssa_v1 =
        reinterpret_cast<__gm__ int32_t *>(swa_indices_inline636__ssa_v1_tensor->buffer.addr) +
        swa_indices_inline636__ssa_v1_tensor->start_offset;

    // Unpack tensor: ori_kv_flat_inline1033_inline9732__ssa_v0
    __gm__ Tensor *ori_kv_flat_inline1033_inline9732__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ bfloat16_t *ori_kv_flat_inline1033_inline9732__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(ori_kv_flat_inline1033_inline9732__ssa_v0_tensor->buffer.addr) +
        ori_kv_flat_inline1033_inline9732__ssa_v0_tensor->start_offset;

    // Extract dynamic dim: ori_block_num_inline1046_inline9838__ssa_v0
    int64_t ori_block_num_inline1046_inline9838__ssa_v0 =
        (static_cast<int64_t>(ori_kv_flat_inline1033_inline9732__ssa_v0_tensor->shapes[0]) / 128);

    // Forward to ptoas-generated function
    swa_gather_kv_0(
        swa_kv_flat_inline1000_inline9849__ssa_v0, swa_indices_inline636__ssa_v1,
        ori_kv_flat_inline1033_inline9732__ssa_v0, ori_block_num_inline1046_inline9838__ssa_v0, __pypto_spmd_block_idx,
        __pypto_spmd_block_num
    );
}
