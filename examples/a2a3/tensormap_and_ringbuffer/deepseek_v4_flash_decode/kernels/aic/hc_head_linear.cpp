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
// Kernel Function: hc_head_linear

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
hc_head_linear(__gm__ float *v1, __gm__ float *v2, __gm__ float *v3, int64_t v4, int64_t v5, int32_t v6, int32_t v7) {
    const int64_t v8 = 256;
    const int64_t v9 = 2;
    const int64_t v10 = 1024;
    const int64_t v11 = 16;
    const int64_t v12 = 4;
    const int64_t v13 = 49152;
    const int64_t v14 = 32768;
    const int64_t v15 = 16384;
    const int64_t v16 = 0;
    using T = float;

#if defined(__DAV_CUBE__)
    // pto: %acc_inline13214__phi_v5
    Tile<
        TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v17 = Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v11, v12);
    // pto: %acc_inline13214__phi_v5
    uint64_t v18 = (uint64_t)v16;
    TASSIGN(v17, v18);
    // pto: %task_inline13236__ssa_v0
    int64_t v19 = (int64_t)v6;
    // pto: %11, %12
    int64_t v20 = (int64_t)((uint64_t)(v19 / v11) * (uint64_t)v11);
    // pto: %13, %14
    int64_t v21 = (int64_t)((uint64_t)(v19 % v11) * (uint64_t)v10);
    // The output is padded to 16 rows, but x_flat contains only t_dim rows.
    // Keep the cube tile capacity at 16 while restricting all x loads and
    // matmuls to the rows that are actually present in this block.
    int64_t remaining_rows = v4 - v20;
    int64_t valid_rows = remaining_rows < v16 ? v16 : (remaining_rows < v11 ? remaining_rows : v11);
    // pto: %acc_inline13214__tile
    Tile<
        TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v22 = Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(valid_rows, v11);
    // pto: %acc_inline13214__tile
    uint64_t v23 = (uint64_t)v16;
    TASSIGN(v22, v23);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    for (int64_t i24 = v16; i24 < v12; i24 += v9) {
        // pto: %15
        int64_t v25 = (int64_t)((uint64_t)i24 * (uint64_t)v8);
        // pto: %16
        int64_t v26 = (int64_t)((uint64_t)v21 + (uint64_t)v25);
        // pto: %19, %18
        int64_t v27 = (int64_t)((uint64_t)v21 + (uint64_t)((int64_t)((uint64_t)v25 + (uint64_t)v8)));
        // pto: %x_lin_inline13225__tile
        Tile<
            TileType::Mat, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v28 = Tile<
                TileType::Mat, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(valid_rows, v8);
        // pto: %x_lin_inline13225__tile
        uint64_t v29 = (uint64_t)v16;
        TASSIGN(v28, v29);
        // pto: %20
        int64_t v30 = v20 < v16 ? v16 : v20;
        // pto: %21
        int64_t v31 = v26 < v16 ? v16 : v26;
        // pto: %x_flat_inline13229__ssa_v0_pview
        pto::Shape<1, 1, 1, -1, 256> v32 = pto::Shape<1, 1, 1, -1, 256>(1, 1, 1, valid_rows, v8);
        // pto: %x_flat_inline13229__ssa_v0_pview
        pto::Stride<262144, 262144, 262144, 16384, 1> v33 = pto::Stride<262144, 262144, 262144, 16384, 1>();
        // pto: %x_flat_inline13229__ssa_v0_pview
        GlobalTensor<
            float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<262144, 262144, 262144, 16384, 1>, pto::Layout::ND>
            v34 = GlobalTensor<
                float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<262144, 262144, 262144, 16384, 1>, pto::Layout::ND>(
                v1 + ((v16 + v30 * v15) + v31), v32, v33
            );
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        TLOAD(v28, v34);
        // pto: %w_inline13235__tile
        Tile<
            TileType::Mat, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v35 = Tile<
                TileType::Mat, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v12, v8);
        // pto: %w_inline13235__tile
        uint64_t v36 = (uint64_t)v15;
        TASSIGN(v35, v36);
        // pto: %hc_head_fn__ssa_v0_pview
        pto::Shape<1, 1, 1, 4, 256> v37 = pto::Shape<1, 1, 1, 4, 256>();
        // pto: %hc_head_fn__ssa_v0_pview
        pto::Stride<65536, 65536, 65536, 16384, 1> v38 = pto::Stride<65536, 65536, 65536, 16384, 1>();
        // pto: %hc_head_fn__ssa_v0_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 4, 256>, pto::Stride<65536, 65536, 65536, 16384, 1>, pto::Layout::ND>
            v39 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 4, 256>, pto::Stride<65536, 65536, 65536, 16384, 1>, pto::Layout::ND>(
                v2 + (v16 + v31), v37, v38
            );
        TLOAD(v35, v39);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        // pto: %0
        Tile<
            TileType::Mat, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v40 = Tile<
                TileType::Mat, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(valid_rows, v8);
        // pto: %0
        uint64_t v41 = (uint64_t)v14;
        TASSIGN(v40, v41);
        // pto: %24
        int64_t v42 = v27 < v16 ? v16 : v27;
        // pto: %25
        pto::Shape<1, 1, 1, -1, 256> v43 = pto::Shape<1, 1, 1, -1, 256>(1, 1, 1, valid_rows, v8);
        // pto: %25
        pto::Stride<262144, 262144, 262144, 16384, 1> v44 = pto::Stride<262144, 262144, 262144, 16384, 1>();
        // pto: %25
        GlobalTensor<
            float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<262144, 262144, 262144, 16384, 1>, pto::Layout::ND>
            v45 = GlobalTensor<
                float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<262144, 262144, 262144, 16384, 1>, pto::Layout::ND>(
                v1 + ((v16 + v30 * v15) + v42), v43, v44
            );
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        TLOAD(v40, v45);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        // pto: %1
        Tile<
            TileType::Mat, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v46 = Tile<
                TileType::Mat, float, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v12, v8);
        // pto: %1
        uint64_t v47 = (uint64_t)v13;
        TASSIGN(v46, v47);
        // pto: %27
        pto::Shape<1, 1, 1, 4, 256> v48 = pto::Shape<1, 1, 1, 4, 256>();
        // pto: %27
        pto::Stride<65536, 65536, 65536, 16384, 1> v49 = pto::Stride<65536, 65536, 65536, 16384, 1>();
        // pto: %27
        GlobalTensor<float, pto::Shape<1, 1, 1, 4, 256>, pto::Stride<65536, 65536, 65536, 16384, 1>, pto::Layout::ND>
            v50 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 4, 256>, pto::Stride<65536, 65536, 65536, 16384, 1>, pto::Layout::ND>(
                v2 + (v16 + v42), v48, v49
            );
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        TLOAD(v46, v50);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        // pto: %28
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        if (i24 == v16) {
            // pto: %w_inline13235__tile_t
            Tile<
                TileType::Mat, float, 256, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v51 = Tile<
                    TileType::Mat, float, 256, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                    CompactMode::Null>(v8, v12);
            // pto: %w_inline13235__tile_t
            uint64_t v52 = (uint64_t)v15;
            TASSIGN(v51, v52);
            // pto: %x_lin_inline13225__tile_Left
            Tile<
                TileType::Left, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v53 = Tile<
                    TileType::Left, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>(valid_rows, v8);
            // pto: %x_lin_inline13225__tile_Left
            uint64_t v54 = (uint64_t)v15;
            TASSIGN(v53, v54);
            TMOV(v53, v28);
            // pto: %w_inline13235__tile_t_Right
            Tile<
                TileType::Right, float, 256, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v55 = Tile<
                    TileType::Right, float, 256, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                    CompactMode::Null>(v8, v12);
            // pto: %w_inline13235__tile_t_Right
            uint64_t v56 = (uint64_t)v15;
            TASSIGN(v55, v56);
            TMOV(v55, v51);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            // pto: %2
            Tile<
                TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v57 = Tile<
                    TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(valid_rows, v12);
            // pto: %2
            uint64_t v58 = (uint64_t)v16;
            TASSIGN(v57, v58);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            pipe_barrier(PIPE_M);
            TMATMUL(v57, v53, v55);
        } else {
            // pto: %3
            Tile<
                TileType::Mat, float, 256, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v59 = Tile<
                    TileType::Mat, float, 256, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                    CompactMode::Null>(v8, v12);
            // pto: %3
            uint64_t v60 = (uint64_t)v15;
            TASSIGN(v59, v60);
            // pto: %4
            Tile<
                TileType::Left, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v61 = Tile<
                    TileType::Left, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>(valid_rows, v8);
            // pto: %4
            uint64_t v62 = (uint64_t)v15;
            TASSIGN(v61, v62);
            TMOV(v61, v28);
            // pto: %5
            Tile<
                TileType::Right, float, 256, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v63 = Tile<
                    TileType::Right, float, 256, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                    CompactMode::Null>(v8, v12);
            // pto: %5
            uint64_t v64 = (uint64_t)v15;
            TASSIGN(v63, v64);
            TMOV(v63, v59);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
            // pto: %6
            Tile<
                TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v65 = Tile<
                    TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(valid_rows, v11);
            // pto: %6
            uint64_t v66 = (uint64_t)v16;
            TASSIGN(v65, v66);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v65, v65, v61, v63);
        }
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        // pto: %7
        Tile<
            TileType::Mat, float, 256, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v67 = Tile<
                TileType::Mat, float, 256, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v8, v12);
        // pto: %7
        uint64_t v68 = (uint64_t)v13;
        TASSIGN(v67, v68);
        // pto: %8
        Tile<
            TileType::Left, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v69 = Tile<
                TileType::Left, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(valid_rows, v8);
        // pto: %8
        uint64_t v70 = (uint64_t)v16;
        TASSIGN(v69, v70);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TMOV(v69, v40);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        // pto: %9
        Tile<
            TileType::Right, float, 256, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v71 = Tile<
                TileType::Right, float, 256, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v8, v12);
        // pto: %9
        uint64_t v72 = (uint64_t)v16;
        TASSIGN(v71, v72);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        TMOV(v71, v67);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        // pto: %10
        Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v73 = Tile<
                TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(valid_rows, v11);
        // pto: %10
        uint64_t v74 = (uint64_t)v16;
        TASSIGN(v73, v74);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v73, v73, v69, v71);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    // pto: %mixes_raw_inline13234__ssa_v0_pview
    pto::Shape<1, 1, 1, -1, 16> v75 = pto::Shape<1, 1, 1, -1, 16>(1, 1, 1, valid_rows, v11);
    // pto: %mixes_raw_inline13234__ssa_v0_pview
    pto::Stride<256, 256, 256, 16, 1> v76 = pto::Stride<256, 256, 256, 16, 1>();
    // pto: %29, %mixes_raw_inline13234__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v77 =
        GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(
            v3 + (v16 + (v20 < v16 ? v16 : v20) * v11), v75, v76
        );
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE<
        Tile<
            TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>,
        GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>,
        AtomicType::AtomicAdd>(v77, v22);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
#endif  // __DAV_CUBE__

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

    // Unpack tensor: hc_head_fn__ssa_v0
    __gm__ Tensor *hc_head_fn__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *hc_head_fn__ssa_v0 = reinterpret_cast<__gm__ float *>(hc_head_fn__ssa_v0_tensor->buffer.addr) +
                                       hc_head_fn__ssa_v0_tensor->start_offset;

    // Unpack tensor: mixes_raw_inline13234__ssa_v0
    __gm__ Tensor *mixes_raw_inline13234__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *mixes_raw_inline13234__ssa_v0 =
        reinterpret_cast<__gm__ float *>(mixes_raw_inline13234__ssa_v0_tensor->buffer.addr) +
        mixes_raw_inline13234__ssa_v0_tensor->start_offset;

    // Extract dynamic dim: t_dim_inline13231__ssa_v0
    int64_t t_dim_inline13231__ssa_v0 = static_cast<int64_t>(x_flat_inline13229__ssa_v0_tensor->shapes[0]);

    // Extract dynamic dim: t_linear_inline13230__ssa_v0
    int64_t t_linear_inline13230__ssa_v0 = static_cast<int64_t>(mixes_raw_inline13234__ssa_v0_tensor->shapes[0]);

    // Forward to ptoas-generated function
    hc_head_linear(
        x_flat_inline13229__ssa_v0, hc_head_fn__ssa_v0, mixes_raw_inline13234__ssa_v0, t_dim_inline13231__ssa_v0,
        t_linear_inline13230__ssa_v0, __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
