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
// Kernel Function: hca_rope

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

static __aicore__ void hca_rope(
    __gm__ float *v1, __gm__ float *v2, __gm__ bfloat16_t *v3, __gm__ bfloat16_t *v4, __gm__ int32_t *v5,
    __gm__ bfloat16_t *v6, __gm__ bfloat16_t *v7
) {
    RoundMode v8 = RoundMode::CAST_RINT;
    SaturationMode v9 = SaturationMode::OFF;
    RoundMode v10 = RoundMode::CAST_ROUND;
    const int64_t v11 = 128;
    const int64_t v12 = 2;
    const int64_t v13 = 64;
    const int64_t v14 = 1;
    const int64_t v15 = 32;
    const int64_t v16 = 4;
    const int64_t v17 = 0;
    const int64_t v18 = 256;
    const int64_t v19 = 384;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    for (int64_t i20 = v17; i20 < v16; i20 += v14) {
        // pto: %5
        int64_t v21 = (int64_t)((uint64_t)i20 * (uint64_t)v12);
        // pto: %first_pos_b_inline11557__tile
        int32_t v22 = (v5)[v21];
        // pto: %6
        int64_t v23 = (int64_t)v22;
        // pto: %10, %8, %7, %11
        int64_t v24 = (int64_t)((uint64_t)((int64_t)((uint64_t)v23 +
                                                     (uint64_t)((int64_t)((uint64_t)v11 - (uint64_t)(v23 % v11))))) -
                                (uint64_t)v11);
        // pto: %cmp_cos_row_inline11767__tile
        Tile<
            TileType::Vec, bfloat16_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v25 = Tile<
                TileType::Vec, bfloat16_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v14, v15);
        // pto: %cmp_cos_row_inline11767__tile
        uint64_t v26 = (uint64_t)v19;
        TASSIGN(v25, v26);
        // pto: %13
        int64_t v27 = v24 < v17 ? v17 : v24;
        // pto: %compressed_freqs_cos_inline559__ssa_v0_pview
        pto::Shape<1, 1, 1, 1, 32> v28 = pto::Shape<1, 1, 1, 1, 32>();
        // pto: %compressed_freqs_cos_inline559__ssa_v0_pview
        pto::Stride<64, 64, 64, 64, 1> v29 = pto::Stride<64, 64, 64, 64, 1>();
        // pto: %compressed_freqs_cos_inline559__ssa_v0_pview
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v30 =
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(
                v6 + (v17 + v27 * v13), v28, v29
            );
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v25, v30);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %cmp_sin_row_inline11550__tile
        Tile<
            TileType::Vec, bfloat16_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v31 = Tile<
                TileType::Vec, bfloat16_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v14, v15);
        // pto: %cmp_sin_row_inline11550__tile
        uint64_t v32 = (uint64_t)v18;
        TASSIGN(v31, v32);
        // pto: %compressed_freqs_sin_inline692__ssa_v0_pview
        pto::Shape<1, 1, 1, 1, 32> v33 = pto::Shape<1, 1, 1, 1, 32>();
        // pto: %compressed_freqs_sin_inline692__ssa_v0_pview
        pto::Stride<64, 64, 64, 64, 1> v34 = pto::Stride<64, 64, 64, 64, 1>();
        // pto: %compressed_freqs_sin_inline692__ssa_v0_pview
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v35 =
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(
                v7 + (v17 + v27 * v13), v33, v34
            );
        TLOAD(v31, v35);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %t__tile
        Tile<
            TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v36 = Tile<
                TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v14, v15);
        // pto: %t__tile
        uint64_t v37 = (uint64_t)v17;
        TASSIGN(v36, v37);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        TCVT(v36, v25, v10, v9);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %15
        int64_t v38 = i20 < v17 ? v17 : i20;
        // pto: %cmp_cos_inline11642__iter_v1_pview
        pto::Shape<1, 1, 1, 1, 32> v39 = pto::Shape<1, 1, 1, 1, 32>();
        // pto: %cmp_cos_inline11642__iter_v1_pview
        pto::Stride<32, 32, 32, 32, 1> v40 = pto::Stride<32, 32, 32, 32, 1>();
        // pto: %cmp_cos_inline11642__iter_v1_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v41 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(
                v1 + (v17 + v38 * v15), v39, v40
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v41, v36);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        // pto: %0
        Tile<
            TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v42 = Tile<
                TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v14, v15);
        // pto: %0
        uint64_t v43 = (uint64_t)v17;
        TASSIGN(v42, v43);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TCVT(v42, v31, v10, v9);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        // pto: %cmp_sin_inline11548__iter_v1_pview
        pto::Shape<1, 1, 1, 1, 32> v44 = pto::Shape<1, 1, 1, 1, 32>();
        // pto: %cmp_sin_inline11548__iter_v1_pview
        pto::Stride<32, 32, 32, 32, 1> v45 = pto::Stride<32, 32, 32, 32, 1>();
        // pto: %cmp_sin_inline11548__iter_v1_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v46 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(
                v2 + (v17 + v38 * v15), v44, v45
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSTORE(v46, v42);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        for (int64_t j47 = v17; j47 < v12; j47 += v14) {
            // pto: %18
            int64_t v48 = (int64_t)((uint64_t)v21 + (uint64_t)j47);
            // pto: %19
            int32_t v49 = (v5)[v48];
            // pto: %20
            int64_t v50 = (int64_t)v49;
            // pto: %1
            Tile<
                TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v51 = Tile<
                    TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v14, v13);
            // pto: %1
            uint64_t v52 = (uint64_t)v19;
            TASSIGN(v51, v52);
            // pto: %21
            int64_t v53 = v50 < v17 ? v17 : v50;
            // pto: %22
            pto::Shape<1, 1, 1, 1, 64> v54 = pto::Shape<1, 1, 1, 1, 64>();
            // pto: %22
            pto::Stride<64, 64, 64, 64, 1> v55 = pto::Stride<64, 64, 64, 64, 1>();
            // pto: %22
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v56 =
                GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(
                    v6 + (v17 + v53 * v13), v54, v55
                );
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            TLOAD(v51, v56);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
            // pto: %step_cos_row_inline11536__tile
            Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v57 = Tile<
                    TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v14, v13);
            // pto: %step_cos_row_inline11536__tile
            uint64_t v58 = (uint64_t)v17;
            TASSIGN(v57, v58);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
            TCVT(v57, v51, v10, v9);
            // pto: %2
            Tile<
                TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v59 = Tile<
                    TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v14, v13);
            // pto: %2
            uint64_t v60 = (uint64_t)v18;
            TASSIGN(v59, v60);
            // pto: %24
            pto::Shape<1, 1, 1, 1, 64> v61 = pto::Shape<1, 1, 1, 1, 64>();
            // pto: %24
            pto::Stride<64, 64, 64, 64, 1> v62 = pto::Stride<64, 64, 64, 64, 1>();
            // pto: %24
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v63 =
                GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(
                    v7 + (v17 + v53 * v13), v61, v62
                );
            TLOAD(v59, v63);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
            // pto: %step_sin_row_inline11572__tile
            Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v64 = Tile<
                    TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v14, v13);
            // pto: %step_sin_row_inline11572__tile
            uint64_t v65 = (uint64_t)v19;
            TASSIGN(v64, v65);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
            pipe_barrier(PIPE_V);
            TCVT(v64, v59, v10, v9);
            // pto: %3
            Tile<
                TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v66 = Tile<
                    TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v14, v13);
            // pto: %3
            uint64_t v67 = (uint64_t)v17;
            TASSIGN(v66, v67);
            TCVT(v66, v57, v8, v9);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
            // pto: %25
            int64_t v68 = v48 < v17 ? v17 : v48;
            // pto: %rope_cos_t_inline11615__iter_v3_pview
            pto::Shape<1, 1, 1, 1, 64> v69 = pto::Shape<1, 1, 1, 1, 64>();
            // pto: %rope_cos_t_inline11615__iter_v3_pview
            pto::Stride<64, 64, 64, 64, 1> v70 = pto::Stride<64, 64, 64, 64, 1>();
            // pto: %rope_cos_t_inline11615__iter_v3_pview
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v71 =
                GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(
                    v3 + (v17 + v68 * v13), v69, v70
                );
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
            TSTORE(v71, v66);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
            // pto: %4
            Tile<
                TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v72 = Tile<
                    TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v14, v13);
            // pto: %4
            uint64_t v73 = (uint64_t)v17;
            TASSIGN(v72, v73);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
            TCVT(v72, v64, v8, v9);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            // pto: %rope_sin_t_inline11554__iter_v3_pview
            pto::Shape<1, 1, 1, 1, 64> v74 = pto::Shape<1, 1, 1, 1, 64>();
            // pto: %rope_sin_t_inline11554__iter_v3_pview
            pto::Stride<64, 64, 64, 64, 1> v75 = pto::Stride<64, 64, 64, 64, 1>();
            // pto: %rope_sin_t_inline11554__iter_v3_pview
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v76 =
                GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(
                    v4 + (v17 + v68 * v13), v74, v75
                );
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
            TSTORE(v76, v72);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        }
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Unpack tensor: cmp_cos_inline11642__ssa_v0
    __gm__ Tensor *cmp_cos_inline11642__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *cmp_cos_inline11642__ssa_v0 =
        reinterpret_cast<__gm__ float *>(cmp_cos_inline11642__ssa_v0_tensor->buffer.addr) +
        cmp_cos_inline11642__ssa_v0_tensor->start_offset;

    // Unpack tensor: cmp_sin_inline11548__ssa_v0
    __gm__ Tensor *cmp_sin_inline11548__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *cmp_sin_inline11548__ssa_v0 =
        reinterpret_cast<__gm__ float *>(cmp_sin_inline11548__ssa_v0_tensor->buffer.addr) +
        cmp_sin_inline11548__ssa_v0_tensor->start_offset;

    // Unpack tensor: rope_cos_t_inline11615__ssa_v0
    __gm__ Tensor *rope_cos_t_inline11615__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ bfloat16_t *rope_cos_t_inline11615__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(rope_cos_t_inline11615__ssa_v0_tensor->buffer.addr) +
        rope_cos_t_inline11615__ssa_v0_tensor->start_offset;

    // Unpack tensor: rope_sin_t_inline11554__ssa_v0
    __gm__ Tensor *rope_sin_t_inline11554__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ bfloat16_t *rope_sin_t_inline11554__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(rope_sin_t_inline11554__ssa_v0_tensor->buffer.addr) +
        rope_sin_t_inline11554__ssa_v0_tensor->start_offset;

    // Unpack tensor: position_ids__ssa_v0
    __gm__ Tensor *position_ids__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ int32_t *position_ids__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(position_ids__ssa_v0_tensor->buffer.addr) +
        position_ids__ssa_v0_tensor->start_offset;

    // Unpack tensor: compressed_freqs_cos_inline559__ssa_v0
    __gm__ Tensor *compressed_freqs_cos_inline559__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[5]);
    __gm__ bfloat16_t *compressed_freqs_cos_inline559__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(compressed_freqs_cos_inline559__ssa_v0_tensor->buffer.addr) +
        compressed_freqs_cos_inline559__ssa_v0_tensor->start_offset;

    // Unpack tensor: compressed_freqs_sin_inline692__ssa_v0
    __gm__ Tensor *compressed_freqs_sin_inline692__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[6]);
    __gm__ bfloat16_t *compressed_freqs_sin_inline692__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(compressed_freqs_sin_inline692__ssa_v0_tensor->buffer.addr) +
        compressed_freqs_sin_inline692__ssa_v0_tensor->start_offset;

    // Forward to ptoas-generated function
    hca_rope(
        cmp_cos_inline11642__ssa_v0, cmp_sin_inline11548__ssa_v0, rope_cos_t_inline11615__ssa_v0,
        rope_sin_t_inline11554__ssa_v0, position_ids__ssa_v0, compressed_freqs_cos_inline559__ssa_v0,
        compressed_freqs_sin_inline692__ssa_v0
    );
}
