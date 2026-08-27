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
// Kernel Function: swa_cache_insert_valid_bias_0

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

static __aicore__ void swa_cache_insert_valid_bias_0(
    __gm__ bfloat16_t *v1, __gm__ int64_t *v2, __gm__ bfloat16_t *v3, __gm__ int32_t *v4, __gm__ float *v5, int64_t v6
) {
    SaturationMode v7 = SaturationMode::OFF;
    RoundMode v8 = RoundMode::CAST_ROUND;
    const float v9 = 1.00000002E+20f;
    const float v10 = 1.0f;
    const float v11 = 0.0f;
    const int32_t v12 = 0;
    const int64_t v13 = 128;
    const int64_t v14 = 8;
    const int64_t v15 = 1;
    const int64_t v16 = 512;
    const int64_t v17 = 4096;
    const int64_t v18 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    for (int64_t i19 = v18; i19 < v14; i19 += v15) {
        // pto: %write_row_i64_inline9598__tile
        int64_t v20 = (v2)[i19];
        // pto: %10
        if (v20 >= v18) {
            // pto: %t__tile
            Tile<
                TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v21 = Tile<
                    TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v15, v16);
            // pto: %t__tile
            uint64_t v22 = (uint64_t)v18;
            TASSIGN(v21, v22);
            // pto: %kv_inline9787__ssa_v0_pview
            pto::Shape<1, 1, 1, 1, 512> v23 = pto::Shape<1, 1, 1, 1, 512>();
            // pto: %kv_inline9787__ssa_v0_pview
            pto::Stride<512, 512, 512, 512, 1> v24 = pto::Stride<512, 512, 512, 512, 1>();
            // pto: %12, %kv_inline9787__ssa_v0_pview
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>
                v25 = GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(
                    v3 + (v18 + (i19 < v18 ? v18 : i19) * v16), v23, v24
                );
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            TLOAD(v21, v25);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            // pto: %kv_cache_flat_inline9842__iter_v1_pview
            pto::Shape<1, 1, 1, 1, 512> v26 = pto::Shape<1, 1, 1, 1, 512>();
            // pto: %kv_cache_flat_inline9842__iter_v1_pview
            pto::Stride<512, 512, 512, 512, 1> v27 = pto::Stride<512, 512, 512, 512, 1>();
            // pto: %13, %kv_cache_flat_inline9842__iter_v1_pview
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>
                v28 = GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(
                    v1 + (v18 + (v20 < v18 ? v18 : v20) * v16), v26, v27
                );
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(v28, v21);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    // pto: %0
    Tile<
        TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v29 = Tile<
            TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v13);
    // pto: %0
    uint64_t v30 = (uint64_t)v18;
    TASSIGN(v29, v30);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    TCI<Tile<
            TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>,
        int32_t, 0>(v29, v12);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    // pto: %v_col_inline9808__tile
    Tile<
        TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v31 = Tile<
            TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v13);
    // pto: %v_col_inline9808__tile
    uint64_t v32 = (uint64_t)v17;
    TASSIGN(v31, v32);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    TCVT(v31, v29, v8, v7);
    // pto: %1
    Tile<
        TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v33 = Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v13);
    // pto: %1
    uint64_t v34 = (uint64_t)v18;
    TASSIGN(v33, v34);
    pipe_barrier(PIPE_V);
    TEXPANDS(v33, v11);
    // pto: %v_col_m_inline9729__tile
    Tile<
        TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v35 = Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v13);
    // pto: %v_col_m_inline9729__tile
    uint64_t v36 = (uint64_t)v18;
    TASSIGN(v35, v36);
    pipe_barrier(PIPE_V);
    TCOLEXPAND(v35, v31);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    // pto: %2
    Tile<
        TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v37 = Tile<
            TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v14);
    // pto: %2
    uint64_t v38 = (uint64_t)v17;
    TASSIGN(v37, v38);
    // pto: %swa_lens_inline628__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 8> v39 = pto::Shape<1, 1, 1, 1, 8>();
    // pto: %swa_lens_inline628__ssa_v0_pview
    pto::Stride<8, 8, 8, 8, 1> v40 = pto::Stride<8, 8, 8, 8, 1>();
    // pto: %swa_lens_inline628__ssa_v0_pview
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND> v41 =
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND>(v4, v39, v40);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v37, v41);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %3
    Tile<
        TileType::Vec, int32_t, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v42 = Tile<
            TileType::Vec, int32_t, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v15);
    // pto: %3
    uint64_t v43 = (uint64_t)v17;
    TASSIGN(v42, v43);
    // pto: %v_lens_inline9610__rm_a0_tmp_v0
    Tile<
        TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v44 = Tile<
            TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v14);
    // pto: %v_lens_inline9610__rm_a0_tmp_v0
    uint64_t v45 = (uint64_t)v17;
    TASSIGN(v44, v45);
    // pto: %v_lens_inline9610__row_major_tmp_v1
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v46 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v14);
    // pto: %v_lens_inline9610__row_major_tmp_v1
    uint64_t v47 = (uint64_t)v17;
    TASSIGN(v46, v47);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(v46, v44, v8, v7);
    // pto: %v_lens_inline9610__tile
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v48 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v15);
    // pto: %v_lens_inline9610__tile
    uint64_t v49 = (uint64_t)v17;
    TASSIGN(v48, v49);
    // pto: %4
    Tile<
        TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v50 = Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v13);
    // pto: %4
    uint64_t v51 = (uint64_t)v18;
    TASSIGN(v50, v51);
    pipe_barrier(PIPE_V);
    TROWEXPANDSUB(v50, v35, v48);
    // pto: %5
    Tile<
        TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v52 = Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v13);
    // pto: %5
    uint64_t v53 = (uint64_t)v18;
    TASSIGN(v52, v53);
    pipe_barrier(PIPE_V);
    TNEG(v52, v50);
    // pto: %6
    Tile<
        TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v54 = Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v13);
    // pto: %6
    uint64_t v55 = (uint64_t)v18;
    TASSIGN(v54, v55);
    pipe_barrier(PIPE_V);
    TMAXS(v54, v52, v11);
    // pto: %v_valid_inline9845__tile
    Tile<
        TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v56 = Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v13);
    // pto: %v_valid_inline9845__tile
    uint64_t v57 = (uint64_t)v18;
    TASSIGN(v56, v57);
    pipe_barrier(PIPE_V);
    TMINS(v56, v54, v10);
    // pto: %7
    Tile<
        TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v58 = Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v13);
    // pto: %7
    uint64_t v59 = (uint64_t)v18;
    TASSIGN(v58, v59);
    pipe_barrier(PIPE_V);
    TSUBS(v58, v56, v10);
    // pto: %8
    Tile<
        TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v60 = Tile<
            TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v13);
    // pto: %8
    uint64_t v61 = (uint64_t)v18;
    TASSIGN(v60, v61);
    pipe_barrier(PIPE_V);
    TMULS(v60, v58, v9);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %sparse_bias_inline9764__ssa_v0_pview
    pto::Shape<1, 1, 1, 8, 128> v62 = pto::Shape<1, 1, 1, 8, 128>();
    // pto: %sparse_bias_inline9764__ssa_v0_pview
    pto::Stride<1024, 1024, 1024, 128, 1> v63 = pto::Stride<1024, 1024, 1024, 128, 1>();
    // pto: %sparse_bias_inline9764__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<1024, 1024, 1024, 128, 1>, pto::Layout::ND> v64 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<1024, 1024, 1024, 128, 1>, pto::Layout::ND>(
            v5, v62, v63
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v64, v60);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Unpack tensor: kv_cache_flat_inline9842__ssa_v0
    __gm__ Tensor *kv_cache_flat_inline9842__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ bfloat16_t *kv_cache_flat_inline9842__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(kv_cache_flat_inline9842__ssa_v0_tensor->buffer.addr) +
        kv_cache_flat_inline9842__ssa_v0_tensor->start_offset;

    // Unpack tensor: swa_slot_mapping_inline576__ssa_v0
    __gm__ Tensor *swa_slot_mapping_inline576__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int64_t *swa_slot_mapping_inline576__ssa_v0 =
        reinterpret_cast<__gm__ int64_t *>(swa_slot_mapping_inline576__ssa_v0_tensor->buffer.addr) +
        swa_slot_mapping_inline576__ssa_v0_tensor->start_offset;

    // Unpack tensor: kv_inline9787__ssa_v0
    __gm__ Tensor *kv_inline9787__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ bfloat16_t *kv_inline9787__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(kv_inline9787__ssa_v0_tensor->buffer.addr) +
        kv_inline9787__ssa_v0_tensor->start_offset;

    // Unpack tensor: swa_lens_inline628__ssa_v0
    __gm__ Tensor *swa_lens_inline628__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ int32_t *swa_lens_inline628__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(swa_lens_inline628__ssa_v0_tensor->buffer.addr) +
        swa_lens_inline628__ssa_v0_tensor->start_offset;

    // Unpack tensor: sparse_bias_inline9764__ssa_v0
    __gm__ Tensor *sparse_bias_inline9764__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ float *sparse_bias_inline9764__ssa_v0 =
        reinterpret_cast<__gm__ float *>(sparse_bias_inline9764__ssa_v0_tensor->buffer.addr) +
        sparse_bias_inline9764__ssa_v0_tensor->start_offset;

    // Extract dynamic dim: ori_block_num_inline9841__ssa_v0
    int64_t ori_block_num_inline9841__ssa_v0 =
        (static_cast<int64_t>(kv_cache_flat_inline9842__ssa_v0_tensor->shapes[0]) / 128);

    // Forward to ptoas-generated function
    swa_cache_insert_valid_bias_0(
        kv_cache_flat_inline9842__ssa_v0, swa_slot_mapping_inline576__ssa_v0, kv_inline9787__ssa_v0,
        swa_lens_inline628__ssa_v0, sparse_bias_inline9764__ssa_v0, ori_block_num_inline9841__ssa_v0
    );
}
