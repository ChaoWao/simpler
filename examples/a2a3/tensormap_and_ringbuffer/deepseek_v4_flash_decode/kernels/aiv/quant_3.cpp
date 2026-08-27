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
// Kernel Function: quant_3

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

static __aicore__ void quant_3(__gm__ float *v1, __gm__ int8_t *v2, __gm__ float *v3, int64_t v4, int64_t v5) {
    RoundMode v6 = RoundMode::CAST_TRUNC;
    RoundMode v7 = RoundMode::CAST_ROUND;
    SaturationMode v8 = SaturationMode::OFF;
    RoundMode v9 = RoundMode::CAST_RINT;
    const half v10 = 0.0f;
    const float v11 = 127.0f;
    const float v12 = 9.99999974E-5f;
    const int64_t v13 = 1024;
    const int64_t v14 = 1;
    const int64_t v15 = 8;
    const int64_t v16 = 65536;
    const int64_t v17 = 32768;
    const int64_t v18 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %oc_amax_inline2067_inline12615__tile
    Tile<
        TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v19 = Tile<
            TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v13);
    // pto: %oc_amax_inline2067_inline12615__tile
    uint64_t v20 = (uint64_t)v18;
    TASSIGN(v19, v20);
    // pto: %0
    int64_t v21 = v4 < v18 ? v18 : v4;
    // pto: %o_r_pad_inline2225_inline12756__ssa_v3_pview
    pto::Shape<1, 1, 1, 8, 1024> v22 = pto::Shape<1, 1, 1, 8, 1024>();
    // pto: %o_r_pad_inline2225_inline12756__ssa_v3_pview
    pto::Stride<65536, 65536, 65536, 8192, 1> v23 = pto::Stride<65536, 65536, 65536, 8192, 1>();
    // pto: %o_r_pad_inline2225_inline12756__ssa_v3_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1024>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND> v24 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1024>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND>(
            v3 + (v18 + v21), v22, v23
        );
    TLOAD(v19, v24);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %g_abs_inline2174_inline12187__tile
    Tile<
        TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v25 = Tile<
            TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v13);
    // pto: %g_abs_inline2174_inline12187__tile
    uint64_t v26 = (uint64_t)v18;
    TASSIGN(v25, v26);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TABS(v25, v19);
    // pto: %tmp_tile
    Tile<
        TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v27 = Tile<
            TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v13);
    // pto: %tmp_tile
    uint64_t v28 = (uint64_t)v17;
    TASSIGN(v27, v28);
    // pto: %g_row_max_inline2066_inline12186__tile
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v29 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v14);
    // pto: %g_row_max_inline2066_inline12186__tile
    uint64_t v30 = (uint64_t)v16;
    TASSIGN(v29, v30);
    pipe_barrier(PIPE_V);
    TROWMAX(v29, v25, v27);
    // pto: %g_row_max_v1_inline2064_inline12375__tile
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v31 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v15);
    // pto: %g_row_max_v1_inline2064_inline12375__tile
    uint64_t v32 = (uint64_t)v16;
    TASSIGN(v31, v32);
    // pto: %g_amax_floor_inline2063_inline12770__tile
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v33 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v15);
    // pto: %g_amax_floor_inline2063_inline12770__tile
    uint64_t v34 = (uint64_t)v18;
    TASSIGN(v33, v34);
    pipe_barrier(PIPE_V);
    TEXPANDS(v33, v12);
    // pto: %g_amax_inline2115_inline12185__tile
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v35 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v15);
    // pto: %g_amax_inline2115_inline12185__tile
    uint64_t v36 = (uint64_t)v18;
    TASSIGN(v35, v36);
    pipe_barrier(PIPE_V);
    TMAX(v35, v33, v31);
    // pto: %g_scale_num_inline2062_inline12184__tile
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v37 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v15);
    // pto: %g_scale_num_inline2062_inline12184__tile
    uint64_t v38 = (uint64_t)v17;
    TASSIGN(v37, v38);
    TEXPANDS(v37, v11);
    // pto: %g_sq_row_inline2061_inline12589__tile
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v39 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v15);
    // pto: %g_sq_row_inline2061_inline12589__tile
    uint64_t v40 = (uint64_t)v17;
    TASSIGN(v39, v40);
    pipe_barrier(PIPE_V);
    TDIV(v39, v37, v35);
    // pto: %t__tile
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v41 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v14, v15);
    // pto: %t__tile
    uint64_t v42 = (uint64_t)v18;
    TASSIGN(v41, v42);
    pipe_barrier(PIPE_V);
    TRECIP(v41, v39);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %act_scale_dq_inline2081_inline12776__iter_v1_pview
    pto::Shape<1, 1, 1, 1, 8> v43 = pto::Shape<1, 1, 1, 1, 8>();
    // pto: %act_scale_dq_inline2081_inline12776__iter_v1_pview
    pto::Stride<8, 8, 8, 8, 1> v44 = pto::Stride<8, 8, 8, 8, 1>();
    // pto: %1, %act_scale_dq_inline2081_inline12776__iter_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND> v45 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND>(
            v1 + (v18 + (v5 < v18 ? v18 : v5) * v15), v43, v44
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v45, v41);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    // pto: %g_sq_col_inline2060_inline12463__tile
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v46 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v14);
    // pto: %g_sq_col_inline2060_inline12463__tile
    uint64_t v47 = (uint64_t)v17;
    TASSIGN(v46, v47);
    // pto: %oc_q_inline2176_inline12183__tile
    Tile<
        TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v48 = Tile<
            TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v13);
    // pto: %oc_q_inline2176_inline12183__tile
    uint64_t v49 = (uint64_t)v18;
    TASSIGN(v48, v49);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v48, v24);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %oq_scaled_inline2059_inline12678__tile
    Tile<
        TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v50 = Tile<
            TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v13);
    // pto: %oq_scaled_inline2059_inline12678__tile
    uint64_t v51 = (uint64_t)v18;
    TASSIGN(v50, v51);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TROWEXPANDMUL(v50, v48, v46);
    // pto: %oq_i32_inline2110_inline12182__tile
    Tile<
        TileType::Vec, int32_t, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v52 = Tile<
            TileType::Vec, int32_t, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v13);
    // pto: %oq_i32_inline2110_inline12182__tile
    uint64_t v53 = (uint64_t)v18;
    TASSIGN(v52, v53);
    pipe_barrier(PIPE_V);
    TCVT(v52, v50, v9, v8);
    // pto: %oq_half_inline2068_inline12181__tile
    Tile<
        TileType::Vec, half, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v54 = Tile<
            TileType::Vec, half, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v13);
    // pto: %oq_half_inline2068_inline12181__tile
    uint64_t v55 = (uint64_t)v18;
    TASSIGN(v54, v55);
    pipe_barrier(PIPE_V);
    TCVT(v54, v52, v7, v8);
    // pto: %oq_i8_inline2058_inline12180__tile
    Tile<
        TileType::Vec, int8_t, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v56 = Tile<
            TileType::Vec, int8_t, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v13);
    // pto: %oq_i8_inline2058_inline12180__tile
    uint64_t v57 = (uint64_t)v18;
    TASSIGN(v56, v57);
    pipe_barrier(PIPE_V);
    TCVT(v56, v54, v6, v8);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %o_r_i8_pad_inline2082_inline12196__iter_v1_pview
    pto::Shape<1, 1, 1, 8, 1024> v58 = pto::Shape<1, 1, 1, 8, 1024>();
    // pto: %o_r_i8_pad_inline2082_inline12196__iter_v1_pview
    pto::Stride<65536, 65536, 65536, 8192, 1> v59 = pto::Stride<65536, 65536, 65536, 8192, 1>();
    // pto: %o_r_i8_pad_inline2082_inline12196__iter_v1_pview
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 1024>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND> v60 =
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 1024>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND>(
            v2 + (v18 + v21), v58, v59
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v60, v56);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    // pto: %zero_half_inline2095_inline12179__tile
    Tile<
        TileType::Vec, half, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v61 = Tile<
            TileType::Vec, half, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v13);
    // pto: %zero_half_inline2095_inline12179__tile
    uint64_t v62 = (uint64_t)v18;
    TASSIGN(v61, v62);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(v61, v10);
    // pto: %zero_i8_inline2157_inline12597__tile
    Tile<
        TileType::Vec, int8_t, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v63 = Tile<
            TileType::Vec, int8_t, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v15, v13);
    // pto: %zero_i8_inline2157_inline12597__tile
    uint64_t v64 = (uint64_t)v18;
    TASSIGN(v63, v64);
    pipe_barrier(PIPE_V);
    TCVT(v63, v61, v6, v8);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    // pto: %o_r_i8_pad_inline2082_inline12196__tile_pview
    pto::Shape<1, 1, 1, 8, 1024> v65 = pto::Shape<1, 1, 1, 8, 1024>();
    // pto: %o_r_i8_pad_inline2082_inline12196__tile_pview
    pto::Stride<65536, 65536, 65536, 8192, 1> v66 = pto::Stride<65536, 65536, 65536, 8192, 1>();
    // pto: %o_r_i8_pad_inline2082_inline12196__tile_pview
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 1024>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND> v67 =
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 1024>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND>(
            v2 + (v16 + v21), v65, v66
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    TSTORE(v67, v63);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Unpack tensor: act_scale_dq_inline2081_inline12776__iter_v1
    __gm__ Tensor *act_scale_dq_inline2081_inline12776__iter_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *act_scale_dq_inline2081_inline12776__iter_v1 =
        reinterpret_cast<__gm__ float *>(act_scale_dq_inline2081_inline12776__iter_v1_tensor->buffer.addr) +
        act_scale_dq_inline2081_inline12776__iter_v1_tensor->start_offset;

    // Unpack tensor: o_r_i8_pad_inline2082_inline12196__iter_v1
    __gm__ Tensor *o_r_i8_pad_inline2082_inline12196__iter_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int8_t *o_r_i8_pad_inline2082_inline12196__iter_v1 =
        reinterpret_cast<__gm__ int8_t *>(o_r_i8_pad_inline2082_inline12196__iter_v1_tensor->buffer.addr) +
        o_r_i8_pad_inline2082_inline12196__iter_v1_tensor->start_offset;

    // Unpack tensor: o_r_pad_inline2225_inline12756__ssa_v3
    __gm__ Tensor *o_r_pad_inline2225_inline12756__ssa_v3_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *o_r_pad_inline2225_inline12756__ssa_v3 =
        reinterpret_cast<__gm__ float *>(o_r_pad_inline2225_inline12756__ssa_v3_tensor->buffer.addr) +
        o_r_pad_inline2225_inline12756__ssa_v3_tensor->start_offset;

    // Unpack scalar: col_g_inline2196_inline12308__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } col_g_inline2196_inline12308__ssa_v0_conv;
    col_g_inline2196_inline12308__ssa_v0_conv.u64 = args[3];
    int64_t col_g_inline2196_inline12308__ssa_v0 = col_g_inline2196_inline12308__ssa_v0_conv.val;

    // Unpack scalar: g_inline2162_inline12195__idx_v0
    union {
        uint64_t u64;
        int64_t val;
    } g_inline2162_inline12195__idx_v0_conv;
    g_inline2162_inline12195__idx_v0_conv.u64 = args[4];
    int64_t g_inline2162_inline12195__idx_v0 = g_inline2162_inline12195__idx_v0_conv.val;

    // Forward to ptoas-generated function
    quant_3(
        act_scale_dq_inline2081_inline12776__iter_v1, o_r_i8_pad_inline2082_inline12196__iter_v1,
        o_r_pad_inline2225_inline12756__ssa_v3, col_g_inline2196_inline12308__ssa_v0, g_inline2162_inline12195__idx_v0
    );
}
