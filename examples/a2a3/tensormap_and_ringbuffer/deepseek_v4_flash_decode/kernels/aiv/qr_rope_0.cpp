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
// Kernel Function: qr_rope_0

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

static __aicore__ void qr_rope_0(
    __gm__ int32_t *v1, __gm__ float *v2, __gm__ float *v3, __gm__ float *v4, __gm__ bfloat16_t *v5, int32_t v6,
    int32_t v7
) {
    SaturationMode v8 = SaturationMode::OFF;
    RoundMode v9 = RoundMode::CAST_RINT;
    const int64_t v10 = 128;
    const int64_t v11 = 1;
    const int64_t v12 = 64;
    const int64_t v13 = 32;
    const int64_t v14 = 8448;
    const int64_t v15 = 8192;
    const int64_t v16 = 0;
    const int64_t v17 = 25600;
    const int64_t v18 = 17408;
    const int64_t v19 = 17152;
    const int64_t v20 = 16896;
    const int64_t v21 = 8704;
    const int64_t v22 = 256;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %idx_inline1975_inline12278__ssa_v0, %3
    int64_t v23 = (int64_t)((uint64_t)((int64_t)v6) * (uint64_t)v13);
    // pto: %4
    int64_t v24 = v23 / v10;
    // pto: %rope_swap_idx_inline1961_inline12708__tile
    Tile<
        TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v25 = Tile<
            TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v13, v12);
    // pto: %rope_swap_idx_inline1961_inline12708__tile
    uint64_t v26 = (uint64_t)v21;
    TASSIGN(v25, v26);
    // pto: %rope_swap_idx_t_inline1970_inline12704__ssa_v1_pview
    pto::Shape<1, 1, 1, 32, 64> v27 = pto::Shape<1, 1, 1, 32, 64>();
    // pto: %rope_swap_idx_t_inline1970_inline12704__ssa_v1_pview
    pto::Stride<2048, 2048, 2048, 64, 1> v28 = pto::Stride<2048, 2048, 2048, 64, 1>();
    // pto: %rope_swap_idx_t_inline1970_inline12704__ssa_v1_pview
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v29 =
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(
            v1, v27, v28
        );
    TLOAD(v25, v29);
    // pto: %cos_row_inline1956_inline12711__tile
    Tile<
        TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v30 = Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v11, v12);
    // pto: %cos_row_inline1956_inline12711__tile
    uint64_t v31 = (uint64_t)v20;
    TASSIGN(v30, v31);
    // pto: %5
    int64_t v32 = v24 < v16 ? v16 : v24;
    // pto: %step_cos_il_inline12430__ssa_v1_pview
    pto::Shape<1, 1, 1, 1, 64> v33 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %step_cos_il_inline12430__ssa_v1_pview
    pto::Stride<64, 64, 64, 64, 1> v34 = pto::Stride<64, 64, 64, 64, 1>();
    // pto: %step_cos_il_inline12430__ssa_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v35 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(
            v2 + (v16 + v32 * v12), v33, v34
        );
    TLOAD(v30, v35);
    // pto: %sin_row_inline1960_inline12712__tile
    Tile<
        TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v36 = Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v11, v12);
    // pto: %sin_row_inline1960_inline12712__tile
    uint64_t v37 = (uint64_t)v19;
    TASSIGN(v36, v37);
    // pto: %step_sin_signed_inline12365__ssa_v1_pview
    pto::Shape<1, 1, 1, 1, 64> v38 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %step_sin_signed_inline12365__ssa_v1_pview
    pto::Stride<64, 64, 64, 64, 1> v39 = pto::Stride<64, 64, 64, 64, 1>();
    // pto: %step_sin_signed_inline12365__ssa_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v40 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(
            v3 + (v16 + v32 * v12), v38, v39
        );
    TLOAD(v36, v40);
    // pto: %qr_nope_slice_inline2037_inline12508__tile
    Tile<
        TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v41 = Tile<
            TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v13, v12);
    // pto: %qr_nope_slice_inline2037_inline12508__tile
    uint64_t v42 = (uint64_t)v18;
    TASSIGN(v41, v42);
    // pto: %7
    int64_t v43 = v23 < v16 ? v16 : v23;
    // pto: %qr_proj_flat_inline1973_inline12473__ssa_v0_pview
    pto::Shape<1, 1, 1, 32, 64> v44 = pto::Shape<1, 1, 1, 32, 64>();
    // pto: %qr_proj_flat_inline1973_inline12473__ssa_v0_pview
    pto::Stride<4096, 4096, 4096, 128, 1> v45 = pto::Stride<4096, 4096, 4096, 128, 1>();
    // pto: %qr_proj_flat_inline1973_inline12473__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v46 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(
            v4 + (v16 + v43 * v10), v44, v45
        );
    TLOAD(v41, v46);
    // pto: %qr_rope_slice_inline1971_inline12714__tile
    Tile<
        TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v47 = Tile<
            TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v13, v12);
    // pto: %qr_rope_slice_inline1971_inline12714__tile
    uint64_t v48 = (uint64_t)v17;
    TASSIGN(v47, v48);
    // pto: %9
    pto::Shape<1, 1, 1, 32, 64> v49 = pto::Shape<1, 1, 1, 32, 64>();
    // pto: %9
    pto::Stride<4096, 4096, 4096, 128, 1> v50 = pto::Stride<4096, 4096, 4096, 128, 1>();
    // pto: %9
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v51 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(
            v4 + (v12 + v43 * v10), v49, v50
        );
    TLOAD(v47, v51);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %gather_acc_init
    Tile<
        TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v52 = Tile<
            TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v13, v12);
    // pto: %gather_acc_init
    uint64_t v53 = (uint64_t)v16;
    TASSIGN(v52, v53);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    for (int64_t i54 = v16; i54 < v13; i54 += v11) {
        // pto: %gather_inp_row
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v55 = Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v11, v12);
        // pto: %gather_inp_row
        uint64_t v56 = (uint64_t)v17;
        TASSIGN(v55, v56);
        // pto: %slice_view
        int64_t v57 = (int64_t)((uint64_t)i54 * (uint64_t)v22);
        // pto: %slice_view
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v58;
        // pto: %slice_view
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v59 = v58;
        // pto: %slice_view
        uint64_t v60 = (uint64_t)((int64_t)((uint64_t)v57 + (uint64_t)v17));
        TASSIGN(v59, v60);
        // pto: %gather_idx_row
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v61 = Tile<
                TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v11, v12);
        // pto: %gather_idx_row
        uint64_t v62 = (uint64_t)v21;
        TASSIGN(v61, v62);
        // pto: %10
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v63;
        // pto: %10
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v64 = v63;
        // pto: %10
        uint64_t v65 = (uint64_t)((int64_t)((uint64_t)v57 + (uint64_t)v21));
        TASSIGN(v64, v65);
        // pto: %gather_row_tmp
        Tile<
            TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v66 = Tile<
                TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v11, v12);
        // pto: %gather_row_tmp
        uint64_t v67 = (uint64_t)v15;
        TASSIGN(v66, v67);
        // pto: %gather_row
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v68 = Tile<
                TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v11, v12);
        // pto: %gather_row
        uint64_t v69 = (uint64_t)v14;
        TASSIGN(v68, v69);
        pipe_barrier(PIPE_V);
        TGATHER(v68, v59, v64, v66);
        // pto: %assemble_view
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v70;
        // pto: %assemble_view
        Tile<
            TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v71 = v70;
        // pto: %assemble_view
        uint64_t v72 = (uint64_t)v57;
        TASSIGN(v71, v72);
        pipe_barrier(PIPE_V);
        TMOV(v71, v68);
    }
    // pto: %qr_swapped_inline2002_inline12718__tile
    Tile<
        TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v73 = Tile<
            TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v13, v12);
    // pto: %qr_swapped_inline2002_inline12718__tile
    uint64_t v74 = (uint64_t)v16;
    TASSIGN(v73, v74);
    // pto: %t__tile
    Tile<
        TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v75 = Tile<
            TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v13, v12);
    // pto: %t__tile
    uint64_t v76 = (uint64_t)v21;
    TASSIGN(v75, v76);
    TCOLEXPANDMUL(v75, v47, v30);
    // pto: %0
    Tile<
        TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v77 = Tile<
            TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v13, v12);
    // pto: %0
    uint64_t v78 = (uint64_t)v17;
    TASSIGN(v77, v78);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v77, v73, v36);
    // pto: %rope_rot_inline2005_inline12370__tile
    Tile<
        TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v79 = Tile<
            TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v13, v12);
    // pto: %rope_rot_inline2005_inline12370__tile
    uint64_t v80 = (uint64_t)v21;
    TASSIGN(v79, v80);
    pipe_barrier(PIPE_V);
    TADD(v79, v75, v77);
    // pto: %1
    Tile<
        TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v81 = Tile<
            TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v13, v12);
    // pto: %1
    uint64_t v82 = (uint64_t)v18;
    TASSIGN(v81, v82);
    TCVT(v81, v41, v9, v8);
    // pto: %2
    Tile<
        TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v83 = Tile<
            TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v13, v12);
    // pto: %2
    uint64_t v84 = (uint64_t)v17;
    TASSIGN(v83, v84);
    pipe_barrier(PIPE_V);
    TCVT(v83, v79, v9, v8);
    // pto: %qr_vec_inline1981_inline12388__tile
    Tile<
        TileType::Vec, bfloat16_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v85 = Tile<
            TileType::Vec, bfloat16_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v13, v10);
    // pto: %qr_vec_inline1981_inline12388__tile
    uint64_t v86 = (uint64_t)v21;
    TASSIGN(v85, v86);
    pipe_barrier(PIPE_V);
    TCONCAT(v85, v81, v83);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %qr_bf16_inline1969_inline12702__ssa_v0_pview
    pto::Shape<1, 1, 1, 32, 128> v87 = pto::Shape<1, 1, 1, 32, 128>();
    // pto: %qr_bf16_inline1969_inline12702__ssa_v0_pview
    pto::Stride<4096, 4096, 4096, 128, 1> v88 = pto::Stride<4096, 4096, 4096, 128, 1>();
    // pto: %qr_bf16_inline1969_inline12702__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v89 =
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(
            v5 + (v16 + v43 * v10), v87, v88
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v89, v85);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: rope_swap_idx_t_inline1970_inline12704__ssa_v1
    __gm__ Tensor *rope_swap_idx_t_inline1970_inline12704__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int32_t *rope_swap_idx_t_inline1970_inline12704__ssa_v1 =
        reinterpret_cast<__gm__ int32_t *>(rope_swap_idx_t_inline1970_inline12704__ssa_v1_tensor->buffer.addr) +
        rope_swap_idx_t_inline1970_inline12704__ssa_v1_tensor->start_offset;

    // Unpack tensor: step_cos_il_inline12430__ssa_v1
    __gm__ Tensor *step_cos_il_inline12430__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *step_cos_il_inline12430__ssa_v1 =
        reinterpret_cast<__gm__ float *>(step_cos_il_inline12430__ssa_v1_tensor->buffer.addr) +
        step_cos_il_inline12430__ssa_v1_tensor->start_offset;

    // Unpack tensor: step_sin_signed_inline12365__ssa_v1
    __gm__ Tensor *step_sin_signed_inline12365__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *step_sin_signed_inline12365__ssa_v1 =
        reinterpret_cast<__gm__ float *>(step_sin_signed_inline12365__ssa_v1_tensor->buffer.addr) +
        step_sin_signed_inline12365__ssa_v1_tensor->start_offset;

    // Unpack tensor: qr_proj_flat_inline1973_inline12473__ssa_v0
    __gm__ Tensor *qr_proj_flat_inline1973_inline12473__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ float *qr_proj_flat_inline1973_inline12473__ssa_v0 =
        reinterpret_cast<__gm__ float *>(qr_proj_flat_inline1973_inline12473__ssa_v0_tensor->buffer.addr) +
        qr_proj_flat_inline1973_inline12473__ssa_v0_tensor->start_offset;

    // Unpack tensor: qr_bf16_inline1969_inline12702__ssa_v0
    __gm__ Tensor *qr_bf16_inline1969_inline12702__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ bfloat16_t *qr_bf16_inline1969_inline12702__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(qr_bf16_inline1969_inline12702__ssa_v0_tensor->buffer.addr) +
        qr_bf16_inline1969_inline12702__ssa_v0_tensor->start_offset;

    // Forward to ptoas-generated function
    qr_rope_0(
        rope_swap_idx_t_inline1970_inline12704__ssa_v1, step_cos_il_inline12430__ssa_v1,
        step_sin_signed_inline12365__ssa_v1, qr_proj_flat_inline1973_inline12473__ssa_v0,
        qr_bf16_inline1969_inline12702__ssa_v0, __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
