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
// Kernel Function: mix_x_5

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
mix_x_5(__gm__ float *v1, __gm__ bfloat16_t *v2, __gm__ float *v3, int64_t v4, int64_t v5, int32_t v6, int32_t v7) {
    SaturationMode v8 = SaturationMode::OFF;
    RoundMode v9 = RoundMode::CAST_RINT;
    const int64_t v10 = 12288;
    const int64_t v11 = 256;
    const int64_t v12 = 2;
    const int64_t v13 = 1024;
    const int64_t v14 = 4;
    const int64_t v15 = 4096;
    const int64_t v16 = 1;
    const int64_t v17 = 8;
    const int64_t v18 = 24576;
    const int64_t v19 = 16384;
    const int64_t v20 = 8192;
    const int64_t v21 = 0;
    const int64_t v22 = 57600;
    const int64_t v23 = 49408;
    const int64_t v24 = 32864;
    const int64_t v25 = 32832;
    const int64_t v26 = 32800;
    const int64_t v27 = 32768;
    const int64_t v28 = 41216;
    const int64_t v29 = 33024;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %blk_inline14879__ssa_v0
    int64_t v30 = (int64_t)v6;
    // pto: %19, %20
    int64_t v31 = (int64_t)((uint64_t)(v30 / v14) * (uint64_t)v17);
    // pto: %21, %22
    int64_t v32 = (int64_t)((uint64_t)(v30 % v14) * (uint64_t)v13);
    // pto: %t__tile
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v33 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %t__tile
    uint64_t v34 = (uint64_t)v29;
    TASSIGN(v33, v34);
    // pto: %23
    int64_t v35 = v31 < v21 ? v21 : v31;
    // pto: %pre_val_store_inline14887__ssa_v1_pview
    pto::Shape<1, 1, 1, 8, 8> v36 = pto::Shape<1, 1, 1, 8, 8>();
    // pto: %pre_val_store_inline14887__ssa_v1_pview
    pto::Stride<64, 64, 64, 8, 1> v37 = pto::Stride<64, 64, 64, 8, 1>();
    // pto: %pre_val_store_inline14887__ssa_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 8>, pto::Stride<64, 64, 64, 8, 1>, pto::Layout::ND> v38 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 8>, pto::Stride<64, 64, 64, 8, 1>, pto::Layout::ND>(
            v1 + (v21 + v35 * v17), v36, v37
        );
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v33, v38);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %transpose_tmp
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v39 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %transpose_tmp
    uint64_t v40 = (uint64_t)v28;
    TASSIGN(v39, v40);
    // pto: %pre_tile_t_inline14888__tile
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v41 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %pre_tile_t_inline14888__tile
    uint64_t v42 = (uint64_t)v27;
    TASSIGN(v41, v42);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TTRANS(v41, v33, v39);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    // pto: %0
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v43 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v17);
    // pto: %0
    uint64_t v44 = (uint64_t)v27;
    TASSIGN(v43, v44);
    // pto: %pre0_inline14970__tile
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v45 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v16);
    // pto: %pre0_inline14970__tile
    uint64_t v46 = (uint64_t)v27;
    TASSIGN(v45, v46);
    // pto: %1
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v47 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v17);
    // pto: %1
    uint64_t v48 = (uint64_t)v26;
    TASSIGN(v47, v48);
    // pto: %pre1_inline14972__tile
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v49 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v16);
    // pto: %pre1_inline14972__tile
    uint64_t v50 = (uint64_t)v26;
    TASSIGN(v49, v50);
    // pto: %2
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v51 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v17);
    // pto: %2
    uint64_t v52 = (uint64_t)v25;
    TASSIGN(v51, v52);
    // pto: %pre2_inline14973__tile
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v53 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v16);
    // pto: %pre2_inline14973__tile
    uint64_t v54 = (uint64_t)v25;
    TASSIGN(v53, v54);
    // pto: %3
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v55 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v16, v17);
    // pto: %3
    uint64_t v56 = (uint64_t)v24;
    TASSIGN(v55, v56);
    // pto: %pre3_inline14971__tile
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v57 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v16);
    // pto: %pre3_inline14971__tile
    uint64_t v58 = (uint64_t)v24;
    TASSIGN(v57, v58);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    for (int64_t i59 = v21; i59 < v14; i59 += v12) {
        // pto: %27
        int64_t v60 = (int64_t)((uint64_t)i59 * (uint64_t)v11);
        // pto: %28
        int64_t v61 = (int64_t)((uint64_t)v32 + (uint64_t)v60);
        // pto: %31, %30
        int64_t v62 = (int64_t)((uint64_t)v32 + (uint64_t)((int64_t)((uint64_t)v60 + (uint64_t)v11)));
        // pto: %x0_inline14977__tile
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v63 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %x0_inline14977__tile
        uint64_t v64 = (uint64_t)v29;
        TASSIGN(v63, v64);
        // pto: %33
        int64_t v65 = v61 < v21 ? v21 : v61;
        // pto: %x_flat_inline14904__ssa_v0_pview
        pto::Shape<1, 1, 1, 8, 256> v66 = pto::Shape<1, 1, 1, 8, 256>();
        // pto: %x_flat_inline14904__ssa_v0_pview
        pto::Stride<131072, 131072, 131072, 16384, 1> v67 = pto::Stride<131072, 131072, 131072, 16384, 1>();
        // pto: %x_flat_inline14904__ssa_v0_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>
            v68 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(
                v3 + ((v21 + v35 * v19) + v65), v66, v67
            );
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        TLOAD(v63, v68);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %x1_inline14955__tile
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v69 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %x1_inline14955__tile
        uint64_t v70 = (uint64_t)v28;
        TASSIGN(v69, v70);
        // pto: %35
        int64_t v71 = (int64_t)((uint64_t)v61 + (uint64_t)v15);
        // pto: %37
        pto::Shape<1, 1, 1, 8, 256> v72 = pto::Shape<1, 1, 1, 8, 256>();
        // pto: %37
        pto::Stride<131072, 131072, 131072, 16384, 1> v73 = pto::Stride<131072, 131072, 131072, 16384, 1>();
        // pto: %37, %36
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>
            v74 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(
                v3 + ((v21 + v35 * v19) + (v71 < v21 ? v21 : v71)), v72, v73
            );
        TLOAD(v69, v74);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %x2_inline14916__tile
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v75 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %x2_inline14916__tile
        uint64_t v76 = (uint64_t)v23;
        TASSIGN(v75, v76);
        // pto: %39
        int64_t v77 = (int64_t)((uint64_t)v61 + (uint64_t)v20);
        // pto: %41
        pto::Shape<1, 1, 1, 8, 256> v78 = pto::Shape<1, 1, 1, 8, 256>();
        // pto: %41
        pto::Stride<131072, 131072, 131072, 16384, 1> v79 = pto::Stride<131072, 131072, 131072, 16384, 1>();
        // pto: %41, %40
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>
            v80 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(
                v3 + ((v21 + v35 * v19) + (v77 < v21 ? v21 : v77)), v78, v79
            );
        TLOAD(v75, v80);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %x3_inline14978__tile
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v81 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %x3_inline14978__tile
        uint64_t v82 = (uint64_t)v22;
        TASSIGN(v81, v82);
        // pto: %43
        int64_t v83 = (int64_t)((uint64_t)v61 + (uint64_t)v10);
        // pto: %45
        pto::Shape<1, 1, 1, 8, 256> v84 = pto::Shape<1, 1, 1, 8, 256>();
        // pto: %45
        pto::Stride<131072, 131072, 131072, 16384, 1> v85 = pto::Stride<131072, 131072, 131072, 16384, 1>();
        // pto: %45, %44
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>
            v86 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(
                v3 + ((v21 + v35 * v19) + (v83 < v21 ? v21 : v83)), v84, v85
            );
        TLOAD(v81, v86);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %4
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v87 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %4
        uint64_t v88 = (uint64_t)v21;
        TASSIGN(v87, v88);
        // pto: %47
        int64_t v89 = v62 < v21 ? v21 : v62;
        // pto: %48
        pto::Shape<1, 1, 1, 8, 256> v90 = pto::Shape<1, 1, 1, 8, 256>();
        // pto: %48
        pto::Stride<131072, 131072, 131072, 16384, 1> v91 = pto::Stride<131072, 131072, 131072, 16384, 1>();
        // pto: %48
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>
            v92 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(
                v3 + ((v21 + v35 * v19) + v89), v90, v91
            );
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        TLOAD(v87, v92);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %5
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v93 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %5
        uint64_t v94 = (uint64_t)v20;
        TASSIGN(v93, v94);
        // pto: %50
        int64_t v95 = (int64_t)((uint64_t)v62 + (uint64_t)v15);
        // pto: %52
        pto::Shape<1, 1, 1, 8, 256> v96 = pto::Shape<1, 1, 1, 8, 256>();
        // pto: %52
        pto::Stride<131072, 131072, 131072, 16384, 1> v97 = pto::Stride<131072, 131072, 131072, 16384, 1>();
        // pto: %52, %51
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>
            v98 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(
                v3 + ((v21 + v35 * v19) + (v95 < v21 ? v21 : v95)), v96, v97
            );
        TLOAD(v93, v98);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        // pto: %6
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v99 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %6
        uint64_t v100 = (uint64_t)v19;
        TASSIGN(v99, v100);
        // pto: %54
        int64_t v101 = (int64_t)((uint64_t)v62 + (uint64_t)v20);
        // pto: %56
        pto::Shape<1, 1, 1, 8, 256> v102 = pto::Shape<1, 1, 1, 8, 256>();
        // pto: %56
        pto::Stride<131072, 131072, 131072, 16384, 1> v103 = pto::Stride<131072, 131072, 131072, 16384, 1>();
        // pto: %56, %55
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>
            v104 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(
                v3 + ((v21 + v35 * v19) + (v101 < v21 ? v21 : v101)), v102, v103
            );
        TLOAD(v99, v104);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        // pto: %7
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v105 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %7
        uint64_t v106 = (uint64_t)v18;
        TASSIGN(v105, v106);
        // pto: %58
        int64_t v107 = (int64_t)((uint64_t)v62 + (uint64_t)v10);
        // pto: %60
        pto::Shape<1, 1, 1, 8, 256> v108 = pto::Shape<1, 1, 1, 8, 256>();
        // pto: %60
        pto::Stride<131072, 131072, 131072, 16384, 1> v109 = pto::Stride<131072, 131072, 131072, 16384, 1>();
        // pto: %60, %59
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>
            v110 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(
                v3 + ((v21 + v35 * v19) + (v107 < v21 ? v21 : v107)), v108, v109
            );
        TLOAD(v105, v110);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %y0_inline14980__tile
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v111 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %y0_inline14980__tile
        uint64_t v112 = (uint64_t)v29;
        TASSIGN(v111, v112);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TROWEXPANDMUL(v111, v63, v45);
        // pto: %y1_inline14956__tile
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v113 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %y1_inline14956__tile
        uint64_t v114 = (uint64_t)v28;
        TASSIGN(v113, v114);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        TROWEXPANDMUL(v113, v69, v49);
        // pto: %y2_inline14974__tile
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v115 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %y2_inline14974__tile
        uint64_t v116 = (uint64_t)v23;
        TASSIGN(v115, v116);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        TROWEXPANDMUL(v115, v75, v53);
        // pto: %y3_inline14885__tile
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v117 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %y3_inline14885__tile
        uint64_t v118 = (uint64_t)v22;
        TASSIGN(v117, v118);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        TROWEXPANDMUL(v117, v81, v57);
        // pto: %8
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v119 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %8
        uint64_t v120 = (uint64_t)v29;
        TASSIGN(v119, v120);
        pipe_barrier(PIPE_V);
        TADD(v119, v111, v113);
        // pto: %9
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v121 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %9
        uint64_t v122 = (uint64_t)v28;
        TASSIGN(v121, v122);
        pipe_barrier(PIPE_V);
        TADD(v121, v115, v117);
        // pto: %y_tile_inline14862__tile
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v123 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %y_tile_inline14862__tile
        uint64_t v124 = (uint64_t)v29;
        TASSIGN(v123, v124);
        pipe_barrier(PIPE_V);
        TADD(v123, v119, v121);
        // pto: %10
        Tile<
            TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v125 = Tile<
                TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %10
        uint64_t v126 = (uint64_t)v29;
        TASSIGN(v125, v126);
        pipe_barrier(PIPE_V);
        TCVT(v125, v123, v9, v8);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %x_mixed_inline11578__iter_v1_pview
        pto::Shape<1, 1, 1, 8, 256> v127 = pto::Shape<1, 1, 1, 8, 256>();
        // pto: %x_mixed_inline11578__iter_v1_pview
        pto::Stride<32768, 32768, 32768, 4096, 1> v128 = pto::Stride<32768, 32768, 32768, 4096, 1>();
        // pto: %x_mixed_inline11578__iter_v1_pview
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>
            v129 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(
                v2 + ((v21 + v35 * v15) + v65), v127, v128
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v129, v125);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        // pto: %11
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v130 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %11
        uint64_t v131 = (uint64_t)v21;
        TASSIGN(v130, v131);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        TROWEXPANDMUL(v130, v87, v45);
        // pto: %12
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v132 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %12
        uint64_t v133 = (uint64_t)v20;
        TASSIGN(v132, v133);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        TROWEXPANDMUL(v132, v93, v49);
        // pto: %13
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v134 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %13
        uint64_t v135 = (uint64_t)v19;
        TASSIGN(v134, v135);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        TROWEXPANDMUL(v134, v99, v53);
        // pto: %14
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v136 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %14
        uint64_t v137 = (uint64_t)v18;
        TASSIGN(v136, v137);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TROWEXPANDMUL(v136, v105, v57);
        // pto: %15
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v138 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %15
        uint64_t v139 = (uint64_t)v21;
        TASSIGN(v138, v139);
        pipe_barrier(PIPE_V);
        TADD(v138, v130, v132);
        // pto: %16
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v140 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %16
        uint64_t v141 = (uint64_t)v20;
        TASSIGN(v140, v141);
        pipe_barrier(PIPE_V);
        TADD(v140, v134, v136);
        // pto: %17
        Tile<
            TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v142 = Tile<
                TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %17
        uint64_t v143 = (uint64_t)v21;
        TASSIGN(v142, v143);
        pipe_barrier(PIPE_V);
        TADD(v142, v138, v140);
        // pto: %18
        Tile<
            TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v144 = Tile<
                TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v17, v11);
        // pto: %18
        uint64_t v145 = (uint64_t)v21;
        TASSIGN(v144, v145);
        pipe_barrier(PIPE_V);
        TCVT(v144, v142, v9, v8);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        // pto: %x_mixed_inline11578__tile_pview
        pto::Shape<1, 1, 1, 8, 256> v146 = pto::Shape<1, 1, 1, 8, 256>();
        // pto: %x_mixed_inline11578__tile_pview
        pto::Stride<32768, 32768, 32768, 4096, 1> v147 = pto::Stride<32768, 32768, 32768, 4096, 1>();
        // pto: %x_mixed_inline11578__tile_pview
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>
            v148 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(
                v2 + ((v21 + v35 * v15) + v89), v146, v147
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v148, v144);
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
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: pre_val_store_inline14887__ssa_v1
    __gm__ Tensor *pre_val_store_inline14887__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *pre_val_store_inline14887__ssa_v1 =
        reinterpret_cast<__gm__ float *>(pre_val_store_inline14887__ssa_v1_tensor->buffer.addr) +
        pre_val_store_inline14887__ssa_v1_tensor->start_offset;

    // Unpack tensor: x_mixed_inline11578__ssa_v0
    __gm__ Tensor *x_mixed_inline11578__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ bfloat16_t *x_mixed_inline11578__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(x_mixed_inline11578__ssa_v0_tensor->buffer.addr) +
        x_mixed_inline11578__ssa_v0_tensor->start_offset;

    // Unpack tensor: x_flat_inline14904__ssa_v0
    __gm__ Tensor *x_flat_inline14904__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *x_flat_inline14904__ssa_v0 =
        reinterpret_cast<__gm__ float *>(x_flat_inline14904__ssa_v0_tensor->buffer.addr) +
        x_flat_inline14904__ssa_v0_tensor->start_offset;

    // Extract dynamic dim: t_linear_inline14909__ssa_v0
    int64_t t_linear_inline14909__ssa_v0 = static_cast<int64_t>(pre_val_store_inline14887__ssa_v1_tensor->shapes[0]);

    // Extract dynamic dim: t_dim_inline14891__ssa_v0
    int64_t t_dim_inline14891__ssa_v0 = static_cast<int64_t>(x_flat_inline14904__ssa_v0_tensor->shapes[0]);

    // Forward to ptoas-generated function
    mix_x_5(
        pre_val_store_inline14887__ssa_v1, x_mixed_inline11578__ssa_v0, x_flat_inline14904__ssa_v0,
        t_linear_inline14909__ssa_v0, t_dim_inline14891__ssa_v0, __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
