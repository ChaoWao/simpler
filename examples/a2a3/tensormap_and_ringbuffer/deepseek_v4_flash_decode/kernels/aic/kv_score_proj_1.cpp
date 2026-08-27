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
// Kernel Function: kv_score_proj_1

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

static __aicore__ void kv_score_proj_1(
    __gm__ bfloat16_t *v1, __gm__ bfloat16_t *v2, __gm__ bfloat16_t *v3, __gm__ float *v4, __gm__ float *v5, int64_t v6,
    int64_t v7, int32_t v8, int32_t v9
) {
    const int64_t v10 = 256;
    const int64_t v11 = 2;
    const int64_t v12 = 64;
    const int64_t v13 = 8;
    const int64_t v14 = 16;
    const int64_t v15 = 512;
    const int64_t v16 = 1;
    const int64_t v17 = 24576;
    const int64_t v18 = 32768;
    const int64_t v19 = 8192;
    const int64_t v20 = 81920;
    const int64_t v21 = 16384;
    const int64_t v22 = 229376;
    const int64_t v23 = 163840;
    const int64_t v24 = 147456;
    const int64_t v25 = 4096;
    const int64_t v26 = 0;
    using T = float;

#if defined(__DAV_CUBE__)
    // pto: %x_flat_inline1388_inline11613__ssa_v0_view
    int64_t v27 = v7 * v25;
    // pto: %x_flat_inline1388_inline11613__ssa_v0_view
    int64_t v28 = v16 * v27;
    // pto: %x_flat_inline1388_inline11613__ssa_v0_view
    pto::Shape<1, 1, 1, -1, -1> v29 = pto::Shape<1, 1, 1, -1, -1>(v16, v16, v16, v7, v25);
    // pto: %x_flat_inline1388_inline11613__ssa_v0_view
    pto::Stride<-1, -1, -1, -1, -1> v30 = pto::Stride<-1, -1, -1, -1, -1>(v16 * v28, v28, v27, v25, v16);
    // pto: %x_flat_inline1388_inline11613__ssa_v0_view
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v31 =
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(
            v1, v29, v30
        );
    // pto: %kv_acc_inline1379_inline11530__phi_v5
    Tile<
        TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v32 = Tile<
            TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v14, v12);
    // pto: %kv_acc_inline1379_inline11530__phi_v5
    uint64_t v33 = (uint64_t)v26;
    TASSIGN(v32, v33);
    // pto: %score_acc_inline1402_inline11794__phi_v5
    Tile<
        TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v34 = Tile<
            TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v14, v12);
    // pto: %score_acc_inline1402_inline11794__phi_v5
    uint64_t v35 = (uint64_t)v25;
    TASSIGN(v34, v35);
    // pto: %idx_inline1390_inline11791__ssa_v0
    int64_t v36 = (int64_t)v8;
    // pto: %35, %36
    int64_t v37 = (int64_t)((uint64_t)(v36 / v13) * (uint64_t)v14);
    // pto: %37, %38
    int64_t v38 = (int64_t)((uint64_t)(v36 % v13) * (uint64_t)v12);
    // pto: %kv_acc_inline1379_inline11530__tile
    Tile<
        TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v39 = Tile<
            TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v14, v12);
    // pto: %kv_acc_inline1379_inline11530__tile
    uint64_t v40 = (uint64_t)v26;
    TASSIGN(v39, v40);
    // pto: %score_acc_inline1402_inline11794__tile
    Tile<
        TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v41 = Tile<
            TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v14, v12);
    // pto: %score_acc_inline1402_inline11794__tile
    uint64_t v42 = (uint64_t)v25;
    TASSIGN(v41, v42);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    for (int64_t i43 = v26; i43 < v13; i43 += v11) {
        // pto: %39
        int64_t v44 = (int64_t)((uint64_t)i43 * (uint64_t)v15);
        // pto: %40
        int64_t v45 = (int64_t)((uint64_t)v7 - (uint64_t)v37);
        // pto: %41
        int64_t v46 = v45 < v14 ? v45 : v14;
        // pto: %43
        int64_t v47 = (int64_t)((uint64_t)v44 + (uint64_t)v15);
        // pto: %x_tile_inline1415_inline11748__tile
        Tile<
            TileType::Mat, bfloat16_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v48 = Tile<
                TileType::Mat, bfloat16_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v46, v15);
        // pto: %x_tile_inline1415_inline11748__tile
        uint64_t v49 = (uint64_t)v24;
        TASSIGN(v48, v49);
        // pto: %46
        int64_t v50 = v37 < v26 ? v26 : v37;
        // pto: %47
        int64_t v51 = v44 < v26 ? v26 : v44;
        // pto: %x_flat_inline1388_inline11613__ssa_v0_pview
        __gm__ bfloat16_t *v52 = PTOAS__GLOBAL_TENSOR_DATA(v31);
        // pto: %x_flat_inline1388_inline11613__ssa_v0_pview
        int64_t v53 = v46 * v25;
        // pto: %x_flat_inline1388_inline11613__ssa_v0_pview
        int64_t v54 = v16 * v53;
        // pto: %x_flat_inline1388_inline11613__ssa_v0_pview
        pto::Shape<1, 1, 1, -1, 512> v55 = pto::Shape<1, 1, 1, -1, 512>(v16, v16, v16, v46, v15);
        // pto: %x_flat_inline1388_inline11613__ssa_v0_pview
        pto::Stride<-1, -1, -1, -1, -1> v56 = pto::Stride<-1, -1, -1, -1, -1>(v16 * v54, v54, v53, v25, v16);
        // pto: %x_flat_inline1388_inline11613__ssa_v0_pview
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v57 =
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(
                v52 + ((v26 + v50 * v25) + v51 * v16), v55, v56
            );
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        TLOAD(v48, v57);
        // pto: %wkv_tile_inline1385_inline11797__tile
        Tile<
            TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v58 = Tile<
                TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v12, v15);
        // pto: %wkv_tile_inline1385_inline11797__tile
        uint64_t v59 = (uint64_t)v23;
        TASSIGN(v58, v59);
        // pto: %48
        int64_t v60 = v38 < v26 ? v26 : v38;
        // pto: %hca_cmp_wkv_hca_inline584__ssa_v0_pview
        pto::Shape<1, 1, 1, 64, 512> v61 = pto::Shape<1, 1, 1, 64, 512>();
        // pto: %hca_cmp_wkv_hca_inline584__ssa_v0_pview
        pto::Stride<262144, 262144, 262144, 4096, 1> v62 = pto::Stride<262144, 262144, 262144, 4096, 1>();
        // pto: %hca_cmp_wkv_hca_inline584__ssa_v0_pview
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<262144, 262144, 262144, 4096, 1>, pto::Layout::ND>
            v63 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<262144, 262144, 262144, 4096, 1>,
                pto::Layout::ND>(v2 + ((v26 + v60 * v25) + v51), v61, v62);
        TLOAD(v58, v63);
        // pto: %wgate_tile_inline1368_inline11592__tile
        Tile<
            TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v64 = Tile<
                TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v12, v15);
        // pto: %wgate_tile_inline1368_inline11592__tile
        uint64_t v65 = (uint64_t)v22;
        TASSIGN(v64, v65);
        // pto: %hca_cmp_wgate_hca_inline630__ssa_v0_pview
        pto::Shape<1, 1, 1, 64, 512> v66 = pto::Shape<1, 1, 1, 64, 512>();
        // pto: %hca_cmp_wgate_hca_inline630__ssa_v0_pview
        pto::Stride<262144, 262144, 262144, 4096, 1> v67 = pto::Stride<262144, 262144, 262144, 4096, 1>();
        // pto: %hca_cmp_wgate_hca_inline630__ssa_v0_pview
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<262144, 262144, 262144, 4096, 1>, pto::Layout::ND>
            v68 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<262144, 262144, 262144, 4096, 1>,
                pto::Layout::ND>(v3 + ((v26 + v60 * v25) + v51), v66, v67);
        TLOAD(v64, v68);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        // pto: %0
        Tile<
            TileType::Mat, bfloat16_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v69 = Tile<
                TileType::Mat, bfloat16_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v46, v15);
        // pto: %0
        uint64_t v70 = (uint64_t)v26;
        TASSIGN(v69, v70);
        // pto: %53
        int64_t v71 = v47 < v26 ? v26 : v47;
        // pto: %54
        __gm__ bfloat16_t *v72 = PTOAS__GLOBAL_TENSOR_DATA(v31);
        // pto: %54
        int64_t v73 = v46 * v25;
        // pto: %54
        int64_t v74 = v16 * v73;
        // pto: %54
        pto::Shape<1, 1, 1, -1, 512> v75 = pto::Shape<1, 1, 1, -1, 512>(v16, v16, v16, v46, v15);
        // pto: %54
        pto::Stride<-1, -1, -1, -1, -1> v76 = pto::Stride<-1, -1, -1, -1, -1>(v16 * v74, v74, v73, v25, v16);
        // pto: %54
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v77 =
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(
                v72 + ((v26 + v50 * v25) + v71 * v16), v75, v76
            );
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        TLOAD(v69, v77);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        // pto: %1
        Tile<
            TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v78 = Tile<
                TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v12, v15);
        // pto: %1
        uint64_t v79 = (uint64_t)v21;
        TASSIGN(v78, v79);
        // pto: %57
        pto::Shape<1, 1, 1, 64, 512> v80 = pto::Shape<1, 1, 1, 64, 512>();
        // pto: %57
        pto::Stride<262144, 262144, 262144, 4096, 1> v81 = pto::Stride<262144, 262144, 262144, 4096, 1>();
        // pto: %57
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<262144, 262144, 262144, 4096, 1>, pto::Layout::ND>
            v82 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<262144, 262144, 262144, 4096, 1>,
                pto::Layout::ND>(v2 + ((v26 + v60 * v25) + v71), v80, v81);
        TLOAD(v78, v82);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        // pto: %2
        Tile<
            TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v83 = Tile<
                TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v12, v15);
        // pto: %2
        uint64_t v84 = (uint64_t)v20;
        TASSIGN(v83, v84);
        // pto: %60
        pto::Shape<1, 1, 1, 64, 512> v85 = pto::Shape<1, 1, 1, 64, 512>();
        // pto: %60
        pto::Stride<262144, 262144, 262144, 4096, 1> v86 = pto::Stride<262144, 262144, 262144, 4096, 1>();
        // pto: %60
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<262144, 262144, 262144, 4096, 1>, pto::Layout::ND>
            v87 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<262144, 262144, 262144, 4096, 1>,
                pto::Layout::ND>(v3 + ((v26 + v60 * v25) + v71), v85, v86);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        TLOAD(v83, v87);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        // pto: %61
        v32.SetValidShape(v46, v12);
        v34.SetValidShape(v46, v12);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        if (v44 == v26) {
            // pto: %wkv_tile_inline1385_inline11797__tile_t
            Tile<
                TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v88 = Tile<
                    TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v12);
            // pto: %wkv_tile_inline1385_inline11797__tile_t
            uint64_t v89 = (uint64_t)v23;
            TASSIGN(v88, v89);
            // pto: %kv_acc_inline1379_inline11530__tile_l0_init_storage
            Tile<
                TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v90 = Tile<
                    TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v14, v12);
            // pto: %kv_acc_inline1379_inline11530__tile_l0_init_storage
            uint64_t v91 = (uint64_t)v26;
            TASSIGN(v90, v91);
            v90.SetValidShape(v46, v12);
            // pto: %kv_acc_inline1379_inline11530__tile_l0_a
            Tile<
                TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Normal>
                v92 = Tile<
                    TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Normal>(v46, v10);
            // pto: %kv_acc_inline1379_inline11530__tile_l0_a
            uint64_t v93 = (uint64_t)v26;
            TASSIGN(v92, v93);
            TEXTRACT(v92, v48, v26, v26);
            // pto: %kv_acc_inline1379_inline11530__tile_l0_b
            Tile<
                TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v94 = Tile<
                    TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v10, v12);
            // pto: %kv_acc_inline1379_inline11530__tile_l0_b
            uint64_t v95 = (uint64_t)v26;
            TASSIGN(v94, v95);
            TEXTRACT(v94, v88, v26, v26);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            // pto: %3
            Tile<
                TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Normal>
                v96 = Tile<
                    TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Normal>(v46, v10);
            // pto: %3
            uint64_t v97 = (uint64_t)v19;
            TASSIGN(v96, v97);
            TEXTRACT(v96, v48, v26, v10);
            // pto: %4
            Tile<
                TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v98 = Tile<
                    TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v10, v12);
            // pto: %4
            uint64_t v99 = (uint64_t)v18;
            TASSIGN(v98, v99);
            TEXTRACT(v98, v88, v10, v26);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
            // pto: %kv_acc_inline1379_inline11530__tile_l0_c_first
            Tile<
                TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v100 = Tile<
                    TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v46, v12);
            // pto: %kv_acc_inline1379_inline11530__tile_l0_c_first
            uint64_t v101 = (uint64_t)v26;
            TASSIGN(v100, v101);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            TMATMUL(v100, v92, v94);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            // pto: %kv_acc_inline1379_inline11530__tile_l0_c_acc
            Tile<
                TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v102 = Tile<
                    TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v46, v12);
            // pto: %kv_acc_inline1379_inline11530__tile_l0_c_acc
            uint64_t v103 = (uint64_t)v26;
            TASSIGN(v102, v103);
            pipe_barrier(PIPE_M);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
            TMATMUL_ACC(v102, v102, v96, v98);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
            // pto: %wgate_tile_inline1368_inline11592__tile_t
            Tile<
                TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v104 = Tile<
                    TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v12);
            // pto: %wgate_tile_inline1368_inline11592__tile_t
            uint64_t v105 = (uint64_t)v22;
            TASSIGN(v104, v105);
            // pto: %score_acc_inline1402_inline11794__tile_l0_init_storage
            Tile<
                TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v106 = Tile<
                    TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v14, v12);
            // pto: %score_acc_inline1402_inline11794__tile_l0_init_storage
            uint64_t v107 = (uint64_t)v25;
            TASSIGN(v106, v107);
            v106.SetValidShape(v46, v12);
            // pto: %score_acc_inline1402_inline11794__tile_l0_a
            Tile<
                TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Normal>
                v108 = Tile<
                    TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Normal>(v46, v10);
            // pto: %score_acc_inline1402_inline11794__tile_l0_a
            uint64_t v109 = (uint64_t)v26;
            TASSIGN(v108, v109);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            TEXTRACT(v108, v48, v26, v26);
            // pto: %score_acc_inline1402_inline11794__tile_l0_b
            Tile<
                TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v110 = Tile<
                    TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v10, v12);
            // pto: %score_acc_inline1402_inline11794__tile_l0_b
            uint64_t v111 = (uint64_t)v26;
            TASSIGN(v110, v111);
            TEXTRACT(v110, v104, v26, v26);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
            // pto: %5
            Tile<
                TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Normal>
                v112 = Tile<
                    TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Normal>(v46, v10);
            // pto: %5
            uint64_t v113 = (uint64_t)v19;
            TASSIGN(v112, v113);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
            TEXTRACT(v112, v48, v26, v10);
            // pto: %6
            Tile<
                TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v114 = Tile<
                    TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v10, v12);
            // pto: %6
            uint64_t v115 = (uint64_t)v18;
            TASSIGN(v114, v115);
            TEXTRACT(v114, v104, v10, v26);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
            // pto: %score_acc_inline1402_inline11794__tile_l0_c_first
            Tile<
                TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v116 = Tile<
                    TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v46, v12);
            // pto: %score_acc_inline1402_inline11794__tile_l0_c_first
            uint64_t v117 = (uint64_t)v25;
            TASSIGN(v116, v117);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
            TMATMUL(v116, v108, v110);
            // pto: %score_acc_inline1402_inline11794__tile_l0_c_acc
            Tile<
                TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v118 = Tile<
                    TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v46, v12);
            // pto: %score_acc_inline1402_inline11794__tile_l0_c_acc
            uint64_t v119 = (uint64_t)v25;
            TASSIGN(v118, v119);
            pipe_barrier(PIPE_M);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
            TMATMUL_ACC(v118, v118, v112, v114);
        } else {
            // pto: %7
            Tile<
                TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v120 = Tile<
                    TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v12);
            // pto: %7
            uint64_t v121 = (uint64_t)v23;
            TASSIGN(v120, v121);
            // pto: %8
            Tile<
                TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Normal>
                v122 = Tile<
                    TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Normal>(v46, v10);
            // pto: %8
            uint64_t v123 = (uint64_t)v26;
            TASSIGN(v122, v123);
            TEXTRACT(v122, v48, v26, v26);
            // pto: %9
            Tile<
                TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v124 = Tile<
                    TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v10, v12);
            // pto: %9
            uint64_t v125 = (uint64_t)v26;
            TASSIGN(v124, v125);
            TEXTRACT(v124, v120, v26, v26);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
            // pto: %10
            Tile<
                TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Normal>
                v126 = Tile<
                    TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Normal>(v46, v10);
            // pto: %10
            uint64_t v127 = (uint64_t)v19;
            TASSIGN(v126, v127);
            TEXTRACT(v126, v48, v26, v10);
            // pto: %11
            Tile<
                TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v128 = Tile<
                    TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v10, v12);
            // pto: %11
            uint64_t v129 = (uint64_t)v18;
            TASSIGN(v128, v129);
            TEXTRACT(v128, v120, v10, v26);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
            // pto: %12
            Tile<
                TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v130 = Tile<
                    TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v14, v12);
            // pto: %12
            uint64_t v131 = (uint64_t)v26;
            TASSIGN(v130, v131);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
            TMATMUL_ACC(v130, v130, v122, v124);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
            // pto: %13
            Tile<
                TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v132 = Tile<
                    TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v14, v12);
            // pto: %13
            uint64_t v133 = (uint64_t)v26;
            TASSIGN(v132, v133);
            pipe_barrier(PIPE_M);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
            TMATMUL_ACC(v132, v132, v126, v128);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
            // pto: %14
            Tile<
                TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v134 = Tile<
                    TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v12);
            // pto: %14
            uint64_t v135 = (uint64_t)v22;
            TASSIGN(v134, v135);
            // pto: %15
            Tile<
                TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Normal>
                v136 = Tile<
                    TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Normal>(v46, v10);
            // pto: %15
            uint64_t v137 = (uint64_t)v26;
            TASSIGN(v136, v137);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
            TEXTRACT(v136, v48, v26, v26);
            // pto: %16
            Tile<
                TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v138 = Tile<
                    TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v10, v12);
            // pto: %16
            uint64_t v139 = (uint64_t)v26;
            TASSIGN(v138, v139);
            TEXTRACT(v138, v134, v26, v26);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
            // pto: %17
            Tile<
                TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Normal>
                v140 = Tile<
                    TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Normal>(v46, v10);
            // pto: %17
            uint64_t v141 = (uint64_t)v19;
            TASSIGN(v140, v141);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
            TEXTRACT(v140, v48, v26, v10);
            // pto: %18
            Tile<
                TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v142 = Tile<
                    TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v10, v12);
            // pto: %18
            uint64_t v143 = (uint64_t)v18;
            TASSIGN(v142, v143);
            TEXTRACT(v142, v134, v10, v26);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
            // pto: %19
            Tile<
                TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v144 = Tile<
                    TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v14, v12);
            // pto: %19
            uint64_t v145 = (uint64_t)v25;
            TASSIGN(v144, v145);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
            TMATMUL_ACC(v144, v144, v136, v138);
            // pto: %20
            Tile<
                TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v146 = Tile<
                    TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v14, v12);
            // pto: %20
            uint64_t v147 = (uint64_t)v25;
            TASSIGN(v146, v147);
            pipe_barrier(PIPE_M);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
            TMATMUL_ACC(v146, v146, v140, v142);
        }
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
        // pto: %21
        Tile<
            TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v148 = Tile<
                TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v12);
        // pto: %21
        uint64_t v149 = (uint64_t)v21;
        TASSIGN(v148, v149);
        // pto: %22
        Tile<
            TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Normal>
            v150 = Tile<
                TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Normal>(v46, v10);
        // pto: %22
        uint64_t v151 = (uint64_t)v21;
        TASSIGN(v150, v151);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v150, v69, v26, v26);
        // pto: %23
        Tile<
            TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v152 = Tile<
                TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v10, v12);
        // pto: %23
        uint64_t v153 = (uint64_t)v26;
        TASSIGN(v152, v153);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        TEXTRACT(v152, v148, v26, v26);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        // pto: %24
        Tile<
            TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Normal>
            v154 = Tile<
                TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Normal>(v46, v10);
        // pto: %24
        uint64_t v155 = (uint64_t)v17;
        TASSIGN(v154, v155);
        TEXTRACT(v154, v69, v26, v10);
        // pto: %25
        Tile<
            TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v156 = Tile<
                TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v10, v12);
        // pto: %25
        uint64_t v157 = (uint64_t)v18;
        TASSIGN(v156, v157);
        TEXTRACT(v156, v148, v10, v26);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        // pto: %26
        Tile<
            TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v158 = Tile<
                TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v14, v12);
        // pto: %26
        uint64_t v159 = (uint64_t)v26;
        TASSIGN(v158, v159);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(v158, v158, v150, v152);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        // pto: %27
        Tile<
            TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v160 = Tile<
                TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v14, v12);
        // pto: %27
        uint64_t v161 = (uint64_t)v26;
        TASSIGN(v160, v161);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        TMATMUL_ACC(v160, v160, v154, v156);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
        // pto: %28
        Tile<
            TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v162 = Tile<
                TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v12);
        // pto: %28
        uint64_t v163 = (uint64_t)v20;
        TASSIGN(v162, v163);
        // pto: %29
        Tile<
            TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Normal>
            v164 = Tile<
                TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Normal>(v46, v10);
        // pto: %29
        uint64_t v165 = (uint64_t)v21;
        TASSIGN(v164, v165);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        TEXTRACT(v164, v69, v26, v26);
        // pto: %30
        Tile<
            TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v166 = Tile<
                TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v10, v12);
        // pto: %30
        uint64_t v167 = (uint64_t)v26;
        TASSIGN(v166, v167);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        TEXTRACT(v166, v162, v26, v26);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        // pto: %31
        Tile<
            TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Normal>
            v168 = Tile<
                TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Normal>(v46, v10);
        // pto: %31
        uint64_t v169 = (uint64_t)v17;
        TASSIGN(v168, v169);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
        TEXTRACT(v168, v69, v26, v10);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        // pto: %32
        Tile<
            TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v170 = Tile<
                TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v10, v12);
        // pto: %32
        uint64_t v171 = (uint64_t)v18;
        TASSIGN(v170, v171);
        TEXTRACT(v170, v162, v10, v26);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        // pto: %33
        Tile<
            TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v172 = Tile<
                TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v14, v12);
        // pto: %33
        uint64_t v173 = (uint64_t)v25;
        TASSIGN(v172, v173);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(v172, v172, v164, v166);
        // pto: %34
        Tile<
            TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v174 = Tile<
                TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v14, v12);
        // pto: %34
        uint64_t v175 = (uint64_t)v25;
        TASSIGN(v174, v175);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        TMATMUL_ACC(v174, v174, v168, v170);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    // pto: %62
    int64_t v176 = v37 < v26 ? v26 : v37;
    // pto: %63
    int64_t v177 = v38 < v26 ? v26 : v38;
    // pto: %kv_proj_pad_inline1372_inline11788__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 64> v178 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %kv_proj_pad_inline1372_inline11788__ssa_v0_pview
    pto::Stride<8192, 8192, 8192, 512, 1> v179 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %kv_proj_pad_inline1372_inline11788__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v180 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
            v4 + ((v26 + v176 * v15) + v177), v178, v179
        );
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(v180, v39);
    // pto: %score_proj_pad_inline1382_inline11463__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 64> v181 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %score_proj_pad_inline1382_inline11463__ssa_v0_pview
    pto::Stride<8192, 8192, 8192, 512, 1> v182 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %score_proj_pad_inline1382_inline11463__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v183 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
            v5 + ((v26 + v176 * v15) + v177), v181, v182
        );
    TSTORE(v183, v41);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
#endif  // __DAV_CUBE__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: x_flat_inline1388_inline11613__ssa_v0
    __gm__ Tensor *x_flat_inline1388_inline11613__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ bfloat16_t *x_flat_inline1388_inline11613__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(x_flat_inline1388_inline11613__ssa_v0_tensor->buffer.addr) +
        x_flat_inline1388_inline11613__ssa_v0_tensor->start_offset;

    // Unpack tensor: hca_cmp_wkv_hca_inline584__ssa_v0
    __gm__ Tensor *hca_cmp_wkv_hca_inline584__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ bfloat16_t *hca_cmp_wkv_hca_inline584__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(hca_cmp_wkv_hca_inline584__ssa_v0_tensor->buffer.addr) +
        hca_cmp_wkv_hca_inline584__ssa_v0_tensor->start_offset;

    // Unpack tensor: hca_cmp_wgate_hca_inline630__ssa_v0
    __gm__ Tensor *hca_cmp_wgate_hca_inline630__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ bfloat16_t *hca_cmp_wgate_hca_inline630__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(hca_cmp_wgate_hca_inline630__ssa_v0_tensor->buffer.addr) +
        hca_cmp_wgate_hca_inline630__ssa_v0_tensor->start_offset;

    // Unpack tensor: kv_proj_pad_inline1372_inline11788__ssa_v0
    __gm__ Tensor *kv_proj_pad_inline1372_inline11788__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ float *kv_proj_pad_inline1372_inline11788__ssa_v0 =
        reinterpret_cast<__gm__ float *>(kv_proj_pad_inline1372_inline11788__ssa_v0_tensor->buffer.addr) +
        kv_proj_pad_inline1372_inline11788__ssa_v0_tensor->start_offset;

    // Unpack tensor: score_proj_pad_inline1382_inline11463__ssa_v0
    __gm__ Tensor *score_proj_pad_inline1382_inline11463__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ float *score_proj_pad_inline1382_inline11463__ssa_v0 =
        reinterpret_cast<__gm__ float *>(score_proj_pad_inline1382_inline11463__ssa_v0_tensor->buffer.addr) +
        score_proj_pad_inline1382_inline11463__ssa_v0_tensor->start_offset;

    // Unpack scalar: bs_inline1375_inline11667__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } bs_inline1375_inline11667__ssa_v0_conv;
    bs_inline1375_inline11667__ssa_v0_conv.u64 = args[5];
    int64_t bs_inline1375_inline11667__ssa_v0 = bs_inline1375_inline11667__ssa_v0_conv.val;

    // Extract dynamic dim: bs_inline1375_inline11667__ssa_v0_1
    int64_t bs_inline1375_inline11667__ssa_v0_1 =
        static_cast<int64_t>(x_flat_inline1388_inline11613__ssa_v0_tensor->shapes[0]);

    // Forward to ptoas-generated function
    kv_score_proj_1(
        x_flat_inline1388_inline11613__ssa_v0, hca_cmp_wkv_hca_inline584__ssa_v0, hca_cmp_wgate_hca_inline630__ssa_v0,
        kv_proj_pad_inline1372_inline11788__ssa_v0, score_proj_pad_inline1382_inline11463__ssa_v0,
        bs_inline1375_inline11667__ssa_v0, bs_inline1375_inline11667__ssa_v0_1, __pypto_spmd_block_idx,
        __pypto_spmd_block_num
    );
}
