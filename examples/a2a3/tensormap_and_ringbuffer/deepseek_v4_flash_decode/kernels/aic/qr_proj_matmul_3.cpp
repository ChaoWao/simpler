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
// Kernel Function: qr_proj_matmul_3

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

static __aicore__ void qr_proj_matmul_3(
    __gm__ float *v1, __gm__ bfloat16_t *v2, __gm__ bfloat16_t *v3, int64_t v4, int64_t v5, int64_t v6, int64_t v7,
    int32_t v8, int32_t v9
) {
    const int64_t v10 = 1024;
    const int64_t v11 = 256;
    const int64_t v12 = 8;
    const int64_t v13 = 16;
    const int64_t v14 = 2048;
    const int64_t v15 = 128;
    const int64_t v16 = 2;
    const int64_t v17 = 1;
    const int64_t v18 = 4096;
    const int64_t v19 = 32768;
    const int64_t v20 = 12288;
    const int64_t v21 = 81920;
    const int64_t v22 = 73728;
    const int64_t v23 = 8192;
    const int64_t v24 = 0;
    using T = float;

#if defined(__DAV_CUBE__)
    // pto: %x_view_inline1710_inline12436__ssa_v0_view
    int64_t v25 = v7 * v18;
    // pto: %x_view_inline1710_inline12436__ssa_v0_view
    int64_t v26 = v17 * v25;
    // pto: %x_view_inline1710_inline12436__ssa_v0_view
    pto::Shape<1, 1, 1, -1, -1> v27 = pto::Shape<1, 1, 1, -1, -1>(v17, v17, v17, v7, v18);
    // pto: %x_view_inline1710_inline12436__ssa_v0_view
    pto::Stride<-1, -1, -1, -1, -1> v28 = pto::Stride<-1, -1, -1, -1, -1>(v17 * v26, v26, v25, v18, v17);
    // pto: %x_view_inline1710_inline12436__ssa_v0_view
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v29 =
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(
            v2, v27, v28
        );
    // pto: %q_acc_inline1780_inline12492__phi_v5
    Tile<
        TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v30 = Tile<
            TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v13, v15);
    // pto: %q_acc_inline1780_inline12492__phi_v5
    uint64_t v31 = (uint64_t)v24;
    TASSIGN(v30, v31);
    // pto: %qbg_idx_inline1715_inline12573__ssa_v0
    int64_t v32 = (int64_t)v8;
    // pto: %16, %17
    int64_t v33 = (int64_t)((uint64_t)(v32 / v16) * (uint64_t)v15);
    // pto: %18
    int64_t v34 = v32 % v16;
    // pto: %19
    int64_t v35 = (int64_t)((uint64_t)v34 * (uint64_t)v14);
    // pto: %20
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    for (int64_t i36 = v24; i36 < (v4 / v13); i36 += v17) {
        // pto: %21
        int64_t v37 = (int64_t)((uint64_t)i36 * (uint64_t)v13);
        // pto: %q_acc_inline1780_inline12492__tile
        Tile<
            TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v38 = Tile<
                TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v13, v15);
        // pto: %q_acc_inline1780_inline12492__tile
        uint64_t v39 = (uint64_t)v24;
        TASSIGN(v38, v39);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        for (int64_t j40 = v24; j40 < v12; j40 += v16) {
            // pto: %22
            int64_t v41 = (int64_t)((uint64_t)j40 * (uint64_t)v11);
            // pto: %23
            int64_t v42 = (int64_t)((uint64_t)v35 + (uint64_t)v41);
            // pto: %24
            int64_t v43 = (int64_t)((uint64_t)v7 - (uint64_t)v37);
            // pto: %25
            int64_t v44 = v43 < v13 ? v43 : v13;
            // pto: %28, %27
            int64_t v45 = (int64_t)((uint64_t)v35 + (uint64_t)((int64_t)((uint64_t)v41 + (uint64_t)v11)));
            // pto: %q_x_chunk_bf16_inline1737_inline12536__tile
            Tile<
                TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v46 = Tile<
                    TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v44, v11);
            // pto: %q_x_chunk_bf16_inline1737_inline12536__tile
            uint64_t v47 = (uint64_t)v24;
            TASSIGN(v46, v47);
            // pto: %31
            int64_t v48 = v37 < v24 ? v24 : v37;
            // pto: %32
            int64_t v49 = v42 < v24 ? v24 : v42;
            // pto: %x_view_inline1710_inline12436__ssa_v0_pview
            __gm__ bfloat16_t *v50 = PTOAS__GLOBAL_TENSOR_DATA(v29);
            // pto: %x_view_inline1710_inline12436__ssa_v0_pview
            int64_t v51 = v44 * v18;
            // pto: %x_view_inline1710_inline12436__ssa_v0_pview
            int64_t v52 = v17 * v51;
            // pto: %x_view_inline1710_inline12436__ssa_v0_pview
            pto::Shape<1, 1, 1, -1, 256> v53 = pto::Shape<1, 1, 1, -1, 256>(v17, v17, v17, v44, v11);
            // pto: %x_view_inline1710_inline12436__ssa_v0_pview
            pto::Stride<-1, -1, -1, -1, -1> v54 = pto::Stride<-1, -1, -1, -1, -1>(v17 * v52, v52, v51, v18, v17);
            // pto: %x_view_inline1710_inline12436__ssa_v0_pview
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>
                v55 = GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(
                    v50 + ((v24 + v48 * v18) + v49 * v17), v53, v54
                );
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
            TLOAD(v46, v55);
            // pto: %w_chunk_inline1681_inline12563__tile
            Tile<
                TileType::Mat, bfloat16_t, 256, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v56 = Tile<
                    TileType::Mat, bfloat16_t, 256, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v11, v15);
            // pto: %w_chunk_inline1681_inline12563__tile
            uint64_t v57 = (uint64_t)v23;
            TASSIGN(v56, v57);
            // pto: %34
            int64_t v58 = v33 < v24 ? v24 : v33;
            // pto: %wq_a_last_inline507__ssa_v0_pview
            pto::Shape<1, 1, 1, 256, 128> v59 = pto::Shape<1, 1, 1, 256, 128>();
            // pto: %wq_a_last_inline507__ssa_v0_pview
            pto::Stride<262144, 262144, 262144, 1024, 1> v60 = pto::Stride<262144, 262144, 262144, 1024, 1>();
            // pto: %wq_a_last_inline507__ssa_v0_pview
            GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 256, 128>, pto::Stride<262144, 262144, 262144, 1024, 1>,
                pto::Layout::ND>
                v61 = GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 256, 128>, pto::Stride<262144, 262144, 262144, 1024, 1>,
                    pto::Layout::ND>(v3 + ((v24 + v49 * v10) + v58), v59, v60);
            TLOAD(v56, v61);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            // pto: %0
            Tile<
                TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v62 = Tile<
                    TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v44, v11);
            // pto: %0
            uint64_t v63 = (uint64_t)v22;
            TASSIGN(v62, v63);
            // pto: %36
            int64_t v64 = v45 < v24 ? v24 : v45;
            // pto: %37
            __gm__ bfloat16_t *v65 = PTOAS__GLOBAL_TENSOR_DATA(v29);
            // pto: %37
            int64_t v66 = v44 * v18;
            // pto: %37
            int64_t v67 = v17 * v66;
            // pto: %37
            pto::Shape<1, 1, 1, -1, 256> v68 = pto::Shape<1, 1, 1, -1, 256>(v17, v17, v17, v44, v11);
            // pto: %37
            pto::Stride<-1, -1, -1, -1, -1> v69 = pto::Stride<-1, -1, -1, -1, -1>(v17 * v67, v67, v66, v18, v17);
            // pto: %37
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>
                v70 = GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(
                    v65 + ((v24 + v48 * v18) + v64 * v17), v68, v69
                );
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
            TLOAD(v62, v70);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
            // pto: %1
            Tile<
                TileType::Mat, bfloat16_t, 256, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v71 = Tile<
                    TileType::Mat, bfloat16_t, 256, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v11, v15);
            // pto: %1
            uint64_t v72 = (uint64_t)v21;
            TASSIGN(v71, v72);
            // pto: %40
            pto::Shape<1, 1, 1, 256, 128> v73 = pto::Shape<1, 1, 1, 256, 128>();
            // pto: %40
            pto::Stride<262144, 262144, 262144, 1024, 1> v74 = pto::Stride<262144, 262144, 262144, 1024, 1>();
            // pto: %40
            GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 256, 128>, pto::Stride<262144, 262144, 262144, 1024, 1>,
                pto::Layout::ND>
                v75 = GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 256, 128>, pto::Stride<262144, 262144, 262144, 1024, 1>,
                    pto::Layout::ND>(v3 + ((v24 + v64 * v10) + v58), v73, v74);
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
            TLOAD(v71, v75);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
            // pto: %41
            v30.SetValidShape(v44, v15);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            if (j40 == v24) {
                // pto: %q_acc_inline1780_inline12492__tile_l0_init_storage
                Tile<
                    TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v76 = Tile<
                        TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v13, v15);
                // pto: %q_acc_inline1780_inline12492__tile_l0_init_storage
                uint64_t v77 = (uint64_t)v24;
                TASSIGN(v76, v77);
                v76.SetValidShape(v44, v15);
                // pto: %q_acc_inline1780_inline12492__tile_l0_a
                Tile<
                    TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Normal>
                    v78 = Tile<
                        TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Normal>(v44, v15);
                // pto: %q_acc_inline1780_inline12492__tile_l0_a
                uint64_t v79 = (uint64_t)v20;
                TASSIGN(v78, v79);
                TEXTRACT(v78, v46, v24, v24);
                // pto: %q_acc_inline1780_inline12492__tile_l0_b
                Tile<
                    TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>
                    v80 = Tile<
                        TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v15, v15);
                // pto: %q_acc_inline1780_inline12492__tile_l0_b
                uint64_t v81 = (uint64_t)v24;
                TASSIGN(v80, v81);
                TEXTRACT(v80, v56, v24, v24);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                // pto: %2
                Tile<
                    TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Normal>
                    v82 = Tile<
                        TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Normal>(v44, v15);
                // pto: %2
                uint64_t v83 = (uint64_t)v24;
                TASSIGN(v82, v83);
                TEXTRACT(v82, v46, v24, v15);
                // pto: %3
                Tile<
                    TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>
                    v84 = Tile<
                        TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v15, v15);
                // pto: %3
                uint64_t v85 = (uint64_t)v19;
                TASSIGN(v84, v85);
                TEXTRACT(v84, v56, v15, v24);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
                // pto: %q_acc_inline1780_inline12492__tile_l0_c_first
                Tile<
                    TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v86 = Tile<
                        TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v44, v15);
                // pto: %q_acc_inline1780_inline12492__tile_l0_c_first
                uint64_t v87 = (uint64_t)v24;
                TASSIGN(v86, v87);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                pipe_barrier(PIPE_M);
                TMATMUL(v86, v78, v80);
                // pto: %q_acc_inline1780_inline12492__tile_l0_c_acc
                Tile<
                    TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v88 = Tile<
                        TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v44, v15);
                // pto: %q_acc_inline1780_inline12492__tile_l0_c_acc
                uint64_t v89 = (uint64_t)v24;
                TASSIGN(v88, v89);
                pipe_barrier(PIPE_M);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
                TMATMUL_ACC(v88, v88, v82, v84);
            } else {
                // pto: %4
                Tile<
                    TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Normal>
                    v90 = Tile<
                        TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Normal>(v44, v15);
                // pto: %4
                uint64_t v91 = (uint64_t)v20;
                TASSIGN(v90, v91);
                TEXTRACT(v90, v46, v24, v24);
                // pto: %5
                Tile<
                    TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>
                    v92 = Tile<
                        TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v15, v15);
                // pto: %5
                uint64_t v93 = (uint64_t)v24;
                TASSIGN(v92, v93);
                TEXTRACT(v92, v56, v24, v24);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
                // pto: %6
                Tile<
                    TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Normal>
                    v94 = Tile<
                        TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Normal>(v44, v15);
                // pto: %6
                uint64_t v95 = (uint64_t)v24;
                TASSIGN(v94, v95);
                TEXTRACT(v94, v46, v24, v15);
                // pto: %7
                Tile<
                    TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>
                    v96 = Tile<
                        TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v15, v15);
                // pto: %7
                uint64_t v97 = (uint64_t)v19;
                TASSIGN(v96, v97);
                TEXTRACT(v96, v56, v15, v24);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
                // pto: %8
                Tile<
                    TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v98 = Tile<
                        TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v13, v15);
                // pto: %8
                uint64_t v99 = (uint64_t)v24;
                TASSIGN(v98, v99);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
                pipe_barrier(PIPE_M);
                TMATMUL_ACC(v98, v98, v90, v92);
                // pto: %9
                Tile<
                    TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v100 = Tile<
                        TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v13, v15);
                // pto: %9
                uint64_t v101 = (uint64_t)v24;
                TASSIGN(v100, v101);
                pipe_barrier(PIPE_M);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
                TMATMUL_ACC(v100, v100, v94, v96);
            }
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
            // pto: %10
            Tile<
                TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Normal>
                v102 = Tile<
                    TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Normal>(v44, v15);
            // pto: %10
            uint64_t v103 = (uint64_t)v18;
            TASSIGN(v102, v103);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
            TEXTRACT(v102, v62, v24, v24);
            // pto: %11
            Tile<
                TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>
                v104 = Tile<
                    TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v15);
            // pto: %11
            uint64_t v105 = (uint64_t)v24;
            TASSIGN(v104, v105);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
            TEXTRACT(v104, v71, v24, v24);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
            // pto: %12
            Tile<
                TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Normal>
                v106 = Tile<
                    TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Normal>(v44, v15);
            // pto: %12
            uint64_t v107 = (uint64_t)v23;
            TASSIGN(v106, v107);
            TEXTRACT(v106, v62, v24, v15);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
            // pto: %13
            Tile<
                TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>
                v108 = Tile<
                    TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v15);
            // pto: %13
            uint64_t v109 = (uint64_t)v19;
            TASSIGN(v108, v109);
            TEXTRACT(v108, v71, v15, v24);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
            // pto: %14
            Tile<
                TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v110 = Tile<
                    TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v13, v15);
            // pto: %14
            uint64_t v111 = (uint64_t)v24;
            TASSIGN(v110, v111);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v110, v110, v102, v104);
            // pto: %15
            Tile<
                TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v112 = Tile<
                    TileType::Acc, float, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v13, v15);
            // pto: %15
            uint64_t v113 = (uint64_t)v24;
            TASSIGN(v112, v113);
            pipe_barrier(PIPE_M);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
            TMATMUL_ACC(v112, v112, v106, v108);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        }
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        // pto: %42, %43
        int64_t v114 = (int64_t)((uint64_t)((int64_t)((uint64_t)v34 * (uint64_t)v4)) + (uint64_t)v37);
        // pto: %qr_partials_inline1686_inline12530__iter_v1_pview
        pto::Shape<1, 1, 1, 16, 128> v115 = pto::Shape<1, 1, 1, 16, 128>();
        // pto: %qr_partials_inline1686_inline12530__iter_v1_pview
        pto::Stride<16384, 16384, 16384, 1024, 1> v116 = pto::Stride<16384, 16384, 16384, 1024, 1>();
        // pto: %44, %qr_partials_inline1686_inline12530__iter_v1_pview, %45
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<16384, 16384, 16384, 1024, 1>, pto::Layout::ND>
            v117 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<16384, 16384, 16384, 1024, 1>, pto::Layout::ND>(
                v1 + ((v24 + (v114 < v24 ? v24 : v114) * v10) + (v33 < v24 ? v24 : v33)), v115, v116
            );
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        pipe_barrier(PIPE_FIX);
        TSTORE(v117, v38);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    }
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
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

    // Unpack tensor: qr_partials_inline1686_inline12530__ssa_v0
    __gm__ Tensor *qr_partials_inline1686_inline12530__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *qr_partials_inline1686_inline12530__ssa_v0 =
        reinterpret_cast<__gm__ float *>(qr_partials_inline1686_inline12530__ssa_v0_tensor->buffer.addr) +
        qr_partials_inline1686_inline12530__ssa_v0_tensor->start_offset;

    // Unpack tensor: x_view_inline1710_inline12436__ssa_v0
    __gm__ Tensor *x_view_inline1710_inline12436__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ bfloat16_t *x_view_inline1710_inline12436__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(x_view_inline1710_inline12436__ssa_v0_tensor->buffer.addr) +
        x_view_inline1710_inline12436__ssa_v0_tensor->start_offset;

    // Unpack tensor: wq_a_last_inline507__ssa_v0
    __gm__ Tensor *wq_a_last_inline507__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ bfloat16_t *wq_a_last_inline507__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(wq_a_last_inline507__ssa_v0_tensor->buffer.addr) +
        wq_a_last_inline507__ssa_v0_tensor->start_offset;

    // Unpack scalar: t_matmul_inline1739_inline12394__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } t_matmul_inline1739_inline12394__ssa_v0_conv;
    t_matmul_inline1739_inline12394__ssa_v0_conv.u64 = args[3];
    int64_t t_matmul_inline1739_inline12394__ssa_v0 = t_matmul_inline1739_inline12394__ssa_v0_conv.val;

    // Unpack scalar: t_dim_inline1772_inline12422__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } t_dim_inline1772_inline12422__ssa_v0_conv;
    t_dim_inline1772_inline12422__ssa_v0_conv.u64 = args[4];
    int64_t t_dim_inline1772_inline12422__ssa_v0 = t_dim_inline1772_inline12422__ssa_v0_conv.val;

    // Extract dynamic dim: qr_partial_rows_inline1729_inline12458__ssa_v0
    int64_t qr_partial_rows_inline1729_inline12458__ssa_v0 =
        static_cast<int64_t>(qr_partials_inline1686_inline12530__ssa_v0_tensor->shapes[0]);

    // Extract dynamic dim: t_dim_inline1772_inline12422__ssa_v0_1
    int64_t t_dim_inline1772_inline12422__ssa_v0_1 =
        static_cast<int64_t>(x_view_inline1710_inline12436__ssa_v0_tensor->shapes[0]);

    // Forward to ptoas-generated function
    qr_proj_matmul_3(
        qr_partials_inline1686_inline12530__ssa_v0, x_view_inline1710_inline12436__ssa_v0, wq_a_last_inline507__ssa_v0,
        t_matmul_inline1739_inline12394__ssa_v0, t_dim_inline1772_inline12422__ssa_v0,
        qr_partial_rows_inline1729_inline12458__ssa_v0, t_dim_inline1772_inline12422__ssa_v0_1, __pypto_spmd_block_idx,
        __pypto_spmd_block_num
    );
}
