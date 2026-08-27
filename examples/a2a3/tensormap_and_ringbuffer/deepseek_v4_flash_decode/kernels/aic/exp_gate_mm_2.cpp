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
// Kernel Function: exp_gate_mm_2

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

static __aicore__ void exp_gate_mm_2(
    __gm__ int32_t *v1, __gm__ int8_t *v2, __gm__ int8_t *v3, int64_t v4, int64_t v5, int32_t v6, int32_t v7
) {
    const int64_t v8 = 128;
    const int64_t v9 = 256;
    const int64_t v10 = 4;
    const int64_t v11 = 1024;
    const int64_t v12 = 512;
    const int64_t v13 = 1;
    const int64_t v14 = 16;
    const int64_t v15 = 4096;
    const int64_t v16 = 2048;
    const int64_t v17 = 32768;
    const int64_t v18 = 6144;
    const int64_t v19 = 147456;
    const int64_t v20 = 139264;
    const int64_t v21 = 131072;
    const int64_t v22 = 0;
    using T = float;

#if defined(__DAV_CUBE__)
    // pto: %gate_acc_inline2785_inline12137__phi_v5
    Tile<
        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v23 = Tile<
            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v14, v9);
    // pto: %gate_acc_inline2785_inline12137__phi_v5
    uint64_t v24 = (uint64_t)v22;
    TASSIGN(v23, v24);
    // pto: %gate_acc_inline2785_inline12137__tile_l0_c_phi
    Tile<
        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v25 = Tile<
            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v14, v9);
    // pto: %gate_acc_inline2785_inline12137__tile_l0_c_phi
    uint64_t v26 = (uint64_t)v22;
    TASSIGN(v25, v26);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
    for (int64_t i27 = v22; i27 < v10; i27 += v13) {
        // pto: %nb_idx_inline2761_inline12133__ssa_v0, %16, %18, %17
        int64_t v28 = (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)v6) * (uint64_t)v11)) +
                                (uint64_t)((int64_t)((uint64_t)i27 * (uint64_t)v9)));
        // pto: %gate_acc_inline2785_inline12137__tile
        Tile<
            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v29 = Tile<
                TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v14, v9);
        // pto: %gate_acc_inline2785_inline12137__tile
        uint64_t v30 = (uint64_t)v22;
        TASSIGN(v29, v30);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        for (int64_t j31 = v22; j31 < v15; j31 += v11) {
            // pto: %x_k_inline2772_inline11910__tile
            Tile<
                TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v32 = Tile<
                    TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>(v14, v12);
            // pto: %x_k_inline2772_inline11910__tile
            uint64_t v33 = (uint64_t)v21;
            TASSIGN(v32, v33);
            // pto: %19
            int64_t v34 = v4 < v22 ? v22 : v4;
            // pto: %20
            int64_t v35 = j31 < v22 ? v22 : j31;
            // pto: %recv_x_flat_inline2781_inline12084__ssa_v0_pview
            pto::Shape<1, 1, 1, 16, 512> v36 = pto::Shape<1, 1, 1, 16, 512>();
            // pto: %recv_x_flat_inline2781_inline12084__ssa_v0_pview
            pto::Stride<65536, 65536, 65536, 4096, 1> v37 = pto::Stride<65536, 65536, 65536, 4096, 1>();
            // pto: %recv_x_flat_inline2781_inline12084__ssa_v0_pview
            GlobalTensor<
                int8_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>
                v38 = GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(
                    v2 + ((v22 + v34 * v15) + v35), v36, v37
                );
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
            TLOAD(v32, v38);
            // pto: %0
            Tile<
                TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v39 = Tile<
                    TileType::Mat, int8_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>(v14, v12);
            // pto: %0
            uint64_t v40 = (uint64_t)v20;
            TASSIGN(v39, v40);
            // pto: %22
            int64_t v41 = (int64_t)((uint64_t)j31 + (uint64_t)v12);
            // pto: %23
            int64_t v42 = v41 < v22 ? v22 : v41;
            // pto: %24
            pto::Shape<1, 1, 1, 16, 512> v43 = pto::Shape<1, 1, 1, 16, 512>();
            // pto: %24
            pto::Stride<65536, 65536, 65536, 4096, 1> v44 = pto::Stride<65536, 65536, 65536, 4096, 1>();
            // pto: %24
            GlobalTensor<
                int8_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>
                v45 = GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(
                    v2 + ((v22 + v34 * v15) + v42), v43, v44
                );
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
            TLOAD(v39, v45);
            // pto: %w1_k_inline2790_inline12027__tile
            Tile<
                TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v46 = Tile<
                    TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>(v9, v12);
            // pto: %w1_k_inline2790_inline12027__tile
            uint64_t v47 = (uint64_t)v19;
            TASSIGN(v46, v47);
            // pto: %25, %26
            int64_t v48 = (int64_t)((uint64_t)((int64_t)((uint64_t)v5 * (uint64_t)v16)) + (uint64_t)v28);
            // pto: %27
            int64_t v49 = v48 < v22 ? v22 : v48;
            // pto: %w1_k_inline2790_inline12027__tile_view2d_pview
            pto::Shape<1, 1, 1, 256, 512> v50 = pto::Shape<1, 1, 1, 256, 512>();
            // pto: %w1_k_inline2790_inline12027__tile_view2d_pview
            pto::Stride<1048576, 1048576, 1048576, 4096, 1> v51 = pto::Stride<1048576, 1048576, 1048576, 4096, 1>();
            // pto: %w1_k_inline2790_inline12027__tile_view2d_pview
            GlobalTensor<
                int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<1048576, 1048576, 1048576, 4096, 1>, pto::Layout::ND>
                v52 = GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<1048576, 1048576, 1048576, 4096, 1>,
                    pto::Layout::ND>(v3 + ((v22 + v49 * v15) + v35), v50, v51);
            TLOAD(v46, v52);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            // pto: %29
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            if (j31 == v22) {
                // pto: %w1_k_inline2790_inline12027__tile_t
                Tile<
                    TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v53 = Tile<
                        TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v12, v9);
                // pto: %w1_k_inline2790_inline12027__tile_t
                uint64_t v54 = (uint64_t)v19;
                TASSIGN(v53, v54);
                // pto: %gate_acc_inline2785_inline12137__tile_l0_init
                Tile<
                    TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v55 = Tile<
                        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v14, v9);
                // pto: %gate_acc_inline2785_inline12137__tile_l0_init
                uint64_t v56 = (uint64_t)v22;
                TASSIGN(v55, v56);
                for (int64_t k57 = v22; k57 < v12; k57 += v9) {
                    // pto: %gate_acc_inline2785_inline12137__tile_l0_a
                    Tile<
                        TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v58 = Tile<
                            TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                            PadValue::Null, CompactMode::Null>(v14, v8);
                    // pto: %gate_acc_inline2785_inline12137__tile_l0_a
                    uint64_t v59 = (uint64_t)v18;
                    TASSIGN(v58, v59);
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
                    pipe_barrier(PIPE_MTE1);
                    TEXTRACT(v58, v32, v22, k57);
                    // pto: %gate_acc_inline2785_inline12137__tile_l0_b
                    Tile<
                        TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v60 = Tile<
                            TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                            PadValue::Null, CompactMode::Null>(v8, v9);
                    // pto: %gate_acc_inline2785_inline12137__tile_l0_b
                    uint64_t v61 = (uint64_t)v17;
                    TASSIGN(v60, v61);
                    TEXTRACT(v60, v53, k57, v22);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                    // pto: %1
                    Tile<
                        TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v62 = Tile<
                            TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                            PadValue::Null, CompactMode::Null>(v14, v8);
                    // pto: %1
                    uint64_t v63 = (uint64_t)v22;
                    TASSIGN(v62, v63);
                    // pto: %30
                    int64_t v64 = (int64_t)((uint64_t)k57 + (uint64_t)v8);
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
                    TEXTRACT(v62, v32, v22, v64);
                    // pto: %2
                    Tile<
                        TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v65 = Tile<
                            TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                            PadValue::Null, CompactMode::Null>(v8, v9);
                    // pto: %2
                    uint64_t v66 = (uint64_t)v22;
                    TASSIGN(v65, v66);
                    TEXTRACT(v65, v53, v64, v22);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
                    // pto: %32
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                    if (k57 == v22) {
                        // pto: %gate_acc_inline2785_inline12137__tile_l0_c_first
                        Tile<
                            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                            PadValue::Null, CompactMode::Null>
                            v67 = Tile<
                                TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                                PadValue::Null, CompactMode::Null>(v14, v9);
                        // pto: %gate_acc_inline2785_inline12137__tile_l0_c_first
                        uint64_t v68 = (uint64_t)v22;
                        TASSIGN(v67, v68);
                        pipe_barrier(PIPE_M);
                        TMATMUL(v67, v58, v60);
                    } else {
                        // pto: %gate_acc_inline2785_inline12137__tile_l0_c_acc
                        Tile<
                            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                            PadValue::Null, CompactMode::Null>
                            v69 = Tile<
                                TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                                PadValue::Null, CompactMode::Null>(v14, v9);
                        // pto: %gate_acc_inline2785_inline12137__tile_l0_c_acc
                        uint64_t v70 = (uint64_t)v22;
                        TASSIGN(v69, v70);
                        pipe_barrier(PIPE_M);
                        TMATMUL_ACC(v69, v69, v58, v60);
                    }
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
                    // pto: %3
                    Tile<
                        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>
                        v71 = Tile<
                            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                            PadValue::Null, CompactMode::Null>(v14, v9);
                    // pto: %3
                    uint64_t v72 = (uint64_t)v22;
                    TASSIGN(v71, v72);
                    pipe_barrier(PIPE_M);
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
                    TMATMUL_ACC(v71, v71, v62, v65);
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
                }
            } else {
                // pto: %4
                Tile<
                    TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v73 = Tile<
                        TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v12, v9);
                // pto: %4
                uint64_t v74 = (uint64_t)v19;
                TASSIGN(v73, v74);
                for (int64_t k75 = v22; k75 < v12; k75 += v9) {
                    // pto: %gate_acc_inline2785_inline12137__iter_v1_l0_a
                    Tile<
                        TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v76 = Tile<
                            TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                            PadValue::Null, CompactMode::Null>(v14, v8);
                    // pto: %gate_acc_inline2785_inline12137__iter_v1_l0_a
                    uint64_t v77 = (uint64_t)v18;
                    TASSIGN(v76, v77);
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
                    TEXTRACT(v76, v32, v22, k75);
                    // pto: %gate_acc_inline2785_inline12137__iter_v1_l0_b
                    Tile<
                        TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v78 = Tile<
                            TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                            PadValue::Null, CompactMode::Null>(v8, v9);
                    // pto: %gate_acc_inline2785_inline12137__iter_v1_l0_b
                    uint64_t v79 = (uint64_t)v17;
                    TASSIGN(v78, v79);
                    TEXTRACT(v78, v73, k75, v22);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
                    // pto: %5
                    Tile<
                        TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v80 = Tile<
                            TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                            PadValue::Null, CompactMode::Null>(v14, v8);
                    // pto: %5
                    uint64_t v81 = (uint64_t)v22;
                    TASSIGN(v80, v81);
                    // pto: %33
                    int64_t v82 = (int64_t)((uint64_t)k75 + (uint64_t)v8);
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
                    TEXTRACT(v80, v32, v22, v82);
                    // pto: %6
                    Tile<
                        TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v83 = Tile<
                            TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                            PadValue::Null, CompactMode::Null>(v8, v9);
                    // pto: %6
                    uint64_t v84 = (uint64_t)v22;
                    TASSIGN(v83, v84);
                    TEXTRACT(v83, v73, v82, v22);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
                    // pto: %gate_acc_inline2785_inline12137__iter_v1_l0_c_acc
                    Tile<
                        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>
                        v85 = Tile<
                            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                            PadValue::Null, CompactMode::Null>(v14, v9);
                    // pto: %gate_acc_inline2785_inline12137__iter_v1_l0_c_acc
                    uint64_t v86 = (uint64_t)v22;
                    TASSIGN(v85, v86);
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
                    pipe_barrier(PIPE_M);
                    TMATMUL_ACC(v85, v85, v76, v78);
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
                    // pto: %7
                    Tile<
                        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>
                        v87 = Tile<
                            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                            PadValue::Null, CompactMode::Null>(v14, v9);
                    // pto: %7
                    uint64_t v88 = (uint64_t)v22;
                    TASSIGN(v87, v88);
                    pipe_barrier(PIPE_M);
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
                    TMATMUL_ACC(v87, v87, v80, v83);
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
                }
            }
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
            // pto: %8
            Tile<
                TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v89 = Tile<
                    TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>(v9, v12);
            // pto: %8
            uint64_t v90 = (uint64_t)v22;
            TASSIGN(v89, v90);
            // pto: %41
            pto::Shape<1, 1, 1, 256, 512> v91 = pto::Shape<1, 1, 1, 256, 512>();
            // pto: %41
            pto::Stride<1048576, 1048576, 1048576, 4096, 1> v92 = pto::Stride<1048576, 1048576, 1048576, 4096, 1>();
            // pto: %41
            GlobalTensor<
                int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<1048576, 1048576, 1048576, 4096, 1>, pto::Layout::ND>
                v93 = GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<1048576, 1048576, 1048576, 4096, 1>,
                    pto::Layout::ND>(v3 + ((v22 + v49 * v15) + v42), v91, v92);
            TLOAD(v89, v93);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
            // pto: %9
            Tile<
                TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>
                v94 = Tile<
                    TileType::Mat, int8_t, 512, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                    CompactMode::Null>(v12, v9);
            // pto: %9
            uint64_t v95 = (uint64_t)v22;
            TASSIGN(v94, v95);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
            for (int64_t k96 = v22; k96 < v12; k96 += v9) {
                // pto: %10
                Tile<
                    TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v97 = Tile<
                        TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>(v14, v8);
                // pto: %10
                uint64_t v98 = (uint64_t)v16;
                TASSIGN(v97, v98);
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
                TEXTRACT(v97, v39, v22, k96);
                // pto: %11
                Tile<
                    TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>
                    v99 = Tile<
                        TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v8, v9);
                // pto: %11
                uint64_t v100 = (uint64_t)v17;
                TASSIGN(v99, v100);
                pipe_barrier(PIPE_MTE1);
                TEXTRACT(v99, v94, k96, v22);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
                // pto: %12
                Tile<
                    TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v101 = Tile<
                        TileType::Left, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Null>(v14, v8);
                // pto: %12
                uint64_t v102 = (uint64_t)v15;
                TASSIGN(v101, v102);
                // pto: %43
                int64_t v103 = (int64_t)((uint64_t)k96 + (uint64_t)v8);
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
                TEXTRACT(v101, v39, v22, v103);
                // pto: %13
                Tile<
                    TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>
                    v104 = Tile<
                        TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v8, v9);
                // pto: %13
                uint64_t v105 = (uint64_t)v22;
                TASSIGN(v104, v105);
                TEXTRACT(v104, v94, v103, v22);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
                // pto: %14
                Tile<
                    TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v106 = Tile<
                        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v14, v9);
                // pto: %14
                uint64_t v107 = (uint64_t)v22;
                TASSIGN(v106, v107);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
                pipe_barrier(PIPE_M);
                TMATMUL_ACC(v106, v106, v97, v99);
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
                // pto: %15
                Tile<
                    TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v108 = Tile<
                        TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v14, v9);
                // pto: %15
                uint64_t v109 = (uint64_t)v22;
                TASSIGN(v108, v109);
                pipe_barrier(PIPE_M);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
                TMATMUL_ACC(v108, v108, v101, v104);
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
            }
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        }
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        // pto: %t__tile
        Tile<
            TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v110 = Tile<
                TileType::Acc, int32_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v14, v9);
        // pto: %t__tile
        uint64_t v111 = (uint64_t)v22;
        TASSIGN(v110, v111);
        // pto: %gate_tile_i32_inline2749_inline11981__iter_v1_pview
        pto::Shape<1, 1, 1, 16, 256> v112 = pto::Shape<1, 1, 1, 16, 256>();
        // pto: %gate_tile_i32_inline2749_inline11981__iter_v1_pview
        pto::Stride<32768, 32768, 32768, 2048, 1> v113 = pto::Stride<32768, 32768, 32768, 2048, 1>();
        // pto: %45, %gate_tile_i32_inline2749_inline11981__iter_v1_pview
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND>
            v114 = GlobalTensor<
                int32_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<32768, 32768, 32768, 2048, 1>, pto::Layout::ND>(
                v1 + (v22 + (v28 < v22 ? v22 : v28)), v112, v113
            );
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        pipe_barrier(PIPE_FIX);
        TSTORE(v114, v110);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    }
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
#endif  // __DAV_CUBE__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: gate_tile_i32_inline2749_inline11981__ssa_v0
    __gm__ Tensor *gate_tile_i32_inline2749_inline11981__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int32_t *gate_tile_i32_inline2749_inline11981__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(gate_tile_i32_inline2749_inline11981__ssa_v0_tensor->buffer.addr) +
        gate_tile_i32_inline2749_inline11981__ssa_v0_tensor->start_offset;

    // Unpack tensor: recv_x_flat_inline2781_inline12084__ssa_v0
    __gm__ Tensor *recv_x_flat_inline2781_inline12084__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int8_t *recv_x_flat_inline2781_inline12084__ssa_v0 =
        reinterpret_cast<__gm__ int8_t *>(recv_x_flat_inline2781_inline12084__ssa_v0_tensor->buffer.addr) +
        recv_x_flat_inline2781_inline12084__ssa_v0_tensor->start_offset;

    // Unpack tensor: routed_w1_hca_inline615__ssa_v0
    __gm__ Tensor *routed_w1_hca_inline615__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ int8_t *routed_w1_hca_inline615__ssa_v0 =
        reinterpret_cast<__gm__ int8_t *>(routed_w1_hca_inline615__ssa_v0_tensor->buffer.addr) +
        routed_w1_hca_inline615__ssa_v0_tensor->start_offset;

    // Unpack scalar: flat_t0_inline2786_inline12094__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } flat_t0_inline2786_inline12094__ssa_v0_conv;
    flat_t0_inline2786_inline12094__ssa_v0_conv.u64 = args[3];
    int64_t flat_t0_inline2786_inline12094__ssa_v0 = flat_t0_inline2786_inline12094__ssa_v0_conv.val;

    // Unpack scalar: local_i_inline2779_inline11869__idx_v0
    union {
        uint64_t u64;
        int64_t val;
    } local_i_inline2779_inline11869__idx_v0_conv;
    local_i_inline2779_inline11869__idx_v0_conv.u64 = args[4];
    int64_t local_i_inline2779_inline11869__idx_v0 = local_i_inline2779_inline11869__idx_v0_conv.val;

    // Forward to ptoas-generated function
    exp_gate_mm_2(
        gate_tile_i32_inline2749_inline11981__ssa_v0, recv_x_flat_inline2781_inline12084__ssa_v0,
        routed_w1_hca_inline615__ssa_v0, flat_t0_inline2786_inline12094__ssa_v0, local_i_inline2779_inline11869__idx_v0,
        __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
