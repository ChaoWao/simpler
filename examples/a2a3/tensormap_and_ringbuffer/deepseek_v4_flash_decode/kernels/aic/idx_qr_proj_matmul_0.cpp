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
// Kernel Function: idx_qr_proj_matmul_0

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
idx_qr_proj_matmul_0(__gm__ int32_t *v1, __gm__ int8_t *v2, __gm__ int8_t *v3, int32_t v4, int32_t v5) {
    const int64_t v6 = 8192;
    const int64_t v7 = 64;
    const int64_t v8 = 128;
    const int64_t v9 = 256;
    const int64_t v10 = 2;
    const int64_t v11 = 4;
    const int64_t v12 = 512;
    const int64_t v13 = 8;
    const int64_t v14 = 16;
    const int64_t v15 = 2048;
    const int64_t v16 = 1024;
    const int64_t v17 = 32768;
    const int64_t v18 = 3072;
    const int64_t v19 = 139264;
    const int64_t v20 = 135168;
    const int64_t v21 = 4096;
    const int64_t v22 = 0;
    using T = float;

#if defined(__DAV_CUBE__)
    // pto: %qr_acc_inline1974_inline12784__phi_v5
    Tile<
        TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v23 = Tile<
            TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v13, v12);
    // pto: %qr_acc_inline1974_inline12784__phi_v5
    uint64_t v24 = (uint64_t)v22;
    TASSIGN(v23, v24);
    // pto: %qr_acc_inline1974_inline12784__tile_l0_c_phi
    Tile<
        TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v25 = Tile<
            TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v13, v12);
    // pto: %qr_acc_inline1974_inline12784__tile_l0_c_phi
    uint64_t v26 = (uint64_t)v22;
    TASSIGN(v25, v26);
    // pto: %ot_inline1983_inline12479__ssa_v0, %17
    int64_t v27 = (int64_t)((uint64_t)((int64_t)v4) * (uint64_t)v16);
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
    for (int64_t i28 = v22; i28 < v16; i28 += v12) {
        // pto: %qr_acc_inline1974_inline12784__tile
        Tile<
            TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v29 = Tile<
                TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v14, v12);
        // pto: %qr_acc_inline1974_inline12784__tile
        uint64_t v30 = (uint64_t)v22;
        TASSIGN(v29, v30);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        for (int64_t j31 = v22; j31 < v11; j31 += v10) {
            // pto: %18
            int64_t v32 = (int64_t)((uint64_t)j31 * (uint64_t)v9);
            // pto: %20
            int64_t v33 = (int64_t)((uint64_t)v32 + (uint64_t)v9);
            // pto: %qr_tile_inline1993_inline12420__tile
            Tile<
                TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v34 = Tile<
                    TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>(v13, v9);
            // pto: %qr_tile_inline1993_inline12420__tile
            uint64_t v35 = (uint64_t)v22;
            TASSIGN(v34, v35);
            // pto: %21
            int64_t v36 = v32 < v22 ? v22 : v32;
            // pto: %qr_inline12486__ssa_v0_pview
            pto::Shape<1, 1, 1, 8, 256> v37 = pto::Shape<1, 1, 1, 8, 256>();
            // pto: %qr_inline12486__ssa_v0_pview
            pto::Stride<8192, 8192, 8192, 1024, 1> v38 = pto::Stride<8192, 8192, 8192, 1024, 1>();
            // pto: %qr_inline12486__ssa_v0_pview
            GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND>
                v39 = GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND>(
                    v2 + (v22 + v36), v37, v38
                );
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
            TLOAD(v34, v39);
            // pto: %wq_tile_inline1988_inline12299__tile
            Tile<
                TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v40 = Tile<
                    TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>(v9, v12);
            // pto: %wq_tile_inline1988_inline12299__tile
            uint64_t v41 = (uint64_t)v21;
            TASSIGN(v40, v41);
            // pto: %23
            int64_t v42 = (int64_t)((uint64_t)v27 + (uint64_t)i28);
            // pto: %24
            int64_t v43 = v42 < v22 ? v22 : v42;
            // pto: %csa_idx_wq_b_last_inline605__ssa_v0_pview
            pto::Shape<1, 1, 1, 256, 512> v44 = pto::Shape<1, 1, 1, 256, 512>();
            // pto: %csa_idx_wq_b_last_inline605__ssa_v0_pview
            pto::Stride<2097152, 2097152, 2097152, 8192, 1> v45 = pto::Stride<2097152, 2097152, 2097152, 8192, 1>();
            // pto: %csa_idx_wq_b_last_inline605__ssa_v0_pview
            GlobalTensor<
                int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>, pto::Layout::ND>
                v46 = GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>,
                    pto::Layout::ND>(v3 + ((v22 + v36 * v6) + v43), v44, v45);
            TLOAD(v40, v46);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            // pto: %0
            Tile<
                TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v47 = Tile<
                    TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>(v13, v9);
            // pto: %0
            uint64_t v48 = (uint64_t)v20;
            TASSIGN(v47, v48);
            // pto: %25
            int64_t v49 = v33 < v22 ? v22 : v33;
            // pto: %26
            pto::Shape<1, 1, 1, 8, 256> v50 = pto::Shape<1, 1, 1, 8, 256>();
            // pto: %26
            pto::Stride<8192, 8192, 8192, 1024, 1> v51 = pto::Stride<8192, 8192, 8192, 1024, 1>();
            // pto: %26
            GlobalTensor<int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND>
                v52 = GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 8, 256>, pto::Stride<8192, 8192, 8192, 1024, 1>, pto::Layout::ND>(
                    v2 + (v22 + v49), v50, v51
                );
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
            TLOAD(v47, v52);
            // pto: %1
            Tile<
                TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v53 = Tile<
                    TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Null>(v9, v12);
            // pto: %1
            uint64_t v54 = (uint64_t)v19;
            TASSIGN(v53, v54);
            // pto: %30
            pto::Shape<1, 1, 1, 256, 512> v55 = pto::Shape<1, 1, 1, 256, 512>();
            // pto: %30
            pto::Stride<2097152, 2097152, 2097152, 8192, 1> v56 = pto::Stride<2097152, 2097152, 2097152, 8192, 1>();
            // pto: %30
            GlobalTensor<
                int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>, pto::Layout::ND>
                v57 = GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>,
                    pto::Layout::ND>(v3 + ((v22 + v49 * v6) + v43), v55, v56);
            TLOAD(v53, v57);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
            // pto: %31
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            if (v32 == v22) {
                // pto: %qr_acc_inline1974_inline12784__tile_l0_init_storage
                Tile<
                    TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v58 = Tile<
                        TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v14, v12);
                // pto: %qr_acc_inline1974_inline12784__tile_l0_init_storage
                uint64_t v59 = (uint64_t)v22;
                TASSIGN(v58, v59);
                v58.SetValidShape(v13, v12);
                for (int64_t k60 = v22; k60 < v9; k60 += v8) {
                    // pto: %qr_acc_inline1974_inline12784__tile_l0_a
                    Tile<
                        TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Normal>
                        v61 = Tile<
                            TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                            PadValue::Null, CompactMode::Normal>(v13, v7);
                    // pto: %qr_acc_inline1974_inline12784__tile_l0_a
                    uint64_t v62 = (uint64_t)v18;
                    TASSIGN(v61, v62);
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
                    pipe_barrier(PIPE_MTE1);
                    TEXTRACT(v61, v34, v22, k60);
                    // pto: %qr_acc_inline1974_inline12784__tile_l0_b
                    Tile<
                        TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v63 = Tile<
                            TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                            PadValue::Null, CompactMode::Null>(v7, v12);
                    // pto: %qr_acc_inline1974_inline12784__tile_l0_b
                    uint64_t v64 = (uint64_t)v22;
                    TASSIGN(v63, v64);
                    TEXTRACT(v63, v40, k60, v22);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                    // pto: %2
                    Tile<
                        TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Normal>
                        v65 = Tile<
                            TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                            PadValue::Null, CompactMode::Normal>(v13, v7);
                    // pto: %2
                    uint64_t v66 = (uint64_t)v22;
                    TASSIGN(v65, v66);
                    // pto: %32
                    int64_t v67 = (int64_t)((uint64_t)k60 + (uint64_t)v7);
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
                    TEXTRACT(v65, v34, v22, v67);
                    // pto: %3
                    Tile<
                        TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v68 = Tile<
                            TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                            PadValue::Null, CompactMode::Null>(v7, v12);
                    // pto: %3
                    uint64_t v69 = (uint64_t)v17;
                    TASSIGN(v68, v69);
                    TEXTRACT(v68, v40, v67, v22);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
                    // pto: %34
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                    if (k60 == v22) {
                        // pto: %qr_acc_inline1974_inline12784__tile_l0_c_first
                        Tile<
                            TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                            PadValue::Null, CompactMode::Null>
                            v70 = Tile<
                                TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                                PadValue::Null, CompactMode::Null>(v13, v12);
                        // pto: %qr_acc_inline1974_inline12784__tile_l0_c_first
                        uint64_t v71 = (uint64_t)v22;
                        TASSIGN(v70, v71);
                        pipe_barrier(PIPE_M);
                        TMATMUL(v70, v61, v63);
                    } else {
                        // pto: %qr_acc_inline1974_inline12784__tile_l0_c_acc
                        Tile<
                            TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                            PadValue::Null, CompactMode::Null>
                            v72 = Tile<
                                TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                                PadValue::Null, CompactMode::Null>(v13, v12);
                        // pto: %qr_acc_inline1974_inline12784__tile_l0_c_acc
                        uint64_t v73 = (uint64_t)v22;
                        TASSIGN(v72, v73);
                        pipe_barrier(PIPE_M);
                        TMATMUL_ACC(v72, v72, v61, v63);
                    }
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
                    // pto: %4
                    Tile<
                        TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>
                        v74 = Tile<
                            TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                            PadValue::Null, CompactMode::Null>(v13, v12);
                    // pto: %4
                    uint64_t v75 = (uint64_t)v22;
                    TASSIGN(v74, v75);
                    pipe_barrier(PIPE_M);
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
                    TMATMUL_ACC(v74, v74, v65, v68);
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
                }
            } else {
                for (int64_t k76 = v22; k76 < v9; k76 += v8) {
                    // pto: %5
                    Tile<
                        TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Normal>
                        v77 = Tile<
                            TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                            PadValue::Null, CompactMode::Normal>(v13, v7);
                    // pto: %5
                    uint64_t v78 = (uint64_t)v18;
                    TASSIGN(v77, v78);
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
                    TEXTRACT(v77, v34, v22, k76);
                    // pto: %6
                    Tile<
                        TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v79 = Tile<
                            TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                            PadValue::Null, CompactMode::Null>(v7, v12);
                    // pto: %6
                    uint64_t v80 = (uint64_t)v22;
                    TASSIGN(v79, v80);
                    TEXTRACT(v79, v40, k76, v22);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
                    // pto: %7
                    Tile<
                        TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Normal>
                        v81 = Tile<
                            TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                            PadValue::Null, CompactMode::Normal>(v13, v7);
                    // pto: %7
                    uint64_t v82 = (uint64_t)v22;
                    TASSIGN(v81, v82);
                    // pto: %36
                    int64_t v83 = (int64_t)((uint64_t)k76 + (uint64_t)v7);
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
                    TEXTRACT(v81, v34, v22, v83);
                    // pto: %8
                    Tile<
                        TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>
                        v84 = Tile<
                            TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                            PadValue::Null, CompactMode::Null>(v7, v12);
                    // pto: %8
                    uint64_t v85 = (uint64_t)v17;
                    TASSIGN(v84, v85);
                    TEXTRACT(v84, v40, v83, v22);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
                    // pto: %9
                    Tile<
                        TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>
                        v86 = Tile<
                            TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                            PadValue::Null, CompactMode::Null>(v14, v12);
                    // pto: %9
                    uint64_t v87 = (uint64_t)v22;
                    TASSIGN(v86, v87);
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
                    pipe_barrier(PIPE_M);
                    TMATMUL_ACC(v86, v86, v77, v79);
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
                    // pto: %10
                    Tile<
                        TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>
                        v88 = Tile<
                            TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                            PadValue::Null, CompactMode::Null>(v14, v12);
                    // pto: %10
                    uint64_t v89 = (uint64_t)v22;
                    TASSIGN(v88, v89);
                    pipe_barrier(PIPE_M);
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
                    TMATMUL_ACC(v88, v88, v81, v84);
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
                }
            }
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
            for (int64_t k90 = v22; k90 < v9; k90 += v8) {
                // pto: %11
                Tile<
                    TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Normal>
                    v91 = Tile<
                        TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Normal>(v13, v7);
                // pto: %11
                uint64_t v92 = (uint64_t)v16;
                TASSIGN(v91, v92);
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
                TEXTRACT(v91, v47, v22, k90);
                // pto: %12
                Tile<
                    TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v93 = Tile<
                        TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v7, v12);
                // pto: %12
                uint64_t v94 = (uint64_t)v22;
                TASSIGN(v93, v94);
                pipe_barrier(PIPE_MTE1);
                TEXTRACT(v93, v53, k90, v22);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
                // pto: %13
                Tile<
                    TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                    CompactMode::Normal>
                    v95 = Tile<
                        TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                        PadValue::Null, CompactMode::Normal>(v13, v7);
                // pto: %13
                uint64_t v96 = (uint64_t)v15;
                TASSIGN(v95, v96);
                // pto: %39
                int64_t v97 = (int64_t)((uint64_t)k90 + (uint64_t)v7);
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
                TEXTRACT(v95, v47, v22, v97);
                // pto: %14
                Tile<
                    TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                    CompactMode::Null>
                    v98 = Tile<
                        TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                        PadValue::Null, CompactMode::Null>(v7, v12);
                // pto: %14
                uint64_t v99 = (uint64_t)v17;
                TASSIGN(v98, v99);
                TEXTRACT(v98, v53, v97, v22);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
                // pto: %15
                Tile<
                    TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v100 = Tile<
                        TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v14, v12);
                // pto: %15
                uint64_t v101 = (uint64_t)v22;
                TASSIGN(v100, v101);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
                pipe_barrier(PIPE_M);
                TMATMUL_ACC(v100, v100, v91, v93);
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
                // pto: %16
                Tile<
                    TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>
                    v102 = Tile<
                        TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024,
                        PadValue::Null, CompactMode::Null>(v14, v12);
                // pto: %16
                uint64_t v103 = (uint64_t)v22;
                TASSIGN(v102, v103);
                pipe_barrier(PIPE_M);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
                TMATMUL_ACC(v102, v102, v95, v98);
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
            }
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        }
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        // pto: %41
        int64_t v104 = (int64_t)((uint64_t)v27 + (uint64_t)i28);
        // pto: %qr_acc_pad_inline2023_inline12697__iter_v1_pview
        pto::Shape<1, 1, 1, 16, 512> v105 = pto::Shape<1, 1, 1, 16, 512>();
        // pto: %qr_acc_pad_inline2023_inline12697__iter_v1_pview
        pto::Stride<131072, 131072, 131072, 8192, 1> v106 = pto::Stride<131072, 131072, 131072, 8192, 1>();
        // pto: %42, %qr_acc_pad_inline2023_inline12697__iter_v1_pview
        GlobalTensor<
            int32_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<131072, 131072, 131072, 8192, 1>, pto::Layout::ND>
            v107 = GlobalTensor<
                int32_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<131072, 131072, 131072, 8192, 1>, pto::Layout::ND>(
                v1 + (v22 + (v104 < v22 ? v22 : v104)), v105, v106
            );
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        pipe_barrier(PIPE_FIX);
        TSTORE(v107, v29);
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

    // Unpack tensor: qr_acc_pad_inline2023_inline12697__ssa_v0
    __gm__ Tensor *qr_acc_pad_inline2023_inline12697__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int32_t *qr_acc_pad_inline2023_inline12697__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(qr_acc_pad_inline2023_inline12697__ssa_v0_tensor->buffer.addr) +
        qr_acc_pad_inline2023_inline12697__ssa_v0_tensor->start_offset;

    // Unpack tensor: qr_inline12486__ssa_v0
    __gm__ Tensor *qr_inline12486__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int8_t *qr_inline12486__ssa_v0 =
        reinterpret_cast<__gm__ int8_t *>(qr_inline12486__ssa_v0_tensor->buffer.addr) +
        qr_inline12486__ssa_v0_tensor->start_offset;

    // Unpack tensor: csa_idx_wq_b_last_inline605__ssa_v0
    __gm__ Tensor *csa_idx_wq_b_last_inline605__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ int8_t *csa_idx_wq_b_last_inline605__ssa_v0 =
        reinterpret_cast<__gm__ int8_t *>(csa_idx_wq_b_last_inline605__ssa_v0_tensor->buffer.addr) +
        csa_idx_wq_b_last_inline605__ssa_v0_tensor->start_offset;

    // Forward to ptoas-generated function
    idx_qr_proj_matmul_0(
        qr_acc_pad_inline2023_inline12697__ssa_v0, qr_inline12486__ssa_v0, csa_idx_wq_b_last_inline605__ssa_v0,
        __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
