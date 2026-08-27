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
// Kernel Function: hca_gather_kv

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

static __aicore__ void hca_gather_kv(
    __gm__ bfloat16_t *v1, __gm__ int32_t *v2, __gm__ bfloat16_t *v3, __gm__ int32_t *v4, __gm__ int32_t *v5,
    __gm__ bfloat16_t *v6, int64_t v7, int64_t v8, int32_t v9, int32_t v10
) {
    const bfloat16_t v11 = 0.0f;
    const int64_t v12 = 15;
    const int64_t v13 = 16;
    const int64_t v14 = 256;
    const int64_t v15 = 2;
    const int64_t v16 = 32;
    const int64_t v17 = 4;
    const int64_t v18 = 128;
    const int64_t v19 = 1;
    const int64_t v20 = 512;
    const int64_t v21 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %g_task_inline1531_inline11478__ssa_v0
    int64_t v22 = (int64_t)v9;
    // pto: %5
    int64_t v23 = v22 / v17;
    // pto: %9
    int64_t v24 = (int64_t)((uint64_t)v23 * (uint64_t)v14);
    // pto: %7, %6, %10
    int64_t v25 =
        (int64_t)((uint64_t)((int64_t)((uint64_t)v22 - (uint64_t)((int64_t)((uint64_t)v23 * (uint64_t)v17)))) *
                  (uint64_t)v16);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    for (int64_t i26 = v21; i26 < v15; i26 += v19) {
        // pto: %12, %11
        int64_t v27 = (int64_t)((uint64_t)v25 + (uint64_t)((int64_t)((uint64_t)i26 * (uint64_t)v13)));
        // pto: %13
        int64_t v28 = (int64_t)((uint64_t)v24 + (uint64_t)v27);
        // pto: %flat_offset_mul
        int64_t v29 = (int64_t)((uint64_t)v23 * (uint64_t)v18);
        // pto: %flat_offset, %g_first_inline1545_inline11446__tile
        int32_t v30 = (v2)[(int64_t)((uint64_t)v29 + (uint64_t)v27)];
        // pto: %16, %14, %g_last_inline1590_inline11474__tile
        int32_t v31 = (v2)[(int64_t)((uint64_t)v29 + (uint64_t)((int64_t)((uint64_t)v27 + (uint64_t)v12)))];
        // pto: %19
        int64_t v32 = (int64_t)v30;
        // pto: %17, %18, %22, %20, %21, %23
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        if ((int64_t)((uint64_t)((int64_t)((int32_t)((uint32_t)v31 - (uint32_t)v30))) +
                      (uint64_t)((int64_t)((uint64_t)(v32 < v21 ? v32 : v21) * (uint64_t)v13))) == v12) {
            // pto: %t__tile
            Tile<
                TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v33 = Tile<
                    TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                    PadValue::Null, CompactMode::Null>(v13, v20);
            // pto: %t__tile
            uint64_t v34 = (uint64_t)v21;
            TASSIGN(v33, v34);
            // pto: %ori_kv_flat_inline1489_inline11377__ssa_v0_pview
            pto::Shape<1, 1, 1, 16, 512> v35 = pto::Shape<1, 1, 1, 16, 512>();
            // pto: %ori_kv_flat_inline1489_inline11377__ssa_v0_pview
            pto::Stride<8192, 8192, 8192, 512, 1> v36 = pto::Stride<8192, 8192, 8192, 512, 1>();
            // pto: %25, %ori_kv_flat_inline1489_inline11377__ssa_v0_pview
            GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>
                v37 = GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
                    v3 + (v21 + (v32 < v21 ? v21 : v32) * v20), v35, v36
                );
            TLOAD(v33, v37);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            // pto: %hca_kv_flat_inline1485_inline11368__iter_v1_pview
            pto::Shape<1, 1, 1, 16, 512> v38 = pto::Shape<1, 1, 1, 16, 512>();
            // pto: %hca_kv_flat_inline1485_inline11368__iter_v1_pview
            pto::Stride<8192, 8192, 8192, 512, 1> v39 = pto::Stride<8192, 8192, 8192, 512, 1>();
            // pto: %26, %hca_kv_flat_inline1485_inline11368__iter_v1_pview
            GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>
                v40 = GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(
                    v1 + (v21 + (v28 < v21 ? v21 : v28) * v20), v38, v39
                );
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(v40, v33);
        } else {
            for (int64_t j41 = v21; j41 < v13; j41 += v19) {
                // pto: %27
                int64_t v42 = (int64_t)((uint64_t)v28 + (uint64_t)j41);
                // pto: %30, %28, %g_win_slot_i32_inline1512_inline11795__tile
                int32_t v43 = (v2)[(int64_t)((uint64_t)v29 + (uint64_t)((int64_t)((uint64_t)v27 + (uint64_t)j41)))];
                // pto: %31
                int64_t v44 = (int64_t)v43;
                // pto: %32
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
                if (v44 >= v21) {
                    // pto: %0
                    Tile<
                        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                        PadValue::Null, CompactMode::Null>
                        v45 = Tile<
                            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                            PadValue::Null, CompactMode::Null>(v19, v20);
                    // pto: %0
                    uint64_t v46 = (uint64_t)v21;
                    TASSIGN(v45, v46);
                    // pto: %35
                    pto::Shape<1, 1, 1, 1, 512> v47 = pto::Shape<1, 1, 1, 1, 512>();
                    // pto: %35
                    pto::Stride<512, 512, 512, 512, 1> v48 = pto::Stride<512, 512, 512, 512, 1>();
                    // pto: %34, %35
                    GlobalTensor<
                        bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>
                        v49 = GlobalTensor<
                            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>,
                            pto::Layout::ND>(v3 + (v21 + (v44 < v21 ? v21 : v44) * v20), v47, v48);
                    TLOAD(v45, v49);
                    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
                    // pto: %hca_kv_flat_inline1485_inline11368__iter_v4_pview
                    pto::Shape<1, 1, 1, 1, 512> v50 = pto::Shape<1, 1, 1, 1, 512>();
                    // pto: %hca_kv_flat_inline1485_inline11368__iter_v4_pview
                    pto::Stride<512, 512, 512, 512, 1> v51 = pto::Stride<512, 512, 512, 512, 1>();
                    // pto: %37, %hca_kv_flat_inline1485_inline11368__iter_v4_pview
                    GlobalTensor<
                        bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>
                        v52 = GlobalTensor<
                            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>,
                            pto::Layout::ND>(v1 + (v21 + (v42 < v21 ? v21 : v42) * v20), v50, v51);
                    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
                    TSTORE(v52, v45);
                } else {
                    // pto: %1
                    Tile<
                        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                        PadValue::Null, CompactMode::Null>
                        v53 = Tile<
                            TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                            PadValue::Null, CompactMode::Null>(v19, v20);
                    // pto: %1
                    uint64_t v54 = (uint64_t)v21;
                    TASSIGN(v53, v54);
                    TEXPANDS(v53, v11);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    // pto: %40
                    pto::Shape<1, 1, 1, 1, 512> v55 = pto::Shape<1, 1, 1, 1, 512>();
                    // pto: %40
                    pto::Stride<512, 512, 512, 512, 1> v56 = pto::Stride<512, 512, 512, 512, 1>();
                    // pto: %39, %40
                    GlobalTensor<
                        bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>
                        v57 = GlobalTensor<
                            bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>,
                            pto::Layout::ND>(v1 + (v21 + (v42 < v21 ? v21 : v42) * v20), v55, v56);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    TSTORE(v57, v53);
                }
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            }
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    for (int64_t i58 = v21; i58 < v16; i58 += v19) {
        // pto: %42, %43, %44
        int64_t v59 =
            (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)v24 + (uint64_t)v25)) + (uint64_t)v18)) +
                      (uint64_t)i58);
        // pto: %45
        int64_t v60 = (int64_t)((uint64_t)v25 + (uint64_t)i58);
        // pto: %46
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        if (v60 < v18) {
            // pto: %47, %48, %g_ridx_inline1482_inline11678__tile
            int32_t v61 = (v4)[(int64_t)((uint64_t)((int64_t)((uint64_t)v23 * (uint64_t)v18)) + (uint64_t)v60)];
            // pto: %49
            int64_t v62 = (int64_t)v61;
            // pto: %50
            if (v62 >= v21) {
                // pto: %8, %54, %55, %53, %51
                int32_t v63 = (v5)[(int64_t)((uint64_t)((int64_t)((uint64_t)(v23 / v15) * (uint64_t)v16)) +
                                             (uint64_t)(v62 / v18))];
                // pto: %56, %57, %60, %59
                int64_t v64 =
                    (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)v63) * (uint64_t)v18)) + (uint64_t)(v62 % v18));
                // pto: %2
                Tile<
                    TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v65 = Tile<
                        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                        PadValue::Null, CompactMode::Null>(v19, v20);
                // pto: %2
                uint64_t v66 = (uint64_t)v21;
                TASSIGN(v65, v66);
                // pto: %cmp_kv_flat_inline1500_inline11375__ssa_v0_pview
                pto::Shape<1, 1, 1, 1, 512> v67 = pto::Shape<1, 1, 1, 1, 512>();
                // pto: %cmp_kv_flat_inline1500_inline11375__ssa_v0_pview
                pto::Stride<512, 512, 512, 512, 1> v68 = pto::Stride<512, 512, 512, 512, 1>();
                // pto: %61, %cmp_kv_flat_inline1500_inline11375__ssa_v0_pview
                GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>
                    v69 = GlobalTensor<
                        bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(
                        v6 + (v21 + (v64 < v21 ? v21 : v64) * v20), v67, v68
                    );
                TLOAD(v65, v69);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
                // pto: %hca_kv_flat_inline1485_inline11368__iter_v10_pview
                pto::Shape<1, 1, 1, 1, 512> v70 = pto::Shape<1, 1, 1, 1, 512>();
                // pto: %hca_kv_flat_inline1485_inline11368__iter_v10_pview
                pto::Stride<512, 512, 512, 512, 1> v71 = pto::Stride<512, 512, 512, 512, 1>();
                // pto: %63, %hca_kv_flat_inline1485_inline11368__iter_v10_pview
                GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>
                    v72 = GlobalTensor<
                        bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(
                        v1 + (v21 + (v59 < v21 ? v21 : v59) * v20), v70, v71
                    );
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
                TSTORE(v72, v65);
            } else {
                // pto: %3
                Tile<
                    TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v73 = Tile<
                        TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                        PadValue::Null, CompactMode::Null>(v19, v20);
                // pto: %3
                uint64_t v74 = (uint64_t)v21;
                TASSIGN(v73, v74);
                TEXPANDS(v73, v11);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                // pto: %66
                pto::Shape<1, 1, 1, 1, 512> v75 = pto::Shape<1, 1, 1, 1, 512>();
                // pto: %66
                pto::Stride<512, 512, 512, 512, 1> v76 = pto::Stride<512, 512, 512, 512, 1>();
                // pto: %65, %66
                GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>
                    v77 = GlobalTensor<
                        bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(
                        v1 + (v21 + (v59 < v21 ? v21 : v59) * v20), v75, v76
                    );
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                TSTORE(v77, v73);
            }
        } else {
            // pto: %4
            Tile<
                TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v78 = Tile<
                    TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v19, v20);
            // pto: %4
            uint64_t v79 = (uint64_t)v21;
            TASSIGN(v78, v79);
            TEXPANDS(v78, v11);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
            // pto: %69
            pto::Shape<1, 1, 1, 1, 512> v80 = pto::Shape<1, 1, 1, 1, 512>();
            // pto: %69
            pto::Stride<512, 512, 512, 512, 1> v81 = pto::Stride<512, 512, 512, 512, 1>();
            // pto: %68, %69
            GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>
                v82 = GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(
                    v1 + (v21 + (v59 < v21 ? v21 : v59) * v20), v80, v81
                );
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
            TSTORE(v82, v78);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: hca_kv_flat_inline1485_inline11368__ssa_v0
    __gm__ Tensor *hca_kv_flat_inline1485_inline11368__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ bfloat16_t *hca_kv_flat_inline1485_inline11368__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(hca_kv_flat_inline1485_inline11368__ssa_v0_tensor->buffer.addr) +
        hca_kv_flat_inline1485_inline11368__ssa_v0_tensor->start_offset;

    // Unpack tensor: swa_indices_inline636__ssa_v1
    __gm__ Tensor *swa_indices_inline636__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int32_t *swa_indices_inline636__ssa_v1 =
        reinterpret_cast<__gm__ int32_t *>(swa_indices_inline636__ssa_v1_tensor->buffer.addr) +
        swa_indices_inline636__ssa_v1_tensor->start_offset;

    // Unpack tensor: ori_kv_flat_inline1489_inline11377__ssa_v0
    __gm__ Tensor *ori_kv_flat_inline1489_inline11377__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ bfloat16_t *ori_kv_flat_inline1489_inline11377__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(ori_kv_flat_inline1489_inline11377__ssa_v0_tensor->buffer.addr) +
        ori_kv_flat_inline1489_inline11377__ssa_v0_tensor->start_offset;

    // Unpack tensor: topk_all_inline11389__ssa_v0
    __gm__ Tensor *topk_all_inline11389__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ int32_t *topk_all_inline11389__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(topk_all_inline11389__ssa_v0_tensor->buffer.addr) +
        topk_all_inline11389__ssa_v0_tensor->start_offset;

    // Unpack tensor: cmp_block_table__ssa_v0
    __gm__ Tensor *cmp_block_table__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ int32_t *cmp_block_table__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(cmp_block_table__ssa_v0_tensor->buffer.addr) +
        cmp_block_table__ssa_v0_tensor->start_offset;

    // Unpack tensor: cmp_kv_flat_inline1500_inline11375__ssa_v0
    __gm__ Tensor *cmp_kv_flat_inline1500_inline11375__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[5]);
    __gm__ bfloat16_t *cmp_kv_flat_inline1500_inline11375__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(cmp_kv_flat_inline1500_inline11375__ssa_v0_tensor->buffer.addr) +
        cmp_kv_flat_inline1500_inline11375__ssa_v0_tensor->start_offset;

    // Extract dynamic dim: ori_block_num_inline1520_inline11500__ssa_v0
    int64_t ori_block_num_inline1520_inline11500__ssa_v0 =
        (static_cast<int64_t>(ori_kv_flat_inline1489_inline11377__ssa_v0_tensor->shapes[0]) / 128);

    // Extract dynamic dim: cmp_block_num_inline1530_inline11379__ssa_v0
    int64_t cmp_block_num_inline1530_inline11379__ssa_v0 =
        (static_cast<int64_t>(cmp_kv_flat_inline1500_inline11375__ssa_v0_tensor->shapes[0]) / 128);

    // Forward to ptoas-generated function
    hca_gather_kv(
        hca_kv_flat_inline1485_inline11368__ssa_v0, swa_indices_inline636__ssa_v1,
        ori_kv_flat_inline1489_inline11377__ssa_v0, topk_all_inline11389__ssa_v0, cmp_block_table__ssa_v0,
        cmp_kv_flat_inline1500_inline11375__ssa_v0, ori_block_num_inline1520_inline11500__ssa_v0,
        cmp_block_num_inline1530_inline11379__ssa_v0, __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
