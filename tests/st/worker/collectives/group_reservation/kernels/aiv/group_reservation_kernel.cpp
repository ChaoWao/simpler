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

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "platform_comm/comm_context.h"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include "intrinsic.h"

#if defined(__CCE_AICORE__) && !defined(__CPU_SIM) && !defined(__COSTMODEL)
#define GROUP_RESERVATION_USE_DEV_INTRIN 1
#else
#define GROUP_RESERVATION_USE_DEV_INTRIN 0
#endif

enum class Operation : int32_t {
    FIRST_GROUP = 0,
    RELEASE_FIRST_GROUP = 1,
    SECOND_GROUP = 2,
    CHECK_SECOND_GROUP = 3,
};

static __aicore__ inline void store_i32(__gm__ int32_t *ptr, int32_t value) {
#if GROUP_RESERVATION_USE_DEV_INTRIN
    st_dev(static_cast<uint32_t>(value), reinterpret_cast<__gm__ uint32_t *>(ptr), 0);
#else
    *reinterpret_cast<volatile int32_t *>(ptr) = value;
#endif
}

static __aicore__ inline int32_t load_i32(__gm__ int32_t *ptr) {
#if GROUP_RESERVATION_USE_DEV_INTRIN
    return static_cast<int32_t>(ld_dev(reinterpret_cast<__gm__ uint32_t *>(ptr), 0));
#else
    return *reinterpret_cast<volatile int32_t *>(ptr);
#endif
}

static __aicore__ inline void ddr_fence() {
#if GROUP_RESERVATION_USE_DEV_INTRIN
    dsb(DSB_DDR);
#endif
}

template <typename T>
static __aicore__ inline __gm__ T *remote_ptr(__gm__ CommContext *context, __gm__ T *local_ptr, int peer_rank) {
    uint64_t local_base = context->windowsIn[context->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(local_ptr) - local_base;
    return reinterpret_cast<__gm__ T *>(context->windowsIn[peer_rank] + offset);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    if (get_block_idx(args) != 0) {
        return;
    }

    __gm__ ChipTensor *output_tensor = reinterpret_cast<__gm__ ChipTensor *>(args[0]);
    Operation operation = static_cast<Operation>(args[1]);
    __gm__ int32_t *state = reinterpret_cast<__gm__ int32_t *>(args[2]);
    __gm__ CommContext *context = reinterpret_cast<__gm__ CommContext *>(args[3]);
    __gm__ int32_t *output =
        reinterpret_cast<__gm__ int32_t *>(output_tensor->buffer.addr) + output_tensor->start_offset;

    int rank = static_cast<int>(context->rankId);
    int32_t marker = -1;

    switch (operation) {
    case Operation::FIRST_GROUP:
        if (rank == 1) {
            pto::comm::Signal signal(state);
            pto::comm::TWAIT(signal, static_cast<int32_t>(1), pto::comm::WaitCmp::GE);
        }
        marker = 10 + rank;
        break;
    case Operation::RELEASE_FIRST_GROUP: {
        __gm__ int32_t *remote_state = remote_ptr(context, state, 1);
        pto::comm::Signal signal(remote_state);
        pto::comm::TNOTIFY(signal, static_cast<int32_t>(1), pto::comm::NotifyOp::AtomicAdd);
        marker = 20;
        break;
    }
    case Operation::SECOND_GROUP:
        if (rank == 2) {
            store_i32(state + 1, 1);
            ddr_fence();
        }
        marker = 30 + rank;
        break;
    case Operation::CHECK_SECOND_GROUP:
        ddr_fence();
        marker = load_i32(state + 1) >= 1 ? 1 : 0;
        break;
    }

    store_i32(output, marker);
    ddr_fence();
    pipe_barrier(PIPE_ALL);
}
