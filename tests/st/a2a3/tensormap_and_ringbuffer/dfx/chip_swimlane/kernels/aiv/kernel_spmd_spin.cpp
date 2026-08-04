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

/**
 * An AIV blocker that occupies every running slot for one measured second.
 * The wall-clock hold is independent of core execution speed and leaves a
 * large margin for the orchestration's scheduler-loop fences.
 *
 * Each block owns one status cache line:
 *   0 = released, 1 = running.
 *
 * Args:
 *   args[0] = inout ChipTensor*
 *   args[1] = scalar status base cache line
 *   args[2] = scalar status count
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#ifdef PTO_CPUSTUB_HPP
#include <chrono>
#include <thread>
#endif

#include "common/platform_config.h"
#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

#include "intrinsic.h"

static constexpr int32_t FLOATS_PER_CACHE_LINE = 16;
static constexpr float BLOCKER_STARTED = 1.0F;
static constexpr uint64_t BLOCKER_HOLD_TICKS = PLATFORM_PROF_SYS_CNT_FREQ;

#ifdef PTO_CPUSTUB_HPP
#ifndef dcci
#define dcci(...) \
    do {          \
    } while (0)
#endif
#define SPIN_WAIT_HINT() std::this_thread::yield()

static uint64_t blocker_now_ticks() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
}
#else
#define SPIN_WAIT_HINT() ((void)0)

static __aicore__ uint64_t blocker_now_ticks() { return get_sys_cnt(); }
#endif
#ifndef SINGLE_CACHE_LINE
#define SINGLE_CACHE_LINE 0
#endif
#ifndef CACHELINE_OUT
#define CACHELINE_OUT 0
#endif

static __aicore__ void publish_value(__gm__ float *value, float state) {
    *value = state;
    dcci(value, SINGLE_CACHE_LINE, CACHELINE_OUT);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ ChipTensor *out_tensor = reinterpret_cast<__gm__ ChipTensor *>(args[0]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;
    int32_t status_base_cl = static_cast<int32_t>(args[1]);
    int32_t status_count = static_cast<int32_t>(args[2]);
    int32_t block_idx = get_block_idx(args);
    if (block_idx < 0 || block_idx >= status_count) {
        return;
    }

    __gm__ float *status = &out[(status_base_cl + block_idx) * FLOATS_PER_CACHE_LINE];
    uint64_t wait_start = blocker_now_ticks();

    publish_value(status, BLOCKER_STARTED);

    while (blocker_now_ticks() - wait_start < BLOCKER_HOLD_TICKS) {
        SPIN_WAIT_HINT();
    }

    publish_value(status, 0.0F);
}
