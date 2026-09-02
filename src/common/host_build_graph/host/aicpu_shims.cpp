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
 * host_build_graph host-side definitions of two AICPU platform primitives
 *
 * The host orchestrator reaches both through headers that describe device
 * hardware, so each needs a definition that works where that hardware is absent.
 * The AICPU build has its own in platform/.../{device_time,platform_regs}.cpp and
 * never compiles this file; the two definitions are selected by which target
 * compiles which directory, so neither is weak.
 *
 * Both carry hidden visibility, which is load-bearing rather than tidiness. Each
 * stands in for hardware only inside this library, and the AICPU carries a
 * same-named definition whose value means something else — a real cycle counter,
 * a real register window. Keeping these off the dynamic symbol table is what makes
 * it impossible for another module to bind one of those names to a host wall-clock
 * or a dummy sink.
 */

#include <time.h>

#include <cstdint>

#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"

// Monotonic wall-clock in AICPU cycle units, so a cycle-denominated deadline
// evaluated during host orchestration fires at the wall-clock it was sized for.
// A constant 0 would instead make every such backstop a no-op and spin forever.
__attribute__((visibility("hidden"))) uint64_t get_sys_cnt_aicpu() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    // Scale sec and nsec separately (divisor is the constant 1e9): avoids a
    // div-by-zero when PLATFORM_PROF_SYS_CNT_FREQ >= 1 GHz and the truncation
    // error a `1e9 / FREQ` divisor would introduce for non-dividing frequencies.
    return static_cast<uint64_t>(ts.tv_sec) * PLATFORM_PROF_SYS_CNT_FREQ +
           static_cast<uint64_t>(ts.tv_nsec) * PLATFORM_PROF_SYS_CNT_FREQ / 1000000000ull;
}

// AICore register window. The orchestrator's route_ready_once path transitively
// ODR-uses the early-dispatch doorbell inline (scheduler.h ring_one_doorbell), but
// host graph-build gates no core, so the doorbell never fires and the address only
// has to be readable and writable.
__attribute__((visibility("hidden"))) volatile uint32_t *get_reg_ptr(uint64_t, RegId) {
    static volatile uint32_t sink = 0;
    return &sink;
}
