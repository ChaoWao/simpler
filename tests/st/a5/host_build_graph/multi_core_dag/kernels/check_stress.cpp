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

#include "intrinsic.h"
#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ TaskTensor *tensor = reinterpret_cast<__gm__ TaskTensor *>(args[0]);
    __gm__ int64_t *state = reinterpret_cast<__gm__ int64_t *>(tensor->buffer.addr) + tensor->start_offset;
    const int64_t task_id = args[1];
    const int64_t graph_case = args[2];
    const int64_t fanin_count = args[3];
    bool ready = true;
    if (graph_case == 1 && task_id >= 32) {
        for (int64_t producer = 0; producer < 32; ++producer)
            dcci(&state[(producer + 1) * 8], cache_line_t::SINGLE_CACHE_LINE);
        dsb((mem_dsb_t)0);
        for (int64_t producer = 0; producer < 32; ++producer) {
            ready = ready && state[(producer + 1) * 8] == producer + 1;
        }
    } else {
        const int64_t checked = fanin_count < 4 ? fanin_count : 4;
        for (int64_t i = 0; i < checked; ++i) {
            const int64_t producer = args[4 + i];
            if (producer >= 0) dcci(&state[(producer + 1) * 8], cache_line_t::SINGLE_CACHE_LINE);
        }
        dsb((mem_dsb_t)0);
        for (int64_t i = 0; i < checked; ++i) {
            const int64_t producer = args[4 + i];
            ready = ready && producer >= 0 && state[(producer + 1) * 8] == producer + 1;
        }
    }
    state[(task_id + 1) * 8] = ready ? task_id + 1 : -(task_id + 1);
    dcci(&state[(task_id + 1) * 8], cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb((mem_dsb_t)0);
}
