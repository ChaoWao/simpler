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

#pragma once

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

#include "intrinsic.h"

#ifdef PTO_CPUSTUB_HPP
#define dcci(...) \
    do {          \
    } while (0)
#endif
#ifndef SINGLE_CACHE_LINE
#define SINGLE_CACHE_LINE 0
#endif
#ifndef CACHELINE_OUT
#define CACHELINE_OUT 0
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ ChipTensor *x_tensor = reinterpret_cast<__gm__ ChipTensor *>(args[0]);
    __gm__ ChipTensor *padding_tensor = reinterpret_cast<__gm__ ChipTensor *>(args[1]);
    __gm__ float *x = reinterpret_cast<__gm__ float *>(x_tensor->buffer.addr) + x_tensor->start_offset;
    __gm__ float *padding =
        reinterpret_cast<__gm__ float *>(padding_tensor->buffer.addr) + padding_tensor->start_offset;

    x[0] = 2.0f;
    padding[0] = 3.0f;
    dcci(&x[0], SINGLE_CACHE_LINE, CACHELINE_OUT);
    dcci(&padding[0], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
