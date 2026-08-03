/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <cstdint>

#include "tensor.h"

#ifdef __CPU_SIM
#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) { (void)args; }

#else

#include "../kernel/fai_body.hpp"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
  GM_ADDR metadata = tensor_data<uint8_t>(args, 6);
  acquire_qwen_fai_metadata(metadata);
  __gm__ const FAInferTilingData *tiling =
      reinterpret_cast<__gm__ const FAInferTilingData *>(metadata);
  if (tiling->needCoreNum != 0) {
    uint64_t raw_barrier = reinterpret_cast<uint64_t>(
        metadata + qwen_fai_metadata::kBarrierAlignmentOffset);
    uint64_t aligned_barrier =
        (raw_barrier + qwen_fai_metadata::kBarrierAlignmentBytes - 1) &
        ~(static_cast<uint64_t>(qwen_fai_metadata::kBarrierAlignmentBytes) - 1);
    run_qwen_fai<true, true>(
        args, reinterpret_cast<__gm__ int32_t *>(aligned_barrier));
  } else {
    run_qwen_fai<false, true>(args);
  }
}

#endif
