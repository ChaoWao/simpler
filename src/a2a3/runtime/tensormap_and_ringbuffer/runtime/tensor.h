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
 * Both tensor types this runtime deals in: the boundary `ChipTensor` an argument
 * arrives as, and `simpler::tmr::Tensor`, the form everything inside the runtime —
 * orchestration included — actually works with.
 *
 * This sits first on the runtime include path, so a `#include "tensor.h"` from
 * runtime, orchestration or kernel sources lands here.
 */

#pragma once

#include "task_interface/tensor.h"
#include "tensormap_and_ringbuffer/tensor.h"

// The tensor type of whichever runtime this translation unit is being built for.
// A kernel reads a payload element and does not care which orchestrator produced
// it, so kernels name this rather than picking a runtime — several are compiled
// under both.
using TaskTensor = simpler::tmr::Tensor;
