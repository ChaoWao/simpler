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
 * Where the `tmr` runtime keeps the arguments its orchestration entry was called
 * with.
 *
 * This header is runtime-internal, and separate from tensor.h for that reason: it
 * reaches the L3+ argument surface in src/common/task_interface/task_args.h, and
 * through it the address-free wire `Tensor` at global scope in buffer.h. A kernel
 * translation unit needs this runtime's `Tensor` but none of that, and gets it from
 * tensor.h, which stays clear of both.
 */

#pragma once

#include "task_interface/arg_direction.h"
#include "task_interface/task_args.h"
#include "tensor.h"

namespace simpler::tmr {

// `Runtime::set_orch_args` adopts the boundary ChipStorageTaskArgs into this once,
// on the host, before orchestration runs; from there inward nothing holds a bare
// ChipTensor.
using EntryArgsStorage = TaskArgsTpl<Tensor, uint64_t, CHIP_MAX_TENSOR_ARGS, CHIP_MAX_SCALAR_ARGS>;

}  // namespace simpler::tmr
