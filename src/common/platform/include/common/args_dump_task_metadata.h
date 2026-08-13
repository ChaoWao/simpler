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
#include <type_traits>

#include "arg_direction.h"

using ArgsDumpArgMask = uint64_t;

// Bitmask stored with each task when orchestration selects specific
// tensor/scalar arguments for dump. Bit N corresponds to the payload argument
// index: tensors first, then scalars.
constexpr ArgsDumpArgMask ARGS_DUMP_ARG_MASK_NONE = 0;
constexpr uint32_t ARGS_DUMP_ARG_MASK_BITS = 64;
constexpr uint8_t ARGS_DUMP_RECORD_FLAG_ARG_INDEX_AMBIGUOUS = 1u << 0;

struct ArgsDumpTaskMetadata {
    ArgsDumpArgMask dump_arg_mask{ARGS_DUMP_ARG_MASK_NONE};
    ArgsDumpArgMask dump_arg_flags{ARGS_DUMP_ARG_MASK_NONE};
    uint8_t scalar_dtypes[CORE_MAX_SCALAR_ARGS]{};
};

static_assert(std::is_trivially_copyable_v<ArgsDumpTaskMetadata>);
static_assert(std::is_standard_layout_v<ArgsDumpTaskMetadata>);
static_assert(sizeof(ArgsDumpTaskMetadata) == 32);
