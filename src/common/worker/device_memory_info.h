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

#include <stdint.h>

typedef struct DeviceMemoryInfo {
    uint64_t free_bytes;
    uint64_t total_bytes;
} DeviceMemoryInfo;

#ifdef __cplusplus
#include <type_traits>

static_assert(std::is_trivially_copyable_v<DeviceMemoryInfo> && std::is_standard_layout_v<DeviceMemoryInfo>);
static_assert(sizeof(DeviceMemoryInfo) == 2 * sizeof(uint64_t));
#endif
