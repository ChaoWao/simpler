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

#include <stddef.h>
#include <stdint.h>

enum class WorkerChipRegionAccessProfile : uint32_t {
    INVALID = 0,
    ONBOARD_VMM = 1,
    SIM_POSIX_SHM = 2,
};

struct WorkerHostRegionMappingHandle {
    uint64_t id{0};
    WorkerChipRegionAccessProfile profile{WorkerChipRegionAccessProfile::INVALID};
    uint64_t mapping_bytes{0};
};

struct WorkerChipRegionCreateRequest {
    uint64_t magic_version;
    uint64_t request_bytes;
    uint64_t payload_bytes;
    uint64_t counter_bytes;
};

struct WorkerChipRegionCreateReply {
    uint64_t desc[6];
    uint32_t access_profile;
    uint32_t reserved;
    int32_t device_id;
    uint8_t backing_shm[32];
    uint64_t mapping_bytes;
    uint64_t shareable_handle;
};

static constexpr size_t WORKER_CHIP_REGION_CREATE_REQUEST_BYTES = sizeof(WorkerChipRegionCreateRequest);
static constexpr size_t WORKER_CHIP_REGION_CREATE_REPLY_BYTES = sizeof(WorkerChipRegionCreateReply);
