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
 * Host-side tensor access for the host orchestrator. See
 * runtime/host_tensor_access.h for the contract.
 */

#include "host_tensor_access.h"

#include <string.h>

#include <vector>

#include "common/host_api.h"

namespace {

struct HostTensorRegion {
    uint64_t dev_base;
    uint64_t size;
    unsigned char *host_view;
    // The host view is a copy of the device bytes rather than the device bytes
    // themselves, so a write reaches the device only once pushed back.
    bool mirrored;
};

// One entry per tensor staged for the run being orchestrated. A run stages a
// handful of tensors and orchestration reads are cold-path, so a linear scan
// costs less than the map that would replace it.
std::vector<HostTensorRegion> g_regions;

const HostApi *g_host_api = nullptr;

// The region serving the whole of [dev_addr, dev_addr + bytes), or nullptr.
// `*offset` is the span's distance from that region's base.
const HostTensorRegion *find_region(uint64_t dev_addr, uint64_t bytes, uint64_t *offset) {
    for (const HostTensorRegion &region : g_regions) {
        if (dev_addr < region.dev_base) {
            continue;
        }
        uint64_t off = dev_addr - region.dev_base;
        if (off > region.size || bytes > region.size - off) {
            continue;
        }
        *offset = off;
        return &region;
    }
    return nullptr;
}

}  // namespace

bool host_tensor_read(uint64_t dev_addr, void *dst, uint64_t bytes) {
    uint64_t offset = 0;
    const HostTensorRegion *region = find_region(dev_addr, bytes, &offset);
    if (region == nullptr) {
        return false;
    }
    memcpy(dst, region->host_view + offset, bytes);
    return true;
}

bool host_tensor_write(uint64_t dev_addr, const void *src, uint64_t bytes) {
    uint64_t offset = 0;
    const HostTensorRegion *region = find_region(dev_addr, bytes, &offset);
    if (region == nullptr) {
        return false;
    }
    unsigned char *dst = region->host_view + offset;
    memcpy(dst, src, bytes);
    if (!region->mirrored) {
        return true;
    }
    if (g_host_api == nullptr) {
        return false;
    }
    return g_host_api->copy_to_device(reinterpret_cast<void *>(dev_addr), dst, static_cast<size_t>(bytes)) == 0;
}

void host_tensor_access_reset(const HostApi *api) {
    g_regions.clear();
    g_host_api = api;
}

bool host_tensor_access_add(uint64_t dev_base, uint64_t size, void *host_view) {
    if (host_view == nullptr || size == 0) {
        return false;
    }
    HostTensorRegion region;
    region.dev_base = dev_base;
    region.size = size;
    region.host_view = static_cast<unsigned char *>(host_view);
    region.mirrored = reinterpret_cast<uintptr_t>(host_view) != static_cast<uintptr_t>(dev_base);
    g_regions.push_back(region);
    return true;
}
