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
struct HostTensorAccessor::Impl {
    const HostApi *api;
    std::vector<HostTensorRegion> regions;
    std::vector<void *> mappings;
};

// The region serving the whole of [dev_addr, dev_addr + bytes), or nullptr.
// `*offset` is the span's distance from that region's base.
const HostTensorRegion *
find_region(const std::vector<HostTensorRegion> &regions, uint64_t dev_addr, uint64_t bytes, uint64_t *offset) {
    for (const HostTensorRegion &region : regions) {
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

HostTensorAccessor::HostTensorAccessor(const HostApi *api) :
    impl_(new Impl{api, {}, {}}) {}

HostTensorAccessor::~HostTensorAccessor() {
    close();
    delete impl_;
}

bool HostTensorAccessor::add(uint64_t dev_base, uint64_t size, void *fallback_host_view) {
    if (impl_->api == nullptr || dev_base == 0 || size == 0) {
        return false;
    }
    // Both push_backs below run after the registration succeeds, so reserve
    // first: a reallocation that threw there would leak the mapping.
    impl_->regions.reserve(impl_->regions.size() + 1);
    impl_->mappings.reserve(impl_->mappings.size() + 1);
    void *host_view = impl_->api->register_device_memory_to_host(reinterpret_cast<void *>(dev_base), size);
    if (host_view != nullptr) {
        impl_->mappings.push_back(reinterpret_cast<void *>(dev_base));
    } else {
        host_view = fallback_host_view;
    }
    if (host_view == nullptr) {
        return false;
    }
    impl_->regions.push_back(
        {dev_base, size, static_cast<unsigned char *>(host_view), reinterpret_cast<uintptr_t>(host_view) != dev_base}
    );
    return true;
}

bool HostTensorAccessor::read(uint64_t dev_addr, void *dst, uint64_t bytes) const {
    uint64_t offset = 0;
    const HostTensorRegion *region = find_region(impl_->regions, dev_addr, bytes, &offset);
    if (region == nullptr) {
        return false;
    }
    memcpy(dst, region->host_view + offset, bytes);
    return true;
}

bool HostTensorAccessor::write(uint64_t dev_addr, const void *src, uint64_t bytes) const {
    uint64_t offset = 0;
    const HostTensorRegion *region = find_region(impl_->regions, dev_addr, bytes, &offset);
    if (region == nullptr) {
        return false;
    }
    unsigned char *dst = region->host_view + offset;
    memcpy(dst, src, bytes);
    if (!region->mirrored) {
        return true;
    }
    return impl_->api->copy_to_device(reinterpret_cast<void *>(dev_addr), dst, static_cast<size_t>(bytes)) == 0;
}

void HostTensorAccessor::close() noexcept {
    for (void *dev_ptr : impl_->mappings) {
        impl_->api->unregister_device_memory_from_host(dev_ptr);
    }
    impl_->mappings.clear();
    impl_->regions.clear();
}

bool host_tensor_read(HostTensorAccessor *accessor, uint64_t dev_addr, void *dst, uint64_t bytes) {
    return accessor != nullptr && accessor->read(dev_addr, dst, bytes);
}

bool host_tensor_write(HostTensorAccessor *accessor, uint64_t dev_addr, const void *src, uint64_t bytes) {
    return accessor != nullptr && accessor->write(dev_addr, src, bytes);
}
