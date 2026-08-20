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
    // Null on a GM-heap region until its first fill, and after one whose mapping
    // attempt failed. That region is write-only, so `fill` can stage the bytes
    // elsewhere and push them; every other region is read back through this view
    // and is mapped when it is registered.
    unsigned char *host_view;
    // The host view is a copy of the device bytes rather than the device bytes
    // themselves, so a write reaches the device only once pushed back.
    bool mirrored;
    // Whether a mapping has been attempted for this region. A GM-heap region is
    // registered without one — mapping the whole heap costs a `halHostRegister`
    // over hundreds of MB (2 GiB on the dsv4 case) that a run setting no initial
    // value would never use — so the attempt is deferred to the first fill and,
    // either way, made once.
    bool map_attempted;
    HostTensorRegionKind kind;
};

// Bytes of the buffer `fill` stages a viewless region's pattern in. One period
// of the pattern is at most 8 bytes, so this holds thousands of repeats and the
// same buffer serves every chunk of every fill in the run.
constexpr uint64_t kFillStagingBytes = 64 * 1024;

// One entry per tensor staged for the run being orchestrated, plus at most one
// for the GM heap. A run stages a handful of tensors and orchestration reads
// are cold-path, so a linear scan costs less than the map that would replace it.
struct HostTensorAccessor::Impl {
    const HostApi *api;
    std::vector<HostTensorRegion> regions;
    std::vector<void *> mappings;
    // Bytes covered by `mappings`, i.e. excluding regions serving a fallback view.
    uint64_t mapped_bytes;
    // Allocated on the first fill of a viewless region, and only then: a
    // platform that maps the heap never needs it.
    std::vector<unsigned char> fill_staging;
};

// The `kind` region serving the whole of [dev_addr, dev_addr + bytes), or
// nullptr. `*offset` is the span's distance from that region's base. Non-const
// because a GM-heap region binds its host view on the first fill that needs it.
HostTensorRegion *find_region(
    std::vector<HostTensorRegion> &regions, HostTensorRegionKind kind, uint64_t dev_addr, uint64_t bytes,
    uint64_t *offset
) {
    for (HostTensorRegion &region : regions) {
        if (region.kind != kind || dev_addr < region.dev_base) {
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
    impl_(new Impl{api, {}, {}, 0, {}}) {}

HostTensorAccessor::~HostTensorAccessor() {
    close();
    delete impl_;
}

bool HostTensorAccessor::add(uint64_t dev_base, uint64_t size, void *fallback_host_view) {
    return add_region(dev_base, size, fallback_host_view, HostTensorRegionKind::StagedTensor);
}

bool HostTensorAccessor::add_heap(uint64_t dev_base, uint64_t size) {
    return add_region(dev_base, size, nullptr, HostTensorRegionKind::GmHeap);
}

bool HostTensorAccessor::add_region(
    uint64_t dev_base, uint64_t size, void *fallback_host_view, HostTensorRegionKind kind
) {
    if (impl_->api == nullptr || dev_base == 0 || size == 0) {
        return false;
    }
    // Both push_backs below run after the registration succeeds, so reserve
    // first: a reallocation that threw there would leak the mapping.
    impl_->regions.reserve(impl_->regions.size() + 1);
    impl_->mappings.reserve(impl_->mappings.size() + 1);
    // A staged tensor is read through its view, so it is mapped now and must end
    // up with one. The GM heap is only ever written, by `fill`, so its mapping
    // waits for a fill to need it and the common run — which sets no initial
    // value — registers the region without touching the heap at all.
    if (kind == HostTensorRegionKind::GmHeap) {
        impl_->regions.push_back({dev_base, size, nullptr, false, false, kind});
        return true;
    }
    void *host_view = impl_->api->register_device_memory_to_host(reinterpret_cast<void *>(dev_base), size);
    if (host_view != nullptr) {
        impl_->mappings.push_back(reinterpret_cast<void *>(dev_base));
        impl_->mapped_bytes += size;
    } else {
        host_view = fallback_host_view;
    }
    if (host_view == nullptr) {
        return false;
    }
    const bool mirrored = reinterpret_cast<uintptr_t>(host_view) != dev_base;
    impl_->regions.push_back({dev_base, size, static_cast<unsigned char *>(host_view), mirrored, true, kind});
    return true;
}

// Map `region` if that has not been tried yet, so a heap that no fill reaches is
// never mapped. Leaves `host_view` null when the platform has no host-map path,
// which is the caller's cue to stage and push instead.
void HostTensorAccessor::ensure_region_mapped(HostTensorRegion *region) const {
    if (region->map_attempted) {
        return;
    }
    region->map_attempted = true;
    impl_->mappings.reserve(impl_->mappings.size() + 1);
    void *host_view =
        impl_->api->register_device_memory_to_host(reinterpret_cast<void *>(region->dev_base), region->size);
    if (host_view == nullptr) {
        return;
    }
    impl_->mappings.push_back(reinterpret_cast<void *>(region->dev_base));
    impl_->mapped_bytes += region->size;
    region->host_view = static_cast<unsigned char *>(host_view);
    region->mirrored = reinterpret_cast<uintptr_t>(host_view) != region->dev_base;
}

bool HostTensorAccessor::read(uint64_t dev_addr, void *dst, uint64_t bytes) const {
    uint64_t offset = 0;
    const HostTensorRegion *region =
        find_region(impl_->regions, HostTensorRegionKind::StagedTensor, dev_addr, bytes, &offset);
    if (region == nullptr) {
        return false;
    }
    memcpy(dst, region->host_view + offset, bytes);
    return true;
}

bool HostTensorAccessor::write(uint64_t dev_addr, const void *src, uint64_t bytes) const {
    uint64_t offset = 0;
    const HostTensorRegion *region =
        find_region(impl_->regions, HostTensorRegionKind::StagedTensor, dev_addr, bytes, &offset);
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

// Fill a span of a region that has no host view, by staging one buffer of the
// repeating pattern and pushing it across the span.
//
// The chunk is a whole number of elements, so every push starts on a pattern
// boundary and the staged bytes can be reused verbatim. A span that is not a
// whole number of elements ends on a partial one, which is what a buffer whose
// size is not a multiple of its element width should get.
bool HostTensorAccessor::fill_staged(uint64_t dev_addr, uint64_t bytes, uint64_t value, uint64_t elem_size) const {
    const uint64_t chunk = (kFillStagingBytes / elem_size) * elem_size;
    std::vector<unsigned char> &staging = impl_->fill_staging;
    // The buffer is reused across fills but the pattern is not: consecutive
    // fills differ in value and element width, so it is laid down per call.
    staging.resize(static_cast<size_t>(chunk));
    for (uint64_t written = 0; written < chunk; written += elem_size) {
        memcpy(staging.data() + written, &value, static_cast<size_t>(elem_size));
    }
    for (uint64_t pushed = 0; pushed < bytes;) {
        const uint64_t span = (bytes - pushed < chunk) ? bytes - pushed : chunk;
        if (impl_->api->copy_to_device(
                reinterpret_cast<void *>(dev_addr + pushed), staging.data(), static_cast<size_t>(span)
            ) != 0) {
            return false;
        }
        pushed += span;
    }
    return true;
}

HostTensorFillStatus
HostTensorAccessor::fill(uint64_t dev_addr, uint64_t bytes, uint64_t value, uint64_t elem_size) const {
    if (bytes == 0) {
        return HostTensorFillStatus::Ok;
    }
    if (elem_size == 0 || elem_size > sizeof(value)) {
        return HostTensorFillStatus::BadElementSize;
    }
    uint64_t offset = 0;
    HostTensorRegion *region = find_region(impl_->regions, HostTensorRegionKind::GmHeap, dev_addr, bytes, &offset);
    if (region == nullptr) {
        return HostTensorFillStatus::NoRegion;
    }
    ensure_region_mapped(region);
    if (region->host_view == nullptr) {
        // No host-map path on this platform: stage the pattern and push it.
        return fill_staged(dev_addr, bytes, value, elem_size) ? HostTensorFillStatus::Ok :
                                                                HostTensorFillStatus::PushFailed;
    }
    // Seed one element, then double the written prefix into the remainder: the
    // span is filled in log2(bytes / elem_size) memcpys rather than one per
    // element. A span that is not a whole number of elements ends on a partial
    // one, which is what a buffer whose size is not a multiple of its element
    // width should get.
    unsigned char *dst = region->host_view + offset;
    uint64_t seed = (elem_size < bytes) ? elem_size : bytes;
    memcpy(dst, &value, static_cast<size_t>(seed));
    uint64_t filled = seed;
    while (filled < bytes) {
        uint64_t copy_size = ((bytes - filled) < filled) ? (bytes - filled) : filled;
        memcpy(dst + filled, dst, static_cast<size_t>(copy_size));
        filled += copy_size;
    }
    if (!region->mirrored) {
        return HostTensorFillStatus::Ok;
    }
    return impl_->api->copy_to_device(reinterpret_cast<void *>(dev_addr), dst, static_cast<size_t>(bytes)) == 0 ?
               HostTensorFillStatus::Ok :
               HostTensorFillStatus::PushFailed;
}

size_t HostTensorAccessor::mapping_count() const noexcept { return impl_->mappings.size(); }

uint64_t HostTensorAccessor::mapped_bytes() const noexcept { return impl_->mapped_bytes; }

void HostTensorAccessor::close() noexcept {
    for (void *dev_ptr : impl_->mappings) {
        impl_->api->unregister_device_memory_from_host(dev_ptr);
    }
    impl_->mappings.clear();
    impl_->regions.clear();
    impl_->mapped_bytes = 0;
}

bool host_tensor_read(HostTensorAccessor *accessor, uint64_t dev_addr, void *dst, uint64_t bytes) {
    return accessor != nullptr && accessor->read(dev_addr, dst, bytes);
}

bool host_tensor_write(HostTensorAccessor *accessor, uint64_t dev_addr, const void *src, uint64_t bytes) {
    return accessor != nullptr && accessor->write(dev_addr, src, bytes);
}

HostTensorFillStatus
host_tensor_fill(HostTensorAccessor *accessor, uint64_t dev_addr, uint64_t bytes, uint64_t value, uint64_t elem_size) {
    if (accessor == nullptr) return HostTensorFillStatus::NoRegion;
    return accessor->fill(dev_addr, bytes, value, elem_size);
}

const char *host_tensor_fill_status_name(HostTensorFillStatus status) {
    switch (status) {
    case HostTensorFillStatus::Ok:
        return "ok";
    case HostTensorFillStatus::BadElementSize:
        return "unsupported element width";
    case HostTensorFillStatus::NoRegion:
        return "address is not in the GM heap";
    case HostTensorFillStatus::PushFailed:
        return "device copy failed";
    }
    return "unknown";
}
