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
 * Memory Allocator Implementation (Simulation)
 *
 * Uses standard malloc/free to simulate device memory operations.
 */

#include "host/memory_allocator.h"

#include <cstdlib>
#include "common/unified_log.h"

MemoryAllocator::~MemoryAllocator() { finalize(); }

void *MemoryAllocator::alloc(size_t size) {
    void *ptr = std::malloc(size);
    if (ptr == nullptr) {
        LOG_ERROR("malloc failed (size=%zu)", size);
        return nullptr;
    }

    std::scoped_lock<std::mutex> lk(mu_);
    ptr_size_map_[ptr] = size;
    committed_bytes_ += size;
    return ptr;
}

int MemoryAllocator::free(void *ptr) {
    if (ptr == nullptr) {
        return 0;
    }

    std::scoped_lock<std::mutex> lk(mu_);
    auto it = ptr_size_map_.find(ptr);
    if (it == ptr_size_map_.end()) {
        return 0;
    }

    committed_bytes_ -= it->second;
    std::free(ptr);
    ptr_size_map_.erase(it);
    return 0;
}

int MemoryAllocator::finalize() {
    std::scoped_lock<std::mutex> lk(mu_);
    for (const auto &kv : ptr_size_map_) {
        std::free(kv.first);
    }
    ptr_size_map_.clear();
    committed_bytes_ = 0;
    return 0;
}
