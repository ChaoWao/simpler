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

#include "tensormap.h"

void TensorMap::lookup_overlapping(
    RunId run_id, TensorKey key, const TensorFootprint &view, std::vector<TaskSlot> &out
) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = map_.find(RunTensorKey{run_id, key});
    if (it == map_.end()) return;
    for (const Entry &entry : it->second) {
        if (tensor_overlap(view, entry.view) != TensorOverlap::NONE) out.push_back(entry.producer);
    }
}

void TensorMap::insert(RunId run_id, TensorKey key, const TensorFootprint &view, TaskSlot producer) {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<Entry> &entries = map_[RunTensorKey{run_id, key}];
    // A write leaves an earlier producer standing only for the bytes it does not reach, so an
    // entry it covers whole is dead and one it only reaches into is not. Dropping the latter too
    // would lose the dependency a consumer of those outside bytes must still take.
    size_t kept = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (tensor_overlap(view, entries[i].view) == TensorOverlap::COVERED) continue;
        entries[kept++] = entries[i];
    }
    entries.resize(kept);
    entries.push_back(Entry{view, producer});
}

void TensorMap::erase_task_outputs(RunId run_id, TaskSlot owner, const std::vector<TensorKey> &keys) {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto &key : keys) {
        auto it = map_.find(RunTensorKey{run_id, key});
        if (it == map_.end()) continue;
        std::vector<Entry> &entries = it->second;
        size_t kept = 0;
        for (size_t i = 0; i < entries.size(); ++i) {
            if (entries[i].producer == owner) continue;
            entries[kept++] = entries[i];
        }
        entries.resize(kept);
        if (entries.empty()) map_.erase(it);
    }
}

int32_t TensorMap::size() const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t total = 0;
    for (const auto &bucket : map_) {
        total += bucket.second.size();
    }
    return static_cast<int32_t>(total);
}
