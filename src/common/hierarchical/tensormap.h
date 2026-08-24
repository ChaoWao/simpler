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
 * TensorMap — (RunId, TensorKey) → the live producers of a backing's byte ranges.
 *
 * A TensorKey names a whole backing: the canonical identity of a local buffer, scoped by the
 * owning NEXT_LEVEL worker id for device memory (identical device addresses recur across
 * children), or the exported (owner, buffer_id, generation, offset) of a remote one. Two views
 * of one backing therefore land on one key by construction, and the key alone cannot tell a
 * pair that conflicts from a pair that does not.
 *
 * `TensorFootprint` is what tells them apart. Each key holds every producer whose written view is
 * still the last word on some byte of the backing, so `x[0]` and `x[1]` coexist under one key and
 * neither resolves as the other's producer. This mirrors the L2 ChipTensorMap, which hashes by
 * base address only and walks the chain running the same overlap cascade — a bounding-range
 * reject, then a per-dimension hyper-rectangle test that separates two column blocks whose
 * bounding boxes interleave.
 *
 * Conservative in one direction only: a pair the cascade cannot model exactly counts as
 * overlapping, and a producer a later write only partly covers stays live. So an extra edge is
 * possible where the truth is subtler, and a real dependency is never dropped.
 *
 * Unlike the L2 ChipTensorMap, this implementation:
 *   - Uses std::unordered_map (no ring buffer entry pool)
 *   - Cleans up a task's entries when it is CONSUMED, skipping the views a
 *     newer same-run producer has taken over
 */

#pragma once

#include <mutex>
#include <unordered_map>
#include <vector>

#include "types.h"

class TensorMap {
public:
    // Append every live producer under `key` whose written view overlaps `view`. Views that do
    // not intersect are the disjoint slices of one backing, and are not dependencies.
    // `out` is appended to, not cleared: one task accumulates producers across all its args.
    void lookup_overlapping(RunId run_id, TensorKey key, const TensorFootprint &view, std::vector<TaskSlot> &out) const;

    // Register `producer` as the writer of `view` under `key`. Entries this write covers whole
    // are superseded and dropped; one that reaches outside it stays, because it is still the
    // last writer of the bytes this write does not reach.
    void insert(RunId run_id, TensorKey key, const TensorFootprint &view, TaskSlot producer);

    // Remove the entries under 'keys' still owned by 'owner'.
    // Called when a producer task transitions to CONSUMED. A view that a newer same-run
    // producer has since superseded is already gone; one it left standing belongs to nobody
    // else and is dropped here.
    void erase_task_outputs(RunId run_id, TaskSlot owner, const std::vector<TensorKey> &keys);

    // Number of (key, view) producer entries currently tracked. A key holding two disjoint
    // live writers counts twice.
    int32_t size() const;

private:
    struct RunTensorKey {
        RunId run_id{INVALID_RUN_ID};
        TensorKey tensor{};

        bool operator==(const RunTensorKey &other) const { return run_id == other.run_id && tensor == other.tensor; }
    };

    struct RunTensorKeyHash {
        size_t operator()(const RunTensorKey &key) const {
            size_t h = std::hash<RunId>{}(key.run_id);
            size_t tensor_hash = TensorKeyHash{}(key.tensor);
            h ^= tensor_hash + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    struct Entry {
        TensorFootprint view{};
        TaskSlot producer{INVALID_SLOT};
    };

    mutable std::mutex mu_;
    std::unordered_map<RunTensorKey, std::vector<Entry>, RunTensorKeyHash> map_;
};
