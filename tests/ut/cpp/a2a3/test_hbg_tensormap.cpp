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
 * cleanup_retired tests for the host_build_graph copy of PTO2TensorMap.
 *
 * This is a distinct type from the tensormap_and_ringbuffer PTO2TensorMap
 * covered by a2a3/test_tensormap.cpp: single-ring, no entry epochs. Only the
 * per-task entry reclamation path is covered here — the hash / overlap /
 * lazy-invalidation surface is shared logic already exercised by that suite.
 */

#include <gtest/gtest.h>

#include <vector>

#include "utils/device_arena.h"
#include "pto_tensormap.h"

namespace {

struct TestLookupResult {
    struct Entry {
        PTO2TensorMapEntry *entry;
        OverlapStatus overlap_status;
    };
    std::vector<Entry> entries;
    int count = 0;
};

void run_lookup(PTO2TensorMap &tmap, const ChipTensor &tensor, TestLookupResult &out) {
    tmap.lookup(tensor, [&](PTO2TensorMapEntry &e, OverlapStatus s) -> bool {
        out.entries.push_back({&e, s});
        out.count++;
        return true;
    });
}

ChipTensor make_test_tensor(uint64_t addr, uint32_t shape0) {
    uint32_t shapes[MAX_TENSOR_DIMS] = {shape0};
    return make_tensor_external(reinterpret_cast<void *>(addr), shapes, 1, DataType::FLOAT32, false, 0);
}

class HbgTensorMapTest : public ::testing::Test {
protected:
    static constexpr int32_t NUM_BUCKETS = 16;
    static constexpr int32_t POOL_SIZE = 64;
    static constexpr int32_t WINDOW_SIZE = 32;

    PTO2TensorMap tmap{};
    DeviceArena arena;

    void SetUp() override {
        auto layout = PTO2TensorMap::reserve_layout(arena, NUM_BUCKETS, POOL_SIZE, WINDOW_SIZE);
        ASSERT_NE(arena.commit(), nullptr);
        ASSERT_TRUE(tmap.init_data_from_layout(layout, arena));
        tmap.wire_arena_pointers(layout, arena);
    }

    void TearDown() override {
        tmap.destroy();
        arena.release();
    }
};

TEST_F(HbgTensorMapTest, CleanupRetiredRemovesEntriesForRetiredTasks) {
    ChipTensor t = make_test_tensor(0x1000, 256);
    tmap.insert(t, PTO2TaskId::make(0, 0));
    tmap.insert(t, PTO2TaskId::make(0, 1));
    tmap.insert(t, PTO2TaskId::make(0, 2));
    EXPECT_EQ(tmap.valid_count(), 3);

    tmap.cleanup_retired(0, 2);

    EXPECT_EQ(tmap.valid_count(), 1);
    TestLookupResult result;
    run_lookup(tmap, t, result);
    ASSERT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].entry->producer_task_id, PTO2TaskId::make(0, 2));
}

TEST_F(HbgTensorMapTest, CleanupRetiredFreesEveryOutputOfOneTask) {
    ChipTensor t1 = make_test_tensor(0x1000, 256);
    ChipTensor t2 = make_test_tensor(0x2000, 128);
    PTO2TaskId tid = PTO2TaskId::make(0, 5);

    tmap.insert(t1, tid);
    tmap.insert(t2, tid);
    EXPECT_EQ(tmap.valid_count(), 2);

    tmap.cleanup_retired(5, 6);
    EXPECT_EQ(tmap.valid_count(), 0);
    EXPECT_EQ(tmap.free_num, 2);
}

// A later task that reuses a slot (local_id + WINDOW_SIZE) before cleanup has
// run on the earlier task chains its entries under the same task_entry_head.
// cleanup_retired retiring only the earlier task must free that earlier task's
// entries alone and leave the still-live (later) task's entries intact.
TEST_F(HbgTensorMapTest, CleanupRetiredSparesLaterTaskReusingSlot) {
    ChipTensor t = make_test_tensor(0x1000, 256);
    // Task 0 and task 0 + WINDOW_SIZE share slot 0 (local_id & (WINDOW_SIZE-1)).
    tmap.insert(t, PTO2TaskId::make(0, 0));
    tmap.insert(t, PTO2TaskId::make(0, WINDOW_SIZE));
    ASSERT_EQ(tmap.valid_count(), 2);

    // Retire only task 0.
    tmap.cleanup_retired(0, 1);

    // Only task 0's entry is freed; task WINDOW_SIZE's entry survives.
    EXPECT_EQ(tmap.valid_count(), 1);
    TestLookupResult result;
    run_lookup(tmap, t, result);
    ASSERT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].entry->producer_task_id, PTO2TaskId::make(0, WINDOW_SIZE));
}

}  // namespace
