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
#include "pto_dep_compute.h"
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

void run_lookup(PTO2TensorMap &tmap, const ChipTensor &tensor, TensorAccessKind access_kind, TestLookupResult &out) {
    tmap.lookup(tensor, access_kind, [&](PTO2TensorMapEntry &e, OverlapStatus s) -> bool {
        out.entries.push_back({&e, s});
        out.count++;
        return true;
    });
}

DepInputs dep_inputs(const CoreTaskArgs &args) {
    return DepInputs{
        args.tensor_count(),       args.tensor_data(), args.tag_data(), static_cast<int32_t>(args.explicit_dep_count()),
        args.explicit_deps_data(),
    };
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

TEST(HbgTensorMapLayoutTest, DefaultLayoutCapsSparseReaderBuckets) {
    DeviceArena arena;
    auto layout = PTO2TensorMap::reserve_layout_default(arena, 32);
    EXPECT_EQ(layout.num_buckets, PTO2_TENSORMAP_NUM_BUCKETS);
    EXPECT_EQ(layout.num_reader_buckets, PTO2_TENSORMAP_READER_NUM_BUCKETS);
}

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
    EXPECT_EQ(result.entries[0].entry->access_task_id, PTO2TaskId::make(0, 2));
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
    EXPECT_EQ(result.entries[0].entry->access_task_id, PTO2TaskId::make(0, WINDOW_SIZE));
}

TEST_F(HbgTensorMapTest, ReaderAndWriterIndexesImplementRawWarAndOutputExistingOrdering) {
    ChipTensor tensor = make_test_tensor(0x4000, 256);
    const PTO2TaskId writer0 = PTO2TaskId::make(0, 1);
    const PTO2TaskId reader0 = PTO2TaskId::make(0, 2);
    const PTO2TaskId writer1 = PTO2TaskId::make(0, 3);
    const PTO2TaskId reader1 = PTO2TaskId::make(0, 4);
    const PTO2TaskId output_existing = PTO2TaskId::make(0, 5);
    tmap.insert(tensor, writer0, TensorAccessKind::WRITER);

    CoreTaskArgs untracked_input_args;
    untracked_input_args.add_input(tensor);
    std::vector<PTO2TaskId> fanin;
    auto emit = [&](PTO2TaskId id) {
        fanin.push_back(id);
        return true;
    };
    ASSERT_TRUE(compute_task_fanin(dep_inputs(untracked_input_args), tmap, false, emit));
    ASSERT_EQ(fanin, std::vector<PTO2TaskId>({writer0}));
    EXPECT_EQ(count_registrable_accesses(dep_inputs(untracked_input_args), false), 0);
    register_task_accesses(dep_inputs(untracked_input_args), reader0, tmap, false);

    TestLookupResult readers;
    run_lookup(tmap, tensor, TensorAccessKind::READER, readers);
    EXPECT_EQ(readers.count, 0);

    CoreTaskArgs input_args;
    input_args.add_tracked_input(tensor);
    fanin.clear();
    ASSERT_TRUE(compute_task_fanin(dep_inputs(input_args), tmap, false, emit));
    ASSERT_EQ(fanin, std::vector<PTO2TaskId>({writer0}));
    EXPECT_EQ(count_registrable_accesses(dep_inputs(input_args), false), 1);
    register_task_accesses(dep_inputs(input_args), reader0, tmap, false);

    readers = {};
    run_lookup(tmap, tensor, TensorAccessKind::READER, readers);
    ASSERT_EQ(readers.count, 1);
    EXPECT_EQ(readers.entries[0].entry->access_task_id, reader0);
    TestLookupResult writers;
    run_lookup(tmap, tensor, TensorAccessKind::WRITER, writers);
    ASSERT_EQ(writers.count, 1);
    EXPECT_EQ(writers.entries[0].entry->access_task_id, writer0);

    CoreTaskArgs inout_args;
    inout_args.add_inout(tensor);
    fanin.clear();
    ASSERT_TRUE(compute_task_fanin(dep_inputs(inout_args), tmap, false, emit));
    ASSERT_EQ(fanin, std::vector<PTO2TaskId>({writer0, reader0}));
    register_task_accesses(dep_inputs(inout_args), writer1, tmap, false);

    CoreTaskArgs second_input_args;
    second_input_args.add_tracked_input(tensor);
    fanin.clear();
    ASSERT_TRUE(compute_task_fanin(dep_inputs(second_input_args), tmap, false, emit));
    ASSERT_EQ(fanin, std::vector<PTO2TaskId>({writer1}));
    register_task_accesses(dep_inputs(second_input_args), reader1, tmap, false);

    CoreTaskArgs output_args;
    output_args.add_output(tensor);
    fanin.clear();
    ASSERT_TRUE(compute_task_fanin(dep_inputs(output_args), tmap, false, emit));
    ASSERT_EQ(fanin, std::vector<PTO2TaskId>({reader1}));
    register_task_accesses(dep_inputs(output_args), output_existing, tmap, false);

    readers = {};
    run_lookup(tmap, tensor, TensorAccessKind::READER, readers);
    EXPECT_EQ(readers.count, 0);
    writers = {};
    run_lookup(tmap, tensor, TensorAccessKind::WRITER, writers);
    ASSERT_EQ(writers.count, 2);
    EXPECT_EQ(writers.entries[0].entry->access_task_id, output_existing);
    EXPECT_EQ(writers.entries[1].entry->access_task_id, writer1);
}

TEST_F(HbgTensorMapTest, TrackedAccessesSkipManualScopeAndManualTensor) {
    ChipTensor tensor = make_test_tensor(0x5000, 64);
    tmap.insert(tensor, PTO2TaskId::make(0, 1));
    int emitted = 0;
    auto emit = [&](PTO2TaskId) {
        emitted++;
        return true;
    };

    CoreTaskArgs manual_scope_args;
    manual_scope_args.add_tracked_input(tensor);
    ASSERT_TRUE(compute_task_fanin(dep_inputs(manual_scope_args), tmap, true, emit));
    register_task_accesses(dep_inputs(manual_scope_args), PTO2TaskId::make(0, 2), tmap, true);

    tensor.manual_dep = true;
    CoreTaskArgs manual_tensor_args;
    manual_tensor_args.add_tracked_input(tensor);
    ASSERT_TRUE(compute_task_fanin(dep_inputs(manual_tensor_args), tmap, false, emit));
    register_task_accesses(dep_inputs(manual_tensor_args), PTO2TaskId::make(0, 3), tmap, false);

    EXPECT_EQ(emitted, 0);
    EXPECT_EQ(tmap.valid_count(), 1);
    EXPECT_EQ(tmap.current_readers(), 0);
    EXPECT_EQ(tmap.current_writers(), 1);
}

TEST_F(HbgTensorMapTest, NoDepRetainsCreatorWithoutTensorMapAccess) {
    ChipTensor tensor = make_test_tensor(0x5100, 64);
    const PTO2TaskId owner = PTO2TaskId::make(0, 7);
    tensor.owner_task_id = owner;
    tmap.insert(tensor, PTO2TaskId::make(0, 1));
    std::vector<PTO2TaskId> fanin;
    auto emit = [&](PTO2TaskId id) {
        fanin.push_back(id);
        return true;
    };

    CoreTaskArgs no_dep_args;
    no_dep_args.add_no_dep(tensor);
    ASSERT_TRUE(compute_task_fanin(dep_inputs(no_dep_args), tmap, false, emit));
    register_task_accesses(dep_inputs(no_dep_args), PTO2TaskId::make(0, 4), tmap, false);

    EXPECT_EQ(fanin, std::vector<PTO2TaskId>({owner}));
    EXPECT_EQ(tmap.valid_count(), 1);
    EXPECT_EQ(tmap.current_readers(), 0);
    EXPECT_EQ(tmap.current_writers(), 1);
}

}  // namespace
