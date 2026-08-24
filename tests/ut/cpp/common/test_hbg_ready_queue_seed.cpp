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
 * The slot array of a host_build_graph ready queue is reserved past the uploaded
 * range, so it reaches the device holding whatever the pooled allocation last
 * had. seed_slots() is what makes it an empty queue.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "scheduler/scheduler.h"

namespace {

constexpr uint64_t CAPACITY = 8;

// A slots region under the test's control, filled with a pattern standing in for
// unseeded device memory. 0 is deliberate for one case: zeroed memory is the
// tempting assumption, and it is the one that silently behaves like a full queue.
class SlotsRegion {
public:
    explicit SlotsRegion(uint8_t fill) :
        storage_(CAPACITY * sizeof(PTO2ReadyQueueSlot)) {
        std::memset(storage_.data(), fill, storage_.size());
    }

    PTO2ReadyQueueSlot *slots() { return reinterpret_cast<PTO2ReadyQueueSlot *>(storage_.data()); }

private:
    std::vector<std::byte> storage_;
};

// PTO2ReadyQueue holds atomics and so is neither copyable nor movable: the caller
// owns the storage and this only fills it, exactly as the arena path does.
void make_queue(PTO2ReadyQueue *queue, SlotsRegion &region) {
    ready_queue_init_data_from_layout(queue, CAPACITY);
    queue->slots = region.slots();
}

// Distinct non-null slot_state values; the queue only moves the pointer around.
ChipTaskSlotState *fake_slot_state(size_t i) {
    static ChipTaskSlotState states[CAPACITY * 2];
    return &states[i];
}

}  // namespace

// The header alone does not make a usable queue: push claims slots[pos] only when
// its sequence already equals pos, so on zeroed memory position 0 happens to
// match and every later position reads a lower sequence, which is the full-queue
// signal. One push then succeeds and the rest report full.
TEST(HbgReadyQueueSeed, ZeroedSlotsReportFullAfterOnePush) {
    SlotsRegion region(0x00);
    PTO2ReadyQueue queue{};
    make_queue(&queue, region);

    EXPECT_TRUE(queue.push(fake_slot_state(0)));
    EXPECT_FALSE(queue.push(fake_slot_state(1)));
    EXPECT_FALSE(queue.push(fake_slot_state(2)));
}

TEST(HbgReadyQueueSeed, SeededSlotsAcceptCapacityPushes) {
    SlotsRegion region(0xAA);
    PTO2ReadyQueue queue{};
    make_queue(&queue, region);

    queue.seed_slots();

    for (uint64_t i = 0; i < CAPACITY; i++) {
        EXPECT_TRUE(queue.push(fake_slot_state(i))) << "push " << i;
    }
    // Capacity is a hard bound: a full queue rejects rather than overwrites.
    EXPECT_FALSE(queue.push(fake_slot_state(CAPACITY)));

    for (uint64_t i = 0; i < CAPACITY; i++) {
        EXPECT_EQ(queue.pop(), fake_slot_state(i)) << "pop " << i;
    }
    EXPECT_EQ(queue.pop(), nullptr);
}

// The ramp is a once-per-region cost, not a per-run one: pop releases a slot with
// the sequence the next lap's push expects, so a drained queue keeps accepting
// work across laps with no re-seed.
TEST(HbgReadyQueueSeed, DrainedQueueWrapsWithoutReseeding) {
    SlotsRegion region(0xAA);
    PTO2ReadyQueue queue{};
    make_queue(&queue, region);
    queue.seed_slots();

    for (uint64_t lap = 0; lap < 4; lap++) {
        for (uint64_t i = 0; i < CAPACITY; i++) {
            ASSERT_TRUE(queue.push(fake_slot_state(i))) << "lap " << lap << " push " << i;
        }
        for (uint64_t i = 0; i < CAPACITY; i++) {
            ASSERT_EQ(queue.pop(), fake_slot_state(i)) << "lap " << lap << " pop " << i;
        }
    }

    EXPECT_EQ(queue.max_occupancy.load(std::memory_order_relaxed), CAPACITY);
}

// Seeding is unconditional on attach, so it must also recover a region left
// mid-lap by an earlier run rather than assuming a drained predecessor.
TEST(HbgReadyQueueSeed, ReseedRecoversAPartiallyUsedRegion) {
    SlotsRegion region(0xAA);
    PTO2ReadyQueue queue{};
    make_queue(&queue, region);
    queue.seed_slots();
    ASSERT_TRUE(queue.push(fake_slot_state(0)));
    ASSERT_TRUE(queue.push(fake_slot_state(1)));

    // What a fresh bind does: the uploaded header returns to zeroed positions and
    // the device re-establishes the lap-0 ramp underneath it.
    ready_queue_init_data_from_layout(&queue, CAPACITY);
    queue.seed_slots();

    for (uint64_t i = 0; i < CAPACITY; i++) {
        EXPECT_TRUE(queue.push(fake_slot_state(i))) << "push " << i;
    }
    EXPECT_EQ(queue.pop(), fake_slot_state(0));
}
