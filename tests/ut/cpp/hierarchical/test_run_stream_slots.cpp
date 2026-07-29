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

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "host/run_stream_slots.h"

namespace {

// Hands out distinct fake handles and can be told to fail the next N destroys,
// which is what makes the destroy-failure path reachable without a device.
class FakeStreams {
public:
    int create(void **out) {
        if (create_failures_ > 0) {
            --create_failures_;
            return -7;
        }
        *out = reinterpret_cast<void *>(++next_handle_);
        live_.push_back(*out);
        return 0;
    }

    int destroy(void *stream) {
        if (destroy_failures_ > 0) {
            --destroy_failures_;
            return -13;
        }
        for (auto it = live_.begin(); it != live_.end(); ++it) {
            if (*it == stream) {
                live_.erase(it);
                return 0;
            }
        }
        ADD_FAILURE() << "destroyed a handle that was not live";
        return -1;
    }

    void fail_next_destroys(int n) { destroy_failures_ = n; }
    void fail_next_creates(int n) { create_failures_ = n; }
    size_t live_count() const { return live_.size(); }

private:
    std::vector<void *> live_;
    uintptr_t next_handle_{0};
    int destroy_failures_{0};
    int create_failures_{0};
};

RunStreamSlots make_slots(FakeStreams &fake) {
    return RunStreamSlots(
        [&fake](void **out) {
            return fake.create(out);
        },
        [&fake](void *s) {
            return fake.destroy(s);
        }
    );
}

// The AICPU stream is the slot's for the runner's lifetime; the AICore stream
// belongs to one run, so the count advances once per acquire.
TEST(RunStreamSlots, EveryAcquireCreatesAnAicoreStreamAndKeepsTheAicpuOne) {
    FakeStreams fake;
    RunStreamSlots slots = make_slots(fake);

    ASSERT_EQ(slots.acquire(0), 0);
    void *aicpu = slots.aicpu(0);
    void *first_aicore = slots.aicore(0);
    ASSERT_NE(aicpu, nullptr);
    ASSERT_NE(first_aicore, nullptr);
    EXPECT_EQ(slots.created_count(), 1u);

    ASSERT_EQ(slots.retire_aicore(0), 0);
    EXPECT_EQ(slots.aicore(0), nullptr);

    ASSERT_EQ(slots.acquire(0), 0);
    EXPECT_EQ(slots.aicpu(0), aicpu) << "the AICPU stream must not be recreated";
    EXPECT_NE(slots.aicore(0), first_aicore) << "the AICore stream must be a new one";
    EXPECT_EQ(slots.created_count(), 2u);
}

// The three consequences a failed destroy must have.
TEST(RunStreamSlots, AFailedDestroyReportsKeepsTheHandleAndLocksTheSlot) {
    FakeStreams fake;
    RunStreamSlots slots = make_slots(fake);

    ASSERT_EQ(slots.acquire(0), 0);
    void *stranded = slots.aicore(0);

    // 1. the error reaches the caller
    fake.fail_next_destroys(1);
    EXPECT_EQ(slots.retire_aicore(0), -13);

    // 2. the handle survives, so teardown still has something to reclaim
    EXPECT_EQ(slots.aicore(0), stranded);

    // 3. the next run is refused rather than handed a second live stream
    EXPECT_NE(slots.acquire(0), 0);
    EXPECT_EQ(slots.aicore(0), stranded);
    EXPECT_EQ(slots.created_count(), 1u) << "a refused acquire must not create anything";

    // and teardown retries it
    EXPECT_EQ(slots.destroy_all(), 0);
    EXPECT_EQ(slots.aicore(0), nullptr);
    EXPECT_EQ(fake.live_count(), 0u);
}

// A slot locked by a failed destroy must not take the whole runner with it.
TEST(RunStreamSlots, AStrandedSlotDoesNotBlockItsPeer) {
    FakeStreams fake;
    RunStreamSlots slots = make_slots(fake);

    ASSERT_EQ(slots.acquire(0), 0);
    fake.fail_next_destroys(1);
    ASSERT_NE(slots.retire_aicore(0), 0);

    ASSERT_EQ(slots.acquire(1), 0);
    EXPECT_NE(slots.aicore(1), nullptr);
    EXPECT_EQ(slots.retire_aicore(1), 0);
}

// destroy_all reports the first failure and keeps only what it could not free,
// so a second teardown attempt is still meaningful.
TEST(RunStreamSlots, DestroyAllReportsFailureAndRetriesWhatSurvived) {
    FakeStreams fake;
    RunStreamSlots slots = make_slots(fake);
    ASSERT_EQ(slots.acquire(0), 0);
    ASSERT_EQ(slots.acquire(1), 0);

    fake.fail_next_destroys(1);
    EXPECT_EQ(slots.destroy_all(), -13);
    EXPECT_EQ(fake.live_count(), 1u) << "only the stream whose destroy failed may survive";

    EXPECT_EQ(slots.destroy_all(), 0);
    EXPECT_EQ(fake.live_count(), 0u);
}

// A failed create leaves nothing half-owned behind.
TEST(RunStreamSlots, AFailedCreateLeavesTheSlotEmpty) {
    FakeStreams fake;
    RunStreamSlots slots = make_slots(fake);

    fake.fail_next_creates(1);
    EXPECT_NE(slots.acquire(0), 0);
    EXPECT_EQ(slots.aicpu(0), nullptr);
    EXPECT_EQ(slots.aicore(0), nullptr);
    EXPECT_EQ(slots.created_count(), 0u);

    // The AICPU stream lands, the AICore one fails: the slot stays unusable
    // rather than half-ready.
    fake.fail_next_creates(0);
    ASSERT_EQ(slots.acquire(0), 0);
    ASSERT_EQ(slots.retire_aicore(0), 0);
    fake.fail_next_creates(1);
    EXPECT_NE(slots.acquire(0), 0);
    EXPECT_FALSE(slots.ready(0));
}

TEST(RunStreamSlots, RejectsASlotOutsideTheContractDepth) {
    FakeStreams fake;
    RunStreamSlots slots = make_slots(fake);
    const unsigned past_end = static_cast<unsigned>(RunStreamSlots::capacity());
    EXPECT_NE(slots.acquire(past_end), 0);
    EXPECT_NE(slots.retire_aicore(past_end), 0);
    EXPECT_EQ(slots.aicore(past_end), nullptr);
}

}  // namespace
