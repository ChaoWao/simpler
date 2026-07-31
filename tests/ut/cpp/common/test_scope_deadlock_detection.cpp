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

#include <atomic>
#include <string>

#include "pto_ring_buffer.h"

// CMake compiles this source against both the a2a3 and a5 runtime objects.
namespace {

constexpr int32_t WINDOW_SIZE = 16;
constexpr int32_t POOL_CAPACITY = 8;

void make_head_match_old_structural_predicate(PTO2TaskSlotState &head) {
    head.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
    head.fanout_count = PTO2_FANOUT_SCOPE_BIT;
    head.fanout_refcount.store(0, std::memory_order_release);
}

}  // namespace

TEST(ScopeDeadlockDetectionTest, DepPoolUsesTimeoutForDifferentScopeHead) {
    PTO2DepListEntry entries[POOL_CAPACITY]{};
    std::atomic<int32_t> error_code{PTO2_ERROR_NONE};
    PTO2DepListPool pool;
    pool.init(entries, POOL_CAPACITY, &error_code);
    for (int32_t i = 0; i < POOL_CAPACITY; ++i) {
        ASSERT_NE(pool.alloc(), nullptr);
    }

    alignas(64) PTO2TaskSlotState slot_states[WINDOW_SIZE]{};
    PTO2SharedMemoryRingHeader ring{};
    ring.fc.init();
    ring.task_window_size = WINDOW_SIZE;
    ring.task_window_mask = WINDOW_SIZE - 1;
    ring.slot_states = slot_states;
    ring.fc.current_task_index.store(2, std::memory_order_release);
    ring.fc.last_task_alive.store(0, std::memory_order_release);

    make_head_match_old_structural_predicate(slot_states[0]);
    PTO2TaskSlotState *oldest_open_task = &slot_states[1];

    testing::internal::CaptureStderr();
    bool available = pool.ensure_space(ring, 1, oldest_open_task);
    std::string log = testing::internal::GetCapturedStderr();

    EXPECT_FALSE(available);
    EXPECT_EQ(error_code.load(), PTO2_ERROR_DEP_POOL_OVERFLOW);
    EXPECT_NE(log.find("cannot reclaim space after ~500 ms"), std::string::npos);
    EXPECT_EQ(log.find("oldest task owned by an open scope on this ring"), std::string::npos);
}

TEST(ScopeDeadlockDetectionTest, DepPoolRejectsCurrentScopeHeadStructurally) {
    PTO2DepListEntry entries[POOL_CAPACITY]{};
    std::atomic<int32_t> error_code{PTO2_ERROR_NONE};
    PTO2DepListPool pool;
    pool.init(entries, POOL_CAPACITY, &error_code);
    for (int32_t i = 0; i < POOL_CAPACITY; ++i) {
        ASSERT_NE(pool.alloc(), nullptr);
    }

    alignas(64) PTO2TaskSlotState slot_states[WINDOW_SIZE]{};
    PTO2SharedMemoryRingHeader ring{};
    ring.fc.init();
    ring.task_window_size = WINDOW_SIZE;
    ring.task_window_mask = WINDOW_SIZE - 1;
    ring.slot_states = slot_states;
    ring.fc.current_task_index.store(1, std::memory_order_release);
    PTO2TaskSlotState *oldest_open_task = &slot_states[0];

    testing::internal::CaptureStderr();
    bool available = pool.ensure_space(ring, 1, oldest_open_task);
    std::string log = testing::internal::GetCapturedStderr();

    EXPECT_FALSE(available);
    EXPECT_EQ(error_code.load(), PTO2_ERROR_DEP_POOL_OVERFLOW);
    EXPECT_NE(log.find("oldest task owned by an open scope on this ring"), std::string::npos);
}
