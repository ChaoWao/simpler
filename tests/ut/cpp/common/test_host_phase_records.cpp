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

#include "host/host_phase_records.h"

using simpler::dfx::HostPhaseRecordStore;

namespace {

void record_n(HostPhaseRecordPool *pool, HostPhaseKind kind, uint32_t n, uint64_t first_start = 0) {
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t start = first_start + i;
        host_phase_pool_append(pool, kind, start, start + 1, i, i);
    }
}

TEST(HostPhaseRecords, DisarmedPoolCollectsNothing) {
    HostPhaseRecordStore store;
    HostPhaseRecordPool *pool = store.arm(/*collect_records=*/false);
    EXPECT_EQ(pool, nullptr);
    EXPECT_FALSE(store.armed());

    // The producer's record call must tolerate a null pool — that is the common
    // configuration, where only the per-kind counters run.
    record_n(pool, HostPhaseKind::OrchRecordNode, 10);
    store.finish(0, /*invocation_id=*/9);
    EXPECT_TRUE(store.records().empty());
    EXPECT_EQ(store.total_records(), 0u);
    EXPECT_EQ(store.dropped_records(), 0u);
}

TEST(HostPhaseRecords, RecordsSurviveInOrderWithinOneBuffer) {
    HostPhaseRecordStore store;
    HostPhaseRecordPool *pool = store.arm(true);
    ASSERT_NE(pool, nullptr);

    record_n(pool, HostPhaseKind::OrchRecordNode, 337, /*first_start=*/1000);
    store.finish(0, /*invocation_id=*/9);

    const auto records = store.records();
    ASSERT_EQ(records.size(), 337u);
    EXPECT_EQ(store.total_records(), 337u);
    EXPECT_EQ(store.dropped_records(), 0u);
    for (size_t i = 0; i < records.size(); ++i) {
        EXPECT_EQ(records[i].start_ns, 1000u + i) << "record " << i << " out of rotation order";
        EXPECT_EQ(records[i].kind, static_cast<uint32_t>(HostPhaseKind::OrchRecordNode));
    }
}

TEST(HostPhaseRecords, RotationSpansEveryBufferInOrder) {
    HostPhaseRecordStore store;
    HostPhaseRecordPool *pool = store.arm(true);
    ASSERT_NE(pool, nullptr);

    // Exactly fills the pool: one active buffer plus PLATFORM_PROF_SLOT_COUNT
    // spares, so the last record lands without a drop.
    const uint32_t capacity = static_cast<uint32_t>(HostPhaseRecordStore::capacity());
    record_n(pool, HostPhaseKind::OrchGraphSubmit, capacity, /*first_start=*/1);
    store.finish(0, /*invocation_id=*/9);

    const auto records = store.records();
    ASSERT_EQ(records.size(), capacity);
    EXPECT_EQ(store.dropped_records(), 0u);
    EXPECT_EQ(pool->head.current_buf_seq, static_cast<uint32_t>(PLATFORM_HOST_PHASE_BUFFERS));
    for (uint32_t i = 0; i < capacity; ++i) {
        EXPECT_EQ(records[i].start_ns, 1u + i) << "record " << i << " out of rotation order";
    }
}

TEST(HostPhaseRecords, OverflowDropsAndCountsWithoutLosingEarlierRecords) {
    HostPhaseRecordStore store;
    HostPhaseRecordPool *pool = store.arm(true);
    ASSERT_NE(pool, nullptr);

    const uint32_t capacity = static_cast<uint32_t>(HostPhaseRecordStore::capacity());
    const uint32_t overshoot = 25;
    record_n(pool, HostPhaseKind::OrchRecordNode, capacity + overshoot, /*first_start=*/1);
    store.finish(0, /*invocation_id=*/9);

    const auto records = store.records();
    EXPECT_EQ(records.size(), capacity);
    EXPECT_EQ(store.total_records(), capacity + overshoot);
    EXPECT_EQ(store.dropped_records(), overshoot);
    // Dropping is tail-loss, not corruption: the retained prefix is intact.
    ASSERT_FALSE(records.empty());
    EXPECT_EQ(records.front().start_ns, 1u);
    EXPECT_EQ(records.back().start_ns, static_cast<uint64_t>(capacity));
}

TEST(HostPhaseRecords, ReArmClearsThePreviousPass) {
    HostPhaseRecordStore store;
    HostPhaseRecordPool *pool = store.arm(true);
    ASSERT_NE(pool, nullptr);
    record_n(pool, HostPhaseKind::OrchRecordNode, 50);
    store.finish(7, /*invocation_id=*/9);
    EXPECT_EQ(store.records().size(), 50u);
    EXPECT_EQ(store.submitted_tasks(), 7u);

    pool = store.arm(true);
    ASSERT_NE(pool, nullptr);
    EXPECT_TRUE(store.records().empty());
    EXPECT_EQ(store.total_records(), 0u);
    EXPECT_FALSE(store.finished());
    record_n(pool, HostPhaseKind::OrchGraphSubmit, 3);
    store.finish(3, /*invocation_id=*/9);
    EXPECT_EQ(store.records().size(), 3u);
}

TEST(HostPhaseRecords, SubmitProjectionKeepsOnlyTaskSubmittingKinds) {
    HostPhaseRecordStore store;
    HostPhaseRecordPool *pool = store.arm(true);
    ASSERT_NE(pool, nullptr);

    // Mirrors one qwen decode pass: the three task-submitting kinds sum to the
    // pass's total_tasks, and the sub-operations do not count.
    record_n(pool, HostPhaseKind::OrchSubmitTask, 5);
    record_n(pool, HostPhaseKind::OrchAllocTensors, 2);
    record_n(pool, HostPhaseKind::OrchRecordNode, 277);
    record_n(pool, HostPhaseKind::OrchGraphSubmit, 40);
    record_n(pool, HostPhaseKind::OrchBuildDefinition, 1);
    record_n(pool, HostPhaseKind::BindHostOrch, 1);
    store.finish(47, /*invocation_id=*/9);

    EXPECT_EQ(store.records().size(), 326u);
    EXPECT_EQ(store.submit_records().size(), 47u);
    EXPECT_EQ(store.submitted_tasks(), 47u);
    for (const auto &record : store.submit_records()) {
        EXPECT_TRUE(host_phase_kind_submits_task(static_cast<HostPhaseKind>(record.kind)));
    }
}

TEST(HostPhaseRecords, EveryKindHasItsOwnName) {
    for (uint32_t i = 0; i < kHostPhaseKindCount; ++i) {
        const char *name = host_phase_kind_name(static_cast<HostPhaseKind>(i));
        EXPECT_STRNE(name, "unknown") << "kind " << i << " has no name";
        for (uint32_t j = i + 1; j < kHostPhaseKindCount; ++j) {
            EXPECT_STRNE(name, host_phase_kind_name(static_cast<HostPhaseKind>(j)))
                << "kinds " << i << " and " << j << " share a name";
        }
    }
    EXPECT_STREQ(host_phase_kind_name(HostPhaseKind::Count), "unknown");
}

}  // namespace
