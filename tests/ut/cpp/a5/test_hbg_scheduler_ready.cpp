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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

#include "scheduler/scheduler_ready.h"
#include "runtime_types.h"

namespace {

class SchedulerStateBuffer {
public:
    explicit SchedulerStateBuffer(const AicoreSchedulerLayout &layout) :
        base_(std::aligned_alloc(SCHEDULER_STATE_ALIGNMENT, layout.total_size)) {
        EXPECT_NE(base_, nullptr);
        if (base_ != nullptr) EXPECT_TRUE(scheduler_init_data_from_layout(base_, layout));
    }
    ~SchedulerStateBuffer() { std::free(base_); }
    void *base() const { return base_; }

private:
    void *base_{nullptr};
};

class GraphBuffer {
public:
    explicit GraphBuffer(size_t task_count) :
        task_count_(task_count) {
        while (capacity_ < std::max<size_t>(task_count, 1))
            capacity_ <<= 1;
        descriptors_ = std::make_unique<TaskDescriptor[]>(capacity_);
        payloads_ = std::make_unique<TaskPayload[]>(capacity_);
        fanins_ = std::make_unique<int32_t[]>(capacity_ * SCHEDULER_GRAPH_MAX_FANIN);
        for (size_t task = 0; task < capacity_; ++task) {
            descriptors_[task].task_id = TaskId{static_cast<uint64_t>(task)};
            payloads_[task].bind_regions(
                nullptr, nullptr, fanins_.get() + task * static_cast<size_t>(SCHEDULER_GRAPH_MAX_FANIN)
            );
            for (int slot = 0; slot < 3; ++slot)
                descriptors_[task].kernel_id[slot] = INVALID_KERNEL_ID;
        }
    }

    void executable(size_t task, uint8_t subtask_slot, std::vector<int32_t> fanins = {}) {
        ASSERT_LT(task, task_count_);
        ASSERT_LT(subtask_slot, 3);
        ASSERT_LE(fanins.size(), static_cast<size_t>(SCHEDULER_GRAPH_MAX_FANIN));
        descriptors_[task].kernel_id[subtask_slot] = 1;
        payloads_[task].fanin_count = static_cast<int32_t>(fanins.size());
        std::copy(fanins.begin(), fanins.end(), payloads_[task].fanin_data());
    }

    void mixed(size_t task, uint8_t active_mask) {
        ASSERT_LT(task, task_count_);
        for (uint8_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
            if ((active_mask & (1U << subtask_slot)) != 0) descriptors_[task].kernel_id[subtask_slot] = 1;
        }
        payloads_[task].fanin_count = 0;
    }

    SchedulerGraphView graph() const {
        return {
            reinterpret_cast<uint64_t>(descriptors_.get()),
            reinterpret_cast<uint64_t>(payloads_.get()),
            task_count_,
            capacity_ - 1,
        };
    }

private:
    size_t task_count_;
    size_t capacity_{1};
    std::unique_ptr<TaskDescriptor[]> descriptors_;
    std::unique_ptr<TaskPayload[]> payloads_;
    std::unique_ptr<int32_t[]> fanins_;
};

struct FixtureStorage {
    explicit FixtureStorage(uint64_t task_count, uint64_t workers = 2) {
        EXPECT_TRUE(scheduler_plan_layout(task_count, task_count, 0, &layout));
        scheduler_state = std::make_unique<SchedulerStateBuffer>(layout);
        run_control = scheduler_state_at<SchedulerRunControl>(scheduler_state->base(), layout.run_control_offset);
        contexts = scheduler_state_at<SchedulerWorkerContext>(scheduler_state->base(), layout.worker_contexts_offset);
        run_control->aiv_active_worker_count = workers;
        run_control->resolver_count = workers;
        for (uint64_t worker = 0; worker < workers; ++worker) {
            SchedulerWorkerContext &context = contexts[worker];
            context.core_type = static_cast<int32_t>(CoreType::AIV);
            context.active = 1;
            context.task_controls_offset = layout.task_controls_offset;
            context.task_metadata_offset = layout.task_metadata_offset;
            context.completion_inboxes_offset = layout.completion_inboxes_offset;
            context.ready_inboxes_offset = layout.ready_inboxes_offset;
            context.ready_owner_states_offset = layout.ready_owner_states_offset;
            context.ready_directory_offset = layout.ready_directory_offset;
            context.trace_cells_offset = layout.trace_cells_offset;
            context.worker_contexts_offset = layout.worker_contexts_offset;
            context.dispatch_slots_offset = layout.dispatch_slots_offset;
            context.callable_addresses_offset = layout.callable_addresses_offset;
            context.gang_coordinator_offset = layout.gang_coordinator_offset;
            context.gang_cohorts_offset = layout.gang_cohorts_offset;
            context.gang_participants_offset = layout.gang_participants_offset;
            context.gang_commands_offset = layout.gang_commands_offset;
            context.dispatch_payload_offset =
                layout.dispatch_payloads_offset + worker * SCHEDULER_PENDING_SLOT_COUNT * sizeof(DispatchPayload);
            context.graph_task_count = task_count;
            context.runtime_worker_count = workers;
            context.worker_index = worker;
            context.inbox_index = worker;
        }
        metadata = scheduler_state_at<SchedulerTaskMetadata>(scheduler_state->base(), layout.task_metadata_offset);
        for (uint64_t task = 0; task < task_count; ++task) {
            metadata[task].kernel_ids[0] = 1;
            metadata[task].kernel_ids[1] = UINT16_MAX;
            metadata[task].kernel_ids[2] = UINT16_MAX;
            metadata[task].active_mask = 1;
            metadata[task].logical_block_num = 1;
            metadata[task].total_required_subtasks = 1;
            metadata[task].flags = SCHEDULER_TASK_EXECUTABLE;
        }
    }

    AicoreSchedulerLayout layout{};
    std::unique_ptr<SchedulerStateBuffer> scheduler_state;
    SchedulerRunControl *run_control{nullptr};
    SchedulerWorkerContext *contexts{nullptr};
    SchedulerTaskMetadata *metadata{nullptr};
};

TEST(SchedulerBootstrap, RegistersOnlyOnFirstExecutableProducer) {
    FixtureStorage storage(4, 2);
    GraphBuffer graph(4);
    graph.executable(0, 0);
    graph.executable(1, 1, {0});
    graph.executable(3, 1, {2, 1});
    storage.metadata[1].active_mask = 2;
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    storage.metadata[2].flags = 0;
    storage.metadata[3].active_mask = 2;
    storage.metadata[3].flags |= SCHEDULER_TASK_HAS_FANIN;
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    controls[2].state = static_cast<int64_t>(SchedulerTaskState::DONE);
    controls[2].wake_list_head = SCHEDULER_WAKE_LIST_CLOSED;

    SchedulerWakeStats stats{};
    EXPECT_EQ(
        scheduler_bootstrap_route_task(
            graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 1, &stats
        ),
        SchedulerRouteResult::WAITING
    );
    EXPECT_EQ(controls[0].wake_list_head, 1);
    EXPECT_EQ(controls[1].next_waiter, SCHEDULER_WAKE_LIST_OPEN);
    EXPECT_EQ(controls[1].waiting_producer, 0);

    EXPECT_EQ(
        scheduler_bootstrap_route_task(
            graph.graph(), storage.scheduler_state->base(), &storage.contexts[1], storage.run_control, 3, &stats
        ),
        SchedulerRouteResult::WAITING
    );
    EXPECT_EQ(controls[1].wake_list_head, 3);
    EXPECT_EQ(controls[3].next_fanin_index, 1);
    EXPECT_EQ(controls[3].waiting_producer, 1);
    EXPECT_EQ(stats.wake_register_count, 2u);
    EXPECT_EQ(stats.fanin_state_load_count, 0u);
    EXPECT_EQ(stats.wake_cas_retry_count, 0u);
}

TEST(SchedulerBootstrap, PublishesExclusiveInboxAndAggregatesDirectory) {
    FixtureStorage storage(2, 2);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    storage.contexts[0].inbox_index = 1;
    storage.contexts[1].inbox_index = 0;
    SchedulerReadyBatch batch{};
    SchedulerReadyStats stats{};
    ASSERT_TRUE(
        scheduler_bootstrap_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[1], 0, &batch, &stats)
    );
    ASSERT_TRUE(
        scheduler_bootstrap_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[1], 1, &batch, &stats)
    );
    scheduler_cache_barrier();
    uint64_t ready_types = 0;
    ASSERT_TRUE(scheduler_bootstrap_ready_batch_publish(
        storage.scheduler_state->base(), &storage.contexts[1], 0, 0, &batch, &stats, &ready_types
    ));
    auto *directory = scheduler_ready_directory_at(storage.scheduler_state->base(), &storage.contexts[1]);
    directory->bootstrap_ready_types[0] = ready_types;
    scheduler_bootstrap_ready_directory_publish(storage.scheduler_state->base(), &storage.contexts[1], 2);

    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    EXPECT_EQ(controls[0].next_waiter, 1);
    EXPECT_EQ(controls[1].next_waiter, SCHEDULER_INBOX_EMPTY);
    EXPECT_EQ(scheduler_ready_inbox_at(storage.scheduler_state->base(), &storage.contexts[1], 0, 0)->head, 0);
    EXPECT_EQ(directory->core_types[0][0].bits, 1u);
    EXPECT_EQ(directory->core_types[1][0].bits, 0u);
    EXPECT_EQ(stats.enqueue_count, 2u);
    EXPECT_EQ(stats.batch_count, 1u);
}

TEST(SchedulerReadyInbox, BatchPushAndOwnerMaintenancePreserveFifoAndDirectory) {
    constexpr uint64_t kTasks = 4;
    FixtureStorage storage(kTasks, 1);
    GraphBuffer graph(kTasks);
    for (uint64_t task = 0; task < kTasks; ++task)
        graph.executable(task, 0);
    SchedulerReadyBatch batch{};
    SchedulerReadyOwnerState owner_state{};
    SchedulerReadyStats stats{};
    for (uint64_t task = 0; task < kTasks; ++task)
        ASSERT_TRUE(
            scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[0], task, &batch, &stats)
        );
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), &storage.contexts[0], 0, 0, &batch, &stats, &owner_state
    ));

    auto *directory = scheduler_state_at<SchedulerReadyDirectory>(
        storage.scheduler_state->base(), storage.layout.ready_directory_offset
    );
    EXPECT_NE(directory->core_types[0][0].bits & 1, 0u);
    for (uint64_t index = 0; index < kTasks; ++index) {
        int64_t task = SCHEDULER_TASK_ID_INVALID;
        ASSERT_TRUE(scheduler_ready_pop_from_inbox(
            graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task,
            &stats
        ));
        EXPECT_EQ(task, static_cast<int64_t>(index));
    }
    int64_t task = SCHEDULER_TASK_ID_INVALID;
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, &stats
    ));
    EXPECT_EQ(task, SCHEDULER_TASK_ID_INVALID);
    EXPECT_NE(directory->core_types[0][0].bits & 1, 0u);
    ASSERT_TRUE(
        scheduler_ready_owner_maintain_type(storage.scheduler_state->base(), &storage.contexts[0], 0, &owner_state)
    );
    EXPECT_EQ(directory->core_types[0][0].bits & 1, 0u);
    EXPECT_EQ(stats.pop_count, kTasks);
}

TEST(SchedulerReadyInbox, OwnerStateInitializationRestoresEmptySentinels) {
    SchedulerReadyOwnerState owner_state;
    __builtin_memset(&owner_state, 0, sizeof(owner_state));

    scheduler_ready_owner_init(&owner_state);

    for (uint32_t type = 0; type < SCHEDULER_CORE_TYPE_COUNT; ++type) {
        EXPECT_EQ(owner_state.queues[type].pending_endpoints, SCHEDULER_READY_PENDING_EMPTY);
        EXPECT_EQ(owner_state.queues[type].advertised, 0u);
    }
}

TEST(SchedulerReadyInbox, PackedOwnerEndpointsRoundTripAsOneWord) {
    EXPECT_EQ(scheduler_ready_pending_pack(SCHEDULER_INBOX_EMPTY, SCHEDULER_INBOX_EMPTY), UINT64_MAX);
    const uint64_t endpoints = scheduler_ready_pending_pack(INT32_MAX - 1, INT32_MAX);
    EXPECT_EQ(scheduler_ready_pending_head(endpoints), INT32_MAX - 1);
    EXPECT_EQ(scheduler_ready_pending_tail(endpoints), INT32_MAX);
}

TEST(SchedulerReadyInbox, OwnerPromotesPendingBankAfterPublishedBankDrains) {
    constexpr uint64_t kTasks = 4;
    FixtureStorage storage(kTasks, 1);
    GraphBuffer graph(kTasks);
    for (uint64_t task = 0; task < kTasks; ++task)
        graph.executable(task, 0);
    SchedulerReadyOwnerState owner_state{};
    SchedulerReadyStats stats{};
    SchedulerReadyBatch published{};
    SchedulerReadyBatch pending{};
    for (int64_t task = 0; task < 2; ++task)
        ASSERT_TRUE(scheduler_ready_batch_append(
            storage.scheduler_state->base(), &storage.contexts[0], task, &published, &stats
        ));
    for (int64_t task = 2; task < 4; ++task)
        ASSERT_TRUE(
            scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[0], task, &pending, &stats)
        );
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), &storage.contexts[0], 0, 0, &published, &stats, &owner_state
    ));
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), &storage.contexts[0], 0, 0, &pending, &stats, &owner_state
    ));
    const uint64_t endpoints = owner_state.queues[0].pending_endpoints;
    EXPECT_EQ(scheduler_ready_pending_head(endpoints), 2);
    EXPECT_EQ(scheduler_ready_pending_tail(endpoints), 3);

    for (int64_t expected = 0; expected < 2; ++expected) {
        int64_t task = SCHEDULER_TASK_ID_INVALID;
        ASSERT_TRUE(scheduler_ready_pop_from_inbox(
            graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task,
            &stats
        ));
        EXPECT_EQ(task, expected);
    }
    int64_t task = SCHEDULER_TASK_ID_INVALID;
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, &stats
    ));
    EXPECT_EQ(task, SCHEDULER_TASK_ID_INVALID);
    ASSERT_TRUE(
        scheduler_ready_owner_maintain_type(storage.scheduler_state->base(), &storage.contexts[0], 0, &owner_state)
    );
    for (int64_t expected = 2; expected < 4; ++expected) {
        ASSERT_TRUE(scheduler_ready_pop_from_inbox(
            graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task,
            &stats
        ));
        EXPECT_EQ(task, expected);
    }
}

TEST(SchedulerReadyInbox, OlderPendingBankPrecedesBatchArrivingAfterDrain) {
    constexpr uint64_t kTasks = 3;
    FixtureStorage storage(kTasks, 1);
    GraphBuffer graph(kTasks);
    for (uint64_t task = 0; task < kTasks; ++task)
        graph.executable(task, 0);
    SchedulerReadyOwnerState owner_state{};
    SchedulerReadyStats stats{};
    for (int64_t task = 0; task < 2; ++task) {
        SchedulerReadyBatch batch{};
        ASSERT_TRUE(
            scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[0], task, &batch, &stats)
        );
        ASSERT_TRUE(scheduler_ready_batch_push(
            storage.scheduler_state->base(), &storage.contexts[0], 0, 0, &batch, &stats, &owner_state
        ));
    }
    int64_t task = SCHEDULER_TASK_ID_INVALID;
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, &stats
    ));
    ASSERT_EQ(task, 0);

    SchedulerReadyBatch arriving{};
    ASSERT_TRUE(
        scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[0], 2, &arriving, &stats)
    );
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), &storage.contexts[0], 0, 0, &arriving, &stats, &owner_state
    ));
    EXPECT_EQ(scheduler_ready_pending_head(owner_state.queues[0].pending_endpoints), 2);
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, &stats
    ));
    EXPECT_EQ(task, 1);
    ASSERT_TRUE(
        scheduler_ready_owner_maintain_type(storage.scheduler_state->base(), &storage.contexts[0], 0, &owner_state)
    );
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, &stats
    ));
    EXPECT_EQ(task, 2);
}

TEST(SchedulerReadyInbox, ThiefCannotObserveOrPromoteOwnerPendingBank) {
    FixtureStorage storage(2, 2);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    SchedulerReadyOwnerState owner_state{};
    SchedulerReadyStats stats{};
    for (int64_t task = 0; task < 2; ++task) {
        SchedulerReadyBatch batch{};
        ASSERT_TRUE(
            scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[1], task, &batch, &stats)
        );
        ASSERT_TRUE(scheduler_ready_batch_push(
            storage.scheduler_state->base(), &storage.contexts[1], 0, 1, &batch, &stats, &owner_state
        ));
    }
    int64_t task = SCHEDULER_TASK_ID_INVALID;
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 1, &task, &stats
    ));
    ASSERT_EQ(task, 0);
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 1, &task, &stats
    ));
    EXPECT_EQ(task, SCHEDULER_TASK_ID_INVALID);
    EXPECT_EQ(scheduler_ready_pending_head(owner_state.queues[0].pending_endpoints), 1);
    ASSERT_TRUE(
        scheduler_ready_owner_maintain_type(storage.scheduler_state->base(), &storage.contexts[1], 0, &owner_state)
    );
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 1, &task, &stats
    ));
    EXPECT_EQ(task, 1);
}

TEST(SchedulerReadyInbox, StealsOnlyFromMarkedVictim) {
    FixtureStorage storage(1, 2);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerReadyBatch batch{};
    SchedulerReadyStats stats{};
    ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[1], 0, &batch, &stats));
    ASSERT_TRUE(
        scheduler_ready_batch_push(storage.scheduler_state->base(), &storage.contexts[1], 0, 1, &batch, &stats)
    );

    uint64_t cursor = 1;
    SchedulerReadyClaim claim{};
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 2, 0, &cursor,
        &stats, &claim
    ));
    EXPECT_EQ(claim.task_id, 0);
    EXPECT_EQ(claim.inbox_index, 1u);
    EXPECT_EQ(claim.source, SchedulerReadySource::STOLEN);
    EXPECT_EQ(stats.steal_count, 1u);
}

TEST(SchedulerReadyInbox, DirectoryShardIgnoresResolverTail) {
    FixtureStorage storage(1, 9);
    auto *directory = scheduler_ready_directory_at(storage.scheduler_state->base(), &storage.contexts[0]);
    directory->core_types[0][1].bits = UINT64_C(1) << 6;
    EXPECT_EQ(scheduler_load_ready_directory_shard(directory, 9, 0, 7), 0u);

    directory->core_types[0][1].bits = UINT64_C(1) << 1;
    EXPECT_EQ(scheduler_load_ready_directory_shard(directory, 9, 0, 7), UINT64_C(1) << 1);
}

TEST(SchedulerReadyInbox, BootstrapPublishesIndependentDirectoryShards) {
    FixtureStorage storage(1, 14);
    auto *directory = scheduler_ready_directory_at(storage.scheduler_state->base(), &storage.contexts[0]);
    directory->bootstrap_ready_types[0] = UINT64_C(1) << 0;
    directory->bootstrap_ready_types[6] = UINT64_C(1) << 0;
    directory->bootstrap_ready_types[7] = UINT64_C(1) << 1;
    directory->bootstrap_ready_types[13] = (UINT64_C(1) << 0) | (UINT64_C(1) << 1);

    scheduler_bootstrap_ready_directory_publish(storage.scheduler_state->base(), &storage.contexts[0], 14);

    EXPECT_EQ(directory->core_types[0][0].bits, (UINT64_C(1) << 0) | (UINT64_C(1) << 6));
    EXPECT_EQ(directory->core_types[1][0].bits, 0u);
    EXPECT_EQ(directory->core_types[0][1].bits, UINT64_C(1) << 6);
    EXPECT_EQ(directory->core_types[1][1].bits, (UINT64_C(1) << 0) | (UINT64_C(1) << 6));
}

TEST(SchedulerReadyInbox, SparseDirectoryWrapsWithinShard) {
    FixtureStorage storage(2, 14);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    SchedulerReadyStats stats{};
    SchedulerReadyBatch high_batch{};
    SchedulerReadyBatch low_batch{};
    ASSERT_TRUE(
        scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[13], 0, &high_batch, &stats)
    );
    ASSERT_TRUE(
        scheduler_ready_batch_push(storage.scheduler_state->base(), &storage.contexts[13], 0, 13, &high_batch, &stats)
    );
    ASSERT_TRUE(
        scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[8], 1, &low_batch, &stats)
    );
    ASSERT_TRUE(
        scheduler_ready_batch_push(storage.scheduler_state->base(), &storage.contexts[8], 0, 8, &low_batch, &stats)
    );

    uint64_t cursor = 12;
    SchedulerReadyClaim claim{};
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[7], storage.run_control, 14, 0, &cursor,
        &stats, &claim
    ));
    EXPECT_EQ(claim.task_id, 0);
    EXPECT_EQ(claim.inbox_index, 13u);
    EXPECT_EQ(cursor, 7u);

    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[7], storage.run_control, 14, 0, &cursor,
        &stats, &claim
    ));
    EXPECT_EQ(claim.task_id, 1);
    EXPECT_EQ(claim.inbox_index, 8u);
    EXPECT_EQ(cursor, 9u);
}

TEST(SchedulerReadyInbox, DoesNotStealAcrossDirectoryShards) {
    FixtureStorage storage(1, 14);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerReadyBatch batch{};
    SchedulerReadyStats stats{};
    ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[8], 0, &batch, &stats));
    ASSERT_TRUE(
        scheduler_ready_batch_push(storage.scheduler_state->base(), &storage.contexts[8], 0, 8, &batch, &stats)
    );

    uint64_t cursor = 1;
    SchedulerReadyClaim claim{};
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 14, 0, &cursor,
        &stats, &claim
    ));
    EXPECT_EQ(claim.task_id, SCHEDULER_TASK_ID_INVALID);

    cursor = 8;
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[7], storage.run_control, 14, 0, &cursor,
        &stats, &claim
    ));
    EXPECT_EQ(claim.task_id, 0);
    EXPECT_EQ(claim.inbox_index, 8u);
    EXPECT_EQ(claim.source, SchedulerReadySource::STOLEN);
}

TEST(SchedulerReadyInbox, ConcurrentConsumersNeverDuplicateTask) {
    constexpr uint64_t kTasks = 8192;
    constexpr uint64_t kConsumerCount = 8;
    FixtureStorage storage(kTasks, kConsumerCount);
    GraphBuffer graph(kTasks);
    SchedulerReadyBatch batch{};
    for (uint64_t task = 0; task < kTasks; ++task) {
        graph.executable(task, 0);
        ASSERT_TRUE(
            scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[0], task, &batch, nullptr)
        );
    }
    ASSERT_TRUE(
        scheduler_ready_batch_push(storage.scheduler_state->base(), &storage.contexts[0], 0, 0, &batch, nullptr)
    );
    std::vector<std::atomic<uint32_t>> seen(kTasks);
    std::atomic<uint64_t> claimed{0};
    auto consume = [&](uint64_t worker) {
        while (claimed.load(std::memory_order_relaxed) < kTasks) {
            int64_t task = SCHEDULER_TASK_ID_INVALID;
            if (!scheduler_ready_pop_from_inbox(
                    graph.graph(), storage.scheduler_state->base(), &storage.contexts[worker], storage.run_control, 0,
                    0, &task, nullptr
                ))
                return;
            if (task >= 0) {
                seen[static_cast<size_t>(task)].fetch_add(1, std::memory_order_relaxed);
                claimed.fetch_add(1, std::memory_order_relaxed);
            }
            std::this_thread::yield();
        }
    };
    std::vector<std::thread> consumers;
    consumers.reserve(kConsumerCount);
    for (uint64_t worker = 0; worker < kConsumerCount; ++worker)
        consumers.emplace_back(consume, worker);
    for (auto &consumer : consumers)
        consumer.join();
    EXPECT_EQ(claimed.load(), kTasks);
    for (const auto &count : seen)
        EXPECT_EQ(count.load(), 1u);
}

TEST(SchedulerReadyWake, WakeResolvePublishesConsumerToResolverLocalInbox) {
    FixtureStorage storage(2, 1);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    SchedulerWakeStats wake{};
    SchedulerReadyStats ready{};
    SchedulerCompletionStats completion{};
    EXPECT_EQ(
        scheduler_route_task(
            graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 1, &wake
        ),
        SchedulerRouteResult::WAITING
    );
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    controls[0].state = static_cast<int64_t>(SchedulerTaskState::DONE);
    ASSERT_TRUE(scheduler_resolve_completion(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, &wake, &ready,
        &completion
    ));
    int64_t task = SCHEDULER_TASK_ID_INVALID;
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, &ready
    ));
    EXPECT_EQ(task, 1);
    EXPECT_EQ(wake.wake_register_count, 1u);
    EXPECT_EQ(wake.wake_migrate_count, 1u);
}

TEST(SchedulerReadyWake, WakeResolveQueuesBehindOlderPublishedWork) {
    FixtureStorage storage(3, 1);
    GraphBuffer graph(3);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    graph.executable(2, 0);
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    SchedulerWakeStats wake{};
    SchedulerReadyStats ready{};
    SchedulerCompletionStats completion{};
    SchedulerReadyOwnerState owner_state{};
    EXPECT_EQ(
        scheduler_route_task(
            graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 1, &wake
        ),
        SchedulerRouteResult::WAITING
    );
    SchedulerReadyBatch older{};
    ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[0], 2, &older, &ready));
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), &storage.contexts[0], 0, 0, &older, &ready, &owner_state
    ));
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    controls[0].state = static_cast<int64_t>(SchedulerTaskState::DONE);
    ASSERT_TRUE(scheduler_resolve_completion(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, &wake, &ready,
        &completion, false, true, nullptr, &owner_state
    ));
    EXPECT_EQ(scheduler_ready_pending_head(owner_state.queues[0].pending_endpoints), 1);

    int64_t task = SCHEDULER_TASK_ID_INVALID;
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, &ready
    ));
    ASSERT_EQ(task, 2);
    ASSERT_TRUE(
        scheduler_ready_owner_maintain_type(storage.scheduler_state->base(), &storage.contexts[0], 0, &owner_state)
    );
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, &ready
    ));
    EXPECT_EQ(task, 1);
}

}  // namespace
