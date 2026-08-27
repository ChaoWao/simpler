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

#include "scheduler/scheduler_dispatch.h"
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

void configure_normal_aiv_cluster(FixtureStorage &storage, uint64_t task_count) {
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    SchedulerWorkerContext &resolver = storage.contexts[1];
    resolver.is_resolver = 1;
    resolver.resolver_index = 0;
    resolver.resolver_count = 1;
    resolver.inbox_index = 0;
    resolver.cluster_worker_ids[0] = 0;
    resolver.cluster_worker_ids[1] = 1;
    resolver.cluster_worker_ids[2] = 2;
    storage.run_control->resolver_count = 1;
    auto *callables =
        scheduler_state_at<uint64_t>(storage.scheduler_state->base(), storage.layout.callable_addresses_offset);
    callables[1] = 0x1000;
    for (uint64_t worker = 0; worker < 3; ++worker) {
        for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot)
            scheduler_initialize_free_slot(
                scheduler_dispatch_slot_at(storage.scheduler_state->base(), &resolver, worker, slot)
            );
    }
    for (uint64_t task = 0; task < task_count; ++task) {
        storage.metadata[task].kernel_ids[0] = UINT16_MAX;
        storage.metadata[task].kernel_ids[1] = 1;
        storage.metadata[task].active_mask = 2;
        storage.metadata[task].flags = SCHEDULER_TASK_EXECUTABLE;
    }
}

void enqueue_normal_aiv_tasks(
    FixtureStorage &storage, SchedulerWorkerContext &resolver, uint64_t task_begin, uint64_t task_end
) {
    SchedulerReadyBatch batch{};
    SchedulerReadyStats ready_stats{};
    for (uint64_t task = task_begin; task < task_end; ++task) {
        auto *control =
            scheduler_task_control_at(storage.scheduler_state->base(), &resolver, static_cast<int64_t>(task));
        control->state = static_cast<int64_t>(SchedulerTaskState::READY);
        ASSERT_TRUE(scheduler_ready_batch_append(
            storage.scheduler_state->base(), &resolver, static_cast<int64_t>(task), &batch, &ready_stats
        ));
    }
    ASSERT_TRUE(scheduler_ready_batch_push(storage.scheduler_state->base(), &resolver, 1, 0, &batch, &ready_stats));
}

void occupy_normal_slot(
    FixtureStorage &storage, SchedulerWorkerContext &resolver, uint64_t worker_id, uint32_t pending_slot,
    int64_t task_id
) {
    auto *slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &resolver, worker_id, pending_slot);
    slot->task_id = task_id;
    slot->subtask_slot = 1;
    slot->gang = 0;
    scheduler_gm_store(
        slot->publication, scheduler_dispatch_publication(slot->generation, SchedulerDispatchSlotState::READY)
    );
}

TEST(SchedulerCompletionInbox, PacksBothGenerationSlotsInOneDeviceWord) {
    alignas(uint64_t) volatile uint32_t generations[SCHEDULER_PENDING_SLOT_COUNT] = {
        UINT32_C(0x11223344), UINT32_C(0x55667788)
    };
    EXPECT_EQ(scheduler_gm_query_u32_pair(generations), UINT64_C(0x5566778811223344));
    scheduler_gm_store(generations[1], UINT32_C(0));
    EXPECT_EQ(scheduler_gm_query_u32_pair(generations), UINT64_C(0x0000000011223344));
}

TEST(SchedulerClusterCompletion, SpscGenerationCompletesNormalTask) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerWorkerContext &resolver = storage.contexts[1];
    resolver.is_resolver = 1;
    resolver.resolver_index = 0;
    resolver.resolver_count = 1;
    resolver.cluster_worker_ids[0] = 0;
    resolver.cluster_worker_ids[1] = 1;
    resolver.cluster_worker_ids[2] = 2;
    auto *slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &resolver, 0, 0);
    scheduler_initialize_free_slot(slot);
    slot->task_id = 0;
    slot->gang = 0;
    scheduler_gm_store(
        slot->publication, scheduler_dispatch_publication(slot->generation, SchedulerDispatchSlotState::READY)
    );
    auto *completion_line = scheduler_completion_inbox_at(storage.scheduler_state->base(), &resolver, 0);
    completion_line->completed_generations[0] = slot->generation;
    auto *control = scheduler_task_control_at(storage.scheduler_state->base(), &resolver, 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::READY);
    SchedulerWakeStats wake_stats{};
    SchedulerReadyStats ready_stats{};
    SchedulerCompletionStats completion_stats{};
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, &wake_stats, &ready_stats,
        &completion_stats
    ));
    EXPECT_EQ(completion_line->completed_generations[0], 0u);
    EXPECT_EQ(scheduler_dispatch_state(slot->publication), SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(control->wake_list_head, SCHEDULER_WAKE_LIST_CLOSED);
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
}

TEST(SchedulerClusterCompletion, CatchupRefreshIsBoundedToInitiallyEmptySibling) {
    EXPECT_EQ(scheduler_completion_catchup_mask(0), UINT32_C(0));
    EXPECT_EQ(scheduler_completion_catchup_mask(1), UINT32_C(2));
    EXPECT_EQ(scheduler_completion_catchup_mask(2), UINT32_C(1));
    EXPECT_EQ(scheduler_completion_catchup_mask(3), UINT32_C(0));
}

TEST(SchedulerClusterCompletion, DirectlyRefillsCompletedSlotWhenReadyTaskExists) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    SchedulerWorkerContext &resolver = storage.contexts[1];
    resolver.is_resolver = 1;
    resolver.resolver_index = 0;
    resolver.resolver_count = 1;
    resolver.inbox_index = 0;
    resolver.cluster_worker_ids[0] = 0;
    resolver.cluster_worker_ids[1] = 1;
    resolver.cluster_worker_ids[2] = 2;
    auto *callables =
        scheduler_state_at<uint64_t>(storage.scheduler_state->base(), storage.layout.callable_addresses_offset);
    callables[1] = 0x1000;

    auto *slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &resolver, 0, 0);
    scheduler_initialize_free_slot(slot);
    const uint32_t completed_generation = slot->generation;
    slot->task_id = 0;
    slot->subtask_slot = 0;
    slot->gang = 0;
    scheduler_gm_store(
        slot->publication, scheduler_dispatch_publication(completed_generation, SchedulerDispatchSlotState::READY)
    );
    auto *completion_line = scheduler_completion_inbox_at(storage.scheduler_state->base(), &resolver, 0);
    completion_line->completed_generations[0] = completed_generation;
    auto *completed_control = scheduler_task_control_at(storage.scheduler_state->base(), &resolver, 0);
    completed_control->state = static_cast<int64_t>(SchedulerTaskState::READY);
    auto *ready_control = scheduler_task_control_at(storage.scheduler_state->base(), &resolver, 1);
    ready_control->state = static_cast<int64_t>(SchedulerTaskState::READY);
    SchedulerReadyBatch batch{};
    SchedulerReadyStats ready_stats{};
    ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), &resolver, 1, &batch, &ready_stats));
    ASSERT_TRUE(scheduler_ready_batch_push(storage.scheduler_state->base(), &resolver, 0, 0, &batch, &ready_stats));

    SchedulerWakeStats wake_stats{};
    SchedulerCompletionStats completion_stats{};
    uint64_t ready_victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    uint64_t direct_refilled_slot_mask = 0;
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, &wake_stats, &ready_stats,
        &completion_stats, ready_victim_cursors, false, &direct_refilled_slot_mask
    ));

    EXPECT_EQ(completion_line->completed_generations[0], 0u);
    EXPECT_EQ(slot->task_id, 1);
    EXPECT_EQ(slot->generation, completed_generation + 1);
    EXPECT_EQ(scheduler_dispatch_state(slot->publication), SchedulerDispatchSlotState::READY);
    EXPECT_EQ(completed_control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
    EXPECT_EQ(direct_refilled_slot_mask, 1u);
}

TEST(SchedulerDeferredAiv, ReservesOnlyAvailableResolverSlotsBeforeClaiming) {
    FixtureStorage storage(3, 3);
    GraphBuffer graph(3);
    for (uint64_t task = 0; task < 3; ++task)
        graph.executable(task, 1);
    configure_normal_aiv_cluster(storage, 3);
    SchedulerWorkerContext &resolver = storage.contexts[1];
    occupy_normal_slot(storage, resolver, 2, 0, SCHEDULER_TASK_ID_INVALID);
    occupy_normal_slot(storage, resolver, 2, 1, SCHEDULER_TASK_ID_INVALID);
    enqueue_normal_aiv_tasks(storage, resolver, 0, 3);

    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    SchedulerReadyStats ready_stats{};
    SchedulerDeferredAivQueue deferred{};
    ASSERT_TRUE(scheduler_fill_cluster_normal_slots(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, victim_cursors, &ready_stats,
        false, 0, nullptr, &deferred
    ));

    ASSERT_EQ(deferred.count, SCHEDULER_PENDING_SLOT_COUNT);
    for (uint32_t slot_index = 0; slot_index < SCHEDULER_PENDING_SLOT_COUNT; ++slot_index) {
        auto *slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &resolver, 1, slot_index);
        EXPECT_EQ(scheduler_dispatch_state(slot->publication), SchedulerDispatchSlotState::FILLING);
        EXPECT_EQ(slot->task_id, SCHEDULER_TASK_ID_INVALID);
    }
    auto *ready_inbox = scheduler_ready_inbox_at(storage.scheduler_state->base(), &resolver, 1, 0);
    EXPECT_NE(ready_inbox->head, SCHEDULER_INBOX_EMPTY);
}

TEST(SchedulerDeferredAiv, DoesNotClaimWithoutResolverReservation) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    configure_normal_aiv_cluster(storage, 1);
    SchedulerWorkerContext &resolver = storage.contexts[1];
    for (uint64_t worker = 1; worker <= 2; ++worker) {
        for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot)
            occupy_normal_slot(storage, resolver, worker, slot, SCHEDULER_TASK_ID_INVALID);
    }
    enqueue_normal_aiv_tasks(storage, resolver, 0, 1);

    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    SchedulerReadyStats ready_stats{};
    SchedulerDeferredAivQueue deferred{};
    EXPECT_FALSE(scheduler_fill_cluster_normal_slots(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, victim_cursors, &ready_stats,
        false, 0, nullptr, &deferred
    ));
    EXPECT_EQ(deferred.count, 0u);
    auto *ready_inbox = scheduler_ready_inbox_at(storage.scheduler_state->base(), &resolver, 1, 0);
    EXPECT_EQ(ready_inbox->head, 0);
}

TEST(SchedulerDeferredAiv, KeepsReservationForResolverWhenNoPeerIsActive) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    configure_normal_aiv_cluster(storage, 1);
    SchedulerWorkerContext &resolver = storage.contexts[1];
    storage.contexts[2].active = 0;
    enqueue_normal_aiv_tasks(storage, resolver, 0, 1);

    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    SchedulerReadyStats ready_stats{};
    SchedulerDeferredAivQueue deferred{};
    ASSERT_TRUE(scheduler_fill_cluster_normal_slots(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, victim_cursors, &ready_stats,
        false, 0, nullptr, &deferred
    ));
    ASSERT_EQ(deferred.count, 1u);

    SchedulerWakeStats wake_stats{};
    SchedulerCompletionStats completion_stats{};
    EXPECT_TRUE(scheduler_drain_deferred_aiv_to_peer(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, &deferred, &wake_stats,
        &ready_stats, &completion_stats, false
    ));
    EXPECT_EQ(deferred.count, 1u);

    uint32_t self_slot = UINT32_MAX;
    ASSERT_TRUE(scheduler_publish_deferred_aiv_to_resolver(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, &deferred, false, &self_slot
    ));
    EXPECT_EQ(deferred.count, 0u);
    EXPECT_LT(self_slot, SCHEDULER_PENDING_SLOT_COUNT);
}

TEST(SchedulerDeferredAiv, PrefersNewPeerCapacityAndSelfPublishesOnlyOne) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    for (uint64_t task = 0; task < 2; ++task)
        graph.executable(task, 1);
    configure_normal_aiv_cluster(storage, 2);
    SchedulerWorkerContext &resolver = storage.contexts[1];
    occupy_normal_slot(storage, resolver, 2, 0, SCHEDULER_TASK_ID_INVALID);
    occupy_normal_slot(storage, resolver, 2, 1, SCHEDULER_TASK_ID_INVALID);
    enqueue_normal_aiv_tasks(storage, resolver, 0, 2);
    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    SchedulerReadyStats ready_stats{};
    SchedulerDeferredAivQueue deferred{};
    ASSERT_TRUE(scheduler_fill_cluster_normal_slots(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, victim_cursors, &ready_stats,
        false, 0, nullptr, &deferred
    ));
    ASSERT_EQ(deferred.count, 2u);
    const SchedulerFreeSlotClaim first_reservation = deferred.entries[0].reserved_slot;

    auto *peer_slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &resolver, 2, 0);
    peer_slot->task_id = SCHEDULER_TASK_ID_INVALID;
    scheduler_gm_store(
        peer_slot->publication, scheduler_dispatch_publication(peer_slot->generation, SchedulerDispatchSlotState::FREE)
    );
    SchedulerWakeStats wake_stats{};
    SchedulerCompletionStats completion_stats{};
    ASSERT_TRUE(scheduler_drain_deferred_aiv_to_peer(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, &deferred, &wake_stats,
        &ready_stats, &completion_stats, false
    ));
    ASSERT_EQ(deferred.count, 1u);
    EXPECT_EQ(scheduler_dispatch_state(peer_slot->publication), SchedulerDispatchSlotState::READY);
    auto *released = scheduler_dispatch_slot_at(
        storage.scheduler_state->base(), &resolver, first_reservation.worker_id, first_reservation.slot_index
    );
    EXPECT_EQ(scheduler_dispatch_state(released->publication), SchedulerDispatchSlotState::FREE);

    uint32_t self_slot = UINT32_MAX;
    ASSERT_TRUE(scheduler_publish_deferred_aiv_to_resolver(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, &deferred, false, &self_slot
    ));
    EXPECT_EQ(deferred.count, 0u);
    ASSERT_LT(self_slot, SCHEDULER_PENDING_SLOT_COUNT);
    auto *published = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &resolver, 1, self_slot);
    EXPECT_EQ(scheduler_dispatch_state(published->publication), SchedulerDispatchSlotState::READY);
}

TEST(SchedulerDeferredAiv, KeepsSecondReservationAfterOneSelfPublish) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    for (uint64_t task = 0; task < 2; ++task)
        graph.executable(task, 1);
    configure_normal_aiv_cluster(storage, 2);
    SchedulerWorkerContext &resolver = storage.contexts[1];
    occupy_normal_slot(storage, resolver, 2, 0, SCHEDULER_TASK_ID_INVALID);
    occupy_normal_slot(storage, resolver, 2, 1, SCHEDULER_TASK_ID_INVALID);
    enqueue_normal_aiv_tasks(storage, resolver, 0, 2);
    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    SchedulerReadyStats ready_stats{};
    SchedulerDeferredAivQueue deferred{};
    ASSERT_TRUE(scheduler_fill_cluster_normal_slots(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, victim_cursors, &ready_stats,
        false, 0, nullptr, &deferred
    ));
    ASSERT_EQ(deferred.count, 2u);

    uint32_t self_slot = UINT32_MAX;
    ASSERT_TRUE(scheduler_publish_deferred_aiv_to_resolver(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, &deferred, false, &self_slot
    ));
    ASSERT_EQ(deferred.count, 1u);
    auto *published = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &resolver, 1, self_slot);
    EXPECT_EQ(scheduler_dispatch_state(published->publication), SchedulerDispatchSlotState::READY);
    const SchedulerFreeSlotClaim remaining = deferred.entries[0].reserved_slot;
    auto *reserved = scheduler_dispatch_slot_at(
        storage.scheduler_state->base(), &resolver, remaining.worker_id, remaining.slot_index
    );
    EXPECT_EQ(scheduler_dispatch_state(reserved->publication), SchedulerDispatchSlotState::FILLING);
    EXPECT_EQ(reserved->task_id, SCHEDULER_TASK_ID_INVALID);
}

TEST(SchedulerDeferredAiv, RetiresCompletedPeerAndRefillsWithoutFreeDecision) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    graph.executable(0, 1);
    graph.executable(1, 1);
    configure_normal_aiv_cluster(storage, 2);
    SchedulerWorkerContext &resolver = storage.contexts[1];
    occupy_normal_slot(storage, resolver, 2, 0, 0);
    occupy_normal_slot(storage, resolver, 2, 1, SCHEDULER_TASK_ID_INVALID);
    occupy_normal_slot(storage, resolver, 1, 1, SCHEDULER_TASK_ID_INVALID);
    auto *completed_control = scheduler_task_control_at(storage.scheduler_state->base(), &resolver, 0);
    completed_control->state = static_cast<int64_t>(SchedulerTaskState::READY);
    enqueue_normal_aiv_tasks(storage, resolver, 1, 2);

    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    SchedulerReadyStats ready_stats{};
    SchedulerDeferredAivQueue deferred{};
    ASSERT_TRUE(scheduler_fill_cluster_normal_slots(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, victim_cursors, &ready_stats,
        false, 0, nullptr, &deferred
    ));
    ASSERT_EQ(deferred.count, 1u);
    auto *peer_slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &resolver, 2, 0);
    const uint32_t completed_generation = peer_slot->generation;
    auto *completion_line = scheduler_completion_inbox_at(storage.scheduler_state->base(), &resolver, 2);
    completion_line->completed_generations[0] = completed_generation;

    SchedulerWakeStats wake_stats{};
    SchedulerCompletionStats completion_stats{};
    ASSERT_TRUE(scheduler_drain_deferred_aiv_to_peer(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, &deferred, &wake_stats,
        &ready_stats, &completion_stats, false
    ));
    EXPECT_EQ(deferred.count, 0u);
    EXPECT_EQ(completion_line->completed_generations[0], 0u);
    EXPECT_EQ(completed_control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
    EXPECT_EQ(peer_slot->task_id, 1);
    EXPECT_EQ(peer_slot->generation, completed_generation + 1);
    EXPECT_EQ(scheduler_dispatch_state(peer_slot->publication), SchedulerDispatchSlotState::READY);
}

TEST(SchedulerDeferredAiv, ResolverCompletionDoesNotDirectRefillItself) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    graph.executable(0, 1);
    graph.executable(1, 1);
    configure_normal_aiv_cluster(storage, 2);
    SchedulerWorkerContext &resolver = storage.contexts[1];
    occupy_normal_slot(storage, resolver, 1, 0, 0);
    auto *completed_slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &resolver, 1, 0);
    auto *completion_line = scheduler_completion_inbox_at(storage.scheduler_state->base(), &resolver, 1);
    completion_line->completed_generations[0] = completed_slot->generation;
    auto *completed_control = scheduler_task_control_at(storage.scheduler_state->base(), &resolver, 0);
    completed_control->state = static_cast<int64_t>(SchedulerTaskState::READY);
    enqueue_normal_aiv_tasks(storage, resolver, 1, 2);

    SchedulerWakeStats wake_stats{};
    SchedulerReadyStats ready_stats{};
    SchedulerCompletionStats completion_stats{};
    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    uint64_t direct_refilled_slot_mask = 0;
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, &wake_stats, &ready_stats,
        &completion_stats, victim_cursors, false, &direct_refilled_slot_mask
    ));
    EXPECT_EQ(direct_refilled_slot_mask, 0u);
    EXPECT_EQ(completed_slot->task_id, SCHEDULER_TASK_ID_INVALID);
    EXPECT_EQ(scheduler_dispatch_state(completed_slot->publication), SchedulerDispatchSlotState::FREE);
    auto *ready_inbox = scheduler_ready_inbox_at(storage.scheduler_state->base(), &resolver, 1, 0);
    EXPECT_EQ(ready_inbox->head, 1);
}

}  // namespace
