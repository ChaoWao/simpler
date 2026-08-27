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
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

#include "scheduler/scheduler_dispatch.h"
#include "scheduler/scheduler_topology.h"
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

TEST(SchedulerGangPriority, SyncStartPrecedesMixAndSpmd) {
    EXPECT_EQ(scheduler_task_priority_bit(SCHEDULER_TASK_SPMD), 4u);
    EXPECT_EQ(scheduler_task_priority_bit(SCHEDULER_TASK_MIX), 2u);
    EXPECT_EQ(scheduler_task_priority_bit(SCHEDULER_TASK_MIX | SCHEDULER_TASK_SYNC_START), 1u);

    FixtureStorage storage(3, 3);
    GraphBuffer graph(3);
    graph.executable(0, 0);
    graph.mixed(1, 3);
    graph.executable(2, 1);
    auto *coordinator = scheduler_state_at<SchedulerGangCoordinator>(
        storage.scheduler_state->base(), storage.layout.gang_coordinator_offset
    );
    coordinator->gang_task_count = 3;
    coordinator->resolver_count = 1;
    coordinator->ready_priority_bits = 7;
    storage.run_control->resolver_count = 1;
    SchedulerWorkerContext &resolver = storage.contexts[1];
    resolver.is_resolver = 1;
    resolver.resolver_index = 0;
    resolver.resolver_count = 1;
    resolver.cluster_worker_ids[0] = 0;
    resolver.cluster_worker_ids[1] = 1;
    resolver.cluster_worker_ids[2] = 2;
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    for (uint32_t task = 0; task < 3; ++task)
        controls[task].state = static_cast<int64_t>(SchedulerTaskState::READY);
    storage.metadata[0].active_mask = 1;
    storage.metadata[0].logical_block_num = 4;
    storage.metadata[0].total_required_subtasks = 4;
    storage.metadata[0].flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD;
    storage.metadata[1].active_mask = 3;
    storage.metadata[1].logical_block_num = 1;
    storage.metadata[1].total_required_subtasks = 2;
    storage.metadata[1].flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_MIX;
    storage.metadata[2].active_mask = 2;
    storage.metadata[2].logical_block_num = 2;
    storage.metadata[2].total_required_subtasks = 2;
    storage.metadata[2].flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_SYNC_START;

    ASSERT_TRUE(
        scheduler_gang_admit_one(graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control)
    );
    auto *cohort =
        scheduler_state_at<SchedulerGangCohort>(storage.scheduler_state->base(), storage.layout.gang_cohorts_offset);
    EXPECT_EQ(cohort->task_id, 2);
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(SchedulerGangCohortState::DRAINING));
    EXPECT_EQ(controls[2].state, static_cast<int64_t>(SchedulerTaskState::DISPATCHING));
    EXPECT_FALSE(
        scheduler_gang_admit_one(graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control)
    );
}

TEST(SchedulerSyncStart, RejectsImpossibleCohortBeforeDrain) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    SchedulerWorkerContext &resolver = storage.contexts[1];
    resolver.is_resolver = 1;
    resolver.resolver_index = 0;
    resolver.resolver_count = 1;
    resolver.cluster_worker_ids[0] = 0;
    resolver.cluster_worker_ids[1] = 1;
    resolver.cluster_worker_ids[2] = 2;
    storage.run_control->resolver_count = 1;
    storage.metadata[0].kernel_ids[1] = 1;
    storage.metadata[0].active_mask = 2;
    storage.metadata[0].logical_block_num = 3;
    storage.metadata[0].total_required_subtasks = 3;
    storage.metadata[0].flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_SYNC_START;
    auto *control = scheduler_task_control_at(storage.scheduler_state->base(), &resolver, 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::READY);
    auto *coordinator = scheduler_gang_coordinator_at(storage.scheduler_state->base(), &resolver);
    coordinator->gang_task_count = 1;
    coordinator->resolver_count = 1;
    coordinator->ready_priority_bits = 1;

    EXPECT_FALSE(
        scheduler_gang_admit_one(graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control)
    );
    EXPECT_EQ(coordinator->capacity_reject_count, 1u);
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::UNSUPPORTED_SHAPE));
    EXPECT_EQ(storage.run_control->error_reserved[0], 72u);
    EXPECT_EQ(coordinator->active_dispatch_cohort, UINT64_MAX);
}

TEST(SchedulerGangTokens, StaleGenerationCannotCompleteTree) {
    FixtureStorage storage(1, 3);
    SchedulerWorkerContext &resolver = storage.contexts[0];
    resolver.resolver_index = 0;
    resolver.resolver_count = 3;
    constexpr uint64_t kGeneration = 9;
    for (uint64_t participant_index = 0; participant_index < 3; ++participant_index) {
        auto *participant =
            scheduler_gang_participant_at(storage.scheduler_state->base(), &resolver, 0, participant_index);
        participant->dispatch_local_token = kGeneration;
    }
    auto *stale_child = scheduler_gang_participant_at(storage.scheduler_state->base(), &resolver, 0, 2);
    stale_child->dispatch_local_token = kGeneration - 1;
    resolver.resolver_index = 1;
    EXPECT_TRUE(scheduler_gang_update_subtree_token(
        storage.scheduler_state->base(), &resolver, 0, 3, kGeneration, SchedulerGangTokenPhase::DISPATCH
    ));
    resolver.resolver_index = 2;
    EXPECT_FALSE(scheduler_gang_update_subtree_token(
        storage.scheduler_state->base(), &resolver, 0, 3, kGeneration, SchedulerGangTokenPhase::DISPATCH
    ));
    resolver.resolver_index = 0;
    EXPECT_FALSE(scheduler_gang_update_subtree_token(
        storage.scheduler_state->base(), &resolver, 0, 3, kGeneration, SchedulerGangTokenPhase::DISPATCH
    ));
    stale_child->dispatch_local_token = kGeneration;
    resolver.resolver_index = 2;
    EXPECT_TRUE(scheduler_gang_update_subtree_token(
        storage.scheduler_state->base(), &resolver, 0, 3, kGeneration, SchedulerGangTokenPhase::DISPATCH
    ));
    resolver.resolver_index = 0;
    EXPECT_TRUE(scheduler_gang_update_subtree_token(
        storage.scheduler_state->base(), &resolver, 0, 3, kGeneration, SchedulerGangTokenPhase::DISPATCH
    ));
}

TEST(SchedulerGangMix, ReservesWholeClusterOrNothing) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.mixed(0, 7);
    SchedulerWorkerContext &resolver = storage.contexts[1];
    resolver.is_resolver = 1;
    resolver.resolver_index = 0;
    resolver.resolver_count = 1;
    resolver.cluster_worker_ids[0] = 0;
    resolver.cluster_worker_ids[1] = 1;
    resolver.cluster_worker_ids[2] = 2;
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    for (uint32_t worker = 0; worker < 3; ++worker) {
        for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot)
            scheduler_initialize_free_slot(
                scheduler_dispatch_slot_at(storage.scheduler_state->base(), &resolver, worker, pending_slot)
            );
    }
    storage.metadata[0].active_mask = 7;
    storage.metadata[0].kernel_ids[0] = 1;
    storage.metadata[0].kernel_ids[1] = 1;
    storage.metadata[0].kernel_ids[2] = 1;
    storage.metadata[0].logical_block_num = 1;
    storage.metadata[0].total_required_subtasks = 3;
    storage.metadata[0].flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_MIX;
    auto *callables =
        scheduler_state_at<uint64_t>(storage.scheduler_state->base(), storage.layout.callable_addresses_offset);
    callables[1] = 0x1000;
    auto *participant = scheduler_gang_participant_at(storage.scheduler_state->base(), &resolver, 0, 0);
    participant->config_generation = 1;
    participant->task_id = 0;
    participant->active_mask = 7;
    participant->logical_block_num = 1;
    participant->local_expected_subtasks = 3;
    auto *blocked0 = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &resolver, 1, 0);
    auto *blocked1 = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &resolver, 2, 1);
    scheduler_gm_store(
        blocked0->publication, scheduler_dispatch_publication(blocked0->generation, SchedulerDispatchSlotState::READY)
    );
    scheduler_gm_store(
        blocked1->publication, scheduler_dispatch_publication(blocked1->generation, SchedulerDispatchSlotState::READY)
    );
    EXPECT_FALSE(scheduler_gang_fill_mix_block(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, participant, 0, 0,
        SchedulerDispatchSlotState::GATED
    ));
    EXPECT_EQ(participant->local_published_subtasks, 0u);
    scheduler_gm_store(
        blocked0->publication, scheduler_dispatch_publication(blocked0->generation, SchedulerDispatchSlotState::FREE)
    );
    scheduler_gm_store(
        blocked1->publication, scheduler_dispatch_publication(blocked1->generation, SchedulerDispatchSlotState::FREE)
    );
    ASSERT_TRUE(scheduler_gang_fill_mix_block(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, participant, 0, 0,
        SchedulerDispatchSlotState::GATED
    ));
    EXPECT_EQ(participant->local_published_subtasks, 3u);
    for (uint32_t worker = 0; worker < 3; ++worker) {
        auto *slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &resolver, worker, 0);
        EXPECT_EQ(scheduler_dispatch_state(slot->publication), SchedulerDispatchSlotState::GATED);
        EXPECT_EQ(slot->block_idx, 0u);
        EXPECT_EQ(slot->block_num, 1u);
    }
}

TEST(SchedulerSyncStart, DrainsStagesAndReleasesBeforeCompletion) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    SchedulerWorkerContext &resolver = storage.contexts[1];
    resolver.is_resolver = 1;
    resolver.resolver_index = 0;
    resolver.resolver_count = 1;
    resolver.cluster_worker_ids[0] = 0;
    resolver.cluster_worker_ids[1] = 1;
    resolver.cluster_worker_ids[2] = 2;
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.run_control->resolver_count = 1;
    for (uint32_t worker = 0; worker < 3; ++worker) {
        for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot)
            scheduler_initialize_free_slot(
                scheduler_dispatch_slot_at(storage.scheduler_state->base(), &resolver, worker, pending_slot)
            );
    }
    storage.metadata[0].kernel_ids[1] = 1;
    storage.metadata[0].active_mask = 2;
    storage.metadata[0].logical_block_num = 2;
    storage.metadata[0].total_required_subtasks = 2;
    storage.metadata[0].flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_SYNC_START;
    auto *callables =
        scheduler_state_at<uint64_t>(storage.scheduler_state->base(), storage.layout.callable_addresses_offset);
    callables[1] = 0x1000;
    auto *control = scheduler_task_control_at(storage.scheduler_state->base(), &resolver, 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::READY);
    auto *coordinator = scheduler_gang_coordinator_at(storage.scheduler_state->base(), &resolver);
    coordinator->gang_task_count = 1;
    coordinator->resolver_count = 1;
    coordinator->ready_priority_bits = 1;
    SchedulerWakeStats wake_stats{};
    SchedulerReadyStats ready_stats{};
    SchedulerCompletionStats completion_stats{};
    ASSERT_TRUE(
        scheduler_gang_admit_one(graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control)
    );
    auto *cohort = scheduler_gang_cohort_at(storage.scheduler_state->base(), &resolver, 0);
    ASSERT_TRUE(scheduler_service_gang(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, &wake_stats, &ready_stats,
        &completion_stats
    ));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(SchedulerGangCohortState::STAGING));
    ASSERT_TRUE(scheduler_service_gang(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, &wake_stats, &ready_stats,
        &completion_stats
    ));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(SchedulerGangCohortState::RELEASING));
    ASSERT_TRUE(scheduler_service_gang(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, &wake_stats, &ready_stats,
        &completion_stats
    ));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(SchedulerGangCohortState::EXECUTING));
    EXPECT_EQ(coordinator->active_dispatch_cohort, UINT64_MAX);
    for (uint32_t worker = 1; worker <= 2; ++worker) {
        auto *slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &resolver, worker, 0);
        ASSERT_EQ(scheduler_dispatch_state(slot->publication), SchedulerDispatchSlotState::READY);
        scheduler_completion_inbox_at(storage.scheduler_state->base(), &resolver, worker)->completed_generations[0] =
            slot->generation;
    }
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, &wake_stats, &ready_stats,
        &completion_stats
    ));
    ASSERT_TRUE(scheduler_service_gang(
        graph.graph(), storage.scheduler_state->base(), &resolver, storage.run_control, &wake_stats, &ready_stats,
        &completion_stats
    ));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(SchedulerGangCohortState::FREE));
    EXPECT_EQ(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
}

}  // namespace
