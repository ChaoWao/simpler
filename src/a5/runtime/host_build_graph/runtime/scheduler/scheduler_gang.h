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

#pragma once

#include "scheduler_ready.h"

enum class SchedulerGangTokenPhase : uint32_t {
    DRAIN = 0,
    STAGE = 1,
    DISPATCH = 2,
    COMPLETION = 3,
};

inline __host__ __aicore__ __gm__ SchedulerGangCohort *scheduler_gang_cohort_at(
    __gm__ void *scheduler_state_base, __gm__ const SchedulerWorkerContext *context, uint32_t cohort_index
) {
    return scheduler_state_at<SchedulerGangCohort>(
        scheduler_state_base,
        context->gang_cohorts_offset + static_cast<uint64_t>(cohort_index) * sizeof(SchedulerGangCohort)
    );
}

inline __host__ __aicore__ __gm__ SchedulerGangParticipant *scheduler_gang_participant_at(
    __gm__ void *scheduler_state_base, __gm__ const SchedulerWorkerContext *context, uint32_t cohort_index,
    uint64_t resolver_index
) {
    const uint64_t linear = static_cast<uint64_t>(cohort_index) * SCHEDULER_CLUSTER_CAPACITY + resolver_index;
    return scheduler_state_at<SchedulerGangParticipant>(
        scheduler_state_base, context->gang_participants_offset + linear * sizeof(SchedulerGangParticipant)
    );
}

inline __host__ __aicore__ __gm__ SchedulerGangCommand *scheduler_gang_command_at(
    __gm__ void *scheduler_state_base, __gm__ const SchedulerWorkerContext *context, uint64_t resolver_index
) {
    return scheduler_state_at<SchedulerGangCommand>(
        scheduler_state_base, context->gang_commands_offset + resolver_index * sizeof(SchedulerGangCommand)
    );
}

inline __aicore__ void scheduler_gang_publish_command(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *resolver, uint64_t resolver_index,
    uint32_t cohort_index, uint64_t generation, SchedulerGangCohortState state
) {
    __gm__ SchedulerGangCommand *command = scheduler_gang_command_at(scheduler_state_base, resolver, resolver_index);
    command->generation[cohort_index] = generation;
    command->state[cohort_index] = static_cast<uint64_t>(state);
    scheduler_publish_cache_line(command);
}

inline __aicore__ bool scheduler_gang_forward_command(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *resolver,
    __gm__ SchedulerGangParticipant *participant, uint32_t cohort_index, uint64_t generation,
    SchedulerGangCohortState state
) {
    if (participant->forwarded_generation == generation && participant->forwarded_state == static_cast<uint32_t>(state))
        return false;
    const uint64_t children[2] = {resolver->resolver_index * 2 + 1, resolver->resolver_index * 2 + 2};
    for (uint32_t child_slot = 0; child_slot < 2; ++child_slot) {
        if (children[child_slot] >= participant->participant_count) continue;
        scheduler_gang_publish_command(
            scheduler_state_base, resolver, children[child_slot], cohort_index, generation, state
        );
    }
    participant->forwarded_generation = generation;
    participant->forwarded_state = static_cast<uint32_t>(state);
    scheduler_publish_cache_line(participant);
    return true;
}

inline __host__ __aicore__ uint32_t scheduler_gang_popcount(uint32_t mask) {
    return static_cast<uint32_t>(__builtin_popcount(mask));
}

inline __aicore__ __gm__ volatile uint64_t *
scheduler_gang_local_token(__gm__ SchedulerGangParticipant *participant, SchedulerGangTokenPhase phase) {
    if (phase == SchedulerGangTokenPhase::DRAIN) return &participant->drain_local_token;
    if (phase == SchedulerGangTokenPhase::STAGE) return &participant->stage_local_token;
    if (phase == SchedulerGangTokenPhase::DISPATCH) return &participant->dispatch_local_token;
    return &participant->completion_local_token;
}

inline __aicore__ __gm__ volatile uint64_t *
scheduler_gang_subtree_token(__gm__ SchedulerGangParticipant *participant, SchedulerGangTokenPhase phase) {
    if (phase == SchedulerGangTokenPhase::DRAIN) return &participant->drain_subtree_token;
    if (phase == SchedulerGangTokenPhase::STAGE) return &participant->stage_subtree_token;
    if (phase == SchedulerGangTokenPhase::DISPATCH) return &participant->dispatch_subtree_token;
    return &participant->completion_subtree_token;
}

inline __aicore__ bool scheduler_gang_publish_local_token(
    __gm__ SchedulerGangParticipant *participant, SchedulerGangTokenPhase phase, uint64_t generation
) {
    __gm__ volatile uint64_t *token = scheduler_gang_local_token(participant, phase);
    if (*token == generation) return false;
    *token = generation;
    scheduler_publish_cache_line(token);
    return true;
}

inline __aicore__ bool scheduler_gang_update_subtree_token(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *resolver, uint32_t cohort_index,
    uint64_t participant_count, uint64_t generation, SchedulerGangTokenPhase phase
) {
    const uint64_t resolver_index = resolver->resolver_index;
    if (resolver_index >= participant_count) return false;
    __gm__ SchedulerGangParticipant *participant =
        scheduler_gang_participant_at(scheduler_state_base, resolver, cohort_index, resolver_index);
    __gm__ volatile uint64_t *local_token = scheduler_gang_local_token(participant, phase);
    __gm__ volatile uint64_t *subtree_token = scheduler_gang_subtree_token(participant, phase);
    if (*subtree_token == generation) return true;
    scheduler_observe_cache_line(local_token);
    if (*local_token != generation) return false;
    const uint64_t children[2] = {resolver_index * 2 + 1, resolver_index * 2 + 2};
    for (uint32_t child_slot = 0; child_slot < 2; ++child_slot) {
        if (children[child_slot] >= participant_count) continue;
        __gm__ SchedulerGangParticipant *child =
            scheduler_gang_participant_at(scheduler_state_base, resolver, cohort_index, children[child_slot]);
        __gm__ volatile uint64_t *child_token = scheduler_gang_subtree_token(child, phase);
        scheduler_observe_cache_line(child_token);
        if (*child_token != generation) return false;
    }
    *subtree_token = generation;
    scheduler_publish_cache_line(subtree_token);
    return true;
}

inline __aicore__ bool scheduler_gang_root_token_ready(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *resolver, uint32_t cohort_index,
    uint64_t generation, SchedulerGangTokenPhase phase
) {
    __gm__ SchedulerGangParticipant *root =
        scheduler_gang_participant_at(scheduler_state_base, resolver, cohort_index, 0);
    __gm__ volatile uint64_t *token = scheduler_gang_subtree_token(root, phase);
    scheduler_observe_cache_line(token);
    return *token == generation;
}

inline __host__ __aicore__ uint64_t scheduler_gang_retire_token(uint64_t generation) { return ~generation; }

inline __aicore__ void scheduler_gang_publish_retire(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *resolver,
    __gm__ SchedulerGangParticipant *participant, uint32_t cohort_index, uint64_t generation
) {
    const uint64_t retire_token = scheduler_gang_retire_token(generation);
    scheduler_gang_publish_local_token(participant, SchedulerGangTokenPhase::COMPLETION, retire_token);
    (void)scheduler_gang_update_subtree_token(
        scheduler_state_base, resolver, cohort_index, participant->participant_count, retire_token,
        SchedulerGangTokenPhase::COMPLETION
    );
}

inline __aicore__ bool scheduler_fill_explicit_dispatch_slot(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *resolver,
    __gm__ SchedulerRunControl *run_control, uint64_t worker_id, uint32_t pending_slot, int64_t task_id,
    uint8_t subtask_slot, uint32_t block_idx, uint32_t block_num, uint32_t cohort_index, uint32_t cohort_generation,
    SchedulerDispatchSlotState publication
) {
    if (worker_id >= resolver->runtime_worker_count || pending_slot >= SCHEDULER_PENDING_SLOT_COUNT || task_id < 0 ||
        static_cast<uint64_t>(task_id) >= graph.task_count || subtask_slot >= 3 || block_num == 0 ||
        block_idx >= block_num) {
        return false;
    }
    __gm__ SchedulerTaskMetadata *metadata = scheduler_task_metadata_at(scheduler_state_base, resolver, task_id);
    scheduler_observe_cache_line(metadata);
    if ((metadata->active_mask & (1U << subtask_slot)) == 0) return false;
    const uint16_t kernel_id = metadata->kernel_ids[subtask_slot];
    __gm__ uint64_t *callable_addresses =
        scheduler_state_at<uint64_t>(scheduler_state_base, resolver->callable_addresses_offset);
    scheduler_observe_cache_line(&callable_addresses[kernel_id]);
    const uint64_t callable_address = callable_addresses[kernel_id];
    if (kernel_id == UINT16_MAX || callable_address == 0) {
        scheduler_record_error(
            run_control, task_id, SchedulerGraphResult::INVALID_CALLABLE, &graph, resolver, UINT64_C(70)
        );
        return false;
    }

    __gm__ SchedulerWorkerContext *target = scheduler_worker_context_at(scheduler_state_base, resolver, worker_id);
    scheduler_observe_cache_line(target);
    __gm__ SchedulerDispatchSlot *slot =
        scheduler_dispatch_slot_at(scheduler_state_base, resolver, worker_id, pending_slot);
    scheduler_observe_cache_line(slot);
    uint32_t generation = slot->generation + 1;
    if (generation == 0) generation = 1;
    slot->task_id = task_id;
    slot->ready_inbox_index = resolver->resolver_index;
    slot->claim_start_cycles = 0;
    slot->claim_end_cycles = 0;
    slot->claim_worker_id = resolver->worker_index;
    slot->kernel_id = kernel_id;
    slot->subtask_slot = subtask_slot;
    slot->has_fanin = scheduler_task_has_fanin(metadata->flags) ? 1 : 0;
    slot->ready_source = static_cast<uint8_t>(SchedulerReadySource::LOCAL);
    slot->pending_slot = static_cast<uint8_t>(pending_slot);
    slot->block_num = static_cast<uint16_t>(block_num);
    slot->generation = generation;
    slot->block_idx = block_idx;
    slot->cohort_generation = cohort_generation;
    slot->cohort_index = static_cast<uint8_t>(cohort_index);
    slot->gang = 1;
    scheduler_writeback_cache_line(slot);

    SchedulerTaskInfo task{
        task_id,
        static_cast<int32_t>(kernel_id),
        static_cast<int32_t>(subtask_slot),
        subtask_slot == 0 ? CoreType::AIC : CoreType::AIV,
    };
    __gm__ DispatchPayload *payload = scheduler_state_at<DispatchPayload>(
        scheduler_state_base,
        target->dispatch_payload_offset + static_cast<uint64_t>(pending_slot) * sizeof(DispatchPayload)
    );
    SchedulerGraphResult status = scheduler_materialize_task_payload_resolved(
        graph, task, callable_address, payload, static_cast<int32_t>(block_idx), static_cast<int32_t>(block_num)
    );
    if (status != SchedulerGraphResult::OK) {
        scheduler_record_error(run_control, task_id, status, &graph, resolver, UINT64_C(71));
        return false;
    }
    scheduler_publish_dispatch_payload(payload);
    scheduler_gm_store(slot->publication, scheduler_dispatch_publication(generation, publication));
    return true;
}

inline __aicore__ int32_t scheduler_gang_find_free_pending_slot(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *resolver, uint64_t worker_id
) {
    for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
        __gm__ SchedulerDispatchSlot *slot =
            scheduler_dispatch_slot_at(scheduler_state_base, resolver, worker_id, pending_slot);
        if (scheduler_dispatch_state(scheduler_gm_query(slot->publication)) == SchedulerDispatchSlotState::FREE)
            return static_cast<int32_t>(pending_slot);
    }
    return -1;
}

inline __aicore__ bool scheduler_gang_fill_single(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *resolver,
    __gm__ SchedulerRunControl *run_control, __gm__ SchedulerGangParticipant *participant, uint32_t cohort_index,
    uint64_t worker_id, uint8_t subtask_slot, uint32_t block_idx, SchedulerDispatchSlotState publication
) {
    const int32_t pending_slot = scheduler_gang_find_free_pending_slot(scheduler_state_base, resolver, worker_id);
    if (pending_slot < 0) return false;
    __gm__ SchedulerDispatchSlot *slot =
        scheduler_dispatch_slot_at(scheduler_state_base, resolver, worker_id, static_cast<uint32_t>(pending_slot));
    const uint32_t generation = scheduler_dispatch_generation(scheduler_gm_query(slot->publication));
    scheduler_gm_store(
        slot->publication, scheduler_dispatch_publication(generation, SchedulerDispatchSlotState::FILLING)
    );
    if (!scheduler_fill_explicit_dispatch_slot(
            graph, scheduler_state_base, resolver, run_control, worker_id, static_cast<uint32_t>(pending_slot),
            participant->task_id, subtask_slot, block_idx, participant->logical_block_num, cohort_index,
            static_cast<uint32_t>(participant->config_generation), publication
        )) {
        return false;
    }
    ++participant->local_published_subtasks;
    return true;
}

inline __aicore__ bool scheduler_gang_fill_mix_block(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *resolver,
    __gm__ SchedulerRunControl *run_control, __gm__ SchedulerGangParticipant *participant, uint32_t cohort_index,
    uint32_t block_idx, SchedulerDispatchSlotState publication
) {
    int32_t pending_slot = -1;
    for (uint32_t candidate = 0; candidate < SCHEDULER_PENDING_SLOT_COUNT; ++candidate) {
        bool all_free = true;
        for (uint8_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
            if ((participant->active_mask & (1U << subtask_slot)) == 0) continue;
            const uint64_t worker_id = resolver->cluster_worker_ids[subtask_slot];
            __gm__ SchedulerDispatchSlot *slot =
                scheduler_dispatch_slot_at(scheduler_state_base, resolver, worker_id, candidate);
            if (scheduler_dispatch_state(scheduler_gm_query(slot->publication)) != SchedulerDispatchSlotState::FREE) {
                all_free = false;
                break;
            }
        }
        if (all_free) {
            pending_slot = static_cast<int32_t>(candidate);
            break;
        }
    }
    if (pending_slot < 0) return false;
    for (uint8_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
        if ((participant->active_mask & (1U << subtask_slot)) == 0) continue;
        const uint64_t worker_id = resolver->cluster_worker_ids[subtask_slot];
        __gm__ SchedulerDispatchSlot *slot =
            scheduler_dispatch_slot_at(scheduler_state_base, resolver, worker_id, static_cast<uint32_t>(pending_slot));
        const uint32_t generation = scheduler_dispatch_generation(scheduler_gm_query(slot->publication));
        scheduler_gm_store(
            slot->publication, scheduler_dispatch_publication(generation, SchedulerDispatchSlotState::FILLING)
        );
    }
    for (uint8_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
        if ((participant->active_mask & (1U << subtask_slot)) == 0) continue;
        if (!scheduler_fill_explicit_dispatch_slot(
                graph, scheduler_state_base, resolver, run_control, resolver->cluster_worker_ids[subtask_slot],
                static_cast<uint32_t>(pending_slot), participant->task_id, subtask_slot, block_idx,
                participant->logical_block_num, cohort_index, static_cast<uint32_t>(participant->config_generation),
                SchedulerDispatchSlotState::GATED
            )) {
            return false;
        }
    }
    if (publication == SchedulerDispatchSlotState::READY) {
        for (uint8_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
            if ((participant->active_mask & (1U << subtask_slot)) == 0) continue;
            __gm__ SchedulerDispatchSlot *slot = scheduler_dispatch_slot_at(
                scheduler_state_base, resolver, resolver->cluster_worker_ids[subtask_slot],
                static_cast<uint32_t>(pending_slot)
            );
            scheduler_gm_store(
                slot->publication, scheduler_dispatch_publication(slot->generation, SchedulerDispatchSlotState::READY)
            );
        }
    }
    participant->local_published_subtasks += scheduler_gang_popcount(participant->active_mask);
    return true;
}

inline __aicore__ bool scheduler_gang_local_slots_drained(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *resolver, uint32_t active_mask
) {
    const bool single_aiv = scheduler_gang_popcount(active_mask) == 1 && (active_mask & 6U) != 0;
    for (uint32_t cluster_lane = 0; cluster_lane < 3; ++cluster_lane) {
        const bool used =
            cluster_lane == 0 ? (active_mask & 1U) != 0 : (single_aiv || (active_mask & (1U << cluster_lane)) != 0);
        if (!used) continue;
        for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
            __gm__ SchedulerDispatchSlot *slot = scheduler_dispatch_slot_at(
                scheduler_state_base, resolver, resolver->cluster_worker_ids[cluster_lane], pending_slot
            );
            if (scheduler_dispatch_state(scheduler_gm_query(slot->publication)) != SchedulerDispatchSlotState::FREE)
                return false;
        }
    }
    return true;
}

inline __aicore__ void scheduler_gang_fill_participant(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *resolver,
    __gm__ SchedulerRunControl *run_control, __gm__ SchedulerGangParticipant *participant, uint32_t cohort_index,
    SchedulerDispatchSlotState publication
) {
    const uint32_t active_mask = participant->active_mask;
    if (scheduler_gang_popcount(active_mask) > 1) {
        while (participant->next_block[0] < participant->logical_block_num) {
            const uint32_t block_idx = participant->next_block[0];
            if (!scheduler_gang_fill_mix_block(
                    graph, scheduler_state_base, resolver, run_control, participant, cohort_index, block_idx,
                    publication
                ))
                break;
            participant->next_block[0] += participant->block_stride;
        }
    } else if ((active_mask & 1U) != 0) {
        while (participant->next_block[0] < participant->logical_block_num) {
            const uint32_t block_idx = participant->next_block[0];
            if (!scheduler_gang_fill_single(
                    graph, scheduler_state_base, resolver, run_control, participant, cohort_index,
                    resolver->cluster_worker_ids[0], 0, block_idx, publication
                ))
                break;
            participant->next_block[0] += participant->block_stride;
        }
    } else {
        const uint8_t subtask_slot = (active_mask & 2U) != 0 ? 1 : 2;
        for (uint32_t lane = 0; lane < 2; ++lane) {
            while (participant->next_block[lane] < participant->logical_block_num) {
                const uint32_t block_idx = participant->next_block[lane];
                if (!scheduler_gang_fill_single(
                        graph, scheduler_state_base, resolver, run_control, participant, cohort_index,
                        resolver->cluster_worker_ids[lane + 1], subtask_slot, block_idx, publication
                    ))
                    break;
                participant->next_block[lane] += participant->block_stride;
            }
        }
    }
    scheduler_publish_cache_line(participant);
}

inline __aicore__ void scheduler_gang_release_local_slots(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *resolver, uint32_t cohort_index,
    uint64_t generation
) {
    for (uint32_t cluster_lane = 0; cluster_lane < 3; ++cluster_lane) {
        for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
            __gm__ SchedulerDispatchSlot *slot = scheduler_dispatch_slot_at(
                scheduler_state_base, resolver, resolver->cluster_worker_ids[cluster_lane], pending_slot
            );
            const uint64_t publication = scheduler_gm_query(slot->publication);
            if (scheduler_dispatch_state(publication) != SchedulerDispatchSlotState::GATED) continue;
            scheduler_observe_cache_line(slot);
            if (slot->cohort_index != cohort_index || slot->cohort_generation != generation) continue;
            scheduler_gm_store(
                slot->publication, scheduler_dispatch_publication(slot->generation, SchedulerDispatchSlotState::READY)
            );
        }
    }
}

inline __aicore__ int64_t scheduler_gang_select_ready_task(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *resolver, uint64_t priority_bit
) {
    for (uint64_t task_id = 0; task_id < resolver->graph_task_count; ++task_id) {
        __gm__ SchedulerTaskMetadata *metadata =
            scheduler_task_metadata_at(scheduler_state_base, resolver, static_cast<int64_t>(task_id));
        if (!scheduler_task_is_gang(metadata->flags) || scheduler_task_priority_bit(metadata->flags) != priority_bit)
            continue;
        __gm__ SchedulerTaskControl *control =
            scheduler_task_control_at(scheduler_state_base, resolver, static_cast<int64_t>(task_id));
        if (scheduler_gm_query(control->state) == static_cast<int64_t>(SchedulerTaskState::READY))
            return static_cast<int64_t>(task_id);
    }
    return SCHEDULER_TASK_ID_INVALID;
}

inline __aicore__ uint32_t scheduler_gang_assigned_blocks(uint32_t first, uint32_t stride, uint32_t logical_block_num) {
    if (first >= logical_block_num) return 0;
    return 1 + (logical_block_num - 1 - first) / stride;
}

inline __aicore__ bool scheduler_gang_admit_one(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *resolver,
    __gm__ SchedulerRunControl *run_control
) {
    if (resolver->resolver_index != 0) return false;
    __gm__ SchedulerGangCoordinator *coordinator = scheduler_gang_coordinator_at(scheduler_state_base, resolver);
    scheduler_observe_cache_line(coordinator);
    scheduler_observe_cache_line(&coordinator->active_dispatch_cohort);
    if (coordinator->active_dispatch_cohort != UINT64_MAX) return false;
    uint64_t ready_bits = coordinator->ready_priority_bits;
    if (ready_bits == 0) return false;
    const uint64_t priority_bit = (ready_bits & 1U) != 0 ? 1U : ((ready_bits & 2U) != 0 ? 2U : 4U);
    uint32_t cohort_index = UINT32_MAX;
    for (uint32_t candidate = 0; candidate < SCHEDULER_GANG_COHORT_COUNT; ++candidate) {
        __gm__ SchedulerGangCohort *cohort = scheduler_gang_cohort_at(scheduler_state_base, resolver, candidate);
        scheduler_observe_cache_line(cohort);
        if (cohort->state == static_cast<uint64_t>(SchedulerGangCohortState::FREE)) {
            cohort_index = candidate;
            break;
        }
    }
    if (cohort_index == UINT32_MAX) return false;
    const int64_t task_id = scheduler_gang_select_ready_task(scheduler_state_base, resolver, priority_bit);
    if (task_id < 0) {
        scheduler_gm_fetch_and(coordinator->ready_priority_bits, ~priority_bit);
        if (scheduler_gang_select_ready_task(scheduler_state_base, resolver, priority_bit) >= 0)
            scheduler_gm_fetch_or(coordinator->ready_priority_bits, priority_bit);
        return false;
    }
    __gm__ SchedulerTaskMetadata *metadata = scheduler_task_metadata_at(scheduler_state_base, resolver, task_id);
    scheduler_observe_cache_line(metadata);
    const uint32_t active_mask = metadata->active_mask;
    const uint32_t block_num = metadata->logical_block_num;
    const bool mix = scheduler_task_is_mix(metadata->flags);
    const bool single_aiv = !mix && (active_mask & 6U) != 0;
    const uint32_t capacity = single_aiv ? static_cast<uint32_t>(resolver->resolver_count * 2) :
                                           static_cast<uint32_t>(resolver->resolver_count);
    if (scheduler_task_requires_sync_start(metadata->flags) && block_num > capacity) {
        ++coordinator->capacity_reject_count;
        scheduler_publish_cache_line(&coordinator->admitted_count);
        scheduler_record_error(
            run_control, task_id, SchedulerGraphResult::UNSUPPORTED_SHAPE, &graph, resolver, UINT64_C(72)
        );
        return false;
    }
    const uint32_t required_participants = single_aiv ? (block_num + 1) / 2 : block_num;
    const uint32_t participant_count = required_participants < resolver->resolver_count ?
                                           required_participants :
                                           static_cast<uint32_t>(resolver->resolver_count);
    uint64_t generation = ++coordinator->next_generation;
    if (generation == 0) generation = ++coordinator->next_generation;
    for (uint32_t participant_index = 0; participant_index < participant_count; ++participant_index) {
        __gm__ SchedulerGangParticipant *participant =
            scheduler_gang_participant_at(scheduler_state_base, resolver, cohort_index, participant_index);
        participant->config_generation = generation;
        participant->task_id = task_id;
        participant->active_mask = active_mask;
        participant->logical_block_num = block_num;
        participant->local_published_subtasks = 0;
        participant->local_completed_subtasks = 0;
        participant->forwarded_state = UINT32_MAX;
        participant->forwarded_generation = 0;
        participant->drain_local_token = 0;
        participant->drain_subtree_token = 0;
        participant->stage_local_token = 0;
        participant->stage_subtree_token = 0;
        participant->dispatch_local_token = 0;
        participant->dispatch_subtree_token = 0;
        participant->completion_local_token = 0;
        participant->completion_subtree_token = 0;
        if (single_aiv) {
            participant->block_stride = static_cast<uint32_t>(resolver->resolver_count * 2);
            participant->next_block[0] = participant_index * 2;
            participant->next_block[1] = participant_index * 2 + 1;
            participant->local_expected_subtasks =
                scheduler_gang_assigned_blocks(participant->next_block[0], participant->block_stride, block_num) +
                scheduler_gang_assigned_blocks(participant->next_block[1], participant->block_stride, block_num);
        } else {
            participant->block_stride = static_cast<uint32_t>(resolver->resolver_count);
            participant->next_block[0] = participant_index;
            participant->next_block[1] = UINT32_MAX;
            participant->local_expected_subtasks =
                scheduler_gang_assigned_blocks(participant_index, participant->block_stride, block_num) *
                scheduler_gang_popcount(active_mask);
        }
        participant->participant_count = participant_count;
        scheduler_publish_cache_line(participant);
        scheduler_publish_cache_line(&participant->drain_local_token);
    }
    __gm__ SchedulerGangCohort *cohort = scheduler_gang_cohort_at(scheduler_state_base, resolver, cohort_index);
    cohort->task_id = task_id;
    cohort->generation = generation;
    cohort->priority_bit = priority_bit;
    cohort->active_mask = active_mask;
    cohort->logical_block_num = block_num;
    cohort->participant_count = participant_count;
    cohort->local_stride = single_aiv ? resolver->resolver_count * 2 : resolver->resolver_count;
    cohort->admitted_cycles = scheduler_cycles();
    cohort->state = static_cast<uint64_t>(
        scheduler_task_requires_sync_start(metadata->flags) ? SchedulerGangCohortState::DRAINING :
                                                              SchedulerGangCohortState::DISPATCHING
    );
    scheduler_publish_cache_line(cohort);
    __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, resolver, task_id);
    scheduler_gm_store(control->state, static_cast<int64_t>(SchedulerTaskState::DISPATCHING));
    coordinator->active_dispatch_cohort = cohort_index;
    ++coordinator->admitted_count;
    if (priority_bit == 1) ++coordinator->sync_drain_count;
    else if (priority_bit == 2) ++coordinator->mix_dispatch_count;
    else ++coordinator->spmd_dispatch_count;
    scheduler_publish_cache_line(&coordinator->active_dispatch_cohort);
    scheduler_publish_cache_line(&coordinator->admitted_count);
    scheduler_gang_publish_command(
        scheduler_state_base, resolver, 0, cohort_index, generation,
        scheduler_task_requires_sync_start(metadata->flags) ? SchedulerGangCohortState::DRAINING :
                                                              SchedulerGangCohortState::DISPATCHING
    );
    return true;
}

inline __aicore__ bool scheduler_gang_service_participant(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *resolver,
    __gm__ SchedulerRunControl *run_control, uint32_t cohort_index, uint64_t cohort_generation,
    SchedulerGangCohortState state
) {
    __gm__ SchedulerGangParticipant *participant =
        scheduler_gang_participant_at(scheduler_state_base, resolver, cohort_index, resolver->resolver_index);
    scheduler_observe_cache_line(participant);
    if (participant->config_generation != cohort_generation ||
        resolver->resolver_index >= participant->participant_count)
        return false;
    bool progress = scheduler_gang_forward_command(
        scheduler_state_base, resolver, participant, cohort_index, cohort_generation, state
    );
    if (state == SchedulerGangCohortState::DRAINING) {
        if (scheduler_gang_local_slots_drained(scheduler_state_base, resolver, participant->active_mask)) {
            progress =
                scheduler_gang_publish_local_token(participant, SchedulerGangTokenPhase::DRAIN, cohort_generation) ||
                progress;
        }
        (void)scheduler_gang_update_subtree_token(
            scheduler_state_base, resolver, cohort_index, participant->participant_count, cohort_generation,
            SchedulerGangTokenPhase::DRAIN
        );
    } else if (state == SchedulerGangCohortState::STAGING) {
        const uint32_t before = participant->local_published_subtasks;
        scheduler_gang_fill_participant(
            graph, scheduler_state_base, resolver, run_control, participant, cohort_index,
            SchedulerDispatchSlotState::GATED
        );
        progress = participant->local_published_subtasks != before;
        if (participant->local_published_subtasks == participant->local_expected_subtasks)
            scheduler_gang_publish_local_token(participant, SchedulerGangTokenPhase::STAGE, cohort_generation);
        (void)scheduler_gang_update_subtree_token(
            scheduler_state_base, resolver, cohort_index, participant->participant_count, cohort_generation,
            SchedulerGangTokenPhase::STAGE
        );
    } else if (state == SchedulerGangCohortState::RELEASING) {
        scheduler_gang_release_local_slots(scheduler_state_base, resolver, cohort_index, cohort_generation);
        progress =
            scheduler_gang_publish_local_token(participant, SchedulerGangTokenPhase::DISPATCH, cohort_generation) ||
            progress;
        (void)scheduler_gang_update_subtree_token(
            scheduler_state_base, resolver, cohort_index, participant->participant_count, cohort_generation,
            SchedulerGangTokenPhase::DISPATCH
        );
    } else if (state == SchedulerGangCohortState::DISPATCHING) {
        const uint32_t before = participant->local_published_subtasks;
        scheduler_gang_fill_participant(
            graph, scheduler_state_base, resolver, run_control, participant, cohort_index,
            SchedulerDispatchSlotState::READY
        );
        progress = participant->local_published_subtasks != before;
        if (participant->local_published_subtasks == participant->local_expected_subtasks)
            scheduler_gang_publish_local_token(participant, SchedulerGangTokenPhase::DISPATCH, cohort_generation);
        (void)scheduler_gang_update_subtree_token(
            scheduler_state_base, resolver, cohort_index, participant->participant_count, cohort_generation,
            SchedulerGangTokenPhase::DISPATCH
        );
    } else if (state == SchedulerGangCohortState::RETIRING) {
        scheduler_gang_publish_retire(scheduler_state_base, resolver, participant, cohort_index, cohort_generation);
    }
    if (state != SchedulerGangCohortState::RETIRING &&
        participant->local_completed_subtasks == participant->local_expected_subtasks &&
        participant->local_expected_subtasks != 0) {
        scheduler_gang_publish_local_token(participant, SchedulerGangTokenPhase::COMPLETION, cohort_generation);
    }
    if (state != SchedulerGangCohortState::RETIRING) {
        (void)scheduler_gang_update_subtree_token(
            scheduler_state_base, resolver, cohort_index, participant->participant_count, cohort_generation,
            SchedulerGangTokenPhase::COMPLETION
        );
    }
    return progress;
}

inline __aicore__ bool scheduler_gang_service_owner(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *resolver,
    __gm__ SchedulerRunControl *run_control, uint32_t cohort_index, __gm__ SchedulerGangCohort *cohort,
    SchedulerWakeStats *wake_stats, SchedulerReadyStats *ready_stats, SchedulerCompletionStats *completion_stats,
    __gm__ SchedulerReadyOwnerState *owner_state = nullptr
) {
    if (resolver->resolver_index != 0 || cohort->participant_count == 0) return false;
    const uint64_t generation = cohort->generation;
    auto state = static_cast<SchedulerGangCohortState>(cohort->state);
    bool progress = false;
    if (state == SchedulerGangCohortState::DRAINING &&
        scheduler_gang_root_token_ready(
            scheduler_state_base, resolver, cohort_index, generation, SchedulerGangTokenPhase::DRAIN
        )) {
        cohort->drain_complete_cycles = scheduler_cycles();
        cohort->state = static_cast<uint64_t>(SchedulerGangCohortState::STAGING);
        scheduler_publish_cache_line(cohort);
        scheduler_gang_publish_command(
            scheduler_state_base, resolver, 0, cohort_index, generation, SchedulerGangCohortState::STAGING
        );
        progress = true;
    } else if (state == SchedulerGangCohortState::STAGING &&
               scheduler_gang_root_token_ready(
                   scheduler_state_base, resolver, cohort_index, generation, SchedulerGangTokenPhase::STAGE
               )) {
        cohort->stage_complete_cycles = scheduler_cycles();
        cohort->state = static_cast<uint64_t>(SchedulerGangCohortState::RELEASING);
        scheduler_publish_cache_line(cohort);
        scheduler_gang_publish_command(
            scheduler_state_base, resolver, 0, cohort_index, generation, SchedulerGangCohortState::RELEASING
        );
        progress = true;
    } else if ((state == SchedulerGangCohortState::RELEASING || state == SchedulerGangCohortState::DISPATCHING) &&
               scheduler_gang_root_token_ready(
                   scheduler_state_base, resolver, cohort_index, generation, SchedulerGangTokenPhase::DISPATCH
               )) {
        cohort->dispatch_complete_cycles = scheduler_cycles();
        cohort->state = static_cast<uint64_t>(SchedulerGangCohortState::EXECUTING);
        scheduler_publish_cache_line(cohort);
        __gm__ SchedulerGangCoordinator *coordinator = scheduler_gang_coordinator_at(scheduler_state_base, resolver);
        coordinator->active_dispatch_cohort = UINT64_MAX;
        scheduler_publish_cache_line(&coordinator->active_dispatch_cohort);
        scheduler_gang_publish_command(
            scheduler_state_base, resolver, 0, cohort_index, generation, SchedulerGangCohortState::EXECUTING
        );
        progress = true;
    }
    state = static_cast<SchedulerGangCohortState>(cohort->state);
    if (state == SchedulerGangCohortState::EXECUTING &&
        scheduler_gang_root_token_ready(
            scheduler_state_base, resolver, cohort_index, generation, SchedulerGangTokenPhase::COMPLETION
        )) {
        __gm__ SchedulerTaskControl *control =
            scheduler_task_control_at(scheduler_state_base, resolver, cohort->task_id);
        scheduler_gm_store(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
        if (!scheduler_resolve_completion(
                graph, scheduler_state_base, resolver, run_control, cohort->task_id, wake_stats, ready_stats,
                completion_stats, false, false, nullptr, owner_state
            )) {
            return false;
        }
        if (completion_stats != nullptr) ++completion_stats->resolve_count;
        cohort->completion_cycles = scheduler_cycles();
        cohort->state = static_cast<uint64_t>(SchedulerGangCohortState::RETIRING);
        scheduler_publish_cache_line(cohort);
        scheduler_gang_publish_command(
            scheduler_state_base, resolver, 0, cohort_index, generation, SchedulerGangCohortState::FREE
        );
        __gm__ SchedulerGangParticipant *root =
            scheduler_gang_participant_at(scheduler_state_base, resolver, cohort_index, 0);
        scheduler_observe_cache_line(root);
        (void)scheduler_gang_forward_command(
            scheduler_state_base, resolver, root, cohort_index, generation, SchedulerGangCohortState::FREE
        );
        scheduler_gang_publish_retire(scheduler_state_base, resolver, root, cohort_index, generation);
        progress = true;
    }
    state = static_cast<SchedulerGangCohortState>(cohort->state);
    if (state == SchedulerGangCohortState::RETIRING &&
        scheduler_gang_root_token_ready(
            scheduler_state_base, resolver, cohort_index, scheduler_gang_retire_token(generation),
            SchedulerGangTokenPhase::COMPLETION
        )) {
        cohort->task_id = SCHEDULER_TASK_ID_INVALID;
        cohort->state = static_cast<uint64_t>(SchedulerGangCohortState::FREE);
        scheduler_publish_cache_line(cohort);
        // A gang task is globally resolved only after every Resolver has observed
        // the FREE command. Otherwise AICPU may stop the AICore schedulers while
        // the tree still contains a retiring generation, making the cohort unsafe
        // to reuse and failing final scheduler validation.
        scheduler_gm_fetch_add(run_control->resolved_task_count, UINT64_C(1));
        progress = true;
    }
    return progress;
}

inline __aicore__ bool scheduler_service_gang(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *resolver,
    __gm__ SchedulerRunControl *run_control, SchedulerWakeStats *wake_stats, SchedulerReadyStats *ready_stats,
    SchedulerCompletionStats *completion_stats, __gm__ SchedulerReadyOwnerState *owner_state = nullptr
) {
    if (resolver->is_resolver == 0) return false;
    __gm__ SchedulerGangCoordinator *coordinator = scheduler_gang_coordinator_at(scheduler_state_base, resolver);
    if (coordinator->gang_task_count == 0) return false;
    bool progress = false;
    const bool owner = resolver->resolver_index == 0;
    uint64_t command_generations[SCHEDULER_GANG_COHORT_COUNT]{};
    uint64_t command_states[SCHEDULER_GANG_COHORT_COUNT]{};
    if (!owner) {
        __gm__ SchedulerGangCommand *command =
            scheduler_gang_command_at(scheduler_state_base, resolver, resolver->resolver_index);
        scheduler_observe_cache_line(command);
        for (uint32_t cohort_index = 0; cohort_index < SCHEDULER_GANG_COHORT_COUNT; ++cohort_index) {
            command_generations[cohort_index] = command->generation[cohort_index];
            command_states[cohort_index] = command->state[cohort_index];
        }
    }
    for (uint32_t cohort_index = 0; cohort_index < SCHEDULER_GANG_COHORT_COUNT; ++cohort_index) {
        __gm__ SchedulerGangCohort *cohort = nullptr;
        uint64_t generation = command_generations[cohort_index];
        auto state = static_cast<SchedulerGangCohortState>(command_states[cohort_index]);
        if (owner) {
            cohort = scheduler_gang_cohort_at(scheduler_state_base, resolver, cohort_index);
            scheduler_observe_cache_line(cohort);
            generation = cohort->generation;
            state = static_cast<SchedulerGangCohortState>(cohort->state);
        }
        if (state == SchedulerGangCohortState::FREE) {
            if (!owner && generation != 0) {
                __gm__ SchedulerGangParticipant *participant = scheduler_gang_participant_at(
                    scheduler_state_base, resolver, cohort_index, resolver->resolver_index
                );
                // This Resolver necessarily observed its participant config in
                // an earlier active phase before the cohort could complete.
                // Keep that owner-local line hot after FREE instead of forcing
                // a DCCI invalidate on every idle scheduler iteration.
                if (participant->config_generation == generation &&
                    resolver->resolver_index < participant->participant_count) {
                    progress = scheduler_gang_forward_command(
                                   scheduler_state_base, resolver, participant, cohort_index, generation, state
                               ) ||
                               progress;
                    scheduler_gang_publish_retire(
                        scheduler_state_base, resolver, participant, cohort_index, generation
                    );
                }
            }
            continue;
        }
        progress = scheduler_gang_service_participant(
                       graph, scheduler_state_base, resolver, run_control, cohort_index, generation, state
                   ) ||
                   progress;
        if (owner) {
            progress = scheduler_gang_service_owner(
                           graph, scheduler_state_base, resolver, run_control, cohort_index, cohort, wake_stats,
                           ready_stats, completion_stats, owner_state
                       ) ||
                       progress;
        }
    }
    if (owner) progress = scheduler_gang_admit_one(graph, scheduler_state_base, resolver, run_control) || progress;
    return progress;
}
