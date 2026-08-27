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

#include "scheduler_gang.h"

inline __aicore__ bool scheduler_service_cluster_completion_slot(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *resolver,
    __gm__ SchedulerRunControl *run_control, uint32_t cluster_lane, uint32_t pending_slot,
    uint32_t completed_generation, SchedulerWakeStats *wake_stats, SchedulerReadyStats *ready_stats,
    SchedulerCompletionStats *completion_stats, uint64_t *ready_victim_cursors, bool trace_enabled,
    const SchedulerReadyClaim *replacement_ready, bool *direct_refilled,
    SchedulerCompletionServiceTiming *timing = nullptr, __gm__ SchedulerReadyOwnerState *owner_state = nullptr
) {
    if (direct_refilled != nullptr) *direct_refilled = false;
    if (cluster_lane >= 3 || pending_slot >= SCHEDULER_PENDING_SLOT_COUNT || completed_generation == 0) return false;
    const uint64_t worker_id = resolver->cluster_worker_ids[cluster_lane];
    __gm__ SchedulerCompletionInbox *completion_line =
        scheduler_completion_inbox_at(scheduler_state_base, resolver, worker_id);
    __gm__ SchedulerDispatchSlot *slot =
        scheduler_dispatch_slot_at(scheduler_state_base, resolver, worker_id, pending_slot);
    const uint64_t publication = scheduler_gm_query(slot->publication);
    if (scheduler_dispatch_state(publication) != SchedulerDispatchSlotState::READY ||
        scheduler_dispatch_generation(publication) != completed_generation) {
        scheduler_record_error(
            run_control, slot->task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, resolver, UINT64_C(74)
        );
        return false;
    }

    const bool record_timeline = timing != nullptr;
    uint64_t operation_start = record_timeline ? scheduler_cycles() : 0;
    scheduler_observe_cache_line(slot);
    const int64_t task_id = slot->task_id;
    const bool gang = slot->gang != 0;
    const uint8_t completed_subtask_slot = slot->subtask_slot;
    const uint32_t cohort_index = slot->cohort_index;
    const uint32_t cohort_generation = slot->cohort_generation;
    if (gang && replacement_ready != nullptr) return false;
    scheduler_gm_store(completion_line->completed_generations[pending_slot], UINT32_C(0));
    uint64_t operation_end = record_timeline ? scheduler_cycles() : 0;
    if (timing != nullptr) timing->consume_cycles += operation_end - operation_start;
    operation_start = operation_end;
    uint64_t ready_publish_cycles = 0;
    uint64_t refill_cycles = 0;
    uint64_t finalize_cycles = 0;
    uint64_t refill_start_cycles = 0;
    uint64_t refill_end_cycles = 0;
    bool refilled = false;
    if (gang) {
        if (cohort_index >= SCHEDULER_GANG_COHORT_COUNT) return false;
        __gm__ SchedulerGangParticipant *participant =
            scheduler_gang_participant_at(scheduler_state_base, resolver, cohort_index, resolver->resolver_index);
        scheduler_observe_cache_line(participant);
        if (participant->config_generation != cohort_generation || participant->task_id != task_id ||
            participant->local_completed_subtasks >= participant->local_expected_subtasks) {
            scheduler_record_error(
                run_control, task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, resolver, UINT64_C(73)
            );
            return false;
        }
        ++participant->local_completed_subtasks;
        scheduler_publish_cache_line(participant);
    } else {
        __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, resolver, task_id);
        scheduler_gm_store(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
        if (!scheduler_resolve_completion(
                graph, scheduler_state_base, resolver, run_control, task_id, wake_stats, ready_stats, completion_stats,
                false, false, timing == nullptr ? nullptr : &ready_publish_cycles, owner_state
            ))
            return false;
        if (timing != nullptr) timing->ready_publish_cycles += ready_publish_cycles;
        if (completion_stats != nullptr) ++completion_stats->resolve_count;
        uint64_t resolved_count_start = timing == nullptr ? 0 : scheduler_cycles();
        scheduler_gm_fetch_add(run_control->resolved_task_count, UINT64_C(1));
        if (timing != nullptr) {
            finalize_cycles = scheduler_cycles() - resolved_count_start;
            timing->finalize_cycles += finalize_cycles;
        }
        refill_start_cycles = record_timeline ? scheduler_cycles() : 0;
        SchedulerReadyClaim ready{};
        bool ready_available = replacement_ready != nullptr;
        if (ready_available) {
            ready = *replacement_ready;
        } else if (ready_victim_cursors != nullptr && worker_id != resolver->worker_index) {
            // A normal AIV task is never refilled directly onto the Resolver.
            // Its completed slot becomes capacity for late binding instead.
            __gm__ SchedulerGangCoordinator *coordinator =
                scheduler_gang_coordinator_at(scheduler_state_base, resolver);
            bool normal_fill_allowed = coordinator->gang_task_count == 0;
            if (!normal_fill_allowed) {
                scheduler_observe_cache_line(coordinator);
                scheduler_observe_cache_line(&coordinator->active_dispatch_cohort);
                normal_fill_allowed =
                    coordinator->ready_priority_bits == 0 && coordinator->active_dispatch_cohort == UINT64_MAX;
            }
            if (normal_fill_allowed) {
                const uint32_t core_type = scheduler_metadata_core_type_index(completed_subtask_slot);
                if (!scheduler_claim_ready_for_slot(
                        graph, scheduler_state_base, resolver, run_control, resolver->resolver_count, core_type,
                        &ready_victim_cursors[core_type], ready_stats, &ready, trace_enabled, owner_state
                    ))
                    return false;
                ready_available = ready.task_id >= 0;
            }
        }
        if (ready_available) {
            SchedulerFreeSlotClaim claim{worker_id, pending_slot, slot->generation};
            if (!scheduler_fill_dispatch_slot(
                    graph, scheduler_state_base, resolver, run_control, claim, ready, trace_enabled
                ))
                return false;
            refilled = true;
        }
        refill_end_cycles = record_timeline ? scheduler_cycles() : 0;
        if (timing != nullptr) {
            refill_cycles = refill_end_cycles - refill_start_cycles;
            timing->refill_cycles += refill_cycles;
        }
    }
    operation_end = record_timeline ? scheduler_cycles() : 0;
    if (refill_start_cycles == 0) refill_start_cycles = operation_end;
    if (refill_end_cycles == 0) refill_end_cycles = operation_end;
    if (timing != nullptr) {
        uint64_t resolve_total = operation_end - operation_start;
        uint64_t excluded = ready_publish_cycles + refill_cycles + finalize_cycles;
        timing->resolve_cycles += resolve_total > excluded ? resolve_total - excluded : 0;
    }
    operation_start = operation_end;
    if (!refilled) {
        slot->task_id = SCHEDULER_TASK_ID_INVALID;
        scheduler_writeback_cache_line(slot);
        scheduler_gm_store(
            slot->publication, scheduler_dispatch_publication(slot->generation, SchedulerDispatchSlotState::FREE)
        );
    }
    const uint64_t completion_end = record_timeline ? scheduler_cycles() : 0;
    if (timing != nullptr) timing->finalize_cycles += completion_end - operation_start;
    if (direct_refilled != nullptr) *direct_refilled = refilled;
    return true;
}

inline __host__ __aicore__ uint32_t scheduler_completion_catchup_mask(uint32_t initial_completion_mask) {
    constexpr uint32_t all_pending_slots_mask = (1U << SCHEDULER_PENDING_SLOT_COUNT) - 1;
    return initial_completion_mask == 0 ? 0 : all_pending_slots_mask & ~initial_completion_mask;
}

inline __aicore__ bool scheduler_service_cluster_completions(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *resolver,
    __gm__ SchedulerRunControl *run_control, SchedulerWakeStats *wake_stats, SchedulerReadyStats *ready_stats,
    SchedulerCompletionStats *completion_stats, uint64_t *ready_victim_cursors = nullptr, bool trace_enabled = false,
    uint64_t *direct_refilled_slot_mask = nullptr, SchedulerCompletionServiceTiming *timing = nullptr,
    __gm__ SchedulerReadyOwnerState *owner_state = nullptr
) {
    if (resolver->is_resolver == 0) return false;
    if (direct_refilled_slot_mask != nullptr) *direct_refilled_slot_mask = 0;
    bool progress = false;
    for (uint32_t cluster_lane = 0; cluster_lane < 3; ++cluster_lane) {
        const uint64_t worker_id = resolver->cluster_worker_ids[cluster_lane];
        __gm__ SchedulerCompletionInbox *completion_line =
            scheduler_completion_inbox_at(scheduler_state_base, resolver, worker_id);
        uint64_t completed_generations = scheduler_gm_query_u32_pair(completion_line->completed_generations);
        uint32_t initial_completion_mask = 0;
        for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
            if (static_cast<uint32_t>(completed_generations >> (pending_slot * 32)) != 0)
                initial_completion_mask |= 1U << pending_slot;
        }
        uint32_t scan_mask = initial_completion_mask;
        // Completion processing can be much slower than the sibling kernel. Refresh the packed completion line once
        // for slots that were incomplete in the initial scan, excluding newly refilled slots from the catch-up pass.
        for (uint32_t scan_pass = 0; scan_pass < 2 && scan_mask != 0; ++scan_pass) {
            for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
                if ((scan_mask & (1U << pending_slot)) == 0) continue;
                const uint32_t completed_generation =
                    static_cast<uint32_t>(completed_generations >> (pending_slot * 32));
                if (completed_generation == 0) continue;
                bool direct_refilled = false;
                if (!scheduler_service_cluster_completion_slot(
                        graph, scheduler_state_base, resolver, run_control, cluster_lane, pending_slot,
                        completed_generation, wake_stats, ready_stats, completion_stats, ready_victim_cursors,
                        trace_enabled, nullptr, &direct_refilled, timing, owner_state
                    ))
                    return false;
                if (direct_refilled && direct_refilled_slot_mask != nullptr)
                    *direct_refilled_slot_mask |= UINT64_C(1)
                                                  << (cluster_lane * SCHEDULER_PENDING_SLOT_COUNT + pending_slot);
                progress = true;
            }
            scan_mask = scan_pass == 0 ? scheduler_completion_catchup_mask(initial_completion_mask) : 0;
            if (scan_mask != 0)
                completed_generations = scheduler_gm_query_u32_pair(completion_line->completed_generations);
        }
    }
    return progress;
}
