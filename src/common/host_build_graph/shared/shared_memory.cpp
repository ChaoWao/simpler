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
 * host_build_graph shared-memory implementation
 *
 * Implements shared memory allocation, initialization, and management
 * for Orchestrator-Scheduler communication.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "host_build_graph/shared_memory.h"
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include "common/unified_log.h"

// =============================================================================
// Size Calculation
// =============================================================================

uint64_t SharedMemoryHandle::calculate_size(uint64_t max_tasks) {
    // Total SM size = offset just past the last segment, from the single
    // source of truth for the layout (sm_layout::segment_offsets).
    return sm_layout::segment_offsets(max_tasks).end;
}

// =============================================================================
// Creation and Destruction
// =============================================================================

void SharedMemoryHandle::setup_pointers(uint64_t pitch) {
    char *base = (char *)sm_base;
    header = (SharedMemoryHeader *)base;

    // descriptors / payloads / slot_states — offsets from the single source
    // of truth (sm_layout::segment_offsets), so this setup and the
    // device-address helpers cannot drift.
    //
    // Only the four slot-pitched segments, and no argument pool: the pool extents
    // differ between the mirror and the image, while these four depend on the pitch
    // alone. The orchestrator holds the mirror's pool bases; the device resolves an
    // argument region through the payload's delta and needs no base at all.
    auto off = sm_layout::segment_offsets(sm_layout::image_extents({pitch, 0, 0, 0}));
    auto &tasks = header->tasks;
    tasks.task_descriptors = (TaskDescriptor *)(base + off.descriptors);
    tasks.task_payloads = (TaskPayload *)(base + off.payloads);
    tasks.slot_states = (ChipTaskSlotState *)(base + off.slot_states);
    tasks.completion_flags = (std::atomic<uint8_t> *)(base + off.completion_flags);
}

bool SharedMemoryHandle::init(void *sm_base_arg, uint64_t sm_size_arg, uint64_t max_tasks) {
    if (!sm_base_arg || sm_size_arg == 0) return false;
    const uint64_t mirror_bytes = calculate_size(max_tasks);
    // The mirror has to stay inside an int32 delta's reach: a payload names its
    // argument regions that way, and set() leaves a field unbound rather than
    // truncating, so a pool out of reach would read back as null and be dereferenced
    // as one. attach_populated makes the same check on the shipped image.
    //
    // Tested before the region's own size, because this rejects `max_tasks` itself:
    // no region, however large, makes such a reservation usable.
    if (mirror_bytes > static_cast<uint64_t>(INT32_MAX)) return false;
    if (sm_size_arg < mirror_bytes) return false;

    sm_base = sm_base_arg;
    sm_size = sm_size_arg;
    is_owner = false;
    setup_pointers(max_tasks);
    init_header();
    return true;
}

bool SharedMemoryHandle::attach_populated(
    void *sm_base_arg, uint64_t sm_size_arg, uint64_t max_tasks, uint64_t live_slots, uint64_t image_bytes
) {
    if (!sm_base_arg || sm_size_arg == 0) return false;
    // A pitch above `max_tasks` names a slot the host could not have built, and one
    // below what the host used would place every segment past the descriptors
    // short. This is the whole contract between the two sides, checked once at
    // attach rather than trusted.
    if (live_slots == 0 || live_slots > max_tasks) return false;
    // The image must at least hold the four slot-pitched segments; anything beyond
    // that is its argument pools, whose extents are the bind's cursors and are not
    // recomputable here. The first pool's offset is exactly where those four end.
    const uint64_t slots_end = sm_layout::segment_offsets(sm_layout::image_extents({live_slots, 0, 0, 0})).fanin_pool;
    if (image_bytes < slots_end) return false;
    // The region holds the compacted image, so that is what has to fit — the
    // dimensioned size is no longer its size.
    if (sm_size_arg < image_bytes) return false;
    // A slot state names its payload and descriptor by an int32 delta from its own
    // address, and a payload names its argument regions the same way, so no two
    // positions in the image may be further apart than that.
    if (image_bytes > static_cast<uint64_t>(INT32_MAX)) return false;

    sm_base = sm_base_arg;
    sm_size = sm_size_arg;
    is_owner = false;
    setup_pointers(live_slots);
    // Deliberately NO init_header: the SM already holds the host orchestrator's
    // task graph (descriptors, slot states, completion flags).
    return true;
}

SharedMemoryHandle *SharedMemoryHandle::create_and_init_default(DeviceArena &arena) {
    const uint64_t buffer_size = calculate_size(CHIP_DEFAULT_GRAPH_TASKS);
    const size_t off_handle = arena.reserve(sizeof(SharedMemoryHandle), alignof(SharedMemoryHandle));
    const size_t off_buffer = arena.reserve(static_cast<size_t>(buffer_size), CHIP_ALIGN_SIZE);
    if (arena.commit() == nullptr) return nullptr;

    auto *handle = static_cast<SharedMemoryHandle *>(arena.region_ptr(off_handle));
    memset(handle, 0, sizeof(*handle));
    void *buffer = arena.region_ptr(off_buffer);
    memset(buffer, 0, static_cast<size_t>(buffer_size));
    if (!handle->init(buffer, buffer_size, CHIP_DEFAULT_GRAPH_TASKS)) return nullptr;
    return handle;
}

void SharedMemoryHandle::destroy() {
    // Arena-owned wrappers (is_owner == false) are reclaimed by arena.release();
    // calling destroy on them is a no-op so existing callers stay safe.
    if (is_owner && sm_base) {
        free(sm_base);
        free(this);
    }
}

// =============================================================================
// Initialization
// =============================================================================
//
// no need init data in pool, init pool data when used
void SharedMemoryHandle::init_header() {
    // Polling completion: -1 = "no task completed yet"; the first task to
    // complete (local_id 0) advances the watermark to 0.
    header->tasks.completed_watermark.store(-1, std::memory_order_relaxed);

    // 0 until run_host_orchestration writes the run's count; init runs before
    // orchestration, so the real value is not known here yet.
    header->tasks.total_tasks = 0;

    header->orchestrator_done.store(0, std::memory_order_relaxed);

    // Layout info. The descriptors are the first segment, so their offset is where
    // the header's own padded size ends — pitch-independent, unlike every segment
    // after them.
    header->tasks.task_descriptors_offset = CHIP_ALIGN_UP(sizeof(SharedMemoryHeader), CHIP_ALIGN_SIZE);

    header->total_size = sm_size;

    // Error reporting
    header->orch_error_code.store(SIMPLER_ERROR_NONE, std::memory_order_relaxed);
    header->sched_error_bitmap.store(0, std::memory_order_relaxed);
    header->sched_error_code.store(SIMPLER_ERROR_NONE, std::memory_order_relaxed);
    header->sched_error_thread.store(-1, std::memory_order_relaxed);

    // Per-slot init (slot_states.reset_for_reuse() + active_mask, and clearing the
    // completion flag) happens init-on-write in orch::prepare_task as each slot
    // [0, total_tasks) is claimed, so the SM init/upload cost tracks the task
    // count, not the size the table was dimensioned for. The device reads no slot
    // past total_tasks, so the unclaimed tail is left uninitialized.
}

// =============================================================================
// Debug Utilities
// =============================================================================

void SharedMemoryHandle::print_layout() {
    if (!header) return;

    SharedMemoryHeader *h = header;

    LOG_DEBUG("=== Shared Memory Layout ===");
    LOG_DEBUG("Base address:       %p", sm_base);
    LOG_DEBUG("Total size:         %" PRIu64 " bytes", h->total_size);
    LOG_DEBUG("Task table:");
    LOG_DEBUG(
        "  descriptors_off:  %" PRIu64 " (0x%" PRIx64 ")", h->tasks.task_descriptors_offset,
        h->tasks.task_descriptors_offset
    );
    LOG_DEBUG("  completed_wm:     %d", h->tasks.completed_watermark.load(std::memory_order_acquire));
    LOG_DEBUG("orchestrator_done:  %d", h->orchestrator_done.load(std::memory_order_acquire));
    LOG_DEBUG("Error state:");
    LOG_DEBUG("  orch_error_code:    %d", h->orch_error_code.load(std::memory_order_relaxed));
    LOG_DEBUG("  sched_error_bitmap: 0x%x", h->sched_error_bitmap.load(std::memory_order_relaxed));
    LOG_DEBUG("  sched_error_code:   %d", h->sched_error_code.load(std::memory_order_relaxed));
    LOG_DEBUG("  sched_error_thread: %d", h->sched_error_thread.load(std::memory_order_relaxed));
    LOG_DEBUG("================================");
}

bool SharedMemoryHandle::validate() {
    if (!sm_base) return false;
    if (!header) return false;

    const SharedMemoryHeader *h = header;

    // Check that offsets are within bounds
    if (h->tasks.task_descriptors_offset >= h->total_size) return false;

    // Check pointer alignment
    if ((uintptr_t)h->tasks.task_descriptors % CHIP_ALIGN_SIZE != 0) return false;

    return true;
}
