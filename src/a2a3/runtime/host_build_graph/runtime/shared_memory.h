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
 * PTO Runtime2 - Shared Memory Layout
 *
 * Defines the shared memory structure for Orchestrator-Scheduler communication.
 *
 * Memory Layout (single ring):
 *   +---------------------------+
 *   | SharedMemoryHeader        |  (flow control + sync)
 *   +---------------------------+
 *   | TaskDescriptor[]          |
 *   | TaskPayload[]             |
 *   | TaskSlotState[]           |
 *   +---------------------------+
 *
 * Design principles:
 * - Only data needed for Orchestrator<->Scheduler communication is here
 * - TensorMap, scope_stack, ready_queues, dep_pool are in private memory
 * - Flow control via atomic counters/flags (no locks needed for single-word R/W)
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <stddef.h>

#include <cstring>

#include "utils/device_arena.h"
#include "runtime_types.h"

// =============================================================================
// Shared Memory Header
// =============================================================================

struct PTO2SharedMemoryHandle;

/**
 * Per-ring flow control state in shared memory.
 * Written/read by Orchestrator and Scheduler for synchronization.
 */
struct alignas(64) PTO2RingFlowControl {
    // Written by Orchestrator, read by Scheduler. There is no reverse channel:
    // the ring is whole-graph-resident, so the scheduler never reclaims task
    // slots and has nothing to publish back.
    alignas(64) std::atomic<int32_t> current_task_index;  // Task ring head (next to allocate)

    // Per-boot SM reset. PTO2TaskAllocator::init() seeds its private
    // local_task_id_ to 0 *without* dereferencing current_task_index — it
    // relies on this reset running on every AICPU boot so 0 stays in sync. If
    // you ever change the initial fc value or the boot ordering, update
    // PTO2TaskAllocator::init (ring_buffer.h) in the same change, or
    // submit IDs will be off by the divergence.
    void init() { current_task_index.store(0, std::memory_order_relaxed); }

    bool validate(PTO2SharedMemoryHandle *handle) const;
};

static_assert(sizeof(PTO2RingFlowControl) == 64, "PTO2RingFlowControl must be exactly one cache line (64B)");

/**
 * Per-ring shared memory header section.
 *
 * Groups flow-control, layout info, and per-ring data pointers for a single ring.
 * Pointers are host-side only (set by setup_pointers, invalid on device).
 */
struct alignas(64) PTO2SharedMemoryRingHeader {
    PTO2RingFlowControl fc;

    // Highest task_id such that every task with id in [0, completed_watermark]
    // has its completion_flags byte set. Advanced over the full contiguous
    // completed prefix at task-completion time (on_mixed_task_complete). The host
    // consumer-wait gates on it: a producer slot P's consumers have all retired
    // once completed_watermark >= P.last_consumer_local_id. On its own cache line
    // (concurrent CAS-advance by completing threads).
    alignas(64) std::atomic<int32_t> completed_watermark;

    // Layout metadata (set once at init)
    alignas(64) uint64_t task_window_size;
    int32_t task_window_mask;
    uint64_t heap_size;
    uint64_t task_descriptors_offset;  // Offset from SM base, in bytes

    // Per-ring data pointers (host-side, set by setup_pointers)
    PTO2TaskDescriptor *task_descriptors;
    PTO2TaskPayload *task_payloads;
    ChipTaskSlotState *slot_states;

    // Polling-completion state (device-addressed array, one byte per slot).
    // 0 = pending, 1 = task fully COMPLETED. Writer = the task's completer at
    // on_mixed_task_complete; reader = consumer fanin polling (is_completion_flag_set).
    // Cleared per-slot in orch::prepare_task as each slot is claimed. Indexed by
    // local_id & task_window_mask.
    std::atomic<uint8_t> *completion_flags;

    bool is_completion_flag_set(int32_t local_id, std::memory_order order = std::memory_order_acquire) const {
        return completion_flags[local_id & task_window_mask].load(order) != 0;
    }

    void set_completion_flag(int32_t local_id, std::memory_order order = std::memory_order_release) const {
        completion_flags[local_id & task_window_mask].store(1, order);
    }

    // set completion flag first before updating the watermark (logic requirement)
    void update_completed_watermark() {
        int32_t curr_watermark = completed_watermark.load(std::memory_order_acquire);
        const int32_t submitted = fc.current_task_index.load(std::memory_order_acquire);

        int32_t next = curr_watermark;
        while (true) {
            while (next + 1 < submitted && is_completion_flag_set(next + 1)) {
                ++next;
            }
            if (next == curr_watermark) {
                return;
            }

            if (completed_watermark.compare_exchange_strong(
                    curr_watermark, next, std::memory_order_acq_rel, std::memory_order_acquire
                )) {
                curr_watermark = next;
            } else {
                // The acquire release semantics of the successful CAS guarantee that in the case of failure this thread
                // also synchronises with the thread reporting the completion through the intermediary thread(s).
                next = std::max(next, curr_watermark);
            }
        }
    }

    int32_t get_slot_by_task_id(int32_t local_task_id) { return local_task_id & task_window_mask; }

    PTO2TaskDescriptor &get_task_by_slot(int32_t slot) { return task_descriptors[slot]; }

    PTO2TaskDescriptor &get_task_by_task_id(int32_t local_id) {
        return task_descriptors[get_slot_by_task_id(local_id)];
    }

    // No get_payload_by_slot / get_payload_by_task_id here: a payload is reached
    // through its slot state's `payload` delta, which the image's restack rebinds to
    // the real address.

    ChipTaskSlotState &get_slot_state_by_slot(int32_t slot) { return slot_states[slot]; }

    ChipTaskSlotState &get_slot_state_by_task_id(int32_t local_id) {
        return slot_states[get_slot_by_task_id(local_id)];
    }
};

static_assert(sizeof(PTO2SharedMemoryRingHeader) == 192, "PTO2SharedMemoryRingHeader layout drift");
static_assert(
    offsetof(PTO2SharedMemoryRingHeader, task_descriptors_offset) == 152,
    "PTO2SharedMemoryRingHeader task_descriptors_offset layout drift"
);

/**
 * Shared memory header structure
 *
 * Contains per-ring flow control and global layout information.
 */
struct alignas(PTO2_ALIGN_SIZE) PTO2SharedMemoryHeader {
    // === RING FLOW CONTROL + LAYOUT INFO (single ring, set once at init) ===
    PTO2SharedMemoryRingHeader ring;

    // === GLOBAL FIELDS ===
    std::atomic<int32_t> orchestrator_done;  // Flag: orchestration complete

    // Total shared memory size (for validation)
    uint64_t total_size;

    // === ERROR REPORTING ===

    // Orchestrator fatal error code (Orchestrator → Scheduler, AICPU → Host)
    // Non-zero signals fatal error. Written by orchestrator, read by scheduler and host.
    std::atomic<int32_t> orch_error_code;

    // Scheduler error state (Scheduler → Host, independent of orchestrator)
    // Written by scheduler threads on timeout; read by orchestrator and host.
    std::atomic<uint32_t> sched_error_bitmap;  // Bit X set = thread X had error
    std::atomic<int32_t> sched_error_code;     // Last scheduler error code (last-writer-wins)
    std::atomic<int32_t> sched_error_thread;   // Thread index of last error writer
};

static_assert(sizeof(PTO2SharedMemoryHeader) == 256, "PTO2SharedMemoryHeader layout drift");
static_assert(offsetof(PTO2SharedMemoryHeader, total_size) == 200, "PTO2SharedMemoryHeader total_size layout drift");
static_assert(
    offsetof(PTO2SharedMemoryHeader, orch_error_code) == 208, "PTO2SharedMemoryHeader orch_error_code layout drift"
);

// =============================================================================
// Shared Memory Handle
// =============================================================================

/**
 * Handle for shared memory lifecycle management (create/destroy).
 * Runtime components (orchestrator, scheduler) use PTO2SharedMemoryHeader* directly.
 */
struct PTO2SharedMemoryHandle {
    void *sm_base;     // Base address of shared memory
    uint64_t sm_size;  // Total size of shared memory

    PTO2SharedMemoryHeader *header;

    // Ownership flag
    bool is_owner;  // True if this handle allocated the memory

    // === Static helpers ===

    static uint64_t calculate_size(uint64_t task_window_size);

    // UT convenience: reserve wrapper + sm_base on `arena`, commit, and init
    // using default PTO2_TASK_WINDOW_SIZE / PTO2_HEAP_SIZE. Only valid when the
    // arena is otherwise empty (the call performs the single commit). All
    // memory is owned by the arena — caller must not call destroy().
    static PTO2SharedMemoryHandle *create_and_init_default(DeviceArena &arena);

    // === Instance methods ===

    // In-place init for caller-provided wrapper storage (e.g. a region carved
    // out of a DeviceArena). Sets is_owner = false, calls setup_pointers and
    // init_header. Returns false when `sm_size` is too small for the requested
    // `task_window_size`.
    bool init(void *sm_base, uint64_t sm_size, uint64_t task_window_size, uint64_t heap_size);

    // Attach to an ALREADY-populated shared memory region: point the handle and
    // every ring header's data pointers (descriptors / payloads / slot_states)
    // at `sm_base`, but do NOT reset the flow-control counters / slot states.
    // Used by host_build_graph host-orch, where the host orchestrator populated
    // the SM and H2D'd it; the device must re-point at its own SM base without
    // wiping the contents (unlike init, which also resets the header).
    //
    // `live_slots` is the pitch the uploaded arrays were laid out with — the
    // number of slots the host actually submitted, not the ring capacity. It must
    // match what the host used or every segment past the descriptors resolves to
    // the wrong address, so both sides derive it from the same submitted count.
    // The capacity and mask in the header are unchanged, and `local_id & mask`
    // yields `local_id`, which is below `live_slots` for every ring task.
    //
    // `image_bytes` is what the host shipped, pools included. The device cannot
    // recompute it — the pool extents are the bind's cursors, which only the host saw
    // — and it does not need to: a payload names its argument regions by delta, so no
    // pool base is resolved here. The value bounds the region and checks the int32
    // delta reach.
    bool attach_populated(
        void *sm_base, uint64_t sm_size, uint64_t task_window_size, uint64_t live_slots, uint64_t image_bytes
    );

    void destroy();
    void print_layout();
    bool validate();

private:
    void init_header(uint64_t task_window_size, uint64_t heap_size);
    // `pitch` is the slot count the arrays are dimensioned for. init passes the
    // ring capacity (the mirror the orchestrator writes into); attach_populated
    // passes the submitted count (the compacted image that shipped).
    void setup_pointers(uint64_t pitch);
};

// =============================================================================
// SM Device Layout Helpers
// =============================================================================
//
// When the host pre-builds a runtime-arena image, it needs the device-side
// addresses of several SM sub-fields (ring flow-control counters,
// task_descriptors arrays, orch_error_code) so it can wire them into the
// orchestrator / scheduler init_data path without dereferencing the SM —
// the SM lives in device memory and cannot be touched from host.
//
// These helpers compute those addresses by offset arithmetic on the SM
// device base. Pure pointer math, no loads/stores; safe to call from host.
// The same arithmetic happens on AICPU too (via PTO2SharedMemoryHandle's
// own setup_pointers), so values are guaranteed consistent across sides.
namespace pto2_sm_layout {

inline std::atomic<int32_t> *orch_error_code_addr(void *sm_dev_base) noexcept {
    return reinterpret_cast<std::atomic<int32_t> *>(
        static_cast<char *>(sm_dev_base) + offsetof(PTO2SharedMemoryHeader, orch_error_code)
    );
}

inline PTO2SharedMemoryRingHeader *ring_header_addr(void *sm_dev_base) noexcept {
    return reinterpret_cast<PTO2SharedMemoryRingHeader *>(
        static_cast<char *>(sm_dev_base) + offsetof(PTO2SharedMemoryHeader, ring)
    );
}

inline std::atomic<int32_t> *ring_current_task_index_addr(void *sm_dev_base) noexcept {
    return reinterpret_cast<std::atomic<int32_t> *>(
        reinterpret_cast<char *>(ring_header_addr(sm_dev_base)) + offsetof(PTO2SharedMemoryRingHeader, fc) +
        offsetof(PTO2RingFlowControl, current_task_index)
    );
}

// Byte offsets (from the SM base) of the ring's segments. The layout is: header, then
// descriptors -> payloads -> slot_states -> completion_flags -> the three argument
// pools, every segment PTO2_ALIGN_UP-padded. RingImageExtents dimensions them: the
// mirror for the worst case the API allows, the image for what this bind holds, which
// is what makes the live prefixes contiguous and the upload one copy.
//
// The pools sit last because nothing on the device resolves a segment past
// completion_flags: a payload names its argument regions by delta, so the four
// slot-pitched offsets are all the attach path computes.
struct PTO2RingSegmentOffsets {
    uint64_t descriptors;
    uint64_t payloads;
    uint64_t slot_states;
    uint64_t completion_flags;  // polling-completion byte array (1 byte/slot)
    uint64_t fanin_pool;
    uint64_t tensor_pool;
    uint64_t scalar_pool;
    uint64_t end;  // offset just past the last segment (total image size)
};

// How many slots and how many pool elements a layout is dimensioned for.
struct RingImageExtents {
    uint64_t slots;
    uint64_t fanin_elems;
    uint64_t tensor_elems;
    uint64_t scalar_elems;
};

// The mirror the orchestrator writes into: the ring capacity, every pool sized so the
// worst case cannot overflow a bump — task_window tasks each at their full cap. That
// bound is what lets prepare_task advance a cursor with no capacity check.
inline RingImageExtents mirror_extents(uint64_t task_window_size) noexcept {
    return RingImageExtents{
        task_window_size,
        task_window_size * PTO2_MAX_FANIN,
        task_window_size * MAX_TENSOR_ARGS,
        task_window_size * MAX_SCALAR_ARGS,
    };
}

// Single source of truth for the SM segment layout. Returns offsets (not pointers),
// so it serves BOTH the host-side pointer setup (`setup_pointers`, which adds
// `sm_base`) and the device-address helpers below (which add `sm_dev_base`). Adding
// or reordering a segment is a one-line edit here; every consumer follows
// automatically, so the layout walk can never silently disagree across call sites.
inline PTO2RingSegmentOffsets ring_segment_offsets(const RingImageExtents &e) noexcept {
    uint64_t off = PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    PTO2RingSegmentOffsets o{};
    o.descriptors = off;
    off += PTO2_ALIGN_UP(e.slots * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
    o.payloads = off;
    off += PTO2_ALIGN_UP(e.slots * sizeof(PTO2TaskPayload), PTO2_ALIGN_SIZE);
    o.slot_states = off;
    off += PTO2_ALIGN_UP(e.slots * sizeof(ChipTaskSlotState), PTO2_ALIGN_SIZE);
    o.completion_flags = off;
    off += PTO2_ALIGN_UP(e.slots * sizeof(std::atomic<uint8_t>), PTO2_ALIGN_SIZE);
    o.fanin_pool = off;
    off += PTO2_ALIGN_UP(e.fanin_elems * sizeof(int32_t), PTO2_ALIGN_SIZE);
    o.tensor_pool = off;
    off += PTO2_ALIGN_UP(e.tensor_elems * sizeof(ChipTensor), PTO2_ALIGN_SIZE);
    o.scalar_pool = off;
    off += PTO2_ALIGN_UP(e.scalar_elems * sizeof(uint64_t), PTO2_ALIGN_SIZE);
    o.end = off;
    return o;
}

inline PTO2RingSegmentOffsets ring_segment_offsets(uint64_t task_window_size) noexcept {
    return ring_segment_offsets(mirror_extents(task_window_size));
}

// Every per-task region starts on a cache line, which PTO2TaskPayload::init's
// round-up scalar memcpy relies on. ChipTensor is 2 cache lines, so a tensor region
// is aligned for any count; the fanin and scalar strides need it stated.
static_assert(
    (PTO2_MAX_FANIN * sizeof(int32_t)) % ARG_POOL_ALIGN == 0,
    "the fanin region's cap must be a whole number of cache lines"
);
static_assert(
    (MAX_SCALAR_ARGS * sizeof(uint64_t)) % ARG_POOL_ALIGN == 0,
    "the scalar region's cap must be a whole number of cache lines"
);

// The pitch the shipped image uses for a given submitted task count. A bind that
// submits nothing still ships its header and still attaches, and a zero-length
// array has no layout, so the pitch never drops below one slot.
inline uint64_t live_slot_pitch(uint64_t submitted_tasks) noexcept {
    return submitted_tasks == 0 ? 1 : submitted_tasks;
}

// What a bind actually holds, for the shipped image's extents. Each pool ships only the
// prefix its cursor reached, which is why a run of narrow tasks ships a small fraction
// of the mirror's pools.
struct BindUsage {
    uint64_t submitted_tasks;
    uint64_t fanin_elems;
    uint64_t tensor_elems;
    uint64_t scalar_elems;
};

inline RingImageExtents image_extents(const BindUsage &used) noexcept {
    return RingImageExtents{
        live_slot_pitch(used.submitted_tasks),
        used.fanin_elems,
        used.tensor_elems,
        used.scalar_elems,
    };
}

// Restack the live prefix of every ring segment from the ring-pitched mirror the
// orchestrator wrote into an image dimensioned for what this bind holds, where the
// prefixes are contiguous and can travel as one copy.
//
// `out_base` must be PTO2_ALIGN_SIZE-aligned and hold
// `ring_segment_offsets(image_extents(used)).end` bytes. Returns that byte count.
//
// Two things the restack has to fix up, both because the image is not the mirror:
//
//   - the ring header's data pointers name the mirror's arrays, so they leave as
//     null rather than carrying host addresses into device memory (the device
//     resolves them in attach_populated);
//   - a slot state names its payload and descriptor, and a payload names its three
//     argument regions, by a delta from the naming field's own address; the restack
//     changed those distances, so every one is re-taken against the image. A region
//     keeps its position within its pool, so the re-take is the same arithmetic with
//     the image's bases.
inline uint64_t
compact_live_image(const char *mirror_base, uint64_t task_window_size, const BindUsage &used, char *out_base) noexcept {
    // The mirror is dimensioned for the worst case, so a live count or a cursor past
    // it reads beyond the segment it is copying from and ships a corrupt image.
    // attach_populated tests the slot bound again on the device side.
    const RingImageExtents mirror = mirror_extents(task_window_size);
    always_assert(used.submitted_tasks <= mirror.slots);
    always_assert(used.fanin_elems <= mirror.fanin_elems);
    always_assert(used.tensor_elems <= mirror.tensor_elems);
    always_assert(used.scalar_elems <= mirror.scalar_elems);
    const PTO2RingSegmentOffsets from = ring_segment_offsets(mirror);
    const PTO2RingSegmentOffsets to = ring_segment_offsets(image_extents(used));

    // The header and the descriptors offset are pitch-independent, so the header
    // lands where it already was.
    std::memcpy(out_base, mirror_base, to.descriptors);
    auto &out_ring = reinterpret_cast<PTO2SharedMemoryHeader *>(out_base)->ring;
    out_ring.task_descriptors = nullptr;
    out_ring.task_payloads = nullptr;
    out_ring.slot_states = nullptr;
    out_ring.completion_flags = nullptr;

    const uint64_t nt = used.submitted_tasks;
    std::memcpy(out_base + to.descriptors, mirror_base + from.descriptors, nt * sizeof(PTO2TaskDescriptor));
    // One copy, not one per payload: PTO2TaskPayload is fixed-size, so the mirror and
    // the image share a stride. Each pool is likewise one copy of its own prefix.
    std::memcpy(out_base + to.payloads, mirror_base + from.payloads, nt * sizeof(PTO2TaskPayload));
    std::memcpy(out_base + to.slot_states, mirror_base + from.slot_states, nt * sizeof(ChipTaskSlotState));
    std::memcpy(out_base + to.completion_flags, mirror_base + from.completion_flags, nt * sizeof(std::atomic<uint8_t>));
    std::memcpy(out_base + to.fanin_pool, mirror_base + from.fanin_pool, used.fanin_elems * sizeof(int32_t));
    std::memcpy(out_base + to.tensor_pool, mirror_base + from.tensor_pool, used.tensor_elems * sizeof(ChipTensor));
    std::memcpy(out_base + to.scalar_pool, mirror_base + from.scalar_pool, used.scalar_elems * sizeof(uint64_t));

    auto *out_slots = reinterpret_cast<ChipTaskSlotState *>(out_base + to.slot_states);
    auto *out_descriptors = reinterpret_cast<PTO2TaskDescriptor *>(out_base + to.descriptors);
    auto *out_payloads = reinterpret_cast<PTO2TaskPayload *>(out_base + to.payloads);
    const auto *mirror_payloads = reinterpret_cast<const PTO2TaskPayload *>(mirror_base + from.payloads);
    auto *out_fanin = reinterpret_cast<int32_t *>(out_base + to.fanin_pool);
    auto *out_tensors = reinterpret_cast<ChipTensor *>(out_base + to.tensor_pool);
    auto *out_scalars = reinterpret_cast<uint64_t *>(out_base + to.scalar_pool);
    const auto *mirror_fanin = reinterpret_cast<const int32_t *>(mirror_base + from.fanin_pool);
    const auto *mirror_tensors = reinterpret_cast<const ChipTensor *>(mirror_base + from.tensor_pool);
    const auto *mirror_scalars = reinterpret_cast<const uint64_t *>(mirror_base + from.scalar_pool);
    // An unbound region stays unbound: a Graph node's payload never gets a fanin
    // region, and its count is 0, so no consumer resolves it. A bound one is inside
    // its own mirror pool by construction — the only binder is a bump cursor on that
    // pool — and the translation below depends on it, so it is asserted rather than
    // re-derived.
    for (uint64_t i = 0; i < nt; ++i) {
        out_slots[i].bind_buffers(&out_payloads[i], &out_descriptors[i]);
        const PTO2TaskPayload &src = mirror_payloads[i];
        const ChipTensor *src_tensors = src.tensor_data();
        const uint64_t *src_scalars = src.scalar_data();
        const int32_t *src_fanin = src.fanin_data();
        debug_assert(
            src_tensors == nullptr ||
            (src_tensors >= mirror_tensors && src_tensors <= mirror_tensors + used.tensor_elems)
        );
        debug_assert(
            src_scalars == nullptr ||
            (src_scalars >= mirror_scalars && src_scalars <= mirror_scalars + used.scalar_elems)
        );
        debug_assert(
            src_fanin == nullptr || (src_fanin >= mirror_fanin && src_fanin <= mirror_fanin + used.fanin_elems)
        );
        out_payloads[i].bind_regions(
            src_tensors == nullptr ? nullptr : out_tensors + (src_tensors - mirror_tensors),
            src_scalars == nullptr ? nullptr : out_scalars + (src_scalars - mirror_scalars),
            src_fanin == nullptr ? nullptr : out_fanin + (src_fanin - mirror_fanin)
        );
    }
    return to.end;
}

}  // namespace pto2_sm_layout
