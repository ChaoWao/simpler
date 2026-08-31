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
 * Shared Memory Layout
 *
 * Defines the shared memory structure for Orchestrator-Scheduler communication.
 *
 * Memory Layout:
 *   +---------------------------+
 *   | SharedMemoryHeader        |  (completion watermark + scheduler error state)
 *   +---------------------------+
 *   | ChipTaskStorage[]         |  (descriptor + slot state + payload, per task)
 *   | std::atomic<uint8_t>[]    |  (completion flags, one byte per task)
 *   +---------------------------+
 *   | fanin / tensor / scalar   |  (the argument pools payloads name by delta)
 *   +---------------------------+
 *
 * Design principles:
 * - Only data needed for Orchestrator<->Scheduler communication is here
 * - TensorMap, scope_stack and ready_queues are in private memory
 * - Synchronization via atomic counters/flags (no locks needed for single-word R/W)
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <stddef.h>

#include <cstring>

#include "utils/device_arena.h"
#include "graph_execution.h"
#include "host_build_graph/runtime_types.h"

// =============================================================================
// Shared Memory Header
// =============================================================================

struct SharedMemoryHandle;

/**
 * The task table's header in shared memory.
 *
 * Groups the completion watermark, the run's task total, and the pointers to the
 * four slot-pitched segments. Pointers are host-side only (set by setup_pointers,
 * invalid on device).
 *
 * The run's task total sits here too, as a plain scalar. The graph is complete
 * before the device starts, so the host writes it once into the mirror after
 * orchestration and the restack ships it with the rest of the header; the device
 * only ever reads it. Nothing publishes it incrementally, so it needs neither an
 * atomic nor a cache line of its own.
 */
struct alignas(64) SharedMemoryTaskHeader {
    // Highest task_id such that every task with id in [0, completed_watermark]
    // has its completion_flags byte set. Advanced over the full contiguous
    // completed prefix at task-completion time (on_mixed_task_complete). The host
    // consumer-wait gates on it: a producer slot P's consumers have all retired
    // once completed_watermark >= P.last_consumer_local_id. On its own cache line
    // (concurrent CAS-advance by completing threads).
    alignas(64) std::atomic<int32_t> completed_watermark;

    // The task storage array (host-side, set by setup_pointers). One entry per slot,
    // holding that task's descriptor, slot state and payload — see ChipTaskStorage.
    alignas(64) ChipTaskStorage *task_storage;

    // Polling-completion state (device-addressed array, one byte per slot).
    // 0 = pending, 1 = task fully COMPLETED. Writer = the task's completer at
    // on_mixed_task_complete; reader = consumer fanin polling (is_completion_flag_set).
    // Cleared per-slot in orch::prepare_task as each slot is claimed. Indexed by
    // local task id, like the storage array — so it covers GLOBAL tasks only. An
    // IN_GRAPH task holds no slot here and publishes completion through its own
    // ChipTaskSlotState::task_state instead; the Graph's outer shell is the GLOBAL
    // task that carries a flag for the whole body.
    //
    // A byte array of its own rather than a field of ChipTaskStorage: a fanin scan
    // reads many producers' flags at once, which one cache line answers here and
    // would take one line per producer inside the storage stride.
    //
    // A hidden-alloc task is the one flag the host presets to 1: it completes during
    // orchestration, and a consumer polls this array rather than task_state.
    std::atomic<uint8_t> *completion_flags;

    // Tasks this run submitted, i.e. the slot count the two segments above are
    // pitched to. Written once by the host after orchestration (run_host_orchestration)
    // and read-only from then on, so it packs into the padding rather than taking a
    // line of its own. Bounds the completed_watermark walk: no slot at or above it was
    // claimed, and the bytes past completion_flags[total_tasks - 1] are not flags.
    int32_t total_tasks;

    bool is_completion_flag_set(int32_t local_id, std::memory_order order = std::memory_order_acquire) const {
        return completion_flags[local_id].load(order) != 0;
    }

    void set_completion_flag(int32_t local_id, std::memory_order order = std::memory_order_release) const {
        completion_flags[local_id].store(1, order);
    }

    // set completion flag first before updating the watermark (logic requirement)
    void update_completed_watermark() {
        int32_t curr_watermark = completed_watermark.load(std::memory_order_acquire);

        int32_t next = curr_watermark;
        while (true) {
            while (next + 1 < total_tasks && is_completion_flag_set(next + 1)) {
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

    // A task id is its own storage index, so the three records it names are reached
    // by one index into one array.
    ChipTaskStorage &storage_at(int32_t local_id) { return task_storage[local_id]; }
    TaskDescriptor &get_task_by_task_id(int32_t local_id) { return task_storage[local_id].task; }
    ChipTaskSlotState &get_slot_state_by_task_id(int32_t local_id) { return task_storage[local_id].slot; }
};

static_assert(sizeof(SharedMemoryTaskHeader) == 128, "SharedMemoryTaskHeader layout drift");
static_assert(offsetof(SharedMemoryTaskHeader, task_storage) == 64, "SharedMemoryTaskHeader task_storage layout drift");
// The device reads this one out of the H2D'd header, so it is pinned separately from the
// segment pointers above, which are host-side only.
static_assert(offsetof(SharedMemoryTaskHeader, total_tasks) == 80, "SharedMemoryTaskHeader total_tasks layout drift");

/**
 * Shared memory header structure
 *
 * Contains the task table's header plus the scheduler's error state.
 */
struct alignas(CHIP_ALIGN_SIZE) SharedMemoryHeader {
    // === TASK TABLE HEADER (set once at init) ===
    SharedMemoryTaskHeader tasks;

    // === ERROR REPORTING ===

    // Scheduler error state. Written by scheduler threads on timeout; read by the
    // scheduler's own cold path and, after a failed run, by the host through a D2H
    // copy of this header. The orchestrator runs on the host and latches its own
    // fatal code in OrchestratorState, so no orchestrator error crosses here.
    std::atomic<uint32_t> sched_error_bitmap;  // Bit X set = thread X had error
    std::atomic<int32_t> sched_error_code;     // Last scheduler error code (last-writer-wins)
    std::atomic<int32_t> sched_error_thread;   // Thread index of last error writer
};

static_assert(sizeof(SharedMemoryHeader) == 192, "SharedMemoryHeader layout drift");
static_assert(
    offsetof(SharedMemoryHeader, sched_error_bitmap) == 128, "SharedMemoryHeader sched_error_bitmap layout drift"
);

// =============================================================================
// Shared Memory Handle
// =============================================================================

/**
 * Handle for shared memory lifecycle management (create/destroy).
 * Runtime components (orchestrator, scheduler) use SharedMemoryHeader* directly.
 */
struct SharedMemoryHandle {
    void *sm_base;     // Base address of shared memory
    uint64_t sm_size;  // Total size of shared memory

    SharedMemoryHeader *header;

    // Ownership flag
    bool is_owner;  // True if this handle allocated the memory

    // === Static helpers ===

    // Bytes an SM image spans when dimensioned for `max_tasks` slots — the count
    // the bind resolved from runtime_env.ring_task_window.
    static uint64_t calculate_size(uint64_t max_tasks);

    // UT convenience: reserve wrapper + sm_base on `arena`, commit, and init using
    // default CHIP_DEFAULT_GRAPH_TASKS. Only valid when the arena is otherwise empty
    // (the call performs the single commit). All memory is owned by the arena —
    // caller must not call destroy().
    static SharedMemoryHandle *create_and_init_default(DeviceArena &arena);

    // === Instance methods ===

    // In-place init for caller-provided wrapper storage (e.g. a region carved
    // out of a DeviceArena). Sets is_owner = false, calls setup_pointers and
    // init_header. Returns false when `sm_size` is too small for `max_tasks`.
    bool init(void *sm_base, uint64_t sm_size, uint64_t max_tasks);

    // Attach to an ALREADY-populated shared memory region: point the handle and
    // the task header's segment pointers (storage / completion flags)
    // at `sm_base`, but do NOT reset the watermark / slot states.
    // Used by host_build_graph host-orch, where the host orchestrator populated
    // the SM and H2D'd it; the device must re-point at its own SM base without
    // wiping the contents (unlike init, which also resets the header).
    //
    // `live_slots` is the pitch the uploaded arrays were laid out with — the
    // number of slots the host actually submitted, not the `max_tasks` the mirror
    // was dimensioned for. It must match what the host used or every segment past
    // the storage resolves to the wrong address, so both sides derive it from
    // the same count.
    // Every task id is below it, and indexes its slot directly.
    //
    // `image_bytes` is what the host shipped, pools included. The device cannot
    // recompute it — the pool extents are the bind's cursors, which only the host saw
    // — and it does not need to: a payload names its argument regions by delta, so no
    // pool base is resolved here. The value bounds the region and checks the int32
    // delta reach.
    bool
    attach_populated(void *sm_base, uint64_t sm_size, uint64_t max_tasks, uint64_t live_slots, uint64_t image_bytes);

    void destroy();

private:
    void init_header();
    // `pitch` is the slot count the arrays are dimensioned for. init passes the
    // mirror's reservation (what the orchestrator writes into); attach_populated
    // passes the submitted count (the compacted image that shipped).
    void setup_pointers(uint64_t pitch);
};

// =============================================================================
// SM Device Layout Helpers
// =============================================================================
//
// When the host pre-builds a runtime-arena image, it needs the device-side
// addresses of several SM sub-fields (the task header, the task storage
// array) so it can wire them into the scheduler init_data path without
// dereferencing the SM — the SM lives in device memory and cannot be touched
// from host.
//
// These helpers compute those addresses by offset arithmetic on the SM
// device base. Pure pointer math, no loads/stores; safe to call from host.
// The same arithmetic happens on AICPU too (via SharedMemoryHandle's
// own setup_pointers), so values are guaranteed consistent across sides.
namespace sm_layout {

inline SharedMemoryTaskHeader *task_header_addr(void *sm_dev_base) noexcept {
    return reinterpret_cast<SharedMemoryTaskHeader *>(
        static_cast<char *>(sm_dev_base) + offsetof(SharedMemoryHeader, tasks)
    );
}

// Byte offsets (from the SM base) of the image's segments. The layout is: header, then
// storage -> completion_flags -> the three argument pools, every segment
// CHIP_ALIGN_UP-padded. ImageExtents dimensions them: the mirror for the worst case the
// API allows, the image for what this bind holds, which is what makes the live prefixes
// contiguous and the upload one copy.
//
// One storage segment, not three: a task's descriptor, slot state and payload sit in one
// ChipTaskStorage, so their relative positions are that type's layout and survive the
// restack's change of pitch untouched.
//
// The pools sit last because nothing on the device resolves a segment past
// completion_flags: a payload names its argument regions by delta, so the two
// slot-pitched offsets are all the attach path computes.
struct SegmentOffsets {
    uint64_t storage;
    uint64_t completion_flags;  // polling-completion byte array (1 byte/slot)
    uint64_t fanin_pool;
    uint64_t tensor_pool;
    uint64_t scalar_pool;
    uint64_t end;  // offset just past the last segment (total image size)
};

// How many slots and how many pool elements a layout is dimensioned for.
struct ImageExtents {
    uint64_t slots;
    uint64_t fanin_elems;
    uint64_t tensor_elems;
    uint64_t scalar_elems;
};

// The mirror the orchestrator writes into: `max_tasks` slots, every pool sized so the
// worst case cannot overflow a bump — max_tasks tasks each at their full cap. That
// bound is what lets prepare_task advance a cursor with no capacity check.
inline ImageExtents mirror_extents(uint64_t max_tasks) noexcept {
    return ImageExtents{
        max_tasks,
        max_tasks * CHIP_MAX_FANIN,
        max_tasks * MAX_TENSOR_ARGS,
        max_tasks * MAX_SCALAR_ARGS,
    };
}

// Single source of truth for the SM segment layout. Returns offsets (not pointers),
// so it serves BOTH the host-side pointer setup (`setup_pointers`, which adds
// `sm_base`) and the device-address helpers below (which add `sm_dev_base`). Adding
// or reordering a segment is a one-line edit here; every consumer follows
// automatically, so the layout walk can never silently disagree across call sites.
inline SegmentOffsets segment_offsets(const ImageExtents &e) noexcept {
    uint64_t off = CHIP_ALIGN_UP(sizeof(SharedMemoryHeader), CHIP_ALIGN_SIZE);
    SegmentOffsets o{};
    o.storage = off;
    off += CHIP_ALIGN_UP(e.slots * sizeof(ChipTaskStorage), CHIP_ALIGN_SIZE);
    o.completion_flags = off;
    off += CHIP_ALIGN_UP(e.slots * sizeof(std::atomic<uint8_t>), CHIP_ALIGN_SIZE);
    o.fanin_pool = off;
    off += CHIP_ALIGN_UP(e.fanin_elems * sizeof(int32_t), CHIP_ALIGN_SIZE);
    o.tensor_pool = off;
    off += CHIP_ALIGN_UP(e.tensor_elems * sizeof(simpler::hbg::Tensor), CHIP_ALIGN_SIZE);
    o.scalar_pool = off;
    off += CHIP_ALIGN_UP(e.scalar_elems * sizeof(uint64_t), CHIP_ALIGN_SIZE);
    o.end = off;
    return o;
}

inline SegmentOffsets segment_offsets(uint64_t max_tasks) noexcept {
    return segment_offsets(mirror_extents(max_tasks));
}

// Every per-task region starts on a cache line, which TaskPayload::init's
// round-up scalar memcpy relies on. simpler::hbg::Tensor is 2 cache lines, so a tensor region
// is aligned for any count; the fanin and scalar strides need it stated.
static_assert(
    (CHIP_MAX_FANIN * sizeof(int32_t)) % ARG_POOL_ALIGN == 0,
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

inline ImageExtents image_extents(const BindUsage &used) noexcept {
    return ImageExtents{
        live_slot_pitch(used.submitted_tasks),
        used.fanin_elems,
        used.tensor_elems,
        used.scalar_elems,
    };
}

// Where the graph heap ended up, for the image's heap addresses.
//
// The orchestrator allocates the heap out of the HEAP_VIRTUAL_BASE window,
// because the heap's device region is committed only after orchestration, from
// the byte count orchestration turned out to need. `real_base` is that region's
// device base and `used_bytes` is what the allocator handed out, so the window
// actually in play is [HEAP_VIRTUAL_BASE, HEAP_VIRTUAL_BASE + used_bytes].
struct HeapRebase {
    uint64_t real_base;
    uint64_t used_bytes;
};

// Translate one address the image carries. Anything below HEAP_VIRTUAL_BASE is a
// real device address the caller owns — a boundary tensor, or an unset field left
// at 0 — and is returned untouched. At or above it, the address came from the
// graph heap: a recorded in-graph task's outputs live in its Definition as offsets, so no
// Graph-recording address (>= GRAPH_RECORD_VIRTUAL_BASE) reaches the image, and
// the committed-heap bound below rejects one rather than classifying by it.
inline uint64_t rebased_heap_addr(uint64_t addr, const HeapRebase &rebase) noexcept {
    if (addr < HEAP_VIRTUAL_BASE) {
        return addr;
    }
    const uint64_t offset = addr - HEAP_VIRTUAL_BASE;
    always_assert(offset <= rebase.used_bytes && "image address is outside the committed graph heap");
    return rebase.real_base + offset;
}

// Restack the live prefix of every segment from the mirror the
// orchestrator wrote into an image dimensioned for what this bind holds, where the
// prefixes are contiguous and can travel as one copy.
//
// `out_base` must be CHIP_ALIGN_SIZE-aligned and hold
// `segment_offsets(image_extents(used)).end` bytes. Returns that byte count.
//
// Three things the restack has to fix up, all because the image is not the mirror:
//
//   - the task header's segment pointers name the mirror's arrays, so they leave as
//     null rather than carrying host addresses into device memory (the device
//     resolves them in attach_populated);
//   - a payload names its three argument regions by a delta from the naming field's
//     own address; the restack changed those distances, so every one is re-taken
//     against the image. A region keeps its position within its pool, so the re-take
//     is the same arithmetic with the image's bases. A task's own three records need
//     no such re-take: they share one ChipTaskStorage, so their distances are that
//     type's layout and the change of pitch cannot reach them;
//   - every heap address the orchestrator wrote is in the HEAP_VIRTUAL_BASE window,
//     so `rebase` moves it onto the device region committed after orchestration.
//     Three fields carry one: a descriptor's packed buffer bounds (read on the
//     device by the Graph expansion in graph_execution.cpp), a payload's dispatch
//     predicate (dereferenced by the scheduler), and a tensor argument's buffer
//     address — the last one per task and in that task's own element type, since an
//     outer GRAPH task's boundaries are GraphTensors rather than ChipTensors.
//
// Those three are the whole surface: a heap address that reached the device through
// an untyped channel would not be moved, and the scalar pool cannot be swept for
// one, because a scalar's value is arbitrary and a quarter of the 64-bit range
// falls inside the window. Orchestration must therefore pass a runtime-created
// buffer as the tensor it got back, never as a scalar carrying its address.
inline uint64_t compact_live_image(
    const char *mirror_base, uint64_t max_tasks, const BindUsage &used, const HeapRebase &rebase, char *out_base
) noexcept {
    // The mirror is dimensioned for the worst case, so a live count or a cursor past
    // it reads beyond the segment it is copying from and ships a corrupt image.
    // attach_populated tests the slot bound again on the device side.
    const ImageExtents mirror = mirror_extents(max_tasks);
    always_assert(used.submitted_tasks <= mirror.slots);
    always_assert(used.fanin_elems <= mirror.fanin_elems);
    always_assert(used.tensor_elems <= mirror.tensor_elems);
    always_assert(used.scalar_elems <= mirror.scalar_elems);
    const SegmentOffsets from = segment_offsets(mirror);
    const SegmentOffsets to = segment_offsets(image_extents(used));

    // The header and the storage offset are pitch-independent, so the header
    // lands where it already was.
    std::memcpy(out_base, mirror_base, to.storage);
    auto &out_tasks = reinterpret_cast<SharedMemoryHeader *>(out_base)->tasks;
    out_tasks.task_storage = nullptr;
    out_tasks.completion_flags = nullptr;

    const uint64_t nt = used.submitted_tasks;
    // One copy for all three of a task's records: ChipTaskStorage is fixed-size, so
    // the mirror and the image share a stride. Each pool is likewise one copy of its
    // own prefix.
    std::memcpy(out_base + to.storage, mirror_base + from.storage, nt * sizeof(ChipTaskStorage));
    std::memcpy(out_base + to.completion_flags, mirror_base + from.completion_flags, nt * sizeof(std::atomic<uint8_t>));
    std::memcpy(out_base + to.fanin_pool, mirror_base + from.fanin_pool, used.fanin_elems * sizeof(int32_t));
    std::memcpy(
        out_base + to.tensor_pool, mirror_base + from.tensor_pool, used.tensor_elems * sizeof(simpler::hbg::Tensor)
    );
    std::memcpy(out_base + to.scalar_pool, mirror_base + from.scalar_pool, used.scalar_elems * sizeof(uint64_t));

    auto *out_storage = reinterpret_cast<ChipTaskStorage *>(out_base + to.storage);
    const auto *mirror_storage = reinterpret_cast<const ChipTaskStorage *>(mirror_base + from.storage);
    auto *out_fanin = reinterpret_cast<int32_t *>(out_base + to.fanin_pool);
    auto *out_tensors = reinterpret_cast<simpler::hbg::Tensor *>(out_base + to.tensor_pool);
    auto *out_scalars = reinterpret_cast<uint64_t *>(out_base + to.scalar_pool);
    const auto *mirror_fanin = reinterpret_cast<const int32_t *>(mirror_base + from.fanin_pool);
    const auto *mirror_tensors = reinterpret_cast<const simpler::hbg::Tensor *>(mirror_base + from.tensor_pool);
    const auto *mirror_scalars = reinterpret_cast<const uint64_t *>(mirror_base + from.scalar_pool);
    // An unbound region stays unbound: an in-graph task's payload never gets a fanin
    // region, and its count is 0, so no consumer resolves it. A bound one is inside
    // its own mirror pool by construction — the only binder is a bump cursor on that
    // pool — and the translation below depends on it, so it is asserted rather than
    // re-derived.
    for (uint64_t i = 0; i < nt; ++i) {
        ChipTaskStorage &out_entry = out_storage[i];
        TaskPayload &out_payload = out_entry.payload;
        const TaskPayload &src = mirror_storage[i].payload;
        const simpler::hbg::Tensor *src_tensors = src.tensor_data();
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
        out_payload.bind_regions(
            src_tensors == nullptr ? nullptr : out_tensors + (src_tensors - mirror_tensors),
            src_scalars == nullptr ? nullptr : out_scalars + (src_scalars - mirror_scalars),
            src_fanin == nullptr ? nullptr : out_fanin + (src_fanin - mirror_fanin)
        );
        // The tensor pool holds two element types, so each task's own region is walked
        // with the type that task wrote: an outer GRAPH task's boundaries are
        // GraphTensors packed at their own stride, merely occupying the number of
        // simpler::hbg::Tensor slots graph_boundary_tensor_pool_slots reserves for them. Walking
        // the pool itself as one simpler::hbg::Tensor array would reach only the first boundary
        // of each Graph and rewrite bytes in the middle of the rest.
        if (out_entry.slot.task_kind == TaskKind::GRAPH) {
            auto *boundaries = reinterpret_cast<GraphTensor *>(out_payload.tensor_data());
            for (int32_t j = 0; j < out_payload.tensor_count; ++j) {
                boundaries[j].buffer_addr = rebased_heap_addr(boundaries[j].buffer_addr, rebase);
            }
        } else {
            simpler::hbg::Tensor *tensors = out_payload.tensor_data();
            for (int32_t j = 0; j < out_payload.tensor_count; ++j) {
                tensors[j].buffer.addr = rebased_heap_addr(tensors[j].buffer.addr, rebase);
            }
        }
        TaskDescriptor &out_task = out_entry.task;
        out_task.packed_buffer_base = reinterpret_cast<void *>(
            rebased_heap_addr(reinterpret_cast<uint64_t>(out_task.packed_buffer_base), rebase)
        );
        out_task.packed_buffer_end =
            reinterpret_cast<void *>(rebased_heap_addr(reinterpret_cast<uint64_t>(out_task.packed_buffer_end), rebase));
        out_payload.predicate.addr = rebased_heap_addr(out_payload.predicate.addr, rebase);
    }
    return to.end;
}

}  // namespace sm_layout
