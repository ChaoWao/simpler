/*
 * Host-only assembly of one L2 runtime, so the four lines
 *
 *     rt_scope_begin(rt);
 *     entry_points->entry(orch_l2);
 *     rt_scope_end(rt);
 *     rt_orchestration_done(rt);
 *
 * can be driven without CANN, without a device, and without an orchestration
 * .so.
 *
 * WHERE THIS SEQUENCE COMES FROM
 * ------------------------------
 * It is the host half of simpler's own host_build_graph path, in call order:
 *
 *   runtime_maker.cpp:831   sm_size = calculate_size_per_ring(task_window)
 *   runtime_maker.cpp:834   DeviceArena host_arena
 *   runtime_maker.cpp:835   runtime_reserve_layout(arena, task_window, heap)
 *   runtime_maker.cpp:836   arena.commit()
 *   runtime_maker.cpp:885   runtime_init_data_from_layout(..., PTO2_MODE_EXECUTE, ...)
 *   runtime_maker.cpp:891   runtime_wire_arena_pointers(arena, layout, rt)
 *   run_host_orchestration:487   host SM buffer + memset of the header segment
 *   run_host_orchestration:493   orchestrator.init_data_from_layout(host SM)
 *   run_host_orchestration:499   orchestrator.wire_arena_pointers(&rt->scheduler)
 *   run_host_orchestration:502   host_sm_handle.init_per_ring(...)
 *   run_host_orchestration:518   runtime_finalize_after_wire(rt, aic, aiv)
 *   run_host_orchestration:520   rt->mode = PTO2_MODE_EXECUTE
 *   run_host_orchestration:534   entry_points->bind(rt)
 *
 * THREE SUBSTITUTIONS, AND WHY EACH IS SOUND
 * ------------------------------------------
 * 1. `gm_heap` is a host malloc rather than a device GM allocation. The
 *    orchestrator only does address arithmetic on it — it hands out slices as
 *    task output buffers and never dereferences one (the AICore would). A host
 *    address is therefore indistinguishable to the code under measurement.
 *
 * 2. The SM is a plain host buffer. That is not a substitution at all: the
 *    host-orch path already runs the orchestrator against a host SM mirror
 *    (run_host_orchestration:487-499) and only H2Ds the populated image
 *    afterwards. We stop before the H2D.
 *
 * 3. `entry_points->bind(rt)` becomes a direct framework_bind_runtime(rt) call.
 *    In simpler the orchestration code is a dlopen'd .so, so binding has to go
 *    through an exported symbol; here it is linked into the same binary and the
 *    function is the same one (orchestration/common.cpp:42).
 *
 * WHAT IS DELIBERATELY ABSENT
 * ---------------------------
 * Everything after mark_done: upload_graph_submissions, the host->device
 * pointer relocation (runtime_maker.cpp:558+), the H2D of the SM and arena, and
 * the device-side scheduler boot. This package measures graph CONSTRUCTION, not
 * execution. No task ever runs.
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.h"
#include "pto_orchestrator.h"
#include "pto_runtime2.h"
#include "pto_shared_memory.h"
#include "pto_types.h"
#include "utils/device_arena.h"

namespace l2_bench {

struct HarnessConfig {
    uint64_t task_window = 8192;            // ring task slots
    uint64_t heap_bytes = 1ULL << 30;       // GM heap stand-in, per ring
    int32_t aic_count = 24;                 // MIX clusters this "run" advertises
    int32_t aiv_count = 48;                 // AIV cores
    uint64_t callable_hash = 0x1220BE0CULL;  // rt->active_callable_hash
    // DIAGNOSTIC ONLY. Touch every page of the SM before the run.
    //
    // Nothing is ever reclaimed here, so each submit writes into a slot that has
    // never been touched: with task_window=8192 the payload region alone is
    // ~38 MB, and every submit pays a cold miss plus a first-touch page fault on
    // its own ~4.8 KB slot. Production wraps the ring and reuses warm slots, so
    // that cost is an artefact of this harness. Setting this pre-faults the
    // region so the difference can be measured instead of assumed.
    //
    // Semantically safe: every SM byte the engine reads is written first
    // (init-on-write in prepare_task / payload.init), so pre-zeroing cannot
    // change what it observes.
    bool prefault_sm = false;
    // Same diagnostic for the runtime arena, which holds the TensorMap buckets
    // and entry pool that STEP 3/4 walk. Pre-faulting only the SM would leave
    // the arena cold and mis-attribute its first-touch cost to the lookup.
    bool prefault_arena = false;
};

// One complete runtime, torn down on destruction. Non-copyable: PTO2Runtime
// lives inside the arena and holds pointers back into it, so the whole thing is
// pinned in place for its lifetime.
class Harness {
public:
    explicit Harness(const HarnessConfig &cfg) : cfg_(cfg) { build(); }

    Harness(const Harness &) = delete;
    Harness &operator=(const Harness &) = delete;

    PTO2Runtime *rt() const { return rt_; }
    PTO2OrchestratorState &orch() const { return rt_->orchestrator; }

    // Task slots consumed so far — the engine's own count, read straight off
    // the ring allocator rather than tallied by the driver.
    int32_t active_task_count() const { return rt_->orchestrator.ring.task_allocator.active_count(); }

    // Set by mark_done(); the scheduler's "no more work is coming" signal.
    bool orchestration_done() const {
        return rt_->orchestrator.sm_header->orchestrator_done.load(std::memory_order_acquire) != 0;
    }

    // Non-zero once the orchestrator latches a fatal. Any submit after that
    // point is a silent no-op, so a bench MUST check this before believing a
    // task count or a throughput number.
    int32_t error_code() const {
        return rt_->orchestrator.sm_header->orch_error_code.load(std::memory_order_acquire);
    }
    bool fatal() const { return rt_->orchestrator.fatal || error_code() != 0; }

private:
    void build() {
        task_window_[0] = cfg_.task_window;
        heap_sizes_[0] = cfg_.heap_bytes;

        sm_size_ = PTO2SharedMemoryHandle::calculate_size_per_ring(task_window_);

        layout_ = runtime_reserve_layout(arena_, task_window_, heap_sizes_);
        if (arena_.commit(DeviceArena::kDefaultBaseAlign) == nullptr) {
            throw std::runtime_error("DeviceArena::commit failed (arena_size=" + std::to_string(layout_.arena_size) + ")");
        }
        // Must run BEFORE init_data_from_layout writes into the arena, and it is
        // safe for the same reason as the SM: every arena byte the engine reads
        // is written by the init phases first.
        if (cfg_.prefault_arena) std::memset(arena_.base(), 0, arena_.total_size());

        // Stands in for the device GM heap. aligned_alloc keeps it on the same
        // 1024-byte granularity the device allocator guarantees, so the
        // orchestrator's alignment arithmetic sees what it would on-device.
        gm_heap_ = std::aligned_alloc(DeviceArena::kDefaultBaseAlign, round_up(cfg_.heap_bytes));
        if (gm_heap_ == nullptr) throw std::bad_alloc();

        sm_buf_.reset(new uint8_t[sm_size_]);
        void *sm = sm_buf_.get();
        if (cfg_.prefault_sm) std::memset(sm, 0, sm_size_);

        rt_ = runtime_init_data_from_layout(
            arena_, layout_, PTO2_MODE_EXECUTE, sm, sm_size_, gm_heap_, heap_sizes_
        );
        if (rt_ == nullptr) throw std::runtime_error("runtime_init_data_from_layout failed");
        runtime_wire_arena_pointers(arena_, layout_, rt_);

        // Init-on-write: only the fixed header segment is zeroed; per-slot
        // descriptors/payloads/slot_states are written by prepare_task as each
        // slot is claimed. Zeroing the whole SM here would both cost more and
        // hide a missing per-slot init.
        const pto2_sm_layout::PTO2RingSegmentOffsets segs =
            pto2_sm_layout::ring_segment_offsets(task_window_[0]);
        std::memset(sm, 0, segs.descriptors);

        if (!rt_->orchestrator.init_data_from_layout(
                layout_.orch, arena_, sm, gm_heap_, heap_sizes_[0], task_window_[0]
            )) {
            throw std::runtime_error("orchestrator.init_data_from_layout failed");
        }
        rt_->orchestrator.wire_arena_pointers(layout_.orch, arena_, &rt_->scheduler);

        if (!sm_handle_.init_per_ring(sm, sm_size_, task_window_, heap_sizes_)) {
            throw std::runtime_error("PTO2SharedMemoryHandle::init_per_ring failed");
        }

        // Fills rt->ops from the runtime's own s_runtime_ops table and sets the
        // orchestrator's core counts, which submit_task reads for its
        // require_sync_start deadlock check.
        runtime_finalize_after_wire(rt_, cfg_.aic_count, cfg_.aiv_count);
        rt_->mode = PTO2_MODE_EXECUTE;
        rt_->active_callable_hash = cfg_.callable_hash;
        // No host tensor views are staged, so get_tensor_data/set_tensor_data
        // fail closed rather than dereferencing a device address. Payloads that
        // read tensor contents during orchestration are out of scope here.
        rt_->tensor_access = nullptr;

        framework_bind_runtime(rt_);
    }

    static size_t round_up(uint64_t n) {
        const uint64_t a = DeviceArena::kDefaultBaseAlign;
        return static_cast<size_t>((n + a - 1) & ~(a - 1));
    }

    struct HeapDeleter {
        void operator()(void *p) const { std::free(p); }
    };

    HarnessConfig cfg_;
    uint64_t task_window_[PTO2_MAX_RING_DEPTH]{};
    uint64_t heap_sizes_[PTO2_MAX_RING_DEPTH]{};
    uint64_t sm_size_{0};

    DeviceArena arena_;
    PTO2RuntimeArenaLayout layout_{};
    std::unique_ptr<uint8_t[]> sm_buf_;
    void *gm_heap_{nullptr};
    PTO2SharedMemoryHandle sm_handle_{};
    PTO2Runtime *rt_{nullptr};

public:
    ~Harness() {
        framework_bind_runtime(nullptr);
        if (rt_ != nullptr) runtime_destroy(rt_, arena_);
        if (gm_heap_ != nullptr) std::free(gm_heap_);
    }
};

}  // namespace l2_bench
