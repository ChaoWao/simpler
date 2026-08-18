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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/chip_swimlane_profiling.h"
#include "common/platform_config.h"

/**
 * Rotating pool of host phase records, written on host_build_graph's prepare
 * path and read by whichever diagnostics are enabled for the run.
 *
 * The pool is a platform resource with two independently gated readers — the
 * chip-swimlane export at level 4, and the per-event artifact the host trace
 * tooling consumes — so it belongs to neither of them. Storage and lifetime are
 * the platform's; the write cursor lives in the pool head, exactly as the
 * device sched/orch pools put it in shared memory for the AICPU to advance.
 *
 * Recording is a header-inline store into platform-allocated memory rather than
 * a call through HostApi: a pass performs a few hundred operations a few
 * microseconds apart on the very path whose budget is under measurement, so the
 * per-event cost must stay at two clock reads and a store. HostApi carries only
 * the once-per-pass arm and finish.
 */

namespace simpler::dfx {

/**
 * Allocation, lifetime and reader access for one host phase pool.
 *
 * Held by the platform runner. `arm()` is called once per prepare pass, before
 * the first bind segment; `records()` is valid from `finish()` until the next
 * `arm()`.
 */
class HostPhaseRecordStore {
public:
    HostPhaseRecordStore() = default;
    // Neither copying nor moving is representable: pool_ holds raw pointers into
    // buffers_, and the producer holds &pool_ for the duration of a pass. A copy
    // would alias the source's buffers and a move would invalidate the pointer
    // already handed out, both silently.
    HostPhaseRecordStore(const HostPhaseRecordStore &) = delete;
    HostPhaseRecordStore &operator=(const HostPhaseRecordStore &) = delete;
    HostPhaseRecordStore(HostPhaseRecordStore &&) = delete;
    HostPhaseRecordStore &operator=(HostPhaseRecordStore &&) = delete;

    /**
     * Arm a fresh pass.
     *
     * @param collect_records  whether per-event records are wanted at all
     *                         (a per-event reader is enabled for this run)
     * @return the pool to write into, or nullptr when nothing is collected
     */
    HostPhaseRecordPool *arm(bool collect_records);

    /**
     * Close the pass.
     *
     * @param submitted_tasks  what the producer reported submitting, for a
     *                         reader's completeness check
     * @param invocation_id    the run's `[STRACE]` invocation id, so an artifact
     *                         can be joined to that run's span tree
     */
    void finish(uint64_t submitted_tasks, uint64_t invocation_id);

    /**
     * Append this pass's records to `path` as one JSON Lines object.
     *
     * @return 0 on success, -1 when the path cannot be written
     */
    int write_records_jsonl(const std::string &path) const;

    /** Records in rotation order. Empty unless a pass was armed with collection. */
    std::vector<HostPhaseRecord> records() const;

    /** Records whose kind submits a task, in rotation order. */
    std::vector<HostPhaseRecord> submit_records() const;

    /** Records whose kind is a host-to-device transfer, in rotation order. */
    std::vector<HostPhaseRecord> device_upload_records() const;

    bool armed() const { return armed_; }
    bool finished() const { return finished_; }
    uint64_t submitted_tasks() const { return submitted_tasks_; }
    uint64_t invocation_id() const { return invocation_id_; }
    uint64_t total_records() const { return pool_.head.total_record_count; }
    uint64_t dropped_records() const { return pool_.head.dropped_record_count; }
    /** Capacity in records, i.e. how many fit before any is dropped. */
    static constexpr size_t capacity() {
        return static_cast<size_t>(PLATFORM_HOST_PHASE_BUFFERS) * PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER;
    }

private:
    HostPhaseRecordPool pool_{};
    std::vector<HostPhaseRecordBuffer> buffers_{};
    bool armed_{false};
    bool finished_{false};
    uint64_t submitted_tasks_{0};
    uint64_t invocation_id_{0};
};

}  // namespace simpler::dfx

/**
 * Take the next free buffer as the active one.
 *
 * Cold path: reached once per PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER records, and
 * on the pass's first record. Returns nullptr when the free queue is exhausted,
 * which is the pool's only lossy outcome and is counted in the head.
 */
inline HostPhaseRecordBuffer *host_phase_pool_rotate(HostPhaseRecordPool *pool) {
    ChipSwimlaneFreeQueue &free_queue = pool->free_queue;
    if (free_queue.head == free_queue.tail) return nullptr;
    const uint32_t slot = free_queue.head % PLATFORM_PROF_SLOT_COUNT;
    auto *next = reinterpret_cast<HostPhaseRecordBuffer *>(free_queue.buffer_ptrs[slot]);
    free_queue.head = free_queue.head + 1;
    pool->head.current_buf_ptr = reinterpret_cast<uint64_t>(next);
    pool->head.current_buf_seq = pool->head.current_buf_seq + 1;
    return next;
}

/**
 * Append one record. A null pool means this pass collects no records, which is
 * the common case and costs one branch.
 *
 * @param start_ns,end_ns  host monotonic nanoseconds
 * @param payload          task id for kinds that submit a task, else a detail count
 * @param index            submit_idx for orchestrator operations, 0 for bind segments
 */
inline void host_phase_pool_append(
    HostPhaseRecordPool *pool, HostPhaseKind kind, uint64_t start_ns, uint64_t end_ns, uint64_t payload, uint32_t index
) {
    if (pool == nullptr) return;
    pool->head.total_record_count = pool->head.total_record_count + 1;
    auto *active = reinterpret_cast<HostPhaseRecordBuffer *>(pool->head.current_buf_ptr);
    if (active == nullptr || active->count >= PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER) {
        active = host_phase_pool_rotate(pool);
        if (active == nullptr) {
            pool->head.dropped_record_count = pool->head.dropped_record_count + 1;
            return;
        }
    }
    active->records[active->count] = HostPhaseRecord{start_ns, end_ns, payload, static_cast<uint32_t>(kind), index};
    active->count = active->count + 1;
}
