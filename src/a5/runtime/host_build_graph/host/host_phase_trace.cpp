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
 * Prepare-path timing: per-kind counters and the binding to the platform's
 * record pool. See runtime/host_phase_trace.h.
 */

#include "../runtime/host_phase_trace.h"

#include <stdio.h>

#include <atomic>
#include <array>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "common/host_api.h"
#include "common/log_clock.h"
#include "common/strace.h"
#include "common/unified_log.h"
#include "host/host_phase_records.h"

namespace {

// Bind segments carry a few attributes each beyond their duration (byte counts,
// tensor counts). They are formatted when the segment ends and printed at flush,
// so a segment's values need not outlive it and the log write stays off the
// measured path.
constexpr size_t kBindAttrsCapacity = 96;

// One producer's counters and pass-lifetime lease, on its own cache lines.
//
// The producers of a pass are independent -- each records its own Definition and
// counts only its own operations -- so nothing here is shared, and the per-record
// path performs no atomic read-modify-write at all. Sharing them cost the emitting
// thread about 0.8 us per record at eight producers, purely in cache-line
// ownership: every producer of a Graph workload records the same kind
// (`record_node`), so a per-kind counter is a single line eight threads fight
// over, and the `alignas(64)` that separates one kind from another does nothing
// about that.
//
// Plain integers, not atomics, and that is only sound because a lane has exactly
// one writer for the whole pass -- never a shared fallback -- and the reader sums
// the lanes after the pass is closed and its producers drained. A producer that
// cannot get a lane of its own does not record.
struct KindCounter {
    uint64_t total_ns{0};
    uint64_t count{0};
    uint64_t payload_sum{0};
    uint64_t first_start_ns{0};
};

struct alignas(64) ProducerLane {
    // Producers admitted by `active` and not yet finished with the counters and
    // pool. begin/end clear `active` and drain these leases before reusing a lane.
    std::atomic<int32_t> leased{0};
    std::array<KindCounter, kHostPhaseKindCount> counters{};
};

// One lane per thread that can record: the submitting thread plus one per
// concurrently-recorded Definition, so the ceiling is GRAPH_MAX_DEFINITIONS + 1 =
// 17. Matches the pool's buffer count, which is sized the same way, so a producer
// that got a buffer also has a lane and neither denial is reachable in a sized
// run.
constexpr size_t kProducerLanes = PLATFORM_HOST_PHASE_BUFFERS;

struct TraceState {
    std::atomic<HostPhaseRecordPool *> pool{nullptr};
    const HostApi *api;
    std::atomic<bool> active{false};
    std::atomic<uint32_t> generation{0};
    std::atomic<uint32_t> next_lane{0};
    // Records dropped because every lane was taken. Unreachable while the lane
    // count covers the producer ceiling; counted rather than assumed so a workload
    // that outgrows it says so instead of silently under-reporting.
    std::atomic<uint64_t> lane_denied{0};
    std::mutex lifecycle_mutex;
    std::atomic<uint64_t> submitted_tasks{0};
    std::array<ProducerLane, kProducerLanes> lanes;
    std::array<std::array<char, kBindAttrsCapacity>, kHostPhaseKindCount> bind_attrs;
};

TraceState &state() {
    static TraceState s{};
    return s;
}

struct ProducerCursor {
    TraceState *owner{nullptr};
    ProducerLane *lane{nullptr};
    uint32_t generation{0};
    uint32_t depth{0};
    bool denied{false};
};

ProducerCursor &producer_cursor() {
    static thread_local ProducerCursor cursor;
    return cursor;
}

void release_producer(ProducerCursor &cursor) {
    ProducerLane *lane = cursor.lane;
    cursor = {};
    if (lane != nullptr) lane->leased.fetch_sub(1, std::memory_order_release);
}

// Called with `active` already false, so no new producer can be admitted. A
// Graph commit normally joins every worker first; the drain also makes early
// error exits safe without charging two atomic RMWs to every record.
void drain_producers(TraceState &s) {
    for (ProducerLane &lane : s.lanes) {
        while (lane.leased.load(std::memory_order_acquire) != 0) {}
    }
}

bool is_bind_kind(uint32_t kind) { return kind < static_cast<uint32_t>(HostPhaseKind::OrchSubmitTask); }

// Opt-in spelling shared by this runtime's switches, matching the runtime's
// other default-off switch, SIMPLER_TMR_SERIAL_ORCH_SCHED_ENABLE, so that
// `=false` / `=off` / `=no` read as off rather than as a non-zero string.
bool env_opt_in(const char *name) {
    const char *env = std::getenv(name);
    return env != nullptr && (env[0] == '1' || env[0] == 't' || env[0] == 'T');
}

uint32_t current_thread_id() {
    // A thread's identity is fixed for its lifetime, so resolve it once. This
    // runs per record on the path being measured, where a syscall would inflate
    // the very durations it is annotating.
    static thread_local const uint32_t id = []() -> uint32_t {
#if defined(__linux__) && defined(SYS_gettid)
        return static_cast<uint32_t>(syscall(SYS_gettid));
#else
        return static_cast<uint32_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
#endif
    }();
    return id;
}

void record_counter(ProducerLane &lane, uint64_t start_ns, uint64_t end_ns, uint32_t kind, uint64_t payload) {
    KindCounter &counter = lane.counters[kind];
    if (counter.first_start_ns == 0) counter.first_start_ns = start_ns;
    counter.total_ns += end_ns - start_ns;
    counter.count++;
    counter.payload_sum += payload;
}

void append_record(TraceState &s, uint64_t start_ns, uint64_t end_ns, uint32_t kind, uint64_t payload, uint32_t index) {
    // No lock: a producer privately advances the buffer it claimed for this pass.
    // `pool` is republished only after every producer lease is drained.
    HostPhaseRecordPool *pool = s.pool.load(std::memory_order_acquire);
    if (pool == nullptr) return;
    host_phase_pool_append(
        pool, static_cast<HostPhaseKind>(kind), start_ns, end_ns, payload, index, current_thread_id()
    );
}

// A pass's totals for one kind, summed across the producers that recorded it.
// Valid only after the lanes are drained.
KindCounter merged_counter(const TraceState &s, uint32_t kind) {
    KindCounter merged{};
    for (const ProducerLane &lane : s.lanes) {
        const KindCounter &counter = lane.counters[kind];
        merged.total_ns += counter.total_ns;
        merged.count += counter.count;
        merged.payload_sum += counter.payload_sum;
        if (counter.first_start_ns != 0 &&
            (merged.first_start_ns == 0 || counter.first_start_ns < merged.first_start_ns)) {
            merged.first_start_ns = counter.first_start_ns;
        }
    }
    return merged;
}

void reset_lanes(TraceState &s) {
    for (ProducerLane &lane : s.lanes) {
        lane.counters = {};
    }
}

}  // namespace

bool host_phase_breakdown_enabled() {
    // Read once: the value cannot change within a process, and the orchestrator
    // asks per submit-level operation.
    static const bool enabled = env_opt_in("SIMPLER_HBG_BIND_BREAKDOWN_ENABLE");
    return enabled;
}

bool host_phase_records_enabled() {
    static const bool enabled = env_opt_in("SIMPLER_HBG_HOST_PHASE_RECORDS_ENABLE");
    return enabled;
}

void host_phase_producer_begin() {
    TraceState &s = state();
    ProducerCursor &cursor = producer_cursor();
    if (cursor.depth != 0) {
        ++cursor.depth;
        return;
    }
    if (!s.active.load(std::memory_order_acquire)) return;

    const uint32_t generation = s.generation.load(std::memory_order_acquire);
    const uint32_t index = s.next_lane.fetch_add(1, std::memory_order_relaxed);
    ProducerLane *lane = index < kProducerLanes ? &s.lanes[index] : nullptr;
    if (lane == nullptr) {
        cursor = ProducerCursor{&s, nullptr, generation, 1, true};
        return;
    }

    // Publish the lease before re-checking the pass boundary. trace_end clears
    // active before draining, so it either observes this lease or this producer
    // observes the closed/replaced pass and withdraws.
    lane->leased.fetch_add(1, std::memory_order_acq_rel);
    if (!s.active.load(std::memory_order_acquire) || s.generation.load(std::memory_order_acquire) != generation) {
        lane->leased.fetch_sub(1, std::memory_order_release);
        return;
    }
    cursor = ProducerCursor{&s, lane, generation, 1, false};
}

void host_phase_producer_end() {
    ProducerCursor &cursor = producer_cursor();
    if (cursor.depth == 0) return;
    if (--cursor.depth == 0) release_producer(cursor);
}

// Strong definitions overriding the orchestrator core's weak fallbacks, which
// exist so builds without this translation unit still link.
__attribute__((visibility("hidden"))) uint64_t host_phase_now_ns() {
    const ProducerCursor &cursor = producer_cursor();
    if (cursor.depth == 0 || cursor.owner != &state()) return 0;
    return static_cast<uint64_t>(simpler::log::monotonic_now_ns());
}

__attribute__((visibility("hidden"))) void
host_phase_record(uint64_t start_ns, uint64_t end_ns, uint32_t kind, uint64_t payload, uint32_t index) {
    TraceState &s = state();
    if (start_ns == 0 || kind >= kHostPhaseKindCount || end_ns < start_ns) {
        return;
    }
    ProducerCursor &cursor = producer_cursor();
    if (cursor.depth == 0 || cursor.owner != &s || cursor.denied || cursor.lane == nullptr) {
        s.lane_denied.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    record_counter(*cursor.lane, start_ns, end_ns, kind, payload);
    append_record(s, start_ns, end_ns, kind, payload, index);
}

void host_phase_record_bind(uint32_t kind, uint64_t start_ns, const char *attrs, uint64_t payload) {
    TraceState &s = state();
    if (start_ns == 0 || !is_bind_kind(kind)) return;
    ProducerCursor &cursor = producer_cursor();
    if (cursor.depth == 0 || cursor.owner != &s || cursor.denied || cursor.lane == nullptr) {
        s.lane_denied.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const uint64_t end_ns = static_cast<uint64_t>(simpler::log::monotonic_now_ns());
    if (attrs != nullptr) {
        snprintf(s.bind_attrs[kind].data(), kBindAttrsCapacity, "%s", attrs);
    }
    record_counter(*cursor.lane, start_ns, end_ns, kind, payload);
    append_record(s, start_ns, end_ns, kind, payload, 0);
}

void host_phase_trace_begin(const void *host_api) {
    TraceState &s = state();
    std::lock_guard<std::mutex> lock(s.lifecycle_mutex);
    s.active.store(false, std::memory_order_release);
    release_producer(producer_cursor());
    drain_producers(s);
    s.api = static_cast<const HostApi *>(host_api);
    HostPhaseRecordPool *pool =
        s.api != nullptr ?
            static_cast<HostPhaseRecordPool *>(s.api->host_phase_pool_arm(host_phase_records_enabled())) :
            nullptr;
    s.pool.store(pool, std::memory_order_release);
    // Timestamps are taken whenever either sink wants them. Gating the clock on
    // one switch alone would leave the other sink recording zeros.
    s.submitted_tasks.store(0, std::memory_order_relaxed);
    s.lane_denied.store(0, std::memory_order_relaxed);
    reset_lanes(s);
    for (auto &attrs : s.bind_attrs) {
        attrs[0] = '\0';
    }
    // Retires the producers' cached lanes, so a thread that recorded in the last
    // pass claims afresh rather than keeping a lane whose counters were just
    // cleared under it. Published before `active`, so no record can see the new
    // pass with a stale lane. The counter is process-wide monotonic rather than
    // per-pass-reset, so no two passes ever share an identity.
    s.next_lane.store(0, std::memory_order_relaxed);
    s.generation.fetch_add(1, std::memory_order_release);
    s.active.store(s.pool != nullptr || host_phase_breakdown_enabled(), std::memory_order_release);
    host_phase_producer_begin();
}

void host_phase_trace_note_submitted(uint64_t submitted_tasks) {
    state().submitted_tasks.store(submitted_tasks, std::memory_order_relaxed);
}

void host_phase_trace_end() {
    TraceState &s = state();
    std::lock_guard<std::mutex> lifecycle_lock(s.lifecycle_mutex);
    if (!s.active.load(std::memory_order_relaxed)) {
        return;
    }
    s.active.store(false, std::memory_order_release);
    release_producer(producer_cursor());
    // Every admitted producer finishes its counter and pool work before totals
    // are read and the pool is handed back.
    drain_producers(s);
    // Read the pool's own tally before handing it back, and drop the pointer
    // after: the pool belongs to the runner from finish() onward, and the pass is
    // over either way, so nothing here may outlive it. Clearing `active` also
    // makes a second end() without an intervening begin() a no-op rather than a
    // stale read plus a duplicate report.
    HostPhaseRecordPool *pool = s.pool.load(std::memory_order_relaxed);
    // Both ways a record can be lost: no buffer left in the pool, or no lane left
    // to count it in. The second is unreachable while the lane count covers the
    // producer ceiling, and is summed in rather than assumed away.
    const uint64_t dropped = (pool != nullptr ? pool->dropped.load(std::memory_order_relaxed) : 0) +
                             s.lane_denied.load(std::memory_order_relaxed);
    if (s.api != nullptr) {
        uint64_t inv = 0;
#if SIMPLER_HOST_STRACE
        // The invocation id the enclosing run's spans carry, so a per-event
        // artifact joins to them on (pid, inv).
        inv = simpler::strace::StraceScope::current_inv();
#endif
        s.api->host_phase_pool_finish(s.submitted_tasks.load(std::memory_order_relaxed), inv);
    }
    s.pool.store(nullptr, std::memory_order_release);
    if (!host_phase_breakdown_enabled()) {
        // Records were collected for a per-event reader alone, which has its own
        // report; this breakdown was not asked for.
        return;
    }

    for (uint32_t kind = 0; kind < kHostPhaseKindCount; ++kind) {
        const KindCounter counter = merged_counter(s, kind);
        const uint64_t count = counter.count;
        if (count == 0) {
            continue;
        }
        const uint64_t total_ns = counter.total_ns;
        const uint64_t payload_sum = counter.payload_sum;
        const uint64_t first_start_ns = counter.first_start_ns;
        const char *name = host_phase_kind_name(static_cast<HostPhaseKind>(kind));
        if (is_bind_kind(kind)) {
            // One occurrence per pass, so the summed time is the segment's own
            // duration and the twelve of them partition the bind stage. The line
            // carries its own start because every line is written at the end of
            // the pass, off the path being measured — the write time says nothing
            // about when the segment ran.
            LOG_TIMING(
                "bind phase=%s start_ns=%llu dur_ns=%llu %s", name, static_cast<unsigned long long>(first_start_ns),
                static_cast<unsigned long long>(total_ns), s.bind_attrs[kind].data()
            );
            continue;
        }
        // A summed cost share, not an interval: several of these kinds are
        // sub-operations of another, so there is no honest way to draw the total
        // on a timeline. Per-event bars come from the artifact instead.
        LOG_TIMING(
            "host-orch phase=%s total_ns=%llu count=%llu detail_sum=%llu dropped=%llu", name,
            static_cast<unsigned long long>(total_ns), static_cast<unsigned long long>(count),
            static_cast<unsigned long long>(payload_sum), static_cast<unsigned long long>(dropped)
        );
    }
}
