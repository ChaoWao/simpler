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

// record_node and graph_submit are updated concurrently by different cores.
// Keep phase counters on distinct cache lines so lock-free accounting does not
// turn that intended overlap into cache-line ownership ping-pong.
struct alignas(64) KindCounter {
    std::atomic<uint64_t> total_ns{0};
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> payload_sum{0};
    std::atomic<uint64_t> first_start_ns{0};
};
static_assert(sizeof(KindCounter) == 64);

struct TraceState {
    std::atomic<HostPhaseRecordPool *> pool{nullptr};
    const HostApi *api;
    std::atomic<bool> active{false};
    // Records accepted by `active` but not yet finished with the counters and the
    // pool. begin/end clear `active` and then drain this to zero, so a record can
    // never be reported by a pass it does not belong to.
    std::atomic<int32_t> in_flight{0};
    std::mutex lifecycle_mutex;
    std::mutex pool_mutex;
    std::atomic<uint64_t> submitted_tasks{0};
    std::array<KindCounter, kHostPhaseKindCount> counters;
    std::array<std::array<char, kBindAttrsCapacity>, kHostPhaseKindCount> bind_attrs;
};

TraceState &state() {
    static TraceState s{};
    return s;
}

// Publish-then-check against the trace lifecycle's clear-then-drain. Claiming a
// slot before reading `active` is what makes the pair race-free: a record that
// got in before the pass closed is waited for, and one that arrives after sees
// `active` false and withdraws.
class RecordAdmission {
public:
    explicit RecordAdmission(TraceState &s) :
        s_(s) {
        s_.in_flight.fetch_add(1, std::memory_order_acq_rel);
        admitted_ = s_.active.load(std::memory_order_acquire);
        if (!admitted_) s_.in_flight.fetch_sub(1, std::memory_order_acq_rel);
    }
    ~RecordAdmission() {
        if (admitted_) s_.in_flight.fetch_sub(1, std::memory_order_acq_rel);
    }

    RecordAdmission(const RecordAdmission &) = delete;
    RecordAdmission &operator=(const RecordAdmission &) = delete;

    explicit operator bool() const { return admitted_; }

private:
    TraceState &s_;
    bool admitted_{false};
};

// Called with `active` already false, so no new record can be admitted. Only the
// recording worker can still be inside one, and commit joins it before a pass
// ends, so in practice this returns without spinning.
void drain_in_flight_records(TraceState &s) {
    while (s.in_flight.load(std::memory_order_acquire) != 0) {}
}

bool is_bind_kind(uint32_t kind) { return kind < static_cast<uint32_t>(HostPhaseKind::OrchSubmitTask); }

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

void record_counter(TraceState &s, uint64_t start_ns, uint64_t end_ns, uint32_t kind, uint64_t payload) {
    KindCounter &counter = s.counters[kind];
    uint64_t unset = 0;
    (void)counter.first_start_ns.compare_exchange_strong(unset, start_ns, std::memory_order_relaxed);
    counter.total_ns.fetch_add(end_ns - start_ns, std::memory_order_relaxed);
    counter.count.fetch_add(1, std::memory_order_relaxed);
    counter.payload_sum.fetch_add(payload, std::memory_order_relaxed);
}

void append_record(TraceState &s, uint64_t start_ns, uint64_t end_ns, uint32_t kind, uint64_t payload, uint32_t index) {
    if (s.pool.load(std::memory_order_acquire) == nullptr) return;
    std::lock_guard<std::mutex> lock(s.pool_mutex);
    HostPhaseRecordPool *pool = s.pool.load(std::memory_order_relaxed);
    if (pool == nullptr) return;
    host_phase_pool_append(
        pool, static_cast<HostPhaseKind>(kind), start_ns, end_ns, payload, index, current_thread_id()
    );
}

void reset_counter(KindCounter &counter) {
    counter.total_ns.store(0, std::memory_order_relaxed);
    counter.count.store(0, std::memory_order_relaxed);
    counter.payload_sum.store(0, std::memory_order_relaxed);
    counter.first_start_ns.store(0, std::memory_order_relaxed);
}

}  // namespace

bool host_phase_timing_enabled() {
    // Read once: the value cannot change within a process, and the orchestrator
    // asks per submit-level operation. Opt-in spelling matches the runtime's
    // other default-off switch, SIMPLER_TMR_SERIAL_ORCH_SCHED_ENABLE, so that
    // `=false` / `=off` / `=no` read as off rather than as a non-zero string.
    static const bool enabled = [] {
        const char *env = std::getenv("SIMPLER_HBG_BIND_BREAKDOWN_ENABLE");
        return env != nullptr && (env[0] == '1' || env[0] == 't' || env[0] == 'T');
    }();
    return enabled;
}

// Strong definitions overriding the orchestrator core's weak fallbacks, which
// exist so builds without this translation unit still link.
__attribute__((visibility("hidden"))) uint64_t host_phase_now_ns() {
    if (!state().active.load(std::memory_order_acquire)) {
        return 0;
    }
    return static_cast<uint64_t>(simpler::log::monotonic_now_ns());
}

__attribute__((visibility("hidden"))) void
host_phase_record(uint64_t start_ns, uint64_t end_ns, uint32_t kind, uint64_t payload, uint32_t index) {
    TraceState &s = state();
    if (!s.active.load(std::memory_order_acquire) || kind >= kHostPhaseKindCount || end_ns < start_ns) {
        return;
    }
    // ORCH_PHASE_END calls this on every submit-level operation, so the load
    // above stays the whole cost when tracing is off. Admission then re-checks
    // under the in-flight claim, which is what actually closes the boundary.
    RecordAdmission admission(s);
    if (!admission) {
        return;
    }
    record_counter(s, start_ns, end_ns, kind, payload);
    append_record(s, start_ns, end_ns, kind, payload, index);
}

void host_phase_record_bind(uint32_t kind, uint64_t start_ns, const char *attrs, uint64_t payload) {
    TraceState &s = state();
    if (!s.active.load(std::memory_order_acquire) || !is_bind_kind(kind)) {
        return;
    }
    RecordAdmission admission(s);
    if (!admission) {
        return;
    }
    const uint64_t end_ns = static_cast<uint64_t>(simpler::log::monotonic_now_ns());
    if (attrs != nullptr) {
        snprintf(s.bind_attrs[kind].data(), kBindAttrsCapacity, "%s", attrs);
    }
    record_counter(s, start_ns, end_ns, kind, payload);
    append_record(s, start_ns, end_ns, kind, payload, 0);
}

void host_phase_trace_begin(const void *host_api) {
    TraceState &s = state();
    std::lock_guard<std::mutex> lock(s.lifecycle_mutex);
    s.active.store(false, std::memory_order_release);
    drain_in_flight_records(s);
    s.api = static_cast<const HostApi *>(host_api);
    HostPhaseRecordPool *pool =
        s.api != nullptr ? static_cast<HostPhaseRecordPool *>(s.api->host_phase_pool_arm(host_phase_timing_enabled())) :
                           nullptr;
    s.pool.store(pool, std::memory_order_release);
    // Timestamps are taken whenever either sink wants them. Gating the clock on
    // the environment variable alone would leave a level-4 run recording zeros.
    s.submitted_tasks.store(0, std::memory_order_relaxed);
    for (KindCounter &counter : s.counters)
        reset_counter(counter);
    for (auto &attrs : s.bind_attrs) {
        attrs[0] = '\0';
    }
    s.active.store(s.pool != nullptr || host_phase_timing_enabled(), std::memory_order_release);
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
    // Every record already past the `active` check finishes its counter and pool
    // work before the totals below are read and the pool is handed back.
    drain_in_flight_records(s);
    std::lock_guard<std::mutex> pool_lock(s.pool_mutex);
    // Read the pool's own tally before handing it back, and drop the pointer
    // after: the pool belongs to the runner from finish() onward, and the pass is
    // over either way, so nothing here may outlive it. Clearing `active` also
    // makes a second end() without an intervening begin() a no-op rather than a
    // stale read plus a duplicate report.
    HostPhaseRecordPool *pool = s.pool.load(std::memory_order_relaxed);
    const uint64_t dropped = pool != nullptr ? pool->head.dropped_record_count : 0;
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
    if (!host_phase_timing_enabled()) {
        // Records were collected for a per-event reader alone, which has its own
        // report; this breakdown was not asked for.
        return;
    }

    for (uint32_t kind = 0; kind < kHostPhaseKindCount; ++kind) {
        const KindCounter &counter = s.counters[kind];
        const uint64_t count = counter.count.load(std::memory_order_relaxed);
        if (count == 0) {
            continue;
        }
        const uint64_t total_ns = counter.total_ns.load(std::memory_order_relaxed);
        const uint64_t payload_sum = counter.payload_sum.load(std::memory_order_relaxed);
        const uint64_t first_start_ns = counter.first_start_ns.load(std::memory_order_relaxed);
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
