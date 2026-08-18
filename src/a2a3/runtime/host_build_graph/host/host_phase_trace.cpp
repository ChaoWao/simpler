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

#include <array>
#include <cstdlib>

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

struct KindCounter {
    uint64_t total_ns;
    uint64_t count;
    uint64_t payload_sum;
    uint64_t first_start_ns;
};

struct TraceState {
    HostPhaseRecordPool *pool;
    const HostApi *api;
    bool active;
    uint64_t submitted_tasks;
    std::array<KindCounter, kHostPhaseKindCount> counters;
    std::array<std::array<char, kBindAttrsCapacity>, kHostPhaseKindCount> bind_attrs;
};

TraceState &state() {
    static TraceState s{};
    return s;
}

bool is_bind_kind(uint32_t kind) { return kind < static_cast<uint32_t>(HostPhaseKind::OrchSubmitTask); }

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
    if (!state().active) {
        return 0;
    }
    return static_cast<uint64_t>(simpler::log::monotonic_now_ns());
}

__attribute__((visibility("hidden"))) void
host_phase_record(uint64_t start_ns, uint64_t end_ns, uint32_t kind, uint64_t payload, uint32_t index) {
    TraceState &s = state();
    if (!s.active || kind >= kHostPhaseKindCount || end_ns < start_ns) {
        return;
    }
    KindCounter &counter = s.counters[kind];
    if (counter.count == 0) {
        counter.first_start_ns = start_ns;
    }
    counter.total_ns += end_ns - start_ns;
    counter.count++;
    counter.payload_sum += payload;
    host_phase_pool_append(s.pool, static_cast<HostPhaseKind>(kind), start_ns, end_ns, payload, index);
}

void host_phase_record_bind(uint32_t kind, uint64_t start_ns, const char *attrs, uint64_t payload) {
    TraceState &s = state();
    if (!s.active || !is_bind_kind(kind)) {
        return;
    }
    const uint64_t end_ns = static_cast<uint64_t>(simpler::log::monotonic_now_ns());
    if (attrs != nullptr) {
        snprintf(s.bind_attrs[kind].data(), kBindAttrsCapacity, "%s", attrs);
    }
    host_phase_record(start_ns, end_ns, kind, payload, 0);
}

void host_phase_trace_begin(const void *host_api) {
    TraceState &s = state();
    s.api = static_cast<const HostApi *>(host_api);
    s.pool = s.api != nullptr ?
                 static_cast<HostPhaseRecordPool *>(s.api->host_phase_pool_arm(host_phase_timing_enabled())) :
                 nullptr;
    // Timestamps are taken whenever either sink wants them. Gating the clock on
    // the environment variable alone would leave a level-4 run recording zeros.
    s.active = s.pool != nullptr || host_phase_timing_enabled();
    s.submitted_tasks = 0;
    s.counters.fill(KindCounter{});
    for (auto &attrs : s.bind_attrs) {
        attrs[0] = '\0';
    }
}

void host_phase_trace_note_submitted(uint64_t submitted_tasks) { state().submitted_tasks = submitted_tasks; }

void host_phase_trace_end() {
    TraceState &s = state();
    if (!s.active) {
        return;
    }
    // Read the pool's own tally before handing it back, and drop the pointer
    // after: the pool belongs to the runner from finish() onward, and the pass is
    // over either way, so nothing here may outlive it. Clearing `active` also
    // makes a second end() without an intervening begin() a no-op rather than a
    // stale read plus a duplicate report.
    const uint64_t dropped = s.pool != nullptr ? s.pool->head.dropped_record_count : 0;
    if (s.api != nullptr) {
        uint64_t inv = 0;
#if SIMPLER_HOST_STRACE
        // The invocation id the enclosing run's spans carry, so a per-event
        // artifact joins to them on (pid, inv).
        inv = simpler::strace::StraceScope::current_inv();
#endif
        s.api->host_phase_pool_finish(s.submitted_tasks, inv);
    }
    s.pool = nullptr;
    s.active = false;
    if (!host_phase_timing_enabled()) {
        // Records were collected for a per-event reader alone, which has its own
        // report; this breakdown was not asked for.
        return;
    }

    for (uint32_t kind = 0; kind < kHostPhaseKindCount; ++kind) {
        const KindCounter &counter = s.counters[kind];
        if (counter.count == 0) {
            continue;
        }
        const char *name = host_phase_kind_name(static_cast<HostPhaseKind>(kind));
        if (is_bind_kind(kind)) {
            // One occurrence per pass, so the summed time is the segment's own
            // duration and the twelve of them partition the bind stage. The line
            // carries its own start because every line is written at the end of
            // the pass, off the path being measured — the write time says nothing
            // about when the segment ran.
            LOG_TIMING(
                "bind phase=%s start_ns=%llu dur_ns=%llu %s", name,
                static_cast<unsigned long long>(counter.first_start_ns),
                static_cast<unsigned long long>(counter.total_ns), s.bind_attrs[kind].data()
            );
            continue;
        }
        // A summed cost share, not an interval: several of these kinds are
        // sub-operations of another, so there is no honest way to draw the total
        // on a timeline. Per-event bars come from the artifact instead.
        LOG_TIMING(
            "host-orch phase=%s total_ns=%llu count=%llu detail_sum=%llu dropped=%llu", name,
            static_cast<unsigned long long>(counter.total_ns), static_cast<unsigned long long>(counter.count),
            static_cast<unsigned long long>(counter.payload_sum), static_cast<unsigned long long>(dropped)
        );
    }
}
