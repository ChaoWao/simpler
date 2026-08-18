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
 * @file host_span_names.h
 * @brief Full `[STRACE]` names for the host-scheduler span family.
 *
 * `Orchestrator` and `WorkerThread` are level-agnostic — the same code drives
 * next-level children at every level above the chip — so the level word in
 * their span names is a property of the process, not of the call site. Python
 * resolves it once from the Worker's level (`simpler.worker_level.span_prefix`)
 * and pushes it here during init.
 *
 * The names are materialized at that point rather than per span: `SpanScope`
 * keeps the `const char *` it was given and dereferences it in its destructor,
 * so a name must outlive every scope that uses it, and building one per span
 * would allocate on the dispatch path.
 */

#pragma once

#include "profiling_config.h"

#if SIMPLER_HOST_STRACE

#include <array>
#include <cstddef>
#include <string>

namespace simpler::host_trace {

/// The host-scheduler family. One entry per span the orchestrator or a worker
/// thread emits; `kCount` sizes the resolved table.
enum class HostSpan : size_t {
    GraphBuild = 0,
    Submit,
    Dispatch,
    FrameSubmit,
    Activate,
    Complete,
    PostFenceRetirement,
    kCount,
};

namespace detail {

inline constexpr std::array<const char *, static_cast<size_t>(HostSpan::kCount)> kHostSpanSuffixes = {
    ".graph_build", ".submit", ".dispatch", ".frame_submit", ".activate", ".complete", ".post_fence_retirement",
};

/// The bound level word. Defaults to `host`: L3 is the only level that has ever
/// driven this code in a shipped configuration, so an unset prefix names the
/// truth rather than an arbitrary placeholder.
inline std::string &level_word() {
    static std::string word = "host";
    return word;
}

/// Resolved names, owned for the process's lifetime so `c_str()` outlives every
/// `SpanScope`.
inline std::array<std::string, static_cast<size_t>(HostSpan::kCount)> &host_span_table() {
    static std::array<std::string, static_cast<size_t>(HostSpan::kCount)> table = [] {
        std::array<std::string, static_cast<size_t>(HostSpan::kCount)> initial;
        for (size_t i = 0; i < initial.size(); ++i) {
            initial[i] = level_word() + kHostSpanSuffixes[i];
        }
        return initial;
    }();
    return table;
}

}  // namespace detail

/**
 * Bind this process's level word, e.g. `"host"` or `"network1"`.
 *
 * Called once during Worker init, before any worker thread starts and before
 * the first span — the table is then read-only, which is why it needs no lock.
 * A null or empty word leaves the default in place.
 */
inline void set_level_prefix(const char *word) {
    if (word == nullptr || *word == '\0') return;
    detail::level_word() = word;
    auto &table = detail::host_span_table();
    for (size_t i = 0; i < table.size(); ++i) {
        table[i] = detail::level_word() + detail::kHostSpanSuffixes[i];
    }
}

/// The level word currently bound, for a caller that must agree with it.
inline const char *level_prefix() { return detail::level_word().c_str(); }

/// Full dotted name for `which`, stable for the process's lifetime.
inline const char *host_span_name(HostSpan which) {
    return detail::host_span_table()[static_cast<size_t>(which)].c_str();
}

}  // namespace simpler::host_trace

#else

namespace simpler::host_trace {

inline void set_level_prefix(const char *) noexcept {}

}  // namespace simpler::host_trace

#endif  // SIMPLER_HOST_STRACE
