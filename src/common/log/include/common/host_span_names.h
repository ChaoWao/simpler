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

/// The bound level word. Defaults to `node`: L3 is the only level that has ever
/// driven this code in a shipped configuration, so an unset prefix names the
/// truth rather than an arbitrary placeholder. The word names a topology
/// position, so it is never the processor name `host` — that belongs to the
/// `host_span` ABI this file implements, which every level above the chip uses.
inline std::string &level_word() {
    static std::string word = "node";
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

/// Whether the word has been bound. A second binding is refused rather than
/// applied — see `set_level_prefix`.
inline bool &prefix_bound() {
    static bool bound = false;
    return bound;
}

}  // namespace detail

/**
 * Bind this process's level word, e.g. `"node"` or `"network1"`.
 *
 * **The first non-empty word wins; every later one is refused.** Two reasons,
 * and neither is enforceable by documentation alone:
 *
 *  - `SpanScope` stores the `const char *` it was handed and dereferences it in
 *    its destructor. Rebinding reassigns these strings, which may reallocate, so
 *    a scope open across a rebind would emit through a dangling pointer.
 *  - A process that constructs Workers at two levels would otherwise relabel the
 *    first Worker's spans while it is still running.
 *
 * A null or empty word leaves the current binding in place. `level_prefix()`
 * always reports what is actually bound, so a caller that asked for something
 * else can see that it did not get it — Python compares the two and warns,
 * because one process has one vocabulary and that is worth saying out loud
 * rather than discovering in a trace.
 */
inline void set_level_prefix(const char *word) {
    if (word == nullptr || *word == '\0') return;
    if (detail::prefix_bound()) return;
    detail::level_word() = word;
    auto &table = detail::host_span_table();
    for (size_t i = 0; i < table.size(); ++i) {
        table[i] = detail::level_word() + detail::kHostSpanSuffixes[i];
    }
    detail::prefix_bound() = true;
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
