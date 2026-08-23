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

#include <stddef.h>
#include <stdint.h>

#include <type_traits>

#include "task_id.h"
#include "types.h"

struct GraphScopeResult {
    bool execute_block{true};
    bool recording{false};
    TaskId task_id{TaskId::invalid()};
    // The recording this call opened, non-null exactly when `recording` is set.
    // The recording thread hands it back to graph_prepare so binding needs no
    // lookup: several Definitions record at once, and finding one by key would
    // mean traversing the in-flight map under its mutex on a path whose whole
    // purpose is to not contend with the submitting thread.
    void *recording_handle{nullptr};
};

using GraphSubmitResult = GraphScopeResult;

constexpr uint64_t graph_hash_byte(uint64_t h, uint8_t b) { return (h ^ static_cast<uint64_t>(b)) * 1099511628211ULL; }

// FNV-1a, mixing eight bytes per multiply instead of one. The multiply is a
// serial dependency, so a byte-at-a-time pass over a Definition image costs one
// multiply latency per byte — 130 KB of it measured as essentially all of
// graph_build_definition, and the AICPU pays the same pass again to verify the
// uploaded object.
//
// Streaming: callers may update across several calls, and the word split makes
// the result depend on where they split. Every chunk boundary in this codebase
// falls on a multiple of eight (see the static_assert beside
// graph_definition_hash_matches), so a chunked update and a single update over
// the concatenation agree. The value is recomputed from this header on both
// sides of every run and is never persisted, so it carries no cross-version
// contract.
inline uint64_t graph_hash_bytes(uint64_t h, const void *data, size_t bytes) {
    const auto *p = static_cast<const uint8_t *>(data);
    size_t i = 0;
    for (; i + sizeof(uint64_t) <= bytes; i += sizeof(uint64_t)) {
        uint64_t word = 0;
        __builtin_memcpy(&word, p + i, sizeof(word));
        h = (h ^ word) * 1099511628211ULL;
    }
    for (; i < bytes; ++i) {
        h = graph_hash_byte(h, p[i]);
    }
    return h;
}

constexpr uint64_t graph_const_hash_impl(const char *s, uint64_t h) {
    return (*s == '\0') ? h : graph_const_hash_impl(s + 1, graph_hash_byte(h, static_cast<uint8_t>(*s)));
}

constexpr uint64_t GRAPH_KEY(const char *s) { return graph_const_hash_impl(s, 1469598103934665603ULL); }

inline bool rt_graph_args_cacheable(const GraphTaskArgs &args) {
    if (args.has_error || args.tensor_count() <= 0 ||
        args.tensor_count() > static_cast<int32_t>(GRAPH_MAX_TENSOR_ARGS)) {
        return false;
    }
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        // A Graph boundary is caller-owned storage. Runtime-allocated
        // TensorCreateInfo outputs remain on the ordinary submit path.
        if (args.tag(i) == TensorArgType::OUTPUT) return false;
    }
    return true;
}

inline uint64_t rt_graph_make_key(uint64_t graph_id) { return graph_id; }

template <typename T>
inline uint64_t graph_hash_config_value(uint64_t hash, T value) {
    using Value = std::remove_cv_t<std::remove_reference_t<T>>;
    static_assert(
        std::is_integral_v<Value> || std::is_same_v<Value, float> || std::is_same_v<Value, double>,
        "Graph construction parameters must be integral, float, or double values"
    );
    constexpr uint8_t category = std::is_same_v<Value, bool> ? 1 :
                                 std::is_integral_v<Value>   ? (std::is_signed_v<Value> ? 2 : 3) :
                                                               4;
    constexpr uint8_t width = sizeof(Value);
    hash = graph_hash_byte(hash, category);
    hash = graph_hash_byte(hash, width);
    return graph_hash_bytes(hash, &value, sizeof(value));
}

template <typename... Config>
inline uint64_t rt_graph_make_key(uint64_t graph_id, Config... config) {
    uint64_t hash = graph_hash_bytes(1469598103934665603ULL, &graph_id, sizeof(graph_id));
    const uint32_t count = sizeof...(Config);
    hash = graph_hash_bytes(hash, &count, sizeof(count));
    ((hash = graph_hash_config_value(hash, config)), ...);
    return hash;
}
