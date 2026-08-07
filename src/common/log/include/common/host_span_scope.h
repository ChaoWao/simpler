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
 * @file host_span_scope.h
 * @brief C++ emit helpers over the SimplerHostSpan C ABI.
 *
 * Timestamps are ns on CLOCK_MONOTONIC (steady_clock), so spans emitted here
 * are comparable with the STRACE markers in common/strace.h, with Python's
 * time.monotonic_ns(), and across a fork.
 */

#pragma once

#include "common/host_span.h"
#include "profiling_config.h"

#if SIMPLER_HOST_STRACE

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace simpler::host_trace {

inline SimplerLogEmitHostSpanFn &sink_slot() noexcept {
    static SimplerLogEmitHostSpanFn sink = nullptr;
    return sink;
}

/** Bind this module's sink slot to the process logger before threads or forks. */
inline void bind_sink(SimplerLogEmitHostSpanFn sink) noexcept { sink_slot() = sink; }

/** False when this module has not been bound to libsimpler_log.so. */
inline bool sink_available() noexcept { return sink_slot() != nullptr; }

inline int64_t now_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline void emit(
    const char *name, uint64_t invocation_id, uint64_t callable_hash, int32_t depth, int64_t timestamp_ns,
    int64_t duration_ns, const char *attributes
) noexcept {
    SimplerLogEmitHostSpanFn sink = sink_slot();
    if (sink == nullptr) return;
    const SimplerHostSpan span{
        SIMPLER_HOST_SPAN_ABI_VERSION,
        sizeof(SimplerHostSpan),
        invocation_id,
        callable_hash,
        depth,
        0,
        timestamp_ns,
        duration_ns,
        name,
        attributes
    };
    sink(&span);
}

/** Times its own scope and emits on destruction. `name` must outlive the scope;
 * every call site passes a literal. */
class SpanScope {
public:
    SpanScope(const char *name, uint64_t invocation_id, uint64_t callable_hash, int32_t depth, std::string attributes) :
        name_(name),
        invocation_id_(invocation_id),
        callable_hash_(callable_hash),
        depth_(depth),
        timestamp_ns_(now_ns()),
        attributes_(std::move(attributes)) {}

    ~SpanScope() {
        const int64_t end_ns = now_ns();
        emit(name_, invocation_id_, callable_hash_, depth_, timestamp_ns_, end_ns - timestamp_ns_, attributes_.c_str());
    }

    SpanScope(const SpanScope &) = delete;
    SpanScope &operator=(const SpanScope &) = delete;
    SpanScope(SpanScope &&) = delete;
    SpanScope &operator=(SpanScope &&) = delete;

private:
    const char *name_;
    uint64_t invocation_id_;
    uint64_t callable_hash_;
    int32_t depth_;
    int64_t timestamp_ns_;
    std::string attributes_;
};

}  // namespace simpler::host_trace

#else

#include <cstdint>

namespace simpler::host_trace {

inline void bind_sink(SimplerLogEmitHostSpanFn) noexcept {}
inline bool sink_available() noexcept { return false; }
inline int64_t now_ns() noexcept { return 0; }
inline void emit(const char *, uint64_t, uint64_t, int32_t, int64_t, int64_t, const char *) noexcept {}

}  // namespace simpler::host_trace

#endif
