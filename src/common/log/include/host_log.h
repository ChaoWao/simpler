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
 * @file host_log.h
 * @brief Unified Host Logging System
 *
 * One threshold controls DEBUG/INFO/TIMING/WARN/ERROR/NUL. The integer values
 * match Python logging levels. CANN has no TIMING level, so onboard setup maps
 * both TIMING and WARN thresholds to CANN WARN.
 */

#pragma once

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <mutex>

#include <sys/types.h>

#include "common/host_log_state.h"
#include "common/log_level.h"

#if defined(__GNUC__) || defined(__clang__)
#define SIMPLER_HOST_LOG_LOCAL __attribute__((visibility("hidden")))
#else
#define SIMPLER_HOST_LOG_LOCAL
#endif

struct SimplerHostSpan;

class SIMPLER_HOST_LOG_LOCAL HostLogger {
public:
    static HostLogger &get_instance();

    // Bind this module-local logger implementation to process-owned state.
    // Must happen during module init, before the module starts worker threads.
    int bind_state(SimplerHostLogState *state);
    SimplerHostLogState *state() const;

    void log(simpler::log::LogLevel level, const char *func, const char *fmt, ...);

    // va_list-taking primitive used by unified_log_* adapters. Caller is
    // responsible for `va_start` / `va_end`.
    void vlog(simpler::log::LogLevel level, const char *func, const char *fmt, va_list args);

    void set_level(simpler::log::LogLevel level);

    // Runtime modules read this from their bound process-owned state when
    // populating InitArgs.log_level at device init.
    int level() const;
    int cann_level() const;

    // Configure CANN before the device context is opened unless the user has
    // already selected ASCEND_GLOBAL_LOG_LEVEL. The callback shape matches
    // dlog_setlevel and keeps this policy independently testable.
    void configure_cann_log_level(int (*set_level)(int module_id, int level, int enable_event)) const;

    bool is_enabled(simpler::log::LogLevel level) const;

    // Fixed host-span adapter. The logger remains the sole STRACE grammar
    // owner even though its implementation is compiled into several DSOs.
    void log_host_span(const SimplerHostSpan *span);

private:
    HostLogger();
    ~HostLogger() = default;

    HostLogger(const HostLogger &) = delete;
    HostLogger &operator=(const HostLogger &) = delete;
    HostLogger(HostLogger &&) = delete;
    HostLogger &operator=(HostLogger &&) = delete;

    const char *level_name(simpler::log::LogLevel level) const;
    bool emit(const char *level_tag, const char *func, const char *fmt, va_list args);
    bool emit_ungated(const char *level_tag, const char *func, const char *fmt, ...);
    void emit_clock_anchor_if_needed();

    std::atomic<SimplerHostLogState *> state_;
    std::mutex mutex_;
};

#undef SIMPLER_HOST_LOG_LOCAL
