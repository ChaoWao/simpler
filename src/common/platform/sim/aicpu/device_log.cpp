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
 * @file device_log.cpp (sim)
 * @brief Simulation Platform Log Implementation
 *
 * Level flags are populated by host via set_log_level() at AICPU kernel init
 * (see kernel.cpp / aicpu_executor.cpp); this file does not read env vars.
 */

#include "aicpu/device_log.h"
#include "common/host_log_binding.h"

#include <cerrno>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <unistd.h>

// =============================================================================
// Level enable flags (mutated by the setter below)
// =============================================================================

bool g_is_log_enable_debug = false;
bool g_is_log_enable_info = false;
bool g_is_log_enable_timing = true;
bool g_is_log_enable_warn = true;
bool g_is_log_enable_error = true;

namespace {
SimplerHostLogState *g_host_log_state = nullptr;
}

// =============================================================================
// Setters (called by AICPU init from KernelArgs)
// =============================================================================

extern "C" void set_log_level(int level) {
    g_is_log_enable_debug = level <= 10;
    g_is_log_enable_info = level <= 20;
    g_is_log_enable_timing = level <= 25;
    g_is_log_enable_warn = level <= 30;
    g_is_log_enable_error = level <= 40;
}

extern "C" void set_host_log_state(SimplerHostLogState *state) { g_host_log_state = state; }

int bind_orchestration_host_log_state(void *handle, const char **error) {
    return simpler::log::bind_loaded_host_log_state(handle, g_host_log_state, error);
}

// =============================================================================
// init_log_switch: sim respects host-pushed config. The no-op entry point is
// retained for ABI compatibility with onboard, where it queries CANN dlog.
// =============================================================================

void init_log_switch() {
    // Sim has no env / dlog to consult. Defaults already applied at static
    // init; host overrides via set_log_level() before this
    // is called.
}

// =============================================================================
// Low-level dev_log_* / dev_vlog_*
//
// Each record "[TAG] func: body\n" is formatted into a single stack buffer and
// emitted with one write(). The buffer caps the record at 2048 bytes, below
// Linux PIPE_BUF (4096), so a record is delivered atomically when stderr is a
// pipe — concurrent AICPU sim threads and forked chip workers sharing stderr
// never interleave partial records.
// =============================================================================

namespace {

void emit_record(const char *level_tag, const char *func, const char *fmt, va_list args) {
    char buffer[2048];
    constexpr size_t kNewlineSlot = 1;
    constexpr size_t kBodyLimit = sizeof(buffer) - kNewlineSlot;

    int prefix = snprintf(buffer, sizeof(buffer), "[%s] %s: ", level_tag, func);
    size_t len = (prefix < 0) ? 0 : static_cast<size_t>(prefix);
    if (len > kBodyLimit) {
        len = kBodyLimit;  // prefix filled the buffer; reserve the newline slot
    }

    int body = vsnprintf(buffer + len, sizeof(buffer) - len, fmt, args);
    if (body > 0) {
        len += static_cast<size_t>(body);
        if (len > kBodyLimit) {
            len = kBodyLimit;  // body truncated; reserve the newline slot
        }
    }

    buffer[len++] = '\n';
    // On a pipe this transfers the whole record (<= PIPE_BUF) in one atomic
    // call; the loop only iterates for a non-pipe stderr (regular file, socket)
    // where write(2) may be interrupted or short, and must not leave a record
    // without its terminating newline.
    for (size_t off = 0; off < len;) {
        ssize_t written = write(STDERR_FILENO, buffer + off, len - off);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;  // nothing the device-log backend can do on a hard failure
        }
        off += static_cast<size_t>(written);
    }
}

}  // namespace

void dev_vlog_debug(const char *func, const char *fmt, va_list args) { emit_record("DEBUG", func, fmt, args); }

void dev_vlog_info(const char *func, const char *fmt, va_list args) { emit_record("INFO", func, fmt, args); }

void dev_vlog_timing(const char *func, const char *fmt, va_list args) { emit_record("TIMING", func, fmt, args); }

void dev_vlog_warn(const char *func, const char *fmt, va_list args) { emit_record("WARN", func, fmt, args); }

void dev_vlog_error(const char *func, const char *fmt, va_list args) { emit_record("ERROR", func, fmt, args); }
