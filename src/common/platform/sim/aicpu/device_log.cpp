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

#include <cstdarg>
#include <cstdio>

// =============================================================================
// Level enable flags (mutated by the setter below)
// =============================================================================

bool g_is_log_enable_debug = false;
bool g_is_log_enable_info = false;
bool g_is_log_enable_timing = true;
bool g_is_log_enable_warn = true;
bool g_is_log_enable_error = true;

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

// =============================================================================
// init_log_switch: sim respects host-pushed config — this is now a no-op
// (kept for ABI compatibility with onboard, where it queries CANN dlog).
// =============================================================================

void init_log_switch() {
    // Sim has no env / dlog to consult. Defaults already applied at static
    // init; host overrides via set_log_level() before this
    // is called.
}

// =============================================================================
// Low-level dev_log_* / dev_vlog_* (sim: fprintf to stderr; no buffer needed)
// =============================================================================

void dev_vlog_debug(const char *func, const char *fmt, va_list args) {
    fprintf(stderr, "[DEBUG] %s: ", func);
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
}

void dev_vlog_info(const char *func, const char *fmt, va_list args) {
    fprintf(stderr, "[INFO] %s: ", func);
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
}

void dev_vlog_timing(const char *func, const char *fmt, va_list args) {
    fprintf(stderr, "[TIMING] %s: ", func);
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
}

void dev_vlog_warn(const char *func, const char *fmt, va_list args) {
    fprintf(stderr, "[WARN] %s: ", func);
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
}

void dev_vlog_error(const char *func, const char *fmt, va_list args) {
    fprintf(stderr, "[ERROR] %s: ", func);
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
}
