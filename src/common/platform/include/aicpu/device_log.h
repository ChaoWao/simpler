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
 * @file device_log.h
 * @brief Unified Device Logging Interface for AICPU
 *
 * Layered design:
 *   - Low-level dev_log_*() functions are platform-specific (CANN dlog on
 *     real hardware, fprintf(stderr,...) in simulation).
 *   - Onboard fills DEBUG/INFO/WARN/ERROR from CheckLogLevel(AICPU,...);
 *     simulation fills all flags from the host-provided threshold.
 *   - TIMING is a simpler level between INFO and WARN. Both backends gate it
 *     from the host threshold; onboard emits enabled messages through CANN
 *     WARN because CANN has no intermediate level.
 *
 * Platform Support:
 * - a2a3 / a5       : Real hardware with CANN dlog API
 * - a2a3sim / a5sim : Host-based simulation using fprintf(stderr,...)
 */

#pragma once

#include <cstdio>
#include <cstdint>

// =============================================================================
// Platform Detection and Thread ID
// =============================================================================

#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#define GET_TID() syscall(SYS_gettid)
#else
#define GET_TID() 0
#endif

// =============================================================================
// Severity enable flags (defined in platform-specific device_log.cpp)
// =============================================================================

extern bool g_is_log_enable_debug;
extern bool g_is_log_enable_info;
extern bool g_is_log_enable_timing;
extern bool g_is_log_enable_warn;
extern bool g_is_log_enable_error;

// =============================================================================
// Configuration setters (called by AICPU kernel init from KernelArgs)
// =============================================================================

// Levels use Python-compatible thresholds: DEBUG=10, INFO=20, TIMING=25,
// WARN=30, ERROR=40, NUL=60. Onboard applies the threshold to TIMING while
// CANN owns its native levels; simulation applies it to the full flag table.
extern "C" void set_log_level(int level);

// Hand the process-owned host-log state to the simulation AICPU backend, which
// relays it to orchestration SOs it later loads. Simulation defines this; the
// host-side loader resolves it by name from the AICPU SO handle. Declared here
// so both sides agree on the signature at compile time — the struct is only
// forward-declared, keeping <dlfcn.h> and the state layout off device targets.
struct SimplerHostLogState;
extern "C" void set_host_log_state(struct SimplerHostLogState *state);

// Apply the platform's logging policy to a newly loaded orchestration SO.
// Simulation binds the process-owned host state; onboard requires no handoff.
int bind_orchestration_host_log_state(void *handle, const char **error);

// =============================================================================
// Platform-specific logging functions (low-level layer)
//
// va_list primitives used by the unified_log_* adapter to forward a caller's
// variadic args. Both backends format a whole record into one stack buffer and
// emit it in a single call: sim writes it with one write(2), kept under
// PIPE_BUF so concurrent threads / forked workers on a shared stderr never
// interleave partial records; onboard buffers because CANN's dlog API has no
// va_list variant. Caller owns va_start/va_end.
// =============================================================================

#include <cstdarg>

void dev_vlog_debug(const char *func, const char *fmt, va_list args);
void dev_vlog_info(const char *func, const char *fmt, va_list args);
void dev_vlog_timing(const char *func, const char *fmt, va_list args);
void dev_vlog_warn(const char *func, const char *fmt, va_list args);
void dev_vlog_error(const char *func, const char *fmt, va_list args);

// =============================================================================
// Helper Functions
// =============================================================================

inline bool is_log_enable_debug() { return g_is_log_enable_debug; }
inline bool is_log_enable_info() { return g_is_log_enable_info; }
inline bool is_log_enable_timing() { return g_is_log_enable_timing; }
inline bool is_log_enable_warn() { return g_is_log_enable_warn; }
inline bool is_log_enable_error() { return g_is_log_enable_error; }

// Initialize log switch (platform-specific implementation)
void init_log_switch();
