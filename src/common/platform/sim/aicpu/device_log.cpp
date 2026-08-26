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
 * The process-owned HostLogger state is bound by the sim host before AICPU
 * execution begins; this file does not read env vars.
 */

#include "aicpu/device_log.h"
#include "common/host_log_binding.h"
#include "host_log.h"

#include <cstdarg>

namespace {
SimplerHostLogState *g_host_log_state = nullptr;
}

bool is_log_enable_debug() { return HostLogger::get_instance().is_enabled(simpler::log::LogLevel::DEBUG); }

bool is_log_enable_info() { return HostLogger::get_instance().is_enabled(simpler::log::LogLevel::INFO); }

bool is_log_enable_timing() { return HostLogger::get_instance().is_enabled(simpler::log::LogLevel::TIMING); }

bool is_log_enable_warn() { return HostLogger::get_instance().is_enabled(simpler::log::LogLevel::WARN); }

bool is_log_enable_error() { return HostLogger::get_instance().is_enabled(simpler::log::LogLevel::ERROR); }

// =============================================================================
// Setters (called by AICPU init from KernelArgs)
// =============================================================================

extern "C" void set_log_level(int level) {
    if (g_host_log_state != nullptr && simpler::log::is_valid_level(level)) {
        HostLogger::get_instance().set_level(static_cast<simpler::log::LogLevel>(level));
    }
}

extern "C" void set_host_log_state(SimplerHostLogState *state) {
    if (HostLogger::get_instance().bind_state(state) == 0) g_host_log_state = state;
}

int bind_orchestration_host_log_state(void *handle, const char **error) {
    return simpler::log::bind_loaded_host_log_state(handle, g_host_log_state, error);
}

// =============================================================================
// init_log_switch: the sim threshold lives in the bound host-log state. The
// no-op entry point shares the platform interface with the onboard CANN query.
// =============================================================================

void init_log_switch() {
    // Sim has no CANN log switch to query.
}

// =============================================================================
// Low-level dev_log_* / dev_vlog_*
//
// The shared AICPU adapter retains this va_list interface on both platforms.
// HostLogger owns the sim envelope, threshold and configured output; onboard
// supplies the corresponding CANN-backed definitions in its platform
// implementation.
// =============================================================================

void dev_vlog_debug(const char *func, const char *fmt, va_list args) {
    HostLogger::get_instance().vlog(simpler::log::LogLevel::DEBUG, func, fmt, args);
}

void dev_vlog_info(const char *func, const char *fmt, va_list args) {
    HostLogger::get_instance().vlog(simpler::log::LogLevel::INFO, func, fmt, args);
}

void dev_vlog_timing(const char *func, const char *fmt, va_list args) {
    HostLogger::get_instance().vlog(simpler::log::LogLevel::TIMING, func, fmt, args);
}

void dev_vlog_warn(const char *func, const char *fmt, va_list args) {
    HostLogger::get_instance().vlog(simpler::log::LogLevel::WARN, func, fmt, args);
}

void dev_vlog_error(const char *func, const char *fmt, va_list args) {
    HostLogger::get_instance().vlog(simpler::log::LogLevel::ERROR, func, fmt, args);
}
