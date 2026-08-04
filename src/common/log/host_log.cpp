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
 * @file host_log.cpp
 * @brief Implementation of Unified Host Logging System
 */

#include "host_log.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <pthread.h>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

using simpler::log::LogLevel;

namespace {

std::string format_message(const char *fmt, va_list args) {
    va_list sizing_args;
    va_copy(sizing_args, args);
    const int required = vsnprintf(nullptr, 0, fmt, sizing_args);
    va_end(sizing_args);
    if (required < 0) {
        return {};
    }

    std::vector<char> buffer(static_cast<size_t>(required) + 1);
    va_list formatting_args;
    va_copy(formatting_args, args);
    const int written = vsnprintf(buffer.data(), buffer.size(), fmt, formatting_args);
    va_end(formatting_args);
    if (written < 0) {
        return {};
    }
    return std::string(buffer.data(), static_cast<size_t>(written));
}

void write_stderr(const std::string &record) {
    size_t offset = 0;
    while (offset < record.size()) {
        const ssize_t written = ::write(STDERR_FILENO, record.data() + offset, record.size() - offset);
        if (written > 0) {
            offset += static_cast<size_t>(written);
        } else if (written < 0 && errno == EINTR) {
            continue;
        } else {
            break;
        }
    }
}

}  // namespace

HostLogger &HostLogger::get_instance() {
    static HostLogger instance;
    return instance;
}

HostLogger::HostLogger() :
    current_level_(LogLevel::TIMING) {}

void HostLogger::set_level(LogLevel level) {
    std::scoped_lock lock(mutex_);
    current_level_ = level;
}

int HostLogger::level() const { return static_cast<int>(current_level_); }

int HostLogger::cann_level() const { return simpler::log::to_cann_log_level(current_level_); }

void HostLogger::configure_cann_log_level(int (*set_level)(int, int, int)) const {
    if (std::getenv("ASCEND_GLOBAL_LOG_LEVEL") == nullptr) {
        set_level(-1, cann_level(), 0);
    }
}

bool HostLogger::is_enabled(LogLevel level) const {
    // current_level_ is the floor: messages with severity >= floor are kept.
    return static_cast<int>(level) >= static_cast<int>(current_level_) && current_level_ != LogLevel::NUL;
}

const char *HostLogger::level_name(LogLevel level) const {
    switch (level) {
    case LogLevel::DEBUG:
        return "DEBUG";
    case LogLevel::INFO:
        return "INFO";
    case LogLevel::TIMING:
        return "TIMING";
    case LogLevel::WARN:
        return "WARN";
    case LogLevel::ERROR:
        return "ERROR";
    case LogLevel::NUL:
        return "NUL";
    }
    return "?";
}

void HostLogger::emit(const char *level_tag, const char *func, const char *fmt, va_list args) {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto t = system_clock::to_time_t(now);
    auto us = duration_cast<microseconds>(now.time_since_epoch()) % 1'000'000;
    struct tm tm_buf;
    localtime_r(&t, &tm_buf);
    char ts[40];
    size_t n = strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm_buf);
    snprintf(ts + n, sizeof(ts) - n, ".%06lld", static_cast<long long>(us.count()));

    auto tid = static_cast<unsigned long>(reinterpret_cast<uintptr_t>(pthread_self()));

    std::ostringstream stream;
    stream << '[' << ts << "][T0x" << std::hex << tid << "][" << level_tag << "] " << func << ": "
           << format_message(fmt, args);
    std::string record = stream.str();
    if (fmt[0] != '\0' && fmt[strlen(fmt) - 1] != '\n') {
        record.push_back('\n');
    }

    std::scoped_lock lock(mutex_);
    // STRACE records fit within PIPE_BUF, so one write keeps each record
    // indivisible when forked workers share a captured stderr pipe.
    write_stderr(record);
}

void HostLogger::vlog(LogLevel level, const char *func, const char *fmt, va_list args) {
    if (!is_enabled(level)) {
        return;
    }
    emit(level_name(level), func, fmt, args);
}

void HostLogger::log(LogLevel level, const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vlog(level, func, fmt, args);
    va_end(args);
}

// ---------------------------------------------------------------------------
// C ABI entry — resolved by ChipWorker via dlsym from libsimpler_log.so.
//
// Called once early in ChipWorker::init (before host_runtime.so is even
// dlopen'd) to seed the process-wide HostLogger from the user's
// `simpler` Python logger snapshot. Consumers that need the current value
// later (host_runtime.so populating InitArgs.log_level at device init) read it
// via HostLogger::get_instance().level() directly; the value never
// has to travel through any other SO's C ABI.
//
// Level values match Python logging thresholds.
// Returns 0 on success, negative for an unsupported threshold.
// ---------------------------------------------------------------------------
extern "C" int simpler_log_init(int log_level) {
    if (!simpler::log::is_valid_level(log_level)) {
        return -1;
    }
    HostLogger::get_instance().set_level(static_cast<LogLevel>(log_level));
    return 0;
}
