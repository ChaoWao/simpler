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

#ifndef PLATFORM_LOG_LEVEL_H_
#define PLATFORM_LOG_LEVEL_H_

namespace simpler::log {

enum class LogLevel : int {
    DEBUG = 10,
    INFO = 20,
    TIMING = 25,
    WARN = 30,
    ERROR = 40,
    NUL = 60,
};

constexpr int kDefaultThreshold = static_cast<int>(LogLevel::TIMING);

constexpr bool is_valid_level(int level) {
    return level == static_cast<int>(LogLevel::DEBUG) || level == static_cast<int>(LogLevel::INFO) ||
           level == static_cast<int>(LogLevel::TIMING) || level == static_cast<int>(LogLevel::WARN) ||
           level == static_cast<int>(LogLevel::ERROR) || level == static_cast<int>(LogLevel::NUL);
}

constexpr int to_cann_log_level(LogLevel level) {
    switch (level) {
    case LogLevel::DEBUG:
        return 0;
    case LogLevel::INFO:
        return 1;
    case LogLevel::TIMING:
    case LogLevel::WARN:
        return 2;
    case LogLevel::ERROR:
        return 3;
    case LogLevel::NUL:
        return 4;
    }
    return 4;
}

}  // namespace simpler::log

#endif  // PLATFORM_LOG_LEVEL_H_
