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

// HostLogger filtering: one Python-compatible threshold.
// Drives the singleton via a direct setter, captures stderr, and asserts on
// the buffered output.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

#include <gtest/gtest.h>

#include "host_log.h"

using simpler::log::LogLevel;

namespace {

struct CapturedStdio {
    std::string out;
    std::string err;
};

struct CannLogLevelCall {
    int count;
    int module_id;
    int level;
    int enable_event;
};

CannLogLevelCall g_cann_log_level_call{};

int capture_cann_log_level(int module_id, int level, int enable_event) {
    g_cann_log_level_call.count++;
    g_cann_log_level_call.module_id = module_id;
    g_cann_log_level_call.level = level;
    g_cann_log_level_call.enable_event = enable_event;
    return 0;
}

CapturedStdio run_with_config(LogLevel level, void (*fn)()) {
    fflush(stdout);
    fflush(stderr);
    FILE *out_tmp = tmpfile();
    FILE *err_tmp = tmpfile();
    int saved_out = dup(fileno(stdout));
    int saved_err = dup(fileno(stderr));
    dup2(fileno(out_tmp), fileno(stdout));
    dup2(fileno(err_tmp), fileno(stderr));

    HostLogger::get_instance().set_level(level);

    fn();

    fflush(stdout);
    fflush(stderr);
    dup2(saved_out, fileno(stdout));
    dup2(saved_err, fileno(stderr));
    close(saved_out);
    close(saved_err);

    auto slurp = [](FILE *f) {
        std::string s;
        rewind(f);
        char buf[512];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
            s.append(buf, n);
        }
        fclose(f);
        return s;
    };
    return {slurp(out_tmp), slurp(err_tmp)};
}

}  // namespace

TEST(HostLogTest, NulLevelMutesAllSeverities) {
    auto captured = run_with_config(LogLevel::NUL, [] {
        HostLogger::get_instance().log(LogLevel::ERROR, "fn", "err-msg");
        HostLogger::get_instance().log(LogLevel::WARN, "fn", "warn-msg");
        HostLogger::get_instance().log(LogLevel::TIMING, "fn", "timing-msg");
        HostLogger::get_instance().log(LogLevel::INFO, "fn", "info-msg");
        HostLogger::get_instance().log(LogLevel::DEBUG, "fn", "dbg-msg");
    });
    EXPECT_EQ(captured.out, "");
    EXPECT_EQ(captured.err, "");
}

TEST(HostLogTest, ErrorLevelEmitsErrorOnly) {
    auto captured = run_with_config(LogLevel::ERROR, [] {
        HostLogger::get_instance().log(LogLevel::ERROR, "fn", "err-msg");
        HostLogger::get_instance().log(LogLevel::WARN, "fn", "warn-msg");
        HostLogger::get_instance().log(LogLevel::TIMING, "fn", "timing-msg");
    });
    EXPECT_EQ(captured.out, "");
    EXPECT_NE(captured.err.find("err-msg"), std::string::npos);
    EXPECT_EQ(captured.err.find("warn-msg"), std::string::npos);
    EXPECT_EQ(captured.err.find("timing-msg"), std::string::npos);
}

TEST(HostLogTest, TimingLevelKeepsTimingAndHigher) {
    auto captured = run_with_config(LogLevel::TIMING, [] {
        HostLogger::get_instance().log(LogLevel::DEBUG, "fn", "debug-msg");
        HostLogger::get_instance().log(LogLevel::INFO, "fn", "info-msg");
        HostLogger::get_instance().log(LogLevel::TIMING, "fn", "timing-msg");
        HostLogger::get_instance().log(LogLevel::WARN, "fn", "warn-msg");
        HostLogger::get_instance().log(LogLevel::ERROR, "fn", "error-msg");
    });
    EXPECT_EQ(captured.out, "");
    EXPECT_EQ(captured.err.find("debug-msg"), std::string::npos);
    EXPECT_EQ(captured.err.find("info-msg"), std::string::npos);
    EXPECT_NE(captured.err.find("timing-msg"), std::string::npos);
    EXPECT_NE(captured.err.find("warn-msg"), std::string::npos);
    EXPECT_NE(captured.err.find("error-msg"), std::string::npos);
}

TEST(HostLogTest, CannLevelMappingSuppressesInfoAtDefault) {
    EXPECT_EQ(simpler::log::to_cann_log_level(LogLevel::DEBUG), 0);
    EXPECT_EQ(simpler::log::to_cann_log_level(LogLevel::INFO), 1);
    EXPECT_EQ(simpler::log::to_cann_log_level(LogLevel::TIMING), 2);
    EXPECT_EQ(simpler::log::to_cann_log_level(LogLevel::WARN), 2);
    EXPECT_EQ(simpler::log::to_cann_log_level(LogLevel::ERROR), 3);
    EXPECT_EQ(simpler::log::to_cann_log_level(LogLevel::NUL), 4);
}

TEST(HostLogTest, CannConfigurationUsesGlobalModuleAndRespectsExternalOverride) {
    const char *old_env = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");
    const bool had_old_env = old_env != nullptr;
    const std::string old_value = had_old_env ? old_env : "";

    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
    HostLogger::get_instance().set_level(LogLevel::TIMING);
    g_cann_log_level_call = {};
    HostLogger::get_instance().configure_cann_log_level(capture_cann_log_level);
    EXPECT_EQ(g_cann_log_level_call.count, 1);
    EXPECT_EQ(g_cann_log_level_call.module_id, -1);
    EXPECT_EQ(g_cann_log_level_call.level, 2);
    EXPECT_EQ(g_cann_log_level_call.enable_event, 0);

    setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", 1);
    g_cann_log_level_call = {};
    HostLogger::get_instance().configure_cann_log_level(capture_cann_log_level);
    EXPECT_EQ(g_cann_log_level_call.count, 0);

    if (had_old_env) {
        setenv("ASCEND_GLOBAL_LOG_LEVEL", old_value.c_str(), 1);
    } else {
        unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
    }
}

TEST(HostLogTest, EmitPrefixHasTimestampAndTid) {
    auto captured = run_with_config(LogLevel::INFO, [] {
        HostLogger::get_instance().log(LogLevel::ERROR, "fn", "marker");
    });
    // Expected shape: "[YYYY-MM-DD HH:MM:SS.uuuuuu][T0x...][ERROR] fn: marker\n"
    ASSERT_FALSE(captured.err.empty());
    EXPECT_EQ(captured.err[0], '[');
    // Year must be 4 ASCII digits.
    for (int i = 1; i <= 4; ++i) {
        EXPECT_GE(captured.err[i], '0');
        EXPECT_LE(captured.err[i], '9');
    }
    EXPECT_EQ(captured.err[5], '-');
    // Thread-id segment "[T0x" must appear before the level tag.
    auto tid_pos = captured.err.find("][T0x");
    auto level_pos = captured.err.find("][ERROR]");
    ASSERT_NE(tid_pos, std::string::npos);
    ASSERT_NE(level_pos, std::string::npos);
    EXPECT_LT(tid_pos, level_pos);
    // Body still present.
    EXPECT_NE(captured.err.find("marker"), std::string::npos);
}

TEST(HostLogTest, AllOutputGoesToStderr) {
    auto captured = run_with_config(LogLevel::DEBUG, [] {
        HostLogger::get_instance().log(LogLevel::ERROR, "fn", "error-output-marker");
        HostLogger::get_instance().log(LogLevel::WARN, "fn", "warn-output-marker");
        HostLogger::get_instance().log(LogLevel::TIMING, "fn", "timing-output-marker");
        HostLogger::get_instance().log(LogLevel::INFO, "fn", "info-output-marker");
        HostLogger::get_instance().log(LogLevel::DEBUG, "fn", "debug-output-marker");
    });
    EXPECT_EQ(captured.out, "");
    EXPECT_NE(captured.err.find("error-output-marker"), std::string::npos);
    EXPECT_NE(captured.err.find("warn-output-marker"), std::string::npos);
    EXPECT_NE(captured.err.find("timing-output-marker"), std::string::npos);
    EXPECT_NE(captured.err.find("info-output-marker"), std::string::npos);
    EXPECT_NE(captured.err.find("debug-output-marker"), std::string::npos);
}
