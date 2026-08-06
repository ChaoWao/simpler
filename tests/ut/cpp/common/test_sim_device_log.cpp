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

// Sim device-log atomicity: every dev_vlog_* call emits exactly one intact
// physical line, even when many AICPU sim threads or forked chip workers write
// the shared stderr concurrently. Each record is <= PIPE_BUF, so a single
// write(2) to a pipe is atomic and cannot interleave with another writer.

#include <cstdarg>
#include <cstdio>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include "aicpu/device_log.h"

namespace {

constexpr const char *kTags[] = {"DEBUG", "INFO", "TIMING", "WARN", "ERROR"};

// dev_vlog_* gate on nothing (the unified_log_* adapter owns level filtering),
// so these thin wrappers always emit. level_idx selects the backend under test.
void emit(int level_idx, const char *func, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    switch (level_idx % 5) {
    case 0:
        dev_vlog_debug(func, fmt, ap);
        break;
    case 1:
        dev_vlog_info(func, fmt, ap);
        break;
    case 2:
        dev_vlog_timing(func, fmt, ap);
        break;
    case 3:
        dev_vlog_warn(func, fmt, ap);
        break;
    default:
        dev_vlog_error(func, fmt, ap);
        break;
    }
    va_end(ap);
}

std::string record(int level_idx, const char *func, const std::string &body) {
    return std::string("[") + kTags[level_idx % 5] + "] " + func + ": " + body;
}

// Redirect stderr onto a fresh pipe. All writers stay well under the pipe's
// 64 KiB capacity, so they never block before the reader drains.
struct Capture {
    int read_fd = -1;
    int saved_stderr = -1;
};

Capture begin_capture() {
    int fds[2];
    EXPECT_EQ(pipe(fds), 0);
    fflush(stderr);
    Capture cap;
    cap.saved_stderr = dup(STDERR_FILENO);
    EXPECT_GE(cap.saved_stderr, 0);
    EXPECT_GE(dup2(fds[1], STDERR_FILENO), 0);
    close(fds[1]);  // STDERR_FILENO is now the sole write handle
    cap.read_fd = fds[0];
    return cap;
}

// Restore stderr (closing the last write handle so read hits EOF) and slurp
// everything the pipe holds.
std::string end_capture(Capture &cap) {
    fflush(stderr);
    EXPECT_GE(dup2(cap.saved_stderr, STDERR_FILENO), 0);
    close(cap.saved_stderr);

    std::string out;
    char buf[4096];
    ssize_t n;
    while ((n = read(cap.read_fd, buf, sizeof(buf))) > 0) {
        out.append(buf, static_cast<size_t>(n));
    }
    close(cap.read_fd);
    return out;
}

// Assert the captured stream is exactly `expected`, one intact record per
// physical line — a torn (merged or split) record matches no expected string.
void expect_intact(const std::string &captured, std::multiset<std::string> expected) {
    size_t start = 0;
    while (start < captured.size()) {
        size_t nl = captured.find('\n', start);
        ASSERT_NE(nl, std::string::npos) << "record missing terminating newline";
        std::string line = captured.substr(start, nl - start);
        auto it = expected.find(line);
        ASSERT_NE(it, expected.end()) << "torn or unexpected record: '" << line << "'";
        expected.erase(it);
        start = nl + 1;
    }
    EXPECT_TRUE(expected.empty()) << expected.size() << " records never arrived intact";
}

}  // namespace

TEST(SimDeviceLogTest, MultiThreadedRecordsStayIntact) {
    constexpr int kThreads = 4;
    constexpr int kPerThread = 200;

    std::multiset<std::string> expected;
    for (int t = 0; t < kThreads; ++t) {
        for (int i = 0; i < kPerThread; ++i) {
            char body[32];
            snprintf(body, sizeof(body), "t%d-r%04d", t, i);
            expected.insert(record(t, "worker", body));
        }
    }

    Capture cap = begin_capture();
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([t] {
            for (int i = 0; i < kPerThread; ++i) {
                emit(t, "worker", "t%d-r%04d", t, i);
            }
        });
    }
    for (auto &th : threads) {
        th.join();
    }
    std::string captured = end_capture(cap);

    ASSERT_NO_FATAL_FAILURE(expect_intact(captured, std::move(expected)));
}

TEST(SimDeviceLogTest, ForkedProcessesEmitWholeRecords) {
    constexpr int kChildren = 8;
    constexpr int kPerChild = 100;

    std::multiset<std::string> expected;
    for (int c = 0; c < kChildren; ++c) {
        for (int i = 0; i < kPerChild; ++i) {
            char body[32];
            snprintf(body, sizeof(body), "c%d-r%03d", c, i);
            expected.insert(record(c, "chip_worker", body));
        }
    }

    Capture cap = begin_capture();
    std::vector<pid_t> pids;
    pids.reserve(kChildren);
    for (int c = 0; c < kChildren; ++c) {
        pid_t pid = fork();
        ASSERT_GE(pid, 0);
        if (pid == 0) {
            for (int i = 0; i < kPerChild; ++i) {
                emit(c, "chip_worker", "c%d-r%03d", c, i);
            }
            _exit(0);  // skip gtest/atexit teardown so nothing else hits the pipe
        }
        pids.push_back(pid);
    }
    for (pid_t pid : pids) {
        int status = 0;
        ASSERT_EQ(waitpid(pid, &status, 0), pid);
        ASSERT_TRUE(WIFEXITED(status)) << "child terminated abnormally";
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
    std::string captured = end_capture(cap);

    ASSERT_NO_FATAL_FAILURE(expect_intact(captured, std::move(expected)));
}
