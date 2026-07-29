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

#ifndef SRC_COMMON_WORKER_NATIVE_RUN_STATE_H_
#define SRC_COMMON_WORKER_NATIVE_RUN_STATE_H_

#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>

#include "call_config.h"
#include "native_run_launch_signal.h"
#include "runtime.h"

/** Internal phase of the caller-owned opaque native-run storage. */
enum class NativeRunPhase : uint8_t {
    Prepared,
    Launching,
    Running,
    Complete,
};

/**
 * Caller-owned state for one progressable native lifecycle. Runtime and
 * CallConfig are per-run; Runner-owned streams, diagnostics, and timing require
 * exclusive ownership from prepare through finalize.
 */
template <typename Runner>
struct NativeRunState {
    static constexpr uint64_t kMagic = UINT64_C(0x534d504c52554e31);  // "SMPLRUN1"

    NativeRunState(Runner *runner_in, const CallConfig &config_in, uint64_t trace_hid_in) :
        runner(runner_in),
        config(config_in),
        trace_hid(trace_hid_in) {}

    ~NativeRunState() {
        if (executor.joinable()) executor.join();
        if (host_thread_state != nullptr) {
            runner->destroy_native_run_thread_state(host_thread_state);
        }
    }

    /** Move prepare-thread state into the executor before runner->run(). */
    void adopt_host_thread_state() noexcept {
        void *snapshot = host_thread_state;
        host_thread_state = nullptr;
        if (snapshot != nullptr) runner->adopt_native_run_thread_state(snapshot);
    }

    uint64_t magic{kMagic};
    Runner *runner{nullptr};
    CallConfig config{};
    Runtime runtime{};
    uint64_t trace_hid{0};
    unsigned trace_inv{0};
    long long trace_start_ns{0};
    std::thread executor{};
    std::atomic<int> execution_rc{-1};
    std::atomic<bool> execution_done{false};
    std::atomic<NativeRunPhase> phase{NativeRunPhase::Prepared};
    NativeRunLaunchSignal launch_signal{};
    void *host_thread_state{nullptr};
    bool runner_claimed{false};
};

/** End object lifetime, then mark the caller-owned storage reusable. */
template <typename Runner>
void destroy_native_run_state(NativeRunState<Runner> *state) {
    void *storage = state;
    state->~NativeRunState<Runner>();
    constexpr uint64_t kEmpty = 0;
    std::memcpy(storage, &kEmpty, sizeof(kEmpty));
}

#endif  // SRC_COMMON_WORKER_NATIVE_RUN_STATE_H_
