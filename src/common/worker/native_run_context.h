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

#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>

#include "call_config.h"
#include "common/host_api.h"
#include "native_run_launch_signal.h"
#include "pto_runtime_c_api.h"
#include "runtime.h"

/** Internal phase of the caller-owned opaque native-run storage. */
enum class NativeRunPhase : uint8_t {
    Prepared,
    Launching,
    Running,
    Complete,
};

/**
 * The single placement-owned carrier for one progressable native lifecycle.
 * It captures immutable selection, identity, acceptance, configuration, and
 * HostApi binding during prepare; later phases never recover them from a
 * runner field or thread-local. The object is address-stable through finalize.
 */
template <typename Runner>
struct NativeRunContext {
    static constexpr uint64_t kMagic = UINT64_C(0x534d504c52554e31);  // "SMPLRUN1"

    NativeRunContext(
        Runner *runner_in, const CallConfig &config_in, uint64_t trace_hid_in, const NativeRunDescriptor &descriptor_in,
        const HostApiOps *host_api_ops
    ) :
        runner(runner_in),
        config(config_in),
        descriptor(descriptor_in),
        host_api(runner_in, descriptor_in.pipeline_slot, descriptor_in.arena_bank, host_api_ops),
        trace_hid(trace_hid_in),
        launch_signal(descriptor_in.accepted_state, descriptor_in.accepted_value) {
        // Publish the storage tag only after every potentially-throwing member
        // has been constructed. A failed placement construction must leave the
        // caller-owned slot reusable rather than looking like a prepared run.
        magic = kMagic;
    }

    NativeRunContext(const NativeRunContext &) = delete;
    NativeRunContext &operator=(const NativeRunContext &) = delete;
    NativeRunContext(NativeRunContext &&) = delete;
    NativeRunContext &operator=(NativeRunContext &&) = delete;

    ~NativeRunContext() {
        if (executor.joinable()) executor.join();
        if (host_thread_state != nullptr) {
            runner->destroy_native_run_thread_state(host_thread_state);
        }
    }

    /** Move prepare-thread state into the executor before enqueue and drain. */
    void adopt_host_thread_state() noexcept {
        void *snapshot = host_thread_state;
        host_thread_state = nullptr;
        if (snapshot != nullptr) runner->adopt_native_run_thread_state(snapshot);
    }

    uint64_t magic{0};
    Runner *runner{nullptr};
    CallConfig config{};
    NativeRunDescriptor descriptor{};
    HostApi host_api;
    Runtime runtime{};
    uint64_t trace_hid{0};
    unsigned trace_inv{0};
    long long trace_start_ns{0};
    std::thread executor{};
    std::atomic<int> execution_rc{-1};
    std::atomic<bool> execution_done{false};
    std::atomic<NativeRunPhase> phase{NativeRunPhase::Prepared};
    NativeRunLaunchSignal launch_signal;
    void *host_thread_state{nullptr};
    char trace_attrs[192]{};
    bool runner_resources_owned{false};
    bool runner_reserved{false};
    bool runner_claimed{false};
};

/** End object lifetime, then mark the caller-owned storage reusable. */
template <typename Runner>
void destroy_native_run_context(NativeRunContext<Runner> *context) {
    void *storage = context;
    context->~NativeRunContext<Runner>();
    constexpr uint64_t kEmpty = 0;
    std::memcpy(storage, &kEmpty, sizeof(kEmpty));
}
