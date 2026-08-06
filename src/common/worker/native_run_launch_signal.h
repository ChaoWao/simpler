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

#include <cstdint>
#include <condition_variable>
#include <mutex>

#include "native_run_execution.h"

/** Sticky one-shot wakeup for the host thread waiting on launch readiness, and
 * the per-run home of the launch-acceptance target the host binds before launch
 * and the device publishes at its real kernel-launch marker. */
class NativeRunLaunchSignal {
public:
    explicit NativeRunLaunchSignal(volatile int32_t *accepted_state = nullptr, int32_t accepted_value = 0) :
        accepted_state_(accepted_state),
        accepted_value_(accepted_value) {}
    NativeRunLaunchSignal(const NativeRunLaunchSignal &) = delete;
    NativeRunLaunchSignal &operator=(const NativeRunLaunchSignal &) = delete;

    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() {
            return notified_;
        });
    }

    /** Publish only from the receipt for this run's completed launch transaction. */
    bool publish_receipt(const LaunchReceipt &receipt, const NativeRunIdentity &identity) {
        if (!receipt.matches(identity)) return false;
        {
            std::scoped_lock lock(mutex_);
            if (notified_) return true;
            if (accepted_state_ != nullptr) {
                __atomic_store_n(accepted_state_, accepted_value_, __ATOMIC_RELEASE);
            }
            accepted_state_ = nullptr;
            notified_ = true;
        }
        cv_.notify_one();
        return true;
    }

    void notify() {
        {
            std::scoped_lock lock(mutex_);
            if (notified_) return;
            accepted_state_ = nullptr;
            notified_ = true;
        }
        cv_.notify_one();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    volatile int32_t *accepted_state_{nullptr};
    int32_t accepted_value_{0};
    bool notified_{false};
};
