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
#include <utility>

class DeviceRunnerBase;
class NativeRunExecutionTestPeer;
class SimDeviceRunnerBase;

struct NativeRunIdentity {
    uint64_t run_epoch{0};
    uint64_t generation{0};
    uint64_t dispatch_id{0};
    uint32_t pipeline_slot{0};

    bool operator==(const NativeRunIdentity &other) const {
        return run_epoch == other.run_epoch && generation == other.generation && dispatch_id == other.dispatch_id &&
               pipeline_slot == other.pipeline_slot;
    }
    bool operator!=(const NativeRunIdentity &other) const { return !(*this == other); }
};

class LaunchPermit {
public:
    LaunchPermit() = default;
    LaunchPermit(const LaunchPermit &) = delete;
    LaunchPermit &operator=(const LaunchPermit &) = delete;

    LaunchPermit(LaunchPermit &&other) noexcept :
        identity_(other.identity_),
        valid_(other.valid_) {
        other.valid_ = false;
    }

    LaunchPermit &operator=(LaunchPermit &&other) noexcept {
        if (this == &other) return *this;
        identity_ = other.identity_;
        valid_ = other.valid_;
        other.valid_ = false;
        return *this;
    }

    bool valid() const { return valid_; }

private:
    friend class DeviceRunnerBase;
    friend class NativeRunExecutionTestPeer;
    friend class SimDeviceRunnerBase;
    template <typename AicoreSubmit, typename AicpuSubmit>
    friend struct ExactLaunchTransaction;

    explicit LaunchPermit(const NativeRunIdentity &identity) :
        identity_(identity),
        valid_(true) {}

    bool consume(const NativeRunIdentity &identity) {
        if (!valid_ || identity_ != identity) return false;
        valid_ = false;
        return true;
    }

    NativeRunIdentity identity_{};
    bool valid_{false};
};

class LaunchReceipt {
public:
    LaunchReceipt() = default;
    LaunchReceipt(const LaunchReceipt &) = delete;
    LaunchReceipt &operator=(const LaunchReceipt &) = delete;

    LaunchReceipt(LaunchReceipt &&other) noexcept :
        identity_(other.identity_),
        valid_(other.valid_) {
        other.valid_ = false;
    }

    LaunchReceipt &operator=(LaunchReceipt &&other) noexcept {
        if (this == &other) return *this;
        identity_ = other.identity_;
        valid_ = other.valid_;
        other.valid_ = false;
        return *this;
    }

    bool valid() const { return valid_; }
    bool matches(const NativeRunIdentity &identity) const { return valid_ && identity_ == identity; }

private:
    template <typename AicoreSubmit, typename AicpuSubmit>
    friend struct ExactLaunchTransaction;

    explicit LaunchReceipt(const NativeRunIdentity &identity) :
        identity_(identity),
        valid_(true) {}

    NativeRunIdentity identity_{};
    bool valid_{false};
};

enum class LaunchProgress : uint8_t {
    NotStarted,
    Partial,
    Complete,
};

struct LaunchTransactionResult {
    int rc{-1};
    LaunchProgress progress{LaunchProgress::NotStarted};
    LaunchReceipt receipt{};

    bool poisoned() const { return progress == LaunchProgress::Partial; }
};

template <typename AicoreSubmit, typename AicpuSubmit>
struct ExactLaunchTransaction {
    static LaunchTransactionResult
    run(const NativeRunIdentity &identity, LaunchPermit permit, AicoreSubmit &&submit_aicore,
        AicpuSubmit &&submit_aicpu) {
        LaunchTransactionResult result;
        if (!permit.consume(identity)) return result;

        try {
            result.rc = std::forward<AicoreSubmit>(submit_aicore)();
        } catch (...) {
            result.rc = -1;
            // Submission callbacks may throw after starting work. Without a
            // receipt, the caller must retain resources and poison admission.
            result.progress = LaunchProgress::Partial;
            return result;
        }
        if (result.rc != 0) return result;

        result.progress = LaunchProgress::Partial;
        try {
            result.rc = std::forward<AicpuSubmit>(submit_aicpu)();
        } catch (...) {
            result.rc = -1;
        }
        if (result.rc != 0) return result;

        result.progress = LaunchProgress::Complete;
        result.receipt = LaunchReceipt(identity);
        return result;
    }
};

template <typename AicoreSubmit, typename AicpuSubmit>
LaunchTransactionResult exact_launch_transaction(
    const NativeRunIdentity &identity, LaunchPermit permit, AicoreSubmit &&submit_aicore, AicpuSubmit &&submit_aicpu
) {
    return ExactLaunchTransaction<AicoreSubmit, AicpuSubmit>::run(
        identity, std::move(permit), std::forward<AicoreSubmit>(submit_aicore), std::forward<AicpuSubmit>(submit_aicpu)
    );
}
