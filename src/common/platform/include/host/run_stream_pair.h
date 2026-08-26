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

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <functional>
#include <mutex>
#include <utility>

#include "runtime_c_api.h"

/**
 * The one AICPU + AICore stream pair every run submits on.
 *
 * A stream is an ordered queue, so consecutive runs share one pair and may be
 * submitted before their predecessors complete. The pair is not indexed by
 * pipeline slot: slots select run-owned preparation state and completion
 * events, while the stream order serializes device execution.
 *
 * The two streams must stay distinct. The AICPU Run kernel spins in the
 * handshake waiting for the AICore workers, so serializing both onto one queue
 * would leave the AICore submission behind a spin that can never end.
 *
 * The AICore stream carries instruction-cache state: cores may retain code
 * fetched from a GM address whose contents a later registration replaced.
 * Publishing new AICore code therefore marks the stream stale, and the next
 * launch destroys it and creates a replacement. Creating a stream is the only
 * operation known to leave a core free of the previous image's instructions.
 *
 * Submitted owners retire in FIFO order after proven completion. An unproven
 * retirement poisons reuse and defers stream destruction until no queued owner
 * remains. A run that never submitted leaves the live pair alone.
 *
 * Threading: launch and drain are the owning operations, but poll may query the
 * pair from a progress thread while the executor retires it. The pair therefore
 * serializes query with handle mutation. Poll uses try-lock and reports
 * NOT_READY rather than waiting behind retirement. `created_count_` is
 * additionally readable from unrelated threads and remains atomic. Stream
 * creation and destruction are injected so this state machine is exercisable
 * without a device.
 */
class RunStreamPair {
private:
    struct Submission {
        const void *owner{nullptr};
        bool complete{false};
    };
    Submission *find_submission(const void *owner) {
        auto end = submissions_.begin() + static_cast<std::ptrdiff_t>(submission_count_);
        auto match = std::find_if(submissions_.begin(), end, [owner](const Submission &submission) {
            return submission.owner == owner;
        });
        return match == end ? nullptr : &*match;
    }

public:
    using CreateFn = std::function<int(void **out_stream)>;
    using DestroyFn = std::function<int(void *stream)>;
    enum class CompletionStatus { Unproven, Complete };

    RunStreamPair(CreateFn create, DestroyFn destroy) :
        create_(std::move(create)),
        destroy_(std::move(destroy)) {}

    /**
     * Ready the pair for a launch: both streams on first use, and a
     * replacement AICore stream when a code publication marked it stale.
     * A warm pair accepts another submission while earlier runs remain queued.
     * A stale pair can only be replaced after every submitted run retires.
     */
    int ensure() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (aicpu_ == nullptr) {
            int rc = create_(&aicpu_);
            if (rc != 0) {
                aicpu_ = nullptr;
                return rc;
            }
        }
        if (aicore_ != nullptr) {
            if (submission_count_ != 0) return stale_ || unproven_ ? PTO_RUNTIME_ERR_INTERNAL : 0;
            if (!stale_) {
                return 0;
            }
            int rc = destroy_(aicore_);
            if (rc != 0) return rc;
            aicore_ = nullptr;
        }
        int rc = create_(&aicore_);
        if (rc != 0) {
            aicore_ = nullptr;
            return rc;
        }
        created_count_.fetch_add(1, std::memory_order_relaxed);
        stale_ = false;
        unproven_ = false;
        return 0;
    }

    /** Mark the AICore stream stale after new AICore code is published. */
    void mark_stale() {
        std::lock_guard<std::mutex> lock(mutex_);
        stale_ = true;
    }

    /** Append one submitter to the stream-ordered in-flight FIFO. */
    int mark_submitted(const void *owner) {
        if (owner == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        std::lock_guard<std::mutex> lock(mutex_);
        if (aicpu_ == nullptr || aicore_ == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        if (stale_ || unproven_ || submission_count_ >= submissions_.size()) {
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        if (find_submission(owner) != nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        submissions_[submission_count_++] = Submission{owner, false};
        return 0;
    }

    /**
     * Query both streams without waiting behind retirement.
     *
     * A completed result is sticky until the pair is readied for its next run,
     * so a poll racing with successful stream destruction never observes a
     * missing handle as an error.
     */
    template <typename QueryPairFn>
    int poll(const void *owner, QueryPairFn &&query) {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock()) return SIMPLER_NATIVE_RUN_POLL_NOT_READY;
        Submission *submission = find_submission(owner);
        if (submission == nullptr || aicpu_ == nullptr || aicore_ == nullptr) {
            return SIMPLER_NATIVE_RUN_POLL_ERROR;
        }
        if (submission->complete) return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
        const int rc = std::forward<QueryPairFn>(query)(aicpu_, aicore_);
        if (rc == SIMPLER_NATIVE_RUN_POLL_COMPLETE) submission->complete = true;
        return rc;
    }

    /**
     * Retire the pair on behalf of the run that submitted it. Complete keeps
     * the AICore stream for the next launch. Unproven destroys it: a failed
     * launch or an abandoned drain may leave the stream in the error state
     * rtStreamDestroy is the supported teardown for, and the handle survives a
     * failed destroy so teardown can retry it. A caller that never submitted
     * retires nothing.
     */
    int retire(CompletionStatus completion_status, const void *owner) {
        std::lock_guard<std::mutex> lock(mutex_);
        Submission *submission = find_submission(owner);
        if (submission == nullptr) return 0;
        const size_t index = static_cast<size_t>(submission - submissions_.data());
        if (completion_status == CompletionStatus::Complete && index != 0) {
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        // Publish the proven terminal state before destroying the handle. Poll
        // either finishes its in-flight query first or observes this result.
        // An error-path retirement clears a completion that raced ahead of a
        // later failing sync, so the sync error remains authoritative.
        if (completion_status == CompletionStatus::Unproven) unproven_ = true;
        for (size_t i = index + 1; i < submission_count_; ++i)
            submissions_[i - 1] = submissions_[i];
        submissions_[--submission_count_] = Submission{};
        if (submission_count_ != 0 || !unproven_ || aicore_ == nullptr) return 0;
        int rc = destroy_(aicore_);
        if (rc != 0) return rc;
        aicore_ = nullptr;
        unproven_ = false;
        return 0;
    }

    /** Destroy both streams, keeping a handle whose destroy failed. */
    int destroy() {
        std::lock_guard<std::mutex> lock(mutex_);
        int first_error = 0;
        submissions_.fill(Submission{});
        submission_count_ = 0;
        unproven_ = false;
        for (void **stream : {&aicpu_, &aicore_}) {
            if (*stream == nullptr) continue;
            int rc = destroy_(*stream);
            if (rc != 0) {
                if (first_error == 0) first_error = rc;
                continue;
            }
            *stream = nullptr;
        }
        // A handle that survived teardown may still hold the previous image's
        // instructions, so it stays stale for whoever retries it.
        stale_ = aicore_ != nullptr;
        return first_error;
    }

    /** Forget both handles after a device reset without invoking destroy_. */
    void abandon() {
        std::lock_guard<std::mutex> lock(mutex_);
        aicpu_ = nullptr;
        aicore_ = nullptr;
        stale_ = false;
        submissions_.fill(Submission{});
        submission_count_ = 0;
        unproven_ = false;
    }

    // Handle reads are unsynchronized: the claim holder is the only writer once
    // the pair is readied, and poll must never block behind a retirement.
    void *aicpu() const { return aicpu_; }
    void *aicore() const { return aicore_; }
    bool ready() const { return aicpu_ != nullptr && aicore_ != nullptr; }
    size_t created_count() const { return created_count_.load(std::memory_order_relaxed); }

private:
    mutable std::mutex mutex_;
    void *aicpu_{nullptr};
    void *aicore_{nullptr};
    bool stale_{false};
    bool unproven_{false};
    std::array<Submission, PTO_PIPELINE_MAX_DEPTH> submissions_{};
    size_t submission_count_{0};

    CreateFn create_;
    DestroyFn destroy_;
    std::atomic<size_t> created_count_{0};
};
