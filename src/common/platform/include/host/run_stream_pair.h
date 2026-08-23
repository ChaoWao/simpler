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
#include <cstddef>
#include <functional>
#include <mutex>
#include <utility>

#include "runtime_c_api.h"

/**
 * The one AICPU + AICore stream pair every run submits on.
 *
 * A stream is an ordered queue and the execution claim is exclusive, so runs
 * reach the device one at a time and a single pair carries all of them. The
 * pair is not indexed by pipeline slot: a slot exists for resources that
 * *preparation* mutates, and preparing a run writes nothing to a stream — only
 * launch submits, and launch holds the claim.
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
 * Only the run that submitted the pair may retire it. A prepared successor
 * overlaps its predecessor's execution, so an unproven retirement from a run
 * that never submitted must leave the live pair alone.
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
     * Callers hold the execution claim, so the pair is idle here.
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
            // A run that has not retired still owns the pair: a device-complete
            // poll is not a finalized run, and replacing the stream under it
            // would strand a live submission.
            if (owner_ != nullptr) return PTO_RUNTIME_ERR_INTERNAL;
            if (!stale_) {
                submitted_ = false;
                complete_ = false;
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
        submitted_ = false;
        complete_ = false;
        return 0;
    }

    /** Mark the AICore stream stale after new AICore code is published. */
    void mark_stale() {
        std::lock_guard<std::mutex> lock(mutex_);
        stale_ = true;
    }

    /** Make the pair visible to non-blocking poll and record its submitter. */
    int mark_submitted(const void *owner) {
        if (owner == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        std::lock_guard<std::mutex> lock(mutex_);
        if (aicpu_ == nullptr || aicore_ == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        owner_ = owner;
        submitted_ = true;
        complete_ = false;
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
    int poll(QueryPairFn &&query) {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock()) return SIMPLER_NATIVE_RUN_POLL_NOT_READY;
        if (complete_) return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
        if (!submitted_ || aicpu_ == nullptr || aicore_ == nullptr) {
            return SIMPLER_NATIVE_RUN_POLL_ERROR;
        }
        const int rc = std::forward<QueryPairFn>(query)(aicpu_, aicore_);
        if (rc == SIMPLER_NATIVE_RUN_POLL_COMPLETE) complete_ = true;
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
        if (owner == nullptr || owner_ != owner) return 0;
        // Publish the proven terminal state before destroying the handle. Poll
        // either finishes its in-flight query first or observes this result.
        // An error-path retirement clears a completion that raced ahead of a
        // later failing sync, so the sync error remains authoritative.
        submitted_ = false;
        complete_ = completion_status == CompletionStatus::Complete;
        owner_ = nullptr;
        if (completion_status == CompletionStatus::Complete) return 0;
        if (aicore_ == nullptr) return 0;
        int rc = destroy_(aicore_);
        if (rc != 0) return rc;
        aicore_ = nullptr;
        return 0;
    }

    /** Destroy both streams, keeping a handle whose destroy failed. */
    int destroy() {
        std::lock_guard<std::mutex> lock(mutex_);
        int first_error = 0;
        submitted_ = false;
        complete_ = false;
        owner_ = nullptr;
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
        submitted_ = false;
        complete_ = false;
        owner_ = nullptr;
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
    // The run that submitted the pair, or null while no run owns it.
    const void *owner_{nullptr};
    bool stale_{false};
    bool submitted_{false};
    bool complete_{false};

    CreateFn create_;
    DestroyFn destroy_;
    std::atomic<size_t> created_count_{0};
};
