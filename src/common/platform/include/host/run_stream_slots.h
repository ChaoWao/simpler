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

#include <array>
#include <atomic>
#include <cstddef>
#include <functional>
#include <mutex>
#include <utility>

#include "pto_runtime_c_api.h"

/**
 * Per-slot ownership of the two streams a run submits on.
 *
 * The AICPU stream carries no instruction-cache state, so it belongs to its
 * slot for the runner's lifetime. The AICore stream belongs to a single run:
 * the platform offers no instruction-cache invalidation for code replaced at a
 * reused GM address, and creating the stream is the only operation known to
 * leave a core free of the previous image's instructions — selecting an
 * existing one is not.
 *
 * A destroy that fails keeps its handle. Such a stream may still hold the
 * previous image's instructions, so the slot must refuse the next run rather
 * than hand it a fresh stream beside a live one, and teardown needs the handle
 * to retry. Stream creation and destruction are injected so this state machine
 * is exercisable without a device.
 *
 * Threading: prepare and drain remain the owning operations, but poll may query
 * the active slot from a progress thread while the executor retires its AICore
 * stream. Each slot therefore serializes query with handle mutation. Poll uses
 * try-lock and reports NOT_READY rather than waiting behind retirement. Two
 * different slots remain independent, so native prepare can acquire a
 * successor while the executor drains its predecessor. `created_count_` is
 * additionally readable from unrelated threads and remains atomic.
 */
class RunStreamSlots {
public:
    using CreateFn = std::function<int(void **out_stream)>;
    using DestroyFn = std::function<int(void *stream)>;
    enum class CompletionStatus { Unproven, Complete };

    RunStreamSlots(CreateFn create, DestroyFn destroy) :
        create_(std::move(create)),
        destroy_(std::move(destroy)) {}

    /**
     * Ready `slot` for a run: its AICPU stream on first use, and always a fresh
     * AICore stream. Fails when the slot still holds an AICore stream a prior
     * run could not retire.
     */
    int acquire(unsigned slot) {
        if (slot >= slots_.size()) return -1;
        Slot &s = slots_[slot];
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.aicpu == nullptr) {
            int rc = create_(&s.aicpu);
            if (rc != 0) {
                s.aicpu = nullptr;
                return rc;
            }
        }
        if (s.aicore != nullptr) return -1;
        int rc = create_(&s.aicore);
        if (rc != 0) {
            s.aicore = nullptr;
            return rc;
        }
        created_count_.fetch_add(1, std::memory_order_relaxed);
        s.submitted = false;
        s.complete = false;
        return 0;
    }

    /** Make the prepared stream pair visible to non-blocking poll. */
    int mark_submitted(unsigned slot) {
        if (slot >= slots_.size()) return -1;
        Slot &s = slots_[slot];
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.aicpu == nullptr || s.aicore == nullptr) return -1;
        s.submitted = true;
        s.complete = false;
        return 0;
    }

    /**
     * Query both streams without waiting behind retirement.
     *
     * A completed result is sticky until the slot is acquired for its next
     * generation, so a poll racing with successful stream destruction never
     * observes a missing handle as an error.
     */
    template <typename QueryPairFn>
    int poll(unsigned slot, QueryPairFn &&query) {
        if (slot >= slots_.size()) return SIMPLER_NATIVE_RUN_POLL_ERROR;
        Slot &s = slots_[slot];
        std::unique_lock<std::mutex> lock(s.mutex, std::try_to_lock);
        if (!lock.owns_lock()) return SIMPLER_NATIVE_RUN_POLL_NOT_READY;
        if (s.complete) return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
        if (!s.submitted || s.aicpu == nullptr || s.aicore == nullptr) {
            return SIMPLER_NATIVE_RUN_POLL_ERROR;
        }
        const int rc = std::forward<QueryPairFn>(query)(s.aicpu, s.aicore);
        if (rc == SIMPLER_NATIVE_RUN_POLL_COMPLETE) s.complete = true;
        return rc;
    }

    /**
     * Retire `slot`'s AICore stream. The handle survives a failed destroy.
     * Only a successful device query or stream synchronization may pass
     * Complete; launch rollback and abandoned preparation pass Unproven.
     */
    int retire_aicore(unsigned slot, CompletionStatus completion_status) {
        if (slot >= slots_.size()) return -1;
        Slot &s = slots_[slot];
        std::lock_guard<std::mutex> lock(s.mutex);
        // Publish the proven terminal state before destroying the handle. Poll
        // either finishes its in-flight query first or observes this result.
        // An error-path retirement clears a completion that raced ahead of a
        // later failing sync, so the sync error remains authoritative.
        s.submitted = false;
        s.complete = completion_status == CompletionStatus::Complete;
        if (s.aicore == nullptr) return 0;
        int rc = destroy_(s.aicore);
        if (rc != 0) return rc;
        s.aicore = nullptr;
        return 0;
    }

    /** Destroy every stream, keeping handles whose destroy failed. */
    int destroy_all() {
        int first_error = 0;
        for (Slot &s : slots_) {
            std::lock_guard<std::mutex> lock(s.mutex);
            s.submitted = false;
            s.complete = false;
            for (void **stream : {&s.aicpu, &s.aicore}) {
                if (*stream == nullptr) continue;
                int rc = destroy_(*stream);
                if (rc != 0) {
                    if (first_error == 0) first_error = rc;
                    continue;
                }
                *stream = nullptr;
            }
        }
        return first_error;
    }

    /** Forget every handle after device reset without invoking destroy_. */
    void abandon_all() {
        for (Slot &s : slots_) {
            s.aicpu = nullptr;
            s.aicore = nullptr;
        }
    }

    void *aicpu(unsigned slot) const { return slot < slots_.size() ? slots_[slot].aicpu : nullptr; }
    void *aicore(unsigned slot) const { return slot < slots_.size() ? slots_[slot].aicore : nullptr; }
    bool ready(unsigned slot) const { return aicpu(slot) != nullptr && aicore(slot) != nullptr; }
    size_t created_count() const { return created_count_.load(std::memory_order_relaxed); }
    static constexpr size_t capacity() { return PTO_PIPELINE_MAX_DEPTH; }

private:
    struct Slot {
        mutable std::mutex mutex;
        void *aicpu{nullptr};
        void *aicore{nullptr};
        bool submitted{false};
        bool complete{false};
    };

    CreateFn create_;
    DestroyFn destroy_;
    std::array<Slot, PTO_PIPELINE_MAX_DEPTH> slots_{};
    std::atomic<size_t> created_count_{0};
};
