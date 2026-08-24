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

#ifndef SRC_COMMON_PLATFORM_INCLUDE_HOST_RUN_STREAM_SLOTS_H_
#define SRC_COMMON_PLATFORM_INCLUDE_HOST_RUN_STREAM_SLOTS_H_

#include <array>
#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

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
 * Threading: admission gives at most one run owner per slot. A single resident
 * lifecycle worker serializes async retire/replenish for every slot. This keeps
 * two slow, CPU-heavy rtStreamDestroy/rtStreamCreate calls from competing with
 * each other in one rank while the device pipeline remains depth two. The next
 * owner waits only when its own slot is still being replenished. State access is
 * protected separately from the CANN calls, so reading the active peer slot is
 * never blocked behind a slow lifecycle call. `created_count_` is additionally
 * readable from an unrelated thread through `get_run_stream_set_create_count`,
 * so it is atomic. `destroy_all()` walks every slot and requires all runs to be
 * quiesced.
 */
class RunStreamSlots {
public:
    using CreateFn = std::function<int(void **out_stream)>;
    using DestroyFn = std::function<int(void *stream)>;
    using ThreadFactory = std::function<std::thread(std::function<void()>)>;
    using JobWrapper = std::function<std::function<void()>(std::function<void()>)>;

    RunStreamSlots(CreateFn create, DestroyFn destroy) :
        create_(std::move(create)),
        destroy_(std::move(destroy)) {}

    ~RunStreamSlots() { stop_worker(); }

    /**
     * Ready `slot` for a run: its AICPU stream on first use, and always a fresh
     * AICore stream. Fails when the slot still holds an AICore stream a prior
     * run could not retire.
     */
    int acquire(unsigned slot) {
        if (slot >= slots_.size()) return -1;
        {
            std::unique_lock<std::mutex> lock(state_mutex_);
            state_cv_.wait(lock, [this, slot]() {
                return !slots_[slot].destroying;
            });
            Slot &s = slots_[slot];
            if (s.destroy_rc != 0) {
                int rc = s.destroy_rc;
                s.destroy_rc = 0;
                return rc;
            }
            if (s.aicore_stranded) return -1;
            if (s.aicore != nullptr) return 0;
        }

        void *new_aicpu = nullptr;
        void *new_aicore = nullptr;
        {
            std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
            bool need_aicpu = false;
            {
                std::lock_guard<std::mutex> state_lock(state_mutex_);
                need_aicpu = slots_[slot].aicpu == nullptr;
            }
            if (need_aicpu) {
                int rc = create_(&new_aicpu);
                if (rc != 0) return rc;
            }
            int rc = create_(&new_aicore);
            if (rc != 0) {
                if (new_aicpu != nullptr) {
                    std::lock_guard<std::mutex> state_lock(state_mutex_);
                    slots_[slot].aicpu = new_aicpu;
                }
                return rc;
            }
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            Slot &s = slots_[slot];
            if (new_aicpu != nullptr) s.aicpu = new_aicpu;
            s.aicore = new_aicore;
            s.destroy_rc = 0;
            s.aicore_stranded = false;
        }
        created_count_.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    /** Retire `slot`'s AICore stream. The handle survives a failed destroy. */
    int retire_aicore(unsigned slot) {
        if (slot >= slots_.size()) return -1;
        void *retired = nullptr;
        {
            std::unique_lock<std::mutex> lock(state_mutex_);
            state_cv_.wait(lock, [this, slot]() {
                return !slots_[slot].destroying;
            });
            Slot &s = slots_[slot];
            if (s.destroy_rc != 0) {
                int rc = s.destroy_rc;
                s.destroy_rc = 0;
                return rc;
            }
            retired = s.aicore;
            if (retired == nullptr) return 0;
        }
        int rc = 0;
        {
            std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
            rc = destroy_(retired);
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            Slot &s = slots_[slot];
            // The synchronous caller observes `rc` directly. Keep only the
            // poisoned-handle state; otherwise teardown would report the same
            // already-observed destroy failure a second time before retrying it.
            s.destroy_rc = 0;
            s.aicore_stranded = rc != 0;
            if (rc == 0) s.aicore = nullptr;
        }
        return rc;
    }

    /**
     * Start retirement after the stream has been reaped, then immediately
     * create the fresh stream for the slot's next owner. The handle stays in
     * the slot until destroy succeeds; acquire() joins and checks both actions.
     */
    int retire_aicore_async(
        unsigned slot, const ThreadFactory &thread_factory, const JobWrapper &job_wrapper = JobWrapper{}
    ) {
        if (slot >= slots_.size()) return -1;
        std::unique_lock<std::mutex> lock(state_mutex_);
        Slot &s = slots_[slot];
        if (stopping_ || s.destroying || s.aicore == nullptr) return -1;
        if (!worker_started_) {
            try {
                retire_worker_ = thread_factory([this]() {
                    worker_loop();
                });
            } catch (...) {
                return -1;
            }
            if (!retire_worker_.joinable()) return -1;
            worker_started_ = true;
        }
        std::function<void()> job = [this, slot]() {
            retire_and_replenish(slot);
        };
        try {
            if (job_wrapper) job = job_wrapper(std::move(job));
            queue_.push_back(std::move(job));
        } catch (...) {
            return -1;
        }
        s.destroying = true;
        s.destroy_rc = 0;
        lock.unlock();
        worker_cv_.notify_one();
        return 0;
    }

    /** Destroy every stream, keeping handles whose destroy failed. */
    int destroy_all() {
        int first_error = 0;
        for (unsigned slot = 0; slot < slots_.size(); ++slot) {
            void *streams[2] = {nullptr, nullptr};
            {
                std::unique_lock<std::mutex> lock(state_mutex_);
                state_cv_.wait(lock, [this, slot]() {
                    return !slots_[slot].destroying;
                });
                Slot &s = slots_[slot];
                if (s.destroy_rc != 0 && first_error == 0) first_error = s.destroy_rc;
                streams[0] = s.aicpu;
                streams[1] = s.aicore;
            }
            for (unsigned index = 0; index < 2; ++index) {
                if (streams[index] == nullptr) continue;
                int rc = 0;
                {
                    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
                    rc = destroy_(streams[index]);
                }
                if (rc != 0) {
                    if (first_error == 0) first_error = rc;
                    continue;
                }
                std::lock_guard<std::mutex> lock(state_mutex_);
                Slot &s = slots_[slot];
                if (index == 0 && s.aicpu == streams[index]) s.aicpu = nullptr;
                if (index == 1 && s.aicore == streams[index]) {
                    s.aicore = nullptr;
                    s.aicore_stranded = false;
                }
            }
        }
        return first_error;
    }

    void *aicpu(unsigned slot) const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return slot < slots_.size() ? slots_[slot].aicpu : nullptr;
    }
    void *aicore(unsigned slot) const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return slot < slots_.size() ? slots_[slot].aicore : nullptr;
    }
    bool ready(unsigned slot) const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return slot < slots_.size() && !slots_[slot].destroying && slots_[slot].aicpu != nullptr &&
               slots_[slot].aicore != nullptr;
    }
    size_t created_count() const { return created_count_.load(std::memory_order_relaxed); }
    static constexpr size_t capacity() { return PTO_PIPELINE_MAX_DEPTH; }

private:
    struct Slot {
        void *aicpu{nullptr};
        void *aicore{nullptr};
        int destroy_rc{0};
        bool destroying{false};
        bool aicore_stranded{false};
    };

    void worker_loop() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(state_mutex_);
                worker_cv_.wait(lock, [this]() {
                    return stopping_ || !queue_.empty();
                });
                if (stopping_ && queue_.empty()) return;
                job = std::move(queue_.front());
                queue_.pop_front();
            }
            job();
        }
    }

    void retire_and_replenish(unsigned slot) noexcept {
        void *retired = nullptr;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            retired = slots_[slot].aicore;
        }
        void *fresh = retired;
        int rc = -1;
        try {
            std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
            rc = destroy_(retired);
            if (rc == 0) {
                fresh = nullptr;
                rc = create_(&fresh);
            }
        } catch (...) {
            rc = -1;
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            Slot &s = slots_[slot];
            s.aicore = fresh;
            s.destroy_rc = rc;
            s.destroying = false;
            s.aicore_stranded = rc != 0 && fresh == retired;
        }
        if (rc == 0) created_count_.fetch_add(1, std::memory_order_relaxed);
        state_cv_.notify_all();
    }

    void stop_worker() noexcept {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            stopping_ = true;
        }
        worker_cv_.notify_all();
        if (retire_worker_.joinable()) retire_worker_.join();
    }

    CreateFn create_;
    DestroyFn destroy_;
    std::array<Slot, PTO_PIPELINE_MAX_DEPTH> slots_{};
    std::atomic<size_t> created_count_{0};
    mutable std::mutex state_mutex_;
    std::condition_variable state_cv_;
    std::condition_variable worker_cv_;
    std::mutex lifecycle_mutex_;
    std::deque<std::function<void()>> queue_;
    std::thread retire_worker_{};
    bool worker_started_{false};
    bool stopping_{false};
};

#endif  // SRC_COMMON_PLATFORM_INCLUDE_HOST_RUN_STREAM_SLOTS_H_
