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

/**
 * The bounded pool of threads that record Graph bodies, and the copy of a boundary's
 * arguments each queued recording owns.
 *
 * Runtime-owned, and only in the host target: one pool per process, serving every
 * registered callable. It used to live in orchestration_api.h, which put a
 * function-local static — and therefore a pool, eight threads and their recording
 * storage — inside every orchestration .so. Those are dlopen'd one per callable with
 * RTLD_LOCAL and are not released as cases finish, so a process held one pool per
 * registered callable: measured at 3 concurrently mapped orchestration images over a
 * four-case pytest session and 5 over the a2a3 host_build_graph corpus, i.e. 24-40
 * recorder threads where 8 suffice.
 *
 * A worker runs jobs the orchestration .so builds (each captures that .so's generated
 * orchestration function), reached through the ops table. Two properties make that
 * sound and are relied on here:
 *
 *   - No job outlives the bind that queued it. rt_orchestration_done() ->
 *     rt_graph_commit() -> graph_record_wait() drains the pool at the end of every
 *     orchestration, so a job's code cannot still be queued when unregister_callable
 *     dlcloses the .so it lives in.
 *   - The runtime a job binds to is a plain global in its own .so
 *     (orchestration/common.cpp, deliberately not thread_local), so a worker shared
 *     across callables reads the right one: the job's own inlined code reads its own
 *     .so's global.
 */

#include <pthread.h>

#include <array>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

// Deliberately no runtime_core.h: it declares the real RuntimeOps while
// orchestration_api.h declares a partial one, so a translation unit that sees both fails
// to compile. Only the ops implementations in graph_recorder_pool.cpp need it.
#include "graph_host_state.h"        // GRAPH_MAX_DEFINITIONS
#include "host_build_graph/types.h"  // GraphTaskArgs, GRAPH_MAX_{TENSOR,SCALAR}_ARGS

class GraphOwnedArgs {
public:
    GraphOwnedArgs() { std::memset(tensors_.data(), 0, sizeof(tensors_)); }

    // The arrays below are sized to the Graph boundary's own capacity, so a
    // source GraphTaskArgs cannot report more args than they hold and the copy
    // loops need no runtime bound.
    void assign(const GraphTaskArgs &source) {
        args_.reset();
        for (int32_t i = 0; i < source.tensor_count(); ++i) {
            tensors_[static_cast<size_t>(i)].copy(source.tensor(i).ref());
            switch (source.tag(i)) {
            case TensorArgType::INPUT:
                args_.add_input(tensors_[static_cast<size_t>(i)]);
                break;
            case TensorArgType::OUTPUT_EXISTING:
                args_.add_output(tensors_[static_cast<size_t>(i)]);
                break;
            case TensorArgType::INOUT:
                args_.add_inout(tensors_[static_cast<size_t>(i)]);
                break;
            case TensorArgType::NO_DEP:
                args_.add_no_dep(tensors_[static_cast<size_t>(i)]);
                break;
            case TensorArgType::OUTPUT:
                args_.set_error("Runtime-allocated output is not supported at a Graph boundary");
                break;
            }
        }
        for (int32_t i = 0; i < source.scalar_count(); ++i) {
            scalars_[static_cast<size_t>(i)] = source.scalar(i);
            args_.add_scalar(scalars_[static_cast<size_t>(i)]);
        }
        args_.launch_spec = source.launch_spec;
        args_.set_allow_early_resolve(source.allow_early_resolve());
        if (source.task_timing_slot() != TASK_TIMING_SLOT_NONE) {
            args_.set_task_timing_slot(source.task_timing_slot());
        }
        args_.set_predicate(source.predicate());
    }

    GraphTaskArgs &args() { return args_; }

private:
    std::array<simpler::hbg::Tensor, GRAPH_MAX_TENSOR_ARGS> tensors_{};
    std::array<uint64_t, GRAPH_MAX_SCALAR_ARGS> scalars_{};
    GraphTaskArgs args_;
};

class GraphAsyncRecordingState {
public:
    GraphAsyncRecordingState() {
        for (size_t i = 0; i < kJobCapacity; ++i) {
            free_owned_args_[i] = kJobCapacity - i - 1;
        }
    }
    ~GraphAsyncRecordingState() { shutdown(); }

    GraphAsyncRecordingState(const GraphAsyncRecordingState &) = delete;
    GraphAsyncRecordingState &operator=(const GraphAsyncRecordingState &) = delete;

    // A workload that cuts its forward pass into up to eight Definitions should
    // not create pthreads between outer shell submissions. Callable registration
    // parks those workers before the first run; start() can still grow to the
    // Definition limit if a workload records more identities concurrently.
    bool prewarm() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (stopping_) return false;
        while (workers_.size() < kPrewarmedWorkerCount) {
            if (!create_worker_locked()) break;
        }
        const size_t target = workers_.size();
        cv_.wait(lock, [&]() {
            return ready_workers_ >= target || stopping_;
        });
        return !stopping_ && !storage_failed_ && target == kPrewarmedWorkerCount;
    }

    template <typename Job>
    bool start(const GraphTaskArgs &args, Job &&job) {
        std::function<void(GraphTaskArgs &)> next;
        try {
            next = std::forward<Job>(job);
        } catch (...) {
            return false;
        }

        size_t owned_args_index;
        {
            std::scoped_lock lock(mutex_);
            if (stopping_ || free_owned_args_count_ == 0 || job_count_ == kJobCapacity) return false;
            owned_args_index = free_owned_args_[--free_owned_args_count_];
        }
        owned_args_[owned_args_index].assign(args);

        std::unique_lock<std::mutex> lock(mutex_);
        if (stopping_) {
            free_owned_args_[free_owned_args_count_++] = owned_args_index;
            return false;
        }
        PendingJob &pending = jobs_[job_tail_];
        pending.function = std::move(next);
        pending.owned_args_index = owned_args_index;
        job_tail_ = (job_tail_ + 1) % kJobCapacity;
        job_count_++;
        const size_t desired_workers = std::min(kMaxWorkerCount, job_count_ + active_jobs_);
        while (workers_.size() < desired_workers) {
            if (!create_worker_locked()) break;
        }
        if (workers_.empty()) {
            job_tail_ = (job_tail_ + kJobCapacity - 1) % kJobCapacity;
            PendingJob &rollback = jobs_[job_tail_];
            rollback.function = {};
            job_count_--;
            free_owned_args_[free_owned_args_count_++] = rollback.owned_args_index;
            return false;
        }
        lock.unlock();
        cv_.notify_one();
        // graph_begin() has already installed the keyed in-flight entry and
        // submitted the zero-heap outer shell. Enqueuing the private job is
        // therefore the last dependency of the caller; graph_prepare() and the
        // recording of the body may start after later shells are submitted.
        return true;
    }

    // Wait for every queued and running recording. A recording thread returns
    // immediately: it may reach this through rt_orchestration_done in a Graph body
    // and must never wait for its own job, nor for a sibling's — the sibling makes
    // progress independently and waiting on it would trade a recording thread for
    // nothing.
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (is_worker_thread_locked(std::this_thread::get_id())) return;
        cv_.wait(lock, [&]() {
            return job_count_ == 0 && active_jobs_ == 0;
        });
    }

    // Sum of every recording worker's CPU time. Differencing this across a phase says
    // how many threads' worth of work ran alongside the thread that runs the bind,
    // which a wall-clock duration cannot: the recorders overlap it.
    //
    // A per-thread CPU clock is the only instrument that resolves a phase of a
    // millisecond. rusage times are accounted per scheduler tick, 10 ms at
    // CLK_TCK=100, so on such a phase they quantise to either zero or a whole tick.
    //
    // Read under this pool's own mutex, from whichever thread asks, so no handle is
    // sampled while it is being created or destroyed: create_worker_locked() runs under
    // that mutex, and shutdown() empties workers_ under it before it joins anything. A
    // thread that has already exited answers EINVAL and contributes nothing, which is
    // correct: its time belongs to no phase still being measured.
    //
    // Linux only, and zero elsewhere: reading another thread's CPU clock needs
    // pthread_getcpuclockid, which Darwin does not provide. The bind breakdown profiles
    // onboard runs, so the platforms that lack it are the ones that never take this
    // measurement -- but rec_cpu_ns carries no information there, and the tooling says so.
    uint64_t worker_cpu_ns() {
        std::scoped_lock lock(mutex_);
        uint64_t total = 0;
#if defined(__linux__)
        for (std::thread &worker : workers_) {
            if (!worker.joinable()) continue;
            clockid_t clock_id{};
            if (pthread_getcpuclockid(worker.native_handle(), &clock_id) != 0) continue;
            timespec ts{};
            if (clock_gettime(clock_id, &ts) != 0) continue;
            total += static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
        }
#endif
        return total;
    }

private:
    bool storage_failed_{false};

    static constexpr size_t kPrewarmedWorkerCount = 8;
    static constexpr size_t kMaxWorkerCount = GRAPH_MAX_DEFINITIONS;
    static constexpr size_t kJobCapacity = GRAPH_MAX_DEFINITIONS;

    struct PendingJob {
        std::function<void(GraphTaskArgs &)> function;
        size_t owned_args_index{0};
    };

    bool is_worker_thread_locked(std::thread::id id) const {
        for (const std::thread &worker : workers_) {
            if (worker.get_id() == id) return true;
        }
        return false;
    }

    bool create_worker_locked() {
        if (stopping_ || workers_.size() >= kMaxWorkerCount) return false;
        try {
            workers_.emplace_back([this]() {
                run();
            });
            return true;
        } catch (...) {
            return false;
        }
    }

    void run() {
        // Named so a `top -H` or a debugger can tell these apart from the scheduler and
        // collector threads a Worker also holds -- there are eight of them and they are
        // idle most of a run, which otherwise looks like a leak.
#if defined(__linux__)
        pthread_setname_np(pthread_self(), "hbg-recorder");
#endif
        // Before the ready handshake, so prewarm() does not return until every worker's
        // recording storage is standing. A worker that could not stand its storage up
        // still reports ready and prewarm() fails: the lazy stand-up in
        // graph_recording_reset is the backstop for the workers start() adds later.
        if (!graph_recorder_stand_up_storage()) {
            std::scoped_lock lock(mutex_);
            storage_failed_ = true;
        }
        {
            std::scoped_lock lock(mutex_);
            ready_workers_++;
        }
        cv_.notify_all();
        for (;;) {
            PendingJob current;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&]() {
                    return job_count_ != 0 || stopping_;
                });
                if (stopping_ && job_count_ == 0) return;
                PendingJob &pending = jobs_[job_head_];
                current.function = std::move(pending.function);
                current.owned_args_index = pending.owned_args_index;
                job_head_ = (job_head_ + 1) % kJobCapacity;
                job_count_--;
                active_jobs_++;
            }
            current.function(owned_args_[current.owned_args_index].args());
            current.function = {};
            {
                std::scoped_lock lock(mutex_);
                free_owned_args_[free_owned_args_count_++] = current.owned_args_index;
                active_jobs_--;
            }
            cv_.notify_all();
        }
    }

    void shutdown() {
        wait();
        std::vector<std::thread> draining;
        {
            std::scoped_lock lock(mutex_);
            stopping_ = true;
            // The handles leave workers_ under the lock and are joined without it.
            // Joining while holding mutex_ deadlocks: a worker reacquires it inside
            // cv_.wait to observe stopping_ and return. Emptying workers_ here is also
            // what keeps worker_cpu_ns() off a handle mid-join, since it samples under
            // this same mutex and so sees an empty pool for the rest of teardown.
            draining.swap(workers_);
        }
        cv_.notify_all();
        for (std::thread &worker : draining) {
            if (worker.joinable()) worker.join();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::array<GraphOwnedArgs, kJobCapacity> owned_args_;
    std::array<size_t, kJobCapacity> free_owned_args_{};
    std::array<PendingJob, kJobCapacity> jobs_;
    size_t free_owned_args_count_{kJobCapacity};
    size_t job_head_{0};
    size_t job_tail_{0};
    size_t job_count_{0};
    size_t ready_workers_{0};
    size_t active_jobs_{0};
    bool stopping_{false};
};

/**
 * The process's one recorder pool.
 *
 * A function-local static here is one instance per *runtime* .so rather than per
 * orchestration .so, which is the whole point of the move.
 */
GraphAsyncRecordingState &graph_recorder_pool();

/**
 * Park the prewarmed workers and stand each one's recording storage up.
 *
 * Called from callable registration. Idempotent across callables: the second and later
 * registrations find the pool already at kPrewarmedWorkerCount and return immediately,
 * which is what makes one pool per process work.
 *
 * @return false when a worker could not be created or could not stand its storage up.
 */
bool graph_recorder_prewarm();
