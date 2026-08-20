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
/**
 * PTO Orchestration API - Slim header for orchestration .so files
 *
 * This header provides everything an orchestration source needs without
 * pulling in runtime implementation headers.  The orchestration .so has
 * zero link dependencies on runtime .cpp files; all runtime calls go
 * through the PTO2RuntimeOps function-pointer table embedded in
 * PTO2Runtime.
 *
 * Orchestration sources include ONLY this header:
 *   #include "pto_orchestration_api.h"
 *
 * Runtime sources continue to use pto_runtime2.h (which defines the
 * full PTO2Runtime struct with all internal fields).
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// Type headers needed by orchestration
#include "common.h"              // framework_bind_runtime / framework_current_runtime
#include "graph_cache.h"         // Graph Execution key and result helpers
#include "graph_host_state.h"    // GRAPH_MAX_DEFINITIONS
#include "pto_runtime2_types.h"  // PTO2_ERROR_*
#include "pto_submit_types.h"    // MixedKernels, INVALID_KERNEL_ID, subtask slots
#include "pto_types.h"           // Arg, TaskOutputTensors, TensorArgType
#include "task_args.h"           // ChipStorageTaskArgs, ChipTensor
#include "tensor.h"              // ChipTensor, TensorCreateInfo

// =============================================================================
// ChipTensor Factory Helpers
// =============================================================================

// make_tensor_external(...) — canonical factory for pre-allocated external
// memory — is defined in the unified tensor.h (common), so host and runtime
// build ChipTensors through the same controlled path.

// =============================================================================
// Ops Table and Opaque Runtime
// =============================================================================

/**
 * Forward declaration — the orchestration sees PTO2Runtime as a partial
 * struct whose first field is the ops pointer.  The full definition
 * lives in pto_runtime2.h (used only by runtime .cpp files).
 */
typedef struct PTO2Runtime PTO2Runtime;

/**
 * Function-pointer table for runtime operations.
 * Populated by the runtime; called by orchestration through inline wrappers.
 */
typedef struct PTO2RuntimeOps {
    TaskOutputTensors (*submit_task)(PTO2Runtime *rt, const MixedKernels &mixed_kernels, const CoreTaskArgs &args);
    void (*scope_begin)(PTO2Runtime *rt);
    void (*scope_end)(PTO2Runtime *rt);
    void (*orchestration_done)(PTO2Runtime *rt);
    bool (*is_fatal)(PTO2Runtime *rt);
    void (*report_fatal)(PTO2Runtime *rt, int32_t error_code, const char *func, const char *fmt, ...);

    // Logging (populated by runtime, called by orchestration)
    void (*log_error)(const char *func, const char *fmt, ...);
    void (*log_warn)(const char *func, const char *fmt, ...);
    void (*log_timing)(const char *func, const char *fmt, ...);
    void (*log_info)(const char *func, const char *fmt, ...);
    void (*log_debug)(const char *func, const char *fmt, ...);

    // Cross-layer data access (orchestration reads/writes tensor values via runtime)
    // Placed after logging to avoid shifting hot-path field offsets.
    uint64_t (*get_tensor_data)(PTO2Runtime *rt, const ChipTensor &tensor, uint32_t ndims, const uint32_t indices[]);
    void (*set_tensor_data)(
        PTO2Runtime *rt, const ChipTensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value
    );
    TaskOutputTensors (*alloc_tensors)(PTO2Runtime *rt, const CoreTaskArgs &args);
    TaskOutputTensors (*submit_dummy_task)(PTO2Runtime *rt, const CoreTaskArgs &args);

    // This-run core geometry from runtime_finalize_after_wire: MIX clusters
    // (one AIC each) and standalone AIV cores.
    int32_t (*available_cluster_count)(PTO2Runtime *rt);
    int32_t (*available_aiv_count)(PTO2Runtime *rt);
    GraphScopeResult (*graph_begin)(PTO2Runtime *rt, uint64_t graph_key, const GraphTaskArgs &args);
    bool (*graph_prepare)(PTO2Runtime *rt, void *recording_handle, const GraphTaskArgs &args);
    void (*graph_abort)(PTO2Runtime *rt, void *recording_handle);
    bool (*graph_end)(PTO2Runtime *rt);
    void (*graph_commit)(PTO2Runtime *rt);

    // Stash the call-site of the next PTO2ScopeGuard so the [ScopeStats]
    // collector can log it. Always present to keep ops-table layout stable
    // across SIMPLER_DFX settings; set to nullptr at SIMPLER_DFX=0.
    void (*scope_set_site)(const char *file, int line);
} PTO2RuntimeOps;

/**
 * Partial PTO2Runtime definition for orchestration.
 *
 * Exposes the ops pointer (for runtime calls) and pending_scope_mode
 * (read directly by inline scope wrappers).  The real struct (in
 * pto_runtime2.h) has the same first fields, so accessing them through
 * this definition is well-defined (C struct layout guarantee).
 */
struct PTO2Runtime {
    const PTO2RuntimeOps *ops;
    PTO2ScopeMode pending_scope_mode;
};

class GraphOwnedArgs {
public:
    // The arrays below are sized to the Graph boundary's own capacity, so a
    // source GraphTaskArgs cannot report more args than they hold and the copy
    // loops need no runtime bound.
    explicit GraphOwnedArgs(const GraphTaskArgs &source) {
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
    std::array<ChipTensor, GRAPH_MAX_TENSOR_ARGS> tensors_{};
    std::array<uint64_t, GRAPH_MAX_SCALAR_ARGS> scalars_{};
    GraphTaskArgs args_;
};

// Runs Graph recording jobs off the submitting thread. Several Definitions may
// record at once — one per Graph key, which the runtime's in-flight map enforces
// — so this holds a queue and grows a thread per job that has no free worker,
// up to GRAPH_MAX_DEFINITIONS. Growth is lazy and self-sizing: a run with one
// Definition creates exactly one thread.
//
// Growth is eager and on the submitting thread. Deferring it to a worker — each
// one spawning the next after it dequeues — under-provisions: thread creation is
// serialized against jobs that last well under a millisecond, so the chain stops
// growing once an earlier worker goes idle. Measured on the four-Definition
// DeepSeek-V4 decode, that reactive policy produced three recording threads and
// 1.6-1.7x concurrency where eager growth produces four and 2.2-2.5x.
class GraphAsyncRecordingState {
public:
    GraphAsyncRecordingState() = default;
    ~GraphAsyncRecordingState() { shutdown(); }

    GraphAsyncRecordingState(const GraphAsyncRecordingState &) = delete;
    GraphAsyncRecordingState &operator=(const GraphAsyncRecordingState &) = delete;

    template <typename Job>
    bool start(Job &&job) {
        std::function<void()> next;
        try {
            next = std::forward<Job>(job);
        } catch (...) {
            return false;
        }

        std::unique_lock<std::mutex> lock(mutex_);
        if (stopping_) return false;
        try {
            queue_.push_back(std::move(next));
        } catch (...) {
            return false;
        }
        outstanding_.store(queue_.size() + running_, std::memory_order_release);
        // One thread can only serve one recording, so a job that would otherwise
        // queue behind another Definition gets a thread of its own. With no thread
        // at all there is nobody to run it, so that case fails back to the caller;
        // a thread that merely could not be created is a scheduling loss.
        if (queue_.size() > idle_workers_ && !grow_locked() && workers_.empty()) {
            queue_.pop_back();
            outstanding_.store(queue_.size() + running_, std::memory_order_release);
            return false;
        }
        cv_.notify_one();
        // graph_begin() has already installed the keyed in-flight entry and
        // submitted the zero-heap outer shell. Enqueuing the private job is
        // therefore the last dependency of the caller; graph_prepare() and all
        // node recording may start after later shells are submitted.
        return true;
    }

    // Wait for every queued and running recording. A recording thread returns
    // immediately: it may reach this through rt_orchestration_done in a Graph body
    // and must never wait for its own job, nor for a sibling's — the sibling makes
    // progress independently and waiting on it would trade a recording thread for
    // nothing.
    void wait() {
        // The idle case — no Graph recorded yet, or all of them already committed —
        // no Graph recorded yet, or all of them already committed — answers
        // without taking the pool's mutex.
        if (outstanding_.load(std::memory_order_acquire) == 0) return;
        std::unique_lock<std::mutex> lock(mutex_);
        if (is_worker_thread()) return;
        idle_cv_.wait(lock, [&]() {
            return queue_.empty() && running_ == 0;
        });
    }

private:
    bool is_worker_thread() const {
        const std::thread::id self = std::this_thread::get_id();
        for (const std::thread &worker : workers_) {
            if (worker.get_id() == self) return true;
        }
        return false;
    }

    // Caller holds mutex_. A thread this pass cannot create is a scheduling loss,
    // not a correctness one — the job still runs behind an existing worker.
    bool grow_locked() {
        if (stopping_ || workers_.size() >= GRAPH_MAX_DEFINITIONS) return false;
        try {
            workers_.emplace_back([this]() {
                run();
            });
        } catch (...) {
            return false;
        }
        return true;
    }

    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            ++idle_workers_;
            cv_.wait(lock, [&]() {
                return !queue_.empty() || stopping_;
            });
            --idle_workers_;
            if (stopping_) return;
            std::function<void()> current = std::move(queue_.front());
            queue_.pop_front();
            ++running_;
            lock.unlock();
            current();
            lock.lock();
            --running_;
            outstanding_.store(queue_.size() + running_, std::memory_order_release);
            if (queue_.empty() && running_ == 0) idle_cv_.notify_all();
        }
    }

    void shutdown() {
        wait();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        for (std::thread &worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable idle_cv_;
    std::deque<std::function<void()>> queue_;
    // queue_.size() + running_, published for wait()'s lock-free idle test.
    std::atomic<size_t> outstanding_{0};
    size_t idle_workers_{0};
    size_t running_{0};
    bool stopping_{false};
};

// External linkage on purpose: `static inline` would give every translation
// unit that submits a Graph its own recorder and its own worker threads, so a
// commit reached from one TU would not wait for a recording another TU started.
// Vague linkage keeps one instance — and one pool — per loaded SO.
inline GraphAsyncRecordingState &rt_graph_async_recording() {
    // One orchestration SO is invoked serially by its ChipWorker. Keep its
    // recorder alive across runs so steady-state misses pay only a condition-
    // variable wakeup, not a fresh pthread create/join. The SO's destructor
    // stops the idle workers before dlclose unmaps their code.
    static GraphAsyncRecordingState state;
    return state;
}

// =============================================================================
// Inline Convenience Wrappers (call through ops table)
// =============================================================================

static inline PTO2Runtime *current_runtime() { return framework_current_runtime(); }
static inline void rt_graph_commit();

// An ordinary submission depends on nothing a recording produces. The outer
// Graph shell entered the task sequence and registered its TensorMap producers
// at graph_begin, so fanin against it is already correct; the recording only
// builds the Definition image, and the deferred heap block a shell still needs is
// an independent bump allocation that orchestration completion reserves. So none
// of the three wrappers below joins the recorders — a barrier here stalls the
// submitting thread for the rest of every recording in flight, which on the
// four-Definition DeepSeek-V4 decode was a third of the orchestration window.
static inline TaskOutputTensors alloc_tensors(const CoreTaskArgs &args) {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    return rt->ops->alloc_tensors(rt, args);
}

static inline TaskOutputTensors alloc_tensors(const TensorCreateInfo create_infos[], uint32_t count) {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    CoreTaskArgs args;
    for (uint32_t i = 0; i < count; i++) {
        args.add_output(create_infos[i]);
    }
    if (args.has_error) {
        rt->ops->report_fatal(
            rt, PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
            args.error_msg ? args.error_msg : "alloc_tensors failed to construct output-only Arg"
        );
        return TaskOutputTensors{};
    }
    return alloc_tensors(args);
}

template <typename... CIs>
static inline TaskOutputTensors alloc_tensors(const CIs &...cis) {
    static_assert(sizeof...(cis) > 0, "alloc_tensors requires at least one TensorCreateInfo");
    static_assert(
        (std::is_same_v<std::decay_t<CIs>, TensorCreateInfo> && ...),
        "alloc_tensors only accepts TensorCreateInfo arguments"
    );
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    CoreTaskArgs args;
    (args.add_output(cis), ...);
    if (args.has_error) {
        rt->ops->report_fatal(
            rt, PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
            args.error_msg ? args.error_msg : "alloc_tensors failed to construct output-only Arg"
        );
        return TaskOutputTensors{};
    }
    return alloc_tensors(args);
}

static inline TaskOutputTensors rt_submit_task(const MixedKernels &mixed_kernels, const CoreTaskArgs &args) {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    return rt->ops->submit_task(rt, mixed_kernels, args);
}

/**
 * Convenience wrapper: submit an AIC-only task.
 */
static inline TaskOutputTensors rt_submit_aic_task(int32_t kernel_id, const CoreTaskArgs &args) {
    MixedKernels mk;
    mk.aic_kernel_id = kernel_id;
    return rt_submit_task(mk, args);
}

/**
 * Convenience wrapper: submit an AIV-only task (uses AIV0 slot).
 */
static inline TaskOutputTensors rt_submit_aiv_task(int32_t kernel_id, const CoreTaskArgs &args) {
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    return rt_submit_task(mk, args);
}

/**
 * Submit a dependency-only task. Accepts the same Arg shape as rt_submit_task
 * (inputs, outputs, inouts, explicit_deps, scalars) but does not run any
 * AICore kernel. The task still participates in the dependency graph: it
 * waits on its fanin and notifies its fanout. Useful as a synchronization
 * barrier or as a placeholder producer for tests / dep-graph wiring.
 */
static inline TaskOutputTensors rt_submit_dummy_task(const CoreTaskArgs &args) {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    return rt->ops->submit_dummy_task(rt, args);
}

static inline GraphScopeResult rt_graph_begin(uint64_t graph_key, const GraphTaskArgs &args) {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt) || rt->ops->graph_begin == nullptr) {
        return GraphScopeResult{};
    }
    return rt->ops->graph_begin(rt, graph_key, args);
}

// Bind the calling thread to the recording `graph_key` opened. The handle comes
// from the GraphScopeResult that opened it, so a thread can only ever record
// into the recording it was handed.
static inline bool rt_graph_prepare(void *recording_handle, const GraphTaskArgs &args) {
    PTO2Runtime *rt = current_runtime();
    return rt->ops->graph_prepare != nullptr && rt->ops->graph_prepare(rt, recording_handle, args);
}

static inline void rt_graph_abort(void *recording_handle) {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->graph_abort != nullptr) rt->ops->graph_abort(rt, recording_handle);
}

// Finish the recording pass and publish its Definition. The calling thread
// finalizes the already-submitted outer Graph shells in rt_graph_commit.
static inline bool rt_graph_end() {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt) || rt->ops->graph_end == nullptr) {
        return true;
    }
    return rt->ops->graph_end(rt);
}

static inline void rt_graph_commit() {
    GraphAsyncRecordingState &async = rt_graph_async_recording();
    async.wait();

    PTO2Runtime *rt = current_runtime();
    if (!rt->ops->is_fatal(rt) && rt->ops->graph_commit != nullptr) rt->ops->graph_commit(rt);
}

static inline void rt_scope_begin(PTO2ScopeMode mode = PTO2ScopeMode::AUTO) {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return;
    }
    rt->pending_scope_mode = mode;
    rt->ops->scope_begin(rt);
}

static inline void rt_scope_end() {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return;
    }
    rt->ops->scope_end(rt);
}

static inline void rt_orchestration_done() {
    rt_graph_commit();
    PTO2Runtime *rt = current_runtime();
    rt->ops->orchestration_done(rt);
}

/** This-run MIX cluster (= AIC) count. Do not hardcode 24/36; MIX cohorts use this. */
static inline int32_t rt_available_cluster_count() {
    PTO2Runtime *rt = current_runtime();
    return rt->ops->available_cluster_count(rt);
}

/** This-run standalone AIV core count. AIV-only cohorts size themselves on this. */
static inline int32_t rt_available_aiv_count() {
    PTO2Runtime *rt = current_runtime();
    return rt->ops->available_aiv_count(rt);
}

static inline bool rt_is_fatal() {
    PTO2Runtime *rt = current_runtime();
    return rt->ops->is_fatal(rt);
}

#define rt_report_fatal(code, fmt, ...)                                          \
    do {                                                                         \
        PTO2Runtime *_rt = current_runtime();                                    \
        _rt->ops->report_fatal(_rt, (code), __FUNCTION__, (fmt), ##__VA_ARGS__); \
    } while (0)

// =============================================================================
// Logging Macros for Orchestration (call through ops table)
// =============================================================================

#define LOG_ERROR(fmt, ...) current_runtime()->ops->log_error(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) current_runtime()->ops->log_warn(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_TIMING(fmt, ...) current_runtime()->ops->log_timing(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) current_runtime()->ops->log_info(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) current_runtime()->ops->log_debug(__FUNCTION__, fmt, ##__VA_ARGS__)

// =============================================================================
// Cross-Layer Data Access
// =============================================================================

/**
 * Read a value from a tensor at the given multi-dimensional indices.
 *
 * Default T = uint64_t preserves old behavior (raw bits).
 * Specify T to get automatic type conversion:
 *
 *   uint64_t raw = get_tensor_data(tensor, 1, idx);       // old usage unchanged
 *   float val = get_tensor_data<float>(tensor, 1, idx);   // typed read
 *
 * This API reads the registered host view used to stage an external tensor.
 * It is valid while host orchestration is building the graph, before device
 * scheduling starts. A tensor produced by a submitted task cannot become
 * readable during graph construction, and a runtime-created output has no
 * registered host view; either use is reported as an invalid argument.
 */
template <typename T = uint64_t>
static inline T get_tensor_data(const ChipTensor &tensor, uint32_t ndims, const uint32_t indices[]) {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return from_u64<T>(0);
    }
    return from_u64<T>(rt->ops->get_tensor_data(rt, tensor, ndims, indices));
}

/**
 * Write a value to a tensor at the given multi-dimensional indices.
 *
 * Type is deduced from value argument; uint64_t by default:
 *
 *   set_tensor_data(tensor, 1, idx, raw_u64);     // old usage unchanged
 *   set_tensor_data(tensor, 1, idx, 42.0f);       // typed write (T = float)
 *
 * This API updates the registered host view used to stage an external tensor.
 * The updated value becomes part of the graph's initial device data. It is not
 * a synchronization barrier for submitted readers or writers. A tensor with a
 * submitted producer, or a runtime-created output with no registered host view,
 * is rejected as an invalid argument.
 */
template <typename T = uint64_t>
static inline void set_tensor_data(const ChipTensor &tensor, uint32_t ndims, const uint32_t indices[], T value) {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return;
    }
    rt->ops->set_tensor_data(rt, tensor, ndims, indices, to_u64(value));
}

// =============================================================================
// C++ Scope Guards and Macros
// =============================================================================

/**
 * RAII Scope Guard (calls through ops table)
 */
class PTO2ScopeGuard {
public:
    explicit PTO2ScopeGuard(
        PTO2ScopeMode mode = PTO2ScopeMode::AUTO, const char *file = __builtin_FILE(), int line = __builtin_LINE()
    ) :
        rt_(current_runtime()) {
        if (!rt_->ops->is_fatal(rt_)) {
            rt_->pending_scope_mode = mode;
            if (rt_->ops->scope_set_site) rt_->ops->scope_set_site(file, line);
            rt_->ops->scope_begin(rt_);
        }
    }
    ~PTO2ScopeGuard() {
        if (!rt_->ops->is_fatal(rt_)) {
            rt_->ops->scope_end(rt_);
        }
    }

private:
    PTO2Runtime *rt_;
};

// Define or submit a Graph Execution. On a cache miss the function executes
// normally and its sub-DAG is recorded. On a hit the function is skipped and
// one Graph task is submitted; Scheduler expands the cached topology with the
// current invocation's GraphTaskArgs.
using GraphFunction = void (*)(const GraphTaskArgs &);

template <typename Function>
static inline uint64_t rt_graph_function_id(Function function) {
    static_assert(std::is_pointer_v<Function>, "Graph function identity requires a function pointer");
    static_assert(sizeof(function) <= sizeof(uint64_t), "Graph function pointer must fit in a 64-bit identity");
    uint64_t function_id = 0;
    std::memcpy(&function_id, &function, sizeof(function));
    return function_id;
}

// `invoke` is copied into the recording job and runs on a recording thread, which
// outlives this call: the caller returns as soon as the outer shell is submitted,
// and the body runs until orchestration completion joins it. So `invoke` must own
// everything it needs by value. Capturing caller-frame storage by reference —
// including the boundary `args` — is a use-after-free; the recorded body receives
// its own boundary copy as a parameter for exactly that reason.
template <typename Invoke>
static inline GraphSubmitResult rt_submit_graph_impl(uint64_t graph_key, const GraphTaskArgs &args, Invoke invoke) {
    debug_assert(!args.has_error && "Graph boundary GraphTaskArgs construction failed");
    debug_assert(
        args.tensor_count() <= static_cast<int32_t>(GRAPH_MAX_TENSOR_ARGS) && "Graph boundary exceeds the tensor limit"
    );
    debug_assert(
        args.explicit_dep_count() == 0 && "Explicit dependencies crossing the Graph boundary are not supported"
    );
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        debug_assert(
            args.tag(i) != TensorArgType::OUTPUT &&
            "Runtime-allocated TensorCreateInfo is not supported at the Graph boundary"
        );
    }
    if (!rt_graph_args_cacheable(args)) {
        invoke(args);
        return GraphSubmitResult{};
    }
    GraphScopeResult result = rt_graph_begin(graph_key, args);
    if (result.recording) {
        GraphAsyncRecordingState &async = rt_graph_async_recording();
        void *handle = result.recording_handle;
        std::shared_ptr<GraphOwnedArgs> owned_args;
        try {
            owned_args = std::make_shared<GraphOwnedArgs>(args);
        } catch (...) {
            try {
                if (!rt_graph_prepare(handle, args)) {
                    rt_graph_abort(handle);
                    rt_graph_commit();
                    return result;
                }
                invoke(args);
                (void)rt_graph_end();
            } catch (...) {
                rt_graph_abort(handle);
                throw;
            }
            rt_graph_commit();
            return result;
        }
        auto record = [owned_args, invoke, handle]() mutable {
            try {
                if (!rt_graph_prepare(handle, owned_args->args())) {
                    rt_graph_abort(handle);
                    return;
                }
                invoke(owned_args->args());
                (void)rt_graph_end();
            } catch (...) {
                rt_graph_abort(handle);
            }
        };
        if (!async.start(std::move(record))) {
            try {
                if (!rt_graph_prepare(handle, owned_args->args())) {
                    rt_graph_abort(handle);
                    rt_graph_commit();
                    return result;
                }
                invoke(owned_args->args());
                (void)rt_graph_end();
            } catch (...) {
                rt_graph_abort(handle);
                throw;
            }
            rt_graph_commit();
        }
    } else if (result.execute_block) {
        // Un-cacheable at begin, or the Definition cache is full: ordinary path.
        if (!current_runtime()->ops->is_fatal(current_runtime())) invoke(args);
    }
    // A cache hit or an in-flight hit skips the body. Every in-flight Graph task
    // is finalized at orchestration completion.
    return result;
}

static inline GraphSubmitResult rt_submit_graph(uint64_t graph_id, GraphFunction function, const GraphTaskArgs &args) {
    debug_assert(function != nullptr && "Graph function must not be null");
    if (function == nullptr) return GraphSubmitResult{};
    return rt_submit_graph_impl(rt_graph_make_key(graph_id), args, [function](const GraphTaskArgs &record_args) {
        function(record_args);
    });
}

static inline GraphSubmitResult rt_submit_graph(GraphFunction function, const GraphTaskArgs &args) {
    return rt_submit_graph(rt_graph_function_id(function), function, args);
}

template <typename... Config>
using GraphFunctionWithConfig = void (*)(const GraphTaskArgs &, Config...);

template <typename... Config>
static inline GraphSubmitResult rt_submit_graph(
    uint64_t graph_id, GraphFunctionWithConfig<Config...> function, const GraphTaskArgs &args, Config... config
) {
    debug_assert(function != nullptr && "Graph function must not be null");
    if (function == nullptr) return GraphSubmitResult{};
    auto configs = std::make_tuple(config...);
    return rt_submit_graph_impl(
        rt_graph_make_key(graph_id, config...), args, [function, configs](const GraphTaskArgs &record_args) {
            std::apply(
                [&](auto... values) {
                    function(record_args, values...);
                },
                configs
            );
        }
    );
}

template <typename... Config>
static inline GraphSubmitResult
rt_submit_graph(GraphFunctionWithConfig<Config...> function, const GraphTaskArgs &args, Config... config) {
    return rt_submit_graph(rt_graph_function_id(function), function, args, config...);
}

#define _PTO2_CONCATENATE_IMPL(x, y) x##y
#define _PTO2_CONCATENATE(x, y) _PTO2_CONCATENATE_IMPL(x, y)

#define PTO2_SCOPE_GUARD() [[maybe_unused]] PTO2ScopeGuard _PTO2_CONCATENATE(scope_guard_, __COUNTER__)

/**
 * Scoped block macro:
 *   PTO2_SCOPE() {
 *       rt_submit_task(...);
 *   }
 */
#define PTO2_SCOPE(...) if (PTO2ScopeGuard _PTO2_CONCATENATE(scope_guard_, __COUNTER__){__VA_ARGS__}; true)

// =============================================================================
// Orchestration Config
// =============================================================================

/**
 * Configuration exported by orchestration .so via aicpu_orchestration_config().
 * The executor reads these values to set up shared memory and runtime.
 *
 * This struct is defined identically in pto_runtime2.h (with an include
 * guard) so the executor can use the same type without including this header.
 */
#ifndef PTO2_ORCHESTRATION_CONFIG_DEFINED
#define PTO2_ORCHESTRATION_CONFIG_DEFINED
struct PTO2OrchestrationConfig {
    int expected_arg_count;
};
#endif

// Convenience layer (CoreTaskArgsWithDeps<N> + matching rt_submit_*_task overloads).
// Pulled in at the bottom so the wrapper sees CoreTaskArgs, MixedKernels, and the
// rt_submit_*_task primitives defined above. Orchestration sources include
// only this single header to access both the primitive and convenience APIs.
#include "pto_arg_with_deps.h"  // NOLINT(build/include_subdir)
