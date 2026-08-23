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

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "call_config.h"
#include "pipeline_slot_pool.h"
#include "runtime_c_api.h"
#include "task_args.h"
#include "types.h"

#define private public
#include "chip_worker.h"
#undef private
#include "chip_run_lane.h"

#include <gtest/gtest.h>

namespace {

std::unordered_map<void *, uint32_t> g_slots;
std::array<std::atomic<bool>, 2> g_complete{};
std::array<int, 2> g_prepare_rc{};
std::array<int, 2> g_launch_rc{};
std::array<int, 2> g_poll_rc{};
std::array<int, 2> g_wait_rc{};
std::array<int, 2> g_finalize_rc{};
std::array<size_t, 2> g_prepare_count{};
std::array<bool, 2> g_reject_first_prepare{};
size_t g_poll_count{0};
// When nonzero, the poll stub completes the run after this many polls. Only
// UnboundedWaitBlocksInsteadOfPolling sets it, so that a regression there fails
// on the poll count instead of looping forever.
size_t g_poll_completes_after{0};
bool g_supports_successor{true};
std::mutex g_wait_mu;
std::condition_variable g_wait_cv;
bool g_wait_entered{false};
bool g_release_wait{false};
std::thread::id g_prepare_thread;
std::thread::id g_finalize_thread;
std::vector<std::string> g_events;

void record_event(std::string event) {
    std::lock_guard<std::mutex> lock(g_wait_mu);
    g_events.push_back(std::move(event));
}

uint32_t slot_of(void *runtime) { return g_slots.at(runtime); }

int prepare_run(
    void *, void *runtime, int32_t, const void *, const CallConfig *, const NativeRunDescriptor *descriptor
) {
    EXPECT_EQ(slot_of(runtime), descriptor->pipeline_slot);
    g_complete[descriptor->pipeline_slot] = false;
    g_prepare_thread = std::this_thread::get_id();
    record_event("prepare" + std::to_string(descriptor->pipeline_slot));
    ++g_prepare_count[descriptor->pipeline_slot];
    if (g_reject_first_prepare[descriptor->pipeline_slot] && g_prepare_count[descriptor->pipeline_slot] == 1) {
        return PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE;
    }
    return g_prepare_rc[descriptor->pipeline_slot];
}

int launch_run(void *, void *runtime) {
    const uint32_t slot = slot_of(runtime);
    record_event("launch" + std::to_string(slot));
    return g_launch_rc[slot];
}

int poll_run(void *, void *runtime) {
    ++g_poll_count;
    const uint32_t slot = slot_of(runtime);
    if (g_poll_rc[slot] != 0) return g_poll_rc[slot];
    if (g_poll_completes_after != 0 && g_poll_count >= g_poll_completes_after) g_complete[slot] = true;
    return g_complete[slot] ? SIMPLER_NATIVE_RUN_POLL_COMPLETE : SIMPLER_NATIVE_RUN_POLL_NOT_READY;
}

int wait_run(void *, void *runtime) {
    const uint32_t slot = slot_of(runtime);
    record_event("wait" + std::to_string(slot));
    {
        std::unique_lock<std::mutex> lk(g_wait_mu);
        g_wait_entered = true;
        g_wait_cv.notify_all();
        g_wait_cv.wait(lk, [slot] {
            return g_release_wait || g_complete[slot].load();
        });
    }
    g_complete[slot] = true;
    return g_wait_rc[slot];
}

int finalize_run(void *, void *runtime) {
    const uint32_t slot = slot_of(runtime);
    g_finalize_thread = std::this_thread::get_id();
    record_event("finalize" + std::to_string(slot));
    return g_finalize_rc[slot];
}

int supports_successor(void *) { return g_supports_successor ? 1 : 0; }

void prime_worker(ChipWorker &worker) {
    g_slots.clear();
    for (auto &complete : g_complete)
        complete.store(false);
    g_prepare_rc = {};
    g_launch_rc = {};
    g_poll_rc = {};
    g_wait_rc = {};
    g_finalize_rc = {};
    g_prepare_count = {};
    g_reject_first_prepare = {};
    g_poll_count = 0;
    g_poll_completes_after = 0;
    g_supports_successor = true;
    g_prepare_thread = {};
    g_finalize_thread = {};
    {
        std::lock_guard<std::mutex> lk(g_wait_mu);
        g_wait_entered = false;
        g_release_wait = false;
    }
    {
        std::lock_guard<std::mutex> lock(g_wait_mu);
        g_events.clear();
    }
    worker.initialized_ = true;
    worker.pipeline_contract_ = {PTO_PIPELINE_CONTRACT_ABI_VERSION, 0, 2, {}};
    worker.runtime_bufs_.emplace_back(64, alignof(std::max_align_t));
    worker.runtime_bufs_.emplace_back(64, alignof(std::max_align_t));
    for (uint32_t slot = 0; slot < worker.runtime_bufs_.size(); ++slot) {
        g_slots.emplace(worker.runtime_bufs_[slot].data(), slot);
    }
    worker.prepare_run_fn_ = prepare_run;
    worker.launch_run_fn_ = launch_run;
    worker.poll_run_fn_ = poll_run;
    worker.wait_run_fn_ = wait_run;
    worker.finalize_run_fn_ = finalize_run;
    worker.supports_concurrent_native_prepare_fn_ = supports_successor;
}

ChipRun submit(ChipRunLane &lane, uint64_t run_id, uint32_t slot, bool activate = true) {
    ChipStorageTaskArgs args{};
    return lane.submit(1, args, CallConfig{}, PipelineSlotLease{slot, 0, run_id}, run_id, run_id, nullptr, 0, activate);
}

void complete(uint32_t slot) {
    // Publish the predicate while holding the same mutex used by wait_run.
    // Atomic storage alone does not prevent a notification from landing
    // between the predicate check and the condition-variable wait.
    {
        std::lock_guard<std::mutex> lock(g_wait_mu);
        g_complete[slot] = true;
    }
    g_wait_cv.notify_all();
}

bool await_done(ChipRun &run) { return run.wait_until(std::chrono::steady_clock::now() + std::chrono::seconds(5)); }

bool await_terminal(ChipRun &run) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!run.done() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    return run.done();
}

std::vector<std::string> events_without_wait() {
    std::lock_guard<std::mutex> lock(g_wait_mu);
    std::vector<std::string> events;
    std::copy_if(g_events.begin(), g_events.end(), std::back_inserter(events), [](const std::string &event) {
        return event.rfind("wait", 0) != 0;
    });
    return events;
}

bool has_event(const std::string &event) {
    std::lock_guard<std::mutex> lock(g_wait_mu);
    return std::find(g_events.begin(), g_events.end(), event) != g_events.end();
}

}  // namespace

TEST(ChipRunLaneTest, OwnsFifoPreparationAndLaunch) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);

    ChipRun first = submit(lane, 101, 0);
    ChipRun second = submit(lane, 102, 1, false);
    EXPECT_EQ(second.preparation_disposition(), ChipRunPreparationDisposition::NATIVE_PREPARED);
    EXPECT_EQ(events_without_wait(), (std::vector<std::string>{"prepare0", "launch0", "prepare1"}));

    second.activate();
    EXPECT_FALSE(second.done());
    complete(0);
    EXPECT_TRUE(await_done(first));
    EXPECT_FALSE(second.done());
    EXPECT_TRUE(second.launched());
    EXPECT_EQ(
        events_without_wait(), (std::vector<std::string>{"prepare0", "launch0", "prepare1", "finalize0", "launch1"})
    );

    complete(1);
    EXPECT_TRUE(await_done(second));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, ExplicitPrepareDoesNotLaunchBeforeActivation) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);

    ChipRun run = submit(lane, 101, 0, false);
    EXPECT_EQ(run.preparation_disposition(), ChipRunPreparationDisposition::VALIDATED_ONLY);
    EXPECT_FALSE(run.launched());

    run.prepare();
    EXPECT_EQ(run.preparation_disposition(), ChipRunPreparationDisposition::NATIVE_PREPARED);
    EXPECT_FALSE(run.launched());
    EXPECT_EQ(events_without_wait(), (std::vector<std::string>{"prepare0"}));

    run.activate();
    EXPECT_TRUE(run.launched());
    EXPECT_EQ(events_without_wait(), (std::vector<std::string>{"prepare0", "launch0"}));
    complete(0);
    EXPECT_TRUE(await_done(run));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, ValidationOnlySuccessorPreparesAfterPromotion) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipStorageTaskArgs args{};
    CallConfig diagnostic;
    diagnostic.enable_pmu = true;
    diagnostic.output_prefix[0] = 'x';

    ChipRun first = submit(lane, 101, 0);
    ChipRun second = lane.submit(1, args, diagnostic, PipelineSlotLease{1, 0, 102}, 102, 102, nullptr, 0, false);
    EXPECT_EQ(second.preparation_disposition(), ChipRunPreparationDisposition::VALIDATED_ONLY);
    EXPECT_EQ(events_without_wait(), (std::vector<std::string>{"prepare0", "launch0"}));

    second.activate();
    complete(0);
    EXPECT_TRUE(await_done(first));
    EXPECT_FALSE(second.done());
    EXPECT_EQ(
        events_without_wait(), (std::vector<std::string>{"prepare0", "launch0", "finalize0", "prepare1", "launch1"})
    );
    complete(1);
    EXPECT_TRUE(await_done(second));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, IncompatibleSuccessorRetriesAfterPredecessorFence) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_reject_first_prepare[1] = true;

    ChipRun first = submit(lane, 101, 0);
    ChipRun second = submit(lane, 102, 1, false);
    EXPECT_EQ(second.preparation_disposition(), ChipRunPreparationDisposition::VALIDATED_ONLY);
    EXPECT_EQ(events_without_wait(), (std::vector<std::string>{"prepare0", "launch0", "prepare1"}));

    second.activate();
    complete(0);
    EXPECT_TRUE(await_done(first));
    EXPECT_TRUE(second.launched());
    EXPECT_EQ(
        events_without_wait(),
        (std::vector<std::string>{"prepare0", "launch0", "prepare1", "finalize0", "prepare1", "launch1"})
    );

    complete(1);
    EXPECT_TRUE(await_done(second));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, EarlierActiveDispatchOrdersBeforeAnAlreadyStagedSuccessor) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);

    ChipRun successor = submit(lane, 102, 1, false);
    EXPECT_EQ(successor.preparation_disposition(), ChipRunPreparationDisposition::VALIDATED_ONLY);
    ChipRun active = submit(lane, 101, 0, true);
    EXPECT_TRUE(active.launched());
    EXPECT_EQ(successor.preparation_disposition(), ChipRunPreparationDisposition::NATIVE_PREPARED);
    EXPECT_EQ(events_without_wait(), (std::vector<std::string>{"prepare0", "launch0", "prepare1"}));

    successor.activate();
    complete(0);
    EXPECT_TRUE(await_done(active));
    EXPECT_TRUE(successor.launched());
    complete(1);
    EXPECT_TRUE(await_done(successor));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, AcceptsCurrentLeaseGenerationAndRejectsOlderOne) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipRun first = submit(lane, 101, 0);
    complete(0);
    ASSERT_TRUE(await_done(first));

    ChipStorageTaskArgs args{};
    ChipRun repeated = lane.submit(1, args, CallConfig{}, PipelineSlotLease{0, 0, 101}, 102, 102, nullptr, 0, true);
    complete(0);
    ASSERT_TRUE(await_done(repeated));
    EXPECT_THROW(
        lane.submit(1, args, CallConfig{}, PipelineSlotLease{0, 0, 100}, 103, 103, nullptr, 0, true), std::runtime_error
    );
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, ActivateStopsAtTheLaunchFenceBeforePolling) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipRun run = submit(lane, 101, 0, false);

    run.activate();
    EXPECT_TRUE(run.launched());
    EXPECT_EQ(g_poll_count, 0);
    complete(0);
    EXPECT_TRUE(await_done(run));
    EXPECT_LE(g_poll_count, 1u);
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, ResidentLifecycleLaunchesSuccessorWithoutCallerPolling) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);

    ChipRun first = submit(lane, 101, 0);
    ChipRun second = submit(lane, 102, 1, false);
    second.activate();
    complete(0);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    while (!second.launched() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    EXPECT_TRUE(second.launched());

    complete(1);
    EXPECT_TRUE(second.wait_until(std::chrono::steady_clock::now() + std::chrono::seconds(1)));
    EXPECT_TRUE(first.done());
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, DiagnosticFinalizeStaysOnCallingThread) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipStorageTaskArgs args{};
    CallConfig config{};
    config.enable_dep_gen = 1;
    config.output_prefix[0] = 'x';

    const std::thread::id caller = std::this_thread::get_id();
    ChipRun run = lane.submit(1, args, config, PipelineSlotLease{0, 0, 101}, 101, 101, nullptr, 0, true);
    std::thread completer([] {
        {
            std::unique_lock<std::mutex> lk(g_wait_mu);
            (void)g_wait_cv.wait_for(lk, std::chrono::seconds(1), [] {
                return g_wait_entered;
            });
        }
        complete(0);
    });

    EXPECT_TRUE(run.wait_until(ChipRunLane::Deadline::max()));
    completer.join();
    EXPECT_EQ(g_prepare_thread, caller);
    EXPECT_EQ(g_finalize_thread, caller);
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, DirectGenerationDoesNotConsumeTheFirstPipelineLease) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipStorageTaskArgs args{};

    ChipRun direct = lane.submit(1, args, CallConfig{});
    complete(0);
    ASSERT_TRUE(await_done(direct));

    ChipRun leased = submit(lane, 1, 0);
    EXPECT_TRUE(leased.launched());
    complete(0);
    EXPECT_TRUE(await_done(leased));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, DirectCapacityTwoPreparesSuccessorAndBackpressuresThird) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipStorageTaskArgs args{};

    ChipRun first = lane.submit(1, args, CallConfig{});
    ChipRun second = lane.submit(1, args, CallConfig{});
    EXPECT_TRUE(first.launched());
    EXPECT_FALSE(second.launched());
    EXPECT_EQ(second.preparation_disposition(), ChipRunPreparationDisposition::NATIVE_PREPARED);
    EXPECT_EQ(events_without_wait(), (std::vector<std::string>{"prepare0", "launch0", "prepare1"}));

    {
        std::lock_guard<std::mutex> lk(g_wait_mu);
        g_release_wait = false;
    }
    std::optional<ChipRun> third;
    std::exception_ptr submit_error;
    std::thread submitter([&] {
        try {
            third.emplace(lane.submit(1, args, CallConfig{}));
        } catch (...) {
            submit_error = std::current_exception();
        }
    });

    bool entered_wait = false;
    {
        std::unique_lock<std::mutex> lk(g_wait_mu);
        entered_wait = g_wait_cv.wait_for(lk, std::chrono::seconds(1), [] {
            return g_wait_entered;
        });
    }
    EXPECT_TRUE(entered_wait);
    EXPECT_EQ(g_prepare_count[0], 1u) << "third submit prepared before capacity released";
    EXPECT_EQ(g_prepare_count[1], 1u);
    complete(0);
    submitter.join();

    ASSERT_EQ(submit_error, nullptr);
    ASSERT_TRUE(third.has_value());
    EXPECT_TRUE(second.launched());
    EXPECT_FALSE(third->launched());
    EXPECT_EQ(third->preparation_disposition(), ChipRunPreparationDisposition::NATIVE_PREPARED);
    EXPECT_EQ(g_prepare_count[0], 2u);

    complete(1);
    EXPECT_TRUE(await_done(second));
    EXPECT_TRUE(third->launched());
    complete(0);
    EXPECT_TRUE(await_done(*third));
    EXPECT_TRUE(first.done());
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, DirectIncompatibleSuccessorRetriesAfterPredecessorFence) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipStorageTaskArgs args{};

    ChipRun first = lane.submit(1, args, CallConfig{});
    g_reject_first_prepare[1] = true;
    ChipRun second = lane.submit(1, args, CallConfig{});
    EXPECT_EQ(second.preparation_disposition(), ChipRunPreparationDisposition::VALIDATED_ONLY);
    EXPECT_EQ(g_prepare_count[1], 1u);

    complete(0);
    EXPECT_FALSE(second.done());
    EXPECT_TRUE(await_done(first));
    EXPECT_TRUE(second.launched());
    EXPECT_EQ(g_prepare_count[1], 2u);
    complete(1);
    EXPECT_TRUE(await_done(second));
    lane.close();
    worker.finalize();
}

// SIMPLER_ERROR_FLOW_CONTROL_DEADLOCK (3) negated — the status a device latches for
// a flow-control deadlock. Spelled out rather than included: the SIMPLER_ERROR_*
// macros live in each runtime's private runtime_status.h, which this
// worker-level test does not compile against.
constexpr int kLatchedFlowControlDeadlock = -3;

// A latched deadlock is terminal. PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE once held
// this same value, so the lane read the deadlock as "prepare at depth one instead"
// and retried it — reporting a successful run over a device that had wedged.
TEST(ChipRunLaneTest, LatchedFlowControlDeadlockIsTerminalRatherThanRetried) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipStorageTaskArgs args{};

    // A live predecessor is what puts the successor's prepare on the overlapping
    // path, where the incompatible-prepare retry lives.
    ChipRun first = lane.submit(1, args, CallConfig{});
    g_prepare_rc[1] = kLatchedFlowControlDeadlock;
    ChipRun second = lane.submit(1, args, CallConfig{});

    EXPECT_TRUE(second.done());
    EXPECT_THROW(second.wait_until(ChipRunLane::Deadline::max()), std::runtime_error);
    EXPECT_EQ(g_prepare_count[1], 1u) << "latched deadlock was retried as an incompatible prepare";
    EXPECT_FALSE(lane.poisoned());

    complete(0);
    EXPECT_TRUE(await_done(first));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, DirectRuntimeWithoutConcurrentPrepareRetainsDepthOne) {
    ChipWorker worker;
    prime_worker(worker);
    g_supports_successor = false;
    ChipRunLane lane(worker);
    ChipStorageTaskArgs args{};

    ChipRun first = lane.submit(1, args, CallConfig{});
    complete(0);
    ChipRun second = lane.submit(1, args, CallConfig{});

    EXPECT_TRUE(first.done());
    EXPECT_TRUE(second.launched());
    EXPECT_EQ(g_prepare_count[0], 2u);
    EXPECT_EQ(g_prepare_count[1], 0u);
    complete(0);
    EXPECT_TRUE(await_done(second));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, PrepareFailureIsTerminalWithoutPoisoningTheLane) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_prepare_rc[0] = -5;

    ChipRun failed = submit(lane, 101, 0);
    EXPECT_TRUE(failed.done());
    EXPECT_THROW(failed.wait_until(ChipRunLane::Deadline::max()), std::runtime_error);
    EXPECT_FALSE(lane.poisoned());

    ChipRun successor = submit(lane, 102, 1);
    EXPECT_TRUE(successor.launched());
    complete(1);
    EXPECT_TRUE(await_done(successor));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, ReclaimedLaunchFailureDoesNotPoisonTheLane) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_launch_rc[0] = -6;

    ChipRun failed = submit(lane, 101, 0);
    EXPECT_TRUE(failed.done());
    EXPECT_THROW(failed.wait_until(ChipRunLane::Deadline::max()), std::runtime_error);
    EXPECT_FALSE(lane.poisoned());

    ChipRun successor = submit(lane, 102, 1);
    EXPECT_TRUE(successor.launched());
    complete(1);
    EXPECT_TRUE(await_done(successor));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, ExpiredWaitLeavesTheRunLive) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipRun run = submit(lane, 101, 0);

    EXPECT_FALSE(run.wait_until(ChipRunLane::Clock::now()));
    EXPECT_FALSE(run.done());
    complete(0);
    EXPECT_TRUE(run.wait_until(ChipRunLane::Deadline::max()));
    lane.close();
    worker.finalize();
}

// An unbounded wait has no deadline to end its loop, so it must reach the
// device's blocking wait rather than re-polling until the run completes. The
// poll bound is what makes that observable: a re-polling implementation calls
// poll_run without limit, and no assertion on the run's result can see it.
TEST(ChipRunLaneTest, UnboundedWaitBlocksInsteadOfPolling) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipRun run = submit(lane, 101, 0);

    // Complete only after the resident waiter has entered the blocking native
    // wait. This keeps the assertion deterministic and proves that no polling
    // loop is required to make progress.
    ASSERT_FALSE(g_complete[0]);
    g_poll_count = 0;
    std::thread completer([] {
        {
            std::unique_lock<std::mutex> lk(g_wait_mu);
            (void)g_wait_cv.wait_for(lk, std::chrono::seconds(1), [] {
                return g_wait_entered;
            });
        }
        complete(0);
    });
    EXPECT_TRUE(run.wait_until(ChipRunLane::Deadline::max()));
    completer.join();

    EXPECT_TRUE(has_event("wait0")) << "unbounded wait never reached the device's blocking wait";
    EXPECT_LE(g_poll_count, 1u) << "unbounded wait polled " << g_poll_count << " times instead of blocking";
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, FinalizeFailurePoisonsAdmissionAndCloseReportsIt) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipRun first = submit(lane, 101, 0);
    g_finalize_rc[0] = -7;
    complete(0);
    EXPECT_TRUE(await_terminal(first));
    EXPECT_THROW(first.wait_until(ChipRunLane::Clock::time_point::max()), std::runtime_error);
    EXPECT_TRUE(lane.poisoned());

    ChipStorageTaskArgs args{};
    EXPECT_THROW(
        lane.submit(1, args, CallConfig{}, PipelineSlotLease{1, 0, 102}, 102, 102, nullptr, 0, true), std::runtime_error
    );
    EXPECT_THROW(lane.close(), std::runtime_error);
    worker.finalize();
}

TEST(ChipRunLaneTest, WaitFailureReportsTheTerminalNativeError) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_wait_rc[0] = 507015;
    ChipRun run = submit(lane, 101, 0);
    {
        std::unique_lock<std::mutex> lk(g_wait_mu);
        ASSERT_TRUE(g_wait_cv.wait_for(lk, std::chrono::seconds(1), [] {
            return g_wait_entered;
        }));
    }
    complete(0);

    EXPECT_TRUE(await_terminal(run));
    EXPECT_THROW(run.wait_until(ChipRunLane::Deadline::max()), std::runtime_error);
    try {
        run.wait_until(ChipRunLane::Deadline::max());
        FAIL() << "expected the terminal native error";
    } catch (const std::runtime_error &e) {
        EXPECT_NE(std::string(e.what()).find("finalize_native_run failed with code 507015"), std::string::npos);
    }
    EXPECT_TRUE(lane.poisoned());
    EXPECT_THROW(lane.close(), std::runtime_error);
    worker.finalize();
}

TEST(ChipRunLaneTest, CloseDrainsAndRejectsNewSubmissions) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipRun first = submit(lane, 101, 0);
    complete(0);
    lane.close();
    EXPECT_TRUE(first.done());

    ChipStorageTaskArgs args{};
    EXPECT_THROW(
        lane.submit(1, args, CallConfig{}, PipelineSlotLease{1, 0, 102}, 102, 102, nullptr, 0, true), std::runtime_error
    );
    worker.finalize();
}
