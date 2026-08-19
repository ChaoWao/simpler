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

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

#include "pto_orchestration_api.h"

namespace {

// The handshakes below wait for an event the peer thread is guaranteed to
// publish, so this bound only decides how long a genuine failure takes to
// report. All of this repo's self-hosted CPU runners share one machine, so it is
// budgeted for a badly loaded host rather than for the expected microseconds.
constexpr std::chrono::seconds kHandshakeTimeout{5};

PTO2Runtime *g_bound_runtime = nullptr;

extern "C" PTO2Runtime *framework_current_runtime(void) { return g_bound_runtime; }
extern "C" void framework_bind_runtime(PTO2Runtime *rt) { g_bound_runtime = rt; }

struct FakeRuntime {
    const PTO2RuntimeOps *ops;
    PTO2ScopeMode pending_scope_mode{PTO2ScopeMode::AUTO};
    std::mutex mutex;
    std::condition_variable cv;
    bool record_entered{false};
    bool later_submit_entered{false};
    bool overlapped{false};
    int prepare_overlaps{0};
    int begin_calls{0};
    int prepare_calls{0};
    int end_calls{0};
    int commit_calls{0};
    int scope_begin_calls{0};
    int scope_end_calls{0};
    std::thread::id record_thread;
    std::thread::id prepare_thread;
    std::thread::id submit_thread;

    // What graph_prepare actually received, so the deep copy GraphOwnedArgs makes
    // can be compared against the boundary the caller passed.
    bool prepare_saw_args{false};
    int32_t recorded_tensor_count{-1};
    int32_t recorded_scalar_count{-1};
    uint64_t recorded_tensor_addr{0};
    uint64_t recorded_tensor_size{0};
    uint32_t recorded_ndims{0};
    uint64_t recorded_scalar{0};
    TensorArgType recorded_tag{};
    const void *recorded_args_object{nullptr};
    const void *recorded_tensor_storage{nullptr};
};

static_assert(offsetof(FakeRuntime, ops) == 0);
static_assert(offsetof(FakeRuntime, pending_scope_mode) == offsetof(PTO2Runtime, pending_scope_mode));

FakeRuntime *as_fake(PTO2Runtime *rt) { return reinterpret_cast<FakeRuntime *>(rt); }

bool fake_is_fatal(PTO2Runtime *) { return false; }

GraphScopeResult fake_graph_begin(PTO2Runtime *rt, uint64_t, const GraphTaskArgs &) {
    FakeRuntime &fake = *as_fake(rt);
    std::lock_guard<std::mutex> lock(fake.mutex);
    fake.begin_calls++;
    GraphScopeResult result;
    result.execute_block = false;
    result.recording = fake.begin_calls == 1;
    if (!result.recording) {
        fake.later_submit_entered = true;
        fake.cv.notify_all();
    }
    return result;
}

bool fake_graph_prepare(PTO2Runtime *rt, const GraphTaskArgs &args) {
    FakeRuntime &fake = *as_fake(rt);
    std::unique_lock<std::mutex> lock(fake.mutex);
    fake.prepare_calls++;
    fake.prepare_thread = std::this_thread::get_id();
    fake.prepare_saw_args = true;
    fake.recorded_tensor_count = args.tensor_count();
    fake.recorded_scalar_count = args.scalar_count();
    fake.recorded_args_object = &args;
    if (args.tensor_count() > 0) {
        const ChipTensor &tensor = args.tensor(0).ref();
        fake.recorded_tensor_addr = tensor.buffer.addr;
        fake.recorded_tensor_size = tensor.buffer.size;
        fake.recorded_ndims = tensor.ndims;
        fake.recorded_tag = args.tag(0);
        fake.recorded_tensor_storage = &tensor;
    }
    if (args.scalar_count() > 0) {
        fake.recorded_scalar = args.scalar(0);
    }
    // The hash-keyed outer shell is enough for the caller to continue. Hold
    // prepare until the later same-hash submission proves that start() did not
    // retain the old worker-ready handshake.
    const bool saw_later_submit = fake.cv.wait_for(lock, kHandshakeTimeout, [&] {
        return fake.later_submit_entered;
    });
    if (saw_later_submit) fake.prepare_overlaps++;
    return true;
}

void fake_graph_abort(PTO2Runtime *) {}

bool fake_graph_end(PTO2Runtime *rt) {
    FakeRuntime &fake = *as_fake(rt);
    fake.end_calls++;
    fake.submit_thread = std::this_thread::get_id();
    return true;
}

void fake_graph_commit(PTO2Runtime *rt) { as_fake(rt)->commit_calls++; }

void fake_scope_begin(PTO2Runtime *rt) { as_fake(rt)->scope_begin_calls++; }

void fake_scope_end(PTO2Runtime *rt) { as_fake(rt)->scope_end_calls++; }

const PTO2RuntimeOps kFakeOps = {
    .scope_begin = fake_scope_begin,
    .scope_end = fake_scope_end,
    .is_fatal = fake_is_fatal,
    .graph_begin = fake_graph_begin,
    .graph_prepare = fake_graph_prepare,
    .graph_abort = fake_graph_abort,
    .graph_end = fake_graph_end,
    .graph_commit = fake_graph_commit,
};

}  // namespace

TEST(HbgGraphAsyncSubmit, WorkerRecordsWhileMainSubmitsLaterGraphs) {
    FakeRuntime fake{};
    fake.ops = &kFakeOps;
    framework_bind_runtime(reinterpret_cast<PTO2Runtime *>(&fake));

    uint32_t storage[4]{};
    uint32_t shape[] = {4};
    ChipTensor boundary = make_tensor_external(storage, shape, 1);
    GraphTaskArgs args;
    args.add_input(boundary);

    int body_calls = 0;
    const std::thread::id caller = std::this_thread::get_id();
    GraphSubmitResult first;
    {
        PTO2ScopeGuard scope;
        first = rt_submit_graph_impl(0x1715, args, [&](const GraphTaskArgs &) {
            // Graph bodies use the ordinary task wrappers, which commit any
            // preceding outer Graph before submitting. This must not make the
            // recorder wait for its own in-flight job.
            rt_graph_commit();
            std::unique_lock<std::mutex> lock(fake.mutex);
            body_calls++;
            fake.record_entered = true;
            fake.record_thread = std::this_thread::get_id();
            fake.cv.notify_all();
            const bool saw_later_submit = fake.cv.wait_for(lock, kHandshakeTimeout, [&] {
                return fake.later_submit_entered;
            });
            fake.overlapped |= saw_later_submit;
        });
    }
    GraphSubmitResult second;
    {
        PTO2ScopeGuard scope;
        second = rt_submit_graph_impl(0x1715, args, [&](const GraphTaskArgs &) {
            ADD_FAILURE() << "a later submission for an in-flight Graph must not execute the body";
        });
    }
    rt_graph_commit();

    const std::thread::id first_record_thread = fake.record_thread;
    {
        std::lock_guard<std::mutex> lock(fake.mutex);
        fake.begin_calls = 0;
        fake.record_entered = false;
        fake.later_submit_entered = false;
        fake.overlapped = false;
    }
    {
        PTO2ScopeGuard scope;
        first = rt_submit_graph_impl(0x1715, args, [&](const GraphTaskArgs &) {
            rt_graph_commit();
            std::unique_lock<std::mutex> lock(fake.mutex);
            body_calls++;
            fake.record_entered = true;
            fake.record_thread = std::this_thread::get_id();
            fake.cv.notify_all();
            const bool saw_later_submit = fake.cv.wait_for(lock, kHandshakeTimeout, [&] {
                return fake.later_submit_entered;
            });
            fake.overlapped |= saw_later_submit;
        });
    }
    {
        PTO2ScopeGuard scope;
        second = rt_submit_graph_impl(0x1715, args, [&](const GraphTaskArgs &) {
            ADD_FAILURE() << "a later submission for an in-flight Graph must not execute the body";
        });
    }
    rt_graph_commit();

    framework_bind_runtime(nullptr);
    EXPECT_TRUE(first.recording);
    EXPECT_FALSE(second.recording);
    EXPECT_EQ(fake.begin_calls, 2);
    EXPECT_EQ(body_calls, 2);
    EXPECT_EQ(fake.prepare_calls, 2);
    EXPECT_EQ(fake.prepare_overlaps, 2) << "main-thread Graph submission must not wait for worker graph_prepare";
    EXPECT_EQ(fake.end_calls, 2);
    EXPECT_EQ(fake.commit_calls, 4);
    EXPECT_EQ(fake.scope_begin_calls, 4);
    EXPECT_EQ(fake.scope_end_calls, 4);
    EXPECT_TRUE(fake.overlapped);
    EXPECT_NE(fake.record_thread, caller);
    EXPECT_EQ(fake.record_thread, first_record_thread) << "steady-state recording must reuse the worker thread";
    EXPECT_EQ(fake.prepare_thread, fake.record_thread);
    EXPECT_EQ(fake.submit_thread, fake.record_thread);
}

// The recorded body runs on the worker after the caller's frame is free to end,
// so it must read a boundary the worker owns. GraphOwnedArgs is that copy; this
// pins both halves of its contract — the values survive, and the storage they
// live in is not the caller's.
TEST(HbgGraphAsyncSubmit, RecordingReadsAnOwnedCopyOfTheBoundary) {
    FakeRuntime fake{};
    fake.ops = &kFakeOps;
    framework_bind_runtime(reinterpret_cast<PTO2Runtime *>(&fake));

    uint32_t storage[8]{};
    uint32_t shape[] = {8};
    ChipTensor boundary = make_tensor_external(storage, shape, 1);
    constexpr uint64_t kScalar = 0x5eed1715ULL;
    GraphTaskArgs args;
    args.add_input(boundary);
    args.add_scalar(kScalar);

    const void *caller_args_object = &args;
    const void *caller_tensor_storage = &args.tensor(0).ref();

    // This case is about the copy, not the overlap: release the gate that
    // fake_graph_prepare waits on so it returns immediately instead of sitting
    // out the handshake timeout no later submission is going to satisfy.
    fake.later_submit_entered = true;

    {
        PTO2ScopeGuard scope;
        (void)rt_submit_graph_impl(0x1719, args, [](const GraphTaskArgs &) {});
    }
    rt_graph_commit();
    framework_bind_runtime(nullptr);

    ASSERT_TRUE(fake.prepare_saw_args);
    EXPECT_EQ(fake.recorded_tensor_count, args.tensor_count());
    EXPECT_EQ(fake.recorded_scalar_count, args.scalar_count());
    EXPECT_EQ(fake.recorded_tensor_addr, boundary.buffer.addr);
    EXPECT_EQ(fake.recorded_tensor_size, boundary.buffer.size);
    EXPECT_EQ(fake.recorded_ndims, boundary.ndims);
    EXPECT_EQ(fake.recorded_tag, TensorArgType::INPUT);
    EXPECT_EQ(fake.recorded_scalar, kScalar);

    EXPECT_NE(fake.recorded_args_object, caller_args_object) << "the worker must not read the caller's GraphTaskArgs";
    EXPECT_NE(fake.recorded_tensor_storage, caller_tensor_storage)
        << "the worker must not read ChipTensor storage the caller owns";
}
