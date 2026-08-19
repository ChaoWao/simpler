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

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <thread>
#include <vector>

#include "mpi_direct_transport.h"
#include "remote_wire.h"
#include "ring.h"
#include "worker_manager.h"

namespace {

constexpr int32_t WORKER_ID = 7;
constexpr int32_t MPI_RANK = 3;
constexpr uint64_t SESSION_ID = 0x1234;
constexpr size_t MAX_FRAME_BYTES = 40 + remote_l3::MAX_FRAME_PAYLOAD_BYTES;

std::vector<uint8_t> make_frame(remote_l3::FrameType type, uint64_t sequence, std::vector<uint8_t> payload = {}) {
    remote_l3::FrameHeader header;
    header.frame_type = type;
    header.session_id = SESSION_ID;
    header.worker_id = WORKER_ID;
    header.sequence = sequence;
    return remote_l3::encode_frame(header, payload);
}

std::vector<uint8_t> make_hello() {
    remote_l3::HelloPayload hello;
    hello.session_id = SESSION_ID;
    hello.worker_id = WORKER_ID;
    hello.protocol_version = remote_l3::PROTOCOL_VERSION;
    hello.comm_profile = "sim";
    hello.ready_state = remote_l3::ReadyState::READY;
    return make_frame(remote_l3::FrameType::HELLO, 0, remote_l3::encode_hello(hello));
}

std::shared_ptr<MpiDirectTransportHub> ready_hub() {
    auto hub = std::make_shared<MpiDirectTransportHub>(2 * MAX_FRAME_BYTES);
    hub->register_route(WORKER_ID, MPI_RANK, SESSION_ID, "sim");
    hub->deliver(MPI_RANK, MpiDirectTag::LIFECYCLE, make_hello());
    return hub;
}

TEST(MpiDirectTransportHub, RoutesRawSlr3FramesWithoutEnvelope) {
    auto hub = ready_hub();
    MpiDirectTransport transport(hub, WORKER_ID, 1.0, 1.0);
    transport.expect_hello_ready();

    auto task = make_frame(remote_l3::FrameType::TASK, 11, {1, 2, 3});
    transport.submit_frame(task);
    auto outbound = hub->poll_outbound(0.0);
    ASSERT_TRUE(outbound.has_value());
    EXPECT_EQ(outbound->target_rank, MPI_RANK);
    EXPECT_EQ(outbound->tag, MpiDirectTag::COMMAND_REQUEST);
    EXPECT_EQ(outbound->frame, task);
    hub->complete_outbound(outbound->ticket);

    auto completion = make_frame(remote_l3::FrameType::COMPLETION, 11, {4, 5});
    hub->deliver(MPI_RANK, MpiDirectTag::COMMAND_REPLY, completion);
    EXPECT_EQ(transport.wait_for_reply(remote_l3::FrameType::COMPLETION, 11), completion);
    EXPECT_EQ(hub->pending_frame_bytes(), 0U);
}

TEST(MpiDirectTransport, ProgressApiIsNonBlocking) {
    auto hub = ready_hub();
    MpiDirectTransport transport(hub, WORKER_ID, 1.0, 1.0);
    transport.expect_hello_ready();

    auto task = make_frame(remote_l3::FrameType::TASK, 21, {7});
    transport.submit_progress_frame(task);
    std::vector<uint8_t> reply;
    EXPECT_FALSE(transport.poll_progress_reply(remote_l3::FrameType::COMPLETION, 21, reply));

    auto completion = make_frame(remote_l3::FrameType::COMPLETION, 21, {8});
    hub->deliver(MPI_RANK, MpiDirectTag::COMMAND_REPLY, completion);
    ASSERT_TRUE(transport.poll_progress_reply(remote_l3::FrameType::COMPLETION, 21, reply));
    EXPECT_EQ(reply, completion);
}

TEST(MpiDirectTransportHub, RejectsSourceRankMismatchAsTerminal) {
    auto hub = std::make_shared<MpiDirectTransportHub>(MAX_FRAME_BYTES);
    hub->register_route(WORKER_ID, MPI_RANK, SESSION_ID, "sim");
    EXPECT_THROW(hub->deliver(MPI_RANK + 1, MpiDirectTag::LIFECYCLE, make_hello()), std::runtime_error);
    EXPECT_TRUE(hub->terminal());
    EXPECT_NE(hub->terminal_error().find("unknown MPI rank"), std::string::npos);
}

TEST(MpiDirectTransportHub, RejectsTagFrameTypeMismatchAsTerminal) {
    auto hub = std::make_shared<MpiDirectTransportHub>(MAX_FRAME_BYTES);
    hub->register_route(WORKER_ID, MPI_RANK, SESSION_ID, "sim");
    EXPECT_THROW(hub->deliver(MPI_RANK, MpiDirectTag::COMMAND_REPLY, make_hello()), std::runtime_error);
    EXPECT_TRUE(hub->terminal());
}

TEST(MpiDirectTransportHub, PendingByteCreditIncludesMpiInFlightSend) {
    auto hub = std::make_shared<MpiDirectTransportHub>(MAX_FRAME_BYTES);
    hub->register_route(WORKER_ID, MPI_RANK, SESSION_ID, "sim");
    hub->deliver(MPI_RANK, MpiDirectTag::LIFECYCLE, make_hello());
    MpiDirectTransport transport(hub, WORKER_ID, 1.0, 2.0);
    transport.expect_hello_ready();
    std::vector<uint8_t> payload(remote_l3::MAX_FRAME_PAYLOAD_BYTES, 0x5a);
    auto first = make_frame(remote_l3::FrameType::TASK, 1, payload);
    auto second = make_frame(remote_l3::FrameType::TASK, 2, payload);

    transport.submit_frame(first);
    auto outbound = hub->poll_outbound(0.0);
    ASSERT_TRUE(outbound.has_value());

    std::atomic<bool> submit_started{false};
    std::atomic<bool> submit_finished{false};
    std::exception_ptr submit_error;
    std::thread submitter([&] {
        submit_started.store(true, std::memory_order_release);
        try {
            transport.submit_frame(second);
            submit_finished.store(true, std::memory_order_release);
        } catch (...) {
            submit_error = std::current_exception();
        }
    });
    while (!submit_started.load(std::memory_order_acquire))
        std::this_thread::yield();
    EXPECT_FALSE(submit_finished.load(std::memory_order_acquire));
    hub->complete_outbound(outbound->ticket);
    submitter.join();
    ASSERT_EQ(submit_error, nullptr);
    EXPECT_TRUE(submit_finished.load(std::memory_order_acquire));
}

TEST(MpiDirectTransportHub, HealthFrameIsOutOfBand) {
    auto hub = ready_hub();
    MpiDirectTransport transport(hub, WORKER_ID, 1.0, 1.0);
    transport.expect_hello_ready();
    hub->deliver(MPI_RANK, MpiDirectTag::HEALTH, make_frame(remote_l3::FrameType::HEALTH, 9));

    auto completion = make_frame(remote_l3::FrameType::COMPLETION, 10);
    hub->deliver(MPI_RANK, MpiDirectTag::COMMAND_REPLY, completion);
    EXPECT_EQ(transport.wait_for_reply(remote_l3::FrameType::COMPLETION, 10), completion);
}

TEST(MpiDirectTransport, WorkerManagerStopSendsLifecycleShutdownAfterProgressStop) {
    auto hub = ready_hub();
    auto transport = std::make_unique<MpiDirectTransport>(hub, WORKER_ID, 1.0, 1.0);
    transport->expect_hello_ready();

    Ring ring;
    ring.init(/*heap_bytes=*/0);
    WorkerManager manager;
    manager.add_next_level_endpoint(
        std::make_unique<RemoteL3Endpoint>(WORKER_ID, SESSION_ID, "mpi-direct", std::move(transport))
    );
    manager.start(&ring, [](WorkerCompletion) {}, [](WorkerDispatch) {});

    // Worker::close() stops the Scheduler first. Its final progress pass sees
    // the stopped lane before WorkerManager::stop() owns child shutdown.
    manager.stop_workers();
    manager.progress();
    manager.stop();

    auto outbound = hub->poll_outbound(0.0);
    ASSERT_TRUE(outbound.has_value());
    EXPECT_EQ(outbound->target_rank, MPI_RANK);
    EXPECT_EQ(outbound->tag, MpiDirectTag::LIFECYCLE);
    auto frame = remote_l3::decode_frame(outbound->frame);
    EXPECT_EQ(frame.header.frame_type, remote_l3::FrameType::SHUTDOWN);
    EXPECT_EQ(frame.header.session_id, SESSION_ID);
    EXPECT_EQ(frame.header.worker_id, WORKER_ID);
    hub->complete_outbound(outbound->ticket);
    ring.shutdown();
}

}  // namespace
