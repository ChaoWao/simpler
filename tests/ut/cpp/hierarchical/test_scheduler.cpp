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
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "call_config.h"
#include "orchestrator.h"
#include "ring.h"
#include "scheduler.h"
#include "scope.h"
#include "tensormap.h"
#include "types.h"
#include "worker_manager.h"
#include "task_args.h"

// ---------------------------------------------------------------------------
// MockMailboxWorker: in-process stand-in for the forked Python child loop.
//
// The production dispatch path writes (callable digest, config, args_blob) into a
// MAILBOX_SIZE-byte shared region and spin-polls TASK_DONE; the real child
// (`_chip_process_loop` in python/simpler/worker.py) decodes the mailbox and
// dispatches to a `ChipWorker`. For unit testing the Scheduler / WorkerManager
// state machine in isolation, we replace the forked child with a thread inside
// the test process that mimics the same handshake but blocks until the
// test thread releases it via `complete()`.
//
// API parity with the previous MockWorker:
//   - dispatched[i].callable_hash0 / .tensor_key — recorded on TASK_READY
//   - is_running                            — atomic flag the test polls
//   - wait_running()                        — spin-wait until is_running flips
//   - complete()                            — release the parked dispatch so
//                                             the loop writes TASK_DONE
// ---------------------------------------------------------------------------

struct MockMailboxWorker {
    struct Record {
        uint8_t callable_hash0;
        uint64_t tensor_key;  // first tensor's `data` field (unique per submit in tests)
    };

    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    std::vector<Record> dispatched;
    std::mutex dispatched_mu;

    std::mutex run_mu;
    std::condition_variable run_cv;
    std::atomic<bool> should_complete{false};
    std::atomic<bool> drain_mode{false};
    int32_t next_error_code{0};
    std::string next_error_msg;
    std::atomic<bool> is_running{false};
    std::atomic<bool> stop_flag{false};
    std::thread loop_thread;

    void start() {
        // SharedMemory zero-fills, but std::array does not — explicitly
        // store IDLE (=0) to mirror production parity and keep the polling
        // loop's first read deterministic.
        write_state(MailboxState::IDLE);
        loop_thread = std::thread(&MockMailboxWorker::loop, this);
    }

    ~MockMailboxWorker() {
        // Defensive teardown — if a test fails before completing every
        // dispatch, set stop_flag and wake the parked loop so the thread
        // joins instead of leaking. The loop's TASK_READY branch always
        // publishes TASK_DONE before checking stop_flag, so any in-flight
        // LocalMailboxEndpoint::run completes its spin-poll cleanly.
        stop_flag.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(run_mu);
            should_complete.store(true, std::memory_order_release);
            run_cv.notify_one();
        }
        if (loop_thread.joinable()) loop_thread.join();
    }

    void *mailbox_ptr() { return mailbox.data(); }

    void complete() {
        std::lock_guard<std::mutex> lk(run_mu);
        next_error_code = 0;
        next_error_msg.clear();
        should_complete.store(true, std::memory_order_release);
        run_cv.notify_one();
    }

    void complete_with_error(std::string msg) {
        std::lock_guard<std::mutex> lk(run_mu);
        next_error_code = 1;
        next_error_msg = std::move(msg);
        should_complete.store(true, std::memory_order_release);
        run_cv.notify_one();
    }

    // Persistent teardown mode: every dispatch — including one arriving after
    // this call — completes itself, so Scheduler::stop() can always join.
    void drain() {
        drain_mode.store(true, std::memory_order_release);
        std::lock_guard<std::mutex> lk(run_mu);
        should_complete.store(true, std::memory_order_release);
        run_cv.notify_one();
    }

    // The child publishes acceptance into the sticky word, not the state.
    void write_task_accepted() {
        auto *ptr = reinterpret_cast<int32_t *>(static_cast<char *>(mailbox_ptr()) + MAILBOX_OFF_ACCEPTED);
        int32_t v = MAILBOX_TASK_ACCEPTED;
        __atomic_store(ptr, &v, __ATOMIC_RELEASE);
    }

    void wait_running(int timeout_ms = 500) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (!is_running.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    int dispatched_count() {
        std::lock_guard<std::mutex> lk(dispatched_mu);
        return static_cast<int>(dispatched.size());
    }

private:
    // Mirror the acquire/release semantics in
    // worker_manager.cpp::read_mailbox_state / write_mailbox_state. Plain
    // memcpy on the mailbox state would let the parent observe the state
    // flip before the preceding error-field stores are visible.
    MailboxState read_state() const {
        const auto *ptr = reinterpret_cast<const volatile int32_t *>(mailbox.data() + MAILBOX_OFF_STATE);
        int32_t v = __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
        return static_cast<MailboxState>(v);
    }

    void write_state(MailboxState s) {
        auto *ptr = reinterpret_cast<volatile int32_t *>(mailbox.data() + MAILBOX_OFF_STATE);
        __atomic_store_n(ptr, static_cast<int32_t>(s), __ATOMIC_RELEASE);
    }

    void loop() {
        while (true) {
            if (stop_flag.load(std::memory_order_acquire)) return;
            MailboxState s = read_state();
            if (s == MailboxState::TASK_READY) {
                uint8_t callable_hash0 = static_cast<uint8_t>(mailbox[MAILBOX_OFF_TASK_CALLABLE_HASH]);
                int32_t t_count = 0;
                std::memcpy(&t_count, mailbox.data() + MAILBOX_OFF_TASK_ARGS_BLOB, sizeof(int32_t));
                uint64_t tensor_key = 0;
                if (t_count > 0) {
                    Tensor first{};
                    std::memcpy(
                        &first, mailbox.data() + MAILBOX_OFF_TASK_ARGS_BLOB + TASK_ARGS_BLOB_HEADER_SIZE, sizeof(Tensor)
                    );
                    tensor_key = first.buffer.addr;
                }
                {
                    std::lock_guard<std::mutex> lk(dispatched_mu);
                    dispatched.push_back({callable_hash0, tensor_key});
                }
                is_running.store(true, std::memory_order_release);

                {
                    std::unique_lock<std::mutex> lk(run_mu);
                    run_cv.wait(lk, [this] {
                        return should_complete.load(std::memory_order_acquire) ||
                               drain_mode.load(std::memory_order_acquire);
                    });
                    should_complete.store(false, std::memory_order_relaxed);
                }
                int32_t error_code = 0;
                std::string error_msg;
                {
                    std::lock_guard<std::mutex> lk(run_mu);
                    error_code = next_error_code;
                    error_msg = std::move(next_error_msg);
                    next_error_code = 0;
                    next_error_msg.clear();
                }
                is_running.store(false, std::memory_order_release);

                std::memcpy(mailbox.data() + MAILBOX_OFF_ERROR, &error_code, sizeof(int32_t));
                std::memset(mailbox.data() + MAILBOX_OFF_ERROR_MSG, 0, MAILBOX_ERROR_MSG_SIZE);
                if (!error_msg.empty()) {
                    size_t n = std::min(error_msg.size(), MAILBOX_ERROR_MSG_SIZE - 1);
                    std::memcpy(mailbox.data() + MAILBOX_OFF_ERROR_MSG, error_msg.data(), n);
                }
                write_state(MailboxState::TASK_DONE);
            } else if (s == MailboxState::CONTROL_REQUEST) {
                // Acknowledge the control request so a future test using
                // WorkerThread::control_* doesn't hang on the spin-poll.
                // No memory operation is simulated — result stays zero.
                int32_t zero_err = 0;
                std::memcpy(mailbox.data() + MAILBOX_OFF_ERROR, &zero_err, sizeof(int32_t));
                std::memset(mailbox.data() + MAILBOX_OFF_ERROR_MSG, 0, MAILBOX_ERROR_MSG_SIZE);
                uint64_t zero_result = 0;
                std::memcpy(mailbox.data() + CTRL_OFF_RESULT, &zero_result, sizeof(uint64_t));
                write_state(MailboxState::CONTROL_DONE);
            } else if (s == MailboxState::SHUTDOWN) {
                return;
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
    }
};

class FakeEndpoint final : public WorkerEndpoint {
public:
    explicit FakeEndpoint(int32_t worker_id, std::atomic<int> *prepare_count = nullptr) :
        prepare_count_(prepare_count) {
        caps_.kind = WorkerEndpointKind::REMOTE_L3;
        caps_.worker_id = worker_id;
        caps_.remote = true;
        caps_.transport = "test-remote";
    }

    const WorkerEndpointCaps &caps() const override { return caps_; }

    WorkerCompletion run(Ring *ring, const WorkerDispatch &dispatch) override {
        (void)ring;
        WorkerCompletion completion;
        completion.task_slot = dispatch.task_slot;
        completion.group_index = dispatch.group_index;
        completion.outcome = EndpointOutcome::SUCCESS;
        return completion;
    }

    void control_prepare(const uint8_t *) override {
        if (prepare_count_ != nullptr) prepare_count_->fetch_add(1, std::memory_order_relaxed);
    }

private:
    WorkerEndpointCaps caps_;
    std::atomic<int> *prepare_count_{nullptr};
};

// ---------------------------------------------------------------------------
// Helper: build a TaskArgs whose only tensor has the given (data, tag).
// ---------------------------------------------------------------------------

static TaskArgs single_tensor_args(uint64_t data_ptr, TensorArgType tag) {
    TaskArgs a;
    Tensor t{};
    t.buffer.addr = data_ptr;
    t.ndims = 1;
    t.shapes[0] = 1;
    t.dtype = DataType::UINT8;
    a.add_tensor(t, tag);
    return a;
}

static CallableIdentity C(uint8_t seed) {
    CallableIdentity c;
    c.digest.fill(seed);
    return c;
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

// The claim is what makes "pop a READY slot" and "dispatch it" one decision.
// A cancelling run moves its unstarted slots out of READY and consumes them;
// anything the scheduler had already popped must lose the race rather than
// overwrite that state with RUNNING.
TEST(ClaimForDispatch, OnlyAReadySlotCanBeClaimed) {
    TaskSlotState s;

    s.state.store(TaskState::READY, std::memory_order_release);
    EXPECT_TRUE(claim_for_dispatch(s));
    EXPECT_EQ(s.state.load(std::memory_order_acquire), TaskState::RUNNING);

    // Already claimed by this scheduler: a second claim must not re-dispatch.
    EXPECT_FALSE(claim_for_dispatch(s));
    EXPECT_EQ(s.state.load(std::memory_order_acquire), TaskState::RUNNING);

    // Cancelled between the pop and here — the cancelling path owns it now.
    // BUILDING is in the list for a different reason: a slot whose submit has
    // not published it has no final args or fanin count to dispatch on.
    for (TaskState taken :
         {TaskState::FAILED, TaskState::COMPLETED, TaskState::CONSUMED, TaskState::PENDING, TaskState::BUILDING}) {
        s.state.store(taken, std::memory_order_release);
        EXPECT_FALSE(claim_for_dispatch(s)) << "claimed a slot in state " << static_cast<int>(taken);
        EXPECT_EQ(s.state.load(std::memory_order_acquire), taken) << "claim overwrote a state it did not own";
    }
}

// Every failing path — a device completion poisoning its consumers, a run
// cancellation, and a submit that wired onto a producer that had already
// failed — moves a task to FAILED through this one exchange, so exactly one of
// them writes the message and runs the propagation.
TEST(ClaimTaskFailure, ReportsThePriorStateAndIsWonOnce) {
    TaskSlotState s;

    for (TaskState claimable : {TaskState::PENDING, TaskState::READY, TaskState::BUILDING}) {
        s.state.store(claimable, std::memory_order_release);
        s.failure_message.clear();

        std::optional<TaskState> won = claim_task_failure(s, "first");
        ASSERT_TRUE(won.has_value()) << "refused a claimable slot in state " << static_cast<int>(claimable);
        EXPECT_EQ(*won, claimable) << "claim did not report the state it took the slot from";
        EXPECT_EQ(s.state.load(std::memory_order_acquire), TaskState::FAILED);
        EXPECT_EQ(s.failure_message, "first");
        EXPECT_EQ(s.failure_propagation_pending.load(std::memory_order_acquire), claimable == TaskState::BUILDING)
            << "only a mid-wiring claim may advertise propagation takeover debt";

        // A second path reaching the same slot must not overwrite the reason
        // the first one recorded.
        EXPECT_FALSE(claim_task_failure(s, "second").has_value());
        EXPECT_EQ(s.failure_message, "first");

        s.failure_propagation_pending.store(false, std::memory_order_release);
    }
}

TEST(MarkGroupMembersSkipped, RepairsBothBookkeepingVectorsAsOneTransaction) {
    TaskSlotState s;
    s.is_group_ = true;
    s.task_args_list.resize(2);
    s.group_member_states.assign(2, GroupMemberState::NOT_DISPATCHED);
    s.group_member_outcomes.clear();

    mark_group_members_skipped(s, "cancelled");

    ASSERT_EQ(s.group_member_states.size(), 2u);
    ASSERT_EQ(s.group_member_outcomes.size(), 2u);
    EXPECT_EQ(s.group_member_states[0], GroupMemberState::SKIPPED);
    EXPECT_EQ(s.group_member_states[1], GroupMemberState::SKIPPED);
    EXPECT_EQ(s.group_member_outcomes[0], EndpointOutcome::SKIPPED);
    EXPECT_EQ(s.group_member_outcomes[1], EndpointOutcome::SKIPPED);
    EXPECT_EQ(s.group_terminal_count.load(std::memory_order_acquire), 2);
}

// Readiness is one decision over two fields that different threads own.
// Judging the count outside the transition lets a producer pass a comparison
// against a count submit has not published, then act on it after submit has —
// dispatching a task whose remaining producers are still running.
TEST(TryMarkReady, JudgesThePublishedCountNotTheOneItArrivedWith) {
    TaskSlotState s;
    s.state.store(TaskState::BUILDING, std::memory_order_release);

    // A producer completes while the slot is still building. The count it can
    // see is zero, which any release count passes — but nothing is readiable
    // yet, and the release is not lost either.
    s.fanin_released.store(1, std::memory_order_release);
    EXPECT_FALSE(try_mark_ready(s));

    // Submit publishes two live producers alongside the transition.
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        s.fanin_count.store(2, std::memory_order_release);
        s.state.store(TaskState::PENDING, std::memory_order_release);
    }
    EXPECT_FALSE(try_mark_ready(s)) << "one of two producers released and the task was marked ready";

    s.fanin_released.store(2, std::memory_order_release);
    EXPECT_TRUE(try_mark_ready(s));
    EXPECT_EQ(s.state.load(std::memory_order_acquire), TaskState::READY);

    // Exactly one caller owns the enqueue.
    EXPECT_FALSE(try_mark_ready(s));
}

// The publication is not a moment another thread can slip through. A producer
// that has already completed is held outside it for its whole duration, so
// there is no instant at which the slot is PENDING with a count only half the
// deciders have seen — which is the state that dispatches a task whose
// remaining producers are still running.
TEST(TryMarkReady, NoProducerTransitionsTheSlotWhileThePublicationHoldsIt) {
    TaskSlotState s;
    s.state.store(TaskState::BUILDING, std::memory_order_release);
    s.fanin_count.store(0, std::memory_order_release);

    std::atomic<bool> producer_marked_ready{false};
    std::unique_lock<std::mutex> publishing(s.fanout_mu);

    std::thread producer([&] {
        s.fanin_released.fetch_add(1, std::memory_order_acq_rel);
        producer_marked_ready.store(try_mark_ready(s), std::memory_order_release);
    });

    // The producer's release has landed and its decision is now in flight.
    while (s.fanin_released.load(std::memory_order_acquire) == 0)
        std::this_thread::yield();

    // Publish two live producers alongside the transition, then stay in the
    // critical section: a decider that judged the count before entering it
    // would take PENDING to READY right here.
    s.fanin_count.store(2, std::memory_order_release);
    s.state.store(TaskState::PENDING, std::memory_order_release);
    for (int i = 0; i < 1000; ++i)
        std::this_thread::yield();
    EXPECT_EQ(s.state.load(std::memory_order_acquire), TaskState::PENDING)
        << "a producer transitioned the slot from inside the publication";

    publishing.unlock();
    producer.join();
    EXPECT_FALSE(producer_marked_ready.load(std::memory_order_acquire));
    EXPECT_EQ(s.state.load(std::memory_order_acquire), TaskState::PENDING)
        << "a task was made ready with a live producer still running";
}

TEST(ClaimTaskFailure, RefusesASlotItDoesNotOwn) {
    TaskSlotState s;

    // RUNNING is the device's until its completion arrives; the rest are
    // already terminal, and resurrecting one would release its dependency
    // references a second time.
    for (TaskState owned :
         {TaskState::RUNNING, TaskState::COMPLETED, TaskState::FAILED, TaskState::CONSUMED, TaskState::FREE}) {
        s.state.store(owned, std::memory_order_release);
        s.failure_message.clear();
        EXPECT_FALSE(claim_task_failure(s, "cancelled").has_value())
            << "claimed a slot in state " << static_cast<int>(owned);
        EXPECT_EQ(s.state.load(std::memory_order_acquire), owned);
        EXPECT_TRUE(s.failure_message.empty());
    }
}

struct SchedulerFixture : public ::testing::Test {
    TensorMap tm;
    Ring allocator;
    Scope scope;
    ReadyQueue rq_sub;
    NextLevelReadyQueues rq_next_level;
    Orchestrator orch;
    MockMailboxWorker mock_worker;
    WorkerManager manager;
    Scheduler sched;
    CallConfig cfg;
    RunId run_id{INVALID_RUN_ID};

    std::vector<TaskSlot> consumed_slots;
    std::mutex consumed_mu;

    // Set by a test before the Scheduler reaches a claim; see
    // Scheduler::Config::before_claim_cb.
    std::function<void(TaskSlot)> before_claim_hook;

    TaskSlotState &S(TaskSlot id) { return *allocator.slot_state(id); }

    void SetUp() override {
        allocator.init(/*heap_bytes=*/1ULL << 20);

        mock_worker.start();
        manager.add_next_level(mock_worker.mailbox_ptr());
        manager.start(
            &allocator,
            [this](WorkerCompletion completion) {
                sched.worker_done(std::move(completion));
            },
            [this](WorkerDispatch dispatch) {
                orch.mark_task_accepted(dispatch.task_slot);
            }
        );
        rq_next_level.reset(manager.next_level_worker_ids());
        orch.init(&tm, &allocator, &scope, &rq_sub, &rq_next_level, &manager, [this] {
            sched.notify_ready();
        });
        run_id = orch.begin_run();

        Scheduler::Config c;
        c.ring = &allocator;
        c.ready_sub_queue = &rq_sub;
        c.ready_next_level_queues = &rq_next_level;
        c.manager = &manager;
        c.enqueue_ready_cb = [this](TaskSlot slot) {
            orch.enqueue_ready(slot);
        };
        // Same gate Worker::start installs: an active run that is also the
        // EXECUTING FIFO head and still owns its pipeline lease. Testing
        // against the weaker active_run_id() would let a slot dispatch here
        // that production refuses.
        c.active_run_cb = [this] {
            return orch.dispatchable_run_id();
        };
        c.on_consumed_cb = [this](TaskSlot s) {
            orch.on_consumed(s);
            std::lock_guard<std::mutex> lk(consumed_mu);
            consumed_slots.push_back(s);
        };
        c.on_task_failed_cb = [this](TaskSlot s, const std::string &message) {
            orch.report_task_error(s, message);
        };
        c.before_claim_cb = [this](TaskSlot slot) {
            if (before_claim_hook) before_claim_hook(slot);
        };
        sched.start(c);
    }

    void TearDown() override {
        mock_worker.drain();
        sched.stop();
        manager.stop();
        allocator.shutdown();
    }

    void wait_consumed(TaskSlot slot, int timeout_ms = 500) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lk(consumed_mu);
                for (TaskSlot s : consumed_slots)
                    if (s == slot) return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        FAIL() << "Timed out waiting for slot " << slot << " to be consumed";
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(WorkerManagerTest, StartRejectsDuplicateNextLevelWorkerId) {
    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    WorkerManager manager;

    manager.add_next_level(mailbox.data());
    manager.add_next_level_endpoint(std::make_unique<FakeEndpoint>(0));

    bool threw = false;
    try {
        manager.start(&allocator, [](WorkerCompletion) {}, {});
    } catch (const std::runtime_error &e) {
        threw = true;
        EXPECT_NE(std::string(e.what()).find("duplicate NEXT_LEVEL worker_id 0"), std::string::npos);
    }

    manager.stop();
    allocator.shutdown();
    EXPECT_TRUE(threw);
}

TEST(WorkerManagerTest, LocalMailboxPublishesAcceptanceBeforeCompletion) {
    MockMailboxWorker child;
    child.start();

    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    AllocResult ar = allocator.alloc(/*heap_bytes=*/0, /*depth=*/0);
    ASSERT_NE(ar.slot, INVALID_SLOT);
    TaskSlotState *slot = allocator.slot_state(ar.slot);
    ASSERT_NE(slot, nullptr);
    slot->reset();
    slot->callable.digest[0] = 0x42;
    slot->pipeline_lease = PipelineSlotLease{1, 0, 7};

    LocalMailboxEndpoint endpoint(/*worker_id=*/0, child.mailbox_ptr());
    std::promise<WorkerCompletion> result;
    auto done = result.get_future();
    std::atomic<bool> accepted{false};
    std::thread caller([&] {
        result.set_value(endpoint.run_with_accept(&allocator, WorkerDispatch{ar.slot, 0}, [&] {
            accepted.store(true, std::memory_order_release);
        }));
    });

    child.wait_running();
    EXPECT_TRUE(child.is_running.load(std::memory_order_acquire));
    PipelineSlotLease wire_lease{};
    std::memcpy(
        &wire_lease, static_cast<char *>(child.mailbox_ptr()) + MAILBOX_OFF_PIPELINE_LEASE, sizeof(PipelineSlotLease)
    );
    EXPECT_EQ(wire_lease.slot_id, 1u);
    EXPECT_EQ(wire_lease.generation, 7u);
    child.write_task_accepted();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!accepted.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {}
    EXPECT_TRUE(accepted.load(std::memory_order_acquire));
    EXPECT_EQ(done.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);

    // Non-fatal from here on: a fatal assertion would return with `caller`
    // joinable, and ~std::thread would terminate the whole test binary.
    child.complete();
    EXPECT_EQ(done.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    if (done.valid() && done.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        EXPECT_EQ(done.get().outcome, EndpointOutcome::SUCCESS);
    }
    caller.join();
    allocator.shutdown();
}

// The ACK must not be carried by anything TASK_DONE can overwrite. The child
// publishes acceptance only into the sticky word — never into the state — and
// then completes immediately, so an endpoint that looks for acceptance in the
// state word observes none at all.
//
// This does not force the parent to skip a poll between the two writes: the
// parent is already spinning by then and nothing here can stop it. What it
// pins is the property that makes that interleaving harmless — acceptance is
// readable after TASK_DONE, so losing a poll cannot lose the ACK.
TEST(WorkerManagerTest, AcceptanceIsReadableAfterTaskDone) {
    MockMailboxWorker child;
    child.start();

    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    AllocResult ar = allocator.alloc(/*heap_bytes=*/0, /*depth=*/0);
    ASSERT_NE(ar.slot, INVALID_SLOT);
    TaskSlotState *slot = allocator.slot_state(ar.slot);
    ASSERT_NE(slot, nullptr);
    slot->reset();
    slot->callable.digest[0] = 0x42;

    LocalMailboxEndpoint endpoint(/*worker_id=*/0, child.mailbox_ptr());
    std::promise<WorkerCompletion> result;
    auto done = result.get_future();
    std::atomic<bool> accepted{false};

    std::thread caller([&] {
        result.set_value(endpoint.run_with_accept(&allocator, WorkerDispatch{ar.slot, 0}, [&] {
            accepted.store(true, std::memory_order_release);
        }));
    });

    child.wait_running();
    EXPECT_TRUE(child.is_running.load(std::memory_order_acquire));
    // Back to back, with no parent poll in between.
    child.write_task_accepted();
    child.complete();

    // Non-fatal: a fatal assertion would return with `caller` joinable, and
    // ~std::thread would terminate the whole test binary.
    EXPECT_EQ(done.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    if (done.valid() && done.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        EXPECT_EQ(done.get().outcome, EndpointOutcome::SUCCESS);
    }
    EXPECT_TRUE(accepted.load(std::memory_order_acquire))
        << "the endpoint lost the launch ACK to a task that completed first";
    caller.join();
    allocator.shutdown();
}

// A child that dies without publishing CONTROL_DONE must be reported, not
// waited on forever. The mailbox stays at CONTROL_REQUEST exactly as it would
// if the real `_chip_process_loop` had crashed mid-command. Run in a worker
// thread with a bounded join so a regression fails the test instead of
// hanging the suite.
TEST(WorkerManagerTest, ControlCommandFailsWhenChildExitsBeforeCompletion) {
    void *mailbox =
        mmap(nullptr, MAILBOX_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, /*fd=*/-1, /*offset=*/0);
    ASSERT_NE(mailbox, MAP_FAILED);
    std::memset(mailbox, 0, MAILBOX_SIZE);

    pid_t child = fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        _exit(3);
    }

    LocalMailboxEndpoint endpoint(/*worker_id=*/0, mailbox, static_cast<int>(child));

    std::promise<std::string> result;
    auto done = result.get_future();
    std::thread caller([&] {
        try {
            endpoint.control_malloc(64);
            result.set_value("");
        } catch (const std::runtime_error &e) {
            result.set_value(e.what());
        }
    });

    ASSERT_EQ(done.wait_for(std::chrono::seconds(10)), std::future_status::ready)
        << "control_malloc did not observe the dead child; it is spinning on CONTROL_DONE";
    std::string message = done.get();
    caller.join();

    EXPECT_NE(message.find("child process pid=" + std::to_string(child)), std::string::npos) << message;
    EXPECT_NE(message.find("exit_status=3"), std::string::npos) << message;

    // The endpoint is poisoned once the child is gone: a later command reports
    // rather than resuming the spin.
    EXPECT_THROW(endpoint.control_free(0), std::runtime_error);

    ASSERT_EQ(munmap(mailbox, MAILBOX_SIZE), 0);
}

TEST(WorkerManagerTest, ControlPrepareUsesStableNextLevelWorkerId) {
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    WorkerManager manager;
    std::atomic<int> worker7_prepares{0};
    std::atomic<int> worker3_prepares{0};

    manager.add_next_level_endpoint(std::make_unique<FakeEndpoint>(7, &worker7_prepares));
    manager.add_next_level_endpoint(std::make_unique<FakeEndpoint>(3, &worker3_prepares));
    manager.start(&allocator, [](WorkerCompletion) {}, {});

    std::array<uint8_t, CALLABLE_HASH_DIGEST_SIZE> digest{};
    manager.control_prepare(3, digest.data());

    manager.stop();
    allocator.shutdown();
    EXPECT_EQ(worker7_prepares.load(std::memory_order_relaxed), 0);
    EXPECT_EQ(worker3_prepares.load(std::memory_order_relaxed), 1);
}

// The losing side of the dispatch claim, driven by the real cancellation path
// rather than a simulated state write. `before_claim_cb` is the only point that
// can observe the window: everything else is either before the queue pop or
// after the launch.
TEST_F(SchedulerFixture, ACancellationThatWinsTheClaimStopsTheDispatch) {
    std::atomic<int> hook_calls{0};
    before_claim_hook = [this, &hook_calls](TaskSlot) {
        if (hook_calls.fetch_add(1) != 0) return;
        orch.fail_run_submission(run_id, std::make_exception_ptr(std::runtime_error("cancelled mid-dispatch")));
    };

    auto task = orch.submit_next_level(C(60), single_tensor_args(0x6000, TensorArgType::OUTPUT), cfg, 0);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (hook_calls.load() == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_GT(hook_calls.load(), 0) << "the Scheduler never reached a dispatch claim";

    // Give the Scheduler its whole loop iteration; a lost claim must leave the
    // slot alone rather than continue into the launch.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_NE(S(task.task_slot).state.load(std::memory_order_acquire), TaskState::RUNNING)
        << "the dispatch overwrote a slot the cancellation already owned";
    {
        std::lock_guard<std::mutex> lk(mock_worker.dispatched_mu);
        EXPECT_TRUE(mock_worker.dispatched.empty()) << "a cancelled task was still handed to a worker";
    }
}

TEST_F(SchedulerFixture, IndependentTaskDispatchedAndConsumed) {
    auto args_a = single_tensor_args(0xCAFE, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level(C(42), args_a, cfg, 0);
    TaskSlot slot = res.task_slot;

    mock_worker.wait_running();
    ASSERT_GE(mock_worker.dispatched_count(), 1);
    EXPECT_EQ(mock_worker.dispatched[0].tensor_key, 0xCAFEu);
    EXPECT_EQ(mock_worker.dispatched[0].callable_hash0, 42u);

    mock_worker.complete();
    wait_consumed(slot);
}

TEST_F(SchedulerFixture, DependentTaskDispatchedAfterProducerCompletes) {
    auto args_a = single_tensor_args(0xBEEF, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(C(10), args_a, cfg, 0);

    auto args_b = single_tensor_args(0xBEEF, TensorArgType::INPUT);
    auto b = orch.submit_next_level(C(11), args_b, cfg, 0);
    EXPECT_EQ(S(b.task_slot).state.load(), TaskState::PENDING);

    mock_worker.wait_running();
    EXPECT_EQ(mock_worker.dispatched[0].callable_hash0, 10u);
    mock_worker.complete();  // A done

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    while (mock_worker.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_GE(mock_worker.dispatched_count(), 2);
    EXPECT_EQ(mock_worker.dispatched[1].callable_hash0, 11u);

    mock_worker.complete();  // B done
    wait_consumed(b.task_slot);
    (void)a;
}

// Issue #1024: composed child kernels can carry far more tensor args than a
// top-level entry (repro: 76 tensors + 2 scalars = 3064-byte blob). The
// mailbox must hold any blob the runtime itself accepts, i.e. up to
// CHIP_MAX_TENSOR_ARGS / CHIP_MAX_SCALAR_ARGS.
TEST_F(SchedulerFixture, ComposedKernelArgsBlobFitsMailbox) {
    constexpr size_t max_blob = TASK_ARGS_BLOB_HEADER_SIZE +
                                static_cast<size_t>(CHIP_MAX_TENSOR_ARGS) * sizeof(Tensor) +
                                static_cast<size_t>(CHIP_MAX_SCALAR_ARGS) * sizeof(uint64_t);
    EXPECT_GE(MAILBOX_ARGS_CAPACITY, max_blob);

    TaskArgs args;
    for (int i = 0; i < 76; ++i) {
        Tensor t{};
        t.buffer.addr = 0x1000u + static_cast<uint64_t>(i) * 0x100u;
        t.ndims = 1;
        t.shapes[0] = 1;
        t.dtype = DataType::UINT8;
        args.add_tensor(t, TensorArgType::OUTPUT);
    }
    args.add_scalar(1);
    args.add_scalar(2);

    auto res = orch.submit_next_level(C(76), args, cfg, 0);

    mock_worker.wait_running();
    ASSERT_GE(mock_worker.dispatched_count(), 1)
        << "dispatch never reached the child: args blob exceeds mailbox capacity";
    EXPECT_EQ(mock_worker.dispatched[0].callable_hash0, 76u);

    mock_worker.complete();
    wait_consumed(res.task_slot);
}

TEST_F(SchedulerFixture, FailedProducerPoisonsDependentTask) {
    auto args_a = single_tensor_args(0xD00D, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(C(21), args_a, cfg, 0);

    auto args_b = single_tensor_args(0xD00D, TensorArgType::INPUT);
    auto b = orch.submit_next_level(C(22), args_b, cfg, 0);
    EXPECT_EQ(S(b.task_slot).state.load(), TaskState::PENDING);

    mock_worker.wait_running();
    ASSERT_EQ(mock_worker.dispatched_count(), 1);
    EXPECT_EQ(mock_worker.dispatched[0].callable_hash0, 21u);

    mock_worker.complete_with_error("root task boom");

    wait_consumed(a.task_slot);
    wait_consumed(b.task_slot);
    EXPECT_TRUE(orch.run_failed(run_id));
    EXPECT_EQ(mock_worker.dispatched_count(), 1) << "poisoned consumer must not dispatch";
    EXPECT_EQ(S(a.task_slot).state.load(), TaskState::CONSUMED);
    EXPECT_EQ(S(b.task_slot).state.load(), TaskState::CONSUMED);
}

// ===========================================================================
// Group task tests -- fixture with 3 MockMailboxWorkers
// ===========================================================================

struct GroupSchedulerFixture : public ::testing::Test {
    TensorMap tm;
    Ring allocator;
    Scope scope;
    ReadyQueue rq_sub;
    NextLevelReadyQueues rq_next_level;
    Orchestrator orch;
    MockMailboxWorker worker_a;
    MockMailboxWorker worker_b;
    MockMailboxWorker worker_c;
    MockMailboxWorker sub_worker_a;
    MockMailboxWorker sub_worker_b;
    WorkerManager manager;
    Scheduler sched;
    CallConfig cfg;
    RunId run_id{INVALID_RUN_ID};

    std::vector<TaskSlot> consumed_slots;
    std::mutex consumed_mu;

    TaskSlotState &S(TaskSlot id) { return *allocator.slot_state(id); }

    void SetUp() override {
        allocator.init(/*heap_bytes=*/1ULL << 20);

        worker_a.start();
        worker_b.start();
        worker_c.start();
        sub_worker_a.start();
        sub_worker_b.start();
        manager.add_next_level(worker_a.mailbox_ptr());
        manager.add_next_level(worker_b.mailbox_ptr());
        manager.add_next_level(worker_c.mailbox_ptr());
        manager.add_sub(sub_worker_a.mailbox_ptr());
        manager.add_sub(sub_worker_b.mailbox_ptr());
        manager.start(
            &allocator,
            [this](WorkerCompletion completion) {
                sched.worker_done(std::move(completion));
            },
            [this](WorkerDispatch dispatch) {
                orch.mark_task_accepted(dispatch.task_slot);
            }
        );
        rq_next_level.reset(manager.next_level_worker_ids());
        orch.init(&tm, &allocator, &scope, &rq_sub, &rq_next_level, &manager, [this] {
            sched.notify_ready();
        });
        run_id = orch.begin_run();

        Scheduler::Config c;
        c.ring = &allocator;
        c.ready_sub_queue = &rq_sub;
        c.ready_next_level_queues = &rq_next_level;
        c.manager = &manager;
        c.enqueue_ready_cb = [this](TaskSlot slot) {
            orch.enqueue_ready(slot);
        };
        // Same gate Worker::start installs. Without it the scheduler takes the
        // unpartitioned branch, which is not the one #1565's group reservation
        // and placement run through.
        c.active_run_cb = [this] {
            return orch.dispatchable_run_id();
        };
        c.on_consumed_cb = [this](TaskSlot s) {
            orch.on_consumed(s);
            std::lock_guard<std::mutex> lk(consumed_mu);
            consumed_slots.push_back(s);
        };
        c.on_task_failed_cb = [this](TaskSlot s, const std::string &message) {
            orch.report_task_error(s, message);
        };
        sched.start(c);
    }

    void TearDown() override {
        worker_a.drain();
        worker_b.drain();
        worker_c.drain();
        sched.stop();
        manager.stop();
        allocator.shutdown();
    }

    void wait_consumed(TaskSlot slot, int timeout_ms = 1000) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lk(consumed_mu);
                for (TaskSlot s : consumed_slots)
                    if (s == slot) return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        FAIL() << "Timed out waiting for slot " << slot << " to be consumed";
    }
};

TEST_F(GroupSchedulerFixture, GroupDispatchesToNWorkers) {
    TaskArgs a0 = single_tensor_args(0xA0, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xA1, TensorArgType::OUTPUT);

    auto res = orch.submit_next_level_group(C(42), {a0, a1}, cfg, {0, 1});
    TaskSlot slot = res.task_slot;

    worker_a.wait_running();
    worker_b.wait_running();

    EXPECT_EQ(worker_a.dispatched_count(), 1);
    EXPECT_EQ(worker_b.dispatched_count(), 1);

    EXPECT_EQ(worker_a.dispatched[0].tensor_key, 0xA0u);
    EXPECT_EQ(worker_b.dispatched[0].tensor_key, 0xA1u);
    (void)slot;

    worker_a.complete();
    worker_b.complete();
    wait_consumed(slot);
}

TEST_F(GroupSchedulerFixture, SubGroupUsesTheAllocationFreeGroupCommitPath) {
    TaskArgs a0 = single_tensor_args(0xA2, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xA3, TensorArgType::OUTPUT);
    auto res = orch.submit_sub_group(C(44), {a0, a1});

    sub_worker_a.wait_running();
    sub_worker_b.wait_running();
    EXPECT_EQ(sub_worker_a.dispatched_count(), 1);
    EXPECT_EQ(sub_worker_b.dispatched_count(), 1);
    EXPECT_EQ(S(res.task_slot).state.load(std::memory_order_acquire), TaskState::RUNNING);

    sub_worker_a.complete();
    sub_worker_b.complete();
    wait_consumed(res.task_slot);
}

TEST_F(GroupSchedulerFixture, GroupMapsEachMemberToItsTargetWorkerIdNotIndex) {
    // Reversed target order: member 0 -> worker id 1 (worker_b), member 1 ->
    // worker id 0 (worker_a). A map-by-registration-index bug would instead
    // send member 0 (a0) to worker_a; the reversed keys catch it.
    TaskArgs a0 = single_tensor_args(0xA0, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xA1, TensorArgType::OUTPUT);

    auto res = orch.submit_next_level_group(C(42), {a0, a1}, cfg, {1, 0});
    TaskSlot slot = res.task_slot;

    worker_a.wait_running();
    worker_b.wait_running();

    EXPECT_EQ(worker_a.dispatched_count(), 1);
    EXPECT_EQ(worker_b.dispatched_count(), 1);

    EXPECT_EQ(worker_b.dispatched[0].tensor_key, 0xA0u);
    EXPECT_EQ(worker_a.dispatched[0].tensor_key, 0xA1u);
    (void)slot;

    worker_a.complete();
    worker_b.complete();
    wait_consumed(slot);
}

TEST_F(GroupSchedulerFixture, GroupCompletesOnlyWhenAllDone) {
    TaskArgs a0 = single_tensor_args(0xB0, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xB1, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level_group(C(42), {a0, a1}, cfg, {0, 1});
    TaskSlot slot = res.task_slot;

    worker_a.wait_running();
    worker_b.wait_running();

    worker_a.complete();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(S(slot).state.load(), TaskState::RUNNING);

    worker_b.complete();
    wait_consumed(slot);
}

TEST_F(GroupSchedulerFixture, BlockedGroupReservesTargetsThatBecomeIdleOneAtATime) {
    auto running_a = orch.submit_next_level(C(70), single_tensor_args(0xF0, TensorArgType::OUTPUT), cfg, 0);
    auto running_b = orch.submit_next_level(C(71), single_tensor_args(0xF1, TensorArgType::OUTPUT), cfg, 1);
    worker_a.wait_running();
    worker_b.wait_running();
    ASSERT_TRUE(worker_a.is_running.load());
    ASSERT_TRUE(worker_b.is_running.load());

    TaskArgs group_a = single_tensor_args(0xF2, TensorArgType::OUTPUT);
    TaskArgs group_b = single_tensor_args(0xF3, TensorArgType::OUTPUT);
    auto group = orch.submit_next_level_group(C(72), {group_a, group_b}, cfg, {1, 0});
    auto single_a = orch.submit_next_level(C(73), single_tensor_args(0xF4, TensorArgType::OUTPUT), cfg, 0);
    auto single_b = orch.submit_next_level(C(74), single_tensor_args(0xF5, TensorArgType::OUTPUT), cfg, 1);
    auto unrelated = orch.submit_next_level(C(75), single_tensor_args(0xF6, TensorArgType::OUTPUT), cfg, 2);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_c.dispatched_count() == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(worker_c.dispatched_count(), 1);
    EXPECT_EQ(worker_c.dispatched[0].callable_hash0, 75u);
    worker_c.complete();

    worker_a.complete();
    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
    while (worker_a.dispatched_count() == 1 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(worker_a.dispatched_count(), 1) << "blocked group must neither dispatch partially nor yield to singles";
    EXPECT_EQ(S(group.task_slot).state.load(), TaskState::READY);

    worker_b.complete();
    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while ((worker_a.dispatched_count() < 2 || worker_b.dispatched_count() < 2) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_GE(worker_a.dispatched_count(), 2);
    ASSERT_GE(worker_b.dispatched_count(), 2);
    EXPECT_EQ(worker_a.dispatched[1].callable_hash0, 72u);
    EXPECT_EQ(worker_b.dispatched[1].callable_hash0, 72u);

    worker_a.complete();
    worker_b.complete();

    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while ((worker_a.dispatched_count() < 3 || worker_b.dispatched_count() < 3) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_GE(worker_a.dispatched_count(), 3);
    ASSERT_GE(worker_b.dispatched_count(), 3);
    EXPECT_EQ(worker_a.dispatched[2].callable_hash0, 73u);
    EXPECT_EQ(worker_b.dispatched[2].callable_hash0, 74u);
    worker_a.complete();
    worker_b.complete();

    wait_consumed(running_a.task_slot);
    wait_consumed(running_b.task_slot);
    wait_consumed(group.task_slot);
    wait_consumed(single_a.task_slot);
    wait_consumed(single_b.task_slot);
    wait_consumed(unrelated.task_slot);
}

TEST(SchedulerDispatchPassTest, ActiveRunSwitchCannotBypassSuccessorGroupReservation) {
    constexpr RunId run_a = 41;
    constexpr RunId run_b = 42;

    Ring allocator;
    ReadyQueue rq_sub;
    NextLevelReadyQueues rq_next_level;
    MockMailboxWorker worker_a;
    MockMailboxWorker worker_b;
    WorkerManager manager;
    Scheduler sched;

    allocator.init(/*heap_bytes=*/0);
    worker_a.start();
    worker_b.start();
    manager.add_next_level(worker_a.mailbox_ptr());
    manager.add_next_level(worker_b.mailbox_ptr());
    manager.start(
        &allocator,
        [&sched](WorkerCompletion completion) {
            sched.worker_done(std::move(completion));
        },
        [](WorkerDispatch) {}
    );
    rq_next_level.reset(manager.next_level_worker_ids());

    auto allocate_slot = [&](RunId run_id, uint8_t callable_seed, int32_t worker_id, bool group) {
        AllocResult allocation = allocator.alloc(/*heap_bytes=*/0, /*scope_depth=*/0);
        TaskSlotState &state = *allocator.slot_state(allocation.slot);
        state.reset();
        state.run_id = run_id;
        state.worker_type = WorkerType::NEXT_LEVEL;
        state.callable = C(callable_seed);
        state.target_worker_ids.push_back(worker_id);
        if (group) {
            state.is_group_ = true;
            state.task_args_list.push_back(single_tensor_args(callable_seed, TensorArgType::OUTPUT));
            rq_next_level.push_group(run_id, allocation.slot);
        } else {
            state.task_args = single_tensor_args(callable_seed, TensorArgType::OUTPUT);
            rq_next_level.push_single(worker_id, run_id, allocation.slot);
        }
        state.state.store(TaskState::READY, std::memory_order_release);
        return allocation.slot;
    };

    // Run A's group occupies worker 1. Run B's group and following single
    // both target worker 0, so the group head owns that worker reservation.
    allocate_slot(run_a, /*callable_seed=*/70, /*worker_id=*/1, /*group=*/true);
    allocate_slot(run_b, /*callable_seed=*/71, /*worker_id=*/0, /*group=*/true);
    allocate_slot(run_b, /*callable_seed=*/72, /*worker_id=*/0, /*group=*/false);

    std::atomic<RunId> active_run{run_a};
    Scheduler::Config config;
    config.ring = &allocator;
    config.ready_sub_queue = &rq_sub;
    config.ready_next_level_queues = &rq_next_level;
    config.manager = &manager;
    config.enqueue_ready_cb = [&](TaskSlot slot) {
        TaskSlotState &state = *allocator.slot_state(slot);
        if (state.is_group()) {
            rq_next_level.push_group(state.run_id, slot);
        } else {
            rq_next_level.push_single(state.target_worker_id(0), state.run_id, slot);
        }
    };
    config.active_run_cb = [&] {
        return active_run.load(std::memory_order_acquire);
    };
    // The run switch lands after the group phase selected A and before the
    // singles phase can select a queue partition.
    config.before_claim_cb = [&](TaskSlot slot) {
        if (allocator.slot_state(slot)->run_id == run_a) active_run.store(run_b, std::memory_order_release);
    };
    config.on_consumed_cb = [&](TaskSlot slot) {
        allocator.slot_state(slot)->state.store(TaskState::CONSUMED, std::memory_order_release);
        allocator.release(slot);
    };
    config.on_task_failed_cb = [](TaskSlot, const std::string &) {};
    sched.start(config);

    worker_a.wait_running();
    worker_b.wait_running();
    EXPECT_EQ(worker_a.dispatched_count(), 1);
    EXPECT_EQ(worker_b.dispatched_count(), 1);
    if (worker_a.dispatched_count() == 1) {
        std::lock_guard<std::mutex> lock(worker_a.dispatched_mu);
        EXPECT_EQ(worker_a.dispatched[0].callable_hash0, 71u)
            << "run B's group must dispatch before its single on the same target";
    }
    if (worker_b.dispatched_count() == 1) {
        std::lock_guard<std::mutex> lock(worker_b.dispatched_mu);
        EXPECT_EQ(worker_b.dispatched[0].callable_hash0, 70u);
    }

    if (worker_a.is_running.load(std::memory_order_acquire)) worker_a.complete();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_a.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(worker_a.dispatched_count(), 2);
    worker_a.wait_running();
    if (worker_a.dispatched_count() == 2) {
        std::lock_guard<std::mutex> lock(worker_a.dispatched_mu);
        EXPECT_EQ(worker_a.dispatched[1].callable_hash0, 72u);
    }

    if (worker_a.is_running.load(std::memory_order_acquire)) worker_a.complete();
    if (worker_b.is_running.load(std::memory_order_acquire)) worker_b.complete();
    sched.stop();
    manager.stop();
    allocator.shutdown();
}

TEST_F(GroupSchedulerFixture, ConsecutiveGroupsReserveOnlyBlockedHeadTargets) {
    SubmitResult first_group;
    SubmitResult second_group;
    SubmitResult single_a;
    SubmitResult single_c;
    {
        std::lock_guard<std::mutex> scheduler_pause(sched.loop_mutex());
        first_group = orch.submit_next_level_group(
            C(80), {single_tensor_args(0x100, TensorArgType::OUTPUT), single_tensor_args(0x101, TensorArgType::OUTPUT)},
            cfg, {0, 1}
        );
        second_group = orch.submit_next_level_group(
            C(81), {single_tensor_args(0x102, TensorArgType::OUTPUT), single_tensor_args(0x103, TensorArgType::OUTPUT)},
            cfg, {1, 2}
        );
        single_a = orch.submit_next_level(C(82), single_tensor_args(0x104, TensorArgType::OUTPUT), cfg, 0);
        single_c = orch.submit_next_level(C(83), single_tensor_args(0x105, TensorArgType::OUTPUT), cfg, 2);
    }

    worker_a.wait_running();
    worker_b.wait_running();
    ASSERT_EQ(worker_a.dispatched_count(), 1);
    ASSERT_EQ(worker_b.dispatched_count(), 1);
    EXPECT_EQ(worker_a.dispatched[0].callable_hash0, 80u);
    EXPECT_EQ(worker_b.dispatched[0].callable_hash0, 80u);
    EXPECT_EQ(worker_c.dispatched_count(), 0);

    worker_a.complete();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_a.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(worker_a.dispatched_count(), 2);
    EXPECT_EQ(worker_a.dispatched[1].callable_hash0, 82u);
    EXPECT_EQ(worker_c.dispatched_count(), 0);
    EXPECT_EQ(S(second_group.task_slot).state.load(), TaskState::READY);

    worker_a.complete();
    worker_b.complete();
    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while ((worker_b.dispatched_count() < 2 || worker_c.dispatched_count() < 1) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(worker_b.dispatched_count(), 2);
    ASSERT_EQ(worker_c.dispatched_count(), 1);
    EXPECT_EQ(worker_b.dispatched[1].callable_hash0, 81u);
    EXPECT_EQ(worker_c.dispatched[0].callable_hash0, 81u);

    worker_b.complete();
    worker_c.complete();
    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_c.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(worker_c.dispatched_count(), 2);
    EXPECT_EQ(worker_c.dispatched[1].callable_hash0, 83u);
    worker_c.complete();

    wait_consumed(first_group.task_slot);
    wait_consumed(second_group.task_slot);
    wait_consumed(single_a.task_slot);
    wait_consumed(single_c.task_slot);
}

TEST_F(GroupSchedulerFixture, TearDownDrainsCurrentAndQueuedDispatches) {
    // The verification is the teardown itself: work is left deliberately in
    // both states — running and queued-but-undispatched — so teardown drains
    // a worker mid-task and one whose dispatch has not happened yet.
    {
        std::lock_guard<std::mutex> scheduler_pause(sched.loop_mutex());
        (void)orch.submit_next_level_group(
            C(84), {single_tensor_args(0x106, TensorArgType::OUTPUT), single_tensor_args(0x107, TensorArgType::OUTPUT)},
            cfg, {0, 1}
        );
        (void)orch.submit_next_level_group(
            C(85), {single_tensor_args(0x108, TensorArgType::OUTPUT), single_tensor_args(0x109, TensorArgType::OUTPUT)},
            cfg, {1, 2}
        );
        (void)orch.submit_next_level(C(86), single_tensor_args(0x10A, TensorArgType::OUTPUT), cfg, 0);
        (void)orch.submit_next_level(C(87), single_tensor_args(0x10B, TensorArgType::OUTPUT), cfg, 2);
    }

    worker_a.wait_running();
    worker_b.wait_running();
    EXPECT_TRUE(worker_a.is_running.load(std::memory_order_acquire));
    EXPECT_TRUE(worker_b.is_running.load(std::memory_order_acquire));
}

TEST_F(GroupSchedulerFixture, LaunchableGroupPrecedesConflictingSingles) {
    auto running_a = orch.submit_next_level(C(73), single_tensor_args(0xF4, TensorArgType::OUTPUT), cfg, 0);
    auto running_b = orch.submit_next_level(C(74), single_tensor_args(0xF5, TensorArgType::OUTPUT), cfg, 1);
    worker_a.wait_running();
    worker_b.wait_running();
    ASSERT_TRUE(worker_a.is_running.load());
    ASSERT_TRUE(worker_b.is_running.load());

    TaskArgs group_a = single_tensor_args(0xF6, TensorArgType::OUTPUT);
    TaskArgs group_b = single_tensor_args(0xF7, TensorArgType::OUTPUT);
    SubmitResult group;
    SubmitResult single_a;
    SubmitResult single_b;
    {
        std::lock_guard<std::mutex> scheduler_pause(sched.loop_mutex());
        group = orch.submit_next_level_group(C(75), {group_a, group_b}, cfg, {0, 1});
        single_a = orch.submit_next_level(C(76), single_tensor_args(0xF8, TensorArgType::OUTPUT), cfg, 0);
        single_b = orch.submit_next_level(C(77), single_tensor_args(0xF9, TensorArgType::OUTPUT), cfg, 1);
        worker_a.complete();
        worker_b.complete();

        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (manager.any_busy() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ASSERT_FALSE(manager.any_busy());
    }

    worker_a.wait_running();
    worker_b.wait_running();
    ASSERT_EQ(worker_a.dispatched_count(), 2);
    ASSERT_EQ(worker_b.dispatched_count(), 2);
    EXPECT_EQ(worker_a.dispatched[1].callable_hash0, 75u);
    EXPECT_EQ(worker_b.dispatched[1].callable_hash0, 75u);

    worker_a.complete();
    worker_b.complete();
    wait_consumed(group.task_slot);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while ((worker_a.dispatched_count() < 3 || worker_b.dispatched_count() < 3) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(worker_a.dispatched_count(), 3);
    ASSERT_EQ(worker_b.dispatched_count(), 3);
    EXPECT_EQ(worker_a.dispatched[2].callable_hash0, 76u);
    EXPECT_EQ(worker_b.dispatched[2].callable_hash0, 77u);
    worker_a.complete();
    worker_b.complete();

    wait_consumed(running_a.task_slot);
    wait_consumed(running_b.task_slot);
    wait_consumed(single_a.task_slot);
    wait_consumed(single_b.task_slot);
}

TEST_F(GroupSchedulerFixture, GroupFailureWaitsForRunningMembersThenConsumes) {
    TaskArgs a0 = single_tensor_args(0xC0, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xC1, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level_group(C(42), {a0, a1}, cfg, {0, 1});
    TaskSlot slot = res.task_slot;

    worker_a.wait_running();
    worker_b.wait_running();

    worker_a.complete_with_error("member boom");
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    while (!orch.run_failed(run_id) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(orch.run_failed(run_id));
    EXPECT_EQ(S(slot).state.load(), TaskState::RUNNING);

    worker_b.complete();
    wait_consumed(slot);
    EXPECT_EQ(S(slot).state.load(), TaskState::CONSUMED);
}

TEST_F(GroupSchedulerFixture, CompletionRepairPreservesRunningPeersWhenOutcomesAreMissing) {
    TaskArgs a0 = single_tensor_args(0xC2, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xC3, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level_group(C(43), {a0, a1}, cfg, {0, 1});
    TaskSlot slot = res.task_slot;

    worker_a.wait_running();
    worker_b.wait_running();
    {
        std::lock_guard<std::mutex> lk(S(slot).group_mu);
        ASSERT_EQ(S(slot).group_member_states.size(), 2u);
        ASSERT_EQ(S(slot).group_member_outcomes.size(), 2u);
        S(slot).group_member_outcomes.clear();
    }

    worker_a.complete_with_error("first member failed");
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    bool repaired = false;
    while (!repaired && std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> lk(S(slot).group_mu);
            repaired = S(slot).group_member_states.size() == 2u && S(slot).group_member_outcomes.size() == 2u &&
                       S(slot).group_member_states[0] == GroupMemberState::FAILED &&
                       S(slot).group_member_states[1] == GroupMemberState::RUNNING;
        }
        if (!repaired) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(repaired) << "worker_done indexed mismatched group bookkeeping instead of repairing both vectors";
    EXPECT_EQ(S(slot).state.load(std::memory_order_acquire), TaskState::RUNNING)
        << "repair discarded a live peer and let group failure consume its slot early";

    worker_b.complete();
    wait_consumed(slot);
    EXPECT_EQ(S(slot).state.load(), TaskState::CONSUMED);
}

TEST_F(GroupSchedulerFixture, InvalidGroupIndexFailsAndConsumesGroup) {
    TaskArgs a0 = single_tensor_args(0xD0, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xD1, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level_group(C(42), {a0, a1}, cfg, {0, 1});
    TaskSlot slot = res.task_slot;

    worker_a.wait_running();
    worker_b.wait_running();

    WorkerCompletion bad;
    bad.task_slot = slot;
    bad.group_index = 99;
    bad.outcome = EndpointOutcome::ENDPOINT_FAILURE;
    bad.error_message = "bad completion index";
    sched.worker_done(std::move(bad));

    wait_consumed(slot);
    EXPECT_EQ(S(slot).state.load(), TaskState::CONSUMED);

    worker_a.complete();
    worker_b.complete();
}

TEST_F(GroupSchedulerFixture, ExplicitTargetWithinEligibilityIsUsed) {
    TaskArgs args = single_tensor_args(0xE0, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level(C(55), args, cfg, 1, {1});
    TaskSlot slot = res.task_slot;

    worker_b.wait_running();
    EXPECT_FALSE(worker_a.is_running.load());
    EXPECT_TRUE(worker_b.is_running.load());
    EXPECT_EQ(worker_a.dispatched_count(), 0);
    EXPECT_EQ(worker_b.dispatched_count(), 1);

    worker_b.complete();
    wait_consumed(slot);
}

TEST_F(GroupSchedulerFixture, BusyTargetDoesNotBlockAnotherWorkerQueue) {
    auto running_args = single_tensor_args(0xE4, TensorArgType::OUTPUT);
    auto running = orch.submit_next_level(C(62), running_args, cfg, 0);
    worker_a.wait_running();
    ASSERT_TRUE(worker_a.is_running.load());

    auto blocked_args = single_tensor_args(0xE5, TensorArgType::OUTPUT);
    auto blocked = orch.submit_next_level(C(63), blocked_args, cfg, 0);
    auto blocked_second_args = single_tensor_args(0xE8, TensorArgType::OUTPUT);
    auto blocked_second = orch.submit_next_level(C(67), blocked_second_args, cfg, 0);
    auto independent_args = single_tensor_args(0xE6, TensorArgType::OUTPUT);
    auto independent = orch.submit_next_level(C(64), independent_args, cfg, 1);

    worker_b.wait_running();
    ASSERT_TRUE(worker_b.is_running.load());
    EXPECT_EQ(worker_b.dispatched_count(), 1);
    EXPECT_EQ(worker_b.dispatched[0].callable_hash0, 64u);

    worker_b.complete();
    wait_consumed(independent.task_slot);
    worker_a.complete();

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_a.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(worker_a.dispatched_count(), 2);
    EXPECT_EQ(worker_a.dispatched[1].callable_hash0, 63u);
    worker_a.complete();

    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_a.dispatched_count() < 3 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(worker_a.dispatched_count(), 3);
    EXPECT_EQ(worker_a.dispatched[2].callable_hash0, 67u);
    worker_a.complete();
    wait_consumed(running.task_slot);
    wait_consumed(blocked.task_slot);
    wait_consumed(blocked_second.task_slot);
}

TEST_F(GroupSchedulerFixture, DependencyReleaseUsesConsumerWorkerQueue) {
    auto producer_args = single_tensor_args(0xE7, TensorArgType::OUTPUT);
    auto producer = orch.submit_next_level(C(65), producer_args, cfg, 0);
    auto consumer_args = single_tensor_args(0xE7, TensorArgType::INPUT);
    auto consumer = orch.submit_next_level(C(66), consumer_args, cfg, 1);
    EXPECT_EQ(S(consumer.task_slot).state.load(), TaskState::PENDING);

    worker_a.wait_running();
    ASSERT_TRUE(worker_a.is_running.load());
    EXPECT_FALSE(worker_b.is_running.load());
    worker_a.complete();

    worker_b.wait_running();
    ASSERT_TRUE(worker_b.is_running.load());
    EXPECT_EQ(worker_b.dispatched[0].callable_hash0, 66u);
    worker_b.complete();
    wait_consumed(producer.task_slot);
    wait_consumed(consumer.task_slot);
}

// A producer that fails while its consumer is still being submitted. Wiring
// happens under each producer's fanout_mu, so the consumer is reachable from
// the failing producer's fanout list well before its own fanin/fanout counters
// are final — which is exactly what BUILDING marks. The poison must stop at the
// claim and leave the propagation to the submitting thread; running it from
// both sides releases every producer reference the consumer holds twice.
//
// The window is opened by holding the *second* producer's fanout_mu: submit
// wires the first producer, then parks on the second, and the failure is
// injected in between.
TEST_F(GroupSchedulerFixture, APoisonThatLandsMidSubmitLeavesThePropagationToSubmit) {
    auto failing = orch.submit_next_level(C(70), single_tensor_args(0xF100, TensorArgType::OUTPUT), cfg, 0);
    auto blocking = orch.submit_next_level(C(71), single_tensor_args(0xB200, TensorArgType::OUTPUT), cfg, 1);
    worker_a.wait_running();
    worker_b.wait_running();

    TaskArgs consumer_args;
    for (uint64_t key : {0xF100ULL, 0xB200ULL}) {
        Tensor t{};
        t.buffer.addr = key;
        t.ndims = 1;
        t.shapes[0] = 1;
        t.dtype = DataType::UINT8;
        consumer_args.add_tensor(t, TensorArgType::INPUT);
    }

    std::unique_lock<std::mutex> parked(S(blocking.task_slot).fanout_mu);
    std::thread submitter([&] {
        (void)orch.submit_next_level(C(72), consumer_args, cfg, 2);
    });

    // Wired into `failing` and now parked on `blocking`: the exact window.
    TaskSlot consumer = INVALID_SLOT;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (consumer == INVALID_SLOT && std::chrono::steady_clock::now() < deadline) {
        std::lock_guard<std::mutex> lk(S(failing.task_slot).fanout_mu);
        if (!S(failing.task_slot).fanout_consumers.empty()) consumer = S(failing.task_slot).fanout_consumers[0];
    }
    ASSERT_NE(consumer, INVALID_SLOT) << "submit never reached the failing producer's fanout list";
    ASSERT_EQ(S(consumer).state.load(std::memory_order_acquire), TaskState::BUILDING);

    worker_a.complete_with_error("producer boom");
    while (S(consumer).state.load(std::memory_order_acquire) == TaskState::BUILDING &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(S(consumer).state.load(std::memory_order_acquire), TaskState::FAILED)
        << "the poison did not claim the consumer while it was building";

    parked.unlock();
    submitter.join();

    // One release per producer reference the consumer holds, from the one
    // thread that knows the wiring is final. `failing` reaches its threshold
    // (fanout_total 1 + the terminal self release) and no further.
    EXPECT_EQ(S(consumer).state.load(std::memory_order_acquire), TaskState::CONSUMED);
    EXPECT_EQ(S(failing.task_slot).fanout_released.load(std::memory_order_acquire), 2);
    EXPECT_EQ(S(blocking.task_slot).fanout_released.load(std::memory_order_acquire), 1);

    worker_b.complete();
    wait_consumed(blocking.task_slot);
}

TEST_F(GroupSchedulerFixture, TargetMustBeInEligibleEndpointSet) {
    TaskArgs args = single_tensor_args(0xE1, TensorArgType::OUTPUT);
    EXPECT_THROW((void)orch.submit_next_level(C(56), args, cfg, 0, {1}), std::invalid_argument);
}

TEST_F(GroupSchedulerFixture, UnknownEligibleWorkerIdIsRejectedBeforeScheduling) {
    TaskArgs args = single_tensor_args(0xE3, TensorArgType::OUTPUT);
    EXPECT_THROW((void)orch.submit_next_level(C(59), args, cfg, 99, {99}), std::invalid_argument);
}

TEST(SchedulerWorkerTargetTest, NextLevelTargetUsesWorkerIdNotVectorIndex) {
    TensorMap tm;
    Ring allocator;
    Scope scope;
    ReadyQueue rq_sub;
    NextLevelReadyQueues rq_next_level;
    Orchestrator orch;
    MockMailboxWorker worker_a;
    MockMailboxWorker worker_b;
    WorkerManager manager;
    Scheduler sched;
    CallConfig cfg;
    std::vector<TaskSlot> consumed_slots;
    std::mutex consumed_mu;

    allocator.init(/*heap_bytes=*/1ULL << 20);
    worker_a.start();
    worker_b.start();
    manager.add_next_level_at(7, worker_a.mailbox_ptr());
    manager.add_next_level_at(9, worker_b.mailbox_ptr());
    manager.start(
        &allocator,
        [&sched](WorkerCompletion completion) {
            sched.worker_done(std::move(completion));
        },
        [&orch](WorkerDispatch dispatch) {
            orch.mark_task_accepted(dispatch.task_slot);
        }
    );
    rq_next_level.reset(manager.next_level_worker_ids());
    orch.init(&tm, &allocator, &scope, &rq_sub, &rq_next_level, &manager, [&sched] {
        sched.notify_ready();
    });
    (void)orch.begin_run();

    Scheduler::Config c;
    c.ring = &allocator;
    c.ready_sub_queue = &rq_sub;
    c.ready_next_level_queues = &rq_next_level;
    c.manager = &manager;
    c.enqueue_ready_cb = [&orch](TaskSlot slot) {
        orch.enqueue_ready(slot);
    };
    c.on_consumed_cb = [&orch, &consumed_slots, &consumed_mu](TaskSlot s) {
        orch.on_consumed(s);
        std::lock_guard<std::mutex> lk(consumed_mu);
        consumed_slots.push_back(s);
    };
    c.on_task_failed_cb = [&orch](TaskSlot s, const std::string &message) {
        orch.report_task_error(s, message);
    };
    sched.start(c);

    auto wait_consumed_slot = [&consumed_slots, &consumed_mu](TaskSlot slot) {
        bool consumed = false;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lk(consumed_mu);
                consumed = std::find(consumed_slots.begin(), consumed_slots.end(), slot) != consumed_slots.end();
            }
            if (consumed) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_TRUE(consumed);
    };

    TaskArgs args = single_tensor_args(0xE2, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level(C(58), args, cfg, 9);

    worker_b.wait_running();
    EXPECT_FALSE(worker_a.is_running.load());
    EXPECT_TRUE(worker_b.is_running.load());
    EXPECT_EQ(worker_a.dispatched_count(), 0);
    EXPECT_EQ(worker_b.dispatched_count(), 1);

    if (worker_a.is_running.load()) worker_a.complete();
    if (worker_b.is_running.load()) worker_b.complete();

    wait_consumed_slot(res.task_slot);

    TaskArgs a0 = single_tensor_args(0xE6, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xE7, TensorArgType::OUTPUT);
    auto group_res = orch.submit_next_level_group(C(61), {a0, a1}, cfg, {7, 9}, {{7}, {9}});

    worker_a.wait_running();
    worker_b.wait_running();
    EXPECT_TRUE(worker_a.is_running.load());
    EXPECT_TRUE(worker_b.is_running.load());
    EXPECT_EQ(worker_a.dispatched_count(), 1);
    EXPECT_EQ(worker_b.dispatched_count(), 2);

    worker_a.complete();
    worker_b.complete();
    wait_consumed_slot(group_res.task_slot);

    sched.stop();
    manager.stop();
    allocator.shutdown();
}

TEST_F(GroupSchedulerFixture, RemoteSidecarRejectsLocalEndpointEligibility) {
    TaskArgs args;
    Tensor tensor{};
    tensor.buffer.addr = 0;
    tensor.ndims = 1;
    tensor.shapes[0] = 1;
    tensor.dtype = DataType::UINT8;
    args.add_tensor(tensor, TensorArgType::OUTPUT);

    RemoteTaskArgsSidecar sidecar;
    sidecar.tensors.resize(1);
    sidecar.tensors[0].present = true;
    sidecar.tensors[0].desc.address_space = RemoteAddressSpace::REMOTE_DEVICE;
    sidecar.tensors[0].desc.owner_worker_id = 7;
    sidecar.tensors[0].desc.buffer_id = 11;
    sidecar.tensors[0].desc.generation = 1;
    sidecar.tensors[0].desc.nbytes = 1;

    EXPECT_THROW((void)orch.submit_next_level(C(57), args, cfg, 0, {0}, sidecar), std::invalid_argument);
}

// ===========================================================================
// Directed NEXT_LEVEL and shared SUB queues do not block each other. Covered
// here with one worker of each type: a SUB task submitted while the exact
// NEXT_LEVEL target is busy must still dispatch immediately.
// ===========================================================================

struct MixedTypeSchedulerFixture : public ::testing::Test {
    TensorMap tm;
    Ring allocator;
    Scope scope;
    ReadyQueue rq_sub;
    NextLevelReadyQueues rq_next_level;
    Orchestrator orch;
    MockMailboxWorker next_level_worker;
    MockMailboxWorker sub_worker;
    WorkerManager manager;
    Scheduler sched;
    CallConfig cfg;
    RunId run_id{INVALID_RUN_ID};

    std::vector<TaskSlot> consumed_slots;
    std::mutex consumed_mu;

    TaskSlotState &S(TaskSlot id) { return *allocator.slot_state(id); }

    void SetUp() override {
        allocator.init(/*heap_bytes=*/1ULL << 20);

        next_level_worker.start();
        sub_worker.start();
        manager.add_next_level(next_level_worker.mailbox_ptr());
        manager.add_sub(sub_worker.mailbox_ptr());
        manager.start(
            &allocator,
            [this](WorkerCompletion completion) {
                sched.worker_done(std::move(completion));
            },
            [this](WorkerDispatch dispatch) {
                orch.mark_task_accepted(dispatch.task_slot);
            }
        );
        rq_next_level.reset(manager.next_level_worker_ids());
        orch.init(&tm, &allocator, &scope, &rq_sub, &rq_next_level, &manager, [this] {
            sched.notify_ready();
        });
        run_id = orch.begin_run();

        Scheduler::Config c;
        c.ring = &allocator;
        c.ready_sub_queue = &rq_sub;
        c.ready_next_level_queues = &rq_next_level;
        c.manager = &manager;
        c.enqueue_ready_cb = [this](TaskSlot slot) {
            orch.enqueue_ready(slot);
        };
        // Same gate Worker::start installs: an active run that is also the
        // EXECUTING FIFO head and still owns its pipeline lease. Testing
        // against the weaker active_run_id() would let a slot dispatch here
        // that production refuses.
        c.active_run_cb = [this] {
            return orch.dispatchable_run_id();
        };
        c.on_consumed_cb = [this](TaskSlot s) {
            orch.on_consumed(s);
            std::lock_guard<std::mutex> lk(consumed_mu);
            consumed_slots.push_back(s);
        };
        c.on_task_failed_cb = [this](TaskSlot s, const std::string &message) {
            orch.report_task_error(s, message);
        };
        sched.start(c);
    }

    void TearDown() override {
        next_level_worker.drain();
        sub_worker.drain();
        sched.stop();
        manager.stop();
        allocator.shutdown();
    }

    bool is_consumed(TaskSlot slot) {
        std::lock_guard<std::mutex> lk(consumed_mu);
        for (TaskSlot s : consumed_slots)
            if (s == slot) return true;
        return false;
    }

    void wait_consumed(TaskSlot slot, int timeout_ms = 500) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            if (is_consumed(slot)) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        FAIL() << "Timed out waiting for slot " << slot << " to be consumed";
    }
};

TEST_F(MixedTypeSchedulerFixture, SubTaskDispatchesWhileNextLevelPoolSaturated) {
    // Submit a next-level task; the only chip worker begins running it and
    // stays blocked until we call complete() on it.
    auto chip_args = single_tensor_args(0xAAA, TensorArgType::OUTPUT);
    auto chip = orch.submit_next_level(C(20), chip_args, cfg, 0);
    next_level_worker.wait_running();
    ASSERT_TRUE(next_level_worker.is_running.load());

    // Now submit a sub task while the chip pool is saturated. With a single
    // shared ready queue this would block behind any next-level task sitting
    // in worker 0's directed FIFO. The independent shared SUB queue must
    // dispatch immediately to the idle SUB worker.
    auto sub_args = single_tensor_args(0xBBB, TensorArgType::OUTPUT);
    auto sub = orch.submit_sub(C(7), sub_args);

    sub_worker.wait_running();
    EXPECT_TRUE(sub_worker.is_running.load());
    EXPECT_TRUE(next_level_worker.is_running.load()) << "chip worker must still be busy";

    // Complete the sub task first; it reaches CONSUMED while the chip task
    // is still running -- demonstrating independent queue dispatch.
    sub_worker.complete();
    wait_consumed(sub.task_slot);
    EXPECT_FALSE(is_consumed(chip.task_slot));

    next_level_worker.complete();
    wait_consumed(chip.task_slot);
}

TEST_F(MixedTypeSchedulerFixture, BusySubWorkerRequeuesWithinTheActiveRun) {
    auto first = orch.submit_sub(C(8), single_tensor_args(0xC01, TensorArgType::OUTPUT));
    sub_worker.wait_running();
    ASSERT_TRUE(sub_worker.is_running.load());

    auto second = orch.submit_sub(C(9), single_tensor_args(0xC02, TensorArgType::OUTPUT));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_EQ(sub_worker.dispatched_count(), 1);

    sub_worker.complete();
    wait_consumed(first.task_slot);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (sub_worker.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(sub_worker.dispatched_count(), 2);
    EXPECT_TRUE(sub_worker.is_running.load());

    sub_worker.complete();
    wait_consumed(second.task_slot);
}

TEST_F(GroupSchedulerFixture, GroupDependencyChain) {
    // Group A (2 workers) produces an OUTPUT at key 0xCAFE.
    // Task B reads INPUT at the same key -- depends on group A.
    TaskArgs a0 = single_tensor_args(0xCAFE, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xCAFE, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level_group(C(42), {a0, a1}, cfg, {0, 1});

    auto args_b = single_tensor_args(0xCAFE, TensorArgType::INPUT);
    auto b = orch.submit_next_level(C(42), args_b, cfg, 0);
    EXPECT_EQ(S(b.task_slot).state.load(), TaskState::PENDING);

    worker_a.wait_running();
    worker_b.wait_running();
    worker_a.complete();
    worker_b.complete();

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_a.dispatched_count() + worker_b.dispatched_count() < 3 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    int total = worker_a.dispatched_count() + worker_b.dispatched_count();
    EXPECT_EQ(total, 3);  // 2 from group A + exactly 1 downstream dispatch

    if (worker_a.is_running.load()) worker_a.complete();
    if (worker_b.is_running.load()) worker_b.complete();
    wait_consumed(b.task_slot);
    (void)a;  // suppress unused
}
