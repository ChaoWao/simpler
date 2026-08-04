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
#include <cstdint>

namespace simpler {

// Rendezvous for N threads that must run a one-shot finalizer after the last
// arrival, then hand exactly one thread the right to tear down what the
// finalizer released.
//
// The release store on cleanup_ready_ happens after finalize() returns and
// pairs with the acquire on a successful claim_cleanup() CAS: a thread that
// wins the claim is guaranteed to observe every write the finalizer made.
// Publishing eligibility before running the finalizer would let a peer tear
// down state the last thread is still using.
//
// thread_count must be the same on every arrival of a given round; it is the
// round's participant count, not a property of one arrival.
//
// reset() is NOT synchronized against the other two operations. It may only be
// called when no thread is between its arrive and the end of its finalizer —
// i.e. from a single thread outside the round (a run's setup or teardown).
class ThreadCompletionGate {
public:
    template <typename Finalize>
    void arrive_and_finalize_if_last(int32_t thread_count, Finalize finalize) {
        int32_t previous = arrived_.fetch_add(1, std::memory_order_acq_rel);
        if (previous + 1 != thread_count) {
            return;
        }

        finalize();
        cleanup_ready_.store(true, std::memory_order_release);
    }

    bool claim_cleanup() {
        bool expected = true;
        return cleanup_ready_.compare_exchange_strong(
            expected, false, std::memory_order_acquire, std::memory_order_relaxed
        );
    }

    void reset() {
        arrived_.store(0, std::memory_order_release);
        cleanup_ready_.store(false, std::memory_order_release);
    }

private:
    std::atomic<int32_t> arrived_{0};
    std::atomic<bool> cleanup_ready_{false};
};

}  // namespace simpler
