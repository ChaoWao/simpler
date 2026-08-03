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

/**
 * Publish fatal teardown before publishing run completion.
 *
 * A thread that observes `completed` with acquire ordering is guaranteed to
 * observe `fatal_started` too, so it cannot enter the healthy per-thread
 * shutdown path for a fatal run and race the emergency broadcast for the same
 * cores. The return value elects exactly one caller to run that broadcast.
 */
inline bool publish_fatal_shutdown(std::atomic<bool> &fatal_started, std::atomic<bool> &completed) noexcept {
    const bool first = !fatal_started.exchange(true, std::memory_order_acq_rel);
    completed.store(true, std::memory_order_release);
    return first;
}

/**
 * Drive the fatal-path device reset until an attempt confirms the card clean.
 *
 * Each attempt drains the device before resetting it, so a later attempt runs
 * against a settled card and can confirm clean where the first did not.
 *
 * `max_attempts` must be 1 whenever the device still holds CP-process SDMA
 * streams: there a reset that does not confirm blocks on the driver's
 * 150/300-second remote-event timeout, and a further attempt only multiplies
 * that wait without a new completion condition. Values below 1 are treated
 * as 1.
 *
 * Returns 0 on the first confirming attempt, otherwise the last attempt's
 * error; the caller quarantines host-side handles on a non-zero return.
 */
template <typename ResetFn>
inline int attempt_fatal_reset(ResetFn &&reset, int max_attempts) {
    const int attempts = max_attempts < 1 ? 1 : max_attempts;
    int rc = 0;
    for (int attempt = 0; attempt < attempts; ++attempt) {
        rc = reset();
        if (rc == 0) break;
    }
    return rc;
}
