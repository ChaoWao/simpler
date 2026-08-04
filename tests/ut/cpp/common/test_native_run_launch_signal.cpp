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
#include <ctime>
#include <future>
#include <thread>

#include "native_run_launch_signal.h"

TEST(NativeRunLaunchSignalTest, WaitBlocksWithoutConsumingCpu) {
    NativeRunLaunchSignal signal;
    std::promise<void> waiter_started;
    std::promise<void> waiter_returned;
    auto started = waiter_started.get_future();
    auto returned = waiter_returned.get_future();

    std::thread waiter([&]() {
        waiter_started.set_value();
        signal.wait();
        waiter_returned.set_value();
    });

    started.wait();
    const std::clock_t cpu_start = std::clock();
    const std::future_status wait_status = returned.wait_for(std::chrono::milliseconds(200));
    const double cpu_seconds = static_cast<double>(std::clock() - cpu_start) / CLOCKS_PER_SEC;
    signal.notify();
    const std::future_status return_status = returned.wait_for(std::chrono::seconds(1));
    waiter.join();

    EXPECT_EQ(wait_status, std::future_status::timeout);
    EXPECT_EQ(return_status, std::future_status::ready);
    EXPECT_LT(cpu_seconds, 0.05);
}

TEST(NativeRunLaunchSignalTest, NotificationBeforeWaitIsRemembered) {
    NativeRunLaunchSignal signal;
    signal.notify();

    auto waiter = std::async(std::launch::async, [&]() {
        signal.wait();
    });
    EXPECT_EQ(waiter.wait_for(std::chrono::seconds(1)), std::future_status::ready);
}

TEST(NativeRunLaunchSignalTest, PublishAcceptanceStoresOnceAndKeepsNotificationSticky) {
    volatile int32_t accepted_state = 0;
    NativeRunLaunchSignal signal(&accepted_state, 17);

    signal.publish_acceptance();
    EXPECT_EQ(__atomic_load_n(&accepted_state, __ATOMIC_ACQUIRE), 17);

    __atomic_store_n(&accepted_state, 23, __ATOMIC_RELEASE);
    signal.publish_acceptance();
    EXPECT_EQ(__atomic_load_n(&accepted_state, __ATOMIC_ACQUIRE), 23);

    auto waiter = std::async(std::launch::async, [&]() {
        signal.wait();
    });
    EXPECT_EQ(waiter.wait_for(std::chrono::seconds(1)), std::future_status::ready);
}

TEST(NativeRunLaunchSignalTest, NotificationBeforeLaunchMarkerSuppressesAcceptance) {
    volatile int32_t accepted_state = 5;
    NativeRunLaunchSignal signal(&accepted_state, 17);

    signal.notify();
    signal.publish_acceptance();

    EXPECT_EQ(__atomic_load_n(&accepted_state, __ATOMIC_ACQUIRE), 5);
}
