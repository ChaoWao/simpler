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
 * Host-view resolution for the host orchestrator's tensor reads and writes.
 *
 * The mirror path (a platform that cannot map device memory into the host
 * address space) has no reachable call site on a2a3, whose SVM map always
 * succeeds, so these tests are the only place it executes.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "common/host_api.h"
#include "host_tensor_access.h"

namespace {

// Stands in for a device address range that no host load can reach. Only its
// arithmetic is exercised — nothing dereferences it.
constexpr uint64_t kFakeDeviceBase = 0x7000'0000'0000ull;

struct CopyCall {
    void *dev_ptr;
    const void *host_ptr;
    size_t size;
};

std::vector<CopyCall> g_copies;
int g_copy_result = 0;

int record_copy(void *dev_ptr, const void *host_ptr, size_t size) {
    g_copies.push_back({dev_ptr, host_ptr, size});
    return g_copy_result;
}

// host_tensor_access_reset retains the HostApi pointer for the run, so this
// static test object outlives every binding. The adapter ignores runner_ctx.
int record_copy_for_api(void * /*runner_ctx*/, void *dev_ptr, const void *host_ptr, size_t size) {
    return record_copy(dev_ptr, host_ptr, size);
}
const HostApiOps kRecordCopyOps{.copy_to_device = record_copy_for_api};
HostApi record_copy_api(nullptr, 0, 0, &kRecordCopyOps);

class HostTensorAccessTest : public ::testing::Test {
protected:
    void SetUp() override {
        g_copies.clear();
        g_copy_result = 0;
        host_tensor_access_reset(&record_copy_api);
    }
    void TearDown() override { host_tensor_access_reset(nullptr); }
};

// A buffer the host can load directly: the device address IS a host address,
// as on a2a3 after halHostRegister and on sim.
TEST_F(HostTensorAccessTest, DirectRegionReadsAndWritesInPlace) {
    int32_t buffer[4] = {10, 20, 30, 40};
    const uint64_t base = reinterpret_cast<uint64_t>(buffer);
    ASSERT_TRUE(host_tensor_access_add(base, sizeof(buffer), buffer));

    int32_t value = 0;
    ASSERT_TRUE(host_tensor_read(base + 2 * sizeof(int32_t), &value, sizeof(value)));
    EXPECT_EQ(value, 30);

    const int32_t written = 99;
    ASSERT_TRUE(host_tensor_write(base + sizeof(int32_t), &written, sizeof(written)));
    EXPECT_EQ(buffer[1], 99);
    // The bytes are already the device's, so no push-back is issued.
    EXPECT_TRUE(g_copies.empty());
}

// A buffer the host reaches only through the staging copy.
TEST_F(HostTensorAccessTest, MirroredRegionReadsFromTheHostView) {
    int32_t mirror[4] = {1, 2, 3, 4};
    ASSERT_TRUE(host_tensor_access_add(kFakeDeviceBase, sizeof(mirror), mirror));

    int32_t value = 0;
    ASSERT_TRUE(host_tensor_read(kFakeDeviceBase + 3 * sizeof(int32_t), &value, sizeof(value)));
    EXPECT_EQ(value, 4);
}

TEST_F(HostTensorAccessTest, MirroredWritePushesTheTouchedBytesToTheDevice) {
    int32_t mirror[4] = {1, 2, 3, 4};
    ASSERT_TRUE(host_tensor_access_add(kFakeDeviceBase, sizeof(mirror), mirror));

    const int32_t written = 77;
    const uint64_t dev_addr = kFakeDeviceBase + 2 * sizeof(int32_t);
    ASSERT_TRUE(host_tensor_write(dev_addr, &written, sizeof(written)));

    EXPECT_EQ(mirror[2], 77);
    ASSERT_EQ(g_copies.size(), 1u);
    EXPECT_EQ(g_copies[0].dev_ptr, reinterpret_cast<void *>(dev_addr));
    EXPECT_EQ(g_copies[0].host_ptr, static_cast<const void *>(&mirror[2]));
    EXPECT_EQ(g_copies[0].size, sizeof(int32_t));
}

TEST_F(HostTensorAccessTest, MirroredWriteFailsWhenThePushBackFails) {
    int32_t mirror[2] = {1, 2};
    ASSERT_TRUE(host_tensor_access_add(kFakeDeviceBase, sizeof(mirror), mirror));

    g_copy_result = -1;
    const int32_t written = 5;
    EXPECT_FALSE(host_tensor_write(kFakeDeviceBase, &written, sizeof(written)));
}

TEST_F(HostTensorAccessTest, MirroredWriteFailsWithNoCopyHook) {
    int32_t mirror[2] = {1, 2};
    host_tensor_access_reset(nullptr);
    ASSERT_TRUE(host_tensor_access_add(kFakeDeviceBase, sizeof(mirror), mirror));

    const int32_t written = 5;
    EXPECT_FALSE(host_tensor_write(kFakeDeviceBase, &written, sizeof(written)));
}

// The fail-closed contract: an address outside every registered region — a
// GM-heap tensor the orchestrator created, or a pass-through child-memory
// buffer — resolves to nothing instead of being dereferenced.
TEST_F(HostTensorAccessTest, UnregisteredAddressFailsAndLeavesTheDestination) {
    int32_t mirror[2] = {1, 2};
    ASSERT_TRUE(host_tensor_access_add(kFakeDeviceBase, sizeof(mirror), mirror));

    int32_t value = 0xABCD;
    EXPECT_FALSE(host_tensor_read(kFakeDeviceBase + 0x100000, &value, sizeof(value)));
    EXPECT_EQ(value, 0xABCD);

    const int32_t written = 5;
    EXPECT_FALSE(host_tensor_write(kFakeDeviceBase + 0x100000, &written, sizeof(written)));
    EXPECT_TRUE(g_copies.empty());
}

TEST_F(HostTensorAccessTest, SpanRunningPastTheRegionEndFails) {
    int32_t mirror[2] = {1, 2};
    ASSERT_TRUE(host_tensor_access_add(kFakeDeviceBase, sizeof(mirror), mirror));

    int64_t value = 0;
    // Starts inside the region, ends past it.
    EXPECT_FALSE(host_tensor_read(kFakeDeviceBase + sizeof(int32_t), &value, sizeof(value)));
    // Starts before it.
    EXPECT_FALSE(host_tensor_read(kFakeDeviceBase - sizeof(int32_t), &value, sizeof(int32_t)));
}

TEST_F(HostTensorAccessTest, ResetDropsEveryRegistration) {
    int32_t mirror[2] = {1, 2};
    ASSERT_TRUE(host_tensor_access_add(kFakeDeviceBase, sizeof(mirror), mirror));

    int32_t value = 0;
    ASSERT_TRUE(host_tensor_read(kFakeDeviceBase, &value, sizeof(value)));

    host_tensor_access_reset(&record_copy_api);
    EXPECT_FALSE(host_tensor_read(kFakeDeviceBase, &value, sizeof(value)));
}

TEST_F(HostTensorAccessTest, EmptyOrNullRegionIsRejected) {
    int32_t mirror[2] = {1, 2};
    EXPECT_FALSE(host_tensor_access_add(kFakeDeviceBase, 0, mirror));
    EXPECT_FALSE(host_tensor_access_add(kFakeDeviceBase, sizeof(mirror), nullptr));
}

// Several tensors are staged per run and each resolves against its own region.
TEST_F(HostTensorAccessTest, RegionsAreResolvedIndependently) {
    int32_t first[2] = {1, 2};
    int32_t second[2] = {3, 4};
    const uint64_t second_base = kFakeDeviceBase + 0x10000;
    ASSERT_TRUE(host_tensor_access_add(kFakeDeviceBase, sizeof(first), first));
    ASSERT_TRUE(host_tensor_access_add(second_base, sizeof(second), second));

    int32_t value = 0;
    ASSERT_TRUE(host_tensor_read(kFakeDeviceBase, &value, sizeof(value)));
    EXPECT_EQ(value, 1);
    ASSERT_TRUE(host_tensor_read(second_base + sizeof(int32_t), &value, sizeof(value)));
    EXPECT_EQ(value, 4);
}

}  // namespace
