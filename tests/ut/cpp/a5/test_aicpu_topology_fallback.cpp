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

#include <cstdint>
#include <vector>

#include "aicpu_topology_probe.h"

extern "C" {
void unified_log_error(const char *, const char *, ...) {}
void unified_log_warn(const char *, const char *, ...) {}
void unified_log_info_v(const char *, int, const char *, ...) {}
}

namespace {

using pto::a5::AicpuLogicalCpu;
using pto::a5::compute_allowed_cpus;
using pto::a5::derive_topology_from_occupy;

TEST(A5AicpuTopologyFallback, EnumeratesEveryOccupiedCpu) {
    std::vector<AicpuLogicalCpu> cpus;

    ASSERT_TRUE(derive_topology_from_occupy(0x3e, cpus));
    ASSERT_EQ(cpus.size(), 5U);

    for (int32_t i = 0; i < 5; ++i) {
        EXPECT_EQ(cpus[i].cpu_id, i + 1);
        EXPECT_EQ(cpus[i].phy_cpu_id, i + 1);
        EXPECT_EQ(cpus[i].hyperthread_id, 0);
        EXPECT_EQ(cpus[i].cluster_id, (i + 1) / 2);
        EXPECT_EQ(cpus[i].die_id, (i + 1) / 4);
    }
}

TEST(A5AicpuTopologyFallback, PreservesAffinitySelection) {
    std::vector<AicpuLogicalCpu> cpus;
    ASSERT_TRUE(derive_topology_from_occupy(0x3e, cpus));

    std::vector<int32_t> allowed;
    ASSERT_TRUE(compute_allowed_cpus(cpus, /*n_sched=*/2, /*n_orch=*/1, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{4, 5, 1}));
}

TEST(A5AicpuTopologyFallback, RejectsEmptyOccupyMask) {
    std::vector<AicpuLogicalCpu> cpus = {{1, 1, 0, 0, 0}};

    EXPECT_FALSE(derive_topology_from_occupy(0, cpus));
    EXPECT_TRUE(cpus.empty());
}

}  // namespace
