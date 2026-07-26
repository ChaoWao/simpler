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

#include "pipeline_contract.h"

namespace {

// A contract a runtime could legitimately ship today: one host-filled region,
// one device-built region, and the two execution handles.
PipelineContract accepted_contract() {
    PipelineContract c{};
    c.abi_version = PTO_PIPELINE_CONTRACT_ABI_VERSION;
    c.resource_count = 4;
    c.pipeline_depth = 1;
    c.resources[0] = {PTO_PIPELINE_TASK_ARGS, PTO_PIPELINE_HOST_PER_RUN, 0};
    c.resources[1] = {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_DEVICE_SCRATCH, 0};
    c.resources[2] = {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0};
    c.resources[3] = {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0};
    return c;
}

TEST(PipelineContract, AcceptsADeclarationThisBuildCanHonor) {
    const PipelineContract c = accepted_contract();
    EXPECT_TRUE(is_valid_depth1_pipeline_contract(&c));
}

// A runtime that exports no contract is handled by the caller, not here.
TEST(PipelineContract, RejectsNull) { EXPECT_FALSE(is_valid_depth1_pipeline_contract(nullptr)); }

TEST(PipelineContract, AcceptsAnEmptyResourceList) {
    PipelineContract c = accepted_contract();
    c.resource_count = 0;
    EXPECT_TRUE(is_valid_depth1_pipeline_contract(&c));
}

TEST(PipelineContract, RejectsAnotherAbiVersion) {
    PipelineContract c = accepted_contract();
    c.abi_version = PTO_PIPELINE_CONTRACT_ABI_VERSION + 1;
    EXPECT_FALSE(is_valid_depth1_pipeline_contract(&c));
    c.abi_version = 0;
    EXPECT_FALSE(is_valid_depth1_pipeline_contract(&c));
}

TEST(PipelineContract, RejectsMoreResourcesThanFit) {
    PipelineContract c = accepted_contract();
    c.resource_count = PTO_PIPELINE_MAX_RESOURCES + 1;
    EXPECT_FALSE(is_valid_depth1_pipeline_contract(&c));
}

// The depth-1 gate is what keeps this build from honoring a contract whose
// resources it would have to replicate.
TEST(PipelineContract, RejectsADepthOtherThanOne) {
    PipelineContract c = accepted_contract();
    c.pipeline_depth = 0;
    EXPECT_FALSE(is_valid_depth1_pipeline_contract(&c));
    c.pipeline_depth = 2;
    EXPECT_FALSE(is_valid_depth1_pipeline_contract(&c));
    c.pipeline_depth = PTO_PIPELINE_MAX_DEPTH;
    EXPECT_FALSE(is_valid_depth1_pipeline_contract(&c));
}

TEST(PipelineContract, RejectsAnOutOfRangeKindOrClass) {
    PipelineContract c = accepted_contract();
    c.resources[0].kind = PTO_PIPELINE_AICORE_STREAM + 1;
    EXPECT_FALSE(is_valid_depth1_pipeline_contract(&c));

    c = accepted_contract();
    c.resources[0].resource_class = PTO_PIPELINE_EXEC_HANDLE + 1;
    EXPECT_FALSE(is_valid_depth1_pipeline_contract(&c));
}

// bytes_per_copy is reserved: nothing sizes anything from it yet, so a runtime
// that populates it is declaring a contract this build does not implement.
TEST(PipelineContract, RejectsANonZeroReservedSize) {
    PipelineContract c = accepted_contract();
    c.resources[1].bytes_per_copy = 4096;
    EXPECT_FALSE(is_valid_depth1_pipeline_contract(&c));
}

TEST(PipelineContract, RejectsAnUnspecifiedKind) {
    PipelineContract c = accepted_contract();
    c.resources[1].kind = PTO_PIPELINE_KIND_UNSPECIFIED;
    EXPECT_FALSE(is_valid_depth1_pipeline_contract(&c));
}

// A resource_count larger than the entries a runtime filled in leaves trailing
// zeroed entries, and zero is not a resource. This must hold whatever the
// filled entries are — a rule keyed on a collision with the first kind would
// pass only when the declaration happens to use that kind.
TEST(PipelineContract, RejectsAResourceCountPastTheFilledEntries) {
    for (uint32_t filled :
         {static_cast<uint32_t>(PTO_PIPELINE_GM_HEAP), static_cast<uint32_t>(PTO_PIPELINE_TASK_ARGS)}) {
        PipelineContract c{};
        c.abi_version = PTO_PIPELINE_CONTRACT_ABI_VERSION;
        c.pipeline_depth = 1;
        c.resources[0] = {filled, PTO_PIPELINE_HOST_PER_RUN, 0};

        c.resource_count = 2;  // overstates by exactly one
        EXPECT_FALSE(is_valid_depth1_pipeline_contract(&c)) << "filled kind " << filled;
        c.resource_count = 3;
        EXPECT_FALSE(is_valid_depth1_pipeline_contract(&c)) << "filled kind " << filled;
    }
}

// A kind names a resource type, not one instance of it, so a runtime that needs
// two of a kind — two AICore streams for parallel branches, say — can say so.
TEST(PipelineContract, AcceptsARepeatedKind) {
    PipelineContract c = accepted_contract();
    c.resources[3].kind = PTO_PIPELINE_AICPU_STREAM;
    EXPECT_TRUE(is_valid_depth1_pipeline_contract(&c));
}

TEST(PipelineContract, AcceptsEveryKindOnce) {
    PipelineContract c{};
    c.abi_version = PTO_PIPELINE_CONTRACT_ABI_VERSION;
    c.pipeline_depth = 1;
    c.resource_count = PTO_PIPELINE_AICORE_STREAM;
    for (uint32_t kind = PTO_PIPELINE_GM_HEAP; kind <= PTO_PIPELINE_AICORE_STREAM; ++kind) {
        c.resources[kind - 1] = {kind, PTO_PIPELINE_HOST_PER_RUN, 0};
    }
    EXPECT_TRUE(is_valid_depth1_pipeline_contract(&c));
}

}  // namespace
