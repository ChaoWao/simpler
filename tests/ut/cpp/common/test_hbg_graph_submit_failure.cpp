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

#include <array>
#include <cstdint>
#include <vector>

#include "graph_host_state.h"
#include "pto_orchestrator.h"
#include "pto_shared_memory.h"
#include "utils/device_arena.h"

class HbgGraphSubmitFailureTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    PTO2SharedMemoryHandle *sm_handle = nullptr;
    PTO2OrchestratorState orch{};
    PTO2SchedulerState sched{};
    PTO2OrchestratorLayout orch_layout{};
    PTO2SchedulerLayout sched_layout{};
    GraphHostStatePtr graph_state;
    std::vector<char> gm_heap;

    void SetUp() override {
        sm_handle = PTO2SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(4096 * PTO2_MAX_RING_DEPTH);

        orch_layout = PTO2OrchestratorState::reserve_layout(runtime_arena, static_cast<int32_t>(PTO2_TASK_WINDOW_SIZE));
        sched_layout = PTO2SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);

        ASSERT_TRUE(orch.init_data_from_layout(
            orch_layout, runtime_arena, sm_handle->sm_base, gm_heap.data(), 4096, PTO2_TASK_WINDOW_SIZE
        ));
        ASSERT_TRUE(sched.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(sched_layout, runtime_arena);
        orch.wire_arena_pointers(orch_layout, runtime_arena, &sched);

        graph_state = make_graph_host_state();
        ASSERT_NE(graph_state, nullptr);
        orch.graph_host_state = graph_state.get();
    }

    void TearDown() override {
        orch.graph_host_state = nullptr;
        graph_state.reset();
        orch.destroy();
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }
};

TEST_F(HbgGraphSubmitFailureTest, FaninFailureLatchesFatalWithoutPartialUpload) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    ChipTensor boundary = make_tensor_external(storage.data(), shape, 1);

    orch.begin_scope();
    CoreTaskArgs boundary_args;
    boundary_args.add_input(boundary);
    const GraphScopeResult graph = orch.graph_begin(0x1715, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);

    CoreTaskArgs node_args;
    node_args.add_input(boundary);
    ASSERT_TRUE(orch.submit_dummy_task(node_args).task_id().is_valid());
    orch.graph_end();
    ASSERT_FALSE(orch.fatal);
    const size_t uploads_before_failure = graph_host_upload_count(*graph_state);

    CoreTaskArgs producer_args;
    producer_args.add_output(boundary);
    for (int32_t i = 0; i < PTO2_MAX_FANIN + 1; ++i) {
        ASSERT_TRUE(orch.submit_dummy_task(producer_args).task_id().is_valid());
    }

    const GraphScopeResult replay = orch.graph_begin(0x1715, boundary_args, 0x1736);

    EXPECT_TRUE(replay.execute_block);
    EXPECT_FALSE(replay.recording);
    EXPECT_FALSE(replay.task_id.is_valid());
    EXPECT_TRUE(orch.fatal);
    EXPECT_EQ(sm_handle->header->orch_error_code.load(std::memory_order_acquire), PTO2_ERROR_DEP_POOL_OVERFLOW);
    EXPECT_EQ(graph_host_upload_count(*graph_state), uploads_before_failure);
}
