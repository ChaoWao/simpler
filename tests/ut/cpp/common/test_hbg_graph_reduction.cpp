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

// Exact transitive reduction of the packed Definition's fanin CSR. Each test
// records a small DAG through the real orchestrator path (graph_begin /
// graph_prepare / submit / graph_end), takes the packed image out of the host
// Definition list, and asserts the reduced edge set plus the invariants
// bind_graph_topology enforces on the device.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "graph_execution.h"
#include "graph_host_state.h"
#include "orchestrator.h"
#include "scheduler/scheduler.h"
#include "shared_memory.h"
#include "utils/device_arena.h"

namespace {

class HbgGraphReductionTest : public ::testing::Test {
protected:
    static constexpr size_t HEAP_BYTES = 256 * 1024;

    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    std::vector<char> gm_heap{};
    PTO2SharedMemoryHandle *sm_handle = nullptr;
    PTO2SchedulerLayout sched_layout{};
    PTO2SchedulerState sched{};
    PTO2OrchestratorState orch{};
    GraphHostStatePtr graph_state{};

    void SetUp() override {
        sm_handle = PTO2SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(HEAP_BYTES);

        sched_layout = PTO2SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);

        ASSERT_TRUE(sched.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(sched_layout, runtime_arena);
        ASSERT_TRUE(orch.init(sm_handle->sm_base, gm_heap.data(), HEAP_BYTES, PTO2_TASK_WINDOW_SIZE, &sched));

        graph_state = make_graph_host_state();
        ASSERT_NE(graph_state, nullptr);
        orch.graph_host_state = graph_state.get();
    }

    void TearDown() override {
        orch.graph_host_state = nullptr;
        graph_state.reset();
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }

    // Records `producer_count` dummy tasks (node 0 .. producer_count-1), then
    // one consumer task consuming tensors produced by `consumer_inputs` of the
    // producers. Dependencies come out of creator retention: the consumer
    // takes producer outputs as INPUT tensors, which records one fanin edge per
    // distinct producer.
    //
    // The chain edges between the producers themselves are laid in by giving
    // each producer p (p > 0) the previous producer's output as an input, so
    // producer rows read [p-1]; the consumer's row reads the chosen subset.
    // A diamond needs the subset to include both a middle hop and its own
    // ancestor.
    const GraphDefinition *
    record_chain_and_consumer(uint32_t producer_count, const std::vector<uint32_t> &consumer_inputs, uint64_t key) {
        std::array<uint32_t, 16> storage{};
        uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
        ChipTensor boundary = make_tensor_external(storage.data(), shape, 1);
        GraphTaskArgs boundary_args;
        boundary_args.add_input(boundary);

        orch.begin_scope();
        const GraphScopeResult begin = orch.graph_begin(key, boundary_args, key ^ 0x5a5a);
        EXPECT_TRUE(begin.recording);
        orch.graph_prepare(begin.recording_handle, boundary_args);

        // Each producer reads the previous producer's output: chain 0<-1<-2...
        std::vector<TaskOutputTensors> outputs;
        for (uint32_t p = 0; p < producer_count; ++p) {
            CoreTaskArgs args;
            if (p == 0) {
                args.add_input(boundary);
            } else {
                args.add_input(outputs[p - 1].get_ref(0));
            }
            TensorCreateInfo out(shape, 1, DataType::UINT32);
            args.add_output(out);
            const auto submitted = orch.submit_dummy_task(args);
            EXPECT_TRUE(submitted.task_id().is_valid());
            outputs.push_back(submitted);
        }

        CoreTaskArgs consumer_args;
        for (uint32_t input : consumer_inputs) {
            consumer_args.add_input(outputs[input].get_ref(0));
        }
        TensorCreateInfo consumer_out(shape, 1, DataType::UINT32);
        consumer_args.add_output(consumer_out);
        EXPECT_TRUE(orch.submit_dummy_task(consumer_args).task_id().is_valid());

        EXPECT_TRUE(orch.graph_end());
        orch.end_scope();

        // Each fixture test records exactly one graph, so the definition list
        // holds the one entry just built. full_key is a callable-hash mix the
        // caller never sees directly.
        const GraphHostDefinitionList definitions = graph_host_definitions(*graph_state);
        if (definitions.entries.size() != 1u) return nullptr;
        return reinterpret_cast<const GraphDefinition *>(definitions.entries[0].data);
    }

    static std::vector<uint16_t> fanin_row(const GraphDefinition &definition, uint32_t consumer) {
        const auto *offsets = reinterpret_cast<const uint32_t *>(
            reinterpret_cast<const std::byte *>(&definition) + definition.off_fanin_offsets
        );
        const auto *indices = reinterpret_cast<const uint16_t *>(
            reinterpret_cast<const std::byte *>(&definition) + definition.off_fanin_indices
        );
        return std::vector<uint16_t>(indices + offsets[consumer], indices + offsets[consumer + 1]);
    }
};

// Diamond: producers 0 -> 1 -> 2 (chain), consumer reads 0 and 2. The direct
// 0 -> consumer edge is implied by 0 -> 1 -> 2 -> consumer, so only 2 must
// remain in the consumer's row.
TEST_F(HbgGraphReductionTest, DiamondShortcutEdgeIsDropped) {
    const GraphDefinition *definition = record_chain_and_consumer(3, {0, 2}, 0xd1a);
    ASSERT_NE(definition, nullptr);
    ASSERT_EQ(definition->task_count, 4u);

    const std::vector<uint16_t> row = fanin_row(*definition, 3);
    ASSERT_EQ(row.size(), 1u);
    EXPECT_EQ(row[0], 2u);
}

// Same shape but the consumer reads 1 and 2: 1 -> 2 -> consumer implies the
// 1 -> consumer shortcut, so only the chain tail remains.
TEST_F(HbgGraphReductionTest, DeeperDiamondStillReducesToOneHop) {
    const GraphDefinition *definition = record_chain_and_consumer(3, {1, 2}, 0xd2);
    ASSERT_NE(definition, nullptr);
    const std::vector<uint16_t> row = fanin_row(*definition, 3);
    ASSERT_EQ(row.size(), 1u);
    EXPECT_EQ(row[0], 2u);
}

// Chain 0 -> 1 -> 2 -> 3, consumer reads 0 and 3: the shortcut spans three
// hops. The 1-hop TMR reducer cannot see this; the host reduction can.
TEST_F(HbgGraphReductionTest, ThreeHopShortcutIsDropped) {
    const GraphDefinition *definition = record_chain_and_consumer(4, {0, 3}, 0xd3);
    ASSERT_NE(definition, nullptr);
    ASSERT_EQ(definition->task_count, 5u);

    const std::vector<uint16_t> row = fanin_row(*definition, 4);
    ASSERT_EQ(row.size(), 1u);
    EXPECT_EQ(row[0], 3u);
}

// A consumer of a single chain tail has nothing redundant: its one edge stays.
TEST_F(HbgGraphReductionTest, IndependentProducersAreKept) {
    const GraphDefinition *definition = record_chain_and_consumer(4, {3}, 0xd4a);
    ASSERT_NE(definition, nullptr);
    EXPECT_EQ(fanin_row(*definition, 4).size(), 1u);
}

// The reduced image must still satisfy the CSR invariants bind_graph_topology
// enforces on the device: monotone offsets, edge_count consistency on both the
// fanin and fanout sides, indices strictly below their consumer, and roots
// exactly the zero-length rows.
TEST_F(HbgGraphReductionTest, ReducedImageHoldsCsrInvariants) {
    const GraphDefinition *definition = record_chain_and_consumer(4, {0, 3}, 0xd5);
    ASSERT_NE(definition, nullptr);

    const auto *bytes = reinterpret_cast<const std::byte *>(definition);
    const auto *fanin_offsets = reinterpret_cast<const uint32_t *>(bytes + definition->off_fanin_offsets);
    const auto *fanin_indices = reinterpret_cast<const uint16_t *>(bytes + definition->off_fanin_indices);
    const auto *fanout_offsets = reinterpret_cast<const uint32_t *>(bytes + definition->off_fanout_offsets);

    ASSERT_EQ(fanin_offsets[0], 0u);
    ASSERT_EQ(fanin_offsets[definition->task_count], definition->edge_count);
    ASSERT_EQ(fanout_offsets[0], 0u);
    ASSERT_EQ(fanout_offsets[definition->task_count], definition->edge_count);

    uint32_t observed_roots = 0;
    std::vector<uint32_t> fanout_counts(definition->task_count + 1, 0);
    for (uint32_t consumer = 0; consumer < definition->task_count; ++consumer) {
        const uint32_t begin = fanin_offsets[consumer];
        const uint32_t end = fanin_offsets[consumer + 1];
        ASSERT_LE(begin, end);
        ASSERT_LE(end, definition->edge_count);
        if (begin == end) observed_roots++;
        for (uint32_t edge = begin; edge < end; ++edge) {
            ASSERT_LT(fanin_indices[edge], consumer);
            fanout_counts[fanin_indices[edge]]++;
        }
    }
    EXPECT_EQ(observed_roots, definition->root_count);
    // The fanout side is derived data rebuilt from the reduced edges, so its
    // per-producer prefix sums must match the fanin-side counts exactly.
    for (uint32_t producer = 0; producer < definition->task_count; ++producer) {
        EXPECT_EQ(fanout_offsets[producer + 1] - fanout_offsets[producer], fanout_counts[producer])
            << "producer " << producer;
    }
}

// Roots (zero-fanin nodes) are unaffected by reduction: the chain head keeps
// its empty row and root_count stays 1.
TEST_F(HbgGraphReductionTest, RootCountUnchanged) {
    const GraphDefinition *definition = record_chain_and_consumer(4, {0, 3}, 0xd6);
    ASSERT_NE(definition, nullptr);
    EXPECT_EQ(definition->root_count, 1u);
    EXPECT_TRUE(fanin_row(*definition, 0).empty());
}

}  // namespace
