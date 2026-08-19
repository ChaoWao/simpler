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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

struct PTO2TaskSlotState;
struct GraphHostState;

inline constexpr size_t GRAPH_MAX_DEFINITIONS = 16;

struct GraphHostStateDeleter {
    void operator()(GraphHostState *state) const noexcept;
};

using GraphHostStatePtr = std::unique_ptr<GraphHostState, GraphHostStateDeleter>;

struct GraphHostUpload {
    PTO2TaskSlotState *outer_slot;
    std::byte *data;
    size_t bytes;
};

// The run's distinct Definition images (already deduplicated by the host-side
// Definition cache), for upload as shared device objects ahead of submissions.
struct GraphHostDefinition {
    uint64_t full_key;
    const std::byte *data;
    size_t bytes;
};

struct GraphHostDefinitionList {
    std::vector<GraphHostDefinition> entries;
};

GraphHostStatePtr make_graph_host_state();
size_t graph_host_upload_count(const GraphHostState &state);
std::optional<GraphHostUpload> graph_host_upload(GraphHostState &state, size_t index);
bool graph_host_upload_h2d_done(const GraphHostState &state, size_t index);
void graph_host_mark_upload_h2d_done(GraphHostState &state, size_t index);
GraphHostDefinitionList graph_host_definitions(GraphHostState &state);

// Optional pinned bump arena for Graph submission POD images. Set by the host
// runtime before orch entry so graph_submit_definition can write each POD in
// place; unset means the fallback std::vector images. The base handed to
// graph_host_set_pinned_arena must satisfy kGraphPinnedBumpAlign — the
// orchestrator aligns offsets relative to it without re-aligning the base.
inline constexpr size_t kGraphPinnedBumpAlign = 64;
void graph_host_set_pinned_arena(std::byte *base, size_t cap);
void graph_host_clear_pinned_arena();
std::byte *graph_host_pinned_base();
size_t graph_host_pinned_used();
