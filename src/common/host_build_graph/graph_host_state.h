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
#include <memory>
#include <optional>

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

GraphHostStatePtr make_graph_host_state();
size_t graph_host_upload_count(const GraphHostState &state);
std::optional<GraphHostUpload> graph_host_upload(GraphHostState &state, size_t index);

// Optional eager H2D hook: invoked right after each Graph POD image is appended
// during orch entry (compute-one-layer, copy-that-layer).
using GraphHostEagerUploadFn = bool (*)(void *ctx, GraphHostState &state, size_t index);
void graph_host_set_eager_upload(GraphHostEagerUploadFn fn, void *ctx);
bool graph_host_upload_h2d_done(const GraphHostState &state, size_t index);
void graph_host_mark_upload_h2d_done(GraphHostState &state, size_t index);

// Optional pinned bump arena for Graph POD images. Set by host runtime before
// orch entry so graph_submit_definition can write PODs in place. Not set →
// fallback std::vector images.
void graph_host_set_pinned_arena(std::byte *base, size_t cap);
void graph_host_clear_pinned_arena();
std::byte *graph_host_pinned_base();
size_t graph_host_pinned_used();
