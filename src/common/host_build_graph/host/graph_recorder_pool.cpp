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
#include "host_build_graph/graph_recorder_pool.h"

#include <functional>
#include <utility>

#include "graph_host_state.h"
// The ops implementations below need the real RuntimeOps/RuntimeContext, which is why the
// header does not include this: a translation unit cannot see both it and
// orchestration_api.h's partial declaration.
#include "host_build_graph/runtime_core.h"

GraphAsyncRecordingState &graph_recorder_pool() {
    static GraphAsyncRecordingState state;
    return state;
}

bool graph_recorder_prewarm() { return graph_recorder_pool().prewarm(); }

// Strong overrides of the weak fallbacks in runtime_core.cpp, which the AICPU build
// links instead: there is no recorder pool on device, and its fallback refuses so the
// caller records synchronously.
//
// `job` points to the caller's std::function. The pool moves the closure out of it
// whether or not it queues it -- start() takes the callable before it checks capacity --
// so the caller must treat it as spent on return. No ownership crosses the .so boundary
// either way: the caller's std::function destructs on its own side.
bool graph_record_start_impl(RuntimeContext *, const GraphTaskArgs &args, void *job) {
    if (job == nullptr) return false;
    auto *record = static_cast<std::function<void(GraphTaskArgs &)> *>(job);
    return graph_recorder_pool().start(args, std::move(*record));
}

void graph_record_wait_impl(RuntimeContext *) { graph_recorder_pool().wait(); }
