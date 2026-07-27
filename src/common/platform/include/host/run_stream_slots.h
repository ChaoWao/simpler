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

#ifndef SRC_COMMON_PLATFORM_INCLUDE_HOST_RUN_STREAM_SLOTS_H_
#define SRC_COMMON_PLATFORM_INCLUDE_HOST_RUN_STREAM_SLOTS_H_

#include <array>
#include <cstddef>
#include <functional>

#include "pto_runtime_c_api.h"

/**
 * Per-slot ownership of the two streams a run submits on.
 *
 * The AICPU stream carries no instruction-cache state, so it belongs to its
 * slot for the runner's lifetime. The AICore stream belongs to a single run:
 * the platform offers no instruction-cache invalidation for code replaced at a
 * reused GM address, and creating the stream is the only operation known to
 * leave a core free of the previous image's instructions — selecting an
 * existing one is not.
 *
 * A destroy that fails keeps its handle. Such a stream may still hold the
 * previous image's instructions, so the slot must refuse the next run rather
 * than hand it a fresh stream beside a live one, and teardown needs the handle
 * to retry. Stream creation and destruction are injected so this state machine
 * is exercisable without a device.
 */
class RunStreamSlots {
public:
    using CreateFn = std::function<int(void **out_stream)>;
    using DestroyFn = std::function<int(void *stream)>;

    RunStreamSlots(CreateFn create, DestroyFn destroy) :
        create_(std::move(create)),
        destroy_(std::move(destroy)) {}

    /**
     * Ready `slot` for a run: its AICPU stream on first use, and always a fresh
     * AICore stream. Fails when the slot still holds an AICore stream a prior
     * run could not retire.
     */
    int acquire(unsigned slot) {
        if (slot >= slots_.size()) return -1;
        Slot &s = slots_[slot];
        if (s.aicpu == nullptr) {
            int rc = create_(&s.aicpu);
            if (rc != 0) {
                s.aicpu = nullptr;
                return rc;
            }
        }
        if (s.aicore != nullptr) return -1;
        int rc = create_(&s.aicore);
        if (rc != 0) {
            s.aicore = nullptr;
            return rc;
        }
        ++created_count_;
        return 0;
    }

    /** Retire `slot`'s AICore stream. The handle survives a failed destroy. */
    int retire_aicore(unsigned slot) {
        if (slot >= slots_.size()) return -1;
        Slot &s = slots_[slot];
        if (s.aicore == nullptr) return 0;
        int rc = destroy_(s.aicore);
        if (rc != 0) return rc;
        s.aicore = nullptr;
        return 0;
    }

    /** Destroy every stream, keeping handles whose destroy failed. */
    int destroy_all() {
        int first_error = 0;
        for (Slot &s : slots_) {
            for (void **stream : {&s.aicpu, &s.aicore}) {
                if (*stream == nullptr) continue;
                int rc = destroy_(*stream);
                if (rc != 0) {
                    if (first_error == 0) first_error = rc;
                    continue;
                }
                *stream = nullptr;
            }
        }
        return first_error;
    }

    void *aicpu(unsigned slot) const { return slot < slots_.size() ? slots_[slot].aicpu : nullptr; }
    void *aicore(unsigned slot) const { return slot < slots_.size() ? slots_[slot].aicore : nullptr; }
    bool ready(unsigned slot) const { return aicpu(slot) != nullptr && aicore(slot) != nullptr; }
    size_t created_count() const { return created_count_; }
    static constexpr size_t capacity() { return PTO_PIPELINE_MAX_DEPTH; }

private:
    struct Slot {
        void *aicpu{nullptr};
        void *aicore{nullptr};
    };

    CreateFn create_;
    DestroyFn destroy_;
    std::array<Slot, PTO_PIPELINE_MAX_DEPTH> slots_{};
    size_t created_count_{0};
};

#endif  // SRC_COMMON_PLATFORM_INCLUDE_HOST_RUN_STREAM_SLOTS_H_
