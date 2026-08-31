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

#include <stdint.h>
#include <cstddef>

/**
 * A pointer to a sibling in the same contiguous block, stored as a byte delta
 * from the field's own address.
 *
 * The block is `memcpy`'d to the device as one image, so a delta between two
 * addresses inside it is invariant under the move while a raw pointer is not.
 * Every user has to satisfy that precondition: a task payload's argument regions
 * live in the same image as the payload itself, in the pools past the task array.
 *
 * A zero delta means unbound — a field can never coincide with its own target.
 * Zeroed memory therefore reads as null, which is what the slot's pristine state
 * is.
 *
 * int32_t bounds the distance at 2 GiB, and set() leaves the field unbound rather
 * than storing a truncated delta, so an out-of-range target reads back as null
 * instead of as unrelated memory. A block whose extent can approach the bound
 * therefore has to reject it up front: attach_populated does so for the shipped
 * shared-memory image.
 */
namespace simpler::hbg {

template <typename T>
class SelfRelativePtr {
public:
    SelfRelativePtr() = default;

    // The stored value means something only relative to where it is stored, so
    // copying it from another field would silently retarget it. Every write goes
    // through set(), which takes the destination's own address into account. A
    // whole-block memcpy is unaffected: it moves the field and its target
    // together, which is the case this representation exists for.
    SelfRelativePtr(const SelfRelativePtr &) = delete;
    SelfRelativePtr &operator=(const SelfRelativePtr &) = delete;

    T *get() const {
        if (delta_ == 0) return nullptr;
        return reinterpret_cast<T *>(reinterpret_cast<uintptr_t>(this) + static_cast<intptr_t>(delta_));
    }

    void set(T *target) {
        if (target == nullptr) {
            delta_ = 0;
            return;
        }
        const intptr_t delta = reinterpret_cast<intptr_t>(target) - reinterpret_cast<intptr_t>(this);
        // Narrowing a delta this large would name unrelated memory that a consumer
        // would then dereference. Unbound is the one value every consumer already
        // tests for.
        if (delta < INT32_MIN || delta > INT32_MAX) {
            delta_ = 0;
            return;
        }
        delta_ = static_cast<int32_t>(delta);
    }

    T *operator->() const { return get(); }
    T &operator*() const { return *get(); }
    explicit operator bool() const { return delta_ != 0; }

    friend bool operator==(const SelfRelativePtr &lhs, std::nullptr_t) { return lhs.delta_ == 0; }
    friend bool operator!=(const SelfRelativePtr &lhs, std::nullptr_t) { return lhs.delta_ != 0; }

private:
    int32_t delta_{0};
};

}  // namespace simpler::hbg
