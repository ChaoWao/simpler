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

#include <limits.h>
#include <stdint.h>

static constexpr uint64_t REGION_COUNTER_LOGICAL_ALIGNMENT = 4;
static constexpr uint64_t REGION_COUNTER_BASE_ALIGNMENT = 64;

struct RegionPartLocalSpan {
    uint64_t base;
    uint64_t logical_bytes;
};

enum class RegionNotifyOp : uint32_t {
    Set = 0,
    Add = 1,
};

enum class RegionWaitCmp : uint32_t {
    EQ = 0,
    NE = 1,
    GT = 2,
    GE = 3,
    LT = 4,
    LE = 5,
};

inline bool region_add_overflows(uint64_t a, uint64_t b) {
#if defined(__clang__) || defined(__GNUC__)
    uint64_t result = 0;
    return __builtin_add_overflow(a, b, &result);
#else
    return a > UINT64_MAX - b;
#endif
}

inline bool region_spans_overlap(RegionPartLocalSpan first, RegionPartLocalSpan second) {
    if (first.logical_bytes == 0 || second.logical_bytes == 0 ||
        region_add_overflows(first.base, first.logical_bytes) ||
        region_add_overflows(second.base, second.logical_bytes)) {
        return false;
    }
    if (first.base < second.base) {
        return second.base - first.base < first.logical_bytes;
    }
    return first.base - second.base < second.logical_bytes;
}

inline bool region_valid_notify_op(RegionNotifyOp op) { return op == RegionNotifyOp::Set || op == RegionNotifyOp::Add; }

inline bool region_valid_wait_cmp(RegionWaitCmp cmp) {
    return cmp == RegionWaitCmp::EQ || cmp == RegionWaitCmp::NE || cmp == RegionWaitCmp::GT ||
           cmp == RegionWaitCmp::GE || cmp == RegionWaitCmp::LT || cmp == RegionWaitCmp::LE;
}

inline bool region_compare_counter(int32_t observed, int32_t cmp_value, RegionWaitCmp cmp) {
    switch (cmp) {
    case RegionWaitCmp::EQ:
        return observed == cmp_value;
    case RegionWaitCmp::NE:
        return observed != cmp_value;
    case RegionWaitCmp::GT:
        return observed > cmp_value;
    case RegionWaitCmp::GE:
        return observed >= cmp_value;
    case RegionWaitCmp::LT:
        return observed < cmp_value;
    case RegionWaitCmp::LE:
        return observed <= cmp_value;
    }
    return false;
}

inline bool region_validate_payload_span(RegionPartLocalSpan span) {
    return span.logical_bytes > 0 && !region_add_overflows(span.base, span.logical_bytes);
}

inline bool region_validate_payload_range(uint64_t offset, uint64_t nbytes, uint64_t logical_bytes) {
    return nbytes > 0 && logical_bytes > 0 && !region_add_overflows(offset, nbytes) && offset + nbytes <= logical_bytes;
}

inline bool region_validate_counter_span(RegionPartLocalSpan span) {
    return span.logical_bytes > 0 && (span.logical_bytes % REGION_COUNTER_LOGICAL_ALIGNMENT) == 0 &&
           (span.base % REGION_COUNTER_BASE_ALIGNMENT) == 0 && !region_add_overflows(span.base, span.logical_bytes);
}

inline bool region_validate_independent_spans(RegionPartLocalSpan payload, RegionPartLocalSpan counter) {
    return region_validate_payload_span(payload) && region_validate_counter_span(counter) &&
           !region_spans_overlap(payload, counter);
}

inline bool region_counter_addr_in_span(RegionPartLocalSpan span, uint64_t counter_addr) {
    if ((counter_addr % REGION_COUNTER_LOGICAL_ALIGNMENT) != 0 ||
        span.logical_bytes < REGION_COUNTER_LOGICAL_ALIGNMENT) {
        return false;
    }
    if (counter_addr < span.base) {
        return false;
    }
    uint64_t max_addr = span.base + span.logical_bytes - REGION_COUNTER_LOGICAL_ALIGNMENT;
    return counter_addr <= max_addr;
}
