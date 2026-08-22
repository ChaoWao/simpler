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

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "common/region_instance_semantics.h"

static constexpr uint32_t WORKER_CHIP_ORCH_COMM_MAGIC = 0x4C334C32u;  // "WorkerChip"
static constexpr uint16_t WORKER_CHIP_ORCH_COMM_ABI_MAJOR = 3;
static constexpr uint16_t WORKER_CHIP_ORCH_COMM_ABI_MINOR = 0;
static constexpr size_t WORKER_CHIP_ORCH_REGION_DESC_SCALAR_COUNT = 6;
static constexpr uint64_t WORKER_CHIP_ORCH_COMM_COUNTER_BYTES = sizeof(int32_t);
static constexpr uint64_t WORKER_CHIP_ORCH_COMM_COUNTER_BASE_ALIGNMENT = 64;

struct WorkerChipOrchRegionDesc {
    uint64_t magic_version;
    uint64_t region_id;
    uint64_t payload_base;
    uint64_t payload_bytes;
    uint64_t counter_base;
    uint64_t counter_bytes;
};

enum class WorkerChipOrchNotifyOp : uint32_t {
    Set = 0,
    Add = 1,
};

enum class WorkerChipOrchWaitCmp : uint32_t {
    EQ = 0,
    NE = 1,
    GT = 2,
    GE = 3,
    LT = 4,
    LE = 5,
};

struct WorkerChipOrchSignalTestResult {
    bool matched;
    int32_t observed;
};

enum class WorkerChipOrchCommValidationError : uint32_t {
    OK = 0,
    BAD_MAGIC_VERSION = 1,
    BAD_REGION_ID = 2,
    BAD_PAYLOAD_RANGE = 3,
    BAD_COUNTER_RANGE = 4,
    OUT_OF_BOUNDS = 5,
    BAD_SCALAR_COUNT = 6,
    NULL_POINTER = 7,
};

namespace worker_chip_orch_comm {

static inline void copy_error_message(char *dst, size_t dst_size, const char *message) {
    if (dst == nullptr || dst_size == 0) {
        return;
    }
    const char *src = message == nullptr ? "" : message;
    size_t n = strnlen(src, dst_size - 1);
    memcpy(dst, src, n);
    dst[n] = '\0';
}

}  // namespace worker_chip_orch_comm

static constexpr uint64_t worker_chip_orch_comm_pack_magic_version(uint32_t magic, uint16_t major, uint16_t minor) {
    return (static_cast<uint64_t>(magic) << 32) | (static_cast<uint64_t>(major) << 16) | static_cast<uint64_t>(minor);
}

static constexpr uint64_t WORKER_CHIP_ORCH_COMM_MAGIC_VERSION = worker_chip_orch_comm_pack_magic_version(
    WORKER_CHIP_ORCH_COMM_MAGIC, WORKER_CHIP_ORCH_COMM_ABI_MAJOR, WORKER_CHIP_ORCH_COMM_ABI_MINOR
);

static inline uint64_t worker_chip_orch_comm_magic_version() { return WORKER_CHIP_ORCH_COMM_MAGIC_VERSION; }

static inline uint32_t worker_chip_orch_comm_magic(uint64_t magic_version) {
    return static_cast<uint32_t>(magic_version >> 32);
}

static inline uint16_t worker_chip_orch_comm_abi_major(uint64_t magic_version) {
    return static_cast<uint16_t>((magic_version >> 16) & 0xFFFFu);
}

static inline uint16_t worker_chip_orch_comm_abi_minor(uint64_t magic_version) {
    return static_cast<uint16_t>(magic_version & 0xFFFFu);
}

static inline bool worker_chip_orch_comm_add_overflows(uint64_t a, uint64_t b) { return region_add_overflows(a, b); }

static inline uint64_t worker_chip_orch_comm_add_sat(uint64_t a, uint64_t b) {
#if defined(__clang__) || defined(__GNUC__)
    uint64_t result = 0;
    return __builtin_add_overflow(a, b, &result) ? UINT64_MAX : result;
#else
    return worker_chip_orch_comm_add_overflows(a, b) ? UINT64_MAX : a + b;
#endif
}

template <uint64_t Align>
static constexpr bool worker_chip_orch_comm_is_aligned(uint64_t value) {
    static_assert(Align > 0 && (Align & (Align - 1)) == 0, "Align must be a power of two");
    return (value & (Align - 1)) == 0;
}

static inline bool worker_chip_orch_comm_is_aligned_runtime(uint64_t value, uint64_t align) {
    return align != 0 && (align & (align - 1)) == 0 && (value & (align - 1)) == 0;
}

static inline bool worker_chip_orch_comm_range_contains(uint64_t base, uint64_t size, uint64_t value) {
    if (size == 0 || worker_chip_orch_comm_add_overflows(base, size)) {
        return false;
    }
    return value >= base && value - base < size;
}

static inline bool worker_chip_orch_comm_ranges_overlap(
    uint64_t first_base, uint64_t first_size, uint64_t second_base, uint64_t second_size
) {
    return region_spans_overlap(
        RegionPartLocalSpan{first_base, first_size}, RegionPartLocalSpan{second_base, second_size}
    );
}

static inline WorkerChipOrchCommValidationError
worker_chip_orch_comm_validate_payload_bounds(uint64_t offset, uint64_t nbytes, uint64_t payload_bytes) {
    if (nbytes == 0 || payload_bytes == 0) {
        return WorkerChipOrchCommValidationError::BAD_PAYLOAD_RANGE;
    }
    if (!region_validate_payload_range(offset, nbytes, payload_bytes)) {
        return WorkerChipOrchCommValidationError::OUT_OF_BOUNDS;
    }
    return WorkerChipOrchCommValidationError::OK;
}

static inline WorkerChipOrchCommValidationError
worker_chip_orch_comm_validate_counter_range(const WorkerChipOrchRegionDesc &desc) {
    RegionPartLocalSpan payload{desc.payload_base, desc.payload_bytes};
    RegionPartLocalSpan counter{desc.counter_base, desc.counter_bytes};
    if (!region_validate_counter_span(counter) || region_spans_overlap(payload, counter)) {
        return WorkerChipOrchCommValidationError::BAD_COUNTER_RANGE;
    }
    return WorkerChipOrchCommValidationError::OK;
}

static inline WorkerChipOrchCommValidationError
worker_chip_orch_comm_validate_desc(const WorkerChipOrchRegionDesc &desc) {
    if (worker_chip_orch_comm_magic(desc.magic_version) != WORKER_CHIP_ORCH_COMM_MAGIC ||
        worker_chip_orch_comm_abi_major(desc.magic_version) != WORKER_CHIP_ORCH_COMM_ABI_MAJOR) {
        return WorkerChipOrchCommValidationError::BAD_MAGIC_VERSION;
    }
    if (desc.region_id == 0) {
        return WorkerChipOrchCommValidationError::BAD_REGION_ID;
    }
    if (desc.payload_bytes == 0 || worker_chip_orch_comm_add_overflows(desc.payload_base, desc.payload_bytes)) {
        return WorkerChipOrchCommValidationError::BAD_PAYLOAD_RANGE;
    }
    WorkerChipOrchCommValidationError counter_error = worker_chip_orch_comm_validate_counter_range(desc);
    if (counter_error != WorkerChipOrchCommValidationError::OK) {
        return counter_error;
    }
    return WorkerChipOrchCommValidationError::OK;
}

static inline bool
worker_chip_orch_comm_encode_desc(const WorkerChipOrchRegionDesc &desc, uint64_t *scalars, size_t scalar_count) {
    if (scalars == nullptr || scalar_count < WORKER_CHIP_ORCH_REGION_DESC_SCALAR_COUNT) {
        return false;
    }
    scalars[0] = desc.magic_version;
    scalars[1] = desc.region_id;
    scalars[2] = desc.payload_base;
    scalars[3] = desc.payload_bytes;
    scalars[4] = desc.counter_base;
    scalars[5] = desc.counter_bytes;
    return true;
}

static inline bool worker_chip_orch_comm_decode_desc(
    const uint64_t *scalars, size_t scalar_count, WorkerChipOrchRegionDesc *out_desc,
    WorkerChipOrchCommValidationError *out_error
) {
    if (out_error != nullptr) {
        *out_error = WorkerChipOrchCommValidationError::OK;
    }
    if (scalars == nullptr || out_desc == nullptr) {
        if (out_error != nullptr) {
            *out_error = WorkerChipOrchCommValidationError::NULL_POINTER;
        }
        return false;
    }
    if (scalar_count < WORKER_CHIP_ORCH_REGION_DESC_SCALAR_COUNT) {
        if (out_error != nullptr) {
            *out_error = WorkerChipOrchCommValidationError::BAD_SCALAR_COUNT;
        }
        return false;
    }
    *out_desc = WorkerChipOrchRegionDesc{
        scalars[0], scalars[1], scalars[2], scalars[3], scalars[4], scalars[5],
    };
    WorkerChipOrchCommValidationError error = worker_chip_orch_comm_validate_desc(*out_desc);
    if (out_error != nullptr) {
        *out_error = error;
    }
    return error == WorkerChipOrchCommValidationError::OK;
}

static inline bool worker_chip_orch_comm_as_region_notify_op(WorkerChipOrchNotifyOp op, RegionNotifyOp &out) {
    switch (op) {
    case WorkerChipOrchNotifyOp::Set:
        out = RegionNotifyOp::Set;
        return true;
    case WorkerChipOrchNotifyOp::Add:
        out = RegionNotifyOp::Add;
        return true;
    }
    return false;
}

static inline bool worker_chip_orch_comm_as_region_wait_cmp(WorkerChipOrchWaitCmp cmp, RegionWaitCmp &out) {
    switch (cmp) {
    case WorkerChipOrchWaitCmp::EQ:
        out = RegionWaitCmp::EQ;
        return true;
    case WorkerChipOrchWaitCmp::NE:
        out = RegionWaitCmp::NE;
        return true;
    case WorkerChipOrchWaitCmp::GT:
        out = RegionWaitCmp::GT;
        return true;
    case WorkerChipOrchWaitCmp::GE:
        out = RegionWaitCmp::GE;
        return true;
    case WorkerChipOrchWaitCmp::LT:
        out = RegionWaitCmp::LT;
        return true;
    case WorkerChipOrchWaitCmp::LE:
        out = RegionWaitCmp::LE;
        return true;
    }
    return false;
}

static inline bool worker_chip_orch_comm_valid_notify_op(WorkerChipOrchNotifyOp op) {
    RegionNotifyOp converted{};
    return worker_chip_orch_comm_as_region_notify_op(op, converted) && region_valid_notify_op(converted);
}

static inline bool worker_chip_orch_comm_valid_wait_cmp(WorkerChipOrchWaitCmp cmp) {
    RegionWaitCmp converted{};
    return worker_chip_orch_comm_as_region_wait_cmp(cmp, converted) && region_valid_wait_cmp(converted);
}

static inline bool
worker_chip_orch_comm_compare_counter(int32_t observed, int32_t cmp_value, WorkerChipOrchWaitCmp cmp) {
    RegionWaitCmp converted{};
    if (!worker_chip_orch_comm_as_region_wait_cmp(cmp, converted)) {
        return false;
    }
    return region_compare_counter(observed, cmp_value, converted);
}

static inline WorkerChipOrchCommValidationError
worker_chip_orch_comm_validate_counter_addr(const WorkerChipOrchRegionDesc &desc, uint64_t counter_addr) {
    if (!worker_chip_orch_comm_is_aligned<WORKER_CHIP_ORCH_COMM_COUNTER_BYTES>(counter_addr)) {
        return WorkerChipOrchCommValidationError::BAD_COUNTER_RANGE;
    }
    if (worker_chip_orch_comm_validate_counter_range(desc) != WorkerChipOrchCommValidationError::OK) {
        return WorkerChipOrchCommValidationError::BAD_COUNTER_RANGE;
    }
    if (!region_counter_addr_in_span(RegionPartLocalSpan{desc.counter_base, desc.counter_bytes}, counter_addr)) {
        return WorkerChipOrchCommValidationError::OUT_OF_BOUNDS;
    }
    return WorkerChipOrchCommValidationError::OK;
}

namespace worker_chip_orch_comm {

static constexpr uint64_t pack_magic_version(uint32_t magic, uint16_t major, uint16_t minor) {
    return ::worker_chip_orch_comm_pack_magic_version(magic, major, minor);
}

static inline uint64_t magic_version() { return ::worker_chip_orch_comm_magic_version(); }

static inline uint32_t magic(uint64_t magic_version_value) {
    return ::worker_chip_orch_comm_magic(magic_version_value);
}

static inline uint16_t abi_major(uint64_t magic_version_value) {
    return ::worker_chip_orch_comm_abi_major(magic_version_value);
}

static inline uint16_t abi_minor(uint64_t magic_version_value) {
    return ::worker_chip_orch_comm_abi_minor(magic_version_value);
}

static inline bool add_overflows(uint64_t a, uint64_t b) { return ::worker_chip_orch_comm_add_overflows(a, b); }

static inline uint64_t add_sat(uint64_t a, uint64_t b) { return ::worker_chip_orch_comm_add_sat(a, b); }

template <uint64_t Align>
static constexpr bool is_aligned(uint64_t value) {
    return ::worker_chip_orch_comm_is_aligned<Align>(value);
}

static inline bool is_aligned_runtime(uint64_t value, uint64_t align) {
    return ::worker_chip_orch_comm_is_aligned_runtime(value, align);
}

static inline bool range_contains(uint64_t base, uint64_t size, uint64_t value) {
    return ::worker_chip_orch_comm_range_contains(base, size, value);
}

static inline bool
ranges_overlap(uint64_t first_base, uint64_t first_size, uint64_t second_base, uint64_t second_size) {
    return ::worker_chip_orch_comm_ranges_overlap(first_base, first_size, second_base, second_size);
}

static inline WorkerChipOrchCommValidationError
validate_payload_bounds(uint64_t offset, uint64_t nbytes, uint64_t payload_bytes) {
    return ::worker_chip_orch_comm_validate_payload_bounds(offset, nbytes, payload_bytes);
}

static inline WorkerChipOrchCommValidationError validate_counter_range(const WorkerChipOrchRegionDesc &desc) {
    return ::worker_chip_orch_comm_validate_counter_range(desc);
}

static inline WorkerChipOrchCommValidationError validate_desc(const WorkerChipOrchRegionDesc &desc) {
    return ::worker_chip_orch_comm_validate_desc(desc);
}

static inline bool encode_desc(const WorkerChipOrchRegionDesc &desc, uint64_t *scalars, size_t scalar_count) {
    return ::worker_chip_orch_comm_encode_desc(desc, scalars, scalar_count);
}

static inline bool decode_desc(
    const uint64_t *scalars, size_t scalar_count, WorkerChipOrchRegionDesc *out_desc,
    WorkerChipOrchCommValidationError *out_error
) {
    return ::worker_chip_orch_comm_decode_desc(scalars, scalar_count, out_desc, out_error);
}

static inline bool valid_notify_op(WorkerChipOrchNotifyOp op) { return ::worker_chip_orch_comm_valid_notify_op(op); }

static inline bool valid_wait_cmp(WorkerChipOrchWaitCmp cmp) { return ::worker_chip_orch_comm_valid_wait_cmp(cmp); }

static inline bool compare_counter(int32_t observed, int32_t cmp_value, WorkerChipOrchWaitCmp cmp) {
    return ::worker_chip_orch_comm_compare_counter(observed, cmp_value, cmp);
}

static inline WorkerChipOrchCommValidationError
validate_counter_addr(const WorkerChipOrchRegionDesc &desc, uint64_t counter_addr) {
    return ::worker_chip_orch_comm_validate_counter_addr(desc, counter_addr);
}

}  // namespace worker_chip_orch_comm
