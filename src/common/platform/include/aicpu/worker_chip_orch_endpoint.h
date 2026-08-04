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

#include "aicpu/cache_maintenance.h"
#include "aicpu/device_time.h"
#include "common/worker_chip_orch_comm.h"

struct WorkerChipOrchPayloadView {
    uint64_t gm_addr;
    uint64_t nbytes;
};

enum class WorkerChipEndpointErrorKind : uint32_t {
    NONE = 0,
    BAD_DESCRIPTOR = 1,
    OUT_OF_BOUNDS = 2,
    SIGNAL_TIMEOUT = 3,
    SIGNAL_PROTOCOL = 4,
};

enum class WorkerChipEndpointOp : uint32_t {
    INIT = 1,
    COUNTER_ADDR = 2,
    PAYLOAD_READ = 3,
    PAYLOAD_WRITE = 4,
    SIGNAL_NOTIFY = 5,
    SIGNAL_TEST = 6,
    SIGNAL_WAIT = 7,
};

inline const char *worker_chip_endpoint_op_to_string(WorkerChipEndpointOp op) {
    switch (op) {
    case WorkerChipEndpointOp::INIT:
        return "init";
    case WorkerChipEndpointOp::COUNTER_ADDR:
        return "counter_addr";
    case WorkerChipEndpointOp::PAYLOAD_READ:
        return "payload_read";
    case WorkerChipEndpointOp::PAYLOAD_WRITE:
        return "payload_write";
    case WorkerChipEndpointOp::SIGNAL_NOTIFY:
        return "signal_notify";
    case WorkerChipEndpointOp::SIGNAL_TEST:
        return "signal_test";
    case WorkerChipEndpointOp::SIGNAL_WAIT:
        return "signal_wait";
    default:
        return "unknown";
    }
}

struct WorkerChipEndpointError {
    WorkerChipEndpointErrorKind kind;
    WorkerChipEndpointOp op;
    uint64_t region_id;
    uint64_t counter_addr;
    int32_t counter_operand;
    int32_t observed_counter;
    char message[256];
};

class WorkerChipOrchEndpoint {
public:
    explicit WorkerChipOrchEndpoint(const WorkerChipOrchRegionDesc &desc) :
        desc_(desc) {
        if (worker_chip_orch_comm::validate_desc(desc_) != WorkerChipOrchCommValidationError::OK) {
            set_error(
                WorkerChipEndpointErrorKind::BAD_DESCRIPTOR, WorkerChipEndpointOp::INIT, desc_.region_id, 0, 0,
                "invalid descriptor"
            );
        }
    }

    WorkerChipOrchEndpoint(const uint64_t *scalars, size_t scalar_count) {
        WorkerChipOrchCommValidationError error = WorkerChipOrchCommValidationError::OK;
        if (!worker_chip_orch_comm::decode_desc(scalars, scalar_count, &desc_, &error)) {
            uint64_t region_id = scalar_count > 1 && scalars != nullptr ? scalars[1] : 0;
            set_error(
                WorkerChipEndpointErrorKind::BAD_DESCRIPTOR, WorkerChipEndpointOp::INIT, region_id, 0, 0,
                "invalid descriptor scalars"
            );
        }
    }

    const WorkerChipEndpointError &error() const { return error_; }

    const WorkerChipOrchRegionDesc &descriptor() const { return desc_; }

    bool counter_addr(uint64_t offset, uint64_t &out_addr) {
        out_addr = 0;
        if (has_error()) {
            return false;
        }
        if (worker_chip_orch_comm_add_overflows(desc_.counter_base, offset)) {
            set_error(
                WorkerChipEndpointErrorKind::OUT_OF_BOUNDS, WorkerChipEndpointOp::COUNTER_ADDR, desc_.region_id, 0, 0,
                "counter offset is out of bounds"
            );
            return false;
        }
        uint64_t addr = desc_.counter_base + offset;
        if (!validate_counter_addr_for_op(
                WorkerChipEndpointOp::COUNTER_ADDR, addr, 0, 0, "counter offset is out of bounds"
            )) {
            return false;
        }
        out_addr = addr;
        return true;
    }

    bool validate_counter_addr(uint64_t counter_addr) const {
        return worker_chip_orch_comm::validate_counter_addr(desc_, counter_addr) ==
               WorkerChipOrchCommValidationError::OK;
    }

    bool payload_read(uint64_t offset, uint64_t nbytes, WorkerChipOrchPayloadView &out) {
        out = WorkerChipOrchPayloadView{0, 0};
        if (has_error()) {
            return false;
        }
        if (!validate_payload_range(WorkerChipEndpointOp::PAYLOAD_READ, offset, nbytes)) {
            return false;
        }
        uint64_t gm_addr = desc_.payload_base + offset;
        cache_invalidate_range(
            reinterpret_cast<const void *>(static_cast<uintptr_t>(gm_addr)), static_cast<size_t>(nbytes)
        );
        out = WorkerChipOrchPayloadView{gm_addr, nbytes};
        return true;
    }

    bool payload_write(uint64_t offset, const void *src, uint64_t nbytes) {
        if (has_error()) {
            return false;
        }
        if (src == nullptr) {
            set_error(
                WorkerChipEndpointErrorKind::OUT_OF_BOUNDS, WorkerChipEndpointOp::PAYLOAD_WRITE, desc_.region_id, 0, 0,
                "null payload source"
            );
            return false;
        }
        if (!validate_payload_range(WorkerChipEndpointOp::PAYLOAD_WRITE, offset, nbytes)) {
            return false;
        }
        void *dst = reinterpret_cast<void *>(static_cast<uintptr_t>(desc_.payload_base + offset));
        memcpy(dst, src, static_cast<size_t>(nbytes));
        cache_flush_range(dst, static_cast<size_t>(nbytes));
        return true;
    }

    bool signal_notify(uint64_t counter_addr, int32_t value, WorkerChipOrchNotifyOp op) {
        if (has_error()) {
            return false;
        }
        if (!validate_counter_addr_for_op(
                WorkerChipEndpointOp::SIGNAL_NOTIFY, counter_addr, value, 0, "invalid counter address"
            )) {
            return false;
        }
        if (!worker_chip_orch_comm::valid_notify_op(op)) {
            set_error(
                WorkerChipEndpointErrorKind::SIGNAL_PROTOCOL, WorkerChipEndpointOp::SIGNAL_NOTIFY, desc_.region_id,
                counter_addr, value, "invalid notify operation"
            );
            return false;
        }

        volatile int32_t *counter = counter_ptr(counter_addr);
        if (op == WorkerChipOrchNotifyOp::Set) {
            *counter = value;
        } else {
            cache_invalidate_range(
                reinterpret_cast<const void *>(static_cast<uintptr_t>(counter_addr)), sizeof(*counter)
            );
            *counter = static_cast<int32_t>(*counter + value);
        }
        cache_flush_range(reinterpret_cast<const void *>(static_cast<uintptr_t>(counter_addr)), sizeof(*counter));
        return true;
    }

    bool signal_test(
        uint64_t counter_addr, int32_t cmp_value, WorkerChipOrchWaitCmp cmp, WorkerChipOrchSignalTestResult &out
    ) {
        out = WorkerChipOrchSignalTestResult{false, 0};
        if (has_error()) {
            return false;
        }
        if (!validate_counter_addr_for_op(
                WorkerChipEndpointOp::SIGNAL_TEST, counter_addr, cmp_value, 0, "invalid counter address"
            )) {
            return false;
        }
        if (!worker_chip_orch_comm::valid_wait_cmp(cmp)) {
            set_error(
                WorkerChipEndpointErrorKind::SIGNAL_PROTOCOL, WorkerChipEndpointOp::SIGNAL_TEST, desc_.region_id,
                counter_addr, cmp_value, "invalid wait comparison"
            );
            return false;
        }
        int32_t observed = load_counter(counter_addr);
        out =
            WorkerChipOrchSignalTestResult{worker_chip_orch_comm::compare_counter(observed, cmp_value, cmp), observed};
        return true;
    }

    bool signal_wait(
        uint64_t counter_addr, int32_t cmp_value, WorkerChipOrchWaitCmp cmp, uint64_t timeout, int32_t &observed
    ) {
        observed = 0;
        if (has_error()) {
            return false;
        }
        if (!validate_counter_addr_for_op(
                WorkerChipEndpointOp::SIGNAL_WAIT, counter_addr, cmp_value, 0, "invalid counter address"
            )) {
            return false;
        }
        if (!worker_chip_orch_comm::valid_wait_cmp(cmp)) {
            set_error(
                WorkerChipEndpointErrorKind::SIGNAL_PROTOCOL, WorkerChipEndpointOp::SIGNAL_WAIT, desc_.region_id,
                counter_addr, cmp_value, "invalid wait comparison"
            );
            return false;
        }

        uint64_t start = device_time_now_ticks();
        uint64_t frequency_hz = device_time_frequency_hz();
        while (true) {
            int32_t current = load_counter(counter_addr);
            observed = current;
            if (worker_chip_orch_comm::compare_counter(current, cmp_value, cmp)) {
                return true;
            }
            uint64_t now = device_time_now_ticks();
            if (timeout == 0 || sys_cnt_elapsed_ns(start, now, frequency_hz) >= timeout) {
                set_error(
                    WorkerChipEndpointErrorKind::SIGNAL_TIMEOUT, WorkerChipEndpointOp::SIGNAL_WAIT, desc_.region_id,
                    counter_addr, cmp_value, current, "wait timed out"
                );
                return false;
            }
        }
    }

private:
    bool has_error() const { return error_.kind != WorkerChipEndpointErrorKind::NONE; }

    bool validate_payload_range(WorkerChipEndpointOp op, uint64_t offset, uint64_t nbytes) {
        WorkerChipOrchCommValidationError error =
            worker_chip_orch_comm::validate_payload_bounds(offset, nbytes, desc_.payload_bytes);
        if (error == WorkerChipOrchCommValidationError::OK) {
            return true;
        }
        set_error(
            WorkerChipEndpointErrorKind::OUT_OF_BOUNDS, op, desc_.region_id, 0, 0, "payload range is out of bounds"
        );
        return false;
    }

    bool validate_counter_addr_for_op(
        WorkerChipEndpointOp op, uint64_t counter_addr, int32_t counter_operand, int32_t observed_counter,
        const char *message
    ) {
        if (worker_chip_orch_comm::validate_counter_addr(desc_, counter_addr) ==
            WorkerChipOrchCommValidationError::OK) {
            return true;
        }
        set_error(
            WorkerChipEndpointErrorKind::OUT_OF_BOUNDS, op, desc_.region_id, counter_addr, counter_operand,
            observed_counter, message
        );
        return false;
    }

    static volatile int32_t *counter_ptr(uint64_t counter_addr) {
        return reinterpret_cast<volatile int32_t *>(static_cast<uintptr_t>(counter_addr));
    }

    static int32_t load_counter(uint64_t counter_addr) {
        volatile int32_t *counter = counter_ptr(counter_addr);
        cache_invalidate_range(reinterpret_cast<const void *>(static_cast<uintptr_t>(counter_addr)), sizeof(*counter));
        return *counter;
    }

    void set_error(
        WorkerChipEndpointErrorKind kind, WorkerChipEndpointOp op, uint64_t region_id, uint64_t counter_addr,
        int32_t counter_operand, const char *message
    ) {
        set_error(kind, op, region_id, counter_addr, counter_operand, 0, message);
    }

    void set_error(
        WorkerChipEndpointErrorKind kind, WorkerChipEndpointOp op, uint64_t region_id, uint64_t counter_addr,
        int32_t counter_operand, int32_t observed_counter, const char *message
    ) {
        if (has_error()) {
            return;
        }
        error_ = WorkerChipEndpointError{kind, op, region_id, counter_addr, counter_operand, observed_counter, ""};
        worker_chip_orch_comm::copy_error_message(error_.message, sizeof(error_.message), message);
    }

    WorkerChipOrchRegionDesc desc_{};
    WorkerChipEndpointError error_{WorkerChipEndpointErrorKind::NONE, WorkerChipEndpointOp::INIT, 0, 0, 0, 0, ""};
};
