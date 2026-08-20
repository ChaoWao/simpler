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

#include "aicpu/region_instance_view.h"
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
        install_view_or_error();
    }

    WorkerChipOrchEndpoint(const uint64_t *scalars, size_t scalar_count) {
        WorkerChipOrchCommValidationError error = WorkerChipOrchCommValidationError::OK;
        if (!worker_chip_orch_comm::decode_desc(scalars, scalar_count, &desc_, &error)) {
            uint64_t region_id = scalar_count > 1 && scalars != nullptr ? scalars[1] : 0;
            set_error(
                WorkerChipEndpointErrorKind::BAD_DESCRIPTOR, WorkerChipEndpointOp::INIT, region_id, 0, 0,
                "invalid descriptor scalars"
            );
            return;
        }
        install_view_or_error();
    }

    const WorkerChipEndpointError &error() const { return error_; }

    const WorkerChipOrchRegionDesc &descriptor() const { return desc_; }

    const RegionInstanceView &view() const { return view_; }

    bool counter_addr(uint64_t offset, uint64_t &out_addr) {
        out_addr = 0;
        if (has_error()) {
            return false;
        }
        uint64_t reported = 0;
        if (!worker_chip_orch_comm_add_overflows(desc_.counter_base, offset)) {
            reported = desc_.counter_base + offset;
        }
        if (!view_.counter_addr(offset, out_addr)) {
            adopt_view_error(WorkerChipEndpointOp::COUNTER_ADDR, reported, 0);
            return false;
        }
        return true;
    }

    bool validate_counter_addr(uint64_t counter_addr) const { return view_.counter_contains(counter_addr); }

    bool payload_read(uint64_t offset, uint64_t nbytes, WorkerChipOrchPayloadView &out) {
        out = WorkerChipOrchPayloadView{0, 0};
        if (has_error()) {
            return false;
        }
        RegionPayloadView inner{};
        if (!view_.payload_read(offset, nbytes, inner)) {
            adopt_view_error(WorkerChipEndpointOp::PAYLOAD_READ, 0, 0);
            return false;
        }
        out = WorkerChipOrchPayloadView{inner.local_addr, inner.nbytes};
        return true;
    }

    bool payload_write(uint64_t offset, const void *src, uint64_t nbytes) {
        if (has_error()) {
            return false;
        }
        if (!view_.payload_write(offset, src, nbytes)) {
            adopt_view_error(WorkerChipEndpointOp::PAYLOAD_WRITE, 0, 0);
            return false;
        }
        return true;
    }

    bool signal_notify(uint64_t counter_addr, int32_t value, WorkerChipOrchNotifyOp op) {
        if (has_error()) {
            return false;
        }
        uint64_t offset = 0;
        if (!absolute_to_counter_offset(counter_addr, offset)) {
            set_error(
                WorkerChipEndpointErrorKind::OUT_OF_BOUNDS, WorkerChipEndpointOp::SIGNAL_NOTIFY, desc_.region_id,
                counter_addr, value, "invalid counter address"
            );
            return false;
        }
        if (!view_.notify(offset, value, static_cast<RegionNotifyOp>(static_cast<uint32_t>(op)))) {
            adopt_view_error(WorkerChipEndpointOp::SIGNAL_NOTIFY, counter_addr, value);
            return false;
        }
        return true;
    }

    bool signal_test(
        uint64_t counter_addr, int32_t cmp_value, WorkerChipOrchWaitCmp cmp, WorkerChipOrchSignalTestResult &out
    ) {
        out = WorkerChipOrchSignalTestResult{false, 0};
        if (has_error()) {
            return false;
        }
        uint64_t offset = 0;
        if (!absolute_to_counter_offset(counter_addr, offset)) {
            set_error(
                WorkerChipEndpointErrorKind::OUT_OF_BOUNDS, WorkerChipEndpointOp::SIGNAL_TEST, desc_.region_id,
                counter_addr, cmp_value, "invalid counter address"
            );
            return false;
        }
        RegionSignalTestResult inner{};
        if (!view_.test(offset, cmp_value, static_cast<RegionWaitCmp>(static_cast<uint32_t>(cmp)), inner)) {
            adopt_view_error(WorkerChipEndpointOp::SIGNAL_TEST, counter_addr, cmp_value);
            return false;
        }
        out = WorkerChipOrchSignalTestResult{inner.matched, inner.observed};
        return true;
    }

    bool signal_wait(
        uint64_t counter_addr, int32_t cmp_value, WorkerChipOrchWaitCmp cmp, uint64_t timeout, int32_t &observed
    ) {
        observed = 0;
        if (has_error()) {
            return false;
        }
        uint64_t offset = 0;
        if (!absolute_to_counter_offset(counter_addr, offset)) {
            set_error(
                WorkerChipEndpointErrorKind::OUT_OF_BOUNDS, WorkerChipEndpointOp::SIGNAL_WAIT, desc_.region_id,
                counter_addr, cmp_value, "invalid counter address"
            );
            return false;
        }
        if (!view_.wait(offset, cmp_value, static_cast<RegionWaitCmp>(static_cast<uint32_t>(cmp)), timeout, observed)) {
            adopt_view_error(WorkerChipEndpointOp::SIGNAL_WAIT, counter_addr, cmp_value);
            return false;
        }
        return true;
    }

private:
    bool has_error() const { return error_.kind != WorkerChipEndpointErrorKind::NONE; }

    void install_view_or_error() {
        if (worker_chip_orch_comm::validate_desc(desc_) != WorkerChipOrchCommValidationError::OK ||
            !view_.assign(
                RegionPartLocalSpan{desc_.payload_base, desc_.payload_bytes},
                RegionPartLocalSpan{desc_.counter_base, desc_.counter_bytes}
            )) {
            set_error(
                WorkerChipEndpointErrorKind::BAD_DESCRIPTOR, WorkerChipEndpointOp::INIT, desc_.region_id, 0, 0,
                "invalid descriptor"
            );
        }
    }

    bool absolute_to_counter_offset(uint64_t counter_addr, uint64_t &offset) const {
        offset = 0;
        if (counter_addr < desc_.counter_base) {
            return false;
        }
        offset = counter_addr - desc_.counter_base;
        return true;
    }

    void adopt_view_error(WorkerChipEndpointOp op, uint64_t counter_addr, int32_t counter_operand) {
        const RegionViewError &ve = view_.error();
        WorkerChipEndpointErrorKind kind = WorkerChipEndpointErrorKind::OUT_OF_BOUNDS;
        switch (ve.kind) {
        case RegionViewErrorKind::INVALID_VIEW:
            kind = WorkerChipEndpointErrorKind::BAD_DESCRIPTOR;
            break;
        case RegionViewErrorKind::OUT_OF_BOUNDS:
            kind = WorkerChipEndpointErrorKind::OUT_OF_BOUNDS;
            break;
        case RegionViewErrorKind::INVALID_ENUM:
            kind = WorkerChipEndpointErrorKind::SIGNAL_PROTOCOL;
            break;
        case RegionViewErrorKind::TIMEOUT:
            kind = WorkerChipEndpointErrorKind::SIGNAL_TIMEOUT;
            break;
        case RegionViewErrorKind::ISSUED_FAILURE:
            kind = WorkerChipEndpointErrorKind::SIGNAL_PROTOCOL;
            break;
        case RegionViewErrorKind::NONE:
            kind = WorkerChipEndpointErrorKind::OUT_OF_BOUNDS;
            break;
        }
        set_error(kind, op, desc_.region_id, counter_addr, counter_operand, ve.observed, ve.message);
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
    RegionInstanceView view_{};
};
