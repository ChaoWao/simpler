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

#include "host/clock_correlation.h"

#include <memory>

#include "aicpu/device_time.h"
#include "common/log_clock.h"

namespace simpler::dfx {
namespace {

class SimClockCorrelationProvider final : public ClockCorrelationProvider {
public:
    const char *name() const override { return "sim_syscnt"; }
    const char *raw_device_timestamp_unit() const override { return "syscnt_cycles"; }

    ClockAnchorSample capture(ClockAnchorPosition position, uint32_t sample_idx) noexcept override {
        ClockAnchorSample sample{};
        sample.position = position;
        sample.sample_idx = sample_idx;
        sample.host_before_ns = static_cast<uint64_t>(simpler::log::monotonic_now_ns());
        sample.raw_device_timestamp = sys_cnt_now_ticks();
        sample.device_cycles = sample.raw_device_timestamp;
        sample.host_after_ns = static_cast<uint64_t>(simpler::log::monotonic_now_ns());
        return sample;
    }

    void release(bool /*abandon_device_resources*/) noexcept override {}
};

}  // namespace

std::unique_ptr<ClockCorrelationProvider> make_clock_correlation_provider() {
    return std::make_unique<SimClockCorrelationProvider>();
}

}  // namespace simpler::dfx
