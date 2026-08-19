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

#include <acl/acl.h>

#include <memory>

#include "common/log_clock.h"
#include "common/platform_config.h"

namespace simpler::dfx {
namespace {

class OnboardClockCorrelationProvider final : public ClockCorrelationProvider {
public:
    ~OnboardClockCorrelationProvider() override { release(false); }

    const char *name() const override { return "acl_event"; }
    const char *raw_device_timestamp_unit() const override { return PLATFORM_ACL_EVENT_TIMESTAMP_UNIT; }

    ClockAnchorSample capture(ClockAnchorPosition position, uint32_t sample_idx) noexcept override {
        ClockAnchorSample sample{};
        sample.position = position;
        sample.sample_idx = sample_idx;
        sample.host_before_ns = static_cast<uint64_t>(simpler::log::monotonic_now_ns());

        if (!ensure_resources(sample)) {
            sample.host_after_ns = static_cast<uint64_t>(simpler::log::monotonic_now_ns());
            return sample;
        }

        aclError rc = aclrtRecordEvent(event_, stream_);
        if (rc != ACL_SUCCESS) {
            sample.error_stage = ClockAnchorErrorStage::RecordEvent;
            sample.error_code = static_cast<int32_t>(rc);
            sample.host_after_ns = static_cast<uint64_t>(simpler::log::monotonic_now_ns());
            return sample;
        }
        rc = aclrtSynchronizeEvent(event_);
        if (rc != ACL_SUCCESS) {
            sample.error_stage = ClockAnchorErrorStage::SynchronizeEvent;
            sample.error_code = static_cast<int32_t>(rc);
            sample.host_after_ns = static_cast<uint64_t>(simpler::log::monotonic_now_ns());
            return sample;
        }
        rc = aclrtEventGetTimestamp(event_, &sample.raw_device_timestamp);
        sample.host_after_ns = static_cast<uint64_t>(simpler::log::monotonic_now_ns());
        if (rc != ACL_SUCCESS) {
            sample.error_stage = ClockAnchorErrorStage::GetTimestamp;
            sample.error_code = static_cast<int32_t>(rc);
            sample.raw_device_timestamp = 0;
        } else {
            if constexpr (PLATFORM_ACL_EVENT_TIMESTAMP_FREQ_HZ != 0) {
                // Preserve the exact backend value, then normalize it to the
                // platform's verified profiling-counter frequency.
                sample.device_cycles = static_cast<uint64_t>(
                    static_cast<__uint128_t>(sample.raw_device_timestamp) * PLATFORM_PROF_SYS_CNT_FREQ /
                    PLATFORM_ACL_EVENT_TIMESTAMP_FREQ_HZ
                );
            }
        }
        return sample;
    }

    void release(bool abandon_device_resources) noexcept override {
        if (event_ != nullptr) {
            if (!abandon_device_resources) (void)aclrtDestroyEvent(event_);
            event_ = nullptr;
        }
        if (stream_ != nullptr) {
            if (!abandon_device_resources) (void)aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
    }

private:
    bool ensure_resources(ClockAnchorSample &sample) noexcept {
        if (stream_ == nullptr) {
            aclError rc = aclrtCreateStream(&stream_);
            if (rc != ACL_SUCCESS) {
                stream_ = nullptr;
                sample.error_stage = ClockAnchorErrorStage::CreateStream;
                sample.error_code = static_cast<int32_t>(rc);
                return false;
            }
        }
        if (event_ == nullptr) {
            aclError rc = aclrtCreateEventExWithFlag(&event_, ACL_EVENT_TIME_LINE);
            if (rc != ACL_SUCCESS) {
                event_ = nullptr;
                sample.error_stage = ClockAnchorErrorStage::CreateEvent;
                sample.error_code = static_cast<int32_t>(rc);
                return false;
            }
        }
        return true;
    }

    aclrtStream stream_{nullptr};
    aclrtEvent event_{nullptr};
};

}  // namespace

std::unique_ptr<ClockCorrelationProvider> make_clock_correlation_provider() {
    return std::make_unique<OnboardClockCorrelationProvider>();
}

}  // namespace simpler::dfx
