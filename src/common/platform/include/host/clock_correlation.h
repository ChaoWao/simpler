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
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simpler::dfx {

constexpr std::size_t kClockAnchorSamplesPerPosition = 3;

// Where in a run the paired reading is taken. It supplies the offset that maps
// this device's timestamps onto the Host clock; the scale comes from the
// platform's counter frequency, so one reading is enough and it pins the two
// domains rather than bounding an interval they must fall inside. A timestamp
// either side of it maps by the same arithmetic.
//
// The serialized name stays "pre_host_orchestration": it is an on-wire value in
// every existing capture. It names a position in one runtime's sequence, which
// both runtimes no longer share — read it as the offset reading.
enum class ClockAnchorPosition : uint32_t {
    HostOffset = 0,
};

enum class ClockAnchorErrorStage : uint32_t {
    None = 0,
    CreateStream,
    CreateEvent,
    RecordEvent,
    SynchronizeEvent,
    GetTimestamp,
};

struct ClockAnchorSample {
    ClockAnchorPosition position{ClockAnchorPosition::HostOffset};
    uint32_t sample_idx{0};
    uint64_t host_before_ns{0};
    // Exact backend value plus the same instant normalized to the platform
    // system-counter frequency.
    uint64_t raw_device_timestamp{0};
    uint64_t device_cycles{0};
    uint64_t host_after_ns{0};
    ClockAnchorErrorStage error_stage{ClockAnchorErrorStage::None};
    int32_t error_code{0};

    bool valid() const {
        return error_stage == ClockAnchorErrorStage::None && error_code == 0 && host_before_ns != 0 &&
               host_after_ns >= host_before_ns && device_cycles != 0;
    }
};

const char *clock_anchor_position_name(ClockAnchorPosition position);
const char *clock_anchor_error_stage_name(ClockAnchorErrorStage stage);

class ClockCorrelationSession {
public:
    void begin(const char *provider_name, const char *raw_device_timestamp_unit);
    void append(std::vector<ClockAnchorSample> samples);
    void finish();
    void reset();

    bool started() const { return started_; }
    bool active() const { return active_; }
    const std::string &provider_name() const { return provider_name_; }
    const std::string &raw_device_timestamp_unit() const { return raw_device_timestamp_unit_; }
    const std::vector<ClockAnchorSample> &samples() const { return samples_; }

private:
    bool started_{false};
    bool active_{false};
    std::string provider_name_{};
    std::string raw_device_timestamp_unit_{};
    std::vector<ClockAnchorSample> samples_{};
};

class ClockCorrelationProvider {
public:
    virtual ~ClockCorrelationProvider() = default;
    ClockCorrelationProvider(const ClockCorrelationProvider &) = delete;
    ClockCorrelationProvider &operator=(const ClockCorrelationProvider &) = delete;

    virtual const char *name() const = 0;
    virtual const char *raw_device_timestamp_unit() const = 0;
    virtual ClockAnchorSample capture(ClockAnchorPosition position, uint32_t sample_idx) noexcept = 0;
    virtual void release(bool abandon_device_resources) noexcept = 0;

protected:
    ClockCorrelationProvider() = default;
};

std::vector<ClockAnchorSample>
capture_clock_anchor_group(ClockCorrelationProvider &provider, ClockAnchorPosition position);
std::unique_ptr<ClockCorrelationProvider> make_clock_correlation_provider();

}  // namespace simpler::dfx
