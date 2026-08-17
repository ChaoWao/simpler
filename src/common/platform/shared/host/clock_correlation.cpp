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

#include <iterator>
#include <utility>

namespace simpler::dfx {

const char *clock_anchor_position_name(ClockAnchorPosition position) {
    switch (position) {
    case ClockAnchorPosition::HostOrchestrationBegin:
        return "pre_host_orchestration";
    case ClockAnchorPosition::DeviceExecutionComplete:
        return "post_device_execution";
    }
    return "unknown";
}

const char *clock_anchor_error_stage_name(ClockAnchorErrorStage stage) {
    switch (stage) {
    case ClockAnchorErrorStage::None:
        return "none";
    case ClockAnchorErrorStage::CreateStream:
        return "create_stream";
    case ClockAnchorErrorStage::CreateEvent:
        return "create_event";
    case ClockAnchorErrorStage::RecordEvent:
        return "record_event";
    case ClockAnchorErrorStage::SynchronizeEvent:
        return "synchronize_event";
    case ClockAnchorErrorStage::GetTimestamp:
        return "get_timestamp";
    }
    return "unknown";
}

void ClockCorrelationSession::begin(const char *provider_name, const char *raw_device_timestamp_unit) {
    reset();
    started_ = true;
    active_ = true;
    provider_name_ = provider_name != nullptr ? provider_name : "unknown";
    raw_device_timestamp_unit_ = raw_device_timestamp_unit != nullptr ? raw_device_timestamp_unit : "unknown";
}

void ClockCorrelationSession::append(std::vector<ClockAnchorSample> samples) {
    if (!active_) return;
    samples_.insert(samples_.end(), std::make_move_iterator(samples.begin()), std::make_move_iterator(samples.end()));
}

void ClockCorrelationSession::finish() { active_ = false; }

void ClockCorrelationSession::reset() {
    started_ = false;
    active_ = false;
    provider_name_.clear();
    raw_device_timestamp_unit_.clear();
    samples_.clear();
}

std::vector<ClockAnchorSample>
capture_clock_anchor_group(ClockCorrelationProvider &provider, ClockAnchorPosition position) {
    std::vector<ClockAnchorSample> samples;
    samples.reserve(kClockAnchorSamplesPerPosition);
    for (std::size_t sample_idx = 0; sample_idx < kClockAnchorSamplesPerPosition; ++sample_idx) {
        samples.push_back(provider.capture(position, static_cast<uint32_t>(sample_idx)));
    }
    return samples;
}

}  // namespace simpler::dfx
