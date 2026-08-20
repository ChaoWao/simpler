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
/**
 * Status-buffer layout shared by the SDMA control-path warmup kernel
 * (src/a2a3/platform/onboard/aicore/sdma_warmup_kernel.cpp) and its host
 * launcher (DeviceRunnerBase::launch_sdma_warmup_kernel).
 */

#pragma once

#include <cstdint>

// One cache line per channel. A packed uint32_t[channel_count] does NOT work:
// the warming cores are AIVs writing concurrently, and each core's dcci clean +
// invalidate drops its neighbours' pending stores that share the line. Measured
// on a2a3 silicon with a 48-block vector-only launch: a packed array reported
// only ~17 of 48 slots written, and the surviving set differed run to run.
constexpr uint32_t kSdmaWarmupStatusStrideBytes = 64U;

// Slot values. 0 (the host-side zero fill) therefore means "no core reached this
// channel", which the launcher reports separately from the warmup declining its
// preconditions on a channel it did reach.
constexpr uint32_t kSdmaWarmupStatusOk = 1U;
constexpr uint32_t kSdmaWarmupStatusFailed = 2U;

// Cores to spread the channel walk over. Measured on a2a3 silicon: the cold cost
// is per-channel (~92 us each) and serializes inside the engine no matter how
// many cores push on it, so block_dim 2..48 all complete in ~4.4 ms. Only a
// single block is worse (~6.9 ms), because then the AICore-side barriers and UB
// copies serialize behind the engine instead of overlapping it. 8 clears that
// cliff and, unlike a channel-count-derived value, stays valid on any AIV count.
constexpr uint32_t kSdmaWarmupBlockDim = 8U;
