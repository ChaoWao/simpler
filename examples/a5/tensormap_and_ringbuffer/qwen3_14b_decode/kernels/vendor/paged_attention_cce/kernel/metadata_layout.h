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

#ifndef PYPTO_QWEN_FAI_METADATA_LAYOUT_H
#define PYPTO_QWEN_FAI_METADATA_LAYOUT_H

#include <cstdint>

namespace qwen_fai_metadata {

constexpr uint32_t kTilingOffset = 0;
constexpr uint32_t kTilingBytes = 2488;
constexpr uint32_t kCumulativeQOffset = 2488;
constexpr uint32_t kLengthArrayBytes = 16 * sizeof(int64_t);
constexpr uint32_t kKvLengthsOffset = kCumulativeQOffset + kLengthArrayBytes;
constexpr uint32_t kBarrierAlignmentOffset = kKvLengthsOffset + kLengthArrayBytes;
constexpr uint32_t kDcciLineBytes = 64;
constexpr uint32_t kBarrierAlignmentBytes = 512;
constexpr uint32_t kBarrierSlotBytes = 512;
constexpr uint32_t kBarrierSlotWords = kBarrierSlotBytes / sizeof(int32_t);
constexpr uint32_t kBarrierSlotCount = 48;
constexpr uint32_t kBarrierBytes = kBarrierSlotCount * kBarrierSlotBytes;
// The pto soft Mix SyncAll counter, one int32 immediately past the slots the
// attention kernel uses for its own barriers. It is a monotonic ticket counter
// (`(before / participants + 1) * participants`), so it MUST start at zero:
// from any other value the first round's arrivals straddle a participant
// boundary and the cores that land above it wait for a count nobody reaches.
// tiling/entry.cpp is the single writer that establishes that zero, and it runs
// once for the whole graph -- the counter then runs monotonically across every
// layer's attention task.
constexpr uint32_t kSyncAllCounterWord = kBarrierSlotCount * kBarrierSlotWords;
constexpr uint32_t kMetadataBytes = 27840;

static_assert(
    kBarrierAlignmentOffset + kBarrierAlignmentBytes - 1 + kBarrierBytes + sizeof(int32_t) <= kMetadataBytes,
    "metadata buffer does not cover the aligned barrier region plus the SyncAll counter"
);

}  // namespace qwen_fai_metadata

#endif
