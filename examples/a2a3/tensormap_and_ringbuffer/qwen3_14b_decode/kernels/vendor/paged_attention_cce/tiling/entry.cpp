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

#include <cstdint>

#include "tensor.h"

#ifdef __CPU_SIM
#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) { (void)args; }

#else

#include "intrinsic.h"

#define QWEN_FAI_TILER_FUNCTION static __aicore__
#include "qwen_fai_runtime_tiler.hpp"

namespace {

template <typename T>
static __aicore__ __attribute__((always_inline)) __gm__ T *tensor_data(__gm__ int64_t *args, int32_t index) {
    __gm__ Tensor *tensor = reinterpret_cast<__gm__ Tensor *>(args[index]);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
}

static __aicore__ __attribute__((always_inline)) __gm__ Tensor *tensor_desc(__gm__ int64_t *args, int32_t index) {
    return reinterpret_cast<__gm__ Tensor *>(args[index]);
}

static __aicore__ __attribute__((always_inline)) __gm__ int32_t *barrier_data(__gm__ uint8_t *metadata) {
    uint64_t raw_barrier = reinterpret_cast<uint64_t>(metadata + qwen_fai_metadata::kBarrierAlignmentOffset);
    uint64_t aligned_barrier = (raw_barrier + qwen_fai_metadata::kBarrierAlignmentBytes - 1) &
                               ~(static_cast<uint64_t>(qwen_fai_metadata::kBarrierAlignmentBytes) - 1);
    return reinterpret_cast<__gm__ int32_t *>(aligned_barrier);
}

static __aicore__ void clear_barrier(__gm__ int32_t *barrier) {
    for (uint32_t slot = 0; slot < qwen_fai_metadata::kBarrierSlotCount; ++slot) {
        __gm__ int32_t *slot_data = barrier + slot * qwen_fai_metadata::kBarrierSlotWords;
        slot_data[0] = 0;
        dcci(slot_data, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
    dsb(DSB_DDR);
}

static __aicore__ void flush_metadata_prefix(__gm__ uint8_t *metadata) {
    uint64_t first_line =
        reinterpret_cast<uint64_t>(metadata) & ~(static_cast<uint64_t>(qwen_fai_metadata::kDcciLineBytes) - 1);
    uint64_t end = reinterpret_cast<uint64_t>(metadata) + qwen_fai_metadata::kBarrierAlignmentOffset;
    for (uint64_t line = first_line; line < end; line += qwen_fai_metadata::kDcciLineBytes) {
        dcci(reinterpret_cast<__gm__ void *>(line), SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
    dsb(DSB_DDR);
}

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *seq_lens_desc = tensor_desc(args, 0);
    __gm__ const int32_t *seq_lens = tensor_data<const int32_t>(args, 0);
    __gm__ uint8_t *metadata = tensor_data<uint8_t>(args, 1);
    __gm__ uint32_t *tiling_out = reinterpret_cast<__gm__ uint32_t *>(metadata + qwen_fai_metadata::kTilingOffset);
    __gm__ int64_t *cumulative_q_out =
        reinterpret_cast<__gm__ int64_t *>(metadata + qwen_fai_metadata::kCumulativeQOffset);
    __gm__ int64_t *kv_lengths_out = reinterpret_cast<__gm__ int64_t *>(metadata + qwen_fai_metadata::kKvLengthsOffset);
    uint32_t batch = static_cast<uint32_t>(seq_lens_desc->shapes[0]);
    uint32_t max_blocks_per_batch = static_cast<uint32_t>(args[2]);
    uint32_t num_blocks = static_cast<uint32_t>(args[3]);

    clear_barrier(barrier_data(metadata));
    if (batch == 0 || batch > qwen_fai_tiler::kMaxBatch) {
        return;
    }

    uint32_t local_seq_lens[qwen_fai_tiler::kMaxBatch] = {};
    int64_t local_cumulative_q[qwen_fai_tiler::kMaxBatch] = {};
    int64_t local_kv_lengths[qwen_fai_tiler::kMaxBatch] = {};
    for (uint32_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        local_seq_lens[batch_idx] = static_cast<uint32_t>(seq_lens[batch_idx]);
    }

    FAInferTilingData tiling;
    bool valid = qwen_fai_tiler::build(
        local_seq_lens, batch, max_blocks_per_batch, num_blocks, tiling, local_cumulative_q, local_kv_lengths
    );
    if (!valid) {
        return;
    }

    uint32_t *tiling_words = reinterpret_cast<uint32_t *>(&tiling);
    for (uint32_t word_idx = 0; word_idx < sizeof(FAInferTilingData) / sizeof(uint32_t); ++word_idx) {
        tiling_out[word_idx] = tiling_words[word_idx];
    }
    for (uint32_t batch_idx = 0; batch_idx < qwen_fai_tiler::kMaxBatch; ++batch_idx) {
        cumulative_q_out[batch_idx] = local_cumulative_q[batch_idx];
        kv_lengths_out[batch_idx] = local_kv_lengths[batch_idx];
    }
    flush_metadata_prefix(metadata);
}

#endif
