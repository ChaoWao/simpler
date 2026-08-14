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

#ifndef PYPTO_QWEN_FAI_RUNTIME_TILER_HPP
#define PYPTO_QWEN_FAI_RUNTIME_TILER_HPP

#include <cstddef>
#include <cstdint>

#include "../generated/kernel_tiling/kernel_tiling.h"
#include "../kernel/metadata_layout.h"

#ifndef QWEN_FAI_TILER_FUNCTION
#define QWEN_FAI_TILER_FUNCTION inline
#endif

namespace qwen_fai_tiler {

static_assert(
    sizeof(FAInferTilingData) == qwen_fai_metadata::kTilingBytes,
    "metadata tiling size does not match FAInferTilingData"
);

constexpr uint32_t kMaxBatch = 16;
constexpr uint32_t kNumHeads = 40;
constexpr uint32_t kNumKvHeads = 8;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kBlockSize = 128;
constexpr uint32_t kPlanningCoreNum = 24;
constexpr uint32_t kMaxCoreNum = 26;
constexpr uint32_t kQTileCeil = 128;
constexpr uint32_t kKvTile = 512;
constexpr uint32_t kPrelaunchNum = 3;
constexpr uint32_t kWorkspaceBlockSize = kQTileCeil * kKvTile;
constexpr uint32_t kUint16Bytes = 2;
constexpr uint32_t kUint32Bytes = 4;
constexpr uint64_t kBaseWorkspaceSize = 66'060'288;
constexpr int64_t kSparseTokenLimit = 2147483647;

struct BatchParams {
    uint32_t q_seqlen;
    uint32_t kv_seqlen;
    uint32_t qn_block_tile;
    uint32_t qn_blocks_per_group;
    uint32_t qn_block_num;
    uint32_t qs_block_tile;
    uint32_t qs_block_num;
    uint32_t ks_block_tile;
    uint32_t ks_block_num;
};

QWEN_FAI_TILER_FUNCTION uint32_t min_u32(uint32_t lhs, uint32_t rhs) { return lhs < rhs ? lhs : rhs; }

QWEN_FAI_TILER_FUNCTION uint32_t ceil_div(uint32_t value, uint32_t divisor) { return (value + divisor - 1) / divisor; }

QWEN_FAI_TILER_FUNCTION BatchParams get_batch_params(uint32_t batch_idx, const uint32_t *kv_seqlens) {
    constexpr uint32_t group_size = kNumHeads / kNumKvHeads;
    BatchParams params{};
    params.q_seqlen = 1;
    params.kv_seqlen = kv_seqlens[batch_idx];
    params.qn_block_tile = min_u32(kQTileCeil, group_size);
    params.qn_blocks_per_group = ceil_div(group_size, params.qn_block_tile);
    params.qn_block_num = params.qn_blocks_per_group * kNumKvHeads;
    params.qs_block_tile = kQTileCeil;
    params.qs_block_num = ceil_div(params.q_seqlen, params.qs_block_tile);
    params.ks_block_tile = kKvTile;
    params.ks_block_num = ceil_div(params.kv_seqlen, params.ks_block_tile);
    return params;
}

QWEN_FAI_TILER_FUNCTION void zero_tiling(FAInferTilingData &tiling) {
    uint32_t *words = reinterpret_cast<uint32_t *>(&tiling);
    for (uint32_t idx = 0; idx < sizeof(FAInferTilingData) / sizeof(uint32_t); ++idx) {
        words[idx] = 0;
    }
}

QWEN_FAI_TILER_FUNCTION void
fill_basic(FAInferTilingData &tiling, uint32_t batch, uint32_t max_blocks_per_batch, uint32_t num_blocks) {
    tiling.numHeads = kNumHeads;
    tiling.embeddingSize = kHeadDim;
    tiling.embeddingSizeV = kHeadDim;
    tiling.numBlocks = num_blocks;
    tiling.blockSize = kBlockSize;
    tiling.kvHeads = kNumKvHeads;
    tiling.batch = batch;
    tiling.maxNumBlocksPerBatch = max_blocks_per_batch;
    tiling.maskType = 1;
    tiling.scaleValue = 0.08838834764831843F;
    tiling.preToken = kSparseTokenLimit;
    tiling.nextToken = kSparseTokenLimit;
    tiling.sparseMode = 3;
}

QWEN_FAI_TILER_FUNCTION void fill_task_counts(FAInferTilingData &tiling, uint32_t batch) {
    constexpr uint32_t tasks_per_batch = kNumKvHeads;
    tiling.firstBatchTaskNum = tasks_per_batch;
    tiling.totalTaskNum = tasks_per_batch * batch;
}

QWEN_FAI_TILER_FUNCTION void init_core_info(FAInferTilingData &tiling, uint32_t planning_core_num) {
    for (uint32_t core_idx = 0; core_idx < planning_core_num; ++core_idx) {
        tiling.coreInfo.startBIdx[core_idx] = 0;
        tiling.coreInfo.startN1Idx[core_idx] = 0;
        tiling.coreInfo.startS1Idx[core_idx] = 0;
        tiling.coreInfo.startS2Idx[core_idx] = 0;
        tiling.coreInfo.endBIdx[core_idx] = 0;
        tiling.coreInfo.endN1Idx[core_idx] = 0;
        tiling.coreInfo.endS1Idx[core_idx] = 0;
        tiling.coreInfo.endS2Idx[core_idx] = 0;
    }
}

QWEN_FAI_TILER_FUNCTION void consume_s2_blocks(
    const uint32_t *kv_seqlens, int64_t &remaining_tasks, uint32_t batch_idx, uint32_t s1_idx, uint32_t &s2_idx
) {
    while (s2_idx < get_batch_params(batch_idx, kv_seqlens).ks_block_num && remaining_tasks > 0) {
        BatchParams params = get_batch_params(batch_idx, kv_seqlens);
        uint32_t remaining_q = s1_idx < params.qs_block_num - 1 ?
                                   params.qs_block_tile :
                                   (params.q_seqlen - s1_idx * params.qs_block_tile) * params.qn_block_tile;
        uint32_t remaining_kv =
            s2_idx < params.ks_block_num - 1 ? params.ks_block_tile : params.kv_seqlen - s2_idx * params.ks_block_tile;
        remaining_tasks -= static_cast<uint64_t>(remaining_q) * remaining_kv;
        ++s2_idx;
    }
}

QWEN_FAI_TILER_FUNCTION void consume_remaining_batches(
    const uint32_t *kv_seqlens, uint32_t batch, int64_t &remaining_tasks, uint32_t &batch_idx, uint32_t &n1_idx,
    uint32_t &s1_idx, uint32_t &s2_idx
) {
    while (batch_idx < batch && remaining_tasks > 0) {
        BatchParams params = get_batch_params(batch_idx, kv_seqlens);
        uint32_t remaining_q =
            params.q_seqlen * (kNumHeads - params.qn_block_tile * n1_idx) - s1_idx * params.qs_block_tile;
        uint32_t remaining_in_batch = remaining_q * params.kv_seqlen;
        if (remaining_tasks < static_cast<int64_t>(remaining_in_batch)) {
            break;
        }
        remaining_tasks -= remaining_in_batch;
        ++batch_idx;
        n1_idx = 0;
        s1_idx = 0;
        s2_idx = 0;
    }
}

QWEN_FAI_TILER_FUNCTION void consume_remaining_n1_groups(
    int64_t &remaining_tasks, const BatchParams &params, uint32_t &n1_idx, uint32_t &s1_idx, uint32_t &s2_idx
) {
    while (n1_idx < params.qn_block_num && remaining_tasks > 0) {
        uint32_t remaining_q = params.q_seqlen * params.qn_block_tile - s1_idx * params.qs_block_tile;
        uint32_t remaining_in_n1 = remaining_q * params.kv_seqlen;
        if (remaining_tasks < static_cast<int64_t>(remaining_in_n1)) {
            break;
        }
        remaining_tasks -= remaining_in_n1;
        ++n1_idx;
        s1_idx = 0;
        s2_idx = 0;
    }
}

QWEN_FAI_TILER_FUNCTION void
consume_remaining_s1_groups(int64_t &remaining_tasks, const BatchParams &params, uint32_t &s1_idx, uint32_t &s2_idx) {
    while (s1_idx < params.qs_block_num && remaining_tasks > 0) {
        uint32_t remaining_q = s1_idx < params.qs_block_num - 1 ?
                                   params.qs_block_tile :
                                   (params.q_seqlen - s1_idx * params.qs_block_tile) * params.qn_block_tile;
        uint64_t remaining_in_s1 = static_cast<uint64_t>(remaining_q) * params.kv_seqlen;
        if (remaining_tasks < static_cast<int64_t>(remaining_in_s1)) {
            break;
        }
        remaining_tasks -= remaining_in_s1;
        ++s1_idx;
        s2_idx = 0;
    }
}

QWEN_FAI_TILER_FUNCTION void
finish_batch(FAInferTilingData &tiling, const uint32_t *kv_seqlens, uint32_t batch, uint32_t core_idx) {
    BatchParams params = get_batch_params(batch - 1, kv_seqlens);
    tiling.coreInfo.endBIdx[core_idx] = batch - 1;
    tiling.coreInfo.endN1Idx[core_idx] = params.qn_block_num - 1;
    tiling.coreInfo.endS1Idx[core_idx] = params.qs_block_num - 1;
    tiling.coreInfo.endS2Idx[core_idx] = params.ks_block_num;
    tiling.needCoreNum = core_idx + 1;
}

QWEN_FAI_TILER_FUNCTION void
advance_counters(const BatchParams &params, uint32_t &batch_idx, uint32_t &n1_idx, uint32_t &s1_idx, uint32_t &s2_idx) {
    if (s2_idx == params.ks_block_num) {
        ++s1_idx;
        s2_idx = 0;
    }
    if (s1_idx == params.qs_block_num) {
        ++n1_idx;
        s1_idx = 0;
        s2_idx = 0;
    }
    if (n1_idx == params.qn_block_num) {
        ++batch_idx;
        n1_idx = 0;
        s1_idx = 0;
        s2_idx = 0;
    }
}

QWEN_FAI_TILER_FUNCTION void init_split_info(FAInferTilingData &tiling, uint32_t planning_core_num) {
    for (uint32_t split_idx = 0; split_idx < planning_core_num + 1; ++split_idx) {
        tiling.splitInfo.batchIdx[split_idx] = 0;
        tiling.splitInfo.headStartIdx[split_idx] = 0;
        tiling.splitInfo.headEndIdx[split_idx] = 0;
        tiling.splitInfo.qStartIdx[split_idx] = 0;
        tiling.splitInfo.qEndIdx[split_idx] = 0;
        tiling.splitInfo.splitNum[split_idx] = 0;
        tiling.splitInfo.lseTaskOffset[split_idx] = 0;
        tiling.splitInfo.oTaskOffset[split_idx] = 0;
    }
}

QWEN_FAI_TILER_FUNCTION void process_core_split_info(
    FAInferTilingData &tiling, const uint32_t *kv_seqlens, uint32_t core_idx, int32_t &split_idx,
    int32_t &prev_batch_idx, int32_t &prev_n1_idx, int32_t &prev_s1_idx, int64_t &current_lse_offset,
    int64_t &current_o_offset, uint32_t planning_core_num
) {
    int32_t start_batch_idx = tiling.coreInfo.startBIdx[core_idx];
    int32_t start_n1_idx = tiling.coreInfo.startN1Idx[core_idx];
    int32_t start_s1_idx = tiling.coreInfo.startS1Idx[core_idx];
    int32_t start_s2_idx = tiling.coreInfo.startS2Idx[core_idx];
    int32_t end_batch_idx = tiling.coreInfo.endBIdx[core_idx];
    int32_t end_n1_idx = tiling.coreInfo.endN1Idx[core_idx];
    int32_t end_s1_idx = tiling.coreInfo.endS1Idx[core_idx];
    int32_t end_s2_idx = tiling.coreInfo.endS2Idx[core_idx];

    tiling.coreInfo.firstSplitKVTaskLseOffset[core_idx] = 0;
    tiling.coreInfo.firstSplitKVTaskOOffset[core_idx] = 0;
    bool found_first_split = false;

    for (int32_t batch_idx = start_batch_idx; batch_idx <= end_batch_idx; ++batch_idx) {
        BatchParams params = get_batch_params(batch_idx, kv_seqlens);
        int32_t current_start_n1 = batch_idx == start_batch_idx ? start_n1_idx : 0;
        int32_t current_end_n1 = batch_idx == end_batch_idx ? end_n1_idx : params.qn_block_num - 1;
        for (int32_t n1_idx = current_start_n1; n1_idx <= current_end_n1; ++n1_idx) {
            int32_t current_start_s1 = batch_idx == start_batch_idx && n1_idx == start_n1_idx ? start_s1_idx : 0;
            int32_t current_end_s1 =
                batch_idx == end_batch_idx && n1_idx == end_n1_idx ? end_s1_idx : params.qs_block_num - 1;
            for (int32_t s1_idx = current_start_s1; s1_idx <= current_end_s1; ++s1_idx) {
                int32_t current_start_s2 =
                    batch_idx == start_batch_idx && n1_idx == start_n1_idx && s1_idx == start_s1_idx ? start_s2_idx : 0;
                int32_t current_end_s2 = batch_idx == end_batch_idx && n1_idx == end_n1_idx && s1_idx == end_s1_idx ?
                                             end_s2_idx :
                                             params.ks_block_num;
                uint32_t covered_s2 = current_end_s2 - current_start_s2;
                bool is_split = covered_s2 > 0 && covered_s2 < params.ks_block_num;
                if (!is_split) {
                    continue;
                }

                int64_t temporary_lse_offset = current_lse_offset;
                int64_t temporary_o_offset = current_o_offset;
                uint32_t n1_per_group = n1_idx % params.qn_blocks_per_group;
                uint32_t kv_head_idx = n1_idx / params.qn_blocks_per_group;
                uint32_t head_start = kv_head_idx * (kNumHeads / kNumKvHeads) + n1_per_group * params.qn_block_tile;
                uint32_t head_end =
                    min_u32(head_start + params.qn_block_tile, (kv_head_idx + 1) * (kNumHeads / kNumKvHeads));
                uint32_t q_start = s1_idx * params.qs_block_tile;
                uint32_t q_end = min_u32(q_start + params.qs_block_tile, params.q_seqlen);
                uint32_t head_len = head_end - head_start;
                uint32_t q_len = q_end - q_start;

                if (batch_idx != prev_batch_idx || n1_idx != prev_n1_idx || s1_idx != prev_s1_idx) {
                    ++split_idx;
                    if (split_idx >= 0 && split_idx < static_cast<int32_t>(planning_core_num + 1)) {
                        tiling.splitInfo.batchIdx[split_idx] = batch_idx;
                        tiling.splitInfo.splitNum[split_idx] = 0;
                        tiling.splitInfo.headStartIdx[split_idx] = head_start;
                        tiling.splitInfo.headEndIdx[split_idx] = head_end;
                        tiling.splitInfo.qStartIdx[split_idx] = q_start;
                        tiling.splitInfo.qEndIdx[split_idx] = q_end;
                        tiling.splitInfo.lseTaskOffset[split_idx] = current_lse_offset;
                        tiling.splitInfo.oTaskOffset[split_idx] = current_o_offset;
                    }
                    prev_batch_idx = batch_idx;
                    prev_n1_idx = n1_idx;
                    prev_s1_idx = s1_idx;
                }
                if (split_idx >= 0 && split_idx < static_cast<int32_t>(planning_core_num + 1)) {
                    ++tiling.splitInfo.splitNum[split_idx];
                    current_lse_offset += static_cast<int64_t>(head_len) * q_len;
                    current_o_offset += static_cast<int64_t>(head_len) * q_len * kHeadDim;
                }
                if (!found_first_split) {
                    found_first_split = true;
                    tiling.coreInfo.firstSplitKVTaskLseOffset[core_idx] = temporary_lse_offset;
                    tiling.coreInfo.firstSplitKVTaskOOffset[core_idx] = temporary_o_offset;
                }
            }
        }
    }
}

QWEN_FAI_TILER_FUNCTION void fill_workspace(FAInferTilingData &tiling) {
    tiling.mm1OutSize = static_cast<uint64_t>(kPlanningCoreNum) * kWorkspaceBlockSize * kUint32Bytes * kPrelaunchNum;
    tiling.smOnlineOutSize =
        static_cast<uint64_t>(kPlanningCoreNum) * kWorkspaceBlockSize * kUint16Bytes * kPrelaunchNum;
    tiling.mm2OutSize = tiling.mm1OutSize;
    tiling.UpdateSize = tiling.mm1OutSize;
    tiling.workSpaceSize = kBaseWorkspaceSize + tiling.splitLseTotalSize + tiling.splitOTotalSize;
}

QWEN_FAI_TILER_FUNCTION void
initialize_tiling(FAInferTilingData &tiling, uint32_t batch, uint32_t max_blocks_per_batch, uint32_t num_blocks) {
    zero_tiling(tiling);
    fill_basic(tiling, batch, max_blocks_per_batch, num_blocks);
    fill_task_counts(tiling, batch);
}

QWEN_FAI_TILER_FUNCTION bool build(
    const uint32_t *kv_seqlens, uint32_t batch, uint32_t max_blocks_per_batch, uint32_t num_blocks,
    FAInferTilingData &tiling, int64_t *cumulative_q_lengths, int64_t *kv_lengths
) {
    if (batch == 0 || batch > kMaxBatch) {
        return false;
    }
    for (uint32_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        cumulative_q_lengths[batch_idx] = batch_idx + 1;
        kv_lengths[batch_idx] = kv_seqlens[batch_idx];
    }
    initialize_tiling(tiling, batch, max_blocks_per_batch, num_blocks);
    fill_workspace(tiling);
    return true;
}

}  // namespace qwen_fai_tiler

#endif
