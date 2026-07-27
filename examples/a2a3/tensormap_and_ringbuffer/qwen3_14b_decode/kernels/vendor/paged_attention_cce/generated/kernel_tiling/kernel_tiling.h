/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Modifications Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PYPTO_QWEN_FAI_KERNEL_TILING_H
#define PYPTO_QWEN_FAI_KERNEL_TILING_H

#include <cstddef>
#include <cstdint>

// Device-side mirror of flash_attention_infer_tiling.h.
struct coreNode {
    int32_t startBIdx[26];
    int32_t startN1Idx[26];
    int32_t startS1Idx[26];
    int32_t startS2Idx[26];
    int32_t endBIdx[26];
    int32_t endN1Idx[26];
    int32_t endS1Idx[26];
    int32_t endS2Idx[26];
    int64_t firstSplitKVTaskLseOffset[26];
    int64_t firstSplitKVTaskOOffset[26];
};

struct splitNode {
    int32_t batchIdx[26];
    int32_t headStartIdx[26];
    int32_t headEndIdx[26];
    int32_t qStartIdx[26];
    int32_t qEndIdx[26];
    int32_t splitNum[26];
    int64_t lseTaskOffset[26];
    int64_t oTaskOffset[26];
};

struct FAInferTilingData {
    uint32_t numHeads;
    uint32_t embeddingSize;
    uint32_t embeddingSizeV;
    uint32_t numBlocks;
    uint32_t blockSize;
    uint32_t maxQSeqlen;
    uint32_t maxKvSeqlen;
    uint32_t kvHeads;
    uint32_t batch;
    uint32_t maxNumBlocksPerBatch;
    uint32_t firstBatchTaskNum;
    uint32_t totalTaskNum;
    uint32_t maskType;
    uint32_t _pad_before_workspace_sizes;
    uint64_t mm1OutSize;
    uint64_t smOnlineOutSize;
    uint64_t mm2OutSize;
    uint64_t UpdateSize;
    uint64_t workSpaceSize;
    float scaleValue;
    uint32_t _pad_before_pse;
    uint64_t pseQ;
    uint64_t pseKv;
    uint32_t padding3;
    uint32_t _pad_before_tokens;
    int64_t preToken;
    int64_t nextToken;
    uint32_t sparseMode;
    uint32_t _pad_before_split_sizes;
    uint64_t splitLseTotalSize;
    uint64_t splitOTotalSize;
    uint32_t totalSplitNodeNum;
    uint32_t needCoreNum;
    uint32_t mainLoopTaskNum;
    uint32_t tailLoopTaskNum;
    uint32_t tailStartBatch;
    uint32_t tailStartN2;
    uint32_t tailKvNBlockTile;
    uint32_t _pad_before_nodes;
    coreNode coreInfo;
    splitNode splitInfo;
};

static_assert(sizeof(coreNode) == 1248, "coreNode ABI mismatch");
static_assert(sizeof(splitNode) == 1040, "splitNode ABI mismatch");
static_assert(offsetof(FAInferTilingData, mm1OutSize) == 56, "FAInfer scalar ABI mismatch");
static_assert(offsetof(FAInferTilingData, coreInfo) == 200, "FAInfer node ABI mismatch");
static_assert(sizeof(FAInferTilingData) == 2488, "FAInferTilingData ABI mismatch");

#endif
