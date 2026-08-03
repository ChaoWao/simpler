/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_HPP

#include <type_traits>
#include <limits>
#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/cross_core_sync.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/tile_copy.hpp"
#include "../../../attn_infra/gemm_coord.hpp"
#include "../../../attn_infra/matrix_coord.hpp"
#include "utils/std/algorithm.h"

namespace NpuArch::Epilogue::Block {

struct SinkLoopParam
{
    uint32_t rowOffsetIoGm;
    uint32_t rowNumCurLoop;
    uint32_t qSBlockSize;
    uint32_t rowOffsetThisSubBlock;

    __aicore__ inline
    SinkLoopParam(
        uint32_t rowOffsetIoGm_,
        uint32_t rowNumCurLoop_,
        uint32_t qSBlockSize_,
        uint32_t rowOffsetThisSubBlock_
    ) :
        rowOffsetIoGm(rowOffsetIoGm_),
        rowNumCurLoop(rowNumCurLoop_),
        qSBlockSize(qSBlockSize_),
        rowOffsetThisSubBlock(rowOffsetThisSubBlock_)
    {}
};

template <
    class OutputType_,
    class InputType_,
    class MaskType_,
    class SinkType_,
    class FullType_,
    LseMode LSE_MODE_,
    SinkMode SINK_MODE_,
    MaskMode MASK_MODE_>
class BlockEpilogue<
    EpilogueAtlasA2OnlineSoftmax<LSE_MODE_, SINK_MODE_, MASK_MODE_, float>,
    OutputType_,
    InputType_,
    MaskType_,
    SinkType_,
    FullType_>
{
public:
    using DispatchPolicy = EpilogueAtlasA2OnlineSoftmax<LSE_MODE_, SINK_MODE_, MASK_MODE_, float>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = typename InputType_::Element;
    using ElementMask = typename MaskType_::Element;
    using ElementSink = typename SinkType_::Element;
    using ElementFull = typename FullType_::Element;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;
    using LayoutMask = typename MaskType_::Layout;
    using LayoutFull = typename FullType_::Layout;

    static constexpr LseMode LSE_MODE = DispatchPolicy::LSE_MODE;
    static constexpr SinkMode SINK_MODE = DispatchPolicy::SINK_MODE;
    static constexpr MaskMode MASK_MODE = DispatchPolicy::MASK_MODE;

    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;
    static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 8192;

    static constexpr uint32_t REDUCE_UB_SIZE = 1024;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;
    static constexpr uint32_t HEAD_NUM_2 = 2;

    static constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

    __aicore__ inline
    BlockEpilogue() {}

    __aicore__ inline
    void init(Arch::Resource<ArchTag> &resource, float scaleValue_)
    {
        ptoSubBlockIdx = resource.ptoTopology.subBlockIdx;
        ptoLanesPerBlock = resource.ptoTopology.lanesPerBlock;
        // Allocate UB space
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t MASK_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t MASK32_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t FULL32_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t MASK_UB_PREMASK_TENSOR_OFFSET = 5 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 8 * UB_UINT8_VECTOR_SIZE;

        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t LL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 11 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t SEL_MASK_UB_TENSOR_OFFSET = LL_UB_TENSOR_OFFSET;

        constexpr uint32_t MASK16_UB_TENSOR_OFFSET = 11 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t FULL16_UB_TENSOR_OFFSET = 11 * UB_UINT8_BLOCK_SIZE;
        scaleValue = scaleValue_;
        lsUbTensor = resource.ubBuf.template GetBufferByByte<float>(LS_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
        maskUbTensor = resource.ubBuf.template GetBufferByByte<ElementMask>(MASK_UB_TENSOR_OFFSET);
        maskUbTensorUint8 = resource.ubBuf.template GetBufferByByte<uint8_t>(MASK_UB_TENSOR_OFFSET);
        maskUbTensor16 = resource.ubBuf.template GetBufferByByte<half>(MASK16_UB_TENSOR_OFFSET);
        maskUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(MASK32_UB_TENSOR_OFFSET);
        fullUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementFull>(FULL16_UB_TENSOR_OFFSET);
        fullUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(FULL32_UB_TENSOR_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<float>(LM_UB_TENSOR_OFFSET);
        hmUbTensor = resource.ubBuf.template GetBufferByByte<float>(HM_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        llUbTensor = resource.ubBuf.template GetBufferByByte<float>(LL_UB_TENSOR_OFFSET);
        selMaskUbTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(SEL_MASK_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        tempMaskTensor = resource.ubBuf.template GetBufferByByte<half>(MASK_UB_PREMASK_TENSOR_OFFSET);
    }

    __aicore__ inline
    ~BlockEpilogue() {}

    template <typename T>
    __aicore__ inline T Min(T a, T b)
    {
        return (a > b) ? b : a;
    }

    __aicore__ inline
    void SetVecMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = len % FLOAT_VECTOR_SIZE;
        for (int64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }

        if (len == VECTOR_SIZE || len == 0) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        }
    }

    __aicore__ inline
    void SetBlockReduceMask(int32_t len)
    {
        if (len > 8 || len < 1) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        uint64_t subMask = ((uint64_t)1 << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask + (subMask << 56) +
                             (subMask << 40) + (subMask << 24) + (subMask << 8);
        AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
    }

    __aicore__ inline
    void RowsumSPECTILE512(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowsumUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        AscendC::BlockReduceSum<float, false>(
            tvUbTensor,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::BlockReduceSum<float, false>(
            tvUbTensor[REDUCE_UB_SIZE],
            tvUbTensor,
            numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum<float, false>(
            rowsumUb,
            tvUbTensor[REDUCE_UB_SIZE],
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void RowsumSPECTILE256(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowsumUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        AscendC::BlockReduceSum<float, false>(
            tvUbTensor,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        SetVecMask(ROW_OPS_SPEC_MASK_32);
        AscendC::BlockReduceSum<float, false>(
            tvUbTensor[REDUCE_UB_SIZE],
            tvUbTensor,
            numRowsRound,
            0, 1, 1, 4);
        AscendC::PipeBarrier<PIPE_V>();
        SetBlockReduceMask(ROW_OPS_SPEC_MASK_4);
        AscendC::BlockReduceSum<float, false>(
            rowsumUb,
            tvUbTensor[REDUCE_UB_SIZE],
            NpuArch::Detail::Alignment::CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }

    __aicore__ inline
    void RowsumTAILTILE(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowsumUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        if (numElems >= FLOAT_VECTOR_SIZE) {
            AscendC::BlockReduceSum<float, false>(
                tvUbTensor,
                srcUb,
                numRowsRound,
                0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceSum<float, false>(
                rowsumUb,
                tvUbTensor,
                NpuArch::Detail::Alignment::CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint64_t rowSumIdx = 1; rowSumIdx < (uint64_t)numElems / FLOAT_VECTOR_SIZE; ++rowSumIdx) {
                AscendC::BlockReduceSum<float, false>(
                    tvUbTensor,
                    srcUb[rowSumIdx * FLOAT_VECTOR_SIZE],
                    numRowsRound,
                    0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::BlockReduceSum<float, false>(
                    tvUbTensor[REDUCE_UB_SIZE],
                    tvUbTensor,
                    NpuArch::Detail::Alignment::CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Add<float, false>(
                    rowsumUb,
                    rowsumUb,
                    tvUbTensor[REDUCE_UB_SIZE],
                    (uint64_t)0,
                    1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(numElems % FLOAT_VECTOR_SIZE);
            AscendC::BlockReduceSum<float, false>(
                tvUbTensor,
                srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                numRowsRound,
                0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            SetBlockReduceMask(NpuArch::Detail::Alignment::CeilDiv(numElems % FLOAT_VECTOR_SIZE, FLOAT_BLOCK_SIZE));
            if (numElems < FLOAT_VECTOR_SIZE) {
                AscendC::BlockReduceSum<float, false>(
                    rowsumUb,
                    tvUbTensor,
                    NpuArch::Detail::Alignment::CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::BlockReduceSum<float, false>(
                    tvUbTensor[REDUCE_UB_SIZE],
                    tvUbTensor,
                    NpuArch::Detail::Alignment::CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Add<float, false>(
                    rowsumUb,
                    rowsumUb,
                    tvUbTensor[REDUCE_UB_SIZE],
                    (uint64_t)0,
                    1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }

    __aicore__ inline
    void RowmaxSPECTILE512(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowmaxUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        AscendC::BlockReduceMax<float, false>(
            tvUbTensor,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceMax<float, false>(
            tvUbTensor[REDUCE_UB_SIZE],
            tvUbTensor,
            numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceMax<float, false>(
            rowmaxUb,
            tvUbTensor[REDUCE_UB_SIZE],
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void RowmaxSPECTILE256(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowmaxUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        AscendC::BlockReduceMax<float, false>(
            tvUbTensor,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        SetVecMask(ROW_OPS_SPEC_MASK_32);
        AscendC::BlockReduceMax<float, false>(
            tvUbTensor[REDUCE_UB_SIZE],
            tvUbTensor,
            numRowsRound,
            0, 1, 1, 4);
        AscendC::PipeBarrier<PIPE_V>();
        SetBlockReduceMask(ROW_OPS_SPEC_MASK_4);
        AscendC::BlockReduceMax<float, false>(
            rowmaxUb,
            tvUbTensor[REDUCE_UB_SIZE],
            NpuArch::Detail::Alignment::CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }

    __aicore__ inline
    void RowmaxTAILTILE(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowmaxUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        if (numElems >= FLOAT_VECTOR_SIZE) {
            AscendC::BlockReduceMax<float, false>(
                tvUbTensor,
                srcUb,
                numRowsRound,
                0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceMax<float, false>(
                rowmaxUb,
                tvUbTensor,
                NpuArch::Detail::Alignment::CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint64_t rowmax_idx = 1; rowmax_idx < (uint64_t)numElems / FLOAT_VECTOR_SIZE; ++rowmax_idx) {
                AscendC::BlockReduceMax<float, false>(
                    tvUbTensor,
                    srcUb[rowmax_idx * FLOAT_VECTOR_SIZE],
                    numRowsRound,
                    0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::BlockReduceMax<float, false>(
                    tvUbTensor[REDUCE_UB_SIZE],
                    tvUbTensor,
                    NpuArch::Detail::Alignment::CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Max<float, false>(rowmaxUb,
                    rowmaxUb,
                    tvUbTensor[REDUCE_UB_SIZE],
                    (uint64_t)0,
                    1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(numElems % FLOAT_VECTOR_SIZE);
            AscendC::BlockReduceMax<float, false>(
                tvUbTensor,
                srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                numRowsRound,
                0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            SetBlockReduceMask(NpuArch::Detail::Alignment::CeilDiv(numElems % FLOAT_VECTOR_SIZE, FLOAT_BLOCK_SIZE));
            if (numElems < FLOAT_VECTOR_SIZE) {
                AscendC::BlockReduceMax<float, false>(rowmaxUb,
                    tvUbTensor,
                    NpuArch::Detail::Alignment::CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::BlockReduceMax<float, false>(tvUbTensor[REDUCE_UB_SIZE],
                    tvUbTensor,
                    NpuArch::Detail::Alignment::CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Max<float, false>(rowmaxUb,
                    rowmaxUb,
                    tvUbTensor[REDUCE_UB_SIZE],
                    (uint64_t)0,
                    1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }

    __aicore__ inline
    void CopySGmToUb(
        AscendC::GlobalTensor<ElementInput> gInput,
        uint32_t sUbOffset,
        uint32_t rowNumCurLoop,
        uint32_t columnNumRound,
        uint32_t columnNumPad)
    {
        AscendC::DataCopy(
            lsUbTensor[sUbOffset],
            gInput,
            AscendC::DataCopyParams(
                rowNumCurLoop, columnNumRound / FLOAT_BLOCK_SIZE,
                (columnNumPad - columnNumRound) / FLOAT_BLOCK_SIZE, 0));
    }

    __aicore__ inline void OperatePreMaskUb(uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        UpCastMask<half, ElementMask>(
            maskUbTensor16,
            maskUbTensor,
            rowNumCurLoop,
            columnNumRound
        );
        AscendC::CompareScalar(
            maskUbTensorUint8,
            maskUbTensor16,
            static_cast<half>(1.0),
            AscendC::CMPMODE::NE,
            REPEAT_SIZE_IN_BYTE / sizeof(half),
            (rowNumCurLoop * columnNumRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE,
            AscendC::UnaryRepeatParams(1, 1, 8, 8)
        );
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<half>(tempMaskTensor, static_cast<half>(1), rowNumCurLoop * columnNumRound);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(maskUbTensor16, maskUbTensorUint8, tempMaskTensor, static_cast<half>(0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, rowNumCurLoop * columnNumRound);
        AscendC::PipeBarrier<PIPE_V>();
        UpCastMask<float, half>(maskUbTensor32, maskUbTensor16, rowNumCurLoop, columnNumRound);
    }

    __aicore__ inline void OperateNextMaskUb(uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        UpCastMask<half, ElementMask>(
            maskUbTensor16,
            maskUbTensor[MAX_UB_S_ELEM_NUM],
            rowNumCurLoop,
            columnNumRound
        );
        UpCastMask<float, half>(
            maskUbTensor32,
            maskUbTensor16,
            rowNumCurLoop,
            columnNumRound
        );
    }


    __aicore__ inline
    void CopyMaskGmToUb(
        AscendC::GlobalTensor<ElementMask> gMask,
        uint32_t columnNum, uint32_t columnNumRound,
        uint32_t maskStride, uint32_t tokenNumPerHead,
        uint32_t proTokenIdx, uint32_t proTokenNum,
        uint32_t integralHeadNum, uint32_t epiTokenNum, bool isNextMask)
    {
        uint32_t innerUbRowOffset = isNextMask ? MAX_UB_S_ELEM_NUM : 0;
        if (proTokenNum != 0) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset], gMask[proTokenIdx * maskStride],
                AscendC::DataCopyExtParams(
                    proTokenNum, columnNum * sizeof(ElementMask),
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
            innerUbRowOffset += proTokenNum * columnNumRound;
        }
        for (uint32_t headIdx = 0; headIdx < integralHeadNum; headIdx++) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset], gMask,
                AscendC::DataCopyExtParams(
                    tokenNumPerHead, columnNum * sizeof(ElementMask),
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
            innerUbRowOffset += tokenNumPerHead * columnNumRound;
        }
        if (epiTokenNum != 0) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset], gMask,
                AscendC::DataCopyExtParams(
                    epiTokenNum, columnNum * sizeof(ElementMask),
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
        }
    }

    __aicore__ inline
    void CalcGmFullShift(int64_t &offsetFull, const LayoutInput &layoutMask,
    uint32_t rowOffer, uint32_t kvSStartIdx, uint32_t maskOffsetThisSubBlock)
    {
        uint32_t fullBlockStart = rowOffer;
        uint32_t gmOffsetFullRow = fullBlockStart + maskOffsetThisSubBlock ;
        uint32_t gmOffsetFullColumn = kvSStartIdx;
        offsetFull = layoutMask.GetOffset(MatrixCoord(gmOffsetFullRow, gmOffsetFullColumn));
    }

    __aicore__ inline
    void CopyFullGmToUb(
        AscendC::GlobalTensor<ElementFull> gFull,
        uint32_t columnNum, uint32_t columnNumRound, uint32_t maskStride, uint32_t tokenNumPerHead,
        uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t integralHeadNum, uint32_t epiTokenNum,
        uint32_t &qNStartIdxVec, uint32_t qNThisSubBlock, uint32_t rowNumCurLoop, uint32_t BIdx,
        uint32_t qHeads, int64_t offsetFull, int64_t pseQ, int64_t pseKv)
    {
        uint32_t innerUbRowOffset = 0;
        int64_t gFullOffset = 0;
        if (proTokenNum != 0) {
            gFullOffset = BIdx * qHeads * pseQ * pseKv + qNStartIdxVec * pseQ * pseKv + offsetFull;
            AscendC::DataCopyPad(
                      fullUbTensor16[innerUbRowOffset], gFull[gFullOffset + proTokenIdx * maskStride],
            AscendC::DataCopyExtParams(
                          proTokenNum, columnNum * sizeof(ElementFull),
                          (maskStride - columnNum) * sizeof(ElementFull),
                          (columnNumRound - columnNum) * sizeof(ElementFull) / BLOCK_SIZE_IN_BYTE, 0),
            AscendC::DataCopyPadExtParams<ElementFull>(false, 0, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
            UpCastMask<float, ElementFull>(fullUbTensor32[innerUbRowOffset], fullUbTensor16[innerUbRowOffset],
                rowNumCurLoop, columnNumRound);
             innerUbRowOffset += proTokenNum * columnNumRound;
        }
        for (uint32_t headIdx = 0; headIdx < integralHeadNum; headIdx++) {
            if (qNThisSubBlock >= HEAD_NUM_2) {
                if (proTokenNum > 0 && headIdx == 0) {
                    qNStartIdxVec++;
                }
                if (headIdx > 0) {
                    qNStartIdxVec++;
                }
            }
            gFullOffset = BIdx * qHeads * pseQ * pseKv + qNStartIdxVec * pseQ * pseKv + offsetFull;
            AscendC::DataCopyPad(
                fullUbTensor16[innerUbRowOffset], gFull[gFullOffset],
                AscendC::DataCopyExtParams(
                        tokenNumPerHead, columnNum * sizeof(ElementFull),
                        (maskStride - columnNum) * sizeof(ElementFull),
                        (columnNumRound - columnNum) * sizeof(ElementFull) / BLOCK_SIZE_IN_BYTE, 0),
                AscendC::DataCopyPadExtParams<ElementFull>(false, 0, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
            UpCastMask<float, ElementFull>(fullUbTensor32[innerUbRowOffset], fullUbTensor16[innerUbRowOffset],
                rowNumCurLoop, columnNumRound);
            innerUbRowOffset += tokenNumPerHead * columnNumRound;
        }
        if (epiTokenNum != 0) {
            if (qNThisSubBlock >= HEAD_NUM_2) {
                qNStartIdxVec++;
            }
            gFullOffset = BIdx * qHeads * pseQ * pseKv + qNStartIdxVec * pseQ * pseKv + offsetFull;
            AscendC::DataCopyPad(
                fullUbTensor16[innerUbRowOffset], gFull[gFullOffset],
                AscendC::DataCopyExtParams(
                        epiTokenNum, columnNum * sizeof(ElementFull),
                        (maskStride - columnNum) * sizeof(ElementFull),
                        (columnNumRound - columnNum) * sizeof(ElementFull) / BLOCK_SIZE_IN_BYTE, 0),
                AscendC::DataCopyPadExtParams<ElementFull>(false, 0, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
            UpCastMask<float, ElementFull>(fullUbTensor32[innerUbRowOffset], fullUbTensor16[innerUbRowOffset],
                rowNumCurLoop, columnNumRound);
        }
    }

    __aicore__ inline
    void Applyfull(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        AscendC::Add<float, false>(
            lsUbTensor[sUbOffset],
            lsUbTensor[sUbOffset],
            fullUbTensor32,
            (uint64_t)0,
            NpuArch::Detail::Alignment::CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void ScaleS(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        AscendC::Muls<float, false>(
            lsUbTensor[sUbOffset],
            lsUbTensor[sUbOffset],
            scaleValue,
            (uint64_t)0,
            NpuArch::Detail::Alignment::CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
            AscendC::UnaryRepeatParams(1, 1, 8, 8));

        AscendC::PipeBarrier<PIPE_V>();
    }

    template<typename ElementMaskDst, typename ElementMaskSrc>
    __aicore__ inline
    void UpCastMask(
        const AscendC::LocalTensor<ElementMaskDst> &maskUbTensorDst,
        const AscendC::LocalTensor<ElementMaskSrc> &maskUbTensorSrc,
        uint32_t rowNumCurLoop,
        uint32_t columnNumRound)
    {
        AscendC::Cast<ElementMaskDst, ElementMaskSrc, false>(
            maskUbTensorDst, maskUbTensorSrc, AscendC::RoundMode::CAST_NONE, (uint64_t)0,
            NpuArch::Detail::Alignment::CeilDiv(
                rowNumCurLoop * columnNumRound, (uint32_t)(REPEAT_SIZE_IN_BYTE / sizeof(ElementMaskDst))),
            AscendC::UnaryRepeatParams(1, 1, 8, 4));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void ApplyMask(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound, uint32_t maskColumnRound,
        uint32_t addMaskUbOffset)
    {
        AscendC::Muls<float, false>(
            maskUbTensor32,
            maskUbTensor32,
            (float)-3e38,
            (uint64_t)0,
            NpuArch::Detail::Alignment::CeilDiv(rowNumCurLoop * maskColumnRound, FLOAT_VECTOR_SIZE),
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        if (maskColumnRound == columnNumRound) {
            AscendC::Add<float, false>(
                lsUbTensor[sUbOffset],
                lsUbTensor[sUbOffset],
                maskUbTensor32,
                (uint64_t)0,
                NpuArch::Detail::Alignment::CeilDiv(rowNumCurLoop * maskColumnRound, FLOAT_VECTOR_SIZE),
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        } else {
            uint32_t loop = maskColumnRound / FLOAT_VECTOR_SIZE;
            for (uint32_t i = 0; i < loop; i++) {
                AscendC::Add<float, false>(lsUbTensor[sUbOffset][addMaskUbOffset + i * FLOAT_VECTOR_SIZE],
                    lsUbTensor[sUbOffset][addMaskUbOffset + i * FLOAT_VECTOR_SIZE],
                    maskUbTensor32[i * FLOAT_VECTOR_SIZE],
                    (uint64_t)0,
                    rowNumCurLoop,
                    AscendC::BinaryRepeatParams(
                        1, 1, 1,
                        columnNumRound / FLOAT_BLOCK_SIZE,
                        columnNumRound / FLOAT_BLOCK_SIZE,
                        maskColumnRound / FLOAT_BLOCK_SIZE));
            }
            if (maskColumnRound % FLOAT_VECTOR_SIZE > 0) {
                SetVecMask(maskColumnRound % FLOAT_VECTOR_SIZE);
                AscendC::Add<float, false>(lsUbTensor[sUbOffset][addMaskUbOffset + loop * FLOAT_VECTOR_SIZE],
                    lsUbTensor[sUbOffset][addMaskUbOffset + loop * FLOAT_VECTOR_SIZE],
                    maskUbTensor32[loop * FLOAT_VECTOR_SIZE],
                    (uint64_t)0,
                    rowNumCurLoop,
                    AscendC::BinaryRepeatParams(
                        1, 1, 1,
                        columnNumRound / FLOAT_BLOCK_SIZE,
                        columnNumRound / FLOAT_BLOCK_SIZE,
                        maskColumnRound / FLOAT_BLOCK_SIZE));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void CalcLocalRowMax(uint32_t sUbOffset, uint32_t rowNumCurLoopRound, uint32_t columnNum, uint32_t columnNumRound,
        uint32_t rowOffset)
    {
        if (columnNum == 512) {
            RowmaxSPECTILE512(
                lsUbTensor[sUbOffset],
                lmUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        } else if (columnNum == 256) {
            RowmaxSPECTILE256(
                lsUbTensor[sUbOffset],
                lmUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        } else {
            RowmaxTAILTILE(
                lsUbTensor[sUbOffset],
                lmUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        }
    }

    __aicore__ inline
    void UpdateGlobalRowMax(uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound, uint32_t columnNum,
        uint32_t columnNumRound, uint32_t dmUbOffsetCurCycle, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            AscendC::DataCopy(
                hmUbTensor[rowOffset],
                lmUbTensor[rowOffset],
                AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(rowNumCurLoop);
            // *** hm = vmax(lm, gm)
            AscendC::Max<float, false>(
                hmUbTensor[rowOffset],
                lmUbTensor[rowOffset],
                gmUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** dm = gm - hm
            AscendC::Sub<float, false>(
                dmUbTensor[dmUbOffsetCurCycle],
                gmUbTensor[rowOffset],
                hmUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** dm = exp(dm)
            AscendC::Exp<float, false>(
                dmUbTensor[dmUbOffsetCurCycle],
                dmUbTensor[dmUbOffsetCurCycle],
                (uint64_t)0,
                1,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::PipeBarrier<PIPE_V>();
        // *** gm = hm
        AscendC::DataCopy(
            gmUbTensor[rowOffset],
            hmUbTensor[rowOffset],
            AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void UpdateGlobalRowMax(AscendC::GlobalTensor<ElementSink> gSink, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound, uint32_t columnNum,
        uint32_t columnNumRound, uint32_t dmUbOffsetCurCycle, uint32_t rowOffset, uint32_t isFirstStackTile, bool isLastStackTile, SinkLoopParam &curLoop)
    {
        if (isFirstStackTile) {
            AscendC::DataCopy(
                hmUbTensor[rowOffset],
                lmUbTensor[rowOffset],
                AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
            // hm = Maxs(hm, sink)
            if constexpr (SINK_MODE == SinkMode::ENABLE){
                if (isLastStackTile) {
                    UpdateRowMaxWithSink(gSink, rowOffset, dmUbOffsetCurCycle, curLoop);
                }
            }
        } else {
            SetVecMask(rowNumCurLoop);
            // *** hm = vmax(lm, gm)
            AscendC::Max<float, false>(
                hmUbTensor[rowOffset],
                lmUbTensor[rowOffset],
                gmUbTensor[rowOffset],
                (uint64_t)0, 1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();

            // hm = Maxs(hm, sink)
            if constexpr (SINK_MODE == SinkMode::ENABLE){
                if (isLastStackTile) {
                    UpdateRowMaxWithSink(gSink, rowOffset, dmUbOffsetCurCycle, curLoop);
                    SetVecMask(rowNumCurLoop);
                }
            }

            // *** dm = gm - hm
            AscendC::Sub<float, false>(
                dmUbTensor[dmUbOffsetCurCycle],
                gmUbTensor[rowOffset],
                hmUbTensor[rowOffset],
                (uint64_t)0,  1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** dm = exp(dm)
            AscendC::Exp<float, false>(
                dmUbTensor[dmUbOffsetCurCycle],
                dmUbTensor[dmUbOffsetCurCycle],
                (uint64_t)0, 1,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::PipeBarrier<PIPE_V>();

        // *** gm = hm
        AscendC::DataCopy(
            gmUbTensor[rowOffset],
            hmUbTensor[rowOffset],
            AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void CalcExp(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound, uint32_t columnNum,
        uint32_t columnNumRound, uint32_t rowOffset)
    {
        // *** hm_block = expand_to_block(hm), 存放于 tv
        AscendC::Brcb(
            tvUbTensor.template ReinterpretCast<uint32_t>(),
            hmUbTensor[rowOffset].template ReinterpretCast<uint32_t>(),
            rowNumCurLoopRound / FLOAT_BLOCK_SIZE,
            AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        // *** ls = ls - hm_block
        for (uint32_t subIdx = 0; subIdx < columnNum / FLOAT_VECTOR_SIZE; ++subIdx) {
            AscendC::Sub<float, false>(
                lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE],
                lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE],
                tvUbTensor,
                (uint64_t)0,
                rowNumCurLoop,
                AscendC::BinaryRepeatParams(
                    1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, 1));
        }
        if (columnNum % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(columnNum % FLOAT_VECTOR_SIZE);
            AscendC::Sub<float, false>(
                lsUbTensor[sUbOffset][columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                lsUbTensor[sUbOffset][columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                tvUbTensor,
                (uint64_t)0,
                rowNumCurLoop,
                AscendC::BinaryRepeatParams(
                    1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, 1));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // *** ls = exp(ls)
        AscendC::Exp<float, false>(
            lsUbTensor[sUbOffset],
            lsUbTensor[sUbOffset],
            (uint64_t)0,
            NpuArch::Detail::Alignment::CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void CalcLocalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoopRound, uint32_t columnNum, uint32_t columnNumRound,
        uint32_t rowOffset)
    {
        // *** ll = rowsum(ls32)
        if (columnNum == 512) {
            RowsumSPECTILE512(
                lsUbTensor[sUbOffset],
                llUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        } else if (columnNum == 256) {
            RowsumSPECTILE256(
                lsUbTensor[sUbOffset],
                llUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        } else {
            RowsumTAILTILE(
                lsUbTensor[sUbOffset],
                llUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        }
    }

    __aicore__ inline
    void UpdateGlobalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound,
        uint32_t dmUbOffsetCurCycle, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            // *** gl = ll
            AscendC::DataCopy(
                glUbTensor[rowOffset],
                llUbTensor[rowOffset],
                AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(rowNumCurLoop);
            // *** gl = dm * gl
            AscendC::Mul<float, false>(
                glUbTensor[rowOffset],
                dmUbTensor[dmUbOffsetCurCycle],
                glUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** gl = ll + gl
            AscendC::Add<float, false>(
                glUbTensor[rowOffset],
                glUbTensor[rowOffset],
                llUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }

    __aicore__ inline
    void UpdateGlobalRowSum(AscendC::GlobalTensor<ElementSink> gSink, uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound,
        uint32_t dmUbOffsetCurCycle, uint32_t rowOffset, uint32_t isFirstStackTile, bool isLastStackTile, SinkLoopParam &curLoop)
    {
        if (isFirstStackTile) {
            // *** gl = ll
            AscendC::DataCopy(
                glUbTensor[rowOffset],
                llUbTensor[rowOffset],
                AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();

        } else {
            SetVecMask(rowNumCurLoop);
            // *** gl = dm * gl
            AscendC::Mul<float, false>(
                glUbTensor[rowOffset],
                dmUbTensor[dmUbOffsetCurCycle],
                glUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();

            // *** gl = ll + gl
            AscendC::Add<float, false>(
                glUbTensor[rowOffset],
                glUbTensor[rowOffset],
                llUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }

        // gl = gl + exp(sink-lm)
        if constexpr (SINK_MODE == SinkMode::ENABLE) {
            if (isLastStackTile) {
                UpdateRowSumWithSink(rowOffset, curLoop.rowNumCurLoop);
            }
        }
    }

    __aicore__ inline
    void DownCastP(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        // *** lp = castfp32to16(ls)
        if (std::is_same<ElementOutput, bfloat16_t>::value) {
            AscendC::Cast<ElementOutput, float, false>(
                lpUbTensor[sUbOffset],
                lsUbTensor[sUbOffset],
                AscendC::RoundMode::CAST_RINT,
                (uint64_t)0,
                NpuArch::Detail::Alignment::CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
                AscendC::UnaryRepeatParams(1, 1, 4, 8));
        } else {
            AscendC::Cast<ElementOutput, float, false>(
                lpUbTensor[sUbOffset],
                lsUbTensor[sUbOffset],
                AscendC::RoundMode::CAST_NONE,
                (uint64_t)0,
                NpuArch::Detail::Alignment::CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
                AscendC::UnaryRepeatParams(1, 1, 4, 8));
        }
    }

    __aicore__ inline
    void CopyPUbToGm(AscendC::GlobalTensor<ElementOutput> gOutput, uint32_t sUbOffset, uint32_t rowNumCurLoop,
        uint32_t columnNumRound, uint32_t columnNumPad)
    {
        AscendC::DataCopy(
            gOutput,
            lpUbTensor[sUbOffset],
            AscendC::DataCopyParams(
                rowNumCurLoop, columnNumRound / BLOCK_SIZE, 0, (columnNumPad - columnNumRound) / BLOCK_SIZE));
    }

    template <bool doTriUMask>
    __aicore__ inline
    void SubCoreCompute(
        AscendC::GlobalTensor<ElementOutput> gOutput, const LayoutOutput &layoutOutput,
        uint32_t rowOffset, uint32_t isFirstStackTile, uint32_t isLastNoMaskStackTile,
        uint32_t isFirstRowLoop, uint32_t isLastRowLoop,
        uint32_t columnNumRound, uint32_t pingpongFlag,
        uint32_t curStackTileMod)
    {
        uint32_t rowNumCurLoop = layoutOutput.shape(0);
        uint32_t rowNumCurLoopRound = NpuArch::Detail::Alignment::RoundUp(rowNumCurLoop, FLOAT_BLOCK_SIZE);
        uint32_t columnNum = layoutOutput.shape(1);
        uint32_t columnNumPad = layoutOutput.stride(0);
        uint32_t sUbOffset = pingpongFlag * MAX_UB_S_ELEM_NUM;
        uint32_t dmUbOffsetCurCycle = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffset;

        if constexpr (LSE_MODE_ == LseMode::OUT_ONLY) {
            // In lse out-only mode, tv is used in the last stack tile to transport lse
            if (isFirstStackTile && isFirstRowLoop) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            }
        }
        CalcLocalRowMax(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        UpdateGlobalRowMax(
            rowNumCurLoop, rowNumCurLoopRound,
            columnNum, columnNumRound,
            dmUbOffsetCurCycle,
            rowOffset,
            isFirstStackTile);

        CalcExp(sUbOffset, rowNumCurLoop, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        if constexpr (!doTriUMask) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
        }

        DownCastP(sUbOffset, rowNumCurLoop, columnNumRound);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);

        CalcLocalRowSum(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);
        CopyPUbToGm(gOutput, sUbOffset, rowNumCurLoop, columnNumRound, columnNumPad);
        if constexpr (!doTriUMask) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
            if (isLastNoMaskStackTile && isLastRowLoop) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        UpdateGlobalRowSum(
            sUbOffset, rowNumCurLoop, rowNumCurLoopRound, dmUbOffsetCurCycle, rowOffset, isFirstStackTile);
    }

    template <bool doTriUMask>
    __aicore__ inline
    void SubCoreCompute(
        AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<ElementSink> gSink, const LayoutOutput &layoutOutput,
        uint32_t rowOffset, uint32_t isFirstStackTile, uint32_t isLastNoMaskStackTile,
        uint32_t isFirstRowLoop, uint32_t isLastRowLoop,
        uint32_t columnNumRound, uint32_t pingpongFlag,
        uint32_t curStackTileMod, SinkLoopParam& sinkLoopParam, bool isLastStackTile, bool isSplitKV, bool startsWithMaskThenNomaskFlag)
    {
        uint32_t rowNumCurLoop = layoutOutput.shape(0);
        uint32_t rowNumCurLoopRound = NpuArch::Detail::Alignment::RoundUp(rowNumCurLoop, FLOAT_BLOCK_SIZE);
        uint32_t columnNum = layoutOutput.shape(1);
        uint32_t columnNumPad = layoutOutput.stride(0);
        uint32_t sUbOffset = pingpongFlag * MAX_UB_S_ELEM_NUM;
        uint32_t dmUbOffsetCurCycle = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffset;

        if constexpr (LSE_MODE_ == LseMode::OUT_ONLY) {
            // In lse out-only mode, tv is used in the last stack tile to transport lse
            if (isFirstStackTile && isFirstRowLoop) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            }
        } else {
            if (isFirstStackTile && isFirstRowLoop && isSplitKV) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            }
        }
        CalcLocalRowMax(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        UpdateGlobalRowMax(
            gSink,
            rowNumCurLoop, rowNumCurLoopRound,
            columnNum, columnNumRound,
            dmUbOffsetCurCycle,
            rowOffset,
            isFirstStackTile,
            isLastStackTile,
            sinkLoopParam);

        CalcExp(sUbOffset, rowNumCurLoop, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        if constexpr (!doTriUMask) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
        }

        DownCastP(sUbOffset, rowNumCurLoop, columnNumRound);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);

        CalcLocalRowSum(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);
        CopyPUbToGm(gOutput, sUbOffset, rowNumCurLoop, columnNumRound, columnNumPad);
        if constexpr (!doTriUMask) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
            if (isLastNoMaskStackTile && isLastRowLoop) {
                if(!startsWithMaskThenNomaskFlag) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        UpdateGlobalRowSum(
            gSink, sUbOffset, rowNumCurLoop, rowNumCurLoopRound, dmUbOffsetCurCycle, rowOffset, isFirstStackTile, isLastStackTile, sinkLoopParam);
    }

    __aicore__ inline
    float ConvertElementSinkToFloat(const ElementSink& rawSinkVal) {
        if constexpr (std::is_same_v<ElementSink, bfloat16_t>) {
            return AscendC::ToFloat(rawSinkVal);
        } else if constexpr (std::is_same_v<ElementSink, half>) {
            return static_cast<float>(rawSinkVal);
        }
    }

    __aicore__ inline
    void SetSinkVecMask(uint32_t elemStart, uint32_t elemEnd) {
        uint64_t mask = 0;
        uint64_t one = 1;
        if (elemStart < 0 || elemStart > elemEnd || elemEnd > FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>((uint64_t)0, (uint64_t)0);
            return;
        }

        for (uint32_t elemIdx = elemStart; elemIdx <= elemEnd; elemIdx++) {
            mask |= (one << elemIdx);
        }
        AscendC::SetVectorMask<int8_t>(0x0, mask);
    }

     __aicore__ inline
    void UpdateRowMaxWithSink(AscendC::GlobalTensor<ElementSink> gSink, uint32_t rowOffset, uint32_t dmUbOffsetCurCycle, SinkLoopParam &curLoop)
    {
        const uint32_t loopStart = curLoop.rowOffsetIoGm;
        const uint32_t loopEnd = curLoop.rowOffsetIoGm + curLoop.rowNumCurLoop - 1;
        const uint32_t qSBlockSize = curLoop.qSBlockSize;

        const uint32_t firstHeadId = loopStart / qSBlockSize;
        const uint32_t lastHeadId = loopEnd / qSBlockSize;

        SetVecMask(curLoop.rowNumCurLoop);
        float zeroNum = 0.0f;
        AscendC::Duplicate<float, false>(lmUbTensor[rowOffset], zeroNum, (uint64_t)0, 1,  1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

        for (uint32_t headId = firstHeadId; headId <= lastHeadId; headId++) {
            uint32_t curHeadQsBlockStartGm = headId * qSBlockSize;
            uint32_t curHeadQsBlockEndGm = curHeadQsBlockStartGm + qSBlockSize - 1U;

            uint32_t headActualStartThisSubCore = AscendC::Std::max(curHeadQsBlockStartGm, loopStart) - curLoop.rowOffsetThisSubBlock - rowOffset;
            uint32_t headActualEndThisSubCore =  AscendC::Std::min(curHeadQsBlockEndGm, loopEnd) - curLoop.rowOffsetThisSubBlock - rowOffset;

            float sinkValue = ConvertElementSinkToFloat(gSink.GetValue(headId));
            uint32_t headRowNumThisSubCore = headActualEndThisSubCore - headActualStartThisSubCore + 1;

            SetSinkVecMask(headActualStartThisSubCore, headActualEndThisSubCore);

            AscendC::UnaryRepeatParams maxsRepeatParams(1, 1, 8, 8);

            // hm = Maxs(hm, sink)
            AscendC::Maxs<float, false>(
                dmUbTensor[dmUbOffsetCurCycle],
                hmUbTensor[rowOffset],
                sinkValue,
                (uint64_t)0, 1,
                maxsRepeatParams
            );

            // sinkTensor
            AscendC::Adds<float, false>(
                lmUbTensor[rowOffset],
                lmUbTensor[rowOffset],
                sinkValue,
                (uint64_t)0,  1,
                maxsRepeatParams
            );

        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        uint64_t mask = static_cast<uint64_t>(curLoop.rowNumCurLoop);

        AscendC::CompareScalar(selMaskUbTensor, hmUbTensor[rowOffset], NEG_INF, AscendC::CMPMODE::EQ,
                mask, 1, AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Select(hmUbTensor[rowOffset], selMaskUbTensor, hmUbTensor[rowOffset], dmUbTensor[dmUbOffsetCurCycle], AscendC::SELMODE::VSEL_CMPMASK_SPR,
                mask, 1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();

    }

    __aicore__ inline
    void UpdateRowSumWithSink(uint32_t rowOffset, uint32_t rowNumCurLoop)
    {
        SetVecMask(rowNumCurLoop);
        // sink = sink - hm
        AscendC::Sub<float, false>(
            lmUbTensor[rowOffset],
            lmUbTensor[rowOffset],
            hmUbTensor[rowOffset],
            (uint64_t)0,  1,
            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();

         // exp(sink-hm)
        AscendC::Exp<float, false>(
            lmUbTensor[rowOffset],
            lmUbTensor[rowOffset],
            (uint64_t)0,
            1,
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();

        // gl+exp(sink -m)
        AscendC::Add<float, false>(
            glUbTensor[rowOffset],
            glUbTensor[rowOffset],
            lmUbTensor[rowOffset],
            (uint64_t)0,  1,
            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }

    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementOutput> gOutputBase,
                    AscendC::GlobalTensor<ElementInput> gInputBase,
                    const LayoutOutput &layoutOutput,
                    const LayoutInput &layoutInput,
                    GemmCoord actualBlockShape,
                    uint32_t isFirstStackTile,
                    uint32_t isLastNoMaskStackTile,
                    uint32_t qSBlockSize,
                    uint32_t qNBlockSize,
                    uint32_t curStackTileMod,
                    uint32_t kvNBlockSize,
                    uint64_t gmOffsetSBase,
                    uint64_t gmOffsetPBase,
                    Arch::CrossCoreFlag qkReady,
                    Arch::CrossCoreFlag softmaxReady)
    {
        Arch::CrossCoreWaitFlag(qkReady);
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = NpuArch::Detail::Alignment::RoundUp(columnNum, BLOCK_SIZE);
        uint32_t columnNumPad = layoutInput.stride(0);

        uint32_t subBlockIdx = ptoSubBlockIdx;
        uint32_t subBlockNum = ptoLanesPerBlock;

        uint32_t kvNSplitSubBlock = kvNBlockSize / subBlockNum;
        uint32_t kvNThisSubBlock = (kvNBlockSize == 1U) ? 0
            : (subBlockIdx == 1U) ? (kvNBlockSize - kvNSplitSubBlock) : kvNSplitSubBlock;

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (kvNBlockSize == 1U) ?
            ((qNBlockSize == 1U) ? 0
                                : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock)
                                                    : qNSplitSubBlock)
            : (kvNThisSubBlock * qNBlockSize);

        uint32_t rowSplitSubBlock = (kvNBlockSize == 1U) ?
            ((qNBlockSize == 1U) ? (qSBlockSize / subBlockNum) : (qSBlockSize * qNSplitSubBlock)) :
            (qSBlockSize * qNBlockSize * kvNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = NpuArch::Detail::Alignment::RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = NpuArch::Detail::Alignment::CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;

        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                uint32_t pingpongFlag = rowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInputBase[offsetInput];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);

                CopySGmToUb(
                    gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop =
                    (delayedRowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutputBase[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);

                // add sink

                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                SubCoreCompute<false>(
                    gOutputCurLoop,
                    layoutOutputCurLoop,
                    rowOffsetCurLoop,
                    isFirstStackTile,
                    isLastNoMaskStackTile,
                    (delayedRowLoopIdx == 0),
                    (delayedRowLoopIdx == rowLoopNum - 1),
                    columnNumRound,
                    pingpongFlag,
                    curStackTileMod
                    );
            }
        }
    }
    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<ElementInput> gInput, AscendC::GlobalTensor<ElementSink> gSink,
        const LayoutOutput &layoutOutput, const LayoutInput &layoutInput, GemmCoord actualBlockShape,
        uint32_t isFirstStackTile, uint32_t isLastNoMaskStackTile, uint32_t qSBlockSize, uint32_t qNBlockSize,
        uint32_t curStackTileMod, bool isLastStackTile, bool isSplitKV = false, bool startsWithMaskTile = false,
        bool startsWithMaskThenNomaskFlag = false)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = NpuArch::Detail::Alignment::RoundUp(columnNum, BLOCK_SIZE);
        uint32_t columnNumPad = layoutInput.stride(0);

        uint32_t subBlockIdx = ptoSubBlockIdx;
        uint32_t subBlockNum = ptoLanesPerBlock;

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1) ?
            0 : (subBlockIdx == 1) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1) ?
            (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;// 在整个块的起始偏移
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = NpuArch::Detail::Alignment::RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = NpuArch::Detail::Alignment::CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;

        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                uint32_t pingpongFlag = rowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                if (startsWithMaskTile && rowLoopIdx == 0) {
                     AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
                CopySGmToUb(
                    gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop =
                    (delayedRowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);

                // add sink
                SinkLoopParam curSinkLoop(rowOffsetIoGm, rowNumCurLoop, qSBlockSize, rowOffsetThisSubBlock);

                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                SubCoreCompute<false>(
                    gOutputCurLoop,
                    gSink,
                    layoutOutputCurLoop,
                    rowOffsetCurLoop,
                    isFirstStackTile,
                    isLastNoMaskStackTile,
                    (delayedRowLoopIdx == 0),
                    (delayedRowLoopIdx == rowLoopNum - 1),
                    columnNumRound,
                    pingpongFlag,
                    curStackTileMod,
                    curSinkLoop,
                    isLastStackTile,
                    isSplitKV,
                    startsWithMaskThenNomaskFlag);
            }
        }
    }

    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<ElementInput> gInput, AscendC::GlobalTensor<ElementSink> gSink,
        AscendC::GlobalTensor<ElementMask> gMask, const LayoutOutput &layoutOutput, const LayoutInput &layoutInput,
        const LayoutInput &layoutMask, GemmCoord actualBlockShape, uint32_t isFirstStackTile, uint32_t qSBlockSize,
        uint32_t qNBlockSize, uint32_t curStackTileMod, Arch::CrossCoreFlag qkReady, uint32_t triUp, uint32_t triDown,
        uint32_t kvSStartIdx, uint32_t kvSEndIdx, bool isLastStackTile, bool isSplitKV = false)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = NpuArch::Detail::Alignment::RoundUp(columnNum, BLOCK_SIZE_IN_BYTE);
        uint32_t columnNumPad = layoutInput.stride(0);
        uint32_t maskStride = layoutMask.stride(0);
        uint32_t subBlockIdx = ptoSubBlockIdx;
        uint32_t subBlockNum = ptoLanesPerBlock;

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1) ?
            0 : (subBlockIdx == 1) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1) ?
            (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ?
            (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;

        uint32_t tokenNumPerHeadThisSubBlock = Min(qSBlockSize, rowActualThisSubBlock);
        uint32_t maskOffsetThisSubBlock = (qNBlockSize == 1) ?
            rowOffsetThisSubBlock : 0;

        // calc mask shift in gm
        uint32_t gmOffsetMaskRow;
        uint32_t gmOffsetMaskColumn;
        uint32_t maskColumn;
        uint32_t addMaskUbOffset;
        if (triUp >= kvSStartIdx) {
            uint32_t triUpRoundDown = NpuArch::Detail::Alignment::RoundDown(triUp, BLOCK_SIZE_IN_BYTE);
            gmOffsetMaskRow = triUp - triUpRoundDown;
            gmOffsetMaskColumn = 0;
            maskColumn = kvSEndIdx - triUpRoundDown;
            addMaskUbOffset = triUpRoundDown - kvSStartIdx;
        } else {
            gmOffsetMaskRow = 0;
            gmOffsetMaskColumn = kvSStartIdx - triUp;
            maskColumn = columnNum;
            addMaskUbOffset = 0;
        }
        uint32_t maskColumnRound = NpuArch::Detail::Alignment::RoundUp(maskColumn, BLOCK_SIZE_IN_BYTE);

        int64_t offsetMask =
            layoutMask.GetOffset(MatrixCoord(gmOffsetMaskRow + maskOffsetThisSubBlock, gmOffsetMaskColumn));
        auto gMaskThisSubBlock = gMask[offsetMask];
        auto layoutMaskThisSubBlock = layoutMask;

        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = NpuArch::Detail::Alignment::RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = NpuArch::Detail::Alignment::CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;

        if (rowActualThisSubBlock == 0) {
            Arch::CrossCoreWaitFlag(qkReady);
            return;
        }

        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                uint32_t pingpongFlag = rowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                // loop 0 mask load before cross core sync
                if (rowLoopIdx == 0) {
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum =
                        Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    CopyMaskGmToUb(
                        gMaskThisSubBlock,
                        maskColumn, maskColumnRound, maskStride,
                        tokenNumPerHeadThisSubBlock,
                        proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum, false);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                    Arch::CrossCoreWaitFlag(qkReady);
                }
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                CopySGmToUb(
                    gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowNumCurLoop = (delayedRowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                UpCastMask<half, ElementMask>(maskUbTensor16, maskUbTensor, rowNumCurLoop, columnNumRound);
                UpCastMask<float, half>(maskUbTensor32, maskUbTensor16, rowNumCurLoop, columnNumRound);

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                ApplyMask(
                    (pingpongFlag * MAX_UB_S_ELEM_NUM),
                    rowNumCurLoop, columnNumRound,
                    maskColumnRound, addMaskUbOffset);
                // next loop mask load
                if (rowLoopIdx < rowLoopNum) {
                    uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                    uint32_t rowNumCurLoop =
                        (rowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum =
                        Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    CopyMaskGmToUb(
                        gMaskThisSubBlock,
                        maskColumn, maskColumnRound, maskStride,
                        tokenNumPerHeadThisSubBlock,
                        proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum, false);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                }
                // online softmax vectorized compute

                // add sink
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                SinkLoopParam curSinkLoop(rowOffsetIoGm, rowNumCurLoop, qSBlockSize, rowOffsetThisSubBlock);

                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                SubCoreCompute<true>(
                    gOutputCurLoop,
                    gSink,
                    layoutOutputCurLoop,
                    rowOffsetCurLoop,
                    isFirstStackTile,
                    0,
                    (delayedRowLoopIdx == 0),
                    (delayedRowLoopIdx == rowLoopNum - 1),
                    columnNumRound,
                    pingpongFlag,
                    curStackTileMod,
                    curSinkLoop,
                    isLastStackTile,
                    isSplitKV,
                    false);
            }
        }
    }

    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<ElementInput> gInput, AscendC::GlobalTensor<ElementSink> gSink,
        AscendC::GlobalTensor<ElementMask> gMask, const LayoutOutput &layoutOutput, const LayoutInput &layoutInput,
        const LayoutInput &layoutMask, GemmCoord actualBlockShape, uint32_t isFirstStackTile, uint32_t qSBlockSize,
        uint32_t qNBlockSize, uint32_t curStackTileMod, Arch::CrossCoreFlag qkReady,
        int32_t kvSStartIdx, bool doTriUPreMask,  bool doTriUNextMask, int32_t preTokenStartLen,
        int32_t preTokenEndLen, int32_t nextTokenStartLen, int32_t nextTokenEndLen, bool isLastStackTile)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = NpuArch::Detail::Alignment::RoundUp(columnNum, BLOCK_SIZE_IN_BYTE);
        uint32_t columnNumPad = layoutInput.stride(0);
        uint32_t maskStride = layoutMask.stride(0);
        uint32_t subBlockIdx = ptoSubBlockIdx;
        uint32_t subBlockNum = ptoLanesPerBlock;

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1) ?
            0 : (subBlockIdx == 1) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1) ?
            (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ?
            (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;

        uint32_t tokenNumPerHeadThisSubBlock = Min(qSBlockSize, rowActualThisSubBlock);
        uint32_t maskOffsetThisSubBlock = (qNBlockSize == 1) ?
            rowOffsetThisSubBlock : 0;

        uint32_t addMaskUbOffset = 0;

        uint32_t gmOffsetMaskRowPre;
        uint32_t gmOffsetMaskColumnPre;
        uint32_t maskColumnPre;

        uint32_t gmOffsetMaskRowNext;
        uint32_t gmOffsetMaskColumnNext;
        uint32_t maskColumnNext;
        if (doTriUPreMask) {
            if (preTokenStartLen > kvSStartIdx) {
                gmOffsetMaskRowPre = preTokenStartLen - kvSStartIdx - 1;
                gmOffsetMaskColumnPre = 0;
                maskColumnPre = columnNumRound;
            } else {
                gmOffsetMaskRowPre = 0;
                gmOffsetMaskColumnPre = kvSStartIdx - preTokenStartLen + 1;
                maskColumnPre = columnNumRound;
            }
        }
        if (doTriUNextMask) {
            if (nextTokenStartLen > kvSStartIdx) {
                gmOffsetMaskRowNext = nextTokenStartLen - kvSStartIdx;
                gmOffsetMaskColumnNext = 0;
                maskColumnNext = columnNumRound;
            } else {
                gmOffsetMaskRowNext = 0;
                gmOffsetMaskColumnNext = kvSStartIdx - nextTokenStartLen;
                maskColumnNext = columnNumRound;
            }
        }
        uint32_t columnNumRoundPre = NpuArch::Detail::Alignment::RoundUp(maskColumnPre, BLOCK_SIZE_IN_BYTE);
        int64_t offsetMaskPre =
                    layoutMask.GetOffset(MatrixCoord(gmOffsetMaskRowPre + maskOffsetThisSubBlock, gmOffsetMaskColumnPre));
        auto gMaskThisSubBlockPre = gMask[offsetMaskPre];

        uint32_t columnNumRoundNext = NpuArch::Detail::Alignment::RoundUp(maskColumnNext, BLOCK_SIZE_IN_BYTE);
        int64_t offsetMaskNext =
                    layoutMask.GetOffset(MatrixCoord(gmOffsetMaskRowNext + maskOffsetThisSubBlock, gmOffsetMaskColumnNext));
        auto gMaskThisSubBlockNext = gMask[offsetMaskNext];
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = NpuArch::Detail::Alignment::RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = NpuArch::Detail::Alignment::CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;

        if (rowActualThisSubBlock == 0) {
            Arch::CrossCoreWaitFlag(qkReady);
            return;
        }

        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                uint32_t pingpongFlag = rowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                // loop 0 mask load before cross core sync
                if (rowLoopIdx == 0) {
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum =
                        Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    if (doTriUPreMask && doTriUNextMask) {
                        CopyMaskGmToUb(
                            gMaskThisSubBlockPre,
                            maskColumnPre, columnNumRoundPre, maskStride,
                            tokenNumPerHeadThisSubBlock,
                            proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum, false);
                    } else if (doTriUPreMask){
                        CopyMaskGmToUb(
                            gMaskThisSubBlockPre,
                            maskColumnPre, columnNumRoundPre, maskStride,
                            tokenNumPerHeadThisSubBlock,
                            proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum, false);
                    } else if (doTriUNextMask) {
                        CopyMaskGmToUb(
                            gMaskThisSubBlockNext,
                            maskColumnNext, columnNumRoundNext, maskStride,
                            tokenNumPerHeadThisSubBlock,
                            proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum, true);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                    Arch::CrossCoreWaitFlag(qkReady);
                }
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                CopySGmToUb(
                    gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowNumCurLoop = (delayedRowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                if (doTriUPreMask && doTriUNextMask)  {
                    OperatePreMaskUb(rowNumCurLoop, columnNumRound);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                    ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                    ApplyMask(
                        (pingpongFlag * MAX_UB_S_ELEM_NUM),
                        rowNumCurLoop, columnNumRound,
                        columnNumRoundPre, addMaskUbOffset);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
                    if (doTriUNextMask) {
                        uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                        // the token num of the prologue part
                        uint32_t proTokenNum =
                            Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                        // the token num of the epilogue part
                        uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                        // the number of integral heads within a cycle
                        uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
                        CopyMaskGmToUb(
                            gMaskThisSubBlockNext,
                            maskColumnNext, columnNumRoundNext, maskStride,
                            tokenNumPerHeadThisSubBlock,
                            proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum, true);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
                        OperateNextMaskUb(rowNumCurLoop, columnNumRound);
                        ApplyMask(
                            (pingpongFlag * MAX_UB_S_ELEM_NUM),
                            rowNumCurLoop, columnNumRound,
                            columnNumRoundNext, addMaskUbOffset);
                    }
                } else if (doTriUPreMask)  {
                    OperatePreMaskUb(rowNumCurLoop, columnNumRound);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                    ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                    ApplyMask(
                        (pingpongFlag * MAX_UB_S_ELEM_NUM),
                        rowNumCurLoop, columnNumRound,
                        columnNumRoundPre, addMaskUbOffset);
                } else if (doTriUNextMask)  {
                    OperateNextMaskUb(rowNumCurLoop, columnNumRound);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                    ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                    ApplyMask(
                        (pingpongFlag * MAX_UB_S_ELEM_NUM),
                        rowNumCurLoop, columnNumRound,
                        columnNumRoundNext, addMaskUbOffset);
                }
                // next loop mask load
                if (rowLoopIdx < rowLoopNum) {
                    uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                    uint32_t rowNumCurLoop =
                        (rowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum =
                        Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    if (doTriUPreMask && doTriUNextMask) {
                        CopyMaskGmToUb(
                            gMaskThisSubBlockPre,
                            maskColumnPre, columnNumRoundPre, maskStride,
                            tokenNumPerHeadThisSubBlock,
                            proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum, false);
                    } else if (doTriUPreMask){
                        CopyMaskGmToUb(
                            gMaskThisSubBlockPre,
                            maskColumnPre, columnNumRoundPre, maskStride,
                            tokenNumPerHeadThisSubBlock,
                            proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum, false);
                    } else if (doTriUNextMask) {
                        CopyMaskGmToUb(
                            gMaskThisSubBlockNext,
                            maskColumnNext, columnNumRoundNext, maskStride,
                            tokenNumPerHeadThisSubBlock,
                            proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum, true);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                }
                // online softmax vectorized compute
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                // add sink
                SinkLoopParam curSinkLoop(rowOffsetIoGm, rowNumCurLoop, qSBlockSize, rowOffsetThisSubBlock);
                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                SubCoreCompute<true>(
                    gOutputCurLoop,
                    gSink,
                    layoutOutputCurLoop,
                    rowOffsetCurLoop,
                    isFirstStackTile,
                    0,
                    (delayedRowLoopIdx == 0),
                    (delayedRowLoopIdx == rowLoopNum - 1),
                    columnNumRound,
                    pingpongFlag,
                    curStackTileMod,
                    curSinkLoop,
                    isLastStackTile,
                    false,
                    false);
            }
        }
    }

    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<ElementInput> gInput,
        AscendC::GlobalTensor<ElementSink> gSink, AscendC::GlobalTensor<ElementFull> gFull,
        const LayoutOutput &layoutOutput, const LayoutInput &layoutInput,
        const LayoutInput &layoutMask, GemmCoord actualBlockShape, uint32_t isFirstStackTile, uint32_t qSBlockSize,
        uint32_t qNBlockSize, uint32_t curStackTileMod, Arch::CrossCoreFlag qkReady, uint32_t rowOffer, uint32_t kvSStartIdx,
        uint32_t kvSEndIdx, uint32_t qNStartIdx, uint32_t BIdx, uint32_t qHeads, int64_t pseQ, int64_t pseKv, bool isLastStackTile)
    {
         uint32_t rowNum = actualBlockShape.m();   //当前fullmask块的总行数
         uint32_t columnNum = actualBlockShape.n();  //fullmask块的总列数
         uint32_t columnNumRound = NpuArch::Detail::Alignment::RoundUp(columnNum, BLOCK_SIZE_IN_BYTE);  //列数向上对齐到32B
         uint32_t columnNumPad = layoutInput.stride(0);   //输入张量stride
         uint32_t maskStride = layoutMask.stride(0);     //掩码张量
         uint32_t subBlockIdx = ptoSubBlockIdx;  // current AIV lane
         uint32_t subBlockNum = ptoLanesPerBlock;
         uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum; // 1  每核分到的头数
         uint32_t qNThisSubBlock = (qNBlockSize == 1) ?
             0 : (subBlockIdx == 1) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock; // 1  本核的头索引
         uint32_t rowSplitSubBlock = (qNBlockSize == 1) ?
             (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);  //每核分到的行数
         uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ?
             (rowNum - rowSplitSubBlock) : rowSplitSubBlock;  //本核实际要处理的行数
         uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;  //本核子块在fullmask的行偏移
         uint32_t tokenNumPerHeadThisSubBlock = Min(qSBlockSize, rowActualThisSubBlock);  //
         uint32_t maskOffsetThisSubBlock = (qNBlockSize == 1) ? rowOffsetThisSubBlock : 0;  //
         // calc qNstartIdx per vec core
         uint32_t qNStartIdxVec =  (qNBlockSize == 1) ? qNStartIdx : (subBlockIdx == 1) ? (qNStartIdx + qNSplitSubBlock) : qNStartIdx;
         // calc Full  shift in gm
         int64_t offsetFull = 0;
         CalcGmFullShift(offsetFull, layoutMask, rowOffer, kvSStartIdx, maskOffsetThisSubBlock);
         uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
         uint32_t rowNumTile = NpuArch::Detail::Alignment::RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
         rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
         uint32_t rowLoopNum = NpuArch::Detail::Alignment::CeilDiv(rowActualThisSubBlock, rowNumTile);
         uint32_t preLoad = 1;

         if (rowActualThisSubBlock == 0) {
             Arch::CrossCoreWaitFlag(qkReady);
             return;
         }
         uint32_t proTokenIdx;
         uint32_t proTokenNum;
         uint32_t integralHeadNum;
         uint32_t epiTokenNum;
         for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
             if (rowLoopIdx < rowLoopNum) {
                uint32_t pingpongFlag = rowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                // loop 0 mask load before cross core sync
                if (rowLoopIdx == 0) {
                    // the token idx of the start token of the prologue part
                    proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    proTokenNum =
                    Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    CopyFullGmToUb(
                            gFull,
                            columnNum, columnNumRound, maskStride,
                            tokenNumPerHeadThisSubBlock, proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum,
                            qNStartIdxVec, qNThisSubBlock, rowNumCurLoop, BIdx, qHeads, offsetFull, pseQ, pseKv);
                    Arch::CrossCoreWaitFlag(qkReady);
                }
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                CopySGmToUb(
                    gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
             }
             if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowNumCurLoop = (delayedRowLoopIdx == rowLoopNum - 1) ?
                (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                Applyfull((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                    // next loop mask load
                    // online softmax vectorized compute

                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                SinkLoopParam curSinkLoop(rowOffsetIoGm, rowNumCurLoop, qSBlockSize, rowOffsetThisSubBlock);

                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                SubCoreCompute<true>(
                    gOutputCurLoop,
                    gSink,
                    layoutOutputCurLoop,
                    rowOffsetCurLoop,
                    isFirstStackTile,
                    0,
                    delayedRowLoopIdx == 0,
                    delayedRowLoopIdx == rowLoopNum - 1,
                    columnNumRound,
                    pingpongFlag,
                    curStackTileMod,
                    curSinkLoop,
                    isLastStackTile,
                    false,
                    false);
                if (rowLoopIdx < rowLoopNum) {
                    uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                    uint32_t rowNumCurLoop =
                    (rowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                    // the token idx of the start token of the prologue part
                    proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    proTokenNum =
                    Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    if ((qNThisSubBlock >= HEAD_NUM_2) && (proTokenIdx == 0) && proTokenNum > 0) {
                        qNStartIdxVec++;
                    }
                    // the number of integral heads within a cycle
                    integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    if ((qNThisSubBlock >= HEAD_NUM_2) && integralHeadNum > 0 && proTokenNum == 0) {
                        qNStartIdxVec++;
                    }
                    // the token num of the epilogue part
                    epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

                    CopyFullGmToUb(
                        gFull,
                        columnNum, columnNumRound, maskStride,
                        tokenNumPerHeadThisSubBlock, proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum,
                        qNStartIdxVec, qNThisSubBlock, rowNumCurLoop, BIdx, qHeads, offsetFull, pseQ, pseKv);   // 当前循环要搬的行数
                }
             }
         }
     }

private:
    uint32_t ptoSubBlockIdx = 0;
    uint32_t ptoLanesPerBlock = 1;
    uint32_t kvheadIdx = 0;
    float scaleValue;
    AscendC::LocalTensor<float> lsUbTensor;
    AscendC::LocalTensor<ElementOutput> lpUbTensor;
    AscendC::LocalTensor<ElementMask> maskUbTensor;
    AscendC::LocalTensor<uint8_t> maskUbTensorUint8;
    AscendC::LocalTensor<half> maskUbTensor16;
    AscendC::LocalTensor<float> maskUbTensor32;
    AscendC::LocalTensor<ElementFull> fullUbTensor16;
    AscendC::LocalTensor<float> fullUbTensor32;
    AscendC::LocalTensor<float> lmUbTensor;
    AscendC::LocalTensor<float> hmUbTensor;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<float> llUbTensor;
    AscendC::LocalTensor<float> tvUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<half> tempMaskTensor;
    AscendC::LocalTensor<uint8_t> selMaskUbTensor;
};
}

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_HPP
