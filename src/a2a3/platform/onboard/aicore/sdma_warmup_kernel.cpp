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
 * One-shot SDMA control-path warmup, launched once per Worker during
 * `provision_dma_workspace` (see DeviceRunnerBase::launch_sdma_warmup_kernel).
 *
 * Without it, the first TPREFETCH_ASYNC of a run pays a cold STARS submit-queue
 * publication on its channel. This kernel walks that path once per channel:
 * touch the channel-info cache line, read the head/tail pair, rewrite the SQE
 * words in place (an identity write -- no transfer is submitted), then ring the
 * doorbell.
 *
 * Unlike the resident executor (kernel.cpp), this is built as a VECTOR-ONLY ELF
 * so `block_dim` maps 1:1 onto AIVs. A MIX launch would cap `get_block_idx()` at
 * the cube count and make two AIVs share a stride slot.
 *
 * Channels are walked grid-stride rather than one-per-block, because the cold
 * cost is per-channel and core-agnostic: warming a channel from any core makes
 * it warm for every core (measured -- one block warming all 48 leaves a
 * subsequent 48-block launch fully warm). So `block_dim` only decides how the
 * walk is split, and `kSdmaWarmupBlockDim` keeps it independent of the channel
 * count and of the AIV count of whatever chip this runs on.
 */

#include <cstdint>

// Must be the pto entry point, not sdma_async_intrin.hpp directly: that header
// is pulled in mid-way through pto-inst.hpp's own include graph, so reaching it
// first leaves pto::comm::sdma half-declared when async_event_impl.hpp uses it.
#include <pto/pto-inst.hpp>

#include "common/sdma_warmup_layout.h"

// CANN resolves handle-registered entries by the `<stub>_<tilingKey>_mix_aiv`
// symbol convention and this launch passes tilingKey 0. The suffix is required
// even for a vector-only ELF: a plainly-named entry makes rtRegisterAllKernel
// fail with 107000.
extern "C" __global__ AICORE void
sdma_warmup_kernel_0_mix_aiv(__gm__ uint8_t *workspace, __gm__ uint8_t *status, uint64_t channel_count) {
    const uint32_t count = static_cast<uint32_t>(channel_count);
    const uint32_t stride = static_cast<uint32_t>(get_block_num());

    // syncId 0 on every core: it only names the event flag of the paired
    // set_flag/wait_flag around the doorbell store, event flags are per-core, and
    // each call drains its own pair before returning -- so neither concurrent
    // cores nor successive iterations on one core can collide.
    for (uint32_t channel_idx = static_cast<uint32_t>(get_block_idx()); channel_idx < count; channel_idx += stride) {
        const bool warmed = pto::comm::sdma::detail::WarmupSdmaControlPathForAiv(workspace, channel_idx, 0U);

        __gm__ uint32_t *slot =
            reinterpret_cast<__gm__ uint32_t *>(status + channel_idx * kSdmaWarmupStatusStrideBytes);
        *slot = warmed ? kSdmaWarmupStatusOk : kSdmaWarmupStatusFailed;
        pipe_barrier(PIPE_ALL);
        dcci(reinterpret_cast<__gm__ void *>(slot), cache_line_t::SINGLE_CACHE_LINE);
    }
    dsb(DSB_DDR);
}
