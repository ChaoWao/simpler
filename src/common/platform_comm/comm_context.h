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
 * CommContext — device-side distributed communication context.
 *
 * This struct is the ABI contract between host (comm_hccl.cpp / comm_sim.cpp)
 * and device kernels. PTO communication instructions (TREDUCE, TGET, TPUT)
 * access remote data through the GVA addresses in windowsIn[]/windowsOut[]
 * via MTE2 DMA.
 *
 * Host fills the struct from scratch:
 *   - comm_hccl.cpp (Path D): allocates a per-rank symmetric pool via the
 *     public ACL IPC primitives (aclrtMalloc + aclrtIpcMemGetExportKey +
 *     SetImportPid + ImportByKey), then writes rankId / rankNum / winSize /
 *     windowsIn[]. No HCCL-private struct is reinterpret_cast'd here; the
 *     layout is owned end-to-end by simpler.
 *   - comm_sim.cpp: same shape, filled with malloc'd host pointers.
 *
 * The leading layout through windowsOut is shared with pto-isa's parallel
 * HcclDeviceContext declaration. Simpler-owned transport fields are appended
 * after that compatible prefix.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

static constexpr uint32_t COMM_MAX_RANK_NUM = 64;

struct CommContext {
    uint64_t sdmaWorkSpace;
    uint64_t sdmaWorkSpaceSize;

    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t windowsIn[COMM_MAX_RANK_NUM];
    uint64_t windowsOut[COMM_MAX_RANK_NUM];

    uint64_t urmaWorkSpace;
    uint64_t urmaWorkSpaceSize;
    // Byte displacement of this context's windowsIn[] view from the symmetric
    // memory registered in urmaWorkSpace. Zero for the base context; non-zero
    // for a derived arena slice.
    uint64_t urmaWindowOffset;
    // Map a domain-local rank to the rank used by the communicator-scoped
    // URMA workspace.  Base contexts contain the identity map; derived
    // contexts may select/reorder any communicator ranks.
    uint32_t urmaRankMap[COMM_MAX_RANK_NUM];
};

// The struct itself lives in this repo, so on the surface these asserts look
// like they only check that we do not contradict ourselves. Their real value
// is that this layout is consumed by *two* out-of-band parties that never see
// this header at the same time:
//
//   1. The pto-isa repo carries a parallel declaration (HcclDeviceContext)
//      that must be prefix-compatible with this struct -- pto-isa kernels read
//      windowsIn[]/winSize/rankId via that mirror. Field names may differ, but
//      any insert/reorder before the simpler-owned tail that is not matched in
//      pto-isa silently shifts
//      the device-side field offsets and corrupts MTE2 reads. The locks below
//      pin our side; pto-isa should add its own mirror asserts.
//
//   2. Device kernels (AICore / AICPU) compiled with CCEC may apply slightly
//      different alignment rules than host gcc. A host-side sizeof/offset
//      lock is a necessary-but-not-sufficient guard.
//
// Treat the numbers below as a tripwire: changing them is a deliberate act
// that forces the editor to coordinate the matching change on the pto-isa
// side, not a routine "oh I just added a field" edit.
static_assert(std::is_trivially_copyable_v<CommContext>, "CommContext must remain trivially copyable");
static_assert(std::is_standard_layout_v<CommContext>, "CommContext must remain standard layout");
static_assert(sizeof(CommContext) == 1336, "CommContext size shifted");
static_assert(offsetof(CommContext, sdmaWorkSpace) == 0, "CommContext layout drift");
static_assert(offsetof(CommContext, sdmaWorkSpaceSize) == 8, "CommContext layout drift");
static_assert(offsetof(CommContext, rankId) == 16, "CommContext layout drift");
static_assert(offsetof(CommContext, rankNum) == 20, "CommContext layout drift");
static_assert(offsetof(CommContext, winSize) == 24, "CommContext layout drift");
static_assert(offsetof(CommContext, windowsIn) == 32, "CommContext layout drift");
static_assert(offsetof(CommContext, windowsOut) == 544, "CommContext layout drift");
static_assert(offsetof(CommContext, urmaWorkSpace) == 1056, "CommContext layout drift");
static_assert(offsetof(CommContext, urmaWorkSpaceSize) == 1064, "CommContext layout drift");
static_assert(offsetof(CommContext, urmaWindowOffset) == 1072, "CommContext layout drift");
static_assert(offsetof(CommContext, urmaRankMap) == 1080, "CommContext layout drift");
