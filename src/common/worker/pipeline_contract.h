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
 * Admission check for the PipelineContract a host runtime declares.
 *
 * Lives beside the contract rather than inside ChipWorker so the rules a build
 * will honor are stated in one place and can be exercised on their own.
 */

#ifndef SRC_COMMON_WORKER_PIPELINE_CONTRACT_H_
#define SRC_COMMON_WORKER_PIPELINE_CONTRACT_H_

#include <cstdint>

#include "pto_runtime_c_api.h"

/**
 * Whether this build can honor `contract`.
 *
 * Rejects a contract whose ABI version is not the one compiled in, that
 * declares more resources than fit, that asks for a pipeline depth other than
 * 1, or whose resources carry an unspecified or out-of-range kind, an
 * out-of-range class, or a non-zero reserved `bytes_per_copy`.
 *
 * The unspecified-kind rule is what catches a `resource_count` larger than the
 * entries a runtime actually filled in: the trailing entries are still zeroed,
 * and zero is not a resource. Repeated kinds stay legal, so a runtime may
 * declare several resources of one kind.
 */
inline bool is_valid_depth1_pipeline_contract(const PipelineContract *contract) {
    if (contract == nullptr || contract->abi_version != PTO_PIPELINE_CONTRACT_ABI_VERSION ||
        contract->resource_count > PTO_PIPELINE_MAX_RESOURCES || contract->pipeline_depth != 1) {
        return false;
    }
    for (uint32_t i = 0; i < contract->resource_count; ++i) {
        const PipelineResource &resource = contract->resources[i];
        if (resource.kind == PTO_PIPELINE_KIND_UNSPECIFIED || resource.kind > PTO_PIPELINE_AICORE_STREAM ||
            resource.resource_class > PTO_PIPELINE_EXEC_HANDLE || resource.bytes_per_copy != 0) {
            return false;
        }
    }
    return true;
}

#endif  // SRC_COMMON_WORKER_PIPELINE_CONTRACT_H_
