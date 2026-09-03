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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pto::a5 {

// Reads the line-oriented companion file written by rtt_die_preflight:
//   build/config/aicpu_affinity_plan.<device_id>.cpus
//   (or sibling of SIMPLER_AICPU_AFFINITY_PLAN)
// Format:
//   soc=<name>
//   source=<token>
//   pool_too_small=0|1
//   cpus=<csv>
bool load_affinity_cpus_side_file(
    const char *soc_name, int32_t device_id, std::vector<int32_t> &allowed_cpus, std::string &plan_source,
    bool &pool_too_small
);

}  // namespace pto::a5
