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

#include "affinity_allowed_file.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace pto::a5 {
namespace {

constexpr char kEnvAffinityPlanPath[] = "SIMPLER_AICPU_AFFINITY_PLAN";
constexpr char kDefaultAffinityPlanRelative[] = "build/config/aicpu_affinity_plan.json";

bool is_supported_affinity_plan_source(const std::string &source) {
    return source == "atomic-flag-orch+cond-die-v1" || source == "manual" || source == "auto-first-run" ||
           source == "pool-too-small-contiguous" || source == "probe-failed-contiguous";
}

std::string affinity_cpus_side_path(int32_t device_id) {
    const char *env = std::getenv(kEnvAffinityPlanPath);
    std::string plan = (env != nullptr && env[0] != '\0') ? env : kDefaultAffinityPlanRelative;
    const std::string suffix = ".json";
    if (plan.size() >= suffix.size() && plan.compare(plan.size() - suffix.size(), suffix.size(), suffix) == 0) {
        plan.resize(plan.size() - suffix.size());
    }
    plan += ".";
    plan += std::to_string(device_id);
    plan += ".cpus";
    return plan;
}

bool parse_csv_int_list(const std::string &text, std::vector<int32_t> &out) {
    out.clear();
    const char *p = text.c_str();
    while (*p != '\0') {
        while (*p == ',' || std::isspace(static_cast<unsigned char>(*p)))
            ++p;
        if (*p == '\0') break;
        char *end = nullptr;
        const long value = std::strtol(p, &end, 10);
        if (end == p || value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max()) {
            return false;
        }
        out.push_back(static_cast<int32_t>(value));
        p = end;
    }
    return !out.empty();
}

}  // namespace

bool load_affinity_cpus_side_file(
    const char *soc_name, int32_t device_id, std::vector<int32_t> &allowed_cpus, std::string &plan_source,
    bool &pool_too_small
) {
    allowed_cpus.clear();
    plan_source.clear();
    pool_too_small = false;
    if (soc_name == nullptr || soc_name[0] == '\0' || device_id < 0) return false;

    std::ifstream input(affinity_cpus_side_path(device_id));
    if (!input) return false;

    std::string file_soc;
    std::string cpus_text;
    bool saw_pool = false;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) return false;
        const std::string key = line.substr(0, eq);
        const std::string value = line.substr(eq + 1);
        if (key == "soc") {
            file_soc = value;
        } else if (key == "source") {
            plan_source = value;
        } else if (key == "pool_too_small") {
            pool_too_small = (value == "1" || value == "true");
            saw_pool = true;
        } else if (key == "cpus") {
            cpus_text = value;
        } else {
            return false;
        }
    }

    if (file_soc != soc_name) return false;
    if (!is_supported_affinity_plan_source(plan_source)) return false;
    if (!parse_csv_int_list(cpus_text, allowed_cpus) || allowed_cpus.size() < 2) return false;
    {
        std::vector<int32_t> unique = allowed_cpus;
        std::sort(unique.begin(), unique.end());
        if (std::unique(unique.begin(), unique.end()) != unique.end()) return false;
    }
    if (!saw_pool) {
        pool_too_small = plan_source == "pool-too-small-contiguous" || plan_source == "probe-failed-contiguous";
    }
    return true;
}

}  // namespace pto::a5
