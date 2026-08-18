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

#include <filesystem>
#include <string>
#include <system_error>

#include "common/unified_log.h"

/**
 * Output path (JSON Lines, one object per prepare pass) for the host phase
 * records a run collected.
 *
 * Only the platform runner knows the per-case directory, so it is the runner that
 * names the file. Filename is fixed — the directory is the per-task uniqueness
 * boundary, mirroring make_deps_json_path() / make_pmu_csv_path().
 */
inline std::string make_host_phase_records_path(const std::string &output_dir) {
    std::filesystem::path dir(output_dir);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        LOG_WARN(
            "Failed to create host phase records output directory %s: %s", output_dir.c_str(), ec.message().c_str()
        );
    }
    return (dir / "host_phase_records.jsonl").string();
}
