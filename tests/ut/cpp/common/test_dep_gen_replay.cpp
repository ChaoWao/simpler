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

#include "dep_gen_replay.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>

#include <unistd.h>

#include "common/dep_gen.h"
#include "tensormap_and_ringbuffer/task_id_encoding.h"

namespace {

std::filesystem::path output_path() {
    return std::filesystem::temp_directory_path() /
           ("simpler_dep_gen_invalid_flags_" + std::to_string(::getpid()) + ".json");
}

}  // namespace

TEST(DepGenReplayTest, RejectsUnknownExplicitDepFlagBits) {
    const std::filesystem::path path = output_path();
    std::filesystem::remove(path);

    DepGenRecord record{};
    record.task_id = simpler::tmr::make_task_id(0, 2).raw;
    record.explicit_dep_count = 1;
    record.explicit_deps[0] = simpler::tmr::make_task_id(0, 1).raw;
    record.explicit_dep_kinds[0] = 0xff;

    EXPECT_EQ(dep_gen_replay_emit_deps_json(&record, 1, path.c_str()), -7);
    EXPECT_FALSE(std::filesystem::exists(path));
}
