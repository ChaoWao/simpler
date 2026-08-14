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

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

namespace reader_fanout_test {

constexpr int32_t FUNC_READER = 0;
constexpr int32_t FUNC_WRITER = 1;

}  // namespace reader_fanout_test

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 6,  // X, reader outputs, padding, reader_count, spin_iters, host access mode
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const ChipTensor &x = orch_args.tensor(0).ref();
    const ChipTensor &reader_outputs = orch_args.tensor(1).ref();
    const ChipTensor &padding = orch_args.tensor(2).ref();
    const int32_t reader_count = static_cast<int32_t>(orch_args.scalar(0));
    const int32_t spin_iters = static_cast<int32_t>(orch_args.scalar(1));
    const int32_t host_access = static_cast<int32_t>(orch_args.scalar(2));

    if (reader_count <= 0 || reader_count > static_cast<int32_t>(reader_outputs.shapes[0])) {
        rt_report_fatal(
            PTO2_ERROR_INVALID_ARGS, "reader_count=%d output_size=%u", reader_count, reader_outputs.shapes[0]
        );
        return;
    }

    // Each reader writes a disjoint result, but all of them read the same X.
    // They therefore depend on a prior X writer (if any), never on each other.
    for (int32_t i = 0; i < reader_count; ++i) {
        uint32_t shape[] = {1, reader_outputs.shapes[1]};
        uint32_t offset[] = {static_cast<uint32_t>(i), 0};
        ChipTensor output = reader_outputs.view(shape, offset);
        CoreTaskArgs args;
        args.add_tracked_input(x);
        args.add_output(output);
        args.add_scalar(spin_iters);
        rt_submit_aiv_task(reader_fanout_test::FUNC_READER, args);
    }

    if (host_access == 1) {
        // Host-side writes have the same WAR obligation. They wait for each
        // reader task itself, but do not wait for those readers' consumers.
        const uint32_t index[] = {0};
        set_tensor_data<float>(x, 1, index, 2.0f);
        set_tensor_data<float>(padding, 1, index, 3.0f);

    } else {
        if (host_access == 2) {
            // A host read waits only for the last writer. The submitted readers
            // above are peers, so waiting for them would deadlock HBG before
            // device execution starts.
            const uint32_t index[] = {0};
            const float observed = get_tensor_data<float>(x, 1, index);
            if (observed != 1.0f) {
                rt_report_fatal(PTO2_ERROR_INVALID_ARGS, "host read observed %.1f, expected 1.0", observed);
                return;
            }
        }
        // A pure device overwrite does not read X, but it must wait for every
        // live reader before changing the storage.
        CoreTaskArgs writer_args;
        writer_args.add_output(x);
        writer_args.add_output(padding);
        rt_submit_aiv_task(reader_fanout_test::FUNC_WRITER, writer_args);
    }

    // A later reader must observe the new writer entry, including when the
    // writer is the HBG scheduler-local host-write node.
    uint32_t shape[] = {1, reader_outputs.shapes[1]};
    uint32_t offset[] = {static_cast<uint32_t>(reader_count), 0};
    ChipTensor verify_output = reader_outputs.view(shape, offset);
    CoreTaskArgs verify_args;
    verify_args.add_input(x);
    verify_args.add_output(verify_output);
    verify_args.add_scalar(0);
    rt_submit_aiv_task(reader_fanout_test::FUNC_READER, verify_args);
}

}  // extern "C"
