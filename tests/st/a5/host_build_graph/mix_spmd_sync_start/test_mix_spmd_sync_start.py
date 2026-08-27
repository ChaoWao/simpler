#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cluster-local Mix, SPMD, and sync-start execution on the HBG AICore scheduler."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

FLOATS_PER_CACHE_LINE = 16
SLOTS_PER_BLOCK = 3
NUM_TASKS = 5
MAX_AIV = 72
MAX_CLUSTERS = 36
MAX_TOTAL_CL = (1 + MAX_AIV // 2 + MAX_CLUSTERS // 2 + MAX_AIV // 2 + MAX_CLUSTERS // 2) * SLOTS_PER_BLOCK


@scene_test(level=2, runtime="host_build_graph")
class TestMixSpmdSyncStart(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/mix_spmd_sync_start_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "../../tensormap_and_ringbuffer/spmd_multiblock_mix/kernels/aic/kernel_spmd_mix.cpp",
                "core_type": "aic",
            },
            {
                "func_id": 1,
                "source": "../../tensormap_and_ringbuffer/spmd_multiblock_mix/kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aiv",
            },
            {
                "func_id": 2,
                "source": "../../tensormap_and_ringbuffer/spmd_multiblock_mix/kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aiv",
            },
        ],
    }

    CASES = [{"name": "mixed_priorities", "platforms": ["a5sim", "a5"], "params": {}}]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("output", torch.zeros(MAX_TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)),
            TensorArg("layout", torch.zeros(NUM_TASKS * 3, dtype=torch.int32)),
        )

    def compute_golden(self, args, params):
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        layout = [int(value) for value in test_args.layout]
        expected = torch.zeros(MAX_TOTAL_CL, dtype=torch.float32)
        for task in range(NUM_TASKS):
            block_num, base_cl, active_mask = layout[task * 3 : task * 3 + 3]
            assert block_num > 0
            assert base_cl + block_num * SLOTS_PER_BLOCK <= MAX_TOTAL_CL
            for block_idx in range(block_num):
                for subtask_slot in range(3):
                    if active_mask & (1 << subtask_slot):
                        expected[base_cl + block_idx * SLOTS_PER_BLOCK + subtask_slot] = float(block_idx)
        actual = test_args.output.reshape(MAX_TOTAL_CL, FLOATS_PER_CACHE_LINE)[:, 0]
        assert torch.equal(actual, expected), f"layout={layout}, actual={actual.tolist()}, expected={expected.tolist()}"


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
