#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SPMD sync_start AIV: 4 AIV tasks testing fast path and drain. Output: 48 CL = 768 float32.

The cohort widths are fractions of the run's core count, derived on device and
reported back in `layout` — a run always takes the whole device, and that width
differs between sim and silicon. The golden is rebuilt from the reported
geometry; `output` is sized for the widest device the platform allows.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

FLOATS_PER_CACHE_LINE = 16
NUM_TASKS = 4
# Widest the platform allows: aiv/12 + aiv/3 + aiv/12 + aiv/2 at 72 AIV cores.
MAX_AIV = 72
MAX_TOTAL_CL = MAX_AIV // 12 + MAX_AIV // 3 + MAX_AIV // 12 + MAX_AIV // 2


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpmdSyncStartAiv(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_sync_start_aiv_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "SPMD_WRITE_AIV",
                "source": "../spmd_multiblock_aiv/kernels/aiv/kernel_spmd_write.cpp",
                "core_type": "aiv",
            },
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a5sim", "a5"],
            "params": {},
        }
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("output", torch.zeros(MAX_TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)),
            TensorArg("layout", torch.zeros(2 * NUM_TASKS, dtype=torch.int32)),
        )

    def compute_golden(self, args, params):
        # Both outputs are checked against the reported layout in compare_outputs.
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        layout = [int(v) for v in test_args.layout]
        expected = torch.zeros(MAX_TOTAL_CL, dtype=torch.float32)
        for i in range(NUM_TASKS):
            block_num, base_cl = layout[2 * i], layout[2 * i + 1]
            assert block_num >= 1, f"task {i} reported block_num {block_num}"
            assert base_cl + block_num * 1 <= MAX_TOTAL_CL, (
                f"task {i} layout ({block_num}, {base_cl}) overflows {MAX_TOTAL_CL} cache lines"
            )
            for block_idx in range(block_num):
                expected[base_cl + block_idx] = float(block_idx)
        actual = test_args.output.reshape(MAX_TOTAL_CL, FLOATS_PER_CACHE_LINE)[:, 0]
        assert torch.equal(actual, expected), (
            f"block slots disagree with the reported layout {layout}: "
            f"got {actual.tolist()}, expected {expected.tolist()}"
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
