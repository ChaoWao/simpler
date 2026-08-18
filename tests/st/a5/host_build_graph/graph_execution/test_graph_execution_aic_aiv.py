#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Graph Execution replays an A5-native AIC/AIV fan-out and fan-in DAG."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@scene_test(level=2, runtime="host_build_graph")
class TestGraphExecutionAicAivHostBuildGraphA5(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/graph_execution_aic_aiv_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.IN, D.OUT, D.OUT, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "../../tensormap_and_ringbuffer/alternating_matmul_add/kernels/aic/kernel_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": "../../tensormap_and_ringbuffer/alternating_matmul_add/kernels/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "record_then_replay_aic_aiv",
            "platforms": ["a5sim", "a5"],
            "manual": ["a5sim"],
            "config": {"aicpu_thread_num": 4},
            "params": {},
        },
    ]

    def generate_args(self, params):
        size = 128 * 128
        return TaskArgsBuilder(
            TensorArg("input", torch.full((size,), 0.01, dtype=torch.float32)),
            TensorArg("weight_1", torch.full((size,), 0.02, dtype=torch.float32)),
            TensorArg("weight_2", torch.full((size,), 0.03, dtype=torch.float32)),
            TensorArg("output_1", torch.zeros(size, dtype=torch.float32)),
            TensorArg("output_2", torch.zeros(size, dtype=torch.float32)),
            TensorArg("output_3", torch.zeros(size, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        input_matrix = args.input.reshape(128, 128)
        left = torch.matmul(input_matrix, args.weight_1.reshape(128, 128))
        right = torch.matmul(input_matrix, args.weight_2.reshape(128, 128))
        expected = (left + right).flatten()
        args.output_1[:] = expected
        args.output_2[:] = expected
        args.output_3[:] = expected


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
