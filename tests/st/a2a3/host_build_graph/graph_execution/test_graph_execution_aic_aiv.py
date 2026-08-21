#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Graph Execution covers a Qwen-style fixed AIC/AIV decoder-layer DAG."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@scene_test(level=2, runtime="host_build_graph")
class TestGraphExecutionAicAivHostBuildGraph(SceneTestCase):
    RTOL = 1e-2
    ATOL = 1e-2

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/graph_execution_aic_aiv_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.IN, D.OUT, D.OUT, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "../matmul/kernels/aiv/kernel_log_sqrt.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": "../matmul/kernels/aic/kernel_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": "../matmul/kernels/aiv/kernel_add_exp.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "record_then_replay_aic_aiv",
            "platforms": ["a2a3sim", "a2a3"],
            "manual": ["a2a3sim"],
            "params": {},
        },
    ]

    def generate_args(self, params):
        rows = 128
        columns = 128
        size = rows * columns
        input_value = torch.exp(torch.tensor(4.0)).item()
        weight_value = 1.0 / (2 * columns)
        return TaskArgsBuilder(
            TensorArg("input", torch.full((size,), input_value, dtype=torch.float16)),
            TensorArg("weight_1", torch.full((size,), weight_value, dtype=torch.float16)),
            TensorArg("weight_2", torch.full((size,), weight_value, dtype=torch.float16)),
            TensorArg("output_1", torch.zeros(size, dtype=torch.float32)),
            TensorArg("output_2", torch.zeros(size, dtype=torch.float32)),
            TensorArg("output_3", torch.zeros(size, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        rows = 128
        columns = 128
        normalized = torch.sqrt(torch.log(args.input.reshape(rows, columns).to(torch.float32)))
        left = torch.matmul(normalized, args.weight_1.reshape(rows, columns).to(torch.float32))
        right = torch.matmul(normalized, args.weight_2.reshape(rows, columns).to(torch.float32))
        expected = torch.exp(left + right).flatten().to(torch.float32)
        args.output_1[:] = expected
        args.output_2[:] = expected
        args.output_3[:] = expected


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
