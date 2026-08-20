#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A TensorCreateInfo initial value reaches the kernel that consumes the allocation.

The buffer is in the GM heap, which the host orchestrator can only write through
the host view registered for the run, so a wrong or absent view shows up as
`out_alloc` / `out_dummy` holding whatever the heap last held.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

ALLOC_FILL = 7.0
DUMMY_FILL = 11.0


@scene_test(level=2, runtime="host_build_graph")
class TestInitialValueHostBuildGraph(SceneTestCase):
    RTOL = 1e-5
    ATOL = 1e-5

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/initial_value_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.OUT, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "../vector_example/kernels/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": "../vector_example/kernels/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "alloc_and_submitted_output",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {"shape": (128 * 128,)},
        },
    ]

    def generate_args(self, params):
        shape = params["shape"]
        return TaskArgsBuilder(
            TensorArg("a", torch.full(shape, 2.0, dtype=torch.float32)),
            TensorArg("out_alloc", torch.zeros(shape, dtype=torch.float32)),
            TensorArg("out_dummy", torch.zeros(shape, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.out_alloc[:] = ALLOC_FILL + args.a
        args.out_dummy[:] = DUMMY_FILL


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
