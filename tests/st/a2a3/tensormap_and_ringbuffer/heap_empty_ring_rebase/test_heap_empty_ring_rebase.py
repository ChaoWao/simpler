#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify that a drained heap ring can reuse its full capacity.

Each scope allocates a 1 MiB scratch on ring 1. The first scope drains the ring
at offset 1 MiB; the second allocation therefore requires an empty-ring rebase
when ring 1 is limited to 1.5 MiB.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

SENTINEL = 42.0
INIT_VAL = -1.0


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestHeapEmptyRingRebase(SceneTestCase):
    """A drained ring parked away from zero accepts a whole-span allocation."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/heap_empty_ring_rebase_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "FILL_CONST",
                "source": "kernels/aic/kernel_write_const.cpp",
                "core_type": "aic",
                "signature": [D.OUT],
            },
            {
                "func_id": 1,
                "name": "COPY_FIRST",
                "source": "kernels/aic/kernel_copy_first.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.INOUT],
            },
        ],
    }

    # One scheduler owns every core, so both scopes drain under the same
    # scheduler topology while only the ring-1 heap capacity forces the rebase.
    CASES = [
        {
            "name": "EmptyRingRebase",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {
                "aicpu_thread_num": 2,
                "runtime_env": {
                    "ring_heap": [268435456, 1572864, 268435456, 268435456],
                },
            },
            "params": {},
        },
    ]

    def generate_args(self, params):
        y1 = torch.full((16,), INIT_VAL, dtype=torch.float32)
        y2 = torch.full((16,), INIT_VAL, dtype=torch.float32)
        return TaskArgsBuilder(
            TensorArg("y1", y1),
            TensorArg("y2", y2),
        )

    def compute_golden(self, args, params):
        args.y1[0] = SENTINEL
        args.y2[0] = SENTINEL


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
