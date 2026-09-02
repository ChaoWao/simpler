#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A two-level early-dispatch chain: gated producer with a gated consumer behind it.

While the slow root P0 runs, P1 (candidate: its producer is flagged) is
pre-staged gated; its staged blocks publish — publication counts placement, not
launch — so C becomes a candidate and pre-stages behind the still-gated P1.
Release cascades in staging order: P0 completes -> P1's doorbells ring -> P1
completes -> C's. The golden checks every cell landed, i.e. the chain neither
deadlocked nor dropped a block. Kernels are shared with the
tensormap_and_ringbuffer early-dispatch scenes.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

FLOATS_PER_CACHE_LINE = 16
BLOCKS = 8
SEGMENTS = 3
TOTAL_CL = SEGMENTS * BLOCKS

SLOW_KERNEL = "../../tensormap_and_ringbuffer/spmd_sync_start_early_dispatch/kernels/aiv/kernel_spmd_write_slow.cpp"


@scene_test(level=2, runtime="host_build_graph")
class TestChainedEarlyDispatchHbg(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/chained_early_dispatch_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "WRITE_AIC",
                "source": SLOW_KERNEL,
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "name": "WRITE_AIV",
                "source": SLOW_KERNEL,
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a2a3sim", "a2a3"],
            "params": {},
        }
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("output", torch.zeros(TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        expected = args.output.reshape(TOTAL_CL, FLOATS_PER_CACHE_LINE)
        for segment in range(SEGMENTS):
            for block_idx in range(BLOCKS):
                expected[segment * BLOCKS + block_idx, 0] = float(block_idx)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
