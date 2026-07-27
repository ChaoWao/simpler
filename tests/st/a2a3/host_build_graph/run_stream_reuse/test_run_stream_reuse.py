#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2A3 run stream sets belong to a pipeline slot, not to a run.

A2A3 submits each run's AICore and AICPU kernels on the stream set of the
selected pipeline slot rather than on the persistent bootstrap pair. A set
carries no per-run content, so it is created on first use and reused; rebuilding
it per run costs two rtStreamCreate plus two rtStreamDestroy (~1.2 ms) on the
synchronous host path around KernelLaunch and buys nothing.

`Worker.run_stream_set_create_count` reports how many sets the bound runner has
built, which is what makes that invariant assertable from here.
"""

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

_VECTOR_KERNELS = "../vector_example/kernels"
_REPEATED_RUNS = 4


@scene_test(level=2, runtime="host_build_graph")
class TestRunStreamReuseHbg(SceneTestCase):
    """Repeated runs on one worker must share a single run stream set."""

    RTOL = 1e-5
    ATOL = 1e-5

    CALLABLE = {
        "orchestration": {
            "source": f"{_VECTOR_KERNELS}/orchestration/example_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": f"{_VECTOR_KERNELS}/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": f"{_VECTOR_KERNELS}/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": f"{_VECTOR_KERNELS}/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "repeated_runs",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {},
        },
    ]

    def generate_args(self, params):
        size = 128 * 128
        return TaskArgsBuilder(
            Tensor("a", torch.full((size,), 2.0, dtype=torch.float32)),
            Tensor("b", torch.full((size,), 3.0, dtype=torch.float32)),
            Tensor("f", torch.zeros(size, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        a, b = args.a, args.b
        args.f[:] = (a + b + 1) * (a + b + 2)

    def test_one_stream_set_serves_repeated_runs(self, st_platform, st_worker):
        """N runs on one worker build one stream set, and every result is right."""
        if st_platform != "a2a3":
            pytest.skip("run stream sets are an a2a3 onboard resource")

        callable_obj = self.build_callable(st_platform)
        self._run_and_validate_l2(st_worker, callable_obj, self.CASES[0], rounds=_REPEATED_RUNS)

        assert st_worker.run_stream_set_create_count == 1, (
            f"expected 1 run stream set for {_REPEATED_RUNS} runs, got "
            f"{st_worker.run_stream_set_create_count} — the set is being rebuilt per run"
        )
