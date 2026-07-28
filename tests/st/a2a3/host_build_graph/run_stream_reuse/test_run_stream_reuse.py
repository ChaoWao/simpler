#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2A3 run streams are reused only while their code image is unchanged.

A2A3 submits each run's AICore and AICPU kernels on the stream set of the
selected pipeline slot rather than on the persistent bootstrap pair. The AICPU
stream belongs to the slot, while the AICore stream is also bound to the loaded
code image. Reusing it after different code replaces the image can execute stale
instructions because the platform has no explicit AICore I-cache invalidation.

`Worker.run_stream_set_create_count` reports how many sets the bound runner has
built, including every fresh AICore stream created for a code transition.
"""

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.scene_test import _build_chip_task_args, _compare_outputs

_VECTOR_KERNELS = "../vector_example/kernels"
_REPEATED_RUNS = 4


@scene_test(level=2, runtime="host_build_graph")
class _SubtractCallable(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": f"{_VECTOR_KERNELS}/orchestration/example_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/kernel_sub.cpp",
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
        self._run_and_validate_l2(st_worker, callable_obj, self.CASES[0], rounds=1)
        after_first = st_worker.run_stream_set_create_count
        self._run_and_validate_l2(st_worker, callable_obj, self.CASES[0], rounds=_REPEATED_RUNS - 1)

        assert st_worker.run_stream_set_create_count == after_first, (
            f"same-image runs advanced stream generation after the first run: "
            f"{after_first} -> {st_worker.run_stream_set_create_count}"
        )

    def _run_registered(self, worker, handle, *, subtract):
        params = self.CASES[0]["params"]
        test_args = self.generate_args(params)
        chip_args, output_names = _build_chip_task_args(test_args, self.CALLABLE["orchestration"]["signature"])
        golden_args = test_args.clone()
        a, b = golden_args.a, golden_args.b
        base = a - b if subtract else a + b
        golden_args.f[:] = (base + 1) * (base + 2)
        worker.run(handle, chip_args, config=self._build_config(self.CASES[0]["config"]))
        _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)

    def test_aicore_stream_tracks_code_image(self, st_platform, st_worker):
        if st_platform != "a2a3":
            pytest.skip("AICore stream code generations are an a2a3 onboard resource")

        add_handle = st_worker.register(self.build_callable(st_platform))
        sub_handle = st_worker.register(_SubtractCallable.compile_chip_callable(st_platform))
        try:
            self._run_registered(st_worker, add_handle, subtract=False)
            after_add = st_worker.run_stream_set_create_count

            self._run_registered(st_worker, add_handle, subtract=False)
            assert st_worker.run_stream_set_create_count == after_add

            for handle, subtract in ((sub_handle, True), (add_handle, False), (sub_handle, True)):
                before_transition = st_worker.run_stream_set_create_count
                self._run_registered(st_worker, handle, subtract=subtract)
                assert st_worker.run_stream_set_create_count == before_transition + 1
        finally:
            st_worker.unregister(sub_handle)
            st_worker.unregister(add_handle)
