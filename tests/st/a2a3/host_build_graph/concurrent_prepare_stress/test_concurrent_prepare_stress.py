#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Concurrent-prepare overlap stress for host_build_graph.

Drives a 2-deep native-run pipeline over the two arena banks: run i is
*prepared* (which runs its full bind — arena build + host orchestration) while
run i-1 is still launched-but-not-finalized. That is exactly the
``overlaps_active_run`` path (a successor prepared into bank B while a
predecessor executes in bank A), which the ordinary blocking run() never hits.

Each iteration uses distinct input data, so any cross-run interference between
the two banks' runtimes/orchestrators (e.g. a shared or under-initialized
orchestrator) would corrupt a result and fail the golden check. The point is to
exercise the concurrent-prepare bind path under many alternating-bank
iterations, not to measure anything.
"""

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _build_chip_task_args, _compare_outputs

_VECTOR_KERNELS = "../vector_example/kernels"
_SIZE = 128 * 128
_ITERS = 40  # 20 reuses per bank


@scene_test(level=2, runtime="host_build_graph")
class TestConcurrentPrepareStressHbg(SceneTestCase):
    """Prepare-while-active stress over the two pipeline banks (a2a3 onboard)."""

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

    _PLATFORMS = ["a2a3"]

    CASES = [
        {
            "name": "overlap_stress",
            "platforms": _PLATFORMS,
            # Consumed by the framework's default test_run (a plain blocking run);
            # the overlap loop below builds its own per-iteration params.
            "params": {"a": 2.0, "b": 3.0},
        },
    ]

    def generate_args(self, params):
        """Build the two constant input vectors and the zero output for ``params``."""
        a, b = params["a"], params["b"]
        return TaskArgsBuilder(
            TensorArg("a", torch.full((_SIZE,), a, dtype=torch.float32)),
            TensorArg("b", torch.full((_SIZE,), b, dtype=torch.float32)),
            TensorArg("f", torch.zeros(_SIZE, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        """Reference output: example_orch computes ``(a + b + 1) * (a + b + 2)``."""
        a, b = args.a, args.b
        args.f[:] = (a + b + 1) * (a + b + 2)

    def _chip_worker(self, worker):
        """Return the L2 ChipWorker backing ``worker`` (asserts it exists)."""
        chip_worker = worker._chip_worker
        assert chip_worker is not None
        return chip_worker

    def test_concurrent_prepare_overlap(self, st_platform, st_worker):
        """Golden-check a 2-deep overlapping pipeline over both arena banks.

        Each submission prepares (fully binds) its run against one bank while the
        predecessor is still active in the other, exercising the
        ``overlaps_active_run`` path many times with distinct data.
        """
        if st_platform != "a2a3":
            pytest.skip("concurrent native prepare / two-bank pipeline is an a2a3 onboard path")

        orch_sig = self.CALLABLE["orchestration"]["signature"]
        callable_obj = self.build_callable(st_platform)
        config = self._build_config({})
        chip_worker = self._chip_worker(st_worker)

        if int(chip_worker.pipeline_depth) < 2:
            pytest.skip(f"need pipeline_depth >= 2 for overlap, have {chip_worker.pipeline_depth}")

        callable_id = 0
        chip_worker._register_callable_at_slot(callable_id, callable_obj)

        # The direct-chip lane is the sole admission authority and follows the
        # runtime PipelineContract: it admits one active plus one prepared
        # compatible successor. Submitting run i while run i-1 is still in flight
        # therefore prepares run i (its full bind — arena build + host
        # orchestration) against the other bank while run i-1 is active, i.e. the
        # overlaps_active_run path. Each in-flight entry keeps BOTH the encoded
        # chip_args and the backing tensors (test_args) alive until the run
        # reaches terminal (the lane copies inputs but the buffers back the
        # resolved descriptors, and the output is copied back at finalize).
        inflight = []  # (chip_run, test_args, golden_args, output_names, chip_args, iteration)

        def _retire(entry):
            """Block one run to terminal (lane finalizes + copies back) and golden-check it."""
            chip_run, test_args, golden_args, output_names, _chip_args, _it = entry
            # Block to the completion fence; the lane finalizes (validate +
            # copy-back) as part of reaching terminal.
            chip_run.wait(-1.0)
            # Distinct per-iteration data → a corrupted/aliased bank fails this.
            _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)

        try:
            for i in range(_ITERS):
                # Distinct inputs per iteration so golden catches cross-bank interference.
                params = {"a": 1.0 + i, "b": 2.0 + 0.5 * i}
                test_args = self.generate_args(params)
                chip_args, output_names = _build_chip_task_args(test_args, orch_sig)
                golden_args = test_args.clone()
                self.compute_golden(golden_args, params)

                chip_run = chip_worker._impl._submit_chip_run_direct(callable_id, chip_args, config)
                inflight.append((chip_run, test_args, golden_args, output_names, chip_args, i))

                # Keep two runs in flight so each submission overlaps its predecessor.
                if len(inflight) >= 2:
                    _retire(inflight.pop(0))

            while inflight:
                _retire(inflight.pop(0))
        finally:
            chip_worker._unregister_slot(callable_id)
