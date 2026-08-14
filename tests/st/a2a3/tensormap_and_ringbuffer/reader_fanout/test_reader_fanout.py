#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reader fan-out followed by a pure overwrite writer (WAR regression)."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestReaderFanoutTmrA2A3(SceneTestCase):
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/reader_fanout_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.OUT, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/reader.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/writer.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.OUT],
            },
        ],
    }
    CASES = [
        {
            "name": "Correctness256",
            "platforms": ["a2a3sim", "a2a3"],
            "params": {"reader_count": 256, "spin_iters": 100000, "host_write": False},
        },
        {
            "name": "HostWrite256",
            "platforms": ["a2a3sim", "a2a3"],
            "params": {"reader_count": 256, "spin_iters": 100000, "host_write": True},
        },
        *[
            {
                "name": f"Fanout{count}",
                "platforms": ["a2a3sim", "a2a3"],
                "manual": True,
                "params": {"reader_count": count, "spin_iters": 0, "host_write": False},
            }
            for count in (1, 8, 32, 64, 128, 256)
        ],
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("x", torch.ones(16, dtype=torch.float32)),
            TensorArg("reader_outputs", torch.zeros((257, 16), dtype=torch.float32)),
            TensorArg("padding", torch.zeros(1, dtype=torch.float32)),
            Scalar("reader_count", int(params["reader_count"])),
            Scalar("spin_iters", int(params["spin_iters"])),
            Scalar("host_write", int(params["host_write"])),
        )

    def compute_golden(self, args, params):
        args.reader_outputs[: params["reader_count"]] = 1.0
        args.reader_outputs[params["reader_count"]] = 2.0
        args.x[0] = 2.0
        args.padding[0] = 3.0


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
