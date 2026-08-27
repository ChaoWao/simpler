#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Manual correctness gates for homogeneous and mixed AIC/AIV dependency graphs."""

import ctypes

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

GRAPH_CASES = {
    "chain_64": (0, 64),
    "diamond_8": (1, 25),
    "fanout_63": (2, 64),
    "fanin_32x32": (3, 64),
    "multi_root_64": (4, 64),
    "random_64": (5, 64),
}


@scene_test(level=2, runtime="host_build_graph")
class TestHbgSingleCoreDag(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/single_core_dag_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/check_dag.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "source": "kernels/check_dag.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": f"{core_type}_{name}",
            "platforms": ["a5sim", "a5"],
            "params": {
                "graph_case": graph_case,
                "task_count": task_count,
                "core_type": core_type,
            },
            "manual": True,
        }
        for core_type, names in {
            "aic": ("chain_64",),
            "aiv": ("chain_64",),
            "mixed": tuple(GRAPH_CASES),
        }.items()
        for name in names
        for graph_case, task_count in (GRAPH_CASES[name],)
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("task_state", torch.zeros(64 * 8, dtype=torch.int64)),
            Scalar("graph_case", ctypes.c_int64(params["graph_case"])),
            Scalar("core_type", ctypes.c_int64({"aic": 0, "aiv": 1, "mixed": 2}[params["core_type"]])),
        )

    def compute_golden(self, args, params):
        count = params["task_count"]
        args.task_state[: count * 8 : 8] = torch.arange(1, count + 1, dtype=torch.int64)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
