#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 HBG loop-carried write-after-read regression."""

import json
import time

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename


@scene_test(level=2, runtime="host_build_graph")
class TestWarRegressionHbgA2A3(SceneTestCase):
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/war_regression_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.OUT],
        },
        "incores": [
            {"func_id": 0, "source": "kernels/aiv/writer.cpp", "core_type": "aiv", "signature": [D.INOUT]},
            {"func_id": 1, "source": "kernels/aiv/reader.cpp", "core_type": "aiv", "signature": [D.IN, D.OUT]},
        ],
    }
    CASES = [
        {
            "name": "LoopCarried",
            "platforms": ["a2a3sim", "a2a3"],
            "params": {"cross_ring": False, "war_pred": 1, "war_succ": 2},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("buffer", torch.zeros(16, dtype=torch.float32)),
            TensorArg("outputs", torch.zeros((2, 16), dtype=torch.float32)),
            Scalar("spin_iters", 100000),
            Scalar("cross_ring", int(params["cross_ring"])),
        )

    def compute_golden(self, args, params):
        args.buffer[:] = 2.0
        args.outputs[0] = 1.0
        args.outputs[1] = 2.0

    def test_run(self, st_platform, st_worker, request):
        run_marker = int(time.time())
        super().test_run(st_platform, st_worker, request)
        if not self._effective_enable_dep_gen(request):
            return
        for case in self.CASES:
            if st_platform in case.get("platforms", []):
                self._validate_war_edge(case, run_marker)

    def _validate_war_edge(self, case, run_marker):
        label = _sanitize_for_filename(f"TestWarRegressionHbgA2A3_{case['name']}")
        matches = [p for p in _outputs_dir().glob(f"{label}_*") if p.stat().st_mtime >= run_marker]
        assert matches, f"no dep_gen output for {case['name']}"
        deps_path = max(matches, key=lambda p: p.stat().st_mtime) / "deps.json"
        assert deps_path.exists(), f"missing {deps_path}"
        edges = json.loads(deps_path.read_text()).get("edges", [])
        expected = (case["params"]["war_pred"], case["params"]["war_succ"])
        war_edges = {
            (int(edge["pred"]), int(edge["succ"]))
            for edge in edges
            if edge.get("source") == "tensormap" and edge.get("hazard") == "WAR" and edge.get("access_kind") == "READER"
        }
        assert expected in war_edges, f"missing READER WAR edge {expected}; got {war_edges}"


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
