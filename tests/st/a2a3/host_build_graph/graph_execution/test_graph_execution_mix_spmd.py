#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Graph Execution preserves MIX active slots and multi-block SPMD metadata."""

import json

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename

FLOATS_PER_CACHE_LINE = 16
SLOTS_PER_BLOCK = 3
MAX_CLUSTERS = 24
TOTAL_FLOATS = MAX_CLUSTERS * SLOTS_PER_BLOCK * FLOATS_PER_CACHE_LINE


@scene_test(level=2, runtime="host_build_graph")
class TestGraphExecutionMixSpmdHostBuildGraph(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/graph_execution_mix_spmd_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "../../tensormap_and_ringbuffer/spmd_multiblock_mix/kernels/aic/kernel_spmd_mix.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "source": "../../tensormap_and_ringbuffer/spmd_multiblock_mix/kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 2,
                "source": "../../tensormap_and_ringbuffer/spmd_multiblock_mix/kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "record_then_replay_mix_spmd",
            "platforms": ["a2a3sim", "a2a3"],
            "params": {},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("blocks_1", torch.zeros(TOTAL_FLOATS, dtype=torch.float32)),
            TensorArg("blocks_2", torch.zeros(TOTAL_FLOATS, dtype=torch.float32)),
            TensorArg("blocks_3", torch.zeros(TOTAL_FLOATS, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        # The exact whole-device cluster count is platform-provided. Validate
        # it from the first execution and require both Graph replays to match.
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        outputs = [test_args.blocks_1, test_args.blocks_2, test_args.blocks_3]
        assert torch.equal(outputs[0], outputs[1])
        assert torch.equal(outputs[0], outputs[2])

        cache_line_heads = outputs[0].reshape(-1, FLOATS_PER_CACHE_LINE)[:, 0]
        nonzero = torch.nonzero(cache_line_heads, as_tuple=False)
        assert nonzero.numel() > 0, "SPMD Graph did not execute any non-zero block"
        cluster_count = int(cache_line_heads.max().item()) + 1
        assert 1 < cluster_count <= MAX_CLUSTERS

        expected = torch.zeros_like(cache_line_heads)
        for block_idx in range(cluster_count):
            begin = block_idx * SLOTS_PER_BLOCK
            expected[begin : begin + SLOTS_PER_BLOCK] = float(block_idx)
        assert torch.equal(cache_line_heads, expected)

    def test_run(self, st_platform, st_worker, request):
        matched_cases = self._matching_cases(st_platform, request)
        prior_mtimes = {case["name"]: self._matching_output_mtimes(case) for case in matched_cases}
        super().test_run(st_platform, st_worker, request)
        if not self._effective_enable_dep_gen(request):
            return
        for case in matched_cases:
            self._validate_outer_graph_capture(case, prior_mtimes[case["name"]])

    @staticmethod
    def _matching_output_mtimes(case):
        safe_label = _sanitize_for_filename(f"TestGraphExecutionMixSpmdHostBuildGraph_{case['name']}")
        return {path: path.stat().st_mtime_ns for path in _outputs_dir().glob(f"{safe_label}_*")}

    @classmethod
    def _validate_outer_graph_capture(cls, case, prior_mtimes):
        matches = [
            path
            for path, mtime_ns in cls._matching_output_mtimes(case).items()
            if path not in prior_mtimes or mtime_ns > prior_mtimes[path]
        ]
        assert matches, f"no dep-gen output directory created for {case['name']}"
        deps_path = max(matches, key=lambda path: path.stat().st_mtime_ns) / "deps.json"
        assert deps_path.exists(), f"deps.json missing for {case['name']}"
        with deps_path.open() as f:
            deps = json.load(f)

        tasks = deps.get("tasks", [])
        assert [int(task["task_id"]) for task in tasks] == [0, 1, 2]
        outer_graph_ids = {
            int(task["task_id"])
            for task in tasks
            if task.get("kernel_ids") == [-1, -1, -1] and len(task.get("args", [])) == 1
        }
        assert outer_graph_ids == {0, 1, 2}
        assert deps.get("edges", []) == []


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
