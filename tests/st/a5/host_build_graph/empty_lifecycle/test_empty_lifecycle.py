#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Manual gate for the A5 HBG empty-graph resident lifecycle."""

from simpler_setup import SceneTestCase, TaskArgsBuilder, scene_test


@scene_test(level=2, runtime="host_build_graph")
class TestHbgEmptyLifecycle(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/empty_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [],
        },
        "incores": [],
    }

    CASES = [
        {
            "name": "empty_lifecycle",
            "platforms": ["a5sim", "a5"],
            # Exercise the minimum supported AICPU population during a no-task shutdown.
            "config": {"aicpu_thread_num": 2},
            "params": {},
            "manual": True,
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder()

    def compute_golden(self, args, params):
        pass


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
