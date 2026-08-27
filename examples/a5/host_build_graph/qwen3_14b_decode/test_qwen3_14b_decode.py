#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 Qwen3-14B decode migrated from TMR to host-build-graph."""

from copy import deepcopy

from examples.a5.tensormap_and_ringbuffer.qwen3_14b_decode.test_qwen3_14b_decode import (
    TestQwen314BDecode as _TmrQwen314BDecode,
)
from simpler_setup import SceneTestCase, scene_test
from simpler_setup.goldens.qwen3_14b_decode import compute_golden, generate_inputs


def _host_build_graph_callable():
    callable_config = deepcopy(_TmrQwen314BDecode.CALLABLE)
    callable_config["orchestration"]["source"] = "kernels/orchestration/decode_fwd_layers.cpp"
    return callable_config


@scene_test(level=2, runtime="host_build_graph")
class TestQwen314BDecodeHostBuildGraph(SceneTestCase):
    """Qwen3-14B decode using the A5 TMR incores with HBG orchestration."""

    RTOL = _TmrQwen314BDecode.RTOL
    ATOL = _TmrQwen314BDecode.ATOL
    CALLABLE = _host_build_graph_callable()

    CASES = [
        {
            "name": "GraphExecutionBatch16Seq3500",
            "platforms": ["a5"],
            "manual": True,
            "params": {"seed": 1234, "seq_len": 3500},
        },
    ]

    def generate_args(self, params):
        return generate_inputs(params["seed"], params["seq_len"], n_layers=40)

    def compute_golden(self, args, params):
        compute_golden(args, n_layers=40)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
