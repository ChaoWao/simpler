#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B 40-layer decode using one recorded host-built Graph per layer."""

import copy
import importlib.util
import sys
from pathlib import Path

from simpler_setup import SceneTestCase, scene_test
from simpler_setup.goldens.qwen3_14b_decode import compute_golden as _decode_golden
from simpler_setup.goldens.qwen3_14b_decode import generate_inputs as _decode_generate_inputs

N_LAYERS = 40
HERE = Path(__file__).resolve().parent
QWEN_CASE_DIR = HERE.parents[1] / "tensormap_and_ringbuffer/qwen3_14b_decode"


def _load_qwen_case():
    module_name = "_qwen3_14b_host_build_graph_base"
    spec = importlib.util.spec_from_file_location(module_name, QWEN_CASE_DIR / "test_qwen3_14b_decode.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the Qwen3-14B decode case")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.TestQwen314BDecode


def _callable():
    callable_cfg = copy.deepcopy(_load_qwen_case().CALLABLE)
    callable_cfg["orchestration"]["source"] = str(HERE / "kernels/orchestration/decode_fwd_layers.cpp")
    for incore in callable_cfg["incores"]:
        incore["source"] = str(QWEN_CASE_DIR / incore["source"])
    return callable_cfg


@scene_test(level=2, runtime="host_build_graph")
class TestQwen314BDecodeHostBuildGraph(SceneTestCase):
    RTOL = 5e-2
    ATOL = 1e-1
    CALLABLE = _callable()

    CASES = [
        {
            "name": "GraphExecutionBatch16Seq3500",
            "platforms": ["a2a3"],
            "manual": True,
            "config": {"aicpu_thread_num": 4, "block_dim": 0},
            "params": {"seed": 1234, "seq_len": 3500},
        },
    ]

    def generate_args(self, params):
        return _decode_generate_inputs(
            seed=params.get("seed", 1234),
            seq_len=params.get("seq_len", 3500),
            n_layers=N_LAYERS,
        )

    def compute_golden(self, args, params):
        _decode_golden(args, n_layers=N_LAYERS)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
