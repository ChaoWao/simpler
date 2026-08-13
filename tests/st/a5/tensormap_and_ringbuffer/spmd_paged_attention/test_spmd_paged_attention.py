#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 port of the benchmark SPMD paged-attention MIX workload."""

from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.goldens.paged_attention import compute_golden as _pa_compute_golden
from simpler_setup.goldens.paged_attention import generate_inputs as _pa_generate_inputs

CASE_DIR = Path(__file__).resolve().parent


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpmdPagedAttentionA5(SceneTestCase):
    RTOL = 1e-2
    ATOL = 1e-2

    CALLABLE = {
        "orchestration": {
            "source": str(CASE_DIR / "kernels/orchestration/spmd_paged_attention_orch.cpp"),
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "PA_AIC",
                "source": str(CASE_DIR / "kernels/mix/paged_attention_parallel.cpp"),
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.INOUT, D.OUT, D.OUT, D.OUT],
            },
            {
                "func_id": 1,
                "name": "PA_AIV",
                "source": str(CASE_DIR / "kernels/mix/paged_attention_parallel.cpp"),
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.INOUT, D.OUT, D.OUT, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "TwoWayOneBlockZeroQuery",
            "platforms": ["a5sim", "a5"],
            "manual": True,
            "params": {
                "batch": 2,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 128,
                "block_size": 128,
                "context_len": 128,
                "max_model_len": 256,
                "dtype": "bfloat16",
            },
        },
    ]

    def generate_args(self, params):
        specs = []
        for name, value in _pa_generate_inputs(params):
            if name == "query":
                value.zero_()
            specs.append(TensorArg(name, value) if isinstance(value, torch.Tensor) else Scalar(name, value))
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        tensors = {s.name: s.value for s in args.specs if isinstance(s, TensorArg)}
        _pa_compute_golden(tensors, params)
        for spec in args.specs:
            if isinstance(spec, TensorArg) and spec.name in tensors:
                getattr(args, spec.name)[:] = tensors[spec.name]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
