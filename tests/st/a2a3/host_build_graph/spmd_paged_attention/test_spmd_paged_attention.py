#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Manual HBG coverage for the disabled SPMD paged-attention workload."""

from copy import deepcopy
from pathlib import Path

from simpler_setup import scene_test
from tests.st.a2a3.tensormap_and_ringbuffer.spmd_paged_attention.test_spmd_paged_attention import (
    TestPagedAttentionUnrollTpushPop as _TmrBase,
)


@scene_test(level=2, runtime="host_build_graph")
class TestSpmdPagedAttentionHbgA2A3(_TmrBase):
    _SHARED_DIR = Path(__file__).resolve().parents[2] / "tensormap_and_ringbuffer/spmd_paged_attention"
    CALLABLE = deepcopy(_TmrBase.CALLABLE)
    CALLABLE["orchestration"]["source"] = str(_SHARED_DIR / "kernels/orchestration/spmd_paged_attention_orch.cpp")
    for _incore in CALLABLE["incores"]:
        _incore["source"] = str(_SHARED_DIR / "kernels/mix/paged_attention_parallel.cpp")

    CASES = [
        {
            "name": "TwoWayOneBlockZeroQuery",
            "platforms": ["a2a3sim", "a2a3"],
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
        }
    ]

    def generate_args(self, params):
        args = super().generate_args(params)
        args.query.zero_()
        return args


if __name__ == "__main__":
    TestSpmdPagedAttentionHbgA2A3.run_module(__name__)
