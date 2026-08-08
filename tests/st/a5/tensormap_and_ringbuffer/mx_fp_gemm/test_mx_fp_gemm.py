#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""mx_fp_gemm: A5 MXFP8/MXFP4 ``TMATMUL_MX`` ST via task interface (onboard only).

Device AIC kernel calls pto-isa ``TMATMUL_MX`` (not scalar MAC). Host inputs +
golden follow pto-isa ``tmatmul_mx`` host-prequant sample:
  M=128, K=64, N=64; scales=FP8E8M0 (ZZ/NN); C=FP32
  mode 0: A/B = FP8E4M3FN
  mode 1: A/B = FP4E2M1x2 (last-dim packed; A [M,K/2], B [K,N/2])

a5sim is omitted: CPU stub TLOAD does not support MX_A_ZZ / MX_B_NN layouts.
"""

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.goldens.mx_fp_gemm import make_mx_fp4_case, make_mx_fp8_case

M, K, N = 128, 64, 64
RTOL = 1e-3
ATOL = 1e-3


def _cases():
    cases = []
    if hasattr(torch, "float8_e4m3fn") and hasattr(torch, "float8_e8m0fnu"):
        cases.append(
            {
                "name": "MXFP8_TMATMUL_MX",
                "platforms": ["a5"],
                "config": {"aicpu_thread_num": 2, "block_dim": 1},
                "params": {"mode": 0},
            }
        )
    if hasattr(torch, "float4_e2m1fn_x2") and hasattr(torch, "float8_e8m0fnu"):
        cases.append(
            {
                "name": "MXFP4_TMATMUL_MX",
                "platforms": ["a5"],
                "config": {"aicpu_thread_num": 2, "block_dim": 1},
                "params": {"mode": 1},
            }
        )
    return cases


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestMxFpGemm(SceneTestCase):
    """Host-prequant MXFP8/MXFP4 → A5 ``TMATMUL_MX`` numerical check."""

    RTOL = RTOL
    ATOL = ATOL

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/mx_fp_gemm_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "MATMUL_MX",
                "source": "kernels/aic/kernel_fp_gemm.cpp",
                "core_type": "aic",
            },
        ],
    }

    CASES = _cases()

    def generate_args(self, params):
        mode = int(params["mode"])
        if mode == 0:
            if not (hasattr(torch, "float8_e4m3fn") and hasattr(torch, "float8_e8m0fnu")):
                pytest.skip("torch.float8_e4m3fn / float8_e8m0fnu required")
            a, a_s, b, b_s, c, _ = make_mx_fp8_case(M, K, N)
        else:
            if not (hasattr(torch, "float4_e2m1fn_x2") and hasattr(torch, "float8_e8m0fnu")):
                pytest.skip("torch.float4_e2m1fn_x2 / float8_e8m0fnu required")
            a, a_s, b, b_s, c, _ = make_mx_fp4_case(M, K, N)
        return TaskArgsBuilder(
            TensorArg("A", a.reshape(-1)),
            TensorArg("As", a_s.reshape(-1)),
            TensorArg("B", b.reshape(-1)),
            TensorArg("Bs", b_s.reshape(-1)),
            TensorArg("C", c.reshape(-1)),
            Scalar("mode", mode),
        )

    def compute_golden(self, args, params):
        mode = int(params["mode"])
        if mode == 0:
            _a, _as, _b, _bs, _c, expected = make_mx_fp8_case(M, K, N)
        else:
            _a, _as, _b, _bs, _c, expected = make_mx_fp4_case(M, K, N)
        args.C[:] = expected.reshape(-1)


if __name__ == "__main__":
    if not TestMxFpGemm.CASES:
        raise SystemExit("Need torch MX dtypes (float8_e4m3fn/float4_e2m1fn_x2 + float8_e8m0fnu)")
    SceneTestCase.run_module(__name__)
