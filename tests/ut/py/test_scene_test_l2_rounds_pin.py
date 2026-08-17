# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""UT: SceneTest L2 pins orchestration IN only when opt-in + --rounds > 1."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg

_SCENE_TEST_MOD = sys.modules["simpler_setup.scene_test"]


class _L2PinCase(SceneTestCase):
    CALLABLE = {"orchestration": {"signature": [D.IN, D.INOUT, D.OUT]}}
    CASES = []
    SKIP_GOLDEN = True

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("x", torch.ones(4, dtype=torch.float32)),
            TensorArg("y", torch.zeros(4, dtype=torch.float32)),
            TensorArg("z", torch.zeros(4, dtype=torch.float32)),
        )


class _L2ReuseCase(_L2PinCase):
    REUSE_L2_IN_ACROSS_ROUNDS = True


def _drive(inst: SceneTestCase, rounds: int):
    inst._st_level = 2
    worker = MagicMock()
    pinned = {"x": object()}
    buffers = [object()]
    with (
        patch.object(
            _SCENE_TEST_MOD,
            "_pin_l2_round_inputs",
            return_value=(pinned, buffers),
        ) as pin,
        patch.object(
            _SCENE_TEST_MOD,
            "_build_l2_ref_args",
            return_value=(object(), ["y", "z"]),
        ) as build,
        patch.object(type(inst), "_build_config", return_value=object()),
    ):
        inst._run_and_validate_l2(worker, object(), {"name": "c", "params": {}}, rounds=rounds)
    return pin, build, worker, pinned, buffers


def test_rounds_one_does_not_pin_even_when_opted_in():
    pin, build, worker, _pinned, _buffers = _drive(_L2ReuseCase(), rounds=1)
    pin.assert_not_called()
    build.assert_called_once()
    assert build.call_args.kwargs.get("pinned_inputs") == {}
    worker.free.assert_not_called()


def test_rounds_gt_one_without_opt_in_does_not_pin():
    pin, build, worker, _pinned, _buffers = _drive(_L2PinCase(), rounds=2)
    pin.assert_not_called()
    assert build.call_args.kwargs.get("pinned_inputs") == {}
    worker.free.assert_not_called()


def test_rounds_gt_one_with_opt_in_pins_in_and_frees_buffers():
    pin, build, worker, pinned, buffers = _drive(_L2ReuseCase(), rounds=2)
    pin.assert_called_once()
    build.assert_called_once()
    assert build.call_args.kwargs.get("pinned_inputs") is pinned
    worker.free.assert_has_calls([call(buf) for buf in buffers])
