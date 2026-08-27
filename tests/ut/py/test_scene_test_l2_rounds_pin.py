# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""UT: multi-round L2 SceneTests keep read-only inputs on the device."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch

import torch
from simpler.buffer import mint_owner_instance_id, wrap_device_malloc
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import DataType

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


def _drive(rounds: int):
    inst = _L2PinCase()
    inst._st_level = 2
    worker = MagicMock()
    pinned = {"x": object()}
    buffers = [object()]
    with (
        patch.object(_SCENE_TEST_MOD, "_pin_l2_round_inputs", return_value=(pinned, buffers)) as pin,
        patch.object(_SCENE_TEST_MOD, "_build_l2_ref_args", return_value=(object(), ["y", "z"])) as build,
        patch.object(type(inst), "_build_config", return_value=object()),
    ):
        inst._run_and_validate_l2(worker, object(), {"name": "c", "params": {}}, rounds=rounds)
    return pin, build, worker, pinned, buffers


def test_rounds_one_keeps_normal_host_staging():
    pin, build, worker, _pinned, _buffers = _drive(rounds=1)
    pin.assert_not_called()
    build.assert_called_once()
    assert build.call_args.kwargs.get("pinned_inputs") == {}
    worker.free.assert_not_called()


def test_rounds_gt_one_pins_inputs_automatically_and_frees_buffers():
    pin, build, worker, pinned, buffers = _drive(rounds=2)
    pin.assert_called_once()
    build.assert_called_once()
    assert build.call_args.kwargs.get("pinned_inputs") is pinned
    worker.free.assert_has_calls([call(buf) for buf in buffers])


def test_pin_helper_uploads_only_nonempty_read_only_inputs():
    test_args = _L2PinCase().generate_args({})
    empty = torch.empty(0, dtype=torch.float32)
    test_args.specs.insert(1, TensorArg("empty", empty))
    signature = [D.IN, D.IN, D.INOUT, D.OUT]
    worker = MagicMock()
    device_buffer = MagicMock()
    device_tensor = object()
    worker.malloc.return_value = device_buffer
    device_buffer.tensor.return_value = device_tensor

    pinned, buffers = _SCENE_TEST_MOD._pin_l2_round_inputs(worker, test_args, signature)

    assert pinned == {"x": device_tensor}
    assert buffers == [device_buffer]
    worker.malloc.assert_called_once_with(test_args.x.numel() * test_args.x.element_size())
    worker.copy_to.assert_called_once_with(device_buffer, test_args.x)


def test_pinned_input_keeps_its_host_view_on_materialized_args():
    test_args = _L2PinCase().generate_args({})
    single_input = TaskArgsBuilder(TensorArg("x", test_args.x))
    worker = MagicMock()
    device_tensor = wrap_device_malloc(
        0x4000,
        test_args.x.numel() * test_args.x.element_size(),
        mint_owner_instance_id(),
        1,
        "L2",
    ).tensor(tuple(test_args.x.shape), int(DataType.FLOAT32.value))

    args, _ = _SCENE_TEST_MOD._build_l2_ref_args(
        single_input,
        [D.IN],
        worker,
        pinned_inputs={"x": device_tensor},
    )

    assert args._host_view(0) == test_args.x.data_ptr()
