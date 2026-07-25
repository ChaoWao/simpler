# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Focused UT for the golden-validation hooks in scene_test.py.

Covers the two escapes from the default "allclose every output against
compute_golden" contract: per-case `skip_golden` and a `compare_outputs`
override.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import torch

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor

_SCENE_TEST_MOD = sys.modules["simpler_setup.scene_test"]

_OUTPUT_NAMES = ["y"]


class _Recorder(SceneTestCase):
    """Level-2 case that records which validation hooks fired."""

    CALLABLE = {"orchestration": {"signature": []}}
    CASES = []

    def __init__(self):
        self.golden_calls = 0
        self.compare_calls = []

    def generate_args(self, params):
        return TaskArgsBuilder(Tensor("y", torch.zeros(4, dtype=torch.float32)))

    def compute_golden(self, args, params):
        self.golden_calls += 1

    def compare_outputs(self, test_args, golden_args, output_names, params):
        self.compare_calls.append(list(output_names))


def _drive(inst, case, *, cli_skip=False):
    """Run one case through the L2 path with the device-touching steps stubbed."""
    inst._st_level = 2
    with (
        patch.object(_SCENE_TEST_MOD, "_build_chip_task_args", return_value=(object(), list(_OUTPUT_NAMES))),
        patch.object(type(inst), "_build_config", return_value=object()),
    ):
        inst._run_and_validate_l2(MagicMock(), object(), case, skip_golden=cli_skip)


def test_default_case_computes_and_compares():
    inst = _Recorder()
    _drive(inst, {"name": "c"})
    assert inst.golden_calls == 1
    assert inst.compare_calls == [_OUTPUT_NAMES]


def test_per_case_skip_golden_bypasses_both_hooks():
    inst = _Recorder()
    _drive(inst, {"name": "c", "skip_golden": True})
    assert inst.golden_calls == 0
    assert inst.compare_calls == []


def test_class_attribute_supplies_the_default():
    class _AllSkipped(_Recorder):
        SKIP_GOLDEN = True

    inst = _AllSkipped()
    _drive(inst, {"name": "c"})
    assert inst.golden_calls == 0
    assert inst.compare_calls == []


def test_per_case_value_overrides_the_class_attribute():
    class _AllSkipped(_Recorder):
        SKIP_GOLDEN = True

    inst = _AllSkipped()
    _drive(inst, {"name": "c", "skip_golden": False})
    assert inst.golden_calls == 1
    assert inst.compare_calls == [_OUTPUT_NAMES]


def test_cli_skip_golden_wins_over_an_opted_in_case():
    inst = _Recorder()
    _drive(inst, {"name": "c", "skip_golden": False}, cli_skip=True)
    assert inst.golden_calls == 0
    assert inst.compare_calls == []


def test_compare_outputs_override_replaces_the_default_allclose():
    """A mismatching output passes when the override drops it from the comparison."""

    class _DropsY(SceneTestCase):
        CALLABLE = {"orchestration": {"signature": []}}
        CASES = []

        def generate_args(self, params):
            return TaskArgsBuilder(Tensor("y", torch.zeros(4, dtype=torch.float32)))

        def compute_golden(self, args, params):
            args.y[:] = 1.0  # deliberately unequal to the un-run test tensor

        def compare_outputs(self, test_args, golden_args, output_names, params):
            super().compare_outputs(test_args, golden_args, [n for n in output_names if n != "y"], params)

    _drive(_DropsY(), {"name": "c"})


def test_default_compare_outputs_raises_on_mismatch():
    """Without the override, the same mismatch is caught."""

    class _Strict(SceneTestCase):
        CALLABLE = {"orchestration": {"signature": []}}
        CASES = []

        def generate_args(self, params):
            return TaskArgsBuilder(Tensor("y", torch.zeros(4, dtype=torch.float32)))

        def compute_golden(self, args, params):
            args.y[:] = 1.0

    import pytest

    with pytest.raises(AssertionError, match="Golden mismatch on 'y'"):
        _drive(_Strict(), {"name": "c"})
