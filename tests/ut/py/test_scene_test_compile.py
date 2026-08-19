# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from contextlib import contextmanager
from threading import Barrier

import pytest

from simpler_setup.tools import scene_test_compile


class _Item:
    def __init__(self, cls, markers=()):
        self.cls = cls
        self._markers = markers

    def iter_markers(self, name=None):
        return (marker for marker in self._markers if name is None or marker.name == name)


def test_compile_collected_scene_tests_deduplicates_classes_and_skips_marked_items():
    calls = []

    class L2:
        _st_level = 2

        @classmethod
        def compile_chip_callable(cls, platform):
            calls.append((cls.__name__, platform))

    class L3:
        _st_level = 3

        @classmethod
        def _compile_l3_callables(cls, platform):
            calls.append((cls.__name__, platform))

    class Skipped:
        _st_level = 2

        @classmethod
        def compile_chip_callable(cls, platform):
            calls.append((cls.__name__, platform))

    count, failures = scene_test_compile.compile_collected_scene_tests(
        [
            _Item(L2),
            _Item(L2),
            _Item(L3),
            _Item(Skipped, [pytest.mark.skip(reason="not selected").mark]),
            _Item(None),
        ],
        "a2a3",
    )

    assert count == 2
    assert failures == []
    assert set(calls) == {("L2", "a2a3"), ("L3", "a2a3")}


def test_compile_collected_scene_tests_evaluates_skipif():
    calls = []

    class ActiveSkip:
        _st_level = 2

        @classmethod
        def compile_chip_callable(cls, platform):
            calls.append(cls.__name__)

    class InactiveSkip:
        _st_level = 2

        @classmethod
        def compile_chip_callable(cls, platform):
            calls.append(cls.__name__)

    count, failures = scene_test_compile.compile_collected_scene_tests(
        [
            _Item(ActiveSkip, [pytest.mark.skipif(True, reason="active").mark]),
            _Item(InactiveSkip, [pytest.mark.skipif(False, reason="inactive").mark]),
        ],
        "a2a3",
    )

    assert count == 1
    assert failures == []
    assert calls == ["InactiveSkip"]


def test_compile_collected_scene_tests_runs_classes_concurrently():
    started = Barrier(2, timeout=1)

    class First:
        _st_level = 2

        @classmethod
        def compile_chip_callable(cls, platform):
            started.wait()

    class Second:
        _st_level = 2

        @classmethod
        def compile_chip_callable(cls, platform):
            started.wait()

    assert scene_test_compile.compile_collected_scene_tests([_Item(First), _Item(Second)], "a2a3", max_workers=2) == (
        2,
        [],
    )


def test_compile_collected_scene_tests_shares_worker_budget_with_callable_compiles(monkeypatch):
    captured = []

    @contextmanager
    def recording_budget(max_workers):
        captured.append(max_workers)
        yield

    class Scene:
        _st_level = 2

        @classmethod
        def compile_chip_callable(cls, platform):
            return None

    monkeypatch.setattr(scene_test_compile, "compile_worker_budget", recording_budget)

    assert scene_test_compile.compile_collected_scene_tests([_Item(Scene)], "a2a3", max_workers=3) == (1, [])
    assert captured == [3]


@pytest.mark.parametrize(("max_workers", "expected"), [(None, 3), (8, 8)])
def test_compile_collected_scene_tests_uses_configured_workers(monkeypatch, max_workers, expected):
    captured = {}

    class RecordingExecutor:
        def __init__(self, max_workers):
            captured["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def map(self, fn, values):
            return map(fn, values)

    class Scene:
        _st_level = 2

        @classmethod
        def compile_chip_callable(cls, platform):
            return None

    monkeypatch.setattr(scene_test_compile, "ThreadPoolExecutor", RecordingExecutor, raising=False)
    monkeypatch.setattr(scene_test_compile, "default_compile_workers", lambda: 3)

    kwargs = {} if max_workers is None else {"max_workers": max_workers}
    assert scene_test_compile.compile_collected_scene_tests([_Item(Scene)], "a2a3", **kwargs) == (1, [])
    assert captured == {"max_workers": expected}


def test_compile_collected_scene_tests_reports_failures_without_raising():
    compiled = []

    class Broken:
        _st_level = 2

        @classmethod
        def compile_chip_callable(cls, platform):
            raise RuntimeError("ccec exploded")

    class Healthy:
        _st_level = 2

        @classmethod
        def compile_chip_callable(cls, platform):
            compiled.append(cls.__name__)

    count, failures = scene_test_compile.compile_collected_scene_tests([_Item(Broken), _Item(Healthy)], "a2a3")

    assert count == 1
    assert compiled == ["Healthy"]
    assert [name for name, _error in failures] == ["Broken"]
    assert isinstance(failures[0][1], RuntimeError)


def test_compile_collected_scene_tests_without_pytest_skip_helper(monkeypatch):
    calls = []

    class Skipped:
        _st_level = 2

        @classmethod
        def compile_chip_callable(cls, platform):
            calls.append(cls.__name__)

    monkeypatch.setattr(scene_test_compile, "evaluate_skip_marks", None)

    count, failures = scene_test_compile.compile_collected_scene_tests(
        [_Item(Skipped, [pytest.mark.skip(reason="ignored without the helper").mark])],
        "a2a3",
    )

    assert (count, failures, calls) == (1, [], ["Skipped"])


def test_main_runs_pytest_collection_only(monkeypatch):
    captured = {}

    def fake_pytest_main(args, plugins):
        captured["args"] = args
        captured["plugins"] = plugins
        return 0

    monkeypatch.setattr("pytest.main", fake_pytest_main)

    result = scene_test_compile.main(["examples", "-m", "not sdma", "--platform", "a2a3"])

    assert result == 0
    assert captured["args"] == [
        "examples",
        "-m",
        "not sdma",
        "--platform",
        "a2a3",
        "--collect-only",
    ]
    assert len(captured["plugins"]) == 1
