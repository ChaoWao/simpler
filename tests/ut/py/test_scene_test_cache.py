# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Regression: SceneTestCase compile cache must release its ChipCallables.

The session-lifetime ``_compile_cache`` in ``simpler_setup.scene_test`` used
to hold every compiled ``ChipCallable`` until Python interpreter shutdown.
At shutdown the nanobind module destructor can run before module globals
are cleared, which surfaces as ``nanobind: leaked N instances of type
_task_interface.ChipCallable`` on stderr. ``clear_compile_cache`` (invoked
from ``pytest_sessionfinish``) drops the cache and forces GC so those
instances die while the extension is still live.
"""

from __future__ import annotations

import importlib

from _task_interface import ArgDirection, ChipCallable  # pyright: ignore[reportMissingImports]

# ``simpler_setup/__init__.py`` re-exports the ``scene_test`` *decorator*,
# which shadows the submodule attribute when accessed via ``simpler_setup``.
# Importing the names directly from the submodule avoids that ambiguity.
from simpler_setup.scene_test import (
    SceneTestCase,
    _compile_cache,
    _pto_isa_compile_cache_token,
    clear_compile_cache,
    l3_compile_cache_key,
)


def _build_chip_callable(tag: str) -> ChipCallable:
    return ChipCallable.build(
        signature=[ArgDirection.IN],
        func_name=tag,
        binary=b"\x00" * 16,
        children=[],
    )


def test_clear_compile_cache_drops_cached_chip_callables():
    """clear_compile_cache empties the dict so nanobind instances can die.

    The leak this guards against is ``_compile_cache`` retaining every
    compiled ``ChipCallable`` for the full pytest session. The regression
    surface is therefore "dict still has entries after the cleanup call"
    — if someone breaks ``clear_compile_cache`` (forgets the ``.clear()``,
    swaps the cache key schema, introduces a secondary holder that the
    cleanup doesn't know about), this assertion fails.
    """
    _compile_cache.clear()
    for i in range(3):
        _compile_cache[("t", "plat", f"rt{i}", "pin")] = _build_chip_callable(f"n{i}")
    assert len(_compile_cache) == 3

    clear_compile_cache()

    assert _compile_cache == {}


def test_pto_isa_compile_cache_token_tracks_pin(monkeypatch):
    """Session cache keys must change when pto_isa.pin changes."""
    pin_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    pin_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    monkeypatch.setattr("simpler_setup.pto_isa.read_pto_isa_pin", lambda: pin_a)
    assert _pto_isa_compile_cache_token() == pin_a
    monkeypatch.setattr("simpler_setup.pto_isa.read_pto_isa_pin", lambda: pin_b)
    assert _pto_isa_compile_cache_token() == pin_b


def test_compile_cache_keys_include_module(monkeypatch):
    """Same-named scene classes from different test modules must not share binaries."""
    monkeypatch.setattr("simpler_setup.pto_isa.read_pto_isa_pin", lambda: "pin")
    captured = []
    scene_test_module = importlib.import_module("simpler_setup.scene_test")
    monkeypatch.setattr(
        scene_test_module,
        "_compile_chip_callable_from_spec",
        lambda spec, platform, runtime, cache_key: captured.append(cache_key),
    )

    first = type("TestScene", (SceneTestCase,), {"__module__": "tests.first", "_st_runtime": "a5", "CALLABLE": {}})
    second = type("TestScene", (SceneTestCase,), {"__module__": "tests.second", "_st_runtime": "a5", "CALLABLE": {}})
    first.compile_chip_callable("a5")
    second.compile_chip_callable("a5")

    assert captured[0] != captured[1]
    assert l3_compile_cache_key("tests.first", "TestScene", "child", "a5", "a5") != l3_compile_cache_key(
        "tests.second", "TestScene", "child", "a5", "a5"
    )
