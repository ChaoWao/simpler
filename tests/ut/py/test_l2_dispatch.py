# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for L2 dispatcher child-command construction."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[3]


def _load_root_conftest():
    spec = importlib.util.spec_from_file_location("_root_conftest_l2_dispatch", _ROOT / "conftest.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Config:
    def __init__(self, *, xdist_active: bool, xdist_blocked: bool, disable_arg: bool = False):
        args = ["first.py", "second.py"]
        if disable_arg:
            args.extend(["-p", "no:xdist"])
        args.extend(["--max-parallel", "2"])
        self.invocation_params = SimpleNamespace(
            args=tuple(args),
            dir=_ROOT,
        )
        self.pluginmanager = SimpleNamespace(
            hasplugin=lambda name: name == "xdist" and xdist_active,
            is_blocked=lambda name: name == "xdist" and xdist_blocked,
        )

    def getoption(self, name, default=None):
        return {
            "--device": "0-1",
            "--exitfirst": False,
            "--platform": "a2a3sim",
            "--manual": "include",
            "--max-parallel": "2",
        }.get(name, default)


def test_l2_dispatch_does_not_add_xdist_options_when_plugin_is_disabled(monkeypatch):
    cf = _load_root_conftest()
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cf.subprocess, "run", fake_run)
    config = _Config(xdist_active=False, xdist_blocked=True, disable_arg=True)
    items = [
        SimpleNamespace(cls=SimpleNamespace(_st_runtime="host_build_graph", _st_level=2)),
        SimpleNamespace(cls=SimpleNamespace(_st_runtime="tensormap_and_ringbuffer", _st_level=2)),
    ]
    session = SimpleNamespace(config=config, items=items, testsfailed=0, testscollected=0)

    assert cf._dispatch_test_phases(session, []) is True
    assert len(commands) == 2
    for command in commands:
        assert "no:xdist" in command
        assert "-n" not in command
        assert "--dist" not in command


def test_l2_dispatch_adds_xdist_options_when_plugin_is_active(monkeypatch):
    cf = _load_root_conftest()
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cf.subprocess, "run", fake_run)
    config = _Config(xdist_active=True, xdist_blocked=False)
    items = [SimpleNamespace(cls=SimpleNamespace(_st_runtime="host_build_graph", _st_level=2))]
    session = SimpleNamespace(config=config, items=items, testsfailed=0, testscollected=0)

    assert cf._dispatch_test_phases(session, []) is True
    assert len(commands) == 1
    assert commands[0][-8:] == [
        "--runtime",
        "host_build_graph",
        "--level",
        "2",
        "-n",
        "2",
        "--dist",
        "loadfile",
    ]
