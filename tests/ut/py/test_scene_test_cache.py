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
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock

import pytest
from _task_interface import ArgDirection, ChipCallable  # pyright: ignore[reportMissingImports]

# ``simpler_setup/__init__.py`` re-exports the ``scene_test`` *decorator*,
# which shadows the submodule attribute when accessed via ``simpler_setup``.
# Importing the names directly from the submodule avoids that ambiguity.
from simpler_setup import scene_test_cache
from simpler_setup.scene_test import (
    SceneTestCase,
    _compile_cache,
    _compile_chip_callable_from_spec,
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


class _FakeKernelCompiler:
    _sanitizers = ""
    cache_schema = 1
    orchestration_compiles = 0
    incore_compiles = 0

    def __init__(self, platform):
        self.platform = platform

    def get_orchestration_include_dirs(self, runtime):
        return []

    def get_incore_include_dirs(self):
        return []

    def get_orchestration_cache_inputs(self, runtime):
        return [], []

    def compile_cache_token(self, runtime, core_types):
        return {"schema": self.cache_schema}

    def compile_orchestration(self, runtime, source):
        type(self).orchestration_compiles += 1
        return Path(source).read_bytes()

    def compile_incore(self, source, **kwargs):
        type(self).incore_compiles += 1
        return Path(source).read_bytes()


def _cacheable_spec(tmp_path):
    header = tmp_path / "shared.h"
    header.write_text("constexpr int VALUE = 1;\n")
    orchestration = tmp_path / "orchestration.cpp"
    orchestration.write_text('#include "shared.h"\nextern "C" void orchestration() {}\n')
    incore = tmp_path / "kernel.cpp"
    incore.write_text('#include "shared.h"\nextern "C" void kernel_entry() {}\n')
    spec = {
        "orchestration": {
            "source": str(orchestration),
            "function_name": "orchestration",
            "signature": [ArgDirection.IN],
        },
        "incores": [
            {
                "source": str(incore),
                "func_id": 0,
                "core_type": "aiv",
                "signature": [ArgDirection.IN],
                "extra_include_dirs": [str(tmp_path)],
            }
        ],
    }
    return spec, header


def _configure_fake_compilation(monkeypatch, tmp_path):
    pto_isa_root = tmp_path / "pto-isa"
    (pto_isa_root / "include" / "pto").mkdir(parents=True)
    monkeypatch.setattr("simpler_setup.kernel_compiler.KernelCompiler", _FakeKernelCompiler)
    monkeypatch.setattr("simpler_setup.pto_isa.ensure_pto_isa_root", lambda: str(pto_isa_root))
    monkeypatch.setattr(scene_test_cache, "KERNEL_CACHE_DIR", tmp_path / "cache")
    _FakeKernelCompiler.cache_schema = 1
    _FakeKernelCompiler.orchestration_compiles = 0
    _FakeKernelCompiler.incore_compiles = 0
    clear_compile_cache()


def test_compile_cache_survives_session_cache_clear(monkeypatch, tmp_path):
    _configure_fake_compilation(monkeypatch, tmp_path)
    spec, _header = _cacheable_spec(tmp_path)
    cache_key = ("TestScene", "a2a3sim", "host_build_graph", "pin")

    first = _compile_chip_callable_from_spec(spec, "a2a3sim", "host_build_graph", cache_key)
    assert _FakeKernelCompiler.orchestration_compiles == 1
    assert _FakeKernelCompiler.incore_compiles == 1

    clear_compile_cache()
    second = _compile_chip_callable_from_spec(spec, "a2a3sim", "host_build_graph", cache_key)

    assert _FakeKernelCompiler.orchestration_compiles == 1
    assert _FakeKernelCompiler.incore_compiles == 1
    assert second.func_name == first.func_name
    assert second.binary_size == first.binary_size
    assert second.child_count == first.child_count


def test_compile_cache_invalidates_transitive_include(monkeypatch, tmp_path):
    _configure_fake_compilation(monkeypatch, tmp_path)
    spec, header = _cacheable_spec(tmp_path)
    cache_key = ("TestScene", "a2a3sim", "host_build_graph", "pin")

    _compile_chip_callable_from_spec(spec, "a2a3sim", "host_build_graph", cache_key)
    clear_compile_cache()
    header.write_text("constexpr int VALUE = 2;\n")
    _compile_chip_callable_from_spec(spec, "a2a3sim", "host_build_graph", cache_key)

    assert _FakeKernelCompiler.orchestration_compiles == 2
    assert _FakeKernelCompiler.incore_compiles == 2


def test_compile_cache_invalidates_compiler_schema(monkeypatch, tmp_path):
    _configure_fake_compilation(monkeypatch, tmp_path)
    spec, _header = _cacheable_spec(tmp_path)
    cache_key = ("TestScene", "a2a3sim", "host_build_graph", "pin")

    _FakeKernelCompiler.cache_schema = 1
    _compile_chip_callable_from_spec(spec, "a2a3sim", "host_build_graph", cache_key)
    clear_compile_cache()
    _FakeKernelCompiler.cache_schema = 2
    _compile_chip_callable_from_spec(spec, "a2a3sim", "host_build_graph", cache_key)

    assert _FakeKernelCompiler.orchestration_compiles == 2
    assert _FakeKernelCompiler.incore_compiles == 2


def test_compile_cache_rebuilds_incomplete_entry(monkeypatch, tmp_path):
    _configure_fake_compilation(monkeypatch, tmp_path)
    spec, _header = _cacheable_spec(tmp_path)
    cache_key = ("TestScene", "a2a3sim", "host_build_graph", "pin")

    _compile_chip_callable_from_spec(spec, "a2a3sim", "host_build_graph", cache_key)
    binary_path = next((tmp_path / "cache").glob("*/callable.bin"))
    binary_path.write_bytes(b"incomplete")
    clear_compile_cache()
    _compile_chip_callable_from_spec(spec, "a2a3sim", "host_build_graph", cache_key)

    assert _FakeKernelCompiler.orchestration_compiles == 2
    assert _FakeKernelCompiler.incore_compiles == 2


def test_compile_artifact_key_is_independent_of_checkout_path(tmp_path):
    keys = []
    for root_name in ("runner-a", "runner-b"):
        root = tmp_path / root_name
        root.mkdir()
        (root / "shared.h").write_text("constexpr int VALUE = 1;\n")
        source = root / "kernel.cpp"
        source.write_text('#include "shared.h"\nextern "C" void kernel_entry() {}\n')
        keys.append(scene_test_cache.compile_artifact_key({"platform": "a2a3"}, [(source, [root])]))

    assert keys[0] == keys[1]


def test_compile_artifact_key_includes_callable_abi(monkeypatch, tmp_path):
    source = tmp_path / "kernel.cpp"
    source.write_text('extern "C" void kernel_entry() {}\n')

    monkeypatch.setattr(scene_test_cache, "_chip_callable_abi_token", lambda: "abi-a")
    first = scene_test_cache.compile_artifact_key({}, [(source, [])])
    monkeypatch.setattr(scene_test_cache, "_chip_callable_abi_token", lambda: "abi-b")
    second = scene_test_cache.compile_artifact_key({}, [(source, [])])

    assert first != second


def test_concurrent_cache_misses_compile_once(monkeypatch, tmp_path):
    monkeypatch.setattr(scene_test_cache, "KERNEL_CACHE_DIR", tmp_path / "cache")
    start = Barrier(2)
    count_lock = Lock()
    compile_count = 0

    def compile_callable():
        nonlocal compile_count
        with count_lock:
            compile_count += 1
        return _build_chip_callable("shared")

    def load_or_compile():
        start.wait()
        return scene_test_cache.get_or_compile("a" * 64, compile_callable)

    with ThreadPoolExecutor(max_workers=2) as executor:
        callables = list(executor.map(lambda _: load_or_compile(), range(2)))

    assert compile_count == 1
    assert [callable_obj.func_name for callable_obj in callables] == ["shared", "shared"]


def test_artifact_logic_token_covers_the_compilation_modules(monkeypatch):
    from simpler_setup import kernel_compiler

    assert set(kernel_compiler._ARTIFACT_LOGIC_MODULES) == {"kernel_compiler.py", "toolchain.py", "elf_parser.py"}

    def token_over(modules):
        monkeypatch.setattr(kernel_compiler, "_ARTIFACT_LOGIC_MODULES", modules)
        kernel_compiler._artifact_logic_token.cache_clear()
        return kernel_compiler._artifact_logic_token()

    over_compiler = token_over(("kernel_compiler.py",))
    over_toolchain = token_over(("toolchain.py",))
    over_both = token_over(("kernel_compiler.py", "toolchain.py"))
    kernel_compiler._artifact_logic_token.cache_clear()

    assert len({over_compiler, over_toolchain, over_both}) == 3


def test_publish_prunes_entries_past_the_retention_window(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(scene_test_cache, "KERNEL_CACHE_DIR", cache_dir)

    stale = cache_dir / ("b" * 64)
    stale.mkdir(parents=True)
    (stale / "callable.bin").write_bytes(b"stale")
    stale_lock = cache_dir / ".locks" / "old.lock"
    stale_lock.parent.mkdir(parents=True)
    stale_lock.write_text("")
    expired = time.time() - scene_test_cache._ENTRY_RETENTION_S - 60
    for path in (stale, stale_lock):
        os.utime(path, (expired, expired))

    scene_test_cache.get_or_compile("c" * 64, lambda: _build_chip_callable("fresh"))

    assert not stale.exists()
    assert not stale_lock.exists()
    assert (cache_dir / ("c" * 64) / "callable.bin").is_file()


def test_cache_hit_refreshes_entry_mtime(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(scene_test_cache, "KERNEL_CACHE_DIR", cache_dir)

    key = "d" * 64
    scene_test_cache.get_or_compile(key, lambda: _build_chip_callable("kept"))
    entry = cache_dir / key
    aged = time.time() - scene_test_cache._ENTRY_RETENTION_S - 60
    os.utime(entry, (aged, aged))

    scene_test_cache.get_or_compile(key, lambda: pytest.fail("cached entry should not recompile"))

    assert entry.stat().st_mtime > aged


def test_unwritable_cache_dir_falls_back_to_plain_compilation(monkeypatch, tmp_path):
    read_only = tmp_path / "read-only"
    read_only.mkdir(mode=0o500)
    monkeypatch.setattr(scene_test_cache, "KERNEL_CACHE_DIR", read_only / "kernels")

    compiled = scene_test_cache.get_or_compile("e" * 64, lambda: _build_chip_callable("uncached"))

    assert compiled.func_name == "uncached"
    assert not (read_only / "kernels").exists()


def test_source_scan_is_memoized_per_file(monkeypatch, tmp_path):
    scene_test_cache._scan_cache.clear()
    header = tmp_path / "shared.h"
    header.write_text("constexpr int VALUE = 1;\n")
    sources = []
    for index in range(3):
        source = tmp_path / f"kernel{index}.cpp"
        source.write_text(f'#include "shared.h"\nextern "C" void kernel_entry{index}() {{}}\n')
        sources.append((source, [tmp_path]))

    reads = []
    original_read_bytes = Path.read_bytes

    def counting_read_bytes(self):
        reads.append(self)
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)
    scene_test_cache.compile_artifact_key({}, sources)

    assert reads.count(header.resolve()) == 1
    assert len(reads) == 4
