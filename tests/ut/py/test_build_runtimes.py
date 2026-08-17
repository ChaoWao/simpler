# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the runtime pre-build entry point."""

import pytest


def test_build_all_prebuilds_sim_context_once(monkeypatch, tmp_path):
    from simpler_setup import build_runtimes  # noqa: PLC0415
    from simpler_setup.runtime_compiler import RuntimeCompiler  # noqa: PLC0415

    calls = []

    class FakeRuntimeBuilder:
        _LIB_DIR = None
        _CACHE_DIR = None

        def __init__(self, platform):
            self.platform = platform

        def ensure_sim_context(self, *, build):
            calls.append(("sim_context", self.platform, build))

        def get_binaries(self, name, *, build, build_shared, profiling_config):
            calls.append(("runtime", self.platform, name, build, build_shared, profiling_config))

    monkeypatch.setattr(build_runtimes, "RuntimeBuilder", FakeRuntimeBuilder)
    monkeypatch.setattr(build_runtimes, "discover_runtimes", lambda arch: ["first", "second"])
    monkeypatch.setattr(RuntimeCompiler, "_sanitizers", [])

    build_runtimes.build_all(
        lib_dir=tmp_path / "lib",
        cache_dir=tmp_path / "cache",
        platforms=["a2a3sim"],
    )

    assert calls.count(("sim_context", "a2a3sim", True)) == 1
    assert sorted(call for call in calls if call[0] == "runtime") == [
        ("runtime", "a2a3sim", "first", True, False, None),
        ("runtime", "a2a3sim", "second", True, False, None),
    ]


@pytest.mark.parametrize(
    "platforms",
    [
        ["a2a3", "a2a3sim"],
        ["a2a3sim", "a2a3"],
        ["a5", "a5sim"],
        ["a5sim", "a5"],
    ],
)
def test_mixed_sanitizer_builds_use_one_host_toolchain_regardless_of_order(monkeypatch, tmp_path, platforms):
    from simpler_setup import build_runtimes, pto_isa, runtime_compiler, toolchain  # noqa: PLC0415

    RuntimeCompiler = runtime_compiler.RuntimeCompiler
    calls = []

    class FakeDeviceToolchain:
        def __init__(self, *args, **kwargs):
            pass

    class FakeRuntimeBuilder:
        _LIB_DIR = None
        _CACHE_DIR = None

        def __init__(self, platform):
            self.platform = platform
            self.compiler = RuntimeCompiler.get_instance(platform)

        def ensure_sim_context(self, *, build):
            calls.append(("sim_context", self.platform, self.compiler.host_target.toolchain.cxx_path))

        def get_binaries(self, name, *, build, build_shared, profiling_config):
            calls.append(("runtime", self.platform, self.compiler.host_target.toolchain.cxx_path))

    monkeypatch.setattr(RuntimeCompiler, "_instances", {})
    monkeypatch.setattr(RuntimeCompiler, "_sanitizers", "")
    monkeypatch.setattr(RuntimeCompiler, "_ensure_host_compilers", lambda self: None)
    monkeypatch.setattr(runtime_compiler.env_manager, "ensure", lambda name: None)
    monkeypatch.setattr(runtime_compiler, "CCECToolchain", FakeDeviceToolchain)
    monkeypatch.setattr(runtime_compiler, "Aarch64GxxToolchain", FakeDeviceToolchain)
    monkeypatch.setattr(pto_isa, "ensure_pto_isa_root", lambda verbose: tmp_path / "pto-isa")
    monkeypatch.setattr(toolchain, "_is_gcc", lambda path: True)
    monkeypatch.setattr(build_runtimes, "RuntimeBuilder", FakeRuntimeBuilder)
    monkeypatch.setattr(build_runtimes, "discover_runtimes", lambda arch: ["runtime"])
    monkeypatch.setattr(build_runtimes, "platform_embeds_pto_isa", lambda platform: False)

    build_runtimes.build_all(
        lib_dir=tmp_path / "lib",
        cache_dir=tmp_path / "cache",
        platforms=platforms,
        sanitizer="asan",
    )

    assert [call for call in calls if call[0] == "sim_context"] == [
        ("sim_context", next(p for p in platforms if p.endswith("sim")), "g++-15")
    ]
    assert {compiler for _, _, compiler in calls} == {"g++-15"}


def test_profiling_config_completes_defaults_and_validates_dependencies():
    from simpler_setup.build_runtimes import _profiling_config  # noqa: PLC0415

    assert _profiling_config(
        {
            "SIMPLER_DFX": None,
            "SIMPLER_ORCH_PROFILING": 1,
            "SIMPLER_SCHED_PROFILING": None,
            "SIMPLER_TENSORMAP_PROFILING": None,
        }
    ) == {
        "SIMPLER_DFX": "1",
        "SIMPLER_ORCH_PROFILING": "1",
        "SIMPLER_SCHED_PROFILING": "0",
        "SIMPLER_TENSORMAP_PROFILING": "0",
    }

    with pytest.raises(ValueError, match="TENSORMAP_PROFILING requires"):
        _profiling_config(
            {
                "SIMPLER_DFX": 1,
                "SIMPLER_ORCH_PROFILING": 0,
                "SIMPLER_SCHED_PROFILING": 0,
                "SIMPLER_TENSORMAP_PROFILING": 1,
            }
        )
