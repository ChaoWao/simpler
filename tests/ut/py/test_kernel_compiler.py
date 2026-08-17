# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for KernelCompiler host toolchain selection."""

import pytest


@pytest.mark.parametrize("platform", ["a2a3", "a2a3sim", "a5", "a5sim"])
@pytest.mark.parametrize(("sanitizers", "expected_cxx"), [("", "g++"), ("address", "g++-15")])
def test_host_orchestration_matches_runtime_sanitizer_toolchain(monkeypatch, platform, sanitizers, expected_cxx):
    from simpler_setup import kernel_compiler, toolchain  # noqa: PLC0415

    class FakeDeviceToolchain:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(kernel_compiler.KernelCompiler, "_sanitizers", sanitizers)
    monkeypatch.setattr(kernel_compiler.env_manager, "ensure", lambda name: None)
    monkeypatch.setattr(kernel_compiler, "CCECToolchain", FakeDeviceToolchain)
    monkeypatch.setattr(kernel_compiler, "Aarch64GxxToolchain", FakeDeviceToolchain)
    monkeypatch.setattr(toolchain, "_is_gcc", lambda path: True)

    compiler = kernel_compiler.KernelCompiler(platform)

    assert compiler.host_gxx.cxx_path == expected_cxx


@pytest.mark.parametrize(
    ("target_platform", "target_runtime", "expects_host_logger"),
    [
        ("a2a3sim", "host_build_graph", True),
        ("a2a3sim", "tensormap_and_ringbuffer", True),
        ("a2a3", "host_build_graph", True),
        ("a2a3", "tensormap_and_ringbuffer", False),
        ("a5sim", "host_build_graph", True),
        ("a5sim", "tensormap_and_ringbuffer", True),
        ("a5", "host_build_graph", True),
        ("a5", "tensormap_and_ringbuffer", False),
    ],
)
def test_host_orchestration_is_a_self_contained_log_consumer(
    monkeypatch, target_platform, target_runtime, expects_host_logger
):
    from simpler_setup import kernel_compiler, toolchain  # noqa: PLC0415

    class FakeDeviceToolchain:
        is_host = False

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(kernel_compiler.env_manager, "ensure", lambda name: None)
    monkeypatch.setattr(kernel_compiler, "CCECToolchain", FakeDeviceToolchain)
    monkeypatch.setattr(kernel_compiler, "Aarch64GxxToolchain", FakeDeviceToolchain)
    monkeypatch.setattr(toolchain, "_is_gcc", lambda path: True)

    compiler = kernel_compiler.KernelCompiler(target_platform)
    include_dirs, sources = compiler.get_orchestration_cache_inputs(target_runtime)
    source_names = {source.rsplit("/", 1)[-1] for source in sources}

    if expects_host_logger:
        assert {"host_log.cpp", "unified_log_host.cpp"} <= source_names
        assert any(include_dir.endswith("src/common/log/include") for include_dir in include_dirs)
        assert "-pthread" in compiler._orchestration_link_flags(compiler._orchestration_toolchain(target_runtime))
    else:
        assert "host_log.cpp" not in source_names
        assert "unified_log_host.cpp" not in source_names
        assert "-pthread" not in compiler._orchestration_link_flags(compiler._orchestration_toolchain(target_runtime))
