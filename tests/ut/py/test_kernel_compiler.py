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
