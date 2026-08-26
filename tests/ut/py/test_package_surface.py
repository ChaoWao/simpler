# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the `simpler` package's public attribute surface."""

import importlib
import subprocess
import sys

import pytest
import simpler


def test_worker_task_interface_and_comm_endpoints_are_advertised():
    assert "Worker" in simpler.__all__
    assert "task_interface" in simpler.__all__
    assert "comm_endpoints" in simpler.__all__
    assert "Worker" in dir(simpler)
    assert "task_interface" in dir(simpler)
    assert "comm_endpoints" in dir(simpler)


def test_logging_helpers_remain_exported():
    for name in ("get_logger", "get_current_config", "DEFAULT_THRESHOLD", "NUL"):
        assert name in simpler.__all__
        assert hasattr(simpler, name)


def test_comm_endpoints_region_access_surface_is_module_scoped():
    ce = importlib.import_module("simpler.comm_endpoints")
    for name in (
        "BufferAccessQuery",
        "buffer_adapter_candidates",
        "RegionAccessDecision",
        "RegionAccessDiagnostics",
        "RegionAccessReasonCode",
        "RegionAccessService",
        "StaticRegionAccessService",
    ):
        assert name in ce.__all__
        assert hasattr(ce, name)
        assert name not in simpler.__all__
        assert not hasattr(simpler, name)

    for removed in (
        "PlatformCapability",
        "CapabilityResult",
        "PlatformCapabilityCache",
        "StaticPlatformCapabilityCache",
        "_AdapterCandidate",
    ):
        assert removed not in ce.__all__


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(simpler, "definitely_not_an_attribute")  # noqa: B009 -- exercises __getattr__


def test_importing_simpler_does_not_load_the_worker_module():
    """`import simpler` must not pull in `simpler.worker`.

    `worker` imports `_task_interface` and `cloudpickle` unguarded, so resolving
    it eagerly here would make `import simpler` fail outright whenever the
    extension is missing or stale — including for callers that only want the
    logging helpers, which `_log` deliberately keeps working via a fallback.
    A subprocess is used because this test session has already imported it.

    Note `_task_interface` itself is a weaker signal: `_log` probes it inside a
    try/except, so it legitimately appears in `sys.modules` when it is present.
    """
    code = "import simpler, sys; print('simpler.worker' in sys.modules)"
    out = subprocess.run(  # noqa: S603 -- fixed argv, no shell
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False", f"import simpler eagerly loaded simpler.worker: {out.stdout!r} {out.stderr!r}"


def test_importing_simpler_survives_without_the_extension():
    """The logging surface must import even when `_task_interface` is absent."""
    code = "import sys; sys.modules['_task_interface'] = None; import simpler; print(callable(simpler.get_logger))"
    out = subprocess.run(  # noqa: S603 -- fixed argv, no shell
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "True", f"{out.stdout!r} {out.stderr!r}"


def test_importing_simpler_keeps_native_default_when_old_extension_lacks_sink_binder():
    """A stale extension may expose the threshold before it exposes the optional sink binder."""
    code = (
        "import sys, types; "
        "stub = types.ModuleType('_task_interface'); "
        "stub.DEFAULT_LOG_THRESHOLD = 30; "
        "sys.modules['_task_interface'] = stub; "
        "import simpler; print(simpler.DEFAULT_THRESHOLD)"
    )
    out = subprocess.run(  # noqa: S603 -- fixed argv, no shell
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "30", f"{out.stdout!r} {out.stderr!r}"


def test_comm_endpoints_requires_the_extension_and_stays_lazy():
    """W2 planning takes `BackendKind` from the extension, so it is not part of the
    no-extension surface.

    An earlier cut of the endpoint model kept its own Python backing enum and was importable
    without `_task_interface`. Reusing the canonical `BackendKind` is what stops a second backing
    vocabulary existing, and it costs that property — so the property is pinned here in its new
    form rather than left unstated: `import simpler` still survives (the submodule is lazy, and
    `_log` keeps its own fallback), while reaching `simpler.comm_endpoints` needs the extension.
    """
    code = (
        "import sys; sys.modules['_task_interface'] = None; import simpler; "
        "reached = True\n"
        "try:\n"
        "    simpler.comm_endpoints\n"
        "except ImportError:\n"
        "    reached = False\n"
        "print(reached)"
    )
    out = subprocess.run(  # noqa: S603 -- fixed argv, no shell
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False", f"comm_endpoints imported without the extension: {out.stdout!r}"

    ce = importlib.import_module("simpler.comm_endpoints")
    assert ce.BackendKind is importlib.import_module("_task_interface").BackendKind


def test_worker_resolves_to_the_same_object_as_the_submodule():
    # Not importorskip: a stale extension raises plain ImportError rather than
    # ModuleNotFoundError, which importorskip is deprecating for pytest 9.1.
    # Only an unusable _task_interface is a skip; `worker` also imports
    # cloudpickle, and losing that must fail rather than quietly skip.
    try:
        worker_module = importlib.import_module("simpler.worker")
    except ImportError as exc:
        if exc.name != "_task_interface":
            raise
        pytest.skip(f"needs a current nanobind extension: {exc}")

    assert simpler.Worker is worker_module.Worker
