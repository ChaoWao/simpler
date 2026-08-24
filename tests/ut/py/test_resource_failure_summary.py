# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for root-conftest behavior: Resource phase failure summaries and
device-poison classification."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from simpler_setup.parallel_scheduler import JobResult

_ROOT = Path(__file__).resolve().parents[3]


def _load_root_conftest():
    spec = importlib.util.spec_from_file_location("_root_conftest_resource_summary", _ROOT / "conftest.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_emit_resource_failure_summary_prints_nodeid_and_annotation(capsys):
    cf = _load_root_conftest()
    results = [
        JobResult(
            label="standalone pass",
            returncode=0,
            device_ids=[0],
            output="pass output\n",
            duration_s=1.0,
        ),
        JobResult(
            label="standalone bad%case\nname",
            returncode=-11,
            device_ids=[4, 5],
            output=(
                "line1\n"
                "E       RuntimeError: run_prepared failed with code 507018\n"
                "simpler runtime failed: orch_error_code=0 sched_error_code=100 runtime_status=-100\n"
                "scheduler timeout sub_class=S1:running-stalled\n"
            ),
            duration_s=12.34,
            nodeid="tests/st/runtime_fatal_codes/test_probe.py::test_bad[param]",
        ),
    ]

    cf._emit_resource_failure_summary(results)

    out = capsys.readouterr().out
    assert "*** Resource phase failed: 1 child job(s) ***" in out
    assert (
        "::error title=Resource phase failed::"
        "tests/st/runtime_fatal_codes/test_probe.py::test_bad[param] "
        "(standalone bad%25case%0Aname) rc=-11 devices=[4,5]"
    ) in out
    assert "- nodeid=tests/st/runtime_fatal_codes/test_probe.py::test_bad[param]" in out
    assert "label=standalone bad%case\nname" in out
    assert "rc=-11 devices=[4, 5] duration=12.3s" in out
    assert "full output is in the Resource child group above" in out
    assert "hint:" not in out
    assert "line1" not in out
    assert "RuntimeError: run_prepared failed with code 507018" not in out
    assert "simpler runtime failed: orch_error_code=0 sched_error_code=100 runtime_status=-100" not in out
    assert "scheduler timeout sub_class=S1:running-stalled" not in out
    assert "standalone pass" not in out


def test_emit_resource_failure_summary_can_emit_compact_recap(capsys):
    cf = _load_root_conftest()
    results = [
        JobResult(
            label="standalone failed",
            returncode=2,
            device_ids=[7],
            output="hidden tail\n",
            duration_s=3.0,
            nodeid="tests/st/test_failed.py::test_failed",
        )
    ]

    cf._emit_resource_failure_summary(
        results,
        emit_annotations=False,
        heading="Resource phase failed recap",
    )

    out = capsys.readouterr().out
    assert "*** Resource phase failed recap: 1 child job(s) ***" in out
    assert "::error" not in out
    assert "nodeid=tests/st/test_failed.py::test_failed" in out
    assert "label=standalone failed" in out
    assert "rc=2 devices=[7] duration=3.0s" in out
    assert "full output is in the Resource child group above" in out
    assert "hidden tail" not in out


def test_device_poison_codes_key_on_the_host_band_not_the_latched_one():
    """Poison classification reads host-band codes only.

    The a5 DeviceRunner fail-fast ("marked unusable") and a poisoned card whose
    CANN code an intermediate layer flattened both report the host-side generic
    code, and both must trigger the worker rebuild. A device-latched code must
    not: SCOPE_DEADLOCK is an orchestration bug, and treating it as a poisoned
    context turns one red test into a misleading runtime-wide skip.

    The two bands are defined in src/common/worker/runtime_c_api.h; the set
    here has to move with them, which is what this test pins.
    """
    cf = _load_root_conftest()

    # Host band: PTO_RUNTIME_ERR_INTERNAL, plus the CANN sticky-error codes.
    assert cf._requires_l2_worker_retirement_msg("prepare_native_run failed with code -1000")
    assert cf._requires_l2_worker_retirement_msg("simpler_init failed with code 507899")

    # Device-latched band (-1..-999): never poison, whatever the mechanism.
    assert not cf._requires_l2_worker_retirement_msg("prepare_native_run failed with code -1")  # SCOPE_DEADLOCK
    assert not cf._requires_l2_worker_retirement_msg("prepare_native_run failed with code -3")  # FLOW_CONTROL_DEADLOCK
    assert not cf._requires_l2_worker_retirement_msg("prepare_native_run failed with code -100")  # SCHEDULER_TIMEOUT
    assert not any(-999 <= code <= -1 for code in cf._DEVICE_POISON_CODES)
