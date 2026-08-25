# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import traceback

import pytest

_RUNTIMES = ("host_build_graph", "tensormap_and_ringbuffer")
_SIM_PLATFORMS = ("a2a3sim", "a5sim")


def _require_runtime(platform: str, runtime: str) -> None:
    from simpler_setup.runtime_builder import RuntimeBuilder

    try:
        RuntimeBuilder(platform=platform).get_binaries(runtime, build=bool(os.environ.get("PTO_UT_BUILD")))
    except FileNotFoundError as exc:
        pytest.skip(f"{platform}/{runtime} runtime binaries unavailable: {exc}")


def _make_worker(*, level: int, platform: str, runtime: str, device_id: int):
    from simpler.worker import Worker

    common = {"level": level, "platform": platform, "runtime": runtime}
    if level == 2:
        return Worker(device_id=device_id, **common)
    return Worker(device_ids=[device_id], num_sub_workers=0, **common)


def _acl_memory_info() -> tuple[int, int]:
    lib = ctypes.CDLL("libascendcl.so")
    get_mem_info = lib.aclrtGetMemInfo
    get_mem_info.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    get_mem_info.restype = ctypes.c_int
    free_bytes = ctypes.c_size_t()
    total_bytes = ctypes.c_size_t()
    acl_hbm_mem = 1
    rc = get_mem_info(acl_hbm_mem, ctypes.byref(free_bytes), ctypes.byref(total_bytes))
    assert rc == 0
    return int(free_bytes.value), int(total_bytes.value)


def _onboard_query_entry(level: int, platform: str, runtime: str, device_id: int, result_queue) -> None:
    """Keep ACL init/reset out of pytest's process so later L3 cases may fork safely."""
    worker = None
    result: dict[str, object] = {}
    try:
        worker = _make_worker(level=level, platform=platform, runtime=runtime, device_id=device_id)
        worker.init()
        info = worker.device_memory_info()
        result["info"] = (int(info.free_bytes), int(info.total_bytes))
        result["unpacked"] = tuple(info)
        if level == 2:
            result["acl"] = _acl_memory_info()
    except BaseException:  # noqa: BLE001 -- marshal child failures back to pytest
        result["error"] = traceback.format_exc()
    finally:
        if worker is not None:
            worker.close()
    result_queue.put(result)


def test_device_memory_info_propagates_worker_lifecycle_errors():
    worker = _make_worker(level=2, platform="a2a3sim", runtime="host_build_graph", device_id=0)
    with pytest.raises(RuntimeError, match=r"requires an initialized \(READY\) worker"):
        worker.device_memory_info()


@pytest.mark.parametrize("platform", _SIM_PLATFORMS)
@pytest.mark.parametrize("runtime", _RUNTIMES)
@pytest.mark.parametrize("level", (2, 3))
def test_simulator_reports_device_memory_info_as_unsupported(platform, runtime, level):
    _require_runtime(platform, runtime)
    worker = _make_worker(level=level, platform=platform, runtime=runtime, device_id=0)
    worker.init()
    try:
        with pytest.raises(NotImplementedError, match="device_memory_info"):
            worker.device_memory_info()
        if level == 2:
            assert worker._chip_worker is not None
            with pytest.raises(NotImplementedError, match="device_memory_info"):
                worker._chip_worker.device_memory_info()
    finally:
        worker.close()


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.parametrize("runtime", _RUNTIMES)
@pytest.mark.parametrize("level", (2, 3))
def test_onboard_device_memory_info_matches_acl(st_platform, st_device_ids, runtime, level):
    device_id = int(st_device_ids[0])
    _require_runtime(st_platform, runtime)
    ctx = mp.get_context("fork")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_onboard_query_entry, args=(level, st_platform, runtime, device_id, result_queue))
    process.start()
    process.join(timeout=300)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        pytest.fail("device_memory_info child did not finish")
    assert process.exitcode == 0, f"device_memory_info child exited with {process.exitcode}"
    result = result_queue.get(timeout=5)
    assert "error" not in result, result.get("error")

    free_bytes, total_bytes = result["info"]
    assert isinstance(free_bytes, int)
    assert isinstance(total_bytes, int)
    assert 0 <= free_bytes <= total_bytes
    assert total_bytes > 0
    assert result["unpacked"] == result["info"]
    if level == 2:
        assert result["info"] == result["acl"]
