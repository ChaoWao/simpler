# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pod ST for the complete L4-to-L3 direct MPI path."""

import importlib.util
import os
import shutil

import pytest

from simpler_setup import SceneTestLevel, scene_level

from .main import run


def _device_spec(device_ids) -> str:
    return ",".join(str(device_id) for device_id in device_ids)


def _require_mpi_direct_pod_env() -> tuple[str, str, str]:
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun is not on PATH")
    if importlib.util.find_spec("mpi4py") is None:
        pytest.skip("mpi4py is not installed")
    local_ip = os.environ.get("POD_LOCAL_IP", "")
    if not local_ip:
        pytest.skip("POD_LOCAL_IP is required for the L4/rank-0 startup gate")
    mpi_python = os.environ.get("POD_MPI_PYTHON", "")
    if not mpi_python:
        pytest.skip("POD_MPI_PYTHON is required on both MPI hosts")
    return mpirun, local_ip, mpi_python


@scene_level(SceneTestLevel.POD)
@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
@pytest.mark.pod_remote_device_count(2)
def test_vector_add_mpi_direct_l3(st_platform, st_device_ids, st_pod_peer, st_pod_remote_device_ids, st_pod_logs):
    mpirun, local_ip, mpi_python = _require_mpi_direct_pod_env()
    remote_host, _daemon_port = st_pod_peer.endpoint.rsplit(":", 1)
    rc = run(
        local_host=local_ip,
        remote_host=remote_host,
        python_executable=mpi_python,
        local_devices=_device_spec(st_device_ids),
        remote_devices=_device_spec(st_pod_remote_device_ids),
        platform=st_platform,
        startup_timeout=st_pod_peer.session_timeout_s,
        session_timeout=st_pod_peer.session_timeout_s,
        mpirun_path=mpirun,
    )
    assert rc == 0
