# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Give the private native-run lifecycle test an empty L2 callable table."""

import pytest


@pytest.fixture(scope="class")
def st_worker(request, st_platform, device_pool):
    cls = request.node.cls
    if cls is None or not hasattr(cls, "_st_runtime"):
        pytest.skip("isolated st_worker requires a SceneTestCase subclass")

    ids = device_pool.allocate(1)
    if not ids:
        pytest.fail("no devices available for isolated L2 worker")
    try:
        from simpler.worker import Worker  # noqa: PLC0415

        worker = Worker(level=2, device_id=ids[0], platform=st_platform, runtime=cls._st_runtime)
        worker.init()
        try:
            yield worker
        finally:
            worker.close()
    finally:
        device_pool.release(ids)
