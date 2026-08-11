# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolated L2 worker for the concurrent-prepare white-box stress.

The default ``st_worker`` (root conftest) is shared across L2 ST classes in a
session-scoped pool — correct for ordinary business tests, but this stress
drives the ChipWorker native-run lane directly (registering a callable slot and
submitting overlapping runs), so it needs a fresh worker whose private slot
table and native-run lane start empty and are not left with residue for other
pooled tests.

Override ``st_worker`` here as class-scope, building a fresh L2 worker that does
**not** enter ``_l2_worker_pool``. Cost: one extra init/close per test class.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="class")
def st_worker(request, st_platform, device_pool):
    """Yield a fresh class-scope L2 worker that does not enter the shared pool."""
    cls = request.node.cls
    if cls is None or not hasattr(cls, "_st_runtime"):
        pytest.skip("isolated st_worker requires a SceneTestCase subclass")

    runtime = cls._st_runtime

    ids = device_pool.allocate(1)
    if not ids:
        pytest.fail("no devices available for isolated L2 worker")
    dev_id = ids[0]
    try:
        from simpler.worker import Worker  # noqa: PLC0415

        w = Worker(
            level=2,
            device_id=dev_id,
            platform=st_platform,
            runtime=runtime,
        )
        w.init()
        try:
            yield w
        finally:
            w.close()
    finally:
        device_pool.release(ids)
