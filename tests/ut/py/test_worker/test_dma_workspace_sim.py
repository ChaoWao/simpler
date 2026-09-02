# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Sim-backend test for the async-DMA workspace request carried by ``simpler_init``.

Simulation provides no async-DMA engine, so ``enable_sdma=True`` has no workspace
to hand a kernel. The contract is that such a Worker fails to come up rather than
reaching its first run reading a zero address it expected to be live — the
rejection is the only thing standing between an unsupported request and a silent
wrong answer, and it is easy to lose to a refactor that derives the provisioned
set from what the platform supports (an empty set looks like success).
"""

from __future__ import annotations

import pytest


def _make_sim_worker(*, enable_sdma: bool):
    from simpler.worker import Worker

    from simpler_setup.runtime_builder import RuntimeBuilder

    try:
        RuntimeBuilder(platform="a2a3sim").get_binaries("tensormap_and_ringbuffer")
    except FileNotFoundError as e:
        pytest.skip(f"a2a3sim runtime binaries unavailable: {e}")
    return Worker(
        level=2,
        device_id=0,
        platform="a2a3sim",
        runtime="tensormap_and_ringbuffer",
        enable_sdma=enable_sdma,
    )


class TestSimRejectsSdma:
    def test_enable_sdma_fails_init(self):
        worker = _make_sim_worker(enable_sdma=True)
        # -1001 is PTO_RUNTIME_ERR_UNSUPPORTED. Pinning the code, not just "some
        # exception", is what distinguishes a rejected request from sim failing
        # to start for an unrelated reason.
        with pytest.raises(RuntimeError, match=r"simpler_init failed with code -1001"):
            worker.init()
        worker.close()

    def test_without_sdma_init_succeeds(self):
        # Positive control: the same Worker comes up when it does not ask for an
        # engine sim has no provider for, so the failure above is attributable to
        # the request rather than to sim being unable to start at all.
        worker = _make_sim_worker(enable_sdma=False)
        worker.init()
        worker.close()
