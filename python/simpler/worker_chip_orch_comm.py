# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3-L2 orchestrator communication facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .comm_provider import RegionPartLocalView, validate_independent_local_views
from .comm_region import (
    MaterializationError,
    NotifyOp,
    RegionInstanceState,
    SignalTestResult,
    WaitCmp,
)

_MAX_SIGNED_CHRONO_TIMEOUT_NS = 2**63 - 1
WORKER_CHIP_ORCH_COMM_MAGIC = 0x4C334C32
WORKER_CHIP_ORCH_COMM_ABI_MAJOR = 3
WORKER_CHIP_ORCH_COMM_ABI_MINOR = 0
_REGION_MAGIC_VERSION = (
    (WORKER_CHIP_ORCH_COMM_MAGIC << 32) | (WORKER_CHIP_ORCH_COMM_ABI_MAJOR << 16) | WORKER_CHIP_ORCH_COMM_ABI_MINOR
)


@dataclass(frozen=True)
class WorkerChipOrchRegionDesc:
    magic_version: int
    region_id: int
    payload_base: int
    payload_bytes: int
    counter_base: int
    counter_bytes: int

    def scalars(self) -> list[int]:
        return [
            int(self.magic_version),
            int(self.region_id),
            int(self.payload_base),
            int(self.payload_bytes),
            int(self.counter_base),
            int(self.counter_bytes),
        ]


def worker_chip_orch_region_desc_from_local_views(
    provider_resource_id: int, payload_view: RegionPartLocalView, counter_view: RegionPartLocalView
) -> WorkerChipOrchRegionDesc:
    payload_view, counter_view = validate_independent_local_views(payload_view, counter_view)
    return WorkerChipOrchRegionDesc(
        magic_version=_REGION_MAGIC_VERSION,
        region_id=int(provider_resource_id),
        payload_base=int(payload_view.local_base),
        payload_bytes=int(payload_view.logical_bytes),
        counter_base=int(counter_view.local_base),
        counter_bytes=int(counter_view.logical_bytes),
    )


class WorkerChipOrchCounter:
    def __init__(self, region: WorkerChipOrchRegion, offset: int) -> None:
        self._region = region
        self._offset = int(offset)

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def addr(self) -> int:
        return int(self._region.descriptor.counter_base) + self._offset

    def notify(self, value: int, op: NotifyOp = NotifyOp.Set) -> None:
        self._region._ensure_live()
        self._region._direct_counter_notify(self._offset, int(value), NotifyOp(op))

    def test(self, cmp_value: int, cmp: WaitCmp) -> SignalTestResult:
        self._region._ensure_live()
        return self._region._direct_counter_test(self._offset, int(cmp_value), WaitCmp(cmp))

    def wait(self, cmp_value: int, cmp: WaitCmp, timeout: float) -> int:
        self._region._ensure_live()
        if timeout is None or float(timeout) <= 0:
            raise ValueError("region counter wait requires a positive timeout")
        timeout_ns = min(int(float(timeout) * 1_000_000_000), _MAX_SIGNED_CHRONO_TIMEOUT_NS)
        return self._region._direct_counter_wait(self._offset, int(cmp_value), WaitCmp(cmp), timeout_ns)


class WorkerChipOrchRegion:
    def __init__(self, owner: Any, instance: Any, desc: WorkerChipOrchRegionDesc) -> None:
        self._owner = owner
        self._instance = instance
        self._worker_id = int(instance.worker_id)
        self._descriptor = desc
        self._released = False

    @property
    def descriptor(self) -> WorkerChipOrchRegionDesc:
        return self._descriptor

    @property
    def region_id(self) -> int:
        return int(self._descriptor.region_id)

    @property
    def expired(self) -> bool:
        try:
            return self._instance.state is not RegionInstanceState.LIVE
        except MaterializationError:
            return True

    def _validate_host_buffer(self, buffer: Any) -> None:
        self._owner._validate_worker_chip_orch_comm_host_buffer(buffer)

    def descriptor_scalars(self) -> list[int]:
        self._ensure_live()
        return self._descriptor.scalars()

    def payload_write(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        self._ensure_live()
        try:
            self._instance.payload_write(offset, host_buffer, nbytes)
        except Exception as exc:
            self._remember_data_plane_error(exc)
            raise

    def payload_read(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        self._ensure_live()
        try:
            self._instance.payload_read(offset, host_buffer, nbytes)
        except Exception as exc:
            self._remember_data_plane_error(exc)
            raise

    def counter(self, offset: int) -> WorkerChipOrchCounter:
        self._ensure_live()
        offset = int(offset)
        self._instance.counter(offset)
        return WorkerChipOrchCounter(self, offset)

    def free(self) -> None:
        if self._released:
            return
        self._released = True

    def _remember_data_plane_error(self, error: BaseException) -> None:
        if self._instance.data_plane_error is None:
            self._instance._data_plane_error = error

    def _ensure_live(self) -> None:
        if self.expired:
            raise RuntimeError(f"L3-L2 region {self.region_id} expired after orchestration run")
        if self._released:
            raise RuntimeError(f"L3-L2 region {self.region_id} has been released")
        if self._instance.data_plane_error is not None:
            raise RuntimeError(f"L3-L2 region {self.region_id} is poisoned")

    def _direct_counter_notify(self, offset: int, value: int, op: NotifyOp) -> None:
        try:
            self._instance.counter(offset).notify(int(value), NotifyOp(op))
        except Exception as exc:
            self._remember_data_plane_error(exc)
            raise

    def _direct_counter_test(self, offset: int, cmp_value: int, cmp: WaitCmp) -> SignalTestResult:
        try:
            return self._instance.counter(offset).test(int(cmp_value), WaitCmp(cmp))
        except Exception as exc:
            self._remember_data_plane_error(exc)
            raise

    def _direct_counter_wait(self, offset: int, cmp_value: int, cmp: WaitCmp, timeout_ns: int) -> int:
        timeout = float(timeout_ns) / 1_000_000_000
        try:
            return self._instance.counter(offset).wait(int(cmp_value), WaitCmp(cmp), timeout)
        except TimeoutError:
            raise
        except Exception as exc:
            self._remember_data_plane_error(exc)
            raise
