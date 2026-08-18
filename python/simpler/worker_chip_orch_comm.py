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

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from _task_interface import (  # pyright: ignore[reportMissingImports]
    _worker_host_mapped_counter_notify,  # noqa: F401
    _worker_host_mapped_counter_test,  # noqa: F401
    _worker_host_mapped_counter_wait,  # noqa: F401
    _worker_host_mapped_payload_read,  # noqa: F401
    _worker_host_mapped_payload_write,  # noqa: F401
    _worker_host_mapped_region_close,
)

from .comm_region import (
    CounterPart,
    HostVmmCopyAccess,
    NotifyOp,
    PayloadPart,
    RegionPartSpan,
    SignalTestResult,
    WaitCmp,
)


class WorkerChipRegionAccessProfile(IntEnum):
    INVALID = 0
    ONBOARD_VMM = 1
    SIM_POSIX_SHM = 2


_DESC = struct.Struct("<6Q")
_WORKER_CHIP_ORCH_REGION_DESC_SCALAR_COUNT = 6
_CTRL_SHM_TOKEN_BYTES = 32
_REGION_CREATE_REQUEST = struct.Struct("<QQQQ")
_REGION_CREATE_REPLY = struct.Struct(f"<6QIIi{_CTRL_SHM_TOKEN_BYTES}s4xQQ")
_REGION_CREATE_REQUEST_BYTES = _REGION_CREATE_REQUEST.size
_REGION_CREATE_REPLY_BYTES = _REGION_CREATE_REPLY.size
_REGION_LAYOUT_ALIGNMENT = 64
_UINT64_MAX = (1 << 64) - 1
_MAX_SIGNED_CHRONO_TIMEOUT_NS = 2**63 - 1
_REGION_MAGIC_VERSION = 0x4C334C3200020000


def _align_up(value: int, align: int) -> int:
    value = int(value)
    if value < 0 or value > _UINT64_MAX:
        raise ValueError("L3-L2 region layout overflowed uint64")
    remainder = value % align
    bump = 0 if remainder == 0 else align - remainder
    result = value + bump
    if result > _UINT64_MAX:
        raise ValueError("L3-L2 region layout overflowed uint64")
    return result


def _checked_add_u64(lhs: int, rhs: int) -> int:
    result = int(lhs) + int(rhs)
    if int(lhs) < 0 or int(rhs) < 0 or result > _UINT64_MAX:
        raise ValueError("L3-L2 region layout overflowed uint64")
    return result


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


@dataclass(frozen=True)
class WorkerChipRegionCreateRequest:
    magic_version: int
    request_bytes: int
    payload_bytes: int
    counter_bytes: int

    def encode_into(self, buf: memoryview, offset: int = 0) -> None:
        _REGION_CREATE_REQUEST.pack_into(
            buf,
            offset,
            int(self.magic_version),
            int(self.request_bytes),
            int(self.payload_bytes),
            int(self.counter_bytes),
        )


@dataclass(frozen=True)
class WorkerChipRegionCreateReply:
    desc: WorkerChipOrchRegionDesc
    access_profile: WorkerChipRegionAccessProfile
    device_id: int
    backing_shm: str
    mapping_bytes: int
    shareable_handle: int


@dataclass
class WorkerHostRegionMapping:
    worker_id: int
    region_id: int
    access_profile: WorkerChipRegionAccessProfile
    total_bytes: int
    payload_offset: int
    payload_bytes: int
    counter_offset: int
    counter_bytes: int
    handle: Any
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        _worker_host_mapped_region_close(int(self.handle))
        self.closed = True


def decode_region_create_reply(buf: memoryview) -> WorkerChipRegionCreateReply:
    fields = _REGION_CREATE_REPLY.unpack_from(buf, 0)
    desc = WorkerChipOrchRegionDesc(*[int(v) for v in fields[:6]])
    access_profile = WorkerChipRegionAccessProfile(int(fields[6]))
    device_id = int(fields[8])
    backing_shm = bytes(fields[9]).split(b"\x00", 1)[0].decode("utf-8", "strict")
    return WorkerChipRegionCreateReply(
        desc=desc,
        access_profile=access_profile,
        device_id=device_id,
        backing_shm=backing_shm,
        mapping_bytes=int(fields[10]),
        shareable_handle=int(fields[11]),
    )


def peek_region_create_reply_region_id(buf: memoryview) -> int:
    """Raw-unpack just the region_id for the create rollback path.

    Must tolerate malformed replies: decode_region_create_reply raises on
    unknown access_profile / non-UTF-8 backing_shm, but the L2 child has
    already created the region by then, so the rollback release still needs
    the id. Call this BEFORE decode_region_create_reply.
    """
    return int(_REGION_CREATE_REPLY.unpack_from(buf, 0)[1])


def validate_region_create_reply(
    reply: WorkerChipRegionCreateReply, expected_access_profile: WorkerChipRegionAccessProfile
) -> tuple[int, int]:
    desc = reply.desc
    if desc.magic_version != _REGION_MAGIC_VERSION:
        raise RuntimeError("create_worker_chip_region: reply magic_version is invalid")
    if desc.region_id == 0:
        raise RuntimeError("create_worker_chip_region: reply region_id must be nonzero")
    if reply.access_profile != expected_access_profile:
        raise RuntimeError(
            f"create_worker_chip_region: reply access_profile must be {expected_access_profile.name.lower()}"
        )
    if desc.payload_bytes <= 0:
        raise RuntimeError("create_worker_chip_region: reply payload_bytes must be positive")
    if desc.counter_bytes <= 0 or desc.counter_bytes % 4 != 0:
        raise RuntimeError("create_worker_chip_region: reply counter_bytes must be positive and a multiple of 4")
    counter_offset = _align_up(desc.payload_bytes, _REGION_LAYOUT_ALIGNMENT)
    total_bytes = _checked_add_u64(counter_offset, desc.counter_bytes)
    expected_counter_base = _checked_add_u64(desc.payload_base, counter_offset)
    if desc.counter_base != expected_counter_base:
        raise RuntimeError("create_worker_chip_region: reply counter_base does not match fixed region layout")
    if desc.counter_base % _REGION_LAYOUT_ALIGNMENT != 0:
        raise RuntimeError("create_worker_chip_region: reply counter_base must be 64-byte aligned")
    if reply.access_profile == WorkerChipRegionAccessProfile.SIM_POSIX_SHM and reply.mapping_bytes != total_bytes:
        profile = reply.access_profile.name.lower()
        raise RuntimeError(f"create_worker_chip_region: {profile} reply mapping_bytes does not match descriptor layout")
    if reply.access_profile == WorkerChipRegionAccessProfile.ONBOARD_VMM:
        if reply.mapping_bytes < total_bytes:
            raise RuntimeError(
                "create_worker_chip_region: onboard_vmm reply mapping_bytes is smaller than descriptor layout"
            )
    return counter_offset, total_bytes


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
    def __init__(
        self,
        owner: Any,
        worker_id: int,
        desc: WorkerChipOrchRegionDesc,
        worker_host_mapping: WorkerHostRegionMapping,
    ) -> None:
        self._owner = owner
        self._worker_id = int(worker_id)
        self._descriptor = desc
        self._worker_host_mapping = worker_host_mapping
        access = HostVmmCopyAccess.from_mapping(worker_host_mapping)
        self._payload_part = PayloadPart(
            RegionPartSpan(
                offset=int(worker_host_mapping.payload_offset), nbytes=int(worker_host_mapping.payload_bytes)
            ),
            access,
        )
        self._counter_part = CounterPart(
            RegionPartSpan(
                offset=int(worker_host_mapping.counter_offset), nbytes=int(worker_host_mapping.counter_bytes)
            ),
            access,
        )
        self._released = False
        self._chip_release_committed = False
        self._poisoned = False
        self._expired = False

    @property
    def descriptor(self) -> WorkerChipOrchRegionDesc:
        return self._descriptor

    @property
    def region_id(self) -> int:
        return int(self._descriptor.region_id)

    @property
    def expired(self) -> bool:
        return self._expired

    def _validate_host_buffer(self, buffer: Any) -> None:
        self._owner._validate_worker_chip_orch_comm_host_buffer(buffer)

    def descriptor_scalars(self) -> list[int]:
        self._ensure_live()
        return self._descriptor.scalars()

    def payload_write(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        self._ensure_live()
        try:
            self._payload_part.write(offset, host_buffer, nbytes)
        except Exception:
            self._poison()
            raise

    def payload_read(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        self._ensure_live()
        try:
            self._payload_part.read(offset, host_buffer, nbytes)
        except Exception:
            self._poison()
            raise

    def counter(self, offset: int) -> WorkerChipOrchCounter:
        self._ensure_live()
        offset = int(offset)
        self._counter_part.counter(offset)
        return WorkerChipOrchCounter(self, offset)

    def free(self) -> None:
        if self._released:
            return
        self._released = True

    def _expire(self) -> None:
        self._expired = True

    def _poison(self) -> None:
        self._poisoned = True

    def _ensure_live(self) -> None:
        if self._expired:
            raise RuntimeError(f"L3-L2 region {self.region_id} expired after orchestration run")
        if self._released:
            raise RuntimeError(f"L3-L2 region {self.region_id} has been released")
        if self._poisoned:
            raise RuntimeError(f"L3-L2 region {self.region_id} is poisoned")

    def _close_worker_host_mapping(self) -> None:
        try:
            self._worker_host_mapping.close()
        except Exception:
            self._poison()
            raise

    def _direct_counter_notify(self, offset: int, value: int, op: NotifyOp) -> None:
        try:
            self._counter_part.notify(offset, int(value), NotifyOp(op))
        except Exception:
            self._poison()
            raise

    def _direct_counter_test(self, offset: int, cmp_value: int, cmp: WaitCmp) -> SignalTestResult:
        try:
            return self._counter_part.test(offset, int(cmp_value), WaitCmp(cmp))
        except Exception:
            self._poison()
            raise

    def _direct_counter_wait(self, offset: int, cmp_value: int, cmp: WaitCmp, timeout_ns: int) -> int:
        timeout = float(timeout_ns) / 1_000_000_000
        try:
            return self._counter_part.wait(offset, int(cmp_value), WaitCmp(cmp), timeout)
        except TimeoutError:
            raise
        except Exception:
            self._poison()
            raise
