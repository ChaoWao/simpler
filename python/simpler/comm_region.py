# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Private region materialization helpers."""

from __future__ import annotations

import ctypes
import itertools
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any

from _task_interface import (  # pyright: ignore[reportMissingImports]
    _host_vmm_copy_from,
    _host_vmm_copy_to,
    _region_counter_notify,
    _region_counter_test,
    _region_counter_wait,
)

from .buffer import AddressSpace, Buffer
from .comm_endpoints import (
    DEVICE_AICPU,
    HOST_CPU,
    AdapterKind,
    AdapterProfile,
    AttachmentRole,
    BackendKind,
    BackendPlan,
    EndpointRecord,
    EndpointRegistry,
    MemberAttachmentPlan,
    RegionLayoutSpec,
    RegionPartPlan,
    SingleOwnerPlan,
    UnsupportedRegionPlan,
    parse_endpoint_path,
)

_GENERATION_COUNTER = itertools.count(1)
_MAX_SIGNED_CHRONO_TIMEOUT_NS = 2**63 - 1
_WAIT_STATUS_TIMEOUT = -1
_WAIT_ERROR_SIGNAL_TIMEOUT = 7


class NotifyOp(IntEnum):
    Set = 0
    Add = 1


class WaitCmp(IntEnum):
    EQ = 0
    NE = 1
    GT = 2
    GE = 3
    LT = 4
    LE = 5


@dataclass(frozen=True)
class SignalTestResult:
    matched: bool
    observed: int


@dataclass(frozen=True)
class RegionPartSpan:
    offset: int
    nbytes: int

    def validate_range(self, offset: int, nbytes: int, label: str) -> None:
        offset = int(offset)
        nbytes = int(nbytes)
        if offset < 0 or nbytes <= 0:
            raise ValueError(f"{label} offset must be non-negative and nbytes must be positive")
        if offset + nbytes > int(self.nbytes):
            raise ValueError(f"{label} range [{offset}, {offset + nbytes}) exceeds part size {int(self.nbytes)}")


class _PinnedBuffer:
    def __init__(self, obj: Any, *, writable: bool = False) -> None:
        self._keepalive: Any = obj
        if isinstance(obj, Buffer):
            if obj.address_space != AddressSpace.HOST:
                raise ValueError("region payload buffer must be host storage, not device storage")
            self.addr = int(obj.base)
            self.nbytes = int(obj.nbytes)
            return

        try:
            view = memoryview(obj)
        except TypeError as exc:
            raise ValueError("region payload buffer must be a contiguous host-accessible byte span") from exc
        if not view.c_contiguous:
            raise ValueError("region payload buffer must be contiguous")
        try:
            byte_view = view if view.itemsize == 1 and view.format in {"B", "b", "c"} else view.cast("B")
        except (TypeError, ValueError) as exc:
            raise ValueError("region payload buffer must be viewable as bytes") from exc
        if writable and byte_view.readonly:
            raise ValueError("region payload read destination must be writable")
        self.nbytes = int(byte_view.nbytes)
        if byte_view.readonly:
            staging = ctypes.create_string_buffer(byte_view.tobytes())
            self._keepalive = staging
            self.addr = ctypes.addressof(staging)
            return
        exported = ctypes.c_char.from_buffer(byte_view)
        self._keepalive = (byte_view, exported)
        self.addr = ctypes.addressof(exported)

    def close(self) -> None:
        self._keepalive = None

    def __enter__(self) -> _PinnedBuffer:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class HostVmmCopyAccess:
    def __init__(self, handle: Any) -> None:
        self._handle = int(handle)

    @classmethod
    def from_mapping(cls, mapping: Any) -> HostVmmCopyAccess:
        return cls(getattr(mapping, "handle"))

    @property
    def handle(self) -> int:
        return self._handle

    def write_bytes(self, span: RegionPartSpan, offset: int, host_ptr: int, nbytes: int) -> None:
        span.validate_range(offset, nbytes, "region byte write")
        _host_vmm_copy_to(self._handle, int(span.offset) + int(offset), int(host_ptr), int(nbytes))

    def read_bytes(self, span: RegionPartSpan, offset: int, host_ptr: int, nbytes: int) -> None:
        span.validate_range(offset, nbytes, "region byte read")
        _host_vmm_copy_from(self._handle, int(span.offset) + int(offset), int(host_ptr), int(nbytes))


class PayloadPart:
    def __init__(self, span: RegionPartSpan, access: Any) -> None:
        self._span = span
        self._access = access

    @property
    def span(self) -> RegionPartSpan:
        return self._span

    def write(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        with _PinnedBuffer(host_buffer) as pinned:
            size = pinned.nbytes if nbytes is None else int(nbytes)
            self._validate_payload_range(offset, size, pinned.nbytes)
            self._access.write_bytes(self._span, int(offset), pinned.addr, size)

    def read(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        with _PinnedBuffer(host_buffer, writable=True) as pinned:
            size = pinned.nbytes if nbytes is None else int(nbytes)
            self._validate_payload_range(offset, size, pinned.nbytes)
            self._access.read_bytes(self._span, int(offset), pinned.addr, size)

    def _validate_payload_range(self, offset: int, nbytes: int, buffer_nbytes: int) -> None:
        if int(nbytes) > int(buffer_nbytes):
            raise ValueError(f"region payload nbytes={int(nbytes)} exceeds host buffer size {int(buffer_nbytes)}")
        self._span.validate_range(offset, nbytes, "region payload")


class CounterPart:
    def __init__(self, span: RegionPartSpan, access: Any, handle: Any | None = None) -> None:
        self._span = span
        self._access = access
        resolved_handle = getattr(access, "handle", handle)
        if resolved_handle is None:
            raise ValueError("region counter access requires a native mapped-region handle")
        self._handle = int(resolved_handle)

    @property
    def span(self) -> RegionPartSpan:
        return self._span

    def counter(self, offset: int) -> RegionCounter:
        self._validate_counter_offset(offset)
        return RegionCounter(self, int(offset))

    def notify(self, offset: int, value: int, op: NotifyOp) -> None:
        self._validate_counter_offset(offset)
        op = NotifyOp(op)
        _region_counter_notify(self._handle, self._mapping_offset(offset), int(value), int(op))

    def test(self, offset: int, cmp_value: int, cmp: WaitCmp) -> SignalTestResult:
        self._validate_counter_offset(offset)
        cmp = WaitCmp(cmp)
        matched, observed = _region_counter_test(self._handle, self._mapping_offset(offset), int(cmp_value), int(cmp))
        return SignalTestResult(matched=bool(matched), observed=int(observed))

    def wait(self, offset: int, cmp_value: int, cmp: WaitCmp, timeout: float) -> int:
        self._validate_counter_offset(offset)
        cmp = WaitCmp(cmp)
        if timeout is None or float(timeout) <= 0:
            raise ValueError("region counter wait requires a positive timeout")
        timeout_ns = min(int(float(timeout) * 1_000_000_000), _MAX_SIGNED_CHRONO_TIMEOUT_NS)
        status, error_kind, observed, _matched, message = _region_counter_wait(
            self._handle, self._mapping_offset(offset), int(cmp_value), int(cmp), int(timeout_ns)
        )
        if int(status) == 0:
            return int(observed)
        msg = str(message) if message else "region counter wait timed out"
        if int(status) == _WAIT_STATUS_TIMEOUT and int(error_kind) == _WAIT_ERROR_SIGNAL_TIMEOUT:
            raise TimeoutError(f"{msg}; observed={int(observed)}")
        raise AssertionError(f"unexpected region counter wait result status={int(status)} error_kind={int(error_kind)}")

    def _mapping_offset(self, offset: int) -> int:
        return int(self._span.offset) + int(offset)

    def _validate_counter_offset(self, offset: int) -> None:
        offset = int(offset)
        if offset < 0 or offset % 4 != 0 or offset + 4 > int(self._span.nbytes):
            raise ValueError("region counter offset must be 4-byte aligned and inside the counter range")


class RegionCounter:
    def __init__(self, part: CounterPart, offset: int) -> None:
        self._part = part
        self._offset = int(offset)

    @property
    def offset(self) -> int:
        return self._offset

    def notify(self, value: int, op: NotifyOp = NotifyOp.Set) -> None:
        self._part.notify(self._offset, int(value), op)

    def test(self, cmp_value: int, cmp: WaitCmp) -> SignalTestResult:
        return self._part.test(self._offset, int(cmp_value), cmp)

    def wait(self, cmp_value: int, cmp: WaitCmp, timeout: float) -> int:
        return self._part.wait(self._offset, int(cmp_value), cmp, timeout)


class RegionInstanceState(str, Enum):
    PLANNED = "PLANNED"
    OWNER_CREATED = "OWNER_CREATED"
    CONSUMER_ATTACHED = "CONSUMER_ATTACHED"
    LIVE = "LIVE"
    ROLLING_BACK = "ROLLING_BACK"
    ROLLED_BACK = "ROLLED_BACK"
    ROLLBACK_FAILED = "ROLLBACK_FAILED"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    CLOSE_FAILED = "CLOSE_FAILED"


_TERMINAL_FAILURE_STATES = frozenset(
    (
        RegionInstanceState.ROLLBACK_FAILED,
        RegionInstanceState.CLOSE_FAILED,
    )
)


class RefusalReason(str, Enum):
    UNSUPPORTED_PLAN = "UNSUPPORTED_PLAN"
    NEEDS_DELEGATION = "NEEDS_DELEGATION"
    UNSUPPORTED_MEMBER_SHAPE = "UNSUPPORTED_MEMBER_SHAPE"
    UNSUPPORTED_PROVIDER_DEPLOYMENT = "UNSUPPORTED_PROVIDER_DEPLOYMENT"
    UNSUPPORTED_BACKEND_KIND = "UNSUPPORTED_BACKEND_KIND"
    UNSUPPORTED_ATTACHMENT = "UNSUPPORTED_ATTACHMENT"
    REGISTRY_MISMATCH = "REGISTRY_MISMATCH"


class MaterializationError(RuntimeError):
    pass


class MaterializationRefusal(MaterializationError):
    def __init__(self, reason: RefusalReason, message: str) -> None:
        self.reason = reason
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class MaterializationContext:
    worker: Any
    registry: EndpointRegistry
    plan: BackendPlan | UnsupportedRegionPlan
    layout: RegionLayoutSpec


@dataclass(frozen=True)
class SingleOwnerRegionShape:
    provider: EndpointRecord
    consumer: EndpointRecord
    worker_id: int


class RegionInstance:
    def __init__(self, ctx: MaterializationContext, shape: SingleOwnerRegionShape) -> None:
        self.plan = ctx.plan
        self.layout = ctx.layout
        self.provider = shape.provider
        self.consumer = shape.consumer
        self.worker_id = int(shape.worker_id)
        self.generation = next(_GENERATION_COUNTER)
        self.diagnostic_label = (
            f"{self.consumer.path} {self.consumer.deployment.value} -> "
            f"{self.provider.path} {self.provider.deployment.value} "
            f"payload={int(self.layout.payload_bytes)} counter={int(self.layout.counter_bytes)}"
        )
        self._worker = ctx.worker
        self._cleanup_resources = getattr(ctx.worker, "_building_run_resources", None)
        self._region = None
        self._payload_part: PayloadPart | None = None
        self._counter_part: CounterPart | None = None
        self._state = RegionInstanceState.PLANNED
        self._cleanup_error: BaseException | None = None

    @property
    def state(self) -> RegionInstanceState:
        return self._state

    @classmethod
    def planned(cls, ctx: MaterializationContext, shape: SingleOwnerRegionShape) -> RegionInstance:
        return cls(ctx, shape)

    def _adopt_worker_chip_region(self, region: Any) -> None:
        self._region = region
        mapping = getattr(region, "_worker_host_mapping", None)
        if mapping is None:
            return
        plan = self.plan
        if not isinstance(plan, BackendPlan):
            raise MaterializationRefusal(RefusalReason.UNSUPPORTED_PLAN, "materialized region requires a BackendPlan")
        payload_access = _select_host_vmm_copy_access(plan.payload, self.provider, self.consumer, mapping)
        counter_access = _select_host_vmm_copy_access(plan.counter, self.provider, self.consumer, mapping)
        self._payload_part = PayloadPart(
            RegionPartSpan(offset=int(mapping.payload_offset), nbytes=int(mapping.payload_bytes)), payload_access
        )
        self._counter_part = CounterPart(
            RegionPartSpan(offset=int(mapping.counter_offset), nbytes=int(mapping.counter_bytes)), counter_access
        )

    def payload_write(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        self._ensure_live()
        self._worker._require_region_control_context("region_instance.payload_write")
        payload_part = self._payload_part
        if payload_part is None:
            region = self._region
            assert region is not None
            region.payload_write(offset, host_buffer, nbytes)
            return
        payload_part.write(offset, host_buffer, nbytes)

    def payload_read(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        self._ensure_live()
        self._worker._require_region_control_context("region_instance.payload_read")
        payload_part = self._payload_part
        if payload_part is None:
            region = self._region
            assert region is not None
            region.payload_read(offset, host_buffer, nbytes)
            return
        payload_part.read(offset, host_buffer, nbytes)

    def counter(self, offset: int):
        self._ensure_live()
        self._worker._require_region_control_context("region_instance.counter")
        counter_part = self._counter_part
        if counter_part is None:
            region = self._region
            assert region is not None
            return region.counter(offset)
        return counter_part.counter(offset)

    def close(self) -> None:
        if self._state is RegionInstanceState.CLOSED:
            return
        if self._state in _TERMINAL_FAILURE_STATES and self._cleanup_error is not None:
            raise self._cleanup_error
        if self._state is RegionInstanceState.ROLLED_BACK:
            return
        if self._state is RegionInstanceState.PLANNED:
            self._state = RegionInstanceState.CLOSED
            return
        if self._region is None:
            self._state = RegionInstanceState.CLOSED
            return
        self._worker._require_region_control_context("region_instance.close")
        self._state = RegionInstanceState.CLOSING
        try:
            self._worker._close_worker_chip_region(
                self._region,
                self._cleanup_resources,
                poison_on_error=True,
            )
        except BaseException as exc:
            self._cleanup_error = exc
            self._state = RegionInstanceState.CLOSE_FAILED
            raise
        self._state = RegionInstanceState.CLOSED

    def rollback(self) -> None:
        if self._state in (RegionInstanceState.CLOSED, RegionInstanceState.ROLLED_BACK):
            return
        if self._state in _TERMINAL_FAILURE_STATES and self._cleanup_error is not None:
            raise self._cleanup_error
        if self._region is None:
            self._state = RegionInstanceState.ROLLED_BACK
            return
        self._worker._require_region_control_context("region_instance.rollback")
        self._state = RegionInstanceState.ROLLING_BACK
        try:
            self._worker._close_worker_chip_region(
                self._region,
                self._cleanup_resources,
                poison_on_error=True,
            )
        except BaseException as exc:
            self._cleanup_error = exc
            self._state = RegionInstanceState.ROLLBACK_FAILED
            raise
        self._state = RegionInstanceState.ROLLED_BACK

    def _rollback_after_failed_materialization(self) -> None:
        self.rollback()

    def _ensure_live(self) -> None:
        if self._state is not RegionInstanceState.LIVE:
            raise MaterializationError(f"region instance is not live: {self._state.value}")
        if self._region is None:
            raise MaterializationError("region instance has no adopted worker-chip region")


def validate_single_owner_region_shape(ctx: MaterializationContext) -> SingleOwnerRegionShape:  # noqa: PLR0912
    _validate_registry_matches_worker(ctx)
    plan = ctx.plan
    if isinstance(plan, UnsupportedRegionPlan):
        raise MaterializationRefusal(RefusalReason.UNSUPPORTED_PLAN, plan.message)
    if not isinstance(plan, BackendPlan):
        raise MaterializationRefusal(RefusalReason.UNSUPPORTED_PLAN, "materializer expects a BackendPlan")
    if int(getattr(ctx.worker, "level", -1)) != 3:
        raise MaterializationRefusal(
            RefusalReason.NEEDS_DELEGATION,
            "Only L3-local worker-chip regions can be materialized directly; higher-level roots require delegation",
        )
    if not isinstance(plan.topology_plan, SingleOwnerPlan):
        raise MaterializationRefusal(RefusalReason.UNSUPPORTED_PLAN, "Only SingleOwner region plans are supported")
    provider = _record_for(ctx, plan.topology_plan.provider_endpoint)
    if not provider.path.startswith("L3/"):
        raise MaterializationRefusal(RefusalReason.NEEDS_DELEGATION, "provider path requires delegated materialization")
    if provider.deployment is not DEVICE_AICPU:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_PROVIDER_DEPLOYMENT,
            "Only DEVICE_AICPU providers are supported for worker-chip regions",
        )
    try:
        provider_path = parse_endpoint_path(provider.path, root_level=ctx.registry.root_level)
    except ValueError as exc:
        raise MaterializationRefusal(RefusalReason.NEEDS_DELEGATION, "provider is not a local L3/L2 endpoint") from exc
    if len(provider_path.segments) != 2:
        raise MaterializationRefusal(RefusalReason.NEEDS_DELEGATION, "provider is not a local L3/L2 endpoint")
    _root, child = provider_path.segments
    if child.level != 2 or child.index is None:
        raise MaterializationRefusal(RefusalReason.NEEDS_DELEGATION, "provider is not a local L3/L2 endpoint")
    worker_id = int(child.index)
    try:
        ctx.worker._validate_worker_chip_id(worker_id)
    except ValueError as exc:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            f"provider worker_id {worker_id} is outside the current L3 device list",
        ) from exc
    member_records = tuple(_record_for(ctx, member) for member in plan.ordered_members)
    if len(member_records) != 2:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            "Only one host consumer and one device provider are supported",
        )
    host_consumers = [member for member in member_records if member.deployment is HOST_CPU]
    if len(host_consumers) != 1 or host_consumers[0].path != "L3":
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            "The current L3 HOST_CPU endpoint must be the only consumer",
        )
    consumer = host_consumers[0]
    if provider.identity not in plan.ordered_members or consumer.identity not in plan.ordered_members:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            "Ordered members must contain the provider and consumer endpoints",
        )
    if not ctx.registry.same_node(provider, consumer):
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            "Worker-chip region materialization only supports local endpoints",
        )
    _validate_part(plan.payload, provider, consumer)
    _validate_part(plan.counter, provider, consumer)
    return SingleOwnerRegionShape(provider=provider, consumer=consumer, worker_id=worker_id)


def materialize_region_instance(ctx: MaterializationContext) -> RegionInstance:
    shape = validate_single_owner_region_shape(ctx)
    instance = RegionInstance.planned(ctx, shape)
    try:
        instance._state = RegionInstanceState.OWNER_CREATED
        region = ctx.worker._create_worker_chip_region(
            shape.worker_id,
            int(ctx.layout.payload_bytes),
            int(ctx.layout.counter_bytes),
        )
        instance._adopt_worker_chip_region(region)
        instance._state = RegionInstanceState.LIVE
        return instance
    except BaseException:
        instance._rollback_after_failed_materialization()
        raise


def _record_for(ctx: MaterializationContext, endpoint: Any) -> EndpointRecord:
    try:
        return ctx.registry.record_for(endpoint)
    except ValueError as exc:
        raise MaterializationRefusal(RefusalReason.REGISTRY_MISMATCH, str(exc)) from exc


def _select_host_vmm_copy_access(
    part: RegionPartPlan,
    provider: EndpointRecord,
    consumer: EndpointRecord,
    mapping: Any,
) -> HostVmmCopyAccess:
    _validate_part(part, provider, consumer)
    return HostVmmCopyAccess.from_mapping(mapping)


def _validate_registry_matches_worker(ctx: MaterializationContext) -> None:
    worker_instance_id = getattr(ctx.worker, "_owner_instance_id", None)
    worker_epoch = getattr(ctx.worker, "_endpoint_registry_epoch", None)
    if worker_instance_id != ctx.registry.session_instance_id or worker_epoch != ctx.registry.registry_epoch:
        raise MaterializationRefusal(
            RefusalReason.REGISTRY_MISMATCH,
            "Region materialization requires a registry from the current worker endpoint epoch",
        )


def _validate_part(part: RegionPartPlan, provider: EndpointRecord, consumer: EndpointRecord) -> None:
    if part.backend_kind is not BackendKind.VMM_WINDOW:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_BACKEND_KIND,
            "Only VMM_WINDOW-backed worker-chip region parts are supported",
        )
    attachments = {attachment.member: attachment for attachment in part.attachments}
    if len(attachments) != len(part.attachments) or set(attachments) != {provider.identity, consumer.identity}:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_ATTACHMENT,
            "Part attachments must match exactly the provider and host consumer",
        )
    _validate_provider_attachment(attachments[provider.identity])
    _validate_consumer_attachment(attachments[consumer.identity])


def _validate_provider_attachment(attachment: MemberAttachmentPlan) -> None:
    if (
        attachment.role is not AttachmentRole.PROVIDER
        or attachment.adapter_kind is not None
        or attachment.adapter_profile is not None
    ):
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_ATTACHMENT,
            "Provider attachment must be a bare PROVIDER attachment",
        )


def _validate_consumer_attachment(attachment: MemberAttachmentPlan) -> None:
    if (
        attachment.role is not AttachmentRole.CONSUMER
        or attachment.adapter_kind is not AdapterKind.OWNER_DELEGATED_COPY
        or attachment.adapter_profile is not AdapterProfile.HOST_VMM_COPY
    ):
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_ATTACHMENT,
            "Host consumer attachment must use OWNER_DELEGATED_COPY/HOST_VMM_COPY",
        )
