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
    _worker_host_mapped_region_close,
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
from .comm_provider import (
    PosixShmImport,
    ProviderReleaseStatus,
    RegionAllocationError,
    RegionAllocationResult,
    RegionAllocationSpec,
    RegionPartAllocationSpec,
    RegionPartKind,
    RegionPartLocalView,
    VmmShareableHandleImport,
    validate_independent_local_views,
)
from .comm_provider_control import ProviderAllocateClient, ProviderReleaseClient

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
        self._handle = handle

    @classmethod
    def from_mapping(cls, mapping: Any) -> HostVmmCopyAccess:
        return cls(getattr(mapping, "handle", mapping))

    @property
    def handle(self) -> Any:
        return self._handle

    def write_bytes(self, span: RegionPartSpan, offset: int, host_ptr: int, nbytes: int) -> None:
        span.validate_range(offset, nbytes, "region byte write")
        _host_vmm_copy_to(int(self._handle), int(span.offset) + int(offset), int(host_ptr), int(nbytes))

    def read_bytes(self, span: RegionPartSpan, offset: int, host_ptr: int, nbytes: int) -> None:
        span.validate_range(offset, nbytes, "region byte read")
        _host_vmm_copy_from(int(self._handle), int(span.offset) + int(offset), int(host_ptr), int(nbytes))


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
    LIVE = "LIVE"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    CLOSE_FAILED = "CLOSE_FAILED"


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


class RegionInstanceRegistry:
    """Strong reachability for consumer instances. Not an ID-addressed data path."""

    def __init__(self) -> None:
        self._instances: dict[int, RegionInstance] = {}
        self._run_scopes: dict[int, Any] = {}

    def track(self, instance: RegionInstance, run_scope: Any) -> None:
        key = id(instance)
        if key in self._instances:
            raise MaterializationError("region instance is already tracked")
        self._instances[key] = instance
        self._run_scopes[key] = run_scope

    def close(self, instance: RegionInstance) -> None:
        try:
            instance._close_owned(poison_on_error=True)
        finally:
            self._settle(instance)

    def cleanup_run(self, run_scope: Any) -> None:
        errors: list[BaseException] = []
        for instance in self._iter_run(run_scope):
            if instance._close_attempted or instance._state in (
                RegionInstanceState.CLOSED,
                RegionInstanceState.CLOSE_FAILED,
            ):
                self._settle(instance)
                continue
            try:
                self.close(instance)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise errors[0]

    def sweep(self) -> None:
        errors: list[BaseException] = []
        for instance in tuple(self._instances.values()):
            if not instance._close_attempted and instance._state not in (
                RegionInstanceState.CLOSED,
                RegionInstanceState.CLOSE_FAILED,
            ):
                try:
                    instance._close_owned(poison_on_error=True)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)
            self._retire(id(instance))
        if errors:
            raise errors[0]

    def record_data_plane_failure(self, run_scope: Any, resource_id: int, error: BaseException) -> None:
        resource_id = int(resource_id)
        matches = [
            instance for instance in self._iter_run(run_scope) if int(instance.provider_resource_id) == resource_id
        ]
        if not matches:
            raise MaterializationError(f"no region instance for resource {resource_id}")
        if len(matches) > 1:
            raise MaterializationError(f"duplicate region instances for resource {resource_id}")
        instance = matches[0]
        if instance._data_plane_error is None:
            instance._data_plane_error = error

    def _iter_run(self, run_scope: Any) -> tuple[RegionInstance, ...]:
        return tuple(instance for key, instance in self._instances.items() if self._run_scopes[key] is run_scope)

    def _settle(self, instance: RegionInstance) -> None:
        if id(instance) not in self._instances:
            return
        if instance._retains_cleanup_only_reachability():
            return
        self._retire(id(instance))

    def _retire(self, key: int) -> None:
        self._instances.pop(key, None)
        self._run_scopes.pop(key, None)


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
        self._payload_mapping: Any | None = None
        self._counter_mapping: Any | None = None
        self._payload_part: PayloadPart | None = None
        self._counter_part: CounterPart | None = None
        self._payload_local_view: RegionPartLocalView | None = None
        self._counter_local_view: RegionPartLocalView | None = None
        self._provider_resource_id = 0
        self._release_client: ProviderReleaseClient | None = None
        self._allocate_client: ProviderAllocateClient | None = None
        self._state: RegionInstanceState | None = None
        self._cleanup_error: BaseException | None = None
        self._close_attempted = False
        self._ever_live = False
        self._provider_release_committed = False
        self._data_plane_error: BaseException | None = None

    @property
    def state(self) -> RegionInstanceState:
        if self._state is None:
            raise MaterializationError("region instance is not published")
        return self._state

    @property
    def provider_resource_id(self) -> int:
        return int(self._provider_resource_id)

    @property
    def data_plane_error(self) -> BaseException | None:
        return self._data_plane_error

    def local_view(self, part: RegionPartKind) -> RegionPartLocalView | None:
        if self._state is not RegionInstanceState.LIVE:
            return None
        selected = RegionPartKind(part)
        if selected is RegionPartKind.PAYLOAD:
            return self._payload_local_view
        if selected is RegionPartKind.COUNTER:
            return self._counter_local_view
        raise ValueError("region part must be PAYLOAD or COUNTER")

    @classmethod
    def planned(cls, ctx: MaterializationContext, shape: SingleOwnerRegionShape) -> RegionInstance:
        return cls(ctx, shape)

    def payload_write(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        self._ensure_live()
        self._worker._require_region_control_context("region_instance.payload_write")
        payload_part = self._payload_part
        assert payload_part is not None
        payload_part.write(offset, host_buffer, nbytes)

    def payload_read(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        self._ensure_live()
        self._worker._require_region_control_context("region_instance.payload_read")
        payload_part = self._payload_part
        assert payload_part is not None
        payload_part.read(offset, host_buffer, nbytes)

    def counter(self, offset: int):
        self._ensure_live()
        self._worker._require_region_control_context("region_instance.counter")
        counter_part = self._counter_part
        assert counter_part is not None
        return counter_part.counter(offset)

    def close(self) -> None:
        if self._state is RegionInstanceState.CLOSED:
            return
        if self._state is RegionInstanceState.CLOSE_FAILED and self._cleanup_error is not None:
            raise self._cleanup_error
        if self._state is None and self._provider_resource_id == 0 and self._payload_mapping is None:
            self._state = RegionInstanceState.CLOSED
            self._worker._region_instance_registry._settle(self)
            return
        self._worker._require_region_control_before_submit("region_instance.close")
        self._worker._region_instance_registry.close(self)

    def _materialize(self, spec: RegionAllocationSpec) -> None:
        plan = self.plan
        if not isinstance(plan, BackendPlan):
            raise MaterializationRefusal(RefusalReason.UNSUPPORTED_PLAN, "materialized region requires a BackendPlan")
        prior = self._worker._consume_worker_host_mapped_cleanup_error("region_instance.materialize")
        if prior is not None:
            raise prior
        native = self._worker._worker
        if native is None:
            raise RuntimeError("region instance materialize requires Worker.init()")
        self._allocate_client = ProviderAllocateClient(native, self.worker_id)
        try:
            result, payload_view, counter_view = self._allocate_client.allocate(spec)
        except RegionAllocationError as exc:
            self._provider_resource_id = int(exc.provisional_resource_id)
            if exc.cleanup_debt_remaining:
                error = self._worker._record_unreclaimable(
                    f"region instance: allocation left cleanup debt for resource "
                    f"{exc.provisional_resource_id} on worker {self.worker_id}; no further work is admitted",
                    exc,
                )
                self._fail_terminal(error)
                raise error
            self._state = RegionInstanceState.CLOSED
            raise
        except BaseException:
            committed = int(self._allocate_client.committed_resource_id)
            if committed:
                self._provider_resource_id = committed
                self._release_client = ProviderReleaseClient(native, self.worker_id)
            raise
        self._provider_resource_id = int(result.provider_resource_id)
        self._release_client = ProviderReleaseClient(native, self.worker_id)
        try:
            validate_committed_region_allocation(
                plan,
                spec,
                result,
                payload_view,
                counter_view,
                expected_capability_type=self._worker._provider_import_capability_type(),
                expected_device_id=self._worker._provider_import_device_id(self.worker_id),
            )
            payload_lease = self._worker._import_region_part_lease(
                self.worker_id, self._provider_resource_id, result.export_descriptor.payload
            )
            self._payload_mapping = payload_lease
            counter_lease = self._worker._import_region_part_lease(
                self.worker_id, self._provider_resource_id, result.export_descriptor.counter
            )
            self._counter_mapping = counter_lease
            self._payload_part = PayloadPart(
                RegionPartSpan(offset=0, nbytes=int(spec.payload.logical_bytes)),
                _select_host_vmm_copy_access(plan.payload, self.provider, self.consumer, payload_lease),
            )
            self._counter_part = CounterPart(
                RegionPartSpan(offset=0, nbytes=int(spec.counter.logical_bytes)),
                _select_host_vmm_copy_access(plan.counter, self.provider, self.consumer, counter_lease),
            )
            self._payload_local_view = payload_view
            self._counter_local_view = counter_view
        except BaseException as exc:
            self._abort_materialization(exc)
            raise
        self._ever_live = True
        self._state = RegionInstanceState.LIVE

    def _abort_materialization(self, cause: BaseException) -> None:
        if isinstance(cause, RegionAllocationError):
            poison = bool(cause.cleanup_debt_remaining)
        else:
            client = self._allocate_client
            poison = bool(client is not None and client.dispatch_started)
        close_error: BaseException | None = None
        try:
            self._close_owned(poison_on_error=False)
        except BaseException as exc:  # noqa: BLE001
            close_error = exc
        if not poison:
            if close_error is None and self._state is not RegionInstanceState.CLOSE_FAILED:
                self._state = RegionInstanceState.CLOSED
            elif close_error is not None:
                self._cleanup_error = close_error
                self._state = RegionInstanceState.CLOSE_FAILED
            return
        if close_error is not None and close_error.__cause__ is None:
            close_error.__cause__ = cause
        self._cleanup_error = self._worker._record_unreclaimable(
            f"region instance: committed allocation could not be published on worker {self.worker_id}; "
            "no further work is admitted",
            close_error or cause,
        )
        self._state = RegionInstanceState.CLOSE_FAILED

    def _close_mapping_leases(self) -> list[BaseException]:
        errors: list[BaseException] = []
        for lease in (self._payload_mapping, self._counter_mapping):
            if lease is None:
                continue
            try:
                closer = getattr(lease, "close", None)
                if closer is not None:
                    closer()
                else:
                    _worker_host_mapped_region_close(int(lease))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        return errors

    def _release_provider_resource(self) -> BaseException | None:
        if self._release_client is None or int(self._provider_resource_id) == 0:
            return None
        try:
            result = self._release_client.release(int(self._provider_resource_id))
        except BaseException as exc:  # noqa: BLE001
            return exc
        if result.status in (ProviderReleaseStatus.RELEASED, ProviderReleaseStatus.ALREADY_GONE):
            self._provider_release_committed = True
            return None
        if result.status is ProviderReleaseStatus.CLEANUP_INCOMPLETE:
            return RuntimeError(
                f"region instance: provider cleanup incomplete for resource {self._provider_resource_id}"
            )
        if result.status is ProviderReleaseStatus.UNKNOWN_RESOURCE:
            return RuntimeError(f"region instance: provider resource {self._provider_resource_id} is unknown")
        return None

    def _close_owned(self, *, poison_on_error: bool) -> None:
        if self._state is RegionInstanceState.CLOSED:
            return
        if self._state is RegionInstanceState.CLOSE_FAILED and self._cleanup_error is not None:
            raise self._cleanup_error
        if self._close_attempted:
            if self._cleanup_error is not None:
                raise self._cleanup_error
            return
        self._close_attempted = True
        if self._state is RegionInstanceState.LIVE:
            self._state = RegionInstanceState.CLOSING
        errors = self._close_mapping_leases()
        release_error = self._release_provider_resource()
        if release_error is not None:
            errors.append(release_error)
        if not errors:
            self._state = RegionInstanceState.CLOSED
            return
        self._cleanup_error = errors[0]
        self._state = RegionInstanceState.CLOSE_FAILED
        if poison_on_error:
            self._worker._record_unreclaimable(
                f"region instance: resource {self._provider_resource_id} on worker {self.worker_id} "
                "could not be fully reclaimed; no further work is admitted",
                self._cleanup_error,
            )
        raise self._cleanup_error

    def _fail_terminal(self, error: BaseException) -> None:
        self._cleanup_error = error
        self._state = RegionInstanceState.CLOSE_FAILED

    def _retains_cleanup_only_reachability(self) -> bool:
        if self._state is RegionInstanceState.CLOSED:
            return False
        if self._state is RegionInstanceState.CLOSE_FAILED:
            return not (self._ever_live and self._provider_release_committed)
        return True

    def _ensure_live(self) -> None:
        if self._state is not RegionInstanceState.LIVE:
            label = "unpublished" if self._state is None else self._state.value
            raise MaterializationError(f"region instance is not live: {label}")


def project_region_allocation_spec(plan: object, layout: object) -> RegionAllocationSpec:
    if not isinstance(plan, BackendPlan):
        raise TypeError("project_region_allocation_spec requires a BackendPlan")
    if not isinstance(layout, RegionLayoutSpec):
        raise TypeError("project_region_allocation_spec requires a RegionLayoutSpec")
    return RegionAllocationSpec(
        payload=RegionPartAllocationSpec(
            planned_backing_kind=plan.payload.backend_kind,
            logical_bytes=int(layout.payload_bytes),
        ),
        counter=RegionPartAllocationSpec(
            planned_backing_kind=plan.counter.backend_kind,
            logical_bytes=int(layout.counter_bytes),
        ),
    )


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
    spec = project_region_allocation_spec(ctx.plan, ctx.layout)
    instance = RegionInstance.planned(ctx, shape)
    resources = getattr(ctx.worker, "_building_run_resources", None)
    ctx.worker._region_instance_registry.track(instance, resources)
    if resources is not None:
        resources.requires_ordered_cleanup = True
    try:
        instance._materialize(spec)
        return instance
    except BaseException as exc:
        if instance._state is None:
            instance._abort_materialization(exc)
        ctx.worker._region_instance_registry._settle(instance)
        raise


def validate_committed_region_allocation(  # noqa: PLR0912
    plan: BackendPlan,
    spec: RegionAllocationSpec,
    result: RegionAllocationResult,
    payload_view: RegionPartLocalView,
    counter_view: RegionPartLocalView,
    *,
    expected_capability_type: type,
    expected_device_id: int | None,
) -> None:
    payload_export = result.export_descriptor.payload
    counter_export = result.export_descriptor.counter
    if payload_export.planned_backing_kind is not plan.payload.backend_kind:
        raise RuntimeError("committed PAYLOAD planned backing does not match the admitted plan")
    if counter_export.planned_backing_kind is not plan.counter.backend_kind:
        raise RuntimeError("committed COUNTER planned backing does not match the admitted plan")
    if payload_export.planned_backing_kind is not spec.payload.planned_backing_kind:
        raise RuntimeError("committed PAYLOAD planned backing does not match the admitted spec")
    if counter_export.planned_backing_kind is not spec.counter.planned_backing_kind:
        raise RuntimeError("committed COUNTER planned backing does not match the admitted spec")
    if int(payload_export.logical_bytes) != int(spec.payload.logical_bytes):
        raise RuntimeError("committed PAYLOAD logical_bytes do not match the admitted spec")
    if int(counter_export.logical_bytes) != int(spec.counter.logical_bytes):
        raise RuntimeError("committed COUNTER logical_bytes do not match the admitted spec")
    if not isinstance(payload_export.import_capability, expected_capability_type) or not isinstance(
        counter_export.import_capability, expected_capability_type
    ):
        raise RuntimeError("committed import capability does not match the current execution environment")
    if expected_capability_type is VmmShareableHandleImport:
        payload_cap = payload_export.import_capability
        counter_cap = counter_export.import_capability
        assert isinstance(payload_cap, VmmShareableHandleImport)
        assert isinstance(counter_cap, VmmShareableHandleImport)
        if int(payload_cap.shareable_handle) == int(counter_cap.shareable_handle):
            raise RuntimeError("committed VMM shareable handles must be distinct")
        if expected_device_id is not None and (
            int(payload_cap.device_id) != int(expected_device_id)
            or int(counter_cap.device_id) != int(expected_device_id)
        ):
            raise RuntimeError("committed VMM device_id is outside this worker's device namespace")
    elif expected_capability_type is PosixShmImport:
        payload_cap = payload_export.import_capability
        counter_cap = counter_export.import_capability
        assert isinstance(payload_cap, PosixShmImport)
        assert isinstance(counter_cap, PosixShmImport)
        if payload_cap.shm_name == counter_cap.shm_name:
            raise RuntimeError("committed POSIX shm tokens must be distinct")
    if payload_view.part is not RegionPartKind.PAYLOAD or counter_view.part is not RegionPartKind.COUNTER:
        raise RuntimeError("committed local views must be PAYLOAD then COUNTER")
    if int(payload_view.logical_bytes) != int(spec.payload.logical_bytes):
        raise RuntimeError("committed PAYLOAD local view bytes do not match the admitted spec")
    if int(counter_view.logical_bytes) != int(spec.counter.logical_bytes):
        raise RuntimeError("committed COUNTER local view bytes do not match the admitted spec")
    try:
        validate_independent_local_views(payload_view, counter_view)
    except ValueError as exc:
        raise RuntimeError(str(exc) or "committed local views are not independent") from exc


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
