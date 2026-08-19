# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Neutral provider-region values, typed codes, and structured results.

This module is the provider-agent value surface for CPU-NPU comm regions. It
does not import worker-chip compatibility types, Worker, mailbox transport, or
W5a transaction identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Union

from _task_interface import BackendKind  # pyright: ignore[reportMissingImports]

_UINT64_MAX = (1 << 64) - 1
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_COUNTER_LOGICAL_ALIGNMENT = 4
_COUNTER_BASE_ALIGNMENT = 64
POSIX_SHM_TOKEN_MAX_BYTES = 32


class RegionPartKind(IntEnum):
    INVALID = 0
    PAYLOAD = 1
    COUNTER = 2


class RegionEnvironmentKind(str, Enum):
    SIM = "SIM"
    ONBOARD = "ONBOARD"


class ProviderRegionStoreState(str, Enum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSE_FAILED = "CLOSE_FAILED"
    CLOSED = "CLOSED"


class ProviderRegionResourceState(str, Enum):
    CREATING = "CREATING"
    ACTIVE = "ACTIVE"
    CLEANUP_PENDING = "CLEANUP_PENDING"


class ProviderPartResourceState(str, Enum):
    SHELL = "SHELL"
    MATERIALIZING = "MATERIALIZING"
    READY = "READY"
    CLEANUP_PENDING = "CLEANUP_PENDING"
    RELEASED = "RELEASED"


class ProviderReleaseStatus(IntEnum):
    RELEASED = 1
    ALREADY_GONE = 2
    UNKNOWN_RESOURCE = 3
    CLEANUP_INCOMPLETE = 4


class RegionControlErrorKind(IntEnum):
    NONE = 0
    BAD_MAGIC_VERSION = 1
    BAD_MESSAGE_SIZE = 2
    INVALID_ENUM_VALUE = 3
    RESERVED_NONZERO = 4
    INVALID_FIELD_VALUE = 5
    STORE_LIFECYCLE = 6
    INTERNAL_INVARIANT = 7
    BACKEND_FAILURE = 8


class RegionOperationKind(IntEnum):
    NONE = 0
    MATERIALIZE = 1
    ZERO_BYTES = 2
    DESCRIBE = 3
    LOCAL_VIEW = 4
    RELEASE = 5


class RegionCleanupCause(IntEnum):
    NONE = 0
    BACKEND_ERROR = 1
    INTERRUPTED = 2
    BACKEND_STATE_MISMATCH = 3


REGION_PARTS = (RegionPartKind.PAYLOAD, RegionPartKind.COUNTER)
_ALLOCATION_ERROR_KINDS = (
    RegionControlErrorKind.BACKEND_FAILURE,
    RegionControlErrorKind.INTERNAL_INVARIANT,
)
_BACKEND_FAILURE_OPERATIONS = (
    RegionOperationKind.MATERIALIZE,
    RegionOperationKind.ZERO_BYTES,
    RegionOperationKind.DESCRIBE,
    RegionOperationKind.LOCAL_VIEW,
)


def _require_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    return value


def _require_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return value


def _require_uint64(name: str, value: object) -> int:
    number = _require_int(name, value)
    if number < 0 or number > _UINT64_MAX:
        raise ValueError(f"{name} overflowed uint64")
    return number


def _require_positive_uint64(name: str, value: object) -> int:
    number = _require_uint64(name, value)
    if number < 1:
        raise ValueError(f"{name} must be positive")
    return number


def _require_nonzero_uint64(name: str, value: object) -> int:
    return _require_positive_uint64(name, value)


def _require_int32(name: str, value: object) -> int:
    number = _require_int(name, value)
    if number < _INT32_MIN or number > _INT32_MAX:
        raise ValueError(f"{name} overflowed int32")
    return number


def _require_enum(enum_cls: type[IntEnum] | type[Enum], value: object, name: str):
    try:
        return enum_cls(value)
    except ValueError as exc:
        raise ValueError(f"unknown {name} {value!r}") from exc


def _require_backend_kind(value: object) -> BackendKind:
    return _require_enum(BackendKind, value, "planned backing kind")


def _require_part(value: object, *, allow_invalid: bool = False) -> RegionPartKind:
    part = _require_enum(RegionPartKind, value, "region part")
    if part is RegionPartKind.INVALID and not allow_invalid:
        raise ValueError("region part must be PAYLOAD or COUNTER")
    return part


def _require_posix_shm_token(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("POSIX shm token must be str")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("POSIX shm token must be ASCII") from exc
    if not encoded or len(encoded) > POSIX_SHM_TOKEN_MAX_BYTES:
        raise ValueError("POSIX shm token length must be in 1..32 bytes")
    if b"\x00" in encoded:
        raise ValueError("POSIX shm token must not contain NUL")
    return value


def _checked_add_u64(lhs: int, rhs: int, name: str) -> int:
    total = lhs + rhs
    if total > _UINT64_MAX:
        raise ValueError(f"{name} overflowed uint64")
    return total


def _require_counter_logical_bytes(value: object) -> int:
    logical_bytes = _require_positive_uint64("COUNTER logical_bytes", value)
    if logical_bytes % _COUNTER_LOGICAL_ALIGNMENT != 0:
        raise ValueError("COUNTER logical_bytes must be a multiple of 4")
    return logical_bytes


@dataclass(frozen=True)
class DeviceAllocationTarget:
    device_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "device_id", _require_int32("device_id", self.device_id))


@dataclass(frozen=True)
class HostAllocationTarget:
    pass


AllocationTarget = Union[DeviceAllocationTarget, HostAllocationTarget]


@dataclass(frozen=True)
class RegionAllocationContext:
    environment_kind: RegionEnvironmentKind
    target: AllocationTarget

    def __post_init__(self) -> None:
        environment = _require_enum(RegionEnvironmentKind, self.environment_kind, "environment kind")
        object.__setattr__(self, "environment_kind", environment)
        if not isinstance(self.target, (DeviceAllocationTarget, HostAllocationTarget)):
            raise TypeError("allocation target must be DeviceAllocationTarget or HostAllocationTarget")


@dataclass(frozen=True)
class RegionPartAllocationSpec:
    planned_backing_kind: BackendKind
    logical_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "planned_backing_kind", _require_backend_kind(self.planned_backing_kind))
        object.__setattr__(self, "logical_bytes", _require_positive_uint64("logical_bytes", self.logical_bytes))


@dataclass(frozen=True)
class RegionAllocationSpec:
    payload: RegionPartAllocationSpec
    counter: RegionPartAllocationSpec

    def __post_init__(self) -> None:
        if not isinstance(self.payload, RegionPartAllocationSpec):
            raise TypeError("payload spec must be RegionPartAllocationSpec")
        if not isinstance(self.counter, RegionPartAllocationSpec):
            raise TypeError("counter spec must be RegionPartAllocationSpec")
        _require_counter_logical_bytes(self.counter.logical_bytes)

    def part(self, kind: RegionPartKind) -> RegionPartAllocationSpec:
        part = _require_part(kind)
        if part is RegionPartKind.PAYLOAD:
            return self.payload
        return self.counter


@dataclass(frozen=True)
class VmmShareableHandleImport:
    device_id: int
    shareable_handle: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "device_id", _require_int32("device_id", self.device_id))
        object.__setattr__(
            self,
            "shareable_handle",
            _require_uint64("shareable_handle", self.shareable_handle),
        )


@dataclass(frozen=True)
class PosixShmImport:
    shm_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "shm_name", _require_posix_shm_token(self.shm_name))


ImportCapability = Union[VmmShareableHandleImport, PosixShmImport]


@dataclass(frozen=True)
class RegionPartExportDescriptor:
    planned_backing_kind: BackendKind
    logical_bytes: int
    mapping_bytes: int
    import_capability: ImportCapability

    def __post_init__(self) -> None:
        object.__setattr__(self, "planned_backing_kind", _require_backend_kind(self.planned_backing_kind))
        logical_bytes = _require_positive_uint64("logical_bytes", self.logical_bytes)
        mapping_bytes = _require_uint64("mapping_bytes", self.mapping_bytes)
        if mapping_bytes < logical_bytes:
            raise ValueError("mapping_bytes must cover logical_bytes")
        if not isinstance(self.import_capability, (VmmShareableHandleImport, PosixShmImport)):
            raise TypeError("import_capability must be a single VMM or POSIX variant")
        object.__setattr__(self, "logical_bytes", logical_bytes)
        object.__setattr__(self, "mapping_bytes", mapping_bytes)


@dataclass(frozen=True)
class RegionExportDescriptor:
    payload: RegionPartExportDescriptor
    counter: RegionPartExportDescriptor

    def __post_init__(self) -> None:
        if not isinstance(self.payload, RegionPartExportDescriptor):
            raise TypeError("payload export descriptor must be RegionPartExportDescriptor")
        if not isinstance(self.counter, RegionPartExportDescriptor):
            raise TypeError("counter export descriptor must be RegionPartExportDescriptor")
        _require_counter_logical_bytes(self.counter.logical_bytes)


@dataclass(frozen=True)
class RegionPartLocalView:
    part: RegionPartKind
    local_base: int
    logical_bytes: int

    def __post_init__(self) -> None:
        part = _require_part(self.part)
        local_base = _require_uint64("local_base", self.local_base)
        if part is RegionPartKind.COUNTER:
            logical_bytes = _require_counter_logical_bytes(self.logical_bytes)
            if local_base % _COUNTER_BASE_ALIGNMENT != 0:
                raise ValueError("COUNTER local_base must be 64-byte aligned")
        else:
            logical_bytes = _require_positive_uint64("logical_bytes", self.logical_bytes)
        _checked_add_u64(local_base, logical_bytes, f"{part.name} local span")
        object.__setattr__(self, "part", part)
        object.__setattr__(self, "local_base", local_base)
        object.__setattr__(self, "logical_bytes", logical_bytes)

    @property
    def local_end(self) -> int:
        return self.local_base + self.logical_bytes


def validate_independent_local_views(
    payload: RegionPartLocalView, counter: RegionPartLocalView
) -> tuple[RegionPartLocalView, RegionPartLocalView]:
    if not isinstance(payload, RegionPartLocalView) or payload.part is not RegionPartKind.PAYLOAD:
        raise ValueError("payload local view must use part PAYLOAD")
    if not isinstance(counter, RegionPartLocalView) or counter.part is not RegionPartKind.COUNTER:
        raise ValueError("counter local view must use part COUNTER")
    if payload.local_base < counter.local_end and counter.local_base < payload.local_end:
        raise ValueError("PAYLOAD and COUNTER local spans must not overlap")
    return payload, counter


@dataclass(frozen=True)
class RegionAllocationResult:
    provider_resource_id: int
    export_descriptor: RegionExportDescriptor

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_resource_id",
            _require_nonzero_uint64("provider_resource_id", self.provider_resource_id),
        )
        if not isinstance(self.export_descriptor, RegionExportDescriptor):
            raise TypeError("export_descriptor must be RegionExportDescriptor")


@dataclass(frozen=True)
class ProviderCleanupFailure:
    part: RegionPartKind
    backend_operation: RegionOperationKind
    typed_cause: RegionCleanupCause

    def __post_init__(self) -> None:
        part = _require_part(self.part)
        operation = _require_enum(RegionOperationKind, self.backend_operation, "backend operation")
        cause = _require_enum(RegionCleanupCause, self.typed_cause, "cleanup cause")
        if operation is RegionOperationKind.NONE:
            raise ValueError("cleanup failure requires a nonzero backend operation")
        if cause is RegionCleanupCause.NONE:
            raise ValueError("cleanup failure requires a nonzero typed cause")
        object.__setattr__(self, "part", part)
        object.__setattr__(self, "backend_operation", operation)
        object.__setattr__(self, "typed_cause", cause)


@dataclass(frozen=True)
class ProviderReleaseResult:
    provider_resource_id: int
    status: ProviderReleaseStatus
    failures: tuple[ProviderCleanupFailure, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_resource_id",
            _require_nonzero_uint64("provider_resource_id", self.provider_resource_id),
        )
        status = _require_enum(ProviderReleaseStatus, self.status, "release status")
        object.__setattr__(self, "status", status)
        failures = tuple(self.failures)
        if any(not isinstance(failure, ProviderCleanupFailure) for failure in failures):
            raise TypeError("release failures must be ProviderCleanupFailure values")
        parts = [failure.part for failure in failures]
        if len(parts) != len(set(parts)):
            raise ValueError("ProviderReleaseResult allows at most one failure per part")
        if status is ProviderReleaseStatus.CLEANUP_INCOMPLETE:
            if not failures:
                raise ValueError("CLEANUP_INCOMPLETE requires structured per-part failures")
        elif failures:
            raise ValueError(f"{status.name} must not carry cleanup failures")
        object.__setattr__(self, "failures", failures)


class RegionProviderError(Exception):
    pass


class RegionControlError(RegionProviderError):
    def __init__(
        self,
        kind: RegionControlErrorKind,
        message: str = "",
        *,
        failed_part: RegionPartKind | int = RegionPartKind.INVALID,
        failed_operation: RegionOperationKind | int = RegionOperationKind.NONE,
    ) -> None:
        control_kind = _require_enum(RegionControlErrorKind, kind, "control error kind")
        if control_kind is RegionControlErrorKind.NONE:
            raise ValueError("RegionControlError requires a nonzero control kind")
        part = _require_part(failed_part, allow_invalid=True)
        operation = _require_enum(RegionOperationKind, failed_operation, "failed operation")
        self.kind = control_kind
        self.failed_part = part
        self.failed_operation = operation
        self.message = str(message)
        super().__init__(self.message or control_kind.name)


class RegionAllocationError(RegionProviderError):
    def __init__(
        self,
        *,
        provisional_resource_id: int,
        control_kind: RegionControlErrorKind,
        failed_part: RegionPartKind | int,
        failed_operation: RegionOperationKind | int,
        cleanup_debt_remaining: bool,
        message: str = "",
    ) -> None:
        resource_id = _require_nonzero_uint64("provisional_resource_id", provisional_resource_id)
        kind = _require_enum(RegionControlErrorKind, control_kind, "control error kind")
        if kind not in _ALLOCATION_ERROR_KINDS:
            raise ValueError("RegionAllocationError control kind must be BACKEND_FAILURE or INTERNAL_INVARIANT")
        part = _require_part(failed_part, allow_invalid=True)
        operation = _require_enum(RegionOperationKind, failed_operation, "failed operation")
        if kind is RegionControlErrorKind.BACKEND_FAILURE and operation not in _BACKEND_FAILURE_OPERATIONS:
            raise ValueError("BACKEND_FAILURE requires a create-path operation")
        if operation is RegionOperationKind.RELEASE:
            raise ValueError("RegionAllocationError must not use RELEASE as the failed operation")
        self.provisional_resource_id = resource_id
        self.control_kind = kind
        self.failed_part = part
        self.failed_operation = operation
        self.cleanup_debt_remaining = _require_bool("cleanup_debt_remaining", cleanup_debt_remaining)
        self.message = str(message)
        super().__init__(self.message or kind.name)
