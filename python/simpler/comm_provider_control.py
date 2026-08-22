# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixed private provider-region control wire, transport handlers, and one-shot clients.

This module owns the versioned mailbox codec and transport adapters. It does not
import worker-chip compatibility types, W2 planning, or W5a identity.
"""

from __future__ import annotations

import logging
import struct
from enum import IntEnum
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Protocol, TypeVar

from _task_interface import BackendKind  # pyright: ignore[reportMissingImports]

from .comm_provider import (
    POSIX_SHM_TOKEN_MAX_BYTES,
    PosixShmImport,
    ProviderCleanupFailure,
    ProviderRegionStore,
    ProviderReleaseResult,
    ProviderReleaseStatus,
    RegionAllocationError,
    RegionAllocationResult,
    RegionAllocationSpec,
    RegionCleanupCause,
    RegionControlError,
    RegionControlErrorKind,
    RegionExportDescriptor,
    RegionOperationKind,
    RegionPartAllocationSpec,
    RegionPartExportDescriptor,
    RegionPartKind,
    RegionPartLocalView,
    VmmShareableHandleImport,
    validate_independent_local_views,
)

PROVIDER_REGION_CTRL_MAGIC = 0x50524354
PROVIDER_REGION_CTRL_ABI_MAJOR = 1
PROVIDER_REGION_CTRL_ABI_MINOR = 0
PROVIDER_REGION_CTRL_MAGIC_VERSION = (
    (PROVIDER_REGION_CTRL_MAGIC << 32) | (PROVIDER_REGION_CTRL_ABI_MAJOR << 16) | PROVIDER_REGION_CTRL_ABI_MINOR
)

ALLOCATE_REQUEST_BYTES = 48
ALLOCATE_REPLY_BYTES = 232
RELEASE_REQUEST_BYTES = 24
RELEASE_REPLY_BYTES = 48
EXPORT_PART_BYTES = 72
LOCAL_VIEW_BYTES = 24
CAPABILITY_PAYLOAD_BYTES = 40
VMM_CAPABILITY_PAYLOAD_BYTES = 16
ALLOCATE_PAYLOAD_EXPORT_OFFSET = 40
ALLOCATE_COUNTER_EXPORT_OFFSET = 112
ALLOCATE_PAYLOAD_VIEW_OFFSET = 184
ALLOCATE_COUNTER_VIEW_OFFSET = 208
RELEASE_PAYLOAD_FAILURE_OFFSET = 32
RELEASE_COUNTER_FAILURE_OFFSET = 40
COMMIT_TAG_OFFSET = 12
FAILURE_MASK_PAYLOAD = 1
FAILURE_MASK_COUNTER = 2

_ALLOCATE_REQUEST = struct.Struct("<QIIIIQQQ")
_ALLOCATE_REPLY_HEADER = struct.Struct("<QIIQIIII")
_EXPORT_PART = struct.Struct("<IIQQII40s")
_LOCAL_VIEW = struct.Struct("<IIQQ")
_RELEASE_REQUEST = struct.Struct("<QIIQ")
_RELEASE_REPLY = struct.Struct("<QIIQIIIIII")
_VMM_CAPABILITY = struct.Struct("<iI Q")
_COMMIT_TAG = struct.Struct("<I")

assert _ALLOCATE_REQUEST.size == ALLOCATE_REQUEST_BYTES
assert _ALLOCATE_REPLY_HEADER.size == ALLOCATE_PAYLOAD_EXPORT_OFFSET
assert _EXPORT_PART.size == EXPORT_PART_BYTES
assert _LOCAL_VIEW.size == LOCAL_VIEW_BYTES
assert _RELEASE_REQUEST.size == RELEASE_REQUEST_BYTES
assert _RELEASE_REPLY.size == RELEASE_REPLY_BYTES
assert ALLOCATE_PAYLOAD_EXPORT_OFFSET + 2 * EXPORT_PART_BYTES + 2 * LOCAL_VIEW_BYTES == ALLOCATE_REPLY_BYTES

_REQUEST_ERROR_KINDS = frozenset(
    {
        RegionControlErrorKind.BAD_MAGIC_VERSION,
        RegionControlErrorKind.BAD_MESSAGE_SIZE,
        RegionControlErrorKind.INVALID_ENUM_VALUE,
        RegionControlErrorKind.RESERVED_NONZERO,
        RegionControlErrorKind.INVALID_FIELD_VALUE,
    }
)
_ALLOCATION_ERROR_KINDS = frozenset(
    {
        RegionControlErrorKind.BACKEND_FAILURE,
        RegionControlErrorKind.INTERNAL_INVARIANT,
    }
)
_RELEASE_ERROR_KINDS = frozenset(
    {
        RegionControlErrorKind.INVALID_FIELD_VALUE,
        RegionControlErrorKind.STORE_LIFECYCLE,
        RegionControlErrorKind.INTERNAL_INVARIANT,
    }
)
_BACKEND_KINDS = {int(kind): kind for kind in BackendKind}


class AllocateReplyTag(IntEnum):
    EMPTY = 0
    SUCCESS = 1
    REQUEST_ERROR = 2
    ALLOCATION_ERROR = 3


class ReleaseReplyTag(IntEnum):
    EMPTY = 0
    RELEASED = 1
    ALREADY_GONE = 2
    UNKNOWN_RESOURCE = 3
    CLEANUP_INCOMPLETE = 4
    RELEASE_ERROR = 5


class ImportCapabilityWireKind(IntEnum):
    INVALID = 0
    VMM_SHAREABLE_HANDLE = 1
    POSIX_SHM = 2


class RegionControlMailbox(Protocol):
    def control_region_allocate(self, worker_id: int, request_shm_name: str, reply_shm_name: str) -> None: ...

    def control_region_release(self, worker_id: int, request_shm_name: str, reply_shm_name: str) -> None: ...


class RegionControlProtocolError(RegionControlError):
    """Committed-reply protocol failure that must not be retried."""


def _as_memoryview(buf: memoryview | bytes | bytearray, size: int, *, writable: bool) -> memoryview:
    view = buf if isinstance(buf, memoryview) else memoryview(buf)
    # POSIX shm mappings may be larger than the requested create size (page
    # rounding). The private wire frame is still exactly `size` bytes and lives
    # in the prefix; trailing mapping bytes are not part of the protocol.
    if view.nbytes < size:
        raise RegionControlError(
            RegionControlErrorKind.BAD_MESSAGE_SIZE,
            f"buffer must hold a {size}-byte wire frame",
        )
    if view.nbytes > size:
        view = memoryview(view).cast("B")[:size]
    if writable and view.readonly:
        raise RegionControlError(
            RegionControlErrorKind.INTERNAL_INVARIANT,
            "reply buffer must be writable",
        )
    return view


def _require_zero_span(view: memoryview, offset: int, length: int) -> None:
    if any(view[offset : offset + length]):
        raise RegionControlError(RegionControlErrorKind.RESERVED_NONZERO, "reserved bytes must be zero")


def _require_magic(magic: int) -> None:
    if magic != PROVIDER_REGION_CTRL_MAGIC_VERSION:
        raise RegionControlError(RegionControlErrorKind.BAD_MAGIC_VERSION, "unsupported provider control version")


_IntEnumT = TypeVar("_IntEnumT", bound=IntEnum)


def _require_shm_buf(shm: SharedMemory) -> memoryview:
    buf = shm.buf
    if buf is None:
        raise RegionControlError(
            RegionControlErrorKind.INTERNAL_INVARIANT,
            "shared memory mapping is missing",
        )
    return buf


def _require_enum(enum_cls: type[_IntEnumT], value: int, *, allow_zero: bool = False) -> _IntEnumT:
    try:
        typed = enum_cls(value)
    except ValueError as exc:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_ENUM_VALUE,
            f"unknown {enum_cls.__name__} {value}",
        ) from exc
    if not allow_zero and int(typed) == 0:
        raise RegionControlError(RegionControlErrorKind.INVALID_ENUM_VALUE, f"{enum_cls.__name__} must be nonzero")
    return typed


def _require_backend_kind(value: int) -> BackendKind:
    kind = _BACKEND_KINDS.get(int(value))
    if kind is None:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_ENUM_VALUE,
            f"unknown planned backing kind {value}",
        )
    return kind


def _publish_tag(view: memoryview, tag: int) -> None:
    _COMMIT_TAG.pack_into(view, COMMIT_TAG_OFFSET, int(tag))


def _acquire_tag(view: memoryview) -> int:
    return int(_COMMIT_TAG.unpack_from(view, COMMIT_TAG_OFFSET)[0])


def encode_allocate_request(buf: memoryview | bytearray, spec: RegionAllocationSpec) -> None:
    view = _as_memoryview(buf, ALLOCATE_REQUEST_BYTES, writable=True)
    view[:] = b"\x00" * ALLOCATE_REQUEST_BYTES
    _ALLOCATE_REQUEST.pack_into(
        view,
        0,
        PROVIDER_REGION_CTRL_MAGIC_VERSION,
        ALLOCATE_REQUEST_BYTES,
        int(spec.payload.planned_backing_kind),
        int(spec.counter.planned_backing_kind),
        0,
        int(spec.payload.logical_bytes),
        int(spec.counter.logical_bytes),
        0,
    )


def decode_allocate_request(buf: memoryview | bytes | bytearray) -> RegionAllocationSpec:
    view = _as_memoryview(buf, ALLOCATE_REQUEST_BYTES, writable=False)
    magic, request_bytes, payload_kind, counter_kind, reserved20, payload_bytes, counter_bytes, reserved40 = (
        _ALLOCATE_REQUEST.unpack_from(view, 0)
    )
    _require_magic(int(magic))
    if int(request_bytes) != ALLOCATE_REQUEST_BYTES:
        raise RegionControlError(RegionControlErrorKind.BAD_MESSAGE_SIZE, "allocate request_bytes must be 48")
    if int(reserved20) != 0 or int(reserved40) != 0:
        raise RegionControlError(
            RegionControlErrorKind.RESERVED_NONZERO, "allocate request reserved fields must be zero"
        )
    try:
        return RegionAllocationSpec(
            payload=RegionPartAllocationSpec(
                planned_backing_kind=_require_backend_kind(int(payload_kind)),
                logical_bytes=int(payload_bytes),
            ),
            counter=RegionPartAllocationSpec(
                planned_backing_kind=_require_backend_kind(int(counter_kind)),
                logical_bytes=int(counter_bytes),
            ),
        )
    except (TypeError, ValueError) as exc:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            str(exc) or "allocate request field is invalid",
        ) from exc


def _capability_wire_kind(capability: VmmShareableHandleImport | PosixShmImport) -> ImportCapabilityWireKind:
    if isinstance(capability, VmmShareableHandleImport):
        return ImportCapabilityWireKind.VMM_SHAREABLE_HANDLE
    return ImportCapabilityWireKind.POSIX_SHM


def _encode_capability_payload(capability: VmmShareableHandleImport | PosixShmImport) -> tuple[int, bytes]:
    payload = bytearray(CAPABILITY_PAYLOAD_BYTES)
    if isinstance(capability, VmmShareableHandleImport):
        _VMM_CAPABILITY.pack_into(payload, 0, int(capability.device_id), 0, int(capability.shareable_handle))
        return VMM_CAPABILITY_PAYLOAD_BYTES, bytes(payload)
    token = capability.shm_name.encode("ascii")
    payload[: len(token)] = token
    return len(token), bytes(payload)


def _decode_capability_payload(
    kind: ImportCapabilityWireKind, payload_bytes: int, payload: bytes
) -> VmmShareableHandleImport | PosixShmImport:
    if len(payload) != CAPABILITY_PAYLOAD_BYTES:
        raise RegionControlError(RegionControlErrorKind.BAD_MESSAGE_SIZE, "capability payload must be 40 bytes")
    if payload_bytes < 0 or payload_bytes > CAPABILITY_PAYLOAD_BYTES:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "capability payload bytes out of range")
    if any(payload[payload_bytes:]):
        raise RegionControlError(RegionControlErrorKind.RESERVED_NONZERO, "inactive capability bytes must be zero")
    if kind is ImportCapabilityWireKind.VMM_SHAREABLE_HANDLE:
        if payload_bytes != VMM_CAPABILITY_PAYLOAD_BYTES:
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE,
                "VMM capability payload_bytes must be 16",
            )
        device_id, reserved, shareable = _VMM_CAPABILITY.unpack_from(payload, 0)
        if int(reserved) != 0:
            raise RegionControlError(RegionControlErrorKind.RESERVED_NONZERO, "VMM capability reserved must be zero")
        try:
            return VmmShareableHandleImport(device_id=int(device_id), shareable_handle=int(shareable))
        except (TypeError, ValueError) as exc:
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE,
                str(exc) or "VMM capability fields are invalid",
            ) from exc
    if kind is ImportCapabilityWireKind.POSIX_SHM:
        if payload_bytes < 1 or payload_bytes > POSIX_SHM_TOKEN_MAX_BYTES:
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE,
                "POSIX shm token length must be in 1..32 bytes",
            )
        token = bytes(payload[:payload_bytes])
        if b"\x00" in token:
            raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "POSIX shm token must not contain NUL")
        try:
            return PosixShmImport(shm_name=token.decode("ascii"))
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE,
                str(exc) or "POSIX shm token is invalid",
            ) from exc
    raise RegionControlError(RegionControlErrorKind.INVALID_ENUM_VALUE, "import capability kind is invalid")


def _encode_export_part(view: memoryview, offset: int, part: RegionPartExportDescriptor) -> None:
    payload_bytes, payload = _encode_capability_payload(part.import_capability)
    _EXPORT_PART.pack_into(
        view,
        offset,
        int(part.planned_backing_kind),
        int(_capability_wire_kind(part.import_capability)),
        int(part.logical_bytes),
        int(part.mapping_bytes),
        int(payload_bytes),
        0,
        payload,
    )


def _decode_export_part(view: memoryview, offset: int) -> RegionPartExportDescriptor:
    backing, cap_kind, logical_bytes, mapping_bytes, payload_bytes, reserved, payload = _EXPORT_PART.unpack_from(
        view, offset
    )
    if int(reserved) != 0:
        raise RegionControlError(RegionControlErrorKind.RESERVED_NONZERO, "export-part reserved must be zero")
    capability = _decode_capability_payload(
        _require_enum(ImportCapabilityWireKind, int(cap_kind)),  # type: ignore[arg-type]
        int(payload_bytes),
        bytes(payload),
    )
    try:
        return RegionPartExportDescriptor(
            planned_backing_kind=_require_backend_kind(int(backing)),
            logical_bytes=int(logical_bytes),
            mapping_bytes=int(mapping_bytes),
            import_capability=capability,
        )
    except (TypeError, ValueError) as exc:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            str(exc) or "export-part fields are invalid",
        ) from exc


def _encode_local_view(view: memoryview, offset: int, local_view: RegionPartLocalView) -> None:
    _LOCAL_VIEW.pack_into(
        view,
        offset,
        int(local_view.part),
        0,
        int(local_view.local_base),
        int(local_view.logical_bytes),
    )


def _decode_local_view(view: memoryview, offset: int, expected: RegionPartKind) -> RegionPartLocalView:
    part, reserved, local_base, logical_bytes = _LOCAL_VIEW.unpack_from(view, offset)
    if int(reserved) != 0:
        raise RegionControlError(RegionControlErrorKind.RESERVED_NONZERO, "local-view reserved must be zero")
    typed_part = _require_enum(RegionPartKind, int(part))
    if typed_part is not expected:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "local-view part does not match slot")
    try:
        return RegionPartLocalView(part=expected, local_base=int(local_base), logical_bytes=int(logical_bytes))
    except (TypeError, ValueError) as exc:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            str(exc) or "local-view fields are invalid",
        ) from exc


def _zero_reply(view: memoryview) -> None:
    view[:] = b"\x00" * view.nbytes


def _discard_control_shm(shm: Any) -> None:
    name = getattr(shm, "name", "<unnamed>")
    logger = logging.getLogger("simpler")
    try:
        shm.close()
    except BaseException as exc:  # noqa: BLE001
        logger.warning(
            "provider control shm cleanup failed: name=%s operation=close error=%s",
            name,
            exc,
        )
    try:
        shm.unlink()
    except FileNotFoundError:
        return
    except BaseException as exc:  # noqa: BLE001
        logger.warning(
            "provider control shm cleanup failed: name=%s operation=unlink error=%s",
            name,
            exc,
        )


def _pack_allocate_header(
    view: memoryview,
    *,
    resource_id: int,
    error_kind: int,
    failed_part: int,
    failed_operation: int,
    debt: int,
) -> None:
    _ALLOCATE_REPLY_HEADER.pack_into(
        view,
        0,
        PROVIDER_REGION_CTRL_MAGIC_VERSION,
        ALLOCATE_REPLY_BYTES,
        0,
        int(resource_id),
        int(error_kind),
        int(failed_part),
        int(failed_operation),
        int(debt),
    )


def peek_allocate_reply_resource_id(buf: memoryview | bytes | bytearray) -> int:
    """Return the SUCCESS resource id, or 0 if the reply is not a committed SUCCESS."""
    try:
        view = _as_memoryview(buf, ALLOCATE_REPLY_BYTES, writable=False)
    except RegionControlError:
        return 0
    tag = _acquire_tag(view)
    if int(tag) != int(AllocateReplyTag.SUCCESS):
        return 0
    return int(_ALLOCATE_REPLY_HEADER.unpack_from(view, 0)[3])


def encode_allocate_success_reply(
    buf: memoryview | bytearray,
    result: RegionAllocationResult,
    payload_view: RegionPartLocalView,
    counter_view: RegionPartLocalView,
) -> None:
    view = _as_memoryview(buf, ALLOCATE_REPLY_BYTES, writable=True)
    validate_independent_local_views(payload_view, counter_view)
    _zero_reply(view)
    _pack_allocate_header(
        view, resource_id=result.provider_resource_id, error_kind=0, failed_part=0, failed_operation=0, debt=0
    )
    _encode_export_part(view, ALLOCATE_PAYLOAD_EXPORT_OFFSET, result.export_descriptor.payload)
    _encode_export_part(view, ALLOCATE_COUNTER_EXPORT_OFFSET, result.export_descriptor.counter)
    _encode_local_view(view, ALLOCATE_PAYLOAD_VIEW_OFFSET, payload_view)
    _encode_local_view(view, ALLOCATE_COUNTER_VIEW_OFFSET, counter_view)
    _publish_tag(view, AllocateReplyTag.SUCCESS)


def encode_allocate_request_error_reply(buf: memoryview | bytearray, kind: RegionControlErrorKind) -> None:
    view = _as_memoryview(buf, ALLOCATE_REPLY_BYTES, writable=True)
    if kind not in _REQUEST_ERROR_KINDS:
        raise RegionControlError(
            RegionControlErrorKind.INTERNAL_INVARIANT,
            "REQUEST_ERROR kind must be a wire syntax code",
        )
    _zero_reply(view)
    _pack_allocate_header(view, resource_id=0, error_kind=int(kind), failed_part=0, failed_operation=0, debt=0)
    _publish_tag(view, AllocateReplyTag.REQUEST_ERROR)


def encode_allocate_allocation_error_reply(buf: memoryview | bytearray, error: RegionAllocationError) -> None:
    view = _as_memoryview(buf, ALLOCATE_REPLY_BYTES, writable=True)
    _zero_reply(view)
    _pack_allocate_header(
        view,
        resource_id=error.provisional_resource_id,
        error_kind=int(error.control_kind),
        failed_part=int(error.failed_part),
        failed_operation=int(error.failed_operation),
        debt=1 if error.cleanup_debt_remaining else 0,
    )
    _publish_tag(view, AllocateReplyTag.ALLOCATION_ERROR)


def decode_allocate_reply(
    buf: memoryview | bytes | bytearray,
) -> tuple[
    AllocateReplyTag,
    RegionAllocationResult | None,
    RegionPartLocalView | None,
    RegionPartLocalView | None,
    RegionControlError | RegionAllocationError | None,
]:
    view = _as_memoryview(buf, ALLOCATE_REPLY_BYTES, writable=False)
    tag = _require_enum(AllocateReplyTag, _acquire_tag(view), allow_zero=True)
    if tag is AllocateReplyTag.EMPTY:
        raise RegionControlProtocolError(
            RegionControlErrorKind.INVALID_FIELD_VALUE, "allocate reply is missing commit tag"
        )
    magic, reply_bytes, _tag, resource_id, error_kind, failed_part, failed_operation, debt = (
        _ALLOCATE_REPLY_HEADER.unpack_from(view, 0)
    )
    _require_magic(int(magic))
    if int(reply_bytes) != ALLOCATE_REPLY_BYTES:
        raise RegionControlError(RegionControlErrorKind.BAD_MESSAGE_SIZE, "allocate reply_bytes must be 232")
    if tag is AllocateReplyTag.SUCCESS:
        if (
            int(resource_id) == 0
            or int(error_kind) != 0
            or int(failed_part) != 0
            or int(failed_operation) != 0
            or int(debt) != 0
        ):
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE,
                "SUCCESS requires a resource id and zero error fields",
            )
        descriptor = RegionExportDescriptor(
            payload=_decode_export_part(view, ALLOCATE_PAYLOAD_EXPORT_OFFSET),
            counter=_decode_export_part(view, ALLOCATE_COUNTER_EXPORT_OFFSET),
        )
        payload_view = _decode_local_view(view, ALLOCATE_PAYLOAD_VIEW_OFFSET, RegionPartKind.PAYLOAD)
        counter_view = _decode_local_view(view, ALLOCATE_COUNTER_VIEW_OFFSET, RegionPartKind.COUNTER)
        validate_independent_local_views(payload_view, counter_view)
        result = RegionAllocationResult(provider_resource_id=int(resource_id), export_descriptor=descriptor)
        return tag, result, payload_view, counter_view, None
    _require_zero_span(view, ALLOCATE_PAYLOAD_EXPORT_OFFSET, 2 * EXPORT_PART_BYTES + 2 * LOCAL_VIEW_BYTES)
    if tag is AllocateReplyTag.REQUEST_ERROR:
        if int(resource_id) != 0 or int(failed_part) != 0 or int(failed_operation) != 0 or int(debt) != 0:
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE,
                "REQUEST_ERROR requires resource id zero and inactive error slots",
            )
        kind = _require_enum(RegionControlErrorKind, int(error_kind))
        if kind not in _REQUEST_ERROR_KINDS:
            raise RegionControlError(
                RegionControlErrorKind.INVALID_ENUM_VALUE,
                "REQUEST_ERROR kind must be a wire syntax code",
            )
        return tag, None, None, None, RegionControlError(kind)  # type: ignore[arg-type]
    if tag is AllocateReplyTag.ALLOCATION_ERROR:
        if int(resource_id) == 0:
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE,
                "ALLOCATION_ERROR requires a provisional resource id",
            )
        if int(debt) not in (0, 1):
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE, "cleanup debt remaining must be 0 or 1"
            )
        kind = _require_enum(RegionControlErrorKind, int(error_kind))
        if kind not in _ALLOCATION_ERROR_KINDS:
            raise RegionControlError(
                RegionControlErrorKind.INVALID_ENUM_VALUE,
                "ALLOCATION_ERROR kind must be BACKEND_FAILURE or INTERNAL_INVARIANT",
            )
        error = RegionAllocationError(
            provisional_resource_id=int(resource_id),
            control_kind=kind,  # type: ignore[arg-type]
            failed_part=int(failed_part),
            failed_operation=int(failed_operation),
            cleanup_debt_remaining=bool(debt),
        )
        return tag, None, None, None, error
    raise RegionControlError(RegionControlErrorKind.INVALID_ENUM_VALUE, f"unknown allocate reply tag {int(tag)}")


def encode_release_request(buf: memoryview | bytearray, provider_resource_id: int) -> None:
    view = _as_memoryview(buf, RELEASE_REQUEST_BYTES, writable=True)
    view[:] = b"\x00" * RELEASE_REQUEST_BYTES
    _RELEASE_REQUEST.pack_into(
        view,
        0,
        PROVIDER_REGION_CTRL_MAGIC_VERSION,
        RELEASE_REQUEST_BYTES,
        0,
        int(provider_resource_id),
    )


def decode_release_request(buf: memoryview | bytes | bytearray) -> int:
    view = _as_memoryview(buf, RELEASE_REQUEST_BYTES, writable=False)
    magic, request_bytes, reserved, resource_id = _RELEASE_REQUEST.unpack_from(view, 0)
    _require_magic(int(magic))
    if int(request_bytes) != RELEASE_REQUEST_BYTES:
        raise RegionControlError(RegionControlErrorKind.BAD_MESSAGE_SIZE, "release request_bytes must be 24")
    if int(reserved) != 0:
        raise RegionControlError(RegionControlErrorKind.RESERVED_NONZERO, "release request reserved must be zero")
    if int(resource_id) == 0:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "provider_resource_id must be nonzero")
    return int(resource_id)


def _pack_release_reply(
    view: memoryview,
    *,
    resource_id: int,
    mask: int,
    error_kind: int,
    payload_op: int,
    payload_cause: int,
    counter_op: int,
    counter_cause: int,
) -> None:
    _RELEASE_REPLY.pack_into(
        view,
        0,
        PROVIDER_REGION_CTRL_MAGIC_VERSION,
        RELEASE_REPLY_BYTES,
        0,
        int(resource_id),
        int(mask),
        int(error_kind),
        int(payload_op),
        int(payload_cause),
        int(counter_op),
        int(counter_cause),
    )


def encode_release_result_reply(buf: memoryview | bytearray, result: ProviderReleaseResult) -> None:
    view = _as_memoryview(buf, RELEASE_REPLY_BYTES, writable=True)
    tag = {
        ProviderReleaseStatus.RELEASED: ReleaseReplyTag.RELEASED,
        ProviderReleaseStatus.ALREADY_GONE: ReleaseReplyTag.ALREADY_GONE,
        ProviderReleaseStatus.UNKNOWN_RESOURCE: ReleaseReplyTag.UNKNOWN_RESOURCE,
        ProviderReleaseStatus.CLEANUP_INCOMPLETE: ReleaseReplyTag.CLEANUP_INCOMPLETE,
    }[result.status]
    mask = 0
    payload_op = payload_cause = counter_op = counter_cause = 0
    if result.status is ProviderReleaseStatus.CLEANUP_INCOMPLETE:
        for failure in result.failures:
            if failure.part is RegionPartKind.PAYLOAD:
                mask |= FAILURE_MASK_PAYLOAD
                payload_op = int(failure.backend_operation)
                payload_cause = int(failure.typed_cause)
            else:
                mask |= FAILURE_MASK_COUNTER
                counter_op = int(failure.backend_operation)
                counter_cause = int(failure.typed_cause)
        if mask == 0:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                "CLEANUP_INCOMPLETE requires a nonzero failure mask",
            )
    _zero_reply(view)
    _pack_release_reply(
        view,
        resource_id=result.provider_resource_id,
        mask=mask,
        error_kind=0,
        payload_op=payload_op,
        payload_cause=payload_cause,
        counter_op=counter_op,
        counter_cause=counter_cause,
    )
    _publish_tag(view, tag)


def encode_release_error_reply(
    buf: memoryview | bytearray, kind: RegionControlErrorKind, provider_resource_id: int = 0
) -> None:
    view = _as_memoryview(buf, RELEASE_REPLY_BYTES, writable=True)
    if kind not in _RELEASE_ERROR_KINDS:
        raise RegionControlError(
            RegionControlErrorKind.INTERNAL_INVARIANT,
            "RELEASE_ERROR kind must be INVALID_FIELD_VALUE, STORE_LIFECYCLE, or INTERNAL_INVARIANT",
        )
    _zero_reply(view)
    _pack_release_reply(
        view,
        resource_id=int(provider_resource_id),
        mask=0,
        error_kind=int(kind),
        payload_op=0,
        payload_cause=0,
        counter_op=0,
        counter_cause=0,
    )
    _publish_tag(view, ReleaseReplyTag.RELEASE_ERROR)


def _failure_from_entry(part: RegionPartKind, operation: int, cause: int) -> ProviderCleanupFailure:
    typed_op = _require_enum(RegionOperationKind, operation)
    typed_cause = _require_enum(RegionCleanupCause, cause)
    return ProviderCleanupFailure(part=part, backend_operation=typed_op, typed_cause=typed_cause)  # type: ignore[arg-type]


def _require_zero_release_side_fields(
    mask: int,
    payload_op: int,
    payload_cause: int,
    counter_op: int,
    counter_cause: int,
    message: str,
) -> None:
    if mask != 0 or payload_op != 0 or payload_cause != 0 or counter_op != 0 or counter_cause != 0:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, message)


def _decode_cleanup_incomplete_failures(
    mask: int,
    payload_op: int,
    payload_cause: int,
    counter_op: int,
    counter_cause: int,
) -> list[ProviderCleanupFailure]:
    if mask & ~(FAILURE_MASK_PAYLOAD | FAILURE_MASK_COUNTER):
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "failure mask has unknown bits")
    if mask == 0:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "CLEANUP_INCOMPLETE requires a nonzero failure mask",
        )
    failures: list[ProviderCleanupFailure] = []
    if mask & FAILURE_MASK_PAYLOAD:
        failures.append(_failure_from_entry(RegionPartKind.PAYLOAD, payload_op, payload_cause))
    elif payload_op != 0 or payload_cause != 0:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "unselected PAYLOAD failure entry must be zero",
        )
    if mask & FAILURE_MASK_COUNTER:
        failures.append(_failure_from_entry(RegionPartKind.COUNTER, counter_op, counter_cause))
    elif counter_op != 0 or counter_cause != 0:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "unselected COUNTER failure entry must be zero",
        )
    return failures


def decode_release_reply(buf: memoryview | bytes | bytearray) -> ProviderReleaseResult | RegionControlError:
    view = _as_memoryview(buf, RELEASE_REPLY_BYTES, writable=False)
    tag = _require_enum(ReleaseReplyTag, _acquire_tag(view), allow_zero=True)
    if tag is ReleaseReplyTag.EMPTY:
        raise RegionControlProtocolError(
            RegionControlErrorKind.INVALID_FIELD_VALUE, "release reply is missing commit tag"
        )
    (
        magic,
        reply_bytes,
        _tag,
        resource_id,
        mask,
        error_kind,
        payload_op,
        payload_cause,
        counter_op,
        counter_cause,
    ) = _RELEASE_REPLY.unpack_from(view, 0)
    _require_magic(int(magic))
    if int(reply_bytes) != RELEASE_REPLY_BYTES:
        raise RegionControlError(RegionControlErrorKind.BAD_MESSAGE_SIZE, "release reply_bytes must be 48")
    if tag is ReleaseReplyTag.RELEASE_ERROR:
        _require_zero_release_side_fields(
            int(mask),
            int(payload_op),
            int(payload_cause),
            int(counter_op),
            int(counter_cause),
            "RELEASE_ERROR requires a zero failure mask and entries",
        )
        kind = _require_enum(RegionControlErrorKind, int(error_kind))
        if kind not in _RELEASE_ERROR_KINDS:
            raise RegionControlError(
                RegionControlErrorKind.INVALID_ENUM_VALUE,
                "RELEASE_ERROR kind is not allowed",
            )
        return RegionControlError(kind)
    if int(resource_id) == 0:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE, "release reply resource id must be nonzero"
        )
    if int(error_kind) != 0:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "lifecycle release replies require a zero control error",
        )
    status = {
        ReleaseReplyTag.RELEASED: ProviderReleaseStatus.RELEASED,
        ReleaseReplyTag.ALREADY_GONE: ProviderReleaseStatus.ALREADY_GONE,
        ReleaseReplyTag.UNKNOWN_RESOURCE: ProviderReleaseStatus.UNKNOWN_RESOURCE,
        ReleaseReplyTag.CLEANUP_INCOMPLETE: ProviderReleaseStatus.CLEANUP_INCOMPLETE,
    }.get(tag)
    if status is None:
        raise RegionControlError(RegionControlErrorKind.INVALID_ENUM_VALUE, f"unknown release reply tag {int(tag)}")
    failures: list[ProviderCleanupFailure] = []
    if status is ProviderReleaseStatus.CLEANUP_INCOMPLETE:
        failures = _decode_cleanup_incomplete_failures(
            int(mask),
            int(payload_op),
            int(payload_cause),
            int(counter_op),
            int(counter_cause),
        )
    else:
        _require_zero_release_side_fields(
            int(mask),
            int(payload_op),
            int(payload_cause),
            int(counter_op),
            int(counter_cause),
            "non-incomplete release replies require zero failure mask and entries",
        )
    return ProviderReleaseResult(
        provider_resource_id=int(resource_id),
        status=status,
        failures=tuple(failures),
    )


def handle_ctrl_region_allocate(req_buf: memoryview, reply_buf: memoryview, store: ProviderRegionStore) -> None:
    reply = _as_memoryview(reply_buf, ALLOCATE_REPLY_BYTES, writable=True)
    _zero_reply(reply)
    try:
        spec = decode_allocate_request(req_buf)
    except RegionControlError as exc:
        if exc.kind in _REQUEST_ERROR_KINDS:
            encode_allocate_request_error_reply(reply, exc.kind)
            return
        raise
    try:
        result = store.allocate_and_export(spec)
    except RegionAllocationError as exc:
        encode_allocate_allocation_error_reply(reply, exc)
        return
    try:
        payload_view = store.local_view(result.provider_resource_id, RegionPartKind.PAYLOAD)
        counter_view = store.local_view(result.provider_resource_id, RegionPartKind.COUNTER)
        encode_allocate_success_reply(reply, result, payload_view, counter_view)
    finally:
        if _acquire_tag(reply) != int(AllocateReplyTag.SUCCESS):
            store.release(result.provider_resource_id)


def handle_ctrl_region_release(req_buf: memoryview, reply_buf: memoryview, store: ProviderRegionStore) -> None:
    reply = _as_memoryview(reply_buf, RELEASE_REPLY_BYTES, writable=True)
    _zero_reply(reply)
    try:
        resource_id = decode_release_request(req_buf)
    except RegionControlError as exc:
        kind = exc.kind if exc.kind in _RELEASE_ERROR_KINDS else RegionControlErrorKind.INVALID_FIELD_VALUE
        encode_release_error_reply(reply, kind)
        return
    try:
        result = store.release(resource_id)
    except RegionControlError as exc:
        if exc.kind in _RELEASE_ERROR_KINDS:
            encode_release_error_reply(reply, exc.kind, provider_resource_id=resource_id)
            return
        raise
    encode_release_result_reply(reply, result)


class ProviderAllocateClient:
    """Mailbox client that publishes one allocate request and waits for a committed reply."""

    def __init__(self, mailbox: RegionControlMailbox, worker_id: int) -> None:
        self._mailbox = mailbox
        self._worker_id = int(worker_id)
        self.dispatch_started = False
        self.committed_resource_id = 0

    def allocate(
        self, spec: RegionAllocationSpec
    ) -> tuple[RegionAllocationResult, RegionPartLocalView, RegionPartLocalView]:
        self.dispatch_started = False
        self.committed_resource_id = 0
        req_shm = SharedMemory(create=True, size=ALLOCATE_REQUEST_BYTES)
        reply_shm = SharedMemory(create=True, size=ALLOCATE_REPLY_BYTES)
        req_buf = memoryview(_require_shm_buf(req_shm))
        reply_buf = memoryview(_require_shm_buf(reply_shm))
        try:
            encode_allocate_request(req_buf, spec)
            self.dispatch_started = True
            try:
                self._mailbox.control_region_allocate(self._worker_id, req_shm.name, reply_shm.name)
            except BaseException:
                self._observe_committed_success(reply_buf)
                raise
            tag, result, payload_view, counter_view, error = decode_allocate_reply(reply_buf)
            if tag is AllocateReplyTag.SUCCESS:
                assert result is not None and payload_view is not None and counter_view is not None
                self.committed_resource_id = int(result.provider_resource_id)
                return result, payload_view, counter_view
            if error is not None:
                raise error
            raise RegionControlProtocolError(
                RegionControlErrorKind.INVALID_FIELD_VALUE, "allocate reply is uncommitted"
            )
        finally:
            del req_buf
            del reply_buf
            _discard_control_shm(req_shm)
            _discard_control_shm(reply_shm)

    def _observe_committed_success(self, reply_buf: memoryview) -> None:
        try:
            view = _as_memoryview(reply_buf, ALLOCATE_REPLY_BYTES, writable=False)
            if _acquire_tag(view) != int(AllocateReplyTag.SUCCESS):
                return
            tag, result, _payload, _counter, _error = decode_allocate_reply(reply_buf)
        except Exception:
            return
        if tag is AllocateReplyTag.SUCCESS and result is not None:
            self.committed_resource_id = int(result.provider_resource_id)


class ProviderReleaseClient:
    """One-shot mailbox client for provider release. Transport and decode failures are terminal."""

    def __init__(self, mailbox: RegionControlMailbox, worker_id: int) -> None:
        self._mailbox = mailbox
        self._worker_id = int(worker_id)
        self._issued = False
        self._terminal: BaseException | ProviderReleaseResult | None = None

    def release(self, provider_resource_id: int) -> ProviderReleaseResult:
        if self._issued:
            if isinstance(self._terminal, ProviderReleaseResult):
                return self._terminal
            assert self._terminal is not None
            raise self._terminal
        self._issued = True
        try:
            result = self._issue(provider_resource_id)
        except BaseException as exc:
            self._terminal = exc
            raise
        self._terminal = result
        return result

    def _issue(self, provider_resource_id: int) -> ProviderReleaseResult:
        req_shm = SharedMemory(create=True, size=RELEASE_REQUEST_BYTES)
        reply_shm = SharedMemory(create=True, size=RELEASE_REPLY_BYTES)
        req_buf = memoryview(_require_shm_buf(req_shm))
        reply_buf = memoryview(_require_shm_buf(reply_shm))
        try:
            encode_release_request(req_buf, provider_resource_id)
            try:
                self._mailbox.control_region_release(self._worker_id, req_shm.name, reply_shm.name)
            except BaseException as exc:
                raise RegionControlProtocolError(
                    RegionControlErrorKind.INTERNAL_INVARIANT,
                    str(exc) or "release transport failed",
                ) from exc
            decoded = decode_release_reply(reply_buf)
            if isinstance(decoded, RegionControlError):
                raise RegionControlProtocolError(decoded.kind, decoded.message or decoded.kind.name)
            return decoded
        finally:
            del req_buf
            del reply_buf
            _discard_control_shm(req_shm)
            _discard_control_shm(reply_shm)
