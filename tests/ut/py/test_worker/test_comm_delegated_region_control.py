# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for the W5a DRCT v1 delegated-region control codec."""

from __future__ import annotations

import ast
import struct
from pathlib import Path

import pytest
from simpler.buffer import BackendKind
from simpler.comm_delegated_region_control import (
    ALLOCATE_COUNTER_EXPORT_OFFSET,
    ALLOCATE_COUNTER_VIEW_OFFSET,
    ALLOCATE_OUTCOME_OFFSET,
    ALLOCATE_PAYLOAD_EXPORT_OFFSET,
    ALLOCATE_PAYLOAD_VIEW_OFFSET,
    ALLOCATE_PROJECTION_BYTES,
    ALLOCATE_REPLY_BYTES,
    ALLOCATE_REQUEST_HARD_CEILING,
    DELEGATED_REGION_CTRL_MAGIC,
    DELEGATED_REGION_CTRL_MAGIC_VERSION,
    PATH_CEILING_BYTES,
    RELEASE_OUTCOME_BYTES,
    RELEASE_REPLY_BYTES,
    RELEASE_REQUEST_HARD_CEILING,
    REPLY_HEADER_BYTES,
    REPLY_TAG_OFFSET,
    REQUEST_HEADER_BYTES,
    DelegatedAllocateReply,
    DelegatedAllocateReplyTag,
    DelegatedAllocateRequest,
    DelegatedRegionOperation,
    DelegatedReleaseReply,
    DelegatedReleaseReplyTag,
    DelegatedReleaseRequest,
    ProviderTransactionTable,
    _hop_staging_copy,
    _inspect_delegated_route,
    encode_reply,
    encode_request,
    handle_terminal_delegated_region,
    parse_reply,
    parse_request,
    publish_reply,
)
from simpler.comm_endpoints import AdapterKind, AdapterProfile, EndpointDeploymentKind, RegionTopologyKind
from simpler.comm_provider import (
    PosixShmImport,
    ProviderCleanupFailure,
    ProviderReleaseResult,
    ProviderReleaseStatus,
    RegionAllocationError,
    RegionAllocationResult,
    RegionCleanupCause,
    RegionControlError,
    RegionControlErrorKind,
    RegionExportDescriptor,
    RegionOperationKind,
    RegionPartExportDescriptor,
    RegionPartKind,
    RegionPartLocalView,
    VmmShareableHandleImport,
)
from tests.ut.py.test_worker.test_comm_provider import FakeShellFactory, _open_store
from tests.ut.py.test_worker.w5a_migration_baseline import (
    W4_5_ALLOCATE_ALLOCATION_ERROR_OUTCOMES,
    W4_5_ALLOCATE_REQUEST_ERROR_OUTCOMES,
    W4_5_ALLOCATE_SUCCESS_OUTCOME,
    W4_5_RELEASE_CLEAN_OUTCOMES,
    W4_5_RELEASE_CLEANUP_INCOMPLETE_OUTCOMES,
    W4_5_RELEASE_ERROR_OUTCOMES,
    W5A_UNKNOWN_TRANSACTION_OUTCOME,
)

_SESSION = b"\x11\x22\x33\x44\x55\x66\x77\x88"
_INITIATOR_PATH = b"L3"
_PROVIDER_PATH = b"L3/L2[0]"
_OLD_PRCT_MAGIC = 0x5052435400010000
_MODULE_PATH = Path(__file__).resolve().parents[4] / "python" / "simpler" / "comm_delegated_region_control.py"


def _allocate_request(
    *,
    session_instance_id: bytes = _SESSION,
    transaction_id: int = 1,
    initiator_path: bytes = _INITIATOR_PATH,
    provider_path: bytes = _PROVIDER_PATH,
) -> DelegatedAllocateRequest:
    return DelegatedAllocateRequest(
        session_instance_id=session_instance_id,
        transaction_id=transaction_id,
        initiator_path=initiator_path,
        provider_path=provider_path,
        payload_logical_bytes=64,
        counter_logical_bytes=8,
    )


def _release_request(
    *,
    session_instance_id: bytes = _SESSION,
    transaction_id: int = 1,
    provider_path: bytes = _PROVIDER_PATH,
) -> DelegatedReleaseRequest:
    return DelegatedReleaseRequest(
        session_instance_id=session_instance_id,
        transaction_id=transaction_id,
        provider_path=provider_path,
    )


def _posix_part(logical_bytes: int, name: str) -> RegionPartExportDescriptor:
    return RegionPartExportDescriptor(
        planned_backing_kind=BackendKind.VMM_WINDOW,
        logical_bytes=logical_bytes,
        mapping_bytes=logical_bytes,
        import_capability=PosixShmImport(shm_name=name),
    )


def _vmm_part(logical_bytes: int, handle: int = 21) -> RegionPartExportDescriptor:
    return RegionPartExportDescriptor(
        planned_backing_kind=BackendKind.VMM_WINDOW,
        logical_bytes=logical_bytes,
        mapping_bytes=logical_bytes,
        import_capability=VmmShareableHandleImport(device_id=2, shareable_handle=handle),
    )


def _success_result() -> tuple[RegionAllocationResult, RegionPartLocalView, RegionPartLocalView]:
    result = RegionAllocationResult(
        provider_resource_id=11,
        export_descriptor=RegionExportDescriptor(payload=_posix_part(64, "/pto_payload_a"), counter=_vmm_part(8)),
    )
    payload = RegionPartLocalView(part=RegionPartKind.PAYLOAD, local_base=0x1000, logical_bytes=64)
    counter = RegionPartLocalView(part=RegionPartKind.COUNTER, local_base=0x2000, logical_bytes=8)
    return result, payload, counter


def _allocated_reply() -> DelegatedAllocateReply:
    result, payload, counter = _success_result()
    return DelegatedAllocateReply(
        tag=DelegatedAllocateReplyTag.ALLOCATED,
        session_instance_id=_SESSION,
        transaction_id=1,
        result=result,
        payload_view=payload,
        counter_view=counter,
    )


def _release_result_reply(tag: DelegatedReleaseReplyTag, result: ProviderReleaseResult) -> DelegatedReleaseReply:
    return DelegatedReleaseReply(
        tag=tag,
        session_instance_id=_SESSION,
        transaction_id=1,
        result=result,
    )


def _canonical_path_of_length(n: int) -> bytes:
    if n < 2:
        raise ValueError("canonical path shorter than L<n> is impossible")
    if n == 2:
        return b"L3"
    remaining = n - 2
    child = "/L1[0]"
    if remaining < 6:
        return f"L{10 ** (n - 2)}".encode()
    count, extra = divmod(remaining, 6)
    if extra == 0:
        return ("L3" + child * count).encode()
    if count == 0:
        return f"L3/L1[{10 ** (remaining - 6)}]".encode()
    last = f"/L1[{10 ** extra}]"
    text = "L3" + child * (count - 1) + last
    assert len(text) == n
    return text.encode()


def _kind(exc: RegionControlError) -> RegionControlErrorKind:
    assert isinstance(exc, RegionControlError)
    return exc.kind


def test_module_does_not_import_old_provider_control():
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert "simpler.comm_provider_control" not in imported
    assert "comm_provider_control" not in imported


def test_fixed_sizes_and_tag_offset_are_asserted():
    assert REQUEST_HEADER_BYTES == 40
    assert ALLOCATE_PROJECTION_BYTES == 64
    assert REPLY_HEADER_BYTES == 40
    assert REPLY_TAG_OFFSET == 12
    assert ALLOCATE_REPLY_BYTES == 256
    assert RELEASE_REPLY_BYTES == 72
    assert ALLOCATE_REQUEST_HARD_CEILING == 616
    assert RELEASE_REQUEST_HARD_CEILING == 296
    assert PATH_CEILING_BYTES == 256
    assert ALLOCATE_PAYLOAD_EXPORT_OFFSET == 64
    assert ALLOCATE_COUNTER_EXPORT_OFFSET == 136
    assert ALLOCATE_PAYLOAD_VIEW_OFFSET == 208
    assert ALLOCATE_COUNTER_VIEW_OFFSET == 232
    assert DELEGATED_REGION_CTRL_MAGIC_VERSION == 0x4452435400010000


def test_allocate_request_round_trip_and_header_layout():
    staged = encode_request(_allocate_request(), staged_capacity=256)
    assert len(staged) == 256
    magic, request_bytes, operation, session, transaction_id, provider_len, initiator_len = struct.unpack_from(
        "<QII8sQII", staged, 0
    )
    assert magic == DELEGATED_REGION_CTRL_MAGIC_VERSION
    assert operation == int(DelegatedRegionOperation.DELEGATED_ALLOCATE)
    assert session == _SESSION
    assert transaction_id == 1
    assert initiator_len == len(_INITIATOR_PATH)
    assert provider_len == len(_PROVIDER_PATH)
    assert request_bytes == REQUEST_HEADER_BYTES + ALLOCATE_PROJECTION_BYTES + initiator_len + provider_len
    assert staged[request_bytes:] == b"\x00" * (256 - request_bytes)
    assert staged[request_bytes - provider_len : request_bytes] == _PROVIDER_PATH
    envelope = parse_request(staged)
    assert envelope.operation is DelegatedRegionOperation.DELEGATED_ALLOCATE
    assert envelope.session_instance_id == _SESSION
    assert envelope.transaction_id == 1
    assert envelope.provider_path == _PROVIDER_PATH
    decoded = envelope.decode_terminal()
    assert isinstance(decoded, DelegatedAllocateRequest)
    assert decoded.initiator_path == _INITIATOR_PATH
    assert decoded.provider_path == _PROVIDER_PATH
    assert decoded.spec.payload.logical_bytes == 64
    assert decoded.spec.counter.logical_bytes == 8
    assert decoded.spec.payload.planned_backing_kind is BackendKind.VMM_WINDOW
    assert decoded.topology is RegionTopologyKind.SINGLE_OWNER
    assert decoded.initiator_deployment is EndpointDeploymentKind.HOST_CPU
    assert decoded.provider_deployment is EndpointDeploymentKind.DEVICE_AICPU
    assert decoded.payload_consumer_adapter_kind is AdapterKind.OWNER_DELEGATED_COPY
    assert decoded.payload_consumer_adapter_profile is AdapterProfile.HOST_VMM_COPY


def test_release_request_round_trip_has_empty_body():
    staged = encode_request(_release_request(), staged_capacity=72)
    magic, request_bytes, operation, session, transaction_id, provider_len, initiator_len = struct.unpack_from(
        "<QII8sQII", staged, 0
    )
    assert operation == int(DelegatedRegionOperation.DELEGATED_RELEASE)
    assert initiator_len == 0
    assert request_bytes == REQUEST_HEADER_BYTES + provider_len
    assert staged[REQUEST_HEADER_BYTES:request_bytes] == _PROVIDER_PATH
    decoded = parse_request(staged).decode_terminal()
    assert isinstance(decoded, DelegatedReleaseRequest)
    assert decoded.session_instance_id == session
    assert decoded.transaction_id == transaction_id
    assert decoded.provider_path == _PROVIDER_PATH


def test_opaque_session_round_trips_byte_for_byte():
    session = bytes(range(8, 0, -1))
    staged = encode_request(_allocate_request(session_instance_id=session, transaction_id=7), staged_capacity=256)
    envelope = parse_request(staged)
    assert envelope.session_instance_id == session
    assert envelope.session_instance_id is not session
    assert int.from_bytes(session, "little") != int.from_bytes(session, "big")
    reply = encode_reply(
        DelegatedAllocateReply(
            tag=DelegatedAllocateReplyTag.ERROR,
            session_instance_id=session,
            transaction_id=7,
            error_kind=RegionControlErrorKind.INVALID_FIELD_VALUE,
        )
    )
    assert parse_reply(reply).session_instance_id == session
    assert parse_reply(reply).decode_outcome().session_instance_id == session


@pytest.mark.parametrize("length", [2, 8, 255, 256])
def test_canonical_path_boundaries_and_provider_tail(length: int):
    path = _canonical_path_of_length(length)
    assert len(path) == length
    staged = encode_request(_release_request(provider_path=path), staged_capacity=max(72, 40 + length))
    envelope = parse_request(staged)
    assert envelope.provider_path == path
    assert envelope.frame.endswith(path)
    assert envelope.request_bytes == REQUEST_HEADER_BYTES + length


def test_path_zero_one_and_257_fail_closed():
    empty_path = bytearray(encode_request(_release_request(), staged_capacity=72))
    struct.pack_into("<I", empty_path, 32, 0)
    struct.pack_into("<I", empty_path, 8, REQUEST_HEADER_BYTES)
    empty_path[REQUEST_HEADER_BYTES:] = b"\x00" * (len(empty_path) - REQUEST_HEADER_BYTES)
    with pytest.raises(RegionControlError) as empty:
        parse_request(empty_path)
    assert _kind(empty.value) is RegionControlErrorKind.INVALID_FIELD_VALUE

    one = bytearray(encode_request(_release_request(), staged_capacity=72))
    struct.pack_into("<I", one, 8, REQUEST_HEADER_BYTES + 1)
    struct.pack_into("<I", one, 32, 1)
    one[REQUEST_HEADER_BYTES : REQUEST_HEADER_BYTES + 1] = b"L"
    one[REQUEST_HEADER_BYTES + 1 :] = b"\x00" * (len(one) - REQUEST_HEADER_BYTES - 1)
    with pytest.raises(RegionControlError) as short:
        parse_request(one)
    assert _kind(short.value) is RegionControlErrorKind.INVALID_FIELD_VALUE

    over = bytearray(REQUEST_HEADER_BYTES + 257)
    struct.pack_into(
        "<QII8sQII",
        over,
        0,
        DELEGATED_REGION_CTRL_MAGIC_VERSION,
        REQUEST_HEADER_BYTES + 257,
        int(DelegatedRegionOperation.DELEGATED_RELEASE),
        _SESSION,
        1,
        257,
        0,
    )
    over[REQUEST_HEADER_BYTES:] = b"A" * 257
    with pytest.raises(RegionControlError) as huge:
        parse_request(over)
    assert _kind(huge.value) is RegionControlErrorKind.BAD_MESSAGE_SIZE


def test_allocate_and_release_hard_ceilings():
    init = _canonical_path_of_length(256)
    prov = _canonical_path_of_length(256)
    staged = encode_request(_allocate_request(initiator_path=init, provider_path=prov), staged_capacity=616)
    assert len(staged) == 616
    envelope = parse_request(staged)
    assert envelope.request_bytes == 616
    assert envelope.decode_terminal().initiator_path == init

    release = encode_request(_release_request(provider_path=prov), staged_capacity=296)
    assert parse_request(release).request_bytes == 296

    over = bytearray(encode_request(_allocate_request(), staged_capacity=256))
    struct.pack_into("<I", over, 8, 617)
    with pytest.raises(RegionControlError) as allocate_over:
        parse_request(over)
    assert _kind(allocate_over.value) is RegionControlErrorKind.BAD_MESSAGE_SIZE

    release_over = bytearray(encode_request(_release_request(), staged_capacity=72))
    struct.pack_into("<I", release_over, 8, 297)
    with pytest.raises(RegionControlError) as release_exc:
        parse_request(release_over)
    assert _kind(release_exc.value) is RegionControlErrorKind.BAD_MESSAGE_SIZE


def test_owned_envelope_survives_source_zero_and_reuse():
    staged = encode_request(_allocate_request(transaction_id=9), staged_capacity=256)
    view = memoryview(staged)
    envelope = parse_request(view)
    snapshot = envelope.frame
    staged[:] = b"\xff" * len(staged)
    view[0] = 0
    decoded = envelope.decode_terminal()
    assert envelope.frame == snapshot
    assert envelope.session_instance_id == _SESSION
    assert decoded.transaction_id == 9
    assert decoded.provider_path == _PROVIDER_PATH

    committed = encode_reply(_allocated_reply())
    reply_buf = bytearray(256)
    publish_reply(memoryview(reply_buf), committed)
    reply_env = parse_reply(reply_buf)
    reply_buf[:] = b"\x00" * 256
    outcome = reply_env.decode_outcome()
    assert outcome.tag is DelegatedAllocateReplyTag.ALLOCATED
    assert outcome.result is not None
    assert outcome.result.provider_resource_id == 11


def test_fail_closed_magic_version_operation_length_enum_reserved_and_tail():
    good = encode_request(_allocate_request(), staged_capacity=256)

    wrong_magic = bytearray(good)
    struct.pack_into("<Q", wrong_magic, 0, _OLD_PRCT_MAGIC)
    with pytest.raises(RegionControlError) as magic:
        parse_request(wrong_magic)
    assert _kind(magic.value) is RegionControlErrorKind.BAD_MAGIC_VERSION

    wrong_version = bytearray(good)
    struct.pack_into("<Q", wrong_version, 0, (DELEGATED_REGION_CTRL_MAGIC << 32) | 0x00020000)
    with pytest.raises(RegionControlError) as version:
        parse_request(wrong_version)
    assert _kind(version.value) is RegionControlErrorKind.BAD_MAGIC_VERSION

    wrong_op = bytearray(good)
    struct.pack_into("<I", wrong_op, 12, 99)
    with pytest.raises(RegionControlError) as operation:
        parse_request(wrong_op)
    assert _kind(operation.value) is RegionControlErrorKind.INVALID_ENUM_VALUE

    short = good[:30]
    with pytest.raises(RegionControlError) as size:
        parse_request(short)
    assert _kind(size.value) is RegionControlErrorKind.BAD_MESSAGE_SIZE

    tail = bytearray(good)
    tail[-1] = 1
    with pytest.raises(RegionControlError) as reserved:
        parse_request(tail)
    assert _kind(reserved.value) is RegionControlErrorKind.RESERVED_NONZERO

    projection = bytearray(good)
    # topology reserved u32 at offset 40+12
    struct.pack_into("<I", projection, REQUEST_HEADER_BYTES + 12, 1)
    with pytest.raises(RegionControlError) as proj:
        parse_request(projection).decode_terminal()
    assert _kind(proj.value) is RegionControlErrorKind.RESERVED_NONZERO

    unknown_topology = bytearray(good)
    struct.pack_into("<I", unknown_topology, REQUEST_HEADER_BYTES, 99)
    with pytest.raises(RegionControlError) as enum_exc:
        parse_request(unknown_topology).decode_terminal()
    assert _kind(enum_exc.value) is RegionControlErrorKind.INVALID_ENUM_VALUE

    known_wrong = bytearray(good)
    struct.pack_into("<I", known_wrong, REQUEST_HEADER_BYTES + 4, int(EndpointDeploymentKind.DEVICE_AICORE))
    with pytest.raises(RegionControlError) as field:
        parse_request(known_wrong).decode_terminal()
    assert _kind(field.value) is RegionControlErrorKind.INVALID_FIELD_VALUE

    tx0 = bytearray(good)
    struct.pack_into("<Q", tx0, 24, 0)
    with pytest.raises(RegionControlError) as tx:
        parse_request(tx0)
    assert _kind(tx.value) is RegionControlErrorKind.INVALID_FIELD_VALUE


def test_decode_terminal_not_required_for_bounded_routing():
    staged = encode_request(_allocate_request(), staged_capacity=256)
    mutated = bytearray(staged)
    struct.pack_into("<I", mutated, REQUEST_HEADER_BYTES, 99)
    envelope = parse_request(mutated)
    assert envelope.provider_path == _PROVIDER_PATH
    assert envelope.session_instance_id == _SESSION
    with pytest.raises(RegionControlError) as exc:
        envelope.decode_terminal()
    assert _kind(exc.value) is RegionControlErrorKind.INVALID_ENUM_VALUE


def test_allocate_reply_tags_and_unique_error_kind():
    allocated = encode_reply(_allocated_reply())
    assert allocated[40:] == W4_5_ALLOCATE_SUCCESS_OUTCOME
    decoded = parse_reply(allocated).decode_outcome()
    assert decoded.tag is DelegatedAllocateReplyTag.ALLOCATED
    assert decoded.error_kind is RegionControlErrorKind.NONE
    assert decoded.result is not None
    assert decoded.payload_view is not None
    assert decoded.counter_view is not None

    for name, outcome in W4_5_ALLOCATE_REQUEST_ERROR_OUTCOMES.items():
        kind = RegionControlErrorKind[name]
        frame = encode_reply(
            DelegatedAllocateReply(
                tag=DelegatedAllocateReplyTag.ERROR,
                session_instance_id=_SESSION,
                transaction_id=1,
                error_kind=kind,
            )
        )
        assert frame[40:] == outcome
        parsed = parse_reply(frame).decode_outcome()
        assert parsed.tag is DelegatedAllocateReplyTag.ERROR
        assert parsed.error_kind is kind
        assert parsed.result is None
        assert parsed.payload_view is None
        assert parsed.provisional_resource_id == 0

    allocation = encode_reply(
        DelegatedAllocateReply(
            tag=DelegatedAllocateReplyTag.ERROR,
            session_instance_id=_SESSION,
            transaction_id=1,
            error=RegionAllocationError(
                provisional_resource_id=7,
                control_kind=RegionControlErrorKind.BACKEND_FAILURE,
                failed_part=RegionPartKind.COUNTER,
                failed_operation=RegionOperationKind.ZERO_BYTES,
                cleanup_debt_remaining=True,
            ),
        )
    )
    assert allocation[40:] == W4_5_ALLOCATE_ALLOCATION_ERROR_OUTCOMES["backend_counter_zero_debt_1"]
    parsed_alloc = parse_reply(allocation).decode_outcome()
    assert parsed_alloc.tag is DelegatedAllocateReplyTag.ERROR
    assert parsed_alloc.error_kind is RegionControlErrorKind.BACKEND_FAILURE
    assert parsed_alloc.provisional_resource_id == 7
    assert parsed_alloc.cleanup_debt_remaining is True


def test_release_reply_tags_and_unknown_transaction_zero_outcome():
    clean = {
        DelegatedReleaseReplyTag.RELEASED: ProviderReleaseStatus.RELEASED,
        DelegatedReleaseReplyTag.ALREADY_GONE: ProviderReleaseStatus.ALREADY_GONE,
        DelegatedReleaseReplyTag.UNKNOWN_RESOURCE: ProviderReleaseStatus.UNKNOWN_RESOURCE,
    }
    for tag, status in clean.items():
        frame = encode_reply(
            _release_result_reply(
                tag,
                ProviderReleaseResult(provider_resource_id=13, status=status),
            )
        )
        assert frame[40:] == W4_5_RELEASE_CLEAN_OUTCOMES[tag.name]
        parsed = parse_reply(frame).decode_outcome()
        assert parsed.tag is tag
        assert parsed.error_kind is RegionControlErrorKind.NONE
        assert parsed.result is not None
        assert parsed.result.status is status

    incomplete = encode_reply(
        _release_result_reply(
            DelegatedReleaseReplyTag.CLEANUP_INCOMPLETE,
            ProviderReleaseResult(
                provider_resource_id=13,
                status=ProviderReleaseStatus.CLEANUP_INCOMPLETE,
                failures=(
                    ProviderCleanupFailure(
                        part=RegionPartKind.PAYLOAD,
                        backend_operation=RegionOperationKind.RELEASE,
                        typed_cause=RegionCleanupCause.BACKEND_ERROR,
                    ),
                    ProviderCleanupFailure(
                        part=RegionPartKind.COUNTER,
                        backend_operation=RegionOperationKind.ZERO_BYTES,
                        typed_cause=RegionCleanupCause.BACKEND_STATE_MISMATCH,
                    ),
                ),
            ),
        )
    )
    assert incomplete[40:] == W4_5_RELEASE_CLEANUP_INCOMPLETE_OUTCOMES["both"]
    parsed_incomplete = parse_reply(incomplete).decode_outcome()
    assert parsed_incomplete.tag is DelegatedReleaseReplyTag.CLEANUP_INCOMPLETE
    assert parsed_incomplete.error_kind is RegionControlErrorKind.NONE

    error = encode_reply(
        DelegatedReleaseReply(
            tag=DelegatedReleaseReplyTag.ERROR,
            session_instance_id=_SESSION,
            transaction_id=1,
            result=ProviderReleaseResult(provider_resource_id=13, status=ProviderReleaseStatus.RELEASED),
            error_kind=RegionControlErrorKind.INVALID_FIELD_VALUE,
        )
    )
    assert error[40:] == W4_5_RELEASE_ERROR_OUTCOMES["INVALID_FIELD_VALUE"]

    unknown = encode_reply(
        DelegatedReleaseReply(
            tag=DelegatedReleaseReplyTag.UNKNOWN_TRANSACTION,
            session_instance_id=_SESSION,
            transaction_id=1,
        )
    )
    assert unknown[40:] == W5A_UNKNOWN_TRANSACTION_OUTCOME
    assert unknown[40:] == bytes(RELEASE_OUTCOME_BYTES)
    parsed_unknown = parse_reply(unknown).decode_outcome()
    assert parsed_unknown.tag is DelegatedReleaseReplyTag.UNKNOWN_TRANSACTION
    assert parsed_unknown.result is None
    assert parsed_unknown.error_kind is RegionControlErrorKind.NONE


def test_publish_reply_zeros_staging_and_commits_tag_last():
    leftover = encode_request(_allocate_request(), staged_capacity=616)
    assert leftover[256:] != b"" or leftover[40:104]
    committed = encode_reply(_allocated_reply())
    publish_reply(memoryview(leftover), committed)
    assert leftover[256:] == b"\x00" * 360
    assert leftover[:256] == committed
    assert leftover[REPLY_TAG_OFFSET : REPLY_TAG_OFFSET + 4] == int(DelegatedAllocateReplyTag.ALLOCATED).to_bytes(
        4, "little"
    )
    parsed = parse_reply(leftover)
    assert parsed.reply_tag == int(DelegatedAllocateReplyTag.ALLOCATED)
    assert parsed.decode_outcome().tag is DelegatedAllocateReplyTag.ALLOCATED


def test_parse_reply_rejects_nonzero_tail_and_reserved():
    committed = bytearray(encode_reply(_allocated_reply()))
    padded = committed + b"\x00" * 8
    parse_reply(padded)
    padded[-1] = 1
    with pytest.raises(RegionControlError) as tail:
        parse_reply(padded)
    assert _kind(tail.value) is RegionControlErrorKind.RESERVED_NONZERO

    reserved = bytearray(committed)
    struct.pack_into("<I", reserved, 20, 1)
    with pytest.raises(RegionControlError) as header:
        parse_reply(reserved)
    assert _kind(header.value) is RegionControlErrorKind.RESERVED_NONZERO


def test_empty_reply_is_not_a_committed_outcome():
    empty = bytearray(ALLOCATE_REPLY_BYTES)
    struct.pack_into(
        "<QIIII8sQ",
        empty,
        0,
        DELEGATED_REGION_CTRL_MAGIC_VERSION,
        ALLOCATE_REPLY_BYTES,
        0,
        int(DelegatedRegionOperation.DELEGATED_ALLOCATE),
        0,
        _SESSION,
        1,
    )
    envelope = parse_reply(empty)
    assert envelope.reply_tag == 0
    with pytest.raises(RegionControlError) as exc:
        envelope.decode_outcome()
    assert _kind(exc.value) is RegionControlErrorKind.INVALID_FIELD_VALUE


def test_error_kind_lives_only_in_outcome():
    frame = encode_reply(
        DelegatedAllocateReply(
            tag=DelegatedAllocateReplyTag.ERROR,
            session_instance_id=_SESSION,
            transaction_id=1,
            error_kind=RegionControlErrorKind.BAD_MESSAGE_SIZE,
        )
    )
    _magic, _reply_bytes, tag, operation, reserved, _session, _tx = struct.unpack_from("<QIIII8sQ", frame, 0)
    assert tag == int(DelegatedAllocateReplyTag.ERROR)
    assert operation == int(DelegatedRegionOperation.DELEGATED_ALLOCATE)
    assert reserved == 0
    resource, error_kind, failed_part, failed_operation, debt = struct.unpack_from("<QIIII", frame, ALLOCATE_OUTCOME_OFFSET)
    assert error_kind == int(RegionControlErrorKind.BAD_MESSAGE_SIZE)
    assert resource == 0
    assert failed_part == 0
    assert failed_operation == 0
    assert debt == 0
    assert frame[ALLOCATE_PAYLOAD_EXPORT_OFFSET:] == b"\x00" * (
        ALLOCATE_REPLY_BYTES - ALLOCATE_PAYLOAD_EXPORT_OFFSET
    )


def test_store_lifecycle_allocate_error_is_zero_resource_protocol_outcome():
    frame = encode_reply(
        DelegatedAllocateReply(
            tag=DelegatedAllocateReplyTag.ERROR,
            session_instance_id=_SESSION,
            transaction_id=4,
            error_kind=RegionControlErrorKind.STORE_LIFECYCLE,
        )
    )
    parsed = parse_reply(frame).decode_outcome()
    assert parsed.tag is DelegatedAllocateReplyTag.ERROR
    assert parsed.error_kind is RegionControlErrorKind.STORE_LIFECYCLE
    assert parsed.provisional_resource_id == 0
    assert frame[ALLOCATE_OUTCOME_OFFSET : ALLOCATE_OUTCOME_OFFSET + 24] == (
        (0).to_bytes(8, "little") + int(RegionControlErrorKind.STORE_LIFECYCLE).to_bytes(4, "little") + b"\x00" * 12
    )


class _CountingStore:
    def __init__(self, store):
        self._store = store
        self.allocate_calls = 0
        self.release_calls = 0
        self.local_view_calls = 0
        self.release_ids: list[int] = []

    def allocate_and_export(self, spec):
        self.allocate_calls += 1
        return self._store.allocate_and_export(spec)

    def local_view(self, provider_resource_id, part):
        self.local_view_calls += 1
        return self._store.local_view(provider_resource_id, part)

    def release(self, provider_resource_id):
        self.release_calls += 1
        self.release_ids.append(int(provider_resource_id))
        return self._store.release(provider_resource_id)

    def sweep(self):
        return self._store.sweep()


def _decode_allocate(committed: bytes) -> DelegatedAllocateReply:
    outcome = parse_reply(committed).decode_outcome()
    assert isinstance(outcome, DelegatedAllocateReply)
    return outcome


def _decode_release(committed: bytes) -> DelegatedReleaseReply:
    outcome = parse_reply(committed).decode_outcome()
    assert isinstance(outcome, DelegatedReleaseReply)
    return outcome


def test_table_first_allocate_then_exact_duplicate_replays_cache():
    store, factory = _open_store()
    counted = _CountingStore(store)
    table = ProviderTransactionTable()
    request = _allocate_request(transaction_id=1)
    first = table.execute(request, counted)
    assert _decode_allocate(first).tag is DelegatedAllocateReplyTag.ALLOCATED
    assert counted.allocate_calls == 1
    second = table.execute(request, counted)
    assert second == first
    assert counted.allocate_calls == 1
    assert counted.release_calls == 0
    assert factory.world.duplicate_releases == []


def test_table_conflict_and_late_allocate_do_not_call_store():
    store, _factory = _open_store()
    counted = _CountingStore(store)
    table = ProviderTransactionTable()
    first = _decode_allocate(table.execute(_allocate_request(transaction_id=2), counted))
    assert first.tag is DelegatedAllocateReplyTag.ALLOCATED
    assert counted.allocate_calls == 1

    conflict = _decode_allocate(
        table.execute(_allocate_request(transaction_id=2, provider_path=b"L3/L2[1]"), counted)
    )
    assert conflict.tag is DelegatedAllocateReplyTag.ERROR
    assert conflict.error_kind is RegionControlErrorKind.INVALID_FIELD_VALUE
    assert counted.allocate_calls == 1

    late = _decode_allocate(table.execute(_allocate_request(transaction_id=1), counted))
    assert late.tag is DelegatedAllocateReplyTag.ERROR
    assert late.error_kind is RegionControlErrorKind.STORE_LIFECYCLE
    assert counted.allocate_calls == 1


def test_table_missing_release_is_unknown_transaction():
    store, _factory = _open_store()
    counted = _CountingStore(store)
    table = ProviderTransactionTable()
    table.execute(_allocate_request(transaction_id=3), counted)
    missing = _decode_release(table.execute(_release_request(transaction_id=1), counted))
    assert missing.tag is DelegatedReleaseReplyTag.UNKNOWN_TRANSACTION
    assert missing.result is None
    assert counted.release_calls == 0


def test_table_known_release_evicts_and_second_release_is_unknown():
    store, factory = _open_store()
    counted = _CountingStore(store)
    table = ProviderTransactionTable()
    allocated = _decode_allocate(table.execute(_allocate_request(transaction_id=1), counted))
    assert allocated.result is not None
    released = _decode_release(table.execute(_release_request(transaction_id=1), counted))
    assert released.tag is DelegatedReleaseReplyTag.RELEASED
    assert counted.release_calls == 1
    second = _decode_release(table.execute(_release_request(transaction_id=1), counted))
    assert second.tag is DelegatedReleaseReplyTag.UNKNOWN_TRANSACTION
    assert counted.release_calls == 1
    assert factory.world.duplicate_releases == []


def test_table_different_identity_same_projection_allocates_twice():
    store, _factory = _open_store()
    counted = _CountingStore(store)
    table = ProviderTransactionTable()
    first = _decode_allocate(table.execute(_allocate_request(transaction_id=1), counted))
    second = _decode_allocate(table.execute(_allocate_request(transaction_id=2), counted))
    assert first.result is not None and second.result is not None
    assert first.result.provider_resource_id != second.result.provider_resource_id
    assert counted.allocate_calls == 2


def test_table_clean_backend_failure_evicts_without_tombstone():
    factory = FakeShellFactory()
    factory.fail_construct[RegionPartKind.PAYLOAD] = RegionControlError(
        RegionControlErrorKind.BACKEND_FAILURE,
        "payload construct failed",
        failed_part=RegionPartKind.PAYLOAD,
        failed_operation=RegionOperationKind.MATERIALIZE,
    )
    store, _ = _open_store(factory)
    counted = _CountingStore(store)
    table = ProviderTransactionTable()
    failed = _decode_allocate(table.execute(_allocate_request(transaction_id=1), counted))
    assert failed.tag is DelegatedAllocateReplyTag.ERROR
    assert failed.error_kind is RegionControlErrorKind.BACKEND_FAILURE
    assert failed.cleanup_debt_remaining is False
    assert counted.allocate_calls == 1
    late = _decode_allocate(table.execute(_allocate_request(transaction_id=1), counted))
    assert late.error_kind is RegionControlErrorKind.STORE_LIFECYCLE
    assert counted.allocate_calls == 1
    factory.fail_construct.clear()
    retry = _decode_allocate(table.execute(_allocate_request(transaction_id=2), counted))
    assert retry.tag is DelegatedAllocateReplyTag.ALLOCATED
    assert counted.allocate_calls == 2


def test_table_local_view_failure_compensates_once():
    store, factory = _open_store()
    counted = _CountingStore(store)
    real_local_view = counted.local_view

    def _boom(resource_id, part):
        raise RegionControlError(
            RegionControlErrorKind.INTERNAL_INVARIANT,
            "local view failed",
            failed_part=part,
            failed_operation=RegionOperationKind.LOCAL_VIEW,
        )

    counted.local_view = _boom  # type: ignore[method-assign]
    table = ProviderTransactionTable()
    failed = _decode_allocate(table.execute(_allocate_request(transaction_id=1), counted))
    assert failed.tag is DelegatedAllocateReplyTag.ERROR
    assert failed.error_kind is RegionControlErrorKind.INTERNAL_INVARIANT
    assert counted.allocate_calls == 1
    assert counted.release_calls == 1
    assert factory.world.duplicate_releases == []
    counted.local_view = real_local_view  # type: ignore[method-assign]
    late = _decode_allocate(table.execute(_allocate_request(transaction_id=1), counted))
    assert late.error_kind is RegionControlErrorKind.STORE_LIFECYCLE
    assert counted.allocate_calls == 1


def test_table_already_gone_requires_known_store_result():
    store, _factory = _open_store()
    counted = _CountingStore(store)
    table = ProviderTransactionTable()
    allocated = _decode_allocate(table.execute(_allocate_request(transaction_id=1), counted))
    assert allocated.result is not None
    store.release(allocated.result.provider_resource_id)
    gone = _decode_release(table.execute(_release_request(transaction_id=1), counted))
    assert gone.tag is DelegatedReleaseReplyTag.ALREADY_GONE
    assert counted.release_calls == 1
    missing = _decode_release(table.execute(_release_request(transaction_id=1), counted))
    assert missing.tag is DelegatedReleaseReplyTag.UNKNOWN_TRANSACTION
    assert counted.release_calls == 1


def test_handler_invalid_request_does_not_advance_waterline():
    store, _factory = _open_store()
    counted = _CountingStore(store)
    table = ProviderTransactionTable()
    staged = encode_request(_allocate_request(transaction_id=1), staged_capacity=256)
    staged[REQUEST_HEADER_BYTES + 12 : REQUEST_HEADER_BYTES + 16] = (1).to_bytes(4, "little")
    handle_terminal_delegated_region(memoryview(staged), table, counted)
    invalid = parse_reply(staged).decode_outcome()
    assert invalid.tag is DelegatedAllocateReplyTag.ERROR
    assert invalid.error_kind is RegionControlErrorKind.RESERVED_NONZERO
    assert counted.allocate_calls == 0

    valid = encode_request(_allocate_request(transaction_id=1), staged_capacity=256)
    handle_terminal_delegated_region(memoryview(valid), table, counted)
    allocated = parse_reply(valid).decode_outcome()
    assert allocated.tag is DelegatedAllocateReplyTag.ALLOCATED
    assert counted.allocate_calls == 1


def test_sweep_then_table_release_does_not_repeat_backend_release():
    store, factory = _open_store()
    counted = _CountingStore(store)
    table = ProviderTransactionTable()
    table.execute(_allocate_request(transaction_id=1), counted)
    payload_releases = [call for call in factory.world.calls if call == (RegionPartKind.PAYLOAD, "release_once")]
    counter_releases = [call for call in factory.world.calls if call == (RegionPartKind.COUNTER, "release_once")]
    assert payload_releases == []
    assert counter_releases == []
    counted.sweep()
    assert factory.world.calls.count((RegionPartKind.PAYLOAD, "release_once")) == 1
    assert factory.world.calls.count((RegionPartKind.COUNTER, "release_once")) == 1
    gone = _decode_release(table.execute(_release_request(transaction_id=1), counted))
    assert gone.tag is DelegatedReleaseReplyTag.ERROR
    assert gone.error_kind is RegionControlErrorKind.STORE_LIFECYCLE
    assert counted.release_calls == 1
    assert factory.world.calls.count((RegionPartKind.PAYLOAD, "release_once")) == 1
    assert factory.world.calls.count((RegionPartKind.COUNTER, "release_once")) == 1
    assert factory.world.duplicate_releases == []


def test_inspect_delegated_route_is_strict_prefix_and_terminates_at_l2():
    l3 = _inspect_delegated_route("L3", b"L3/L2[1]")
    assert l3.child_id == 1
    assert l3.child_level == 2
    l4 = _inspect_delegated_route("L4", b"L4/L3[2]/L2[0]")
    assert l4.child_id == 2
    assert l4.child_level == 3
    hop = _inspect_delegated_route("L4/L3[2]", b"L4/L3[2]/L2[0]")
    assert hop.child_id == 0
    assert hop.child_level == 2
    deeper = _inspect_delegated_route("L5", b"L5/L4[1]/L3[0]/L2[3]")
    assert deeper.child_id == 1
    assert deeper.child_level == 4
    with pytest.raises(RegionControlError, match="strict prefix"):
        _inspect_delegated_route("L3/L2[0]", b"L3/L2[0]")
    with pytest.raises(RegionControlError, match="not a prefix"):
        _inspect_delegated_route("L4/L3[1]", b"L4/L3[2]/L2[0]")
    with pytest.raises(RegionControlError, match="terminate at the selected L2"):
        _inspect_delegated_route("L4/L3[2]", b"L4/L3[2]/L2[0]/L1[0]")


def test_hop_staging_copy_uses_max_of_request_and_fixed_reply():
    allocate = encode_request(_allocate_request(), staged_capacity=ALLOCATE_REQUEST_HARD_CEILING)
    allocate_envelope = parse_request(allocate)
    allocate_hop = _hop_staging_copy(allocate_envelope)
    assert len(allocate_hop) == max(allocate_envelope.request_bytes, ALLOCATE_REPLY_BYTES)
    assert allocate_hop[: allocate_envelope.request_bytes] == allocate_envelope.frame
    assert not any(allocate_hop[allocate_envelope.request_bytes :])
    release = encode_request(_release_request(), staged_capacity=RELEASE_REQUEST_HARD_CEILING)
    release_envelope = parse_request(release)
    release_hop = _hop_staging_copy(release_envelope)
    assert len(release_hop) == max(release_envelope.request_bytes, RELEASE_REPLY_BYTES)
    assert release_hop[: release_envelope.request_bytes] == release_envelope.frame
    assert not any(release_hop[release_envelope.request_bytes :])
