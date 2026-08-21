# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Unit tests for the private provider-region control wire."""

from __future__ import annotations

import struct
from multiprocessing.shared_memory import SharedMemory

import pytest
from simpler.buffer import BackendKind
from simpler.comm_provider import (
    PosixShmImport,
    ProviderCleanupFailure,
    ProviderRegionStore,
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
from simpler.comm_provider_control import (
    ALLOCATE_COUNTER_EXPORT_OFFSET,
    ALLOCATE_COUNTER_VIEW_OFFSET,
    ALLOCATE_PAYLOAD_EXPORT_OFFSET,
    ALLOCATE_PAYLOAD_VIEW_OFFSET,
    ALLOCATE_REPLY_BYTES,
    ALLOCATE_REQUEST_BYTES,
    COMMIT_TAG_OFFSET,
    PROVIDER_REGION_CTRL_MAGIC_VERSION,
    RELEASE_COUNTER_FAILURE_OFFSET,
    RELEASE_PAYLOAD_FAILURE_OFFSET,
    RELEASE_REPLY_BYTES,
    RELEASE_REQUEST_BYTES,
    AllocateReplyTag,
    ImportCapabilityWireKind,
    ProviderAllocateClient,
    ProviderReleaseClient,
    RegionControlProtocolError,
    ReleaseReplyTag,
    decode_allocate_reply,
    decode_allocate_request,
    decode_release_reply,
    decode_release_request,
    encode_allocate_allocation_error_reply,
    encode_allocate_request,
    encode_allocate_request_error_reply,
    encode_allocate_success_reply,
    encode_release_error_reply,
    encode_release_request,
    encode_release_result_reply,
    handle_ctrl_region_allocate,
    handle_ctrl_region_release,
    peek_allocate_reply_resource_id,
)

from tests.ut.py.test_worker.test_comm_provider import FakeShellFactory, _allocation_spec, _sim_context

_OLD_MAGIC = 0x4C334C3200020000


def _posix_part(logical_bytes: int, name: str, mapping_bytes: int | None = None) -> RegionPartExportDescriptor:
    return RegionPartExportDescriptor(
        planned_backing_kind=BackendKind.VMM_WINDOW,
        logical_bytes=logical_bytes,
        mapping_bytes=logical_bytes if mapping_bytes is None else mapping_bytes,
        import_capability=PosixShmImport(shm_name=name),
    )


def _vmm_part(logical_bytes: int, handle: int = 9, mapping_bytes: int | None = None) -> RegionPartExportDescriptor:
    return RegionPartExportDescriptor(
        planned_backing_kind=BackendKind.VMM_WINDOW,
        logical_bytes=logical_bytes,
        mapping_bytes=logical_bytes if mapping_bytes is None else mapping_bytes,
        import_capability=VmmShareableHandleImport(device_id=2, shareable_handle=handle),
    )


def _success_result() -> tuple[RegionAllocationResult, RegionPartLocalView, RegionPartLocalView]:
    result = RegionAllocationResult(
        provider_resource_id=11,
        export_descriptor=RegionExportDescriptor(payload=_posix_part(64, "/pto_payload_a"), counter=_vmm_part(8, 21)),
    )
    payload = RegionPartLocalView(part=RegionPartKind.PAYLOAD, local_base=0x1000, logical_bytes=64)
    counter = RegionPartLocalView(part=RegionPartKind.COUNTER, local_base=0x2000, logical_bytes=8)
    return result, payload, counter


def test_allocate_request_is_exactly_48_bytes_at_normative_offsets():
    spec = _allocation_spec()
    buf = bytearray(ALLOCATE_REQUEST_BYTES)
    encode_allocate_request(buf, spec)
    magic, size, payload_kind, counter_kind, reserved20, payload_bytes, counter_bytes, reserved40 = struct.unpack_from(
        "<QIIIIQQQ", buf, 0
    )
    assert magic == PROVIDER_REGION_CTRL_MAGIC_VERSION
    assert magic != _OLD_MAGIC
    assert size == 48
    assert payload_kind == int(BackendKind.VMM_WINDOW)
    assert counter_kind == int(BackendKind.VMM_WINDOW)
    assert reserved20 == 0
    assert payload_bytes == 64
    assert counter_bytes == 8
    assert reserved40 == 0
    assert decode_allocate_request(buf) == spec


def test_codec_uses_wire_prefix_of_larger_transport_mapping():
    spec = _allocation_spec()
    req = bytearray(4096)
    encode_allocate_request(req, spec)
    assert req[ALLOCATE_REQUEST_BYTES:] == b"\x00" * (4096 - ALLOCATE_REQUEST_BYTES)
    assert decode_allocate_request(req) == spec
    req[ALLOCATE_REQUEST_BYTES] = 1
    assert decode_allocate_request(req) == spec

    release_req = bytearray(4096)
    encode_release_request(release_req, 13)
    assert decode_release_request(release_req) == 13
    release_req[RELEASE_REQUEST_BYTES] = 1
    assert decode_release_request(release_req) == 13

    result, payload, counter = _success_result()
    reply = bytearray(4096)
    encode_allocate_success_reply(reply, result, payload, counter)
    tag, decoded, decoded_payload, decoded_counter, error = decode_allocate_reply(reply)
    assert tag is AllocateReplyTag.SUCCESS
    assert error is None
    assert decoded == result
    assert decoded_payload == payload
    assert decoded_counter == counter
    reply[ALLOCATE_REPLY_BYTES] = 1
    tag, decoded, decoded_payload, decoded_counter, error = decode_allocate_reply(reply)
    assert tag is AllocateReplyTag.SUCCESS
    assert decoded == result

    release_reply = bytearray(4096)
    encode_release_result_reply(
        release_reply,
        ProviderReleaseResult(provider_resource_id=13, status=ProviderReleaseStatus.RELEASED),
    )
    decoded_release = decode_release_reply(release_reply)
    assert isinstance(decoded_release, ProviderReleaseResult)
    assert decoded_release.provider_resource_id == 13
    assert decoded_release.status is ProviderReleaseStatus.RELEASED
    release_reply[RELEASE_REPLY_BYTES] = 1
    decoded_release = decode_release_reply(release_reply)
    assert isinstance(decoded_release, ProviderReleaseResult)
    assert decoded_release.status is ProviderReleaseStatus.RELEASED

    assert peek_allocate_reply_resource_id(reply) == 11
    assert peek_allocate_reply_resource_id(bytearray(ALLOCATE_REPLY_BYTES - 1)) == 0


@pytest.mark.parametrize(
    ("decoder", "size"),
    [
        (decode_allocate_request, ALLOCATE_REQUEST_BYTES),
        (decode_allocate_reply, ALLOCATE_REPLY_BYTES),
        (decode_release_request, RELEASE_REQUEST_BYTES),
        (decode_release_reply, RELEASE_REPLY_BYTES),
    ],
)
def test_codec_rejects_transport_mapping_smaller_than_wire_frame(decoder, size):
    with pytest.raises(RegionControlError) as exc_info:
        decoder(bytearray(size - 1))
    assert exc_info.value.kind is RegionControlErrorKind.BAD_MESSAGE_SIZE


def test_allocate_reply_exports_and_views_use_offsets_40_112_184_208():
    result, payload, counter = _success_result()
    buf = bytearray(ALLOCATE_REPLY_BYTES)
    encode_allocate_success_reply(buf, result, payload, counter)
    assert struct.unpack_from("<I", buf, 8)[0] == 232
    assert struct.unpack_from("<I", buf, COMMIT_TAG_OFFSET)[0] == AllocateReplyTag.SUCCESS
    assert struct.unpack_from("<Q", buf, 16)[0] == 11
    payload_export = struct.unpack_from("<IIQQII40s", buf, ALLOCATE_PAYLOAD_EXPORT_OFFSET)
    counter_export = struct.unpack_from("<IIQQII40s", buf, ALLOCATE_COUNTER_EXPORT_OFFSET)
    payload_view = struct.unpack_from("<IIQQ", buf, ALLOCATE_PAYLOAD_VIEW_OFFSET)
    counter_view = struct.unpack_from("<IIQQ", buf, ALLOCATE_COUNTER_VIEW_OFFSET)
    assert payload_export[1] == ImportCapabilityWireKind.POSIX_SHM
    assert payload_export[4] == len("/pto_payload_a")
    assert bytes(payload_export[6])[: payload_export[4]] == b"/pto_payload_a"
    assert counter_export[1] == ImportCapabilityWireKind.VMM_SHAREABLE_HANDLE
    assert counter_export[4] == 16
    assert struct.unpack_from("<iIQ", counter_export[6], 0) == (2, 0, 21)
    assert payload_view == (RegionPartKind.PAYLOAD, 0, 0x1000, 64)
    assert counter_view == (RegionPartKind.COUNTER, 0, 0x2000, 8)
    tag, decoded, decoded_payload, decoded_counter, error = decode_allocate_reply(buf)
    assert tag is AllocateReplyTag.SUCCESS
    assert error is None
    assert decoded == result
    assert decoded_payload == payload
    assert decoded_counter == counter


def test_allocate_success_writes_commit_tag_last(monkeypatch):
    from simpler import comm_provider_control as control

    body_ready: list[bool] = []
    original = control._publish_tag

    def _spy(view, tag):
        body_ready.append(
            bytes(view[ALLOCATE_PAYLOAD_EXPORT_OFFSET : ALLOCATE_PAYLOAD_EXPORT_OFFSET + 4]) != b"\x00\x00\x00\x00"
        )
        original(view, tag)

    monkeypatch.setattr(control, "_publish_tag", _spy)
    result, payload, counter = _success_result()
    buf = bytearray(ALLOCATE_REPLY_BYTES)
    encode_allocate_success_reply(buf, result, payload, counter)
    assert body_ready == [True]
    assert struct.unpack_from("<I", buf, COMMIT_TAG_OFFSET)[0] == AllocateReplyTag.SUCCESS


def test_allocate_reply_rejects_old_version_and_missing_commit():
    buf = bytearray(ALLOCATE_REPLY_BYTES)
    result, payload, counter = _success_result()
    encode_allocate_success_reply(buf, result, payload, counter)
    struct.pack_into("<Q", buf, 0, _OLD_MAGIC)
    with pytest.raises(RegionControlError) as exc_info:
        decode_allocate_reply(buf)
    assert exc_info.value.kind is RegionControlErrorKind.BAD_MAGIC_VERSION
    empty = bytearray(ALLOCATE_REPLY_BYTES)
    with pytest.raises(RegionControlProtocolError) as empty_info:
        decode_allocate_reply(empty)
    assert "commit" in empty_info.value.message


@pytest.mark.parametrize(
    "kind",
    [
        RegionControlErrorKind.BAD_MAGIC_VERSION,
        RegionControlErrorKind.BAD_MESSAGE_SIZE,
        RegionControlErrorKind.INVALID_ENUM_VALUE,
        RegionControlErrorKind.RESERVED_NONZERO,
        RegionControlErrorKind.INVALID_FIELD_VALUE,
    ],
)
def test_allocate_request_error_has_zero_id_and_zero_sections(kind):
    buf = bytearray(ALLOCATE_REPLY_BYTES)
    encode_allocate_request_error_reply(buf, kind)
    tag, result, payload, counter, error = decode_allocate_reply(buf)
    assert tag is AllocateReplyTag.REQUEST_ERROR
    assert result is None and payload is None and counter is None
    assert isinstance(error, RegionControlError)
    assert error.kind is kind
    assert struct.unpack_from("<Q", buf, 16)[0] == 0
    assert buf[ALLOCATE_PAYLOAD_EXPORT_OFFSET:] == b"\x00" * (ALLOCATE_REPLY_BYTES - ALLOCATE_PAYLOAD_EXPORT_OFFSET)


def test_allocate_allocation_error_carries_provisional_id_and_zero_sections():
    buf = bytearray(ALLOCATE_REPLY_BYTES)
    error = RegionAllocationError(
        provisional_resource_id=7,
        control_kind=RegionControlErrorKind.BACKEND_FAILURE,
        failed_part=RegionPartKind.COUNTER,
        failed_operation=RegionOperationKind.ZERO_BYTES,
        cleanup_debt_remaining=True,
    )
    encode_allocate_allocation_error_reply(buf, error)
    tag, result, payload, counter, decoded = decode_allocate_reply(buf)
    assert tag is AllocateReplyTag.ALLOCATION_ERROR
    assert result is None
    assert isinstance(decoded, RegionAllocationError)
    assert decoded.provisional_resource_id == 7
    assert decoded.cleanup_debt_remaining is True
    assert decoded.failed_part is RegionPartKind.COUNTER
    assert buf[ALLOCATE_PAYLOAD_EXPORT_OFFSET:] == b"\x00" * (ALLOCATE_REPLY_BYTES - ALLOCATE_PAYLOAD_EXPORT_OFFSET)


def test_release_request_and_reply_use_offsets_32_and_40():
    req = bytearray(RELEASE_REQUEST_BYTES)
    encode_release_request(req, 13)
    magic, size, reserved, resource_id = struct.unpack_from("<QIIQ", req, 0)
    assert magic == PROVIDER_REGION_CTRL_MAGIC_VERSION
    assert size == 24
    assert reserved == 0
    assert resource_id == 13
    assert decode_release_request(req) == 13

    reply = bytearray(RELEASE_REPLY_BYTES)
    result = ProviderReleaseResult(
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
                backend_operation=RegionOperationKind.RELEASE,
                typed_cause=RegionCleanupCause.INTERRUPTED,
            ),
        ),
    )
    encode_release_result_reply(reply, result)
    assert struct.unpack_from("<I", reply, COMMIT_TAG_OFFSET)[0] == ReleaseReplyTag.CLEANUP_INCOMPLETE
    assert struct.unpack_from("<I", reply, 24)[0] == 3
    assert struct.unpack_from("<II", reply, RELEASE_PAYLOAD_FAILURE_OFFSET) == (
        RegionOperationKind.RELEASE,
        RegionCleanupCause.BACKEND_ERROR,
    )
    assert struct.unpack_from("<II", reply, RELEASE_COUNTER_FAILURE_OFFSET) == (
        RegionOperationKind.RELEASE,
        RegionCleanupCause.INTERRUPTED,
    )
    decoded = decode_release_reply(reply)
    assert decoded == result


def test_release_error_rejects_nonzero_failure_mask():
    buf = bytearray(RELEASE_REPLY_BYTES)
    encode_release_error_reply(buf, RegionControlErrorKind.STORE_LIFECYCLE, provider_resource_id=4)
    struct.pack_into("<I", buf, 24, 1)
    with pytest.raises(RegionControlError) as exc_info:
        decode_release_reply(buf)
    assert exc_info.value.kind is RegionControlErrorKind.INVALID_FIELD_VALUE


def test_inactive_capability_bytes_must_be_zero():
    result, payload, counter = _success_result()
    buf = bytearray(ALLOCATE_REPLY_BYTES)
    encode_allocate_success_reply(buf, result, payload, counter)
    buf[ALLOCATE_PAYLOAD_EXPORT_OFFSET + 32 + 20] = 1
    with pytest.raises(RegionControlError) as exc_info:
        decode_allocate_reply(buf)
    assert exc_info.value.kind is RegionControlErrorKind.RESERVED_NONZERO


class _LoopbackMailbox:
    def __init__(self, store: ProviderRegionStore) -> None:
        self.store = store
        self.allocate_calls = 0
        self.release_calls = 0

    def control_region_allocate(self, worker_id: int, request_shm_name: str, reply_shm_name: str) -> None:
        del worker_id
        self.allocate_calls += 1
        req = SharedMemory(name=request_shm_name)
        reply = SharedMemory(name=reply_shm_name)
        assert req.buf is not None
        assert reply.buf is not None
        req_view = memoryview(req.buf)[:ALLOCATE_REQUEST_BYTES]
        reply_view = memoryview(reply.buf)[:ALLOCATE_REPLY_BYTES]
        try:
            handle_ctrl_region_allocate(req_view, reply_view, self.store)
        finally:
            req_view.release()
            reply_view.release()
            req.close()
            reply.close()

    def control_region_release(self, worker_id: int, request_shm_name: str, reply_shm_name: str) -> None:
        del worker_id
        self.release_calls += 1
        req = SharedMemory(name=request_shm_name)
        reply = SharedMemory(name=reply_shm_name)
        assert req.buf is not None
        assert reply.buf is not None
        req_view = memoryview(req.buf)[:RELEASE_REQUEST_BYTES]
        reply_view = memoryview(reply.buf)[:RELEASE_REPLY_BYTES]
        try:
            handle_ctrl_region_release(req_view, reply_view, self.store)
        finally:
            req_view.release()
            reply_view.release()
            req.close()
            reply.close()


def test_handler_roundtrip_calls_store_once_and_clients_are_one_shot():
    factory = FakeShellFactory()
    store = ProviderRegionStore(_sim_context(), _shell_factory=factory)
    mailbox = _LoopbackMailbox(store)
    allocated, payload, counter = ProviderAllocateClient(mailbox, 3).allocate(_allocation_spec())
    assert mailbox.allocate_calls == 1
    assert allocated.provider_resource_id == 1
    assert payload.logical_bytes == 64
    assert counter.logical_bytes == 8
    client = ProviderReleaseClient(mailbox, 3)
    released = client.release(allocated.provider_resource_id)
    assert released.status is ProviderReleaseStatus.RELEASED
    assert mailbox.release_calls == 1
    again = client.release(allocated.provider_resource_id)
    assert again.status is ProviderReleaseStatus.RELEASED
    assert mailbox.release_calls == 1


def test_handler_request_error_does_not_call_store():
    factory = FakeShellFactory()
    store = ProviderRegionStore(_sim_context(), _shell_factory=factory)
    req = bytearray(ALLOCATE_REQUEST_BYTES)
    encode_allocate_request(req, _allocation_spec())
    struct.pack_into("<Q", req, 0, _OLD_MAGIC)
    reply = bytearray(ALLOCATE_REPLY_BYTES)
    handle_ctrl_region_allocate(memoryview(req), memoryview(reply), store)
    tag, _result, _payload, _counter, error = decode_allocate_reply(reply)
    assert tag is AllocateReplyTag.REQUEST_ERROR
    assert isinstance(error, RegionControlError)
    assert error.kind is RegionControlErrorKind.BAD_MAGIC_VERSION
    assert factory.world.calls == []


def test_publication_failure_releases_the_active_resource_once():
    factory = FakeShellFactory()
    store = ProviderRegionStore(_sim_context(), _shell_factory=factory)
    req = bytearray(ALLOCATE_REQUEST_BYTES)
    encode_allocate_request(req, _allocation_spec())
    reply = bytearray(ALLOCATE_REPLY_BYTES)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("publish failed")

    from simpler import comm_provider_control as control

    original = control.encode_allocate_success_reply
    control.encode_allocate_success_reply = _boom  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="publish failed"):
            handle_ctrl_region_allocate(memoryview(req), memoryview(reply), store)
    finally:
        control.encode_allocate_success_reply = original  # type: ignore[method-assign]
    assert factory.payloads[0].release_count == 1
    assert factory.counters[0].release_count == 1
    assert store.release(1).status is ProviderReleaseStatus.ALREADY_GONE


def test_release_client_terminalizes_transport_failure_without_retry():
    class _BoomMailbox:
        def __init__(self) -> None:
            self.calls = 0

        def control_region_allocate(self, worker_id: int, request_shm_name: str, reply_shm_name: str) -> None:
            raise AssertionError("allocate must not be called")

        def control_region_release(self, worker_id: int, request_shm_name: str, reply_shm_name: str) -> None:
            self.calls += 1
            raise RuntimeError("mailbox down")

    mailbox = _BoomMailbox()
    client = ProviderReleaseClient(mailbox, 1)
    with pytest.raises(RegionControlProtocolError):
        client.release(4)
    with pytest.raises(RegionControlProtocolError):
        client.release(4)
    assert mailbox.calls == 1


def test_allocate_client_transport_failure_does_not_call_store_or_release():
    class _BoomMailbox:
        def __init__(self) -> None:
            self.allocate_calls = 0

        def control_region_allocate(self, worker_id: int, request_shm_name: str, reply_shm_name: str) -> None:
            del worker_id, request_shm_name, reply_shm_name
            self.allocate_calls += 1
            raise RuntimeError("mailbox down")

        def control_region_release(self, worker_id: int, request_shm_name: str, reply_shm_name: str) -> None:
            del worker_id, request_shm_name, reply_shm_name
            raise AssertionError("allocate transport failure must not issue release")

    mailbox = _BoomMailbox()
    client = ProviderAllocateClient(mailbox, 1)
    with pytest.raises(RuntimeError, match="mailbox down"):
        client.allocate(_allocation_spec())
    assert mailbox.allocate_calls == 1
    assert client.committed_resource_id == 0


def test_allocate_client_missing_commit_is_a_decode_failure():
    class _SilentMailbox:
        def __init__(self) -> None:
            self.allocate_calls = 0

        def control_region_allocate(self, worker_id: int, request_shm_name: str, reply_shm_name: str) -> None:
            del worker_id, request_shm_name, reply_shm_name
            self.allocate_calls += 1

        def control_region_release(self, worker_id: int, request_shm_name: str, reply_shm_name: str) -> None:
            del worker_id, request_shm_name, reply_shm_name
            raise AssertionError("uncommitted allocate must not issue release")

    mailbox = _SilentMailbox()
    client = ProviderAllocateClient(mailbox, 1)
    with pytest.raises(RegionControlProtocolError, match="commit"):
        client.allocate(_allocation_spec())
    assert mailbox.allocate_calls == 1
    assert client.committed_resource_id == 0


def test_allocate_client_old_version_reply_is_a_decode_failure():
    class _OldVersionMailbox:
        def __init__(self) -> None:
            self.allocate_calls = 0

        def control_region_allocate(self, worker_id: int, request_shm_name: str, reply_shm_name: str) -> None:
            del worker_id, request_shm_name
            self.allocate_calls += 1
            reply = SharedMemory(name=reply_shm_name)
            try:
                assert reply.buf is not None
                struct.pack_into("<Q", reply.buf, 0, _OLD_MAGIC)
                struct.pack_into("<I", reply.buf, COMMIT_TAG_OFFSET, int(AllocateReplyTag.SUCCESS))
            finally:
                reply.close()

        def control_region_release(self, worker_id: int, request_shm_name: str, reply_shm_name: str) -> None:
            del worker_id, request_shm_name, reply_shm_name
            raise AssertionError("old-version allocate must not issue release")

    mailbox = _OldVersionMailbox()
    client = ProviderAllocateClient(mailbox, 1)
    with pytest.raises(RegionControlError) as exc_info:
        client.allocate(_allocation_spec())
    assert exc_info.value.kind is RegionControlErrorKind.BAD_MAGIC_VERSION
    assert mailbox.allocate_calls == 1
    assert client.committed_resource_id == 0


def test_allocate_client_transport_failure_does_not_call_store():
    class _BoomMailbox:
        def __init__(self) -> None:
            self.calls = 0

        def control_region_allocate(self, worker_id: int, request_shm_name: str, reply_shm_name: str) -> None:
            del worker_id, request_shm_name, reply_shm_name
            self.calls += 1
            raise RuntimeError("mailbox down")

        def control_region_release(self, worker_id: int, request_shm_name: str, reply_shm_name: str) -> None:
            raise AssertionError("release must not be called")

    mailbox = _BoomMailbox()
    client = ProviderAllocateClient(mailbox, 1)
    with pytest.raises(RuntimeError, match="mailbox down"):
        client.allocate(_allocation_spec())
    assert mailbox.calls == 1


def test_allocate_client_uncommitted_reply_does_not_call_store_again():
    class _EmptyReplyMailbox:
        def __init__(self) -> None:
            self.calls = 0

        def control_region_allocate(self, worker_id: int, request_shm_name: str, reply_shm_name: str) -> None:
            del worker_id, request_shm_name, reply_shm_name
            self.calls += 1

        def control_region_release(self, worker_id: int, request_shm_name: str, reply_shm_name: str) -> None:
            raise AssertionError("release must not be called")

    mailbox = _EmptyReplyMailbox()
    client = ProviderAllocateClient(mailbox, 1)
    with pytest.raises(RegionControlProtocolError, match="commit"):
        client.allocate(_allocation_spec())
    with pytest.raises(RegionControlProtocolError, match="commit"):
        client.allocate(_allocation_spec())
    assert mailbox.calls == 2
