# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for simpler.buffer_handle: identity/descriptor pack-unpack + create/import round trip."""

import ctypes
from dataclasses import replace

import pytest
from _task_interface import (
    BUFFER_HANDLE_DESCRIPTOR_BYTES,
    BUFFER_REF_BYTES,
    CANONICAL_IDENTITY_BYTES,
    OWNER_INSTANCE_ID_BYTES,
    DataType,
    materialize_bufferref_blob,
    read_args_from_blob,
)
from simpler.buffer_handle import (
    AccessMode,
    AddressSpace,
    BackendKind,
    BufferHandleDescriptor,
    CanonicalIdentity,
    ImportRegistry,
    Tensor,
    create_host_shared_buffer,
    intern_worker_path,
    mint_owner_instance_id,
    pack_bufferref_blob,
    re_export,
    wrap_device_malloc,
    wrap_fork_inherited,
)

_OID = bytes(range(0xA0, 0xA0 + OWNER_INSTANCE_ID_BYTES))


def _identity(oid=_OID, buffer_id=7, generation=2):
    return CanonicalIdentity(oid, buffer_id, generation)


def test_identity_roundtrip():
    ident = _identity()
    raw = ident.pack()
    assert len(raw) == CANONICAL_IDENTITY_BYTES
    assert CanonicalIdentity.unpack(raw) == ident


def test_identity_is_fixed_length_no_length_field():
    # The structural guarantee: nothing inside the identity bounds a read, so a hostile blob cannot
    # steer a hash or a compare past the struct. Any CANONICAL_IDENTITY_BYTES decode either yields an
    # identity or is rejected on a value check — never an out-of-bounds read.
    for raw in (b"\xff" * CANONICAL_IDENTITY_BYTES, bytes(range(CANONICAL_IDENTITY_BYTES))):
        try:
            ident = CanonicalIdentity.unpack(raw)
        except ValueError:
            continue
        assert len(ident.pack()) == CANONICAL_IDENTITY_BYTES


def test_identity_rejects_bad_oid_width():
    with pytest.raises(ValueError):
        CanonicalIdentity(b"\x00" * (OWNER_INSTANCE_ID_BYTES + 1), 1, 1)


def test_identity_rejects_reserved_generation_zero():
    raw = bytearray(_identity().pack())
    raw[16:20] = (0).to_bytes(4, "little")  # generation field
    with pytest.raises(ValueError, match="generation 0"):
        CanonicalIdentity.unpack(bytes(raw))


def test_identity_padding_does_not_perturb_the_key():
    # Two decodes of the same backing must key identically even if the wire padding differs, or the
    # same buffer would land in two import-registry / dependency buckets.
    clean = bytearray(_identity().pack())
    dirty = bytearray(clean)
    dirty[20:32] = b"\xa5" * 12  # _pad
    assert CanonicalIdentity.unpack(bytes(clean)) == CanonicalIdentity.unpack(bytes(dirty))
    assert hash(CanonicalIdentity.unpack(bytes(clean))) == hash(CanonicalIdentity.unpack(bytes(dirty)))


def test_identity_distinguishes_generation_and_incarnation():
    a = _identity()
    assert a != _identity(generation=a.generation + 1)  # ABA
    assert a != _identity(oid=bytes(range(1, 1 + OWNER_INSTANCE_ID_BYTES)))  # different incarnation
    assert a != _identity(buffer_id=a.buffer_id + 1)


def test_descriptor_roundtrip_host_and_device():
    host = BufferHandleDescriptor(
        identity=_identity(),
        address_space=AddressSpace.HOST,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.POSIX_SHM,
        nbytes=4096,
        body=b"psm_deadbeef",
    )
    raw = host.pack()
    assert len(raw) == BUFFER_HANDLE_DESCRIPTOR_BYTES
    assert BufferHandleDescriptor.unpack(raw) == host

    dev = BufferHandleDescriptor(
        identity=_identity(buffer_id=99),
        address_space=AddressSpace.DEVICE,
        access=AccessMode.READ,
        backend_kind=BackendKind.VMM_WINDOW,
        nbytes=1 << 20,
        body=(0x7F00ABCD).to_bytes(8, "little"),
    )
    assert BufferHandleDescriptor.unpack(dev.pack()) == dev


def test_descriptor_rejects_bad_magic():
    raw = bytearray(
        BufferHandleDescriptor(
            identity=_identity(),
            address_space=AddressSpace.HOST,
            access=AccessMode.READWRITE,
            backend_kind=BackendKind.POSIX_SHM,
            nbytes=8,
        ).pack()
    )
    raw[0] = raw[0] + 1  # corrupt the leading sentinel (u16 @ offset 0)
    with pytest.raises(ValueError, match="magic"):
        BufferHandleDescriptor.unpack(bytes(raw))


def test_descriptor_rejects_body_len_past_the_array():
    raw = bytearray(
        BufferHandleDescriptor(
            identity=_identity(),
            address_space=AddressSpace.HOST,
            access=AccessMode.READWRITE,
            backend_kind=BackendKind.POSIX_SHM,
            nbytes=8,
            body=b"psm_x",
        ).pack()
    )
    raw[52:54] = (0xFFFF).to_bytes(2, "little")  # body_len field
    with pytest.raises(ValueError, match="body_len"):
        BufferHandleDescriptor.unpack(bytes(raw))


def test_worker_path_is_diagnostic_and_survives_an_unknown_id():
    h = BufferHandleDescriptor(
        identity=_identity(),
        address_space=AddressSpace.HOST,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.POSIX_SHM,
        nbytes=8,
        owner_worker_path_id=intern_worker_path("L4/L3[2]"),
    )
    assert h.owner_worker_path == "L4/L3[2]"
    assert BufferHandleDescriptor.unpack(h.pack()) == h
    # An id minted in another process has no local text, and that is not an error.
    foreign = replace(h, owner_worker_path_id=99_999)
    assert foreign.owner_worker_path == "<path#99999>"
    # The path takes no part in identity: two handles differing only by path are the same backing.
    assert replace(h, owner_worker_path_id=0).identity == h.identity


def test_descriptor_rejects_oversized_body():
    with pytest.raises(ValueError, match="body"):
        BufferHandleDescriptor(
            identity=_identity(),
            address_space=AddressSpace.HOST,
            access=AccessMode.READWRITE,
            backend_kind=BackendKind.POSIX_SHM,
            nbytes=8,
            body=b"x" * 200,  # > DESC_MAX_BYTES
        ).pack()


def test_create_export_import_resolve_zero_copy():
    oid = mint_owner_instance_id()
    handle = create_host_shared_buffer(nbytes=256, owner_instance_id=oid, buffer_id=1, owner_worker_path="L4")
    reg = ImportRegistry()
    try:
        assert handle.backend_kind == BackendKind.POSIX_SHM
        imported = reg.materialize(handle.to_descriptor().pack())
        assert reg.resolve(handle.identity).base == imported.base
        assert reg.materialize(handle.to_descriptor()).base == imported.base  # map-once: same mapping
        assert imported.nbytes == 256
        owner_shm = handle.shm
        consumer_shm = imported.shm
        assert owner_shm is not None
        assert consumer_shm is not None
        owner_buf = owner_shm.buf
        consumer_buf = consumer_shm.buf
        assert owner_buf is not None
        assert consumer_buf is not None
        owner_buf[:4] = b"\xde\xad\xbe\xef"
        assert bytes(consumer_buf[:4]) == b"\xde\xad\xbe\xef"
    finally:
        reg.close()
        handle.close()


def test_resolve_unregistered_raises():
    reg = ImportRegistry()
    with pytest.raises(KeyError):
        reg.resolve(_identity())


def test_tensor_full_view_is_contiguous():
    oid = mint_owner_instance_id()
    h = create_host_shared_buffer(nbytes=1024, owner_instance_id=oid, buffer_id=1)
    try:
        # handle.tensor(shape, dtype) is a contiguous full view: row-major strides, zero offset.
        v = h.tensor(shapes=(4, 8), dtype=DataType.FLOAT32)
        assert v.shapes == (4, 8)
        assert v.strides == (8, 1)
        assert v.ndims == 2
        assert v.byte_offset == 0
        # An explicit stride is carried verbatim; a singleton dim is never normalized away.
        strided = h.tensor(shapes=(4, 1), dtype=DataType.FLOAT32, strides=(8, 3))
        assert strided.strides == (8, 3)
    finally:
        h.close()


def test_tensor_unpack_roundtrip():
    oid = mint_owner_instance_id()
    h = create_host_shared_buffer(64, oid, buffer_id=1, owner_worker_path="L3")
    try:
        ref = h.tensor(shapes=(2, 4), dtype=DataType.FLOAT16, byte_offset=8)
        assert Tensor.unpack(ref.pack()) == ref
    finally:
        h.close()


def test_re_export_preserves_identity_same_backing_no_map():
    # Frozen model §5/§8: canonical identity is invariant across every edge. Re-exporting an L4-owned
    # backing for forwarding keeps the SOURCE identity (owner_instance_id / path / buffer_id /
    # generation) and the same backing, only stripping the mapping.
    l4 = mint_owner_instance_id()
    src = create_host_shared_buffer(64, l4, buffer_id=7, owner_worker_path="L4")
    try:
        sdesc = src.to_descriptor()
        hp = re_export(sdesc)
        assert hp.identity.pack() == src.identity.pack()  # identity invariant across the edge
        assert hp.backend_kind == BackendKind.POSIX_SHM
        assert hp.body == sdesc.body and hp.nbytes == 64  # same backing
        assert hp.shm is None and hp.base == 0  # no map (lazy — a compute leaf maps)
        # a ref built from H' carries the source identity + the same shm body, so L2 can materialize it
        r = hp.tensor(shapes=(16,), dtype=DataType.FLOAT32)
        assert Tensor.unpack(r.pack()).handle.identity.pack() == src.identity.pack()
    finally:
        src.close()


def test_device_malloc_wrap_materialize():
    # A device pointer (from orch.malloc) wrapped as DEVICE_MALLOC: materializes to the pointer with
    # no map, address_space DEVICE (-> a child_memory Tensor).
    oid = mint_owner_instance_id()
    h = wrap_device_malloc(0xDEAD0000, 4096, oid, buffer_id=3, owner_worker_path="L3")
    assert h.backend_kind == BackendKind.DEVICE_MALLOC
    assert h.address_space == AddressSpace.DEVICE
    assert h.shm is None and h.base == 0xDEAD0000
    reg = ImportRegistry()
    imp = reg.materialize(h.to_descriptor())
    assert imp.base == 0xDEAD0000
    assert imp.address_space == AddressSpace.DEVICE
    assert imp.shm is None


def test_fork_inherited_zero_copy_materialize():
    # A pre-fork COW-inherited allocation: the body is the base VA, materialized in place (no shm,
    # no copy). In-process the VA is trivially valid. A plain allocation is copy-on-write, so it is
    # tagged FORK_COW and grants READ only; a MAP_SHARED one is FORK_SHM and may grant writes.
    backing = ctypes.create_string_buffer(64)
    addr = ctypes.addressof(backing)
    oid = mint_owner_instance_id()
    handle = wrap_fork_inherited(addr, 64, owner_instance_id=oid, buffer_id=5, owner_worker_path="L3")
    assert handle.backend_kind == BackendKind.FORK_COW
    assert handle.access == AccessMode.READ
    shared = wrap_fork_inherited(addr, 64, oid, buffer_id=6, access=AccessMode.READWRITE)
    assert shared.backend_kind == BackendKind.FORK_SHM
    assert handle.shm is None
    reg = ImportRegistry()
    try:
        # 15 int32 at byte 4 exactly fills the 64-byte backing; 16 would overrun it by one element.
        ref = handle.tensor(shapes=(15,), strides=(1,), dtype=DataType.INT32, byte_offset=4)
        blob = pack_bufferref_blob([ref])
        src = ctypes.create_string_buffer(blob, len(blob))
        resolved = reg.materialize_blob(ctypes.addressof(src), len(blob))
        assert reg.resolve(handle.identity).base == addr  # same VA, no mapping
        tensor_blob = materialize_bufferref_blob(ctypes.addressof(src), len(blob), resolved)
        dst = ctypes.create_string_buffer(tensor_blob, len(tensor_blob))
        args = read_args_from_blob(ctypes.addressof(dst))
        assert args.tensor(0).data == addr + 4
    finally:
        reg.close()
        handle.close()  # no-op: FORK_SHM owns no shm


def test_materialize_remote_sidecar_rejected():
    desc = BufferHandleDescriptor(
        identity=_identity(),
        address_space=AddressSpace.HOST,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.REMOTE_SIDECAR,
        nbytes=8,
    )
    reg = ImportRegistry()
    with pytest.raises(ValueError, match="REMOTE_SIDECAR"):
        reg.materialize(desc)


def test_owner_instance_ids_are_distinct():
    ids = {mint_owner_instance_id() for _ in range(64)}
    assert len(ids) == 64
    assert all(len(i) == OWNER_INSTANCE_ID_BYTES for i in ids)


def test_materialize_bufferref_blob_to_tensors():
    oid = mint_owner_instance_id()
    h0 = create_host_shared_buffer(nbytes=64, owner_instance_id=oid, buffer_id=1)
    h1 = create_host_shared_buffer(nbytes=128, owner_instance_id=oid, buffer_id=2, generation=3)
    reg = ImportRegistry()
    try:
        # Self-describing: refs embed the full descriptor (built via BufferHandle.ref); the consumer
        # materializes lazily from the blob (no prior register), map-once by identity.
        ref0 = h0.tensor(shapes=(4,), strides=(1,), dtype=DataType.FLOAT32)
        ref1 = h1.tensor(shapes=(2, 4), strides=(4, 1), dtype=DataType.FLOAT16, byte_offset=8)
        assert ref0.handle == h0.to_descriptor()
        blob = pack_bufferref_blob([ref0, ref1], scalars=(42,))
        src = ctypes.create_string_buffer(blob, len(blob))

        resolved = reg.materialize_blob(ctypes.addressof(src), len(blob))
        tensor_blob = materialize_bufferref_blob(ctypes.addressof(src), len(blob), resolved)
        dst = ctypes.create_string_buffer(tensor_blob, len(tensor_blob))
        args = read_args_from_blob(ctypes.addressof(dst))

        assert args.tensor_count() == 2
        assert args.scalar_count() == 1
        assert args.tensor(0).data == reg.resolve(h0.identity).base + 0
        assert args.tensor(0).shapes == (4,)
        assert args.tensor(0).child_memory is False  # HOST
        assert args.tensor(1).data == reg.resolve(h1.identity).base + 8
        assert args.tensor(1).shapes == (2, 4)
        assert args.scalar(0) == 42
    finally:
        reg.close()
        h0.close()
        h1.close()


def test_materialize_strided_views():
    # Explicitly-strided, offset views materialize to strided chip Tensors (matching the strided
    # Tensor wire), not rejected — and no stride is normalized away in transit.
    oid = mint_owner_instance_id()
    h = create_host_shared_buffer(1024, oid, buffer_id=1)
    reg = ImportRegistry()
    try:
        # A transposed view of (4,8): shapes (8,4), strides (1,8).
        t = h.tensor(shapes=(8, 4), dtype=DataType.FLOAT32, strides=(1, 8))
        # An inner-dim sub-range of (4,8) starting at element 2: shapes (4,4), strides (8,1).
        s = h.tensor(shapes=(4, 4), dtype=DataType.FLOAT32, strides=(8, 1), byte_offset=2 * 1 * 4)
        blob = pack_bufferref_blob([t, s])
        src = ctypes.create_string_buffer(blob, len(blob))
        resolved = reg.materialize_blob(ctypes.addressof(src), len(blob))
        tb = materialize_bufferref_blob(ctypes.addressof(src), len(blob), resolved)
        dst = ctypes.create_string_buffer(tb, len(tb))
        args = read_args_from_blob(ctypes.addressof(dst))
        base = reg.resolve(h.identity).base

        tt = args.tensor(0)
        assert tt.shapes == (8, 4) and tt.strides == (1, 8) and not tt.is_contiguous
        assert tt.data == base
        ss = args.tensor(1)
        assert ss.shapes == (4, 4) and ss.strides == (8, 1) and not ss.is_contiguous
        assert ss.data == base + 2 * 1 * 4  # slice(1,2,..) shifts byte_offset by start*stride*elem
    finally:
        reg.close()
        h.close()


def test_materialize_rejects_unresolved_identity():
    oid = mint_owner_instance_id()
    handle = create_host_shared_buffer(nbytes=32, owner_instance_id=oid, buffer_id=9)
    try:
        ref = Tensor(handle.to_descriptor(), byte_offset=0, shapes=(8,), strides=(1,), dtype=DataType.INT32)
        blob = pack_bufferref_blob([ref])
        src = ctypes.create_string_buffer(blob, len(blob))
        with pytest.raises(RuntimeError, match="identity"):
            materialize_bufferref_blob(ctypes.addressof(src), len(blob), {})  # nothing resolved
    finally:
        handle.close()


@pytest.mark.parametrize(
    "space,backend",
    [
        (AddressSpace.HOST, BackendKind.VMM_WINDOW),
        (AddressSpace.HOST, BackendKind.DEVICE_MALLOC),
        (AddressSpace.DEVICE, BackendKind.FORK_SHM),
        (AddressSpace.DEVICE, BackendKind.POSIX_SHM),
    ],
)
def test_descriptor_rejects_bad_capability_combo(space, backend):
    # §4.1 capability matrix: an unsupported address_space×backend_kind fails at construction (before
    # dispatch, before it can ride the wire).
    with pytest.raises(ValueError, match="capability"):
        BufferHandleDescriptor(
            identity=_identity(),
            address_space=space,
            access=AccessMode.READWRITE,
            backend_kind=backend,
            nbytes=64,
            body=b"",
        )


def test_descriptor_accepts_legal_combos():
    for space, backend in [
        (AddressSpace.HOST, BackendKind.FORK_SHM),
        (AddressSpace.HOST, BackendKind.POSIX_SHM),
        (AddressSpace.DEVICE, BackendKind.VMM_WINDOW),
        (AddressSpace.DEVICE, BackendKind.DEVICE_MALLOC),
        (AddressSpace.HOST, BackendKind.REMOTE_SIDECAR),
        (AddressSpace.DEVICE, BackendKind.REMOTE_SIDECAR),
    ]:
        BufferHandleDescriptor(_identity(), space, AccessMode.READWRITE, backend, 64, b"")


# --- the shared validator: every trust boundary rejects a malformed element -----------------------
#
# `TaskArgs.add_tensor` (builder) and blob decode (receive) both run `validate_buffer_ref`, so a
# hand-packed or corrupted element is refused before any of its fields is trusted.


def _valid_packed_tensor() -> tuple[bytes, object]:
    """A well-formed packed Tensor plus its owning handle (caller closes it)."""
    h = create_host_shared_buffer(256, mint_owner_instance_id(), buffer_id=1)
    return h.tensor(shapes=(8,), dtype=DataType.FLOAT32).pack(), h


def _add_packed(packed: bytes) -> None:
    from simpler.task_interface import TaskArgs, TensorArgType  # noqa: PLC0415

    TaskArgs().add_tensor(packed, TensorArgType.INPUT)


@pytest.mark.parametrize(
    "offset,value,match",
    [
        (0, b"\x00\x00", "magic"),  # descriptor magic
        (2, b"\x07", "address_space"),  # address_space out of range
        (3, b"\x09", "access"),  # access out of range
        (4, b"\x63", "backend_kind"),  # backend_kind out of range
        (52, b"\xff\xff", "body_len"),  # body_len past the array
        (24, b"\x00\x00\x00\x00", "generation"),  # identity.generation == 0 (reserved)
        (96, b"\x63\x00\x00\x00", "ndims"),  # ndims out of range
        (140, b"\x7f", "dtype"),  # unknown dtype
        (120, b"\x00\x00\x00\x00", "stride"),  # strides[0] == 0
        (88, b"\x03\x00\x00\x00\x00\x00\x00\x00", "byte_offset"),  # unaligned byte_offset
    ],
)
def test_add_tensor_rejects_corrupted_field(offset, value, match):
    packed, h = _valid_packed_tensor()
    try:
        _add_packed(packed)  # the untouched element is accepted
        corrupted = bytearray(packed)
        corrupted[offset : offset + len(value)] = value
        with pytest.raises(ValueError, match=match):
            _add_packed(bytes(corrupted))
    finally:
        h.close()


def test_add_tensor_rejects_view_past_the_backing():
    h = create_host_shared_buffer(64, mint_owner_instance_id(), buffer_id=1)
    try:
        _add_packed(h.tensor(shapes=(16,), dtype=DataType.FLOAT32).pack())  # exactly 64 B: fits
        with pytest.raises(ValueError, match="past the backing"):
            _add_packed(h.tensor(shapes=(17,), dtype=DataType.FLOAT32).pack())
        with pytest.raises(ValueError, match="past the backing"):
            _add_packed(h.tensor(shapes=(16,), dtype=DataType.FLOAT32, byte_offset=4).pack())
    finally:
        h.close()


def test_add_tensor_rejects_arbitrary_bytes():
    # The builder takes raw bytes, so anything of the right length reaches it. None of it may be
    # accepted, and none of it may crash: every length-like field is bounded before it is used.
    import os as _os  # noqa: PLC0415

    for _ in range(256):
        with pytest.raises(ValueError):
            _add_packed(_os.urandom(BUFFER_REF_BYTES))
    with pytest.raises(ValueError):
        _add_packed(b"\x00" * BUFFER_REF_BYTES)
    with pytest.raises(ValueError):
        _add_packed(b"\xff" * BUFFER_REF_BYTES)
