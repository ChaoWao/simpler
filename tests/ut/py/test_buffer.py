# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for simpler.buffer: identity/descriptor pack-unpack + create/import round trip.

The three wire types are the C++ structs of buffer.h bound directly, so what is pinned here is the
Python-visible contract over them — construction rejects what `validate_buffer_descriptor` rejects,
a round trip through `pack`/`unpack` is the identity, and equality and hashing ignore wire padding.
Imports come from `simpler.buffer` because that is where a caller reaches them, alongside the
registry and the Buffer constructors that are genuinely defined there.
"""

import pytest
from _task_interface import OWNER_INSTANCE_ID_BYTES, DataType
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    BufferDescriptor,
    CanonicalIdentity,
    ImportRegistry,
    Tensor,
    create_host_shared_buffer,
    intern_worker_path,
    mint_owner_instance_id,
    re_export,
    wrap_device_malloc,
)
from simpler.task_interface import ChipTensor

_OID = bytes(range(0xA0, 0xA0 + OWNER_INSTANCE_ID_BYTES))


def test_wire_tensor_and_device_pod_are_distinct_types():
    # The whole point of the two names: one address-free argument, one GM-address-bearing POD. An
    # alias here would give `Tensor` a second meaning and every later cutover would have to keep it.
    assert ChipTensor is not Tensor


def test_wire_tensor_stays_off_the_public_submit_surface():
    # `TaskArgs.add_tensor` still takes a ChipTensor, so re-exporting `Tensor` from
    # simpler.task_interface would advertise a public type that its own submit call rejects. The two
    # facts move together: the wire cutover that makes add_tensor accept a Tensor is what earns it a
    # place on that module, and this test fails then to say so.
    import simpler.task_interface as ti  # noqa: PLC0415

    assert not hasattr(ti, "Tensor")

    h = create_host_shared_buffer(64, mint_owner_instance_id(), buffer_id=1)
    try:
        args = ti.TaskArgs()
        with pytest.raises(TypeError):
            args.add_tensor(h.tensor(shapes=(16,), dtype=DataType.FLOAT32))
    finally:
        h.close()


def _identity(oid=_OID, buffer_id=7, generation=2):
    return CanonicalIdentity(oid, buffer_id, generation)


def test_identity_rejects_bad_oid_width():
    with pytest.raises(ValueError):
        CanonicalIdentity(b"\x00" * (OWNER_INSTANCE_ID_BYTES + 1), 1, 1)


def test_identity_distinguishes_generation_and_incarnation():
    a = _identity()
    assert a != _identity(generation=a.generation + 1)  # ABA
    assert a != _identity(oid=bytes(range(1, 1 + OWNER_INSTANCE_ID_BYTES)))  # different incarnation
    assert a != _identity(buffer_id=a.buffer_id + 1)


def _descriptor_with_path_id(path_id: int) -> BufferDescriptor:
    return BufferDescriptor(
        identity=_identity(),
        address_space=AddressSpace.HOST,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.POSIX_SHM,
        nbytes=8,
        owner_worker_path_id=path_id,
    )


def test_worker_path_is_diagnostic_and_survives_an_unknown_id():
    h = _descriptor_with_path_id(intern_worker_path("L4/L3[2]"))
    assert h.owner_worker_path == "L4/L3[2]"
    # An id minted in another process has no local text, and that is not an error.
    assert _descriptor_with_path_id(99_999).owner_worker_path == "<path#99999>"
    # The path takes no part in identity: two descriptors differing only by path are one backing.
    assert _descriptor_with_path_id(0).identity == h.identity


def test_descriptor_rejects_oversized_body():
    with pytest.raises(ValueError, match="body"):
        BufferDescriptor(
            identity=_identity(),
            address_space=AddressSpace.HOST,
            access=AccessMode.READWRITE,
            backend_kind=BackendKind.POSIX_SHM,
            nbytes=8,
            body=b"x" * 200,  # > DESC_MAX_BYTES
        )


def test_create_export_import_resolve_zero_copy():
    oid = mint_owner_instance_id()
    buffer = create_host_shared_buffer(nbytes=256, owner_instance_id=oid, buffer_id=1, owner_worker_path="L4")
    reg = ImportRegistry()
    try:
        assert buffer.backend_kind == BackendKind.POSIX_SHM
        imported = reg.materialize(buffer.to_descriptor())
        assert reg.resolve(buffer.identity).base == imported.base
        assert reg.materialize(buffer.to_descriptor()).base == imported.base  # map-once: same mapping
        assert imported.nbytes == 256
        owner_shm = buffer.shm
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
        buffer.close()


def test_resolve_unregistered_raises():
    reg = ImportRegistry()
    with pytest.raises(KeyError):
        reg.resolve(_identity())


def test_tensor_full_view_is_contiguous():
    oid = mint_owner_instance_id()
    h = create_host_shared_buffer(nbytes=1024, owner_instance_id=oid, buffer_id=1)
    try:
        # buffer.tensor(shape, dtype) is a contiguous full view: row-major strides, zero offset.
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


def test_re_export_preserves_identity_same_backing_no_map():
    # Frozen model §5/§8: canonical identity is invariant across every edge. Re-exporting an L4-owned
    # backing for forwarding keeps the SOURCE identity (owner_instance_id / path / buffer_id /
    # generation) and the same backing, only stripping the mapping.
    l4 = mint_owner_instance_id()
    src = create_host_shared_buffer(64, l4, buffer_id=7, owner_worker_path="L4")
    try:
        sdesc = src.to_descriptor()
        hp = re_export(sdesc)
        assert hp.identity == src.identity  # identity invariant across the edge
        assert hp.backend_kind == BackendKind.POSIX_SHM
        assert hp.body == sdesc.body and hp.nbytes == 64  # same backing
        assert hp.shm is None and hp.base == 0  # no map (lazy — a compute leaf maps)
        # a tensor built from H' carries the source identity + the same shm body, so L2 can materialize it
        r = hp.tensor(shapes=(16,), dtype=DataType.FLOAT32)
        assert r.buffer.identity == src.identity
        assert r.buffer.body == sdesc.body
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


def test_materialize_remote_sidecar_rejected():
    desc = BufferDescriptor(
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
        BufferDescriptor(
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
        BufferDescriptor(_identity(), space, AccessMode.READWRITE, backend, 64, b"")


# --- the shared validator, on the paths Python can reach --------------------------------------
#
# `validate_tensor` guards two boundaries: construction (here) and blob decode, which after the wire
# flip is `TaskArgsView::tensors(i)` in task_args.h. Both are C++, and Python has no way to turn
# bytes into a Tensor at all — so the malformed-bytes cases (bad magic, unknown backend tag,
# generation 0, body_len past the array, a view that does not fit) are exercised where they can be
# built: tests/ut/cpp/types/test_buffer.cpp. What is reachable from here is the construction gate.


def test_construction_rejects_a_view_past_the_backing():
    h = create_host_shared_buffer(64, mint_owner_instance_id(), buffer_id=1)
    try:
        h.tensor(shapes=(16,), dtype=DataType.FLOAT32)  # exactly 64 B: fits
        with pytest.raises(ValueError, match="past the backing"):
            h.tensor(shapes=(17,), dtype=DataType.FLOAT32)
        with pytest.raises(ValueError, match="past the backing"):
            h.tensor(shapes=(16,), dtype=DataType.FLOAT32, byte_offset=4)
    finally:
        h.close()


def test_construction_rejects_a_zero_stride():
    # strides are element strides and strictly > 0: broadcast and negative step are unsupported, and
    # a 0 would make two coordinates alias without the overlap map ever seeing it.
    h = create_host_shared_buffer(256, mint_owner_instance_id(), buffer_id=1)
    try:
        with pytest.raises(ValueError, match="stride"):
            h.tensor(shapes=(4, 8), dtype=DataType.FLOAT32, strides=(8, 0))
    finally:
        h.close()
