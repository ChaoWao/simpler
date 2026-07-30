# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Owner/consumer side of the BufferHandle ABI (Python mirror of ``buffer_handle.h``).

``CanonicalIdentity`` is fixed-length (opaque ``owner_instance_id`` + ``buffer_id`` + ``generation``)
so hashing and comparison cannot read past it whatever arrives on the wire. The owning worker's tree
path is **not** part of the identity: it is interned to a diagnostic ``owner_worker_path_id`` whose
side table lives only in the owning process. Struct formats mirror ``buffer_handle.h`` byte for byte;
their sizes are asserted at import against the constants the ``_task_interface`` binding exports, so
a layout drift fails loudly here.
"""

from __future__ import annotations

import ctypes
import enum
import os
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any

from _task_interface import (  # pyright: ignore[reportMissingImports]
    BUFFER_DESCRIPTOR_MAGIC,
    BUFFER_HANDLE_DESCRIPTOR_BYTES,
    BUFFER_REF_BYTES,
    BUFFERREF_BLOB_HEADER_BYTES,
    BUFFERREF_BLOB_MAGIC,
    CANONICAL_IDENTITY_BYTES,
    DESC_MAX_BYTES,
    MAX_TENSOR_DIMS,
    OWNER_INSTANCE_ID_BYTES,
    DataType,
    bufferref_blob_descriptors,
    bufferref_blob_refs,
    bufferref_blob_scalars,
)


def _dtype_value(dtype: Any) -> int:
    """The int wire value of a dtype given either a ``DataType`` enum or its int value.

    The nanobind ``DataType`` enum is not directly ``int()``-able, so callers historically passed
    ``dtype.value``; accept either form here to remove that footgun.
    """
    return int(dtype.value) if hasattr(dtype, "value") else int(dtype)


class AddressSpace(enum.IntEnum):
    HOST = 0
    DEVICE = 1


class AccessMode(enum.IntEnum):
    READ = 0
    WRITE = 1
    READWRITE = 2


class BackendKind(enum.IntEnum):
    FORK_SHM = 0
    POSIX_SHM = 1
    VMM_WINDOW = 2
    REMOTE_SIDECAR = 3
    DEVICE_MALLOC = 4


# (address_space, backend_kind) capability gate (§4.1). Absent ⇒ rejected. The two REMOTE_SIDECAR rows
# are allowed at the matrix layer; its P1 blanket reject is ImportRegistry.materialize's job (folding it
# here would double the semantics).
_CAPABILITY_OK: frozenset[tuple[int, int]] = frozenset(
    {
        (AddressSpace.HOST, BackendKind.FORK_SHM),
        (AddressSpace.HOST, BackendKind.POSIX_SHM),
        (AddressSpace.DEVICE, BackendKind.VMM_WINDOW),
        (AddressSpace.DEVICE, BackendKind.DEVICE_MALLOC),
        (AddressSpace.HOST, BackendKind.REMOTE_SIDECAR),
        (AddressSpace.DEVICE, BackendKind.REMOTE_SIDECAR),
    }
)


# ---------------------------------------------------------------------------
# Wire struct formats — mirror buffer_handle.h; sizes pinned to the binding.
# ---------------------------------------------------------------------------
#
# CanonicalIdentity (32 B): owner_instance_id[8] opaque, buffer_id u64, generation u32, _pad[12].
_CANONICAL_IDENTITY = struct.Struct(f"<{OWNER_INSTANCE_ID_BYTES}sQI12x")
# BufferHandleDescriptor (88 B) = prefix(8) + CanonicalIdentity(32) + suffix(48).
_DESC_PREFIX = struct.Struct("<HBBB3x")  # magic, address_space, access, backend_kind
_DESC_SUFFIX = struct.Struct(f"<QIH2x{DESC_MAX_BYTES}s")  # nbytes, owner_worker_path_id, body_len, body

assert _CANONICAL_IDENTITY.size == CANONICAL_IDENTITY_BYTES, (_CANONICAL_IDENTITY.size, CANONICAL_IDENTITY_BYTES)
assert _DESC_PREFIX.size + _CANONICAL_IDENTITY.size + _DESC_SUFFIX.size == BUFFER_HANDLE_DESCRIPTOR_BYTES, (
    "BufferHandleDescriptor layout drift vs buffer_handle.h"
)


# Owner-side residency for the diagnostic worker path. Ids are process-local and meaningful only in
# the owning process: a consumer that cannot resolve one renders it as "<path#N>". Nothing routes,
# gates or keys on a path, so an unresolvable id is never an error. Id 0 means "no path".
_PATH_BY_ID: dict[int, str] = {0: ""}
_ID_BY_PATH: dict[str, int] = {"": 0}


def intern_worker_path(path: str) -> int:
    """The diagnostic id for ``path`` in this process, assigning one on first sight."""
    pid = _ID_BY_PATH.get(path)
    if pid is None:
        pid = len(_ID_BY_PATH)
        _ID_BY_PATH[path] = pid
        _PATH_BY_ID[pid] = path
    return pid


def worker_path_for_id(path_id: int) -> str:
    """``path_id`` rendered for humans; an id minted in another process has no local text."""
    return _PATH_BY_ID.get(int(path_id), f"<path#{int(path_id)}>")


@dataclass(frozen=True)
class CanonicalIdentity:
    """Globally-unique allocation identity; the key of every import registry.

    Fixed-length: no field bounds a read, so hashing and comparison are structurally in-bounds for any
    bytes that arrive. ``owner_instance_id`` is an opaque nonce (bytewise-compared) and is the sole
    source of cross-incarnation uniqueness. ``generation`` starts at 1 and increments on every
    ``buffer_id`` slot reuse; 0 means uninitialized and is rejected on decode.
    """

    owner_instance_id: bytes
    buffer_id: int
    generation: int = 1

    def __post_init__(self) -> None:
        if len(self.owner_instance_id) != OWNER_INSTANCE_ID_BYTES:
            raise ValueError(
                f"owner_instance_id must be {OWNER_INSTANCE_ID_BYTES} bytes, got {len(self.owner_instance_id)}"
            )

    def pack(self) -> bytes:
        return _CANONICAL_IDENTITY.pack(self.owner_instance_id, self.buffer_id, self.generation)

    @classmethod
    def unpack(cls, raw: bytes) -> CanonicalIdentity:
        owner_instance_id, buffer_id, generation = _CANONICAL_IDENTITY.unpack(raw)
        if generation == 0:
            raise ValueError("generation 0 is reserved (uninitialized) — a live identity starts at 1")
        return cls(owner_instance_id, buffer_id, generation)


@dataclass(frozen=True)
class BufferHandleDescriptor:
    """The self-describing handle payload — embedded whole in every ``Tensor`` built over the handle.

    ``body`` is the per-backend materialization (POSIX/fork shm name UTF-8, VMM handle bytes, ...).
    """

    identity: CanonicalIdentity
    address_space: AddressSpace
    access: AccessMode
    backend_kind: BackendKind
    nbytes: int
    body: bytes = b""
    owner_worker_path_id: int = 0

    def __post_init__(self) -> None:
        # §4.1 capability gate: reject an unsupported address_space×backend_kind. Runs on every
        # construction — owner-side (wrap_* → to_descriptor) and wire decode (unpack → cls(...)) — so a
        # bad combo fails before dispatch and can never ride the wire.
        if (self.address_space, self.backend_kind) not in _CAPABILITY_OK:
            raise ValueError(
                f"unsupported address_space×backend: "
                f"{self.address_space.name}×{self.backend_kind.name} (§4.1 capability matrix)"
            )

    @property
    def owner_worker_path(self) -> str:
        """The owning worker's tree path, for diagnostics only; ``<path#N>`` when minted elsewhere."""
        return worker_path_for_id(self.owner_worker_path_id)

    def pack(self) -> bytes:
        if len(self.body) > DESC_MAX_BYTES:
            raise ValueError(f"backend body exceeds DESC_MAX_BYTES ({DESC_MAX_BYTES})")
        prefix = _DESC_PREFIX.pack(
            BUFFER_DESCRIPTOR_MAGIC,
            int(self.address_space),
            int(self.access),
            int(self.backend_kind),
        )
        suffix = _DESC_SUFFIX.pack(self.nbytes, self.owner_worker_path_id, len(self.body), self.body)
        return prefix + self.identity.pack() + suffix

    @classmethod
    def unpack(cls, raw: bytes) -> BufferHandleDescriptor:
        if len(raw) < BUFFER_HANDLE_DESCRIPTOR_BYTES:
            raise ValueError(f"descriptor too small: {len(raw)} < {BUFFER_HANDLE_DESCRIPTOR_BYTES}")
        magic, address_space, access, backend_kind = _DESC_PREFIX.unpack_from(raw, 0)
        if magic != BUFFER_DESCRIPTOR_MAGIC:
            raise ValueError(f"not a BufferHandleDescriptor: magic {magic:#06x} != {BUFFER_DESCRIPTOR_MAGIC:#06x}")
        identity = CanonicalIdentity.unpack(raw[_DESC_PREFIX.size : _DESC_PREFIX.size + _CANONICAL_IDENTITY.size])
        nbytes, path_id, body_len, body = _DESC_SUFFIX.unpack_from(raw, _DESC_PREFIX.size + _CANONICAL_IDENTITY.size)
        if body_len > DESC_MAX_BYTES:
            raise ValueError(f"body_len {body_len} exceeds DESC_MAX_BYTES ({DESC_MAX_BYTES})")
        return cls(
            identity=identity,
            address_space=AddressSpace(address_space),
            access=AccessMode(access),
            backend_kind=BackendKind(backend_kind),
            owner_worker_path_id=path_id,
            nbytes=nbytes,
            body=bytes(body[:body_len]),
        )


# BufferRef (272 B): BufferHandleDescriptor(216) + byte_offset u64, ndims u32, shapes[MAX] u32,
#   strides[MAX] u32, dtype u8, _pad[3].
_BUFFER_REF_TAIL = struct.Struct(f"<QI{MAX_TENSOR_DIMS}I{MAX_TENSOR_DIMS}IB3x")
assert BUFFER_HANDLE_DESCRIPTOR_BYTES + _BUFFER_REF_TAIL.size == BUFFER_REF_BYTES, "BufferRef layout drift"

# BufferRef blob envelope (16 B): magic u32, ref_count i32, scalar_count i32, reserved u32.
_BUFFERREF_BLOB_HEADER = struct.Struct("<IiiI")
assert _BUFFERREF_BLOB_HEADER.size == BUFFERREF_BLOB_HEADER_BYTES, "BufferRef blob header drift"


def _row_major_strides(shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Contiguous (row-major) element strides for ``shapes``: strides[i] = prod(shapes[i+1:])."""
    strides = [1] * len(shapes)
    for i in range(len(shapes) - 2, -1, -1):
        strides[i] = strides[i + 1] * shapes[i + 1]
    return tuple(strides)


@dataclass(frozen=True)
class Tensor:
    """A task argument: a strided view over a ``BufferHandle``, carrying that handle's descriptor.

    Self-describing — the consumer materializes ``handle`` on first receipt (no prior handshake),
    keyed by ``handle.identity``. **Carries no materialized address**: the consuming endpoint
    resolves the descriptor to a local base and adds ``byte_offset`` there.

    ``strides`` are ELEMENT strides and are strictly > 0 (broadcast / negative step unsupported);
    they are carried explicitly and a singleton dimension's stride is never normalized away.
    ``byte_offset`` is a BYTE offset, a multiple of the dtype size (checked at materialization).
    """

    handle: BufferHandleDescriptor
    byte_offset: int
    shapes: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: int | DataType  # normalized to the int wire value in __post_init__

    def __post_init__(self) -> None:
        if not 0 < len(self.shapes) <= MAX_TENSOR_DIMS:
            raise ValueError(f"Tensor ndims must be in [1, {MAX_TENSOR_DIMS}], got {len(self.shapes)}")
        if len(self.strides) != len(self.shapes):
            raise ValueError("Tensor shapes and strides must have equal length")
        object.__setattr__(self, "dtype", _dtype_value(self.dtype))

    @property
    def ndims(self) -> int:
        return len(self.shapes)

    def pack(self) -> bytes:
        ndims = len(self.shapes)
        shapes = list(self.shapes) + [0] * (MAX_TENSOR_DIMS - ndims)
        strides = list(self.strides) + [0] * (MAX_TENSOR_DIMS - ndims)
        tail = _BUFFER_REF_TAIL.pack(self.byte_offset, ndims, *shapes, *strides, self.dtype)
        return self.handle.pack() + tail

    @classmethod
    def unpack(cls, raw: bytes) -> Tensor:
        handle = BufferHandleDescriptor.unpack(raw[:BUFFER_HANDLE_DESCRIPTOR_BYTES])
        vals = _BUFFER_REF_TAIL.unpack(raw[BUFFER_HANDLE_DESCRIPTOR_BYTES:BUFFER_REF_BYTES])
        byte_offset, ndims = vals[0], vals[1]
        shapes = tuple(vals[2 : 2 + ndims])
        strides = tuple(vals[2 + MAX_TENSOR_DIMS : 2 + MAX_TENSOR_DIMS + ndims])
        dtype = vals[2 + 2 * MAX_TENSOR_DIMS]
        return cls(handle=handle, byte_offset=byte_offset, shapes=shapes, strides=strides, dtype=dtype)


def pack_bufferref_blob(tensors: list[Tensor], scalars: tuple[int, ...] = ()) -> bytes:
    """Serialize tensors + scalars into the versioned wire blob (mirror of write_bufferref_blob)."""
    header = _BUFFERREF_BLOB_HEADER.pack(BUFFERREF_BLOB_MAGIC, len(tensors), len(scalars), 0)
    body = b"".join(t.pack() for t in tensors)
    tail = struct.pack(f"<{len(scalars)}Q", *scalars) if scalars else b""
    return header + body + tail


def mint_owner_instance_id() -> bytes:
    """A fresh opaque nonce, unique per owner incarnation (defends identity against ABA).

    Must stay a full-width random draw. It is the only thing separating two Workers' buffer_id spaces,
    so a structured value (timestamp/pid) would hand the same identity to two Workers constructed in
    one process within one second — a routine pattern (an L4 and its L3 built back to back).
    """
    return os.urandom(OWNER_INSTANCE_ID_BYTES)


def _shm_base_addr(shm: SharedMemory) -> int:
    """Mapped base address of ``shm``; valid until ``shm.close()``."""
    view = shm.buf
    assert view is not None
    exporter = ctypes.c_char.from_buffer(view)
    addr = ctypes.addressof(exporter)
    del exporter
    return addr


@dataclass
class BufferHandle:
    """Owner-side registry object for one shared backing; owns the POSIX shm that backs it."""

    identity: CanonicalIdentity
    address_space: AddressSpace
    access: AccessMode
    backend_kind: BackendKind
    nbytes: int
    body: bytes = b""
    owner_worker_path_id: int = 0
    shm: SharedMemory | None = None
    base: int = 0
    # Owner-side only, never serialized into the descriptor: which next-level worker a DEVICE_MALLOC
    # backing lives on (0 for a host backing or an L2 own-device malloc). The device-pointer provenance
    # guard and free/copy key on (owner_worker_id, base).
    owner_worker_id: int = 0

    def to_descriptor(self) -> BufferHandleDescriptor:
        return BufferHandleDescriptor(
            identity=self.identity,
            address_space=self.address_space,
            owner_worker_path_id=self.owner_worker_path_id,
            access=self.access,
            backend_kind=self.backend_kind,
            nbytes=self.nbytes,
            body=self.body,
        )

    def tensor(
        self,
        shapes: tuple[int, ...],
        dtype: int | DataType,
        strides: tuple[int, ...] | None = None,
        byte_offset: int = 0,
    ) -> Tensor:
        """A self-describing ``Tensor`` viewing this handle: embeds the full descriptor + the view.

        ``strides`` default to contiguous (row-major) — ``handle.tensor(shape, dtype)`` names the
        whole buffer as a contiguous view; pass explicit element strides for a strided view.
        ``byte_offset`` must be a multiple of the dtype size (checked at materialization).
        ``dtype`` accepts a ``DataType`` enum or its int value.
        """
        shapes = tuple(shapes)
        strides = _row_major_strides(shapes) if strides is None else tuple(strides)
        return Tensor(
            handle=self.to_descriptor(),
            byte_offset=byte_offset,
            shapes=shapes,
            strides=strides,
            dtype=_dtype_value(dtype),
        )

    def close(self) -> None:
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            self.shm = None


def create_host_shared_buffer(
    nbytes: int,
    owner_instance_id: bytes,
    buffer_id: int,
    owner_worker_path: str = "",
    generation: int = 1,
    access: AccessMode = AccessMode.READWRITE,
) -> BufferHandle:
    """Allocate a POSIX-shm host backing and wrap it as an owner ``BufferHandle`` (backend POSIX_SHM).

    The backend body is the shm name (UTF-8); the consumer maps it by name in ``ImportRegistry``.
    """
    if nbytes <= 0:
        raise ValueError(f"create_host_shared_buffer: nbytes must be positive, got {nbytes}")
    shm = SharedMemory(create=True, size=nbytes)
    identity = CanonicalIdentity(owner_instance_id, buffer_id, generation)
    return BufferHandle(
        identity=identity,
        owner_worker_path_id=intern_worker_path(owner_worker_path),
        address_space=AddressSpace.HOST,
        access=access,
        backend_kind=BackendKind.POSIX_SHM,
        nbytes=nbytes,
        body=shm.name.encode("utf-8"),
        shm=shm,
        base=_shm_base_addr(shm),
    )


def re_export(source: BufferHandleDescriptor) -> BufferHandle:
    """Re-export a received handle descriptor for forwarding — identity **invariant**, no mapping.

    Canonical identity is invariant across every edge (frozen model §5/§8): the re-exported ``H'``
    keeps the SOURCE ``(owner_instance_id, buffer_id, generation)`` and the SAME
    backing (backend_kind / body / nbytes / address_space / access) as ``source`` — an
    L4-owned buffer forwarded L4→L3→L2 carries one identity at all three layers. Only the mapping is
    stripped: ``base=0``, ``shm=None`` (no mmap on the forwarding hop); a downstream compute leaf
    materializes lazily. Dependency inference keys on the (invariant) identity, so an alias /
    retain-release does not split across layers. Re-export is per-backing (memoize by identity), so
    pure forwarding carries no per-ref map cost.
    """
    return BufferHandle(
        identity=source.identity,
        owner_worker_path_id=source.owner_worker_path_id,
        address_space=source.address_space,
        access=source.access,
        backend_kind=source.backend_kind,
        nbytes=source.nbytes,
        body=source.body,
        shm=None,
        base=0,
    )


def remote_sidecar_tensor(
    shapes: tuple[int, ...],
    dtype: int,
    nbytes: int,
    owner_worker_id: int,
    buffer_id: int,
    generation: int,
    address_space: AddressSpace,
) -> Tensor:
    """Build a ``REMOTE_SIDECAR`` ``Tensor`` for a task arg destined for a remote worker.

    An arg passed L4→remote-L3 cannot be materialized from a local backing — the data lives on another
    machine and travels via the remote transport. Its descriptor therefore carries ``backend_kind =
    REMOTE_SIDECAR`` (a consumer decode-rejects a local materialize; the authoritative remote
    descriptor rides in the per-task RemoteTaskArgsSidecar). The identity encodes the remote buffer
    (``owner_worker_id`` folded into the opaque nonce, plus ``buffer_id`` / ``generation``) so
    dependency inference and routing stay stable across the hop.
    """
    oid = int(owner_worker_id).to_bytes(OWNER_INSTANCE_ID_BYTES, "little")
    # A HOST_INLINE placeholder has no backing and so no generation of its own; 0 is the reserved
    # "uninitialized" value a decoder rejects, so the placeholder carries the initial generation.
    identity = CanonicalIdentity(oid, buffer_id, int(generation) or 1)
    handle = BufferHandleDescriptor(
        identity=identity,
        owner_worker_path_id=intern_worker_path(f"remote/{owner_worker_id}"),
        address_space=address_space,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.REMOTE_SIDECAR,
        nbytes=nbytes,
        body=b"",
    )
    shapes = tuple(shapes)
    return Tensor(
        handle=handle,
        byte_offset=0,
        shapes=shapes,
        strides=_row_major_strides(shapes),
        dtype=int(dtype),
    )


def wrap_fork_inherited(
    data_ptr: int,
    nbytes: int,
    owner_instance_id: bytes,
    buffer_id: int,
    owner_worker_path: str = "",
    generation: int = 1,
    access: AccessMode = AccessMode.READ,
) -> BufferHandle:
    """Wrap a pre-fork, fork-inherited host allocation as a zero-copy ``FORK_SHM`` ``BufferHandle``.

    Memory allocated before the children were forked is present in every child at the *same* virtual
    address; the backend body is that base VA (u64 LE) and the consumer materializes to the same VA
    with no mapping and no copy. Read/write sharing depends on the underlying mmap:

    * a plain allocation is ``MAP_PRIVATE`` — copy-on-write, so a child's first write splits the page
      into a private copy the parent never sees. Read-only from the child (input); ``access`` defaults
      to READ.
    * a ``MAP_SHARED`` allocation (e.g. a ``torch.Tensor.share_memory_()``) is truly shared across the
      fork — the child's writes land in the same physical pages the parent reads, so it is usable as an
      OUTPUT the parent reads back. Pass ``access=READWRITE`` for that case.
    """
    identity = CanonicalIdentity(owner_instance_id, buffer_id, generation)
    return BufferHandle(
        identity=identity,
        owner_worker_path_id=intern_worker_path(owner_worker_path),
        address_space=AddressSpace.HOST,
        access=access,
        backend_kind=BackendKind.FORK_SHM,
        nbytes=nbytes,
        body=int(data_ptr).to_bytes(8, "little"),
        shm=None,
        base=int(data_ptr),
    )


def host_ptr_nbytes(obj: Any) -> tuple[int, int]:
    """Host address + byte length of a copy_to/copy_from buffer, without importing torch.

    A torch tensor is read via its ``data_ptr`` / ``numel`` / ``element_size`` (duck-typed); any other
    object goes through the buffer protocol and must be writable so its backing address is stable for
    the duration of the synchronous copy.
    """
    if hasattr(obj, "data_ptr") and hasattr(obj, "numel") and hasattr(obj, "element_size"):
        return int(obj.data_ptr()), int(obj.numel()) * int(obj.element_size())
    mv = memoryview(obj)
    if mv.readonly:
        raise TypeError("copy_to/copy_from host buffer must be a torch tensor or a writable buffer")
    return ctypes.addressof((ctypes.c_char * mv.nbytes).from_buffer(obj)), mv.nbytes


def wrap_device_malloc(
    device_ptr: int,
    nbytes: int,
    owner_instance_id: bytes,
    buffer_id: int,
    owner_worker_path: str = "",
    generation: int = 1,
    access: AccessMode = AccessMode.READWRITE,
    owner_worker_id: int = 0,
) -> BufferHandle:
    """Wrap a device pointer (from a worker device malloc) as a ``DEVICE_MALLOC`` ``BufferHandle``.

    The backend body is the device pointer (u64 LE); the consumer materializes to that pointer with no
    mapping. The pointer is valid only on the chip that allocated it, so a ref over this handle must be
    dispatched only to that chip (a topology invariant, as for the former ``child_memory`` tensor).
    ``owner_worker_id`` records which next-level worker the backing lives on for free/copy provenance.
    """
    identity = CanonicalIdentity(owner_instance_id, buffer_id, generation)
    return BufferHandle(
        identity=identity,
        owner_worker_path_id=intern_worker_path(owner_worker_path),
        address_space=AddressSpace.DEVICE,
        access=access,
        backend_kind=BackendKind.DEVICE_MALLOC,
        nbytes=nbytes,
        body=int(device_ptr).to_bytes(8, "little"),
        shm=None,
        base=int(device_ptr),
        owner_worker_id=int(owner_worker_id),
    )


def wrap_vmm_window(
    device_ptr: int,
    nbytes: int,
    owner_instance_id: bytes,
    buffer_id: int,
    owner_worker_path: str = "",
    generation: int = 1,
    access: AccessMode = AccessMode.READWRITE,
    owner_worker_id: int = 0,
) -> BufferHandle:
    """Wrap a domain-window-carved device VA as a ``VMM_WINDOW`` ``BufferHandle``.

    A comm domain's per-rank window is device memory carved by ``allocate_domain``; each named buffer
    slice is one such backing. The backend body is the device VA (u64 LE); the consumer materializes to
    that VA with no mapping. The VA is valid only on the chip that owns the window, so a ref over this
    handle must be dispatched only to that chip (``owner_worker_id``). Unlike ``DEVICE_MALLOC`` it is
    not freed by ``worker.free`` — the domain owns its lifetime and reclaims it at ``release_domain``.
    """
    identity = CanonicalIdentity(owner_instance_id, buffer_id, generation)
    return BufferHandle(
        identity=identity,
        owner_worker_path_id=intern_worker_path(owner_worker_path),
        address_space=AddressSpace.DEVICE,
        access=access,
        backend_kind=BackendKind.VMM_WINDOW,
        nbytes=nbytes,
        body=int(device_ptr).to_bytes(8, "little"),
        shm=None,
        base=int(device_ptr),
        owner_worker_id=int(owner_worker_id),
    )


@dataclass
class ImportedBuffer:
    """A handle materialized into the consumer's address space: identity -> local base."""

    identity: CanonicalIdentity
    base: int
    nbytes: int
    address_space: AddressSpace = AddressSpace.HOST
    shm: SharedMemory | None = None  # the consumer's own mapping for shm backends


@dataclass
class MappedArg:
    """A Python compute (sub-worker) task arg: a ``Tensor`` materialized into this process, exposing a
    writable ``buffer`` at the view origin plus the view geometry. The callable computes with e.g.
    ``torch.frombuffer(arg.buffer, dtype=<from arg.dtype>, count=prod(arg.shapes))`` — reads/writes
    land in the shared backing the owner sees.
    """

    imported: ImportedBuffer
    byte_offset: int
    shapes: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: int  # DataType value

    @property
    def buffer(self) -> memoryview:
        """A memoryview over the mapped backing at this view's origin (``byte_offset``)."""
        ib = self.imported
        if ib.shm is not None:
            base = ib.shm.buf
            assert base is not None
        else:
            # FORK_SHM (COW): no shm object — wrap the inherited VA range.
            base = memoryview((ctypes.c_char * ib.nbytes).from_address(ib.base))
        return base[self.byte_offset :]


class MappedArgs(Sequence):
    """A Python sub-worker's task args: the mapped tensor args plus the scalar args.

    Indexes and iterates as the tensor ``MappedArg`` list (``args[i].buffer``, ``len(args)``) — the
    common compute-leaf access — and additionally exposes the blob's scalars via ``scalar_count()`` /
    ``scalar(i)`` (uint64, in submission order), mirroring the owner-side ``TaskArgs`` scalar API.
    """

    __slots__ = ("_tensors", "_scalars")

    def __init__(self, tensors: list[MappedArg], scalars: tuple[int, ...]) -> None:
        self._tensors = list(tensors)
        self._scalars = tuple(int(s) for s in scalars)

    def __getitem__(self, i):
        return self._tensors[i]

    def __len__(self) -> int:
        return len(self._tensors)

    def tensor_count(self) -> int:
        return len(self._tensors)

    def scalar_count(self) -> int:
        return len(self._scalars)

    def scalar(self, i: int) -> int:
        return self._scalars[i]


class ImportRegistry:
    """Per-consumer-endpoint lazy import cache: materialize a ``Tensor``'s embedded descriptor to a
    local base on first receipt (map-once), keyed by canonical identity.

    A consumer calls ``materialize`` for each ref's embedded descriptor as it arrives; the first
    sight of an identity maps its backing into this process, later sights reuse the cached base
    (a bumped generation is a distinct identity, materialized fresh). Keyed by the packed canonical
    identity so lookups are exact — never a numeric-range guess.
    """

    def __init__(self) -> None:
        self._by_identity: dict[bytes, ImportedBuffer] = {}

    def materialize(self, descriptor: BufferHandleDescriptor | bytes) -> ImportedBuffer:
        """Map ``descriptor``'s backing into this process on first sight of its identity; reuse the
        cached ImportedBuffer thereafter (map-once)."""
        desc = BufferHandleDescriptor.unpack(descriptor) if isinstance(descriptor, (bytes, bytearray)) else descriptor
        key = desc.identity.pack()
        cached = self._by_identity.get(key)
        if cached is not None:
            return cached
        if desc.backend_kind in (BackendKind.FORK_SHM, BackendKind.DEVICE_MALLOC, BackendKind.VMM_WINDOW):
            # The body is the base pointer (u64 LE), already valid in this process — no mapping.
            # FORK_SHM: a COW-inherited host VA. DEVICE_MALLOC / VMM_WINDOW: a device pointer valid on
            # the chip that allocated / carved it (the ref must only reach that chip — a topology
            # invariant).
            base = int.from_bytes(desc.body, "little")
            imported = ImportedBuffer(desc.identity, base, desc.nbytes, desc.address_space, None)
        elif desc.backend_kind == BackendKind.POSIX_SHM:
            shm = SharedMemory(name=desc.body.decode("utf-8"))
            imported = ImportedBuffer(desc.identity, _shm_base_addr(shm), desc.nbytes, desc.address_space, shm)
        elif desc.backend_kind == BackendKind.REMOTE_SIDECAR:
            raise ValueError("ImportRegistry: REMOTE_SIDECAR backend is reserved for P2")
        else:
            raise NotImplementedError(f"ImportRegistry: backend {desc.backend_kind!r} not supported in P1-B")
        self._by_identity[key] = imported
        return imported

    def materialize_blob(self, blob_ptr: int, capacity: int) -> dict[bytes, tuple[int, int]]:
        """Lazily materialize every embedded descriptor in a wire blob and return the resolved
        map for ``materialize_bufferref_blob``: packed identity -> (local base, address_space)."""
        for desc_bytes in bufferref_blob_descriptors(blob_ptr, capacity):
            self.materialize(desc_bytes)
        return self.materialization_map()

    def mapped_args_from_blob(self, blob_ptr: int, capacity: int) -> MappedArgs:
        """Materialize a wire blob into a Python compute callable's args: every tensor becomes a
        MappedArg (map-once, buffer at the view origin) and the blob's scalars ride alongside. This is
        the compute-leaf map (a sub-worker reads/writes), distinct from pure forwarding (re-export,
        which never maps).
        """
        tensors = []
        for i, ref in enumerate(Tensor.unpack(rb) for rb in bufferref_blob_refs(blob_ptr, capacity)):
            if ref.handle.address_space == AddressSpace.DEVICE:
                # Depth behind the submit-time endpoint check: this process is a host compute leaf, so
                # a device address here would be handed to torch as a host pointer.
                raise ValueError(
                    f"sub-worker argument {i} is a DEVICE-space tensor "
                    f"({ref.handle.backend_kind.name}); it cannot be mapped into a host process"
                )
            tensors.append(MappedArg(self.materialize(ref.handle), ref.byte_offset, ref.shapes, ref.strides, ref.dtype))
        return MappedArgs(tensors, tuple(bufferref_blob_scalars(blob_ptr, capacity)))

    def resolve(self, identity: CanonicalIdentity) -> ImportedBuffer:
        imported = self._by_identity.get(identity.pack())
        if imported is None:
            raise KeyError(f"ImportRegistry: no handle registered for {identity}")
        return imported

    def materialization_map(self) -> dict[bytes, tuple[int, int]]:
        """Snapshot for ``materialize_bufferref_blob``: packed identity -> (local base, address_space)."""
        return {key: (ib.base, int(ib.address_space)) for key, ib in self._by_identity.items()}

    def unregister(self, identity: CanonicalIdentity) -> None:
        imported = self._by_identity.pop(identity.pack(), None)
        if imported is not None and imported.shm is not None:
            imported.shm.close()

    def close(self) -> None:
        for imported in self._by_identity.values():
            if imported.shm is not None:
                imported.shm.close()
        self._by_identity.clear()
