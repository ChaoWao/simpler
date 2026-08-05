# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Owner/consumer side of the Buffer ABI.

The three wire types — ``CanonicalIdentity``, ``BufferDescriptor`` and ``Tensor`` — are the C++
structs of ``buffer.h`` bound directly, so there is one definition of the layout and one validator
(``validate_tensor``) behind every boundary that builds or decodes one. This module re-exports them
and adds what is inherently host-side: ``Buffer``, which owns a POSIX shm, the constructors that wrap
a backing as one, and ``ImportRegistry``, the consumer's map-once cache. Transporting a ``Tensor``
is the TaskArgs mailbox blob's job and lives in ``task_args.h``, not here.

``CanonicalIdentity`` is fixed-length (opaque ``owner_instance_id`` + ``buffer_id`` + ``generation``)
so hashing and comparison cannot read past it whatever arrives on the wire. The owning worker's tree
path is **not** part of the identity: it is interned to a diagnostic ``owner_worker_path_id`` whose
side table lives only in the owning process.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any

from _task_interface import (  # pyright: ignore[reportMissingImports]
    OWNER_INSTANCE_ID_BYTES,
    AccessMode,
    AddressSpace,
    BackendKind,
    BufferDescriptor,
    CanonicalIdentity,
    DataType,
    Tensor,
)

__all__ = [
    "AccessMode",
    "AddressSpace",
    "BackendKind",
    "Buffer",
    "BufferDescriptor",
    "CanonicalIdentity",
    "ImportRegistry",
    "ImportedBuffer",
    "Tensor",
    "create_host_shared_buffer",
    "host_ptr_nbytes",
    "intern_worker_path",
    "mint_owner_instance_id",
    "re_export",
    "remote_sidecar_tensor",
    "worker_path_for_id",
    "wrap_device_malloc",
    "wrap_fork_inherited",
    "wrap_vmm_window",
]


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


# `owner_worker_path` is resolved through this module's intern table, which is process-local, so it
# hangs off the bound descriptor here rather than in the binding.
BufferDescriptor.owner_worker_path = property(
    lambda self: worker_path_for_id(self.owner_worker_path_id),
    doc="The owning worker's tree path, for diagnostics only; ``<path#N>`` when minted elsewhere.",
)


def _row_major_strides(shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Contiguous (row-major) element strides for ``shapes``: strides[i] = prod(shapes[i+1:])."""
    strides = [1] * len(shapes)
    for i in range(len(shapes) - 2, -1, -1):
        strides[i] = strides[i + 1] * shapes[i + 1]
    return tuple(strides)


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
class Buffer:
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

    def to_descriptor(self) -> BufferDescriptor:
        """The wire descriptor for this backing — what a consumer needs to resolve it."""
        return BufferDescriptor(
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
        """A self-describing ``Tensor`` viewing this buffer: embeds the full descriptor + the view.

        ``strides`` default to contiguous (row-major) — ``buffer.tensor(shape, dtype)`` names the
        whole buffer as a contiguous view; pass explicit element strides for a strided view.
        ``byte_offset`` must be a multiple of the dtype size (checked at materialization).
        ``dtype`` accepts a ``DataType`` enum or its int value.
        """
        shapes = tuple(shapes)
        strides = _row_major_strides(shapes) if strides is None else tuple(strides)
        return Tensor(
            buffer=self.to_descriptor(),
            byte_offset=byte_offset,
            shapes=shapes,
            strides=strides,
            dtype=dtype,
        )

    def close(self) -> None:
        """Release the backing. The owner unlinks it, so a later consumer map fails rather than
        resolving a name whose bytes are gone. Idempotent."""
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
) -> Buffer:
    """Allocate a POSIX-shm host backing and wrap it as an owner ``Buffer`` (backend POSIX_SHM).

    The backend body is the shm name (UTF-8); the consumer maps it by name in ``ImportRegistry``.
    """
    if nbytes <= 0:
        raise ValueError(f"create_host_shared_buffer: nbytes must be positive, got {nbytes}")
    shm = SharedMemory(create=True, size=nbytes)
    identity = CanonicalIdentity(owner_instance_id, buffer_id, generation)
    return Buffer(
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


def re_export(source: BufferDescriptor) -> Buffer:
    """Re-export a received buffer descriptor for forwarding — identity **invariant**, no mapping.

    Canonical identity is invariant across every edge (frozen model §5/§8): the re-exported ``H'``
    keeps the SOURCE ``(owner_instance_id, buffer_id, generation)`` and the SAME
    backing (backend_kind / body / nbytes / address_space / access) as ``source`` — an
    L4-owned buffer forwarded L4→L3→L2 carries one identity at all three layers. Only the mapping is
    stripped: ``base=0``, ``shm=None`` (no mmap on the forwarding hop); a downstream compute leaf
    materializes lazily. Dependency inference keys on the (invariant) identity, so an alias /
    retain-release does not split across layers. Re-export is per-backing (memoize by identity), so
    pure forwarding carries no per-tensor map cost.
    """
    return Buffer(
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
    descriptor = BufferDescriptor(
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
        buffer=descriptor,
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
) -> Buffer:
    """Wrap a pre-fork, fork-inherited host allocation as a zero-copy ``Buffer``.

    Memory allocated before the children were forked is present in every child at the *same* virtual
    address; the backend body is that base VA (u64 LE) and the consumer materializes to the same VA
    with no mapping and no copy. The backend tag follows the mmap the caller actually has, which is
    what ``access`` states:

    * ``MAP_SHARED`` (e.g. a ``torch.Tensor.share_memory_()``) — a child's writes land in the pages
      the parent reads, so it can serve as an OUTPUT. Pass ``access=READWRITE``; tagged FORK_SHM.
    * plain ``MAP_PRIVATE`` — copy-on-write: a child's first write splits the page into a private
      copy the parent never sees. Read-only, the default; tagged FORK_COW so the distinction is a
      classification rather than something a reader has to infer from ``access``.
    """
    identity = CanonicalIdentity(owner_instance_id, buffer_id, generation)
    backend = BackendKind.FORK_SHM if access != AccessMode.READ else BackendKind.FORK_COW
    return Buffer(
        identity=identity,
        owner_worker_path_id=intern_worker_path(owner_worker_path),
        address_space=AddressSpace.HOST,
        access=access,
        backend_kind=backend,
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
) -> Buffer:
    """Wrap a device pointer (from a worker device malloc) as a ``DEVICE_MALLOC`` ``Buffer``.

    The backend body is the device pointer (u64 LE); the consumer materializes to that pointer with no
    mapping. The pointer is valid only on the chip that allocated it, so a tensor over this buffer must be
    dispatched only to that chip (a topology invariant, as for the former ``child_memory`` tensor).
    ``owner_worker_id`` records which next-level worker the backing lives on for free/copy provenance.
    """
    identity = CanonicalIdentity(owner_instance_id, buffer_id, generation)
    return Buffer(
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
) -> Buffer:
    """Wrap a domain-window-carved device VA as a ``VMM_WINDOW`` ``Buffer``.

    A comm domain's per-rank window is device memory carved by ``allocate_domain``; each named buffer
    slice is one such backing. The backend body is the device VA (u64 LE); the consumer materializes to
    that VA with no mapping. The VA is valid only on the chip that owns the window, so a tensor over this
    buffer must be dispatched only to that chip (``owner_worker_id``). Unlike ``DEVICE_MALLOC`` it is
    not freed by ``worker.free`` — the domain owns its lifetime and reclaims it at ``release_domain``.
    """
    identity = CanonicalIdentity(owner_instance_id, buffer_id, generation)
    return Buffer(
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
    """A buffer materialized into the consumer's address space: identity -> local base."""

    identity: CanonicalIdentity
    base: int
    nbytes: int
    address_space: AddressSpace = AddressSpace.HOST
    shm: SharedMemory | None = None  # the consumer's own mapping for shm backends


class ImportRegistry:
    """Per-consumer-endpoint lazy import cache: materialize a ``Tensor``'s embedded descriptor to a
    local base on first receipt (map-once), keyed by canonical identity.

    A consumer calls ``materialize`` for each tensor's embedded descriptor as it arrives; the first
    sight of an identity maps its backing into this process, later sights reuse the cached base
    (a bumped generation is a distinct identity, materialized fresh). Keyed by the canonical identity
    itself so lookups are exact — never a numeric-range guess, and never split by the wire padding
    the identity's equality and hashing deliberately ignore.
    """

    def __init__(self) -> None:
        self._by_identity: dict[CanonicalIdentity, ImportedBuffer] = {}

    def materialize(self, desc: BufferDescriptor) -> ImportedBuffer:
        """Map ``desc``'s backing into this process on first sight of its identity; reuse the
        cached ImportedBuffer thereafter (map-once).

        Takes the descriptor, never its bytes: every producer of one — an owner's ``to_descriptor()``,
        a re-export, an element pulled out of a received TaskArgs — hands over the bound type, and a
        caller holding raw bytes decodes them with ``BufferDescriptor.unpack`` so the decode is
        visible at its call site.
        """
        key = desc.identity
        cached = self._by_identity.get(key)
        if cached is not None:
            return cached
        if desc.backend_kind in (
            BackendKind.FORK_SHM,
            BackendKind.FORK_COW,
            BackendKind.DEVICE_MALLOC,
            BackendKind.VMM_WINDOW,
        ):
            # The body is the base pointer (u64 LE), already valid in this process — no mapping.
            # FORK_SHM: a COW-inherited host VA. DEVICE_MALLOC / VMM_WINDOW: a device pointer valid on
            # the chip that allocated / carved it (the tensor must only reach that chip — a topology
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

    def resolve(self, identity: CanonicalIdentity) -> ImportedBuffer:
        """The already-materialized import for ``identity``. Raises ``KeyError`` if this endpoint has
        not materialized that backing — resolution never maps as a side effect."""
        imported = self._by_identity.get(identity)
        if imported is None:
            raise KeyError(f"ImportRegistry: no buffer registered for {identity}")
        return imported

    def unregister(self, identity: CanonicalIdentity) -> None:
        """Drop one import and close its mapping. The owner still holds the backing; only this
        endpoint's view of it goes away. A no-op for an identity that was never materialized."""
        imported = self._by_identity.pop(identity, None)
        if imported is not None and imported.shm is not None:
            imported.shm.close()

    def close(self) -> None:
        """Close every mapping this endpoint made. Consumer-side only — unlinking belongs to the
        owning Worker, so this never destroys a backing."""
        for imported in self._by_identity.values():
            if imported.shm is not None:
                imported.shm.close()
        self._by_identity.clear()
