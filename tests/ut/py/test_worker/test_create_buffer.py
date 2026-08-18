# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Owner-side ``Worker.create_buffer``: the child-topology gate and the identity it mints.

``_create_buffer_locked`` reads only ``level``, the three child-shm lists, and the buffer registry
state, so these run against a Worker built with ``__new__`` — no fork, no device, no ``init()``.
The gate is the point: an L3+ backing is resolved by a forked child mapping the shm by name, and
**every** kind of forked child can do that, so counting only chip and sub children refuses an L4
whose children are local L3 Workers.
"""

from __future__ import annotations

import threading

import pytest
from simpler.buffer import AddressSpace, BackendKind, mint_owner_instance_id
from simpler.worker import Worker, _NoBufferConsumerError, _SharedExclusiveLock


def _bare_worker(level: int, *, chip: int = 0, sub: int = 0, next_level: int = 0) -> Worker:
    w = Worker.__new__(Worker)
    w.level = level
    w._chip_shms = [object()] * chip
    w._sub_shms = [object()] * sub
    w._next_level_shms = [object()] * next_level
    w._registry_lock = threading.Lock()
    w._owner_instance_id = mint_owner_instance_id()
    w._buffer_id_counter = 1
    w._buffers = {}
    w._hierarchical_start_mu = threading.Lock()
    w._hierarchical_start_cv = threading.Condition(w._hierarchical_start_mu)
    w._accepted_run_handles = set()
    w._abandoned_run_handles = []
    # Run admission is shared/exclusive now, not a plain Lock: production code calls
    # `.exclusive()` / `.shared()` on it, so the stand-in has to be the real type.
    w._submit_mu = _SharedExclusiveLock()
    w._chip_run_touched_identities = {}
    w._chip_import_registry = None
    w._reexport_by_source = {}
    w._worker = None
    return w


def _drain(w: Worker) -> None:
    for buffer in list(w._buffers.values()):
        buffer.close()
    w._buffers.clear()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chip": 1},
        {"sub": 1},
        # An L4 whose only children are local L3 Workers: they are forked processes that map a
        # POSIX_SHM backing by name exactly as a chip child does, so the buffer has a consumer.
        {"next_level": 1},
    ],
    ids=["chip", "sub", "next_level"],
)
def test_any_forked_child_admits_a_buffer(kwargs):
    w = _bare_worker(4, **kwargs)
    try:
        buffer = w._create_buffer_locked(64)
        assert buffer.nbytes == 64
        assert buffer.backend_kind == BackendKind.POSIX_SHM
        assert buffer.address_space == AddressSpace.HOST
    finally:
        _drain(w)


def test_childless_l3_plus_is_refused():
    w = _bare_worker(3)
    with pytest.raises(RuntimeError, match="at least one forked"):
        w._create_buffer_locked(64)
    assert not w._buffers


def test_the_childless_refusal_carries_its_own_type():
    # The remote L3 runner catches exactly this refusal to supply a session-scoped backing instead,
    # so it has to stay distinguishable from every other way create_buffer fails.
    w = _bare_worker(3)
    with pytest.raises(_NoBufferConsumerError):
        w._create_buffer_locked(64)


def test_l2_leaf_needs_no_child():
    # An L2 leaf materializes in-process, so it has nothing to hand the backing to and needs no child.
    w = _bare_worker(2)
    try:
        assert w._create_buffer_locked(64).nbytes == 64
    finally:
        _drain(w)


def test_rejects_nonpositive_size():
    w = _bare_worker(2)
    for nbytes in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            w._create_buffer_locked(nbytes)
    assert not w._buffers


def test_buffer_ids_are_unique_within_one_incarnation():
    w = _bare_worker(2)
    try:
        ids = [w._create_buffer_locked(32).identity.buffer_id for _ in range(4)]
        assert len(set(ids)) == 4
        assert all(b.identity.generation == 1 for b in w._buffers.values())
        # One incarnation, one nonce: every buffer this Worker owns shares it.
        nonces = {b.identity.owner_instance_id for b in w._buffers.values()}
        assert len(nonces) == 1
    finally:
        _drain(w)


def test_release_all_buffers_unlinks_and_empties_the_registry():
    w = _bare_worker(2)
    w._create_buffer_locked(32)
    w._create_buffer_locked(32)
    w._release_all_buffers()
    assert not w._buffers


def test_release_buffer_drops_only_its_own_entry():
    w = _bare_worker(2)
    try:
        keep = w._create_buffer_locked(32)
        drop = w._create_buffer_locked(32)
        w.release_buffer(drop)
        assert drop.shm is None
        assert list(w._buffers.values()) == [keep]
        # A second release of the same buffer is a no-op, not a KeyError on the registry.
        w.release_buffer(drop)
        assert list(w._buffers.values()) == [keep]
    finally:
        _drain(w)


def test_release_buffer_keeps_the_entry_when_close_fails():
    # Same discipline as _release_all_buffers: a close that fails leaves the entry behind so the
    # leak is still reported at close() instead of being dropped here.
    w = _bare_worker(2)
    bad = w._create_buffer_locked(32)
    bad_id = next(k for k, v in w._buffers.items() if v is bad)
    real_close = bad.close

    def boom():
        raise OSError("close failed")

    bad.close = boom  # type: ignore[method-assign]
    with pytest.raises(OSError, match="close failed"):
        w.release_buffer(bad)
    assert bad_id in w._buffers
    bad.close = real_close  # type: ignore[method-assign]
    _drain(w)


def test_release_all_buffers_reports_the_failure_and_keeps_the_entry():
    # Per-buffer best-effort: a failing close must neither strand the others nor be swallowed, and
    # its registry entry stays so the cleanup journal can retry it.
    w = _bare_worker(2)
    ok = w._create_buffer_locked(32)
    bad = w._create_buffer_locked(32)
    bad_id = next(k for k, v in w._buffers.items() if v is bad)

    def boom():
        raise OSError("close failed")

    bad.close = boom  # type: ignore[method-assign]
    with pytest.raises(OSError, match="close failed"):
        w._release_all_buffers()
    assert bad_id in w._buffers  # retryable
    assert ok.shm is None  # the healthy one was still released
    w._buffers.clear()
    bad.shm.close()  # type: ignore[union-attr]
    bad.shm.unlink()  # type: ignore[union-attr]
