# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""G2 — device-address guard ②: kind4 (child_memory) pointer provenance.

Device-free tests that pin the guard's contract: a child device allocation is
tracked by its ``(worker_id, base)`` provenance and byte extent. ``free`` and
kind4 dispatch require the exact base; ``copy_to`` / ``copy_from`` accept any
range wholly within a live allocation (a ``base + offset`` partial update), but
still reject a wrong-worker, stale, or out-of-range address. Covers every real
entry — ``Worker.malloc`` (L2), ``Worker.alloc_child_tensor`` / ``copy_to`` /
``copy_from`` / ``free`` (L3, which ``orch.*`` delegates to),
``submit_next_level`` / ``submit_next_level_group`` dispatch, and CommDomain
window pointers — plus the explicit boundary that strict ABA is NOT covered.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import simpler.orchestrator as orch_mod
from _task_interface import DataType, TensorArgType
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    Buffer,
    create_host_shared_buffer,
    mint_owner_instance_id,
    wrap_device_malloc,
    wrap_fork_inherited,
)
from simpler.orchestrator import Orchestrator
from simpler.task_interface import TaskArgs
from simpler.worker import Worker, _ChildProvEntry, _Lifecycle

_F32 = 0  # DataType.FLOAT32 value
_OID = mint_owner_instance_id()
# The host end of an L3 copy is a POSIX-shm Buffer: the forked child reaches it by name.
_HOSTSRC = create_host_shared_buffer(64, _OID, buffer_id=0xF0)


@pytest.fixture(scope="module", autouse=True)
def _release_module_host_buffer():
    yield
    _HOSTSRC.close()


def _l3() -> Worker:
    return Worker(level=3, num_sub_workers=0, platform="a2a3sim", runtime="tensormap_and_ringbuffer")


def _l3_ready(malloc_ret: int = 0x1000, *, chips: int = 2) -> tuple[Worker, MagicMock]:
    """An uninitialized L3 Worker forced READY with a mocked native worker, so the leased device-mem
    ops run in isolation. Returns ``(worker, native_worker_mock)``; the mock's ``malloc`` returns
    ``malloc_ret`` (the fabricated device ptr)."""
    w = _l3()
    w._lifecycle = _Lifecycle.READY
    w._chip_shms = [None] * chips  # type: ignore[list-item]
    nw = MagicMock()
    nw.malloc.return_value = malloc_ret
    w._worker = nw
    return w, nw


@contextlib.contextmanager
def _host_buf(nbytes: int, *, buffer_id: int = 0xF1):
    """A POSIX-shm host ``Buffer`` of ``nbytes``, unlinked on exit."""
    buf = create_host_shared_buffer(nbytes, _OID, buffer_id=buffer_id)
    try:
        yield buf
    finally:
        buf.close()


def _dev_handle(ptr: int, *, wid: int = 0, nbytes: int = 64) -> Buffer:
    """A DEVICE_MALLOC handle keyed at ``(wid, ptr)`` — the free/copy provenance key."""
    return wrap_device_malloc(ptr, nbytes, _OID, buffer_id=ptr, owner_worker_id=wid)


def _device_ptr(desc) -> int:
    """The device pointer a DEVICE_MALLOC / VMM_WINDOW descriptor carries in its backend body."""
    return int.from_bytes(desc.body, "little")


def _child_args(ptr: int, *, n: int = 16) -> TaskArgs:
    # A DEVICE_MALLOC ref carries ``ptr`` in its backend body: the device-pointer provenance key.
    args = TaskArgs()
    dev = wrap_device_malloc(ptr, n * 4, _OID, buffer_id=ptr)
    args.add_tensor(dev.tensor(shapes=(n,), dtype=_F32), TensorArgType.OUTPUT_EXISTING)
    return args


def _record_malloc(w: Worker, worker_id: int, ptr: int, size: int = 64) -> None:
    with w._child_prov_lock:
        w._child_prov_record_malloc(worker_id, ptr, size)


# ----------------------------------------------------------------------------
# Provenance table — typed exact (worker_id, ptr) entries
# ----------------------------------------------------------------------------


class TestProvenanceTable:
    def test_malloc_base_is_live_then_freed(self):
        w = _l3()
        with w._child_prov_lock:
            w._child_prov_record_malloc(0, 0x1000, 64)
            w._child_prov_require_live_range(0, 0x1000, 64, api="copy_to")  # no raise
            w._child_prov_require_malloc_base(0, 0x1000, api="free")  # no raise
            w._child_prov_clear_malloc(0, 0x1000)
            with pytest.raises(ValueError, match="not contained in a live allocation"):
                w._child_prov_require_live_range(0, 0x1000, 64, api="copy_to")

    def test_free_of_unknown_pointer_rejected(self):
        w = _l3()
        with w._child_prov_lock, pytest.raises(ValueError, match="not a live malloc base"):
            w._child_prov_require_malloc_base(0, 0xDEAD, api="free")

    def test_double_free_stale_before_reuse_rejected(self):
        w = _l3()
        with w._child_prov_lock:
            w._child_prov_record_malloc(0, 0x1000, 64)
            w._child_prov_clear_malloc(0, 0x1000)
            with pytest.raises(ValueError, match="already-freed/stale"):
                w._child_prov_require_malloc_base(0, 0x1000, api="free")

    def test_interior_range_is_accepted_edges_rejected(self):
        w = _l3()
        with w._child_prov_lock:
            w._child_prov_record_malloc(0, 0x1000, 64)
            # A range wholly inside the allocation is a valid partial update.
            w._child_prov_require_live_range(0, 0x1020, 16, api="copy_to")
            w._child_prov_require_live_range(0, 0x1000, 64, api="copy_to")  # exact fill
            # One byte past the end overruns the allocation.
            with pytest.raises(ValueError, match="not contained in a live allocation"):
                w._child_prov_require_live_range(0, 0x103F, 2, api="copy_to")
            # A base below the allocation is out of range too.
            with pytest.raises(ValueError, match="not contained in a live allocation"):
                w._child_prov_require_live_range(0, 0x0FFF, 4, api="copy_to")

    def test_free_still_requires_exact_base_not_interior(self):
        # copy accepts interior ranges, but free must still hit the exact base.
        w = _l3()
        with w._child_prov_lock:
            w._child_prov_record_malloc(0, 0x1000, 64)
            with pytest.raises(ValueError, match="not a live malloc base"):
                w._child_prov_require_malloc_base(0, 0x1020, api="free")

    def test_same_numeric_va_on_two_workers_is_independent(self):
        # A raw device VA is not globally unique; the composite key keeps two
        # chips' identical numeric addresses independent.
        w = _l3()
        with w._child_prov_lock:
            w._child_prov_record_malloc(0, 0x4000, 64)
            w._child_prov_record_malloc(1, 0x4000, 64)
            w._child_prov_clear_malloc(0, 0x4000)
            with pytest.raises(ValueError, match="not contained in a live allocation"):
                w._child_prov_require_live_range(0, 0x4000, 64, api="copy_to")
            w._child_prov_require_live_range(1, 0x4000, 64, api="copy_to")  # survives

    def test_domain_pointer_is_not_freeable_but_is_dispatchable(self):
        w = _l3()
        with w._child_prov_lock:
            w._child_prov_record_domain(0, 0x8000, allocation_id=7, extent=64)
            # usable for copy / dispatch
            w._child_prov_require_live_range(0, 0x8000, 64, api="copy_to")
            # but not free-able — domains are revoked by release, not free()
            with pytest.raises(ValueError, match="CommDomain buffer"):
                w._child_prov_require_malloc_base(0, 0x8000, api="free")
            w._child_prov_drop_domain(7)
            with pytest.raises(ValueError, match="not contained in a live allocation"):
                w._child_prov_require_live_range(0, 0x8000, 64, api="copy_to")

    def test_malloc_and_domain_alias_same_pointer(self):
        # A malloc base and a domain buffer can alias the same (worker, ptr).
        # Clearing the malloc role leaves it live via the domain; only dropping
        # the domain removes it.
        w = _l3()
        with w._child_prov_lock:
            w._child_prov_record_malloc(0, 0x9000, 64)
            w._child_prov_record_domain(0, 0x9000, allocation_id=3, extent=64)
            w._child_prov_clear_malloc(0, 0x9000)
            w._child_prov_require_live_range(0, 0x9000, 64, api="copy_to")  # still live via domain
            w._child_prov_drop_domain(3)
            with pytest.raises(ValueError, match="not contained in a live allocation"):
                w._child_prov_require_live_range(0, 0x9000, 64, api="copy_to")

    def test_empty_entry_is_not_live_fail_closed(self):
        # A role-less entry (e.g. one left by an interrupted revoke) must never be
        # treated as live — every check is fail-closed on the roles, not on the
        # key's mere presence.
        w = _l3()
        w._child_alloc_prov[(0, 0x1000)] = _ChildProvEntry()  # both roles empty
        with w._child_prov_lock:
            with pytest.raises(ValueError, match="not contained in a live allocation"):
                w._child_prov_require_live_range(0, 0x1000, 64, api="copy_to")
            with pytest.raises(ValueError, match="not a live malloc base"):
                w._child_prov_require_malloc_base(0, 0x1000, api="free")
            with pytest.raises(ValueError, match="not a live allocation on target worker 0"):
                w._child_prov_check_dispatch([(0x1000, 0)], 0, api="submit_next_level")

    def test_drop_last_role_deletes_entry_no_empty_state(self):
        # Dropping the last role deletes the key outright — it never leaves a
        # role-less entry behind (which a later live check could not tell from a
        # live one without the fail-closed guard).
        w = _l3()
        with w._child_prov_lock:
            w._child_prov_record_domain(0, 0x5000, allocation_id=1, extent=64)
            w._child_prov_drop_domain(1)
            assert (0, 0x5000) not in w._child_alloc_prov
            w._child_prov_record_malloc(0, 0x6000, 64)
            w._child_prov_clear_malloc(0, 0x6000)
            assert (0, 0x6000) not in w._child_alloc_prov

    def test_strict_aba_is_explicitly_not_covered(self):
        # Documentation test: after free + re-malloc of the same numeric VA, the
        # (worker, ptr) becomes live again and an old handle is indistinguishable
        # from the new allocation. Strict ABA is deferred to P1 generation
        # handles; this guard only catches stale-BEFORE-reuse.
        w = _l3()
        with w._child_prov_lock:
            w._child_prov_record_malloc(0, 0x1000, 64)
            w._child_prov_clear_malloc(0, 0x1000)
            w._child_prov_record_malloc(0, 0x1000, 64)  # device handed back the same VA
            w._child_prov_require_live_range(0, 0x1000, 64, api="copy_to")  # NOT rejected — by design


# ----------------------------------------------------------------------------
# Target resolution + dispatch check
# ----------------------------------------------------------------------------


class TestDispatchResolution:
    def test_unique_target_and_live_passes(self):
        w = _l3()
        _record_malloc(w, 0, 0x1000)
        with w._child_prov_lock:
            w._child_prov_check_dispatch([(0x1000, 0)], 0, api="submit_next_level")

    def test_unique_target_but_wrong_worker_rejected(self):
        w = _l3()
        _record_malloc(w, 0, 0x1000)  # lives on worker 0
        with w._child_prov_lock, pytest.raises(ValueError, match="not a live allocation on target worker 1"):
            w._child_prov_check_dispatch([(0x1000, 0)], 1, api="submit_next_level")


# ----------------------------------------------------------------------------
# Worker device memory — alloc_child_tensor / free / copy on a next-level worker
# (the Worker is the sole allocator; orch.* are thin delegates)
# ----------------------------------------------------------------------------


class TestChildDeviceMemoryOps:
    def test_alloc_child_records_then_free_clears(self):
        w, nw = _l3_ready(0x1000)
        h = w.alloc_child_tensor(0, (64,), DataType.UINT8)
        assert h.base == 0x1000
        assert (0, 0x1000) in w._child_alloc_prov
        w.free(h)
        nw.free.assert_called_once_with(0, 0x1000)
        assert (0, 0x1000) not in w._child_alloc_prov

    def test_free_wrong_worker_rejected_without_native_free(self):
        w, nw = _l3_ready(0x1000)
        w.alloc_child_tensor(1, (64,), DataType.UINT8)  # allocated on worker 1
        with pytest.raises(ValueError, match="not a live malloc base"):
            w.free(_dev_handle(0x1000, wid=0))  # freed on worker 0
        nw.free.assert_not_called()

    def test_copy_to_requires_live_device_dst(self):
        w, nw = _l3_ready(0x2000)
        with pytest.raises(ValueError, match="not contained in a live allocation"):
            w.copy_to(_dev_handle(0x2000, wid=0), _HOSTSRC)
        nw.copy_to.assert_not_called()
        w.alloc_child_tensor(0, (64,), DataType.UINT8)
        w.copy_to(_dev_handle(0x2000, wid=0), _HOSTSRC)
        nw.copy_to.assert_called_once()

    def test_copy_to_interior_range_within_allocation_passes(self):
        # A partial update wholly inside a live allocation is valid (issue #1537).
        w, nw = _l3_ready(0x2000)
        w.alloc_child_tensor(0, (64,), DataType.UINT8)
        with _host_buf(16) as src:
            w.copy_to(_dev_handle(0x2020, wid=0), src)  # base + 32, 16 bytes — inside [0x2000, 0x2040)
        nw.copy_to.assert_called_once()
        # copy_to(worker_id, dst_descriptor, src_descriptor, nbytes)
        assert _device_ptr(nw.copy_to.call_args.args[1]) == 0x2020
        assert nw.copy_to.call_args.args[3] == 16

    def test_copy_to_range_overrunning_allocation_rejected(self):
        w, nw = _l3_ready(0x2000)
        w.alloc_child_tensor(0, (64,), DataType.UINT8)
        with _host_buf(2) as src, pytest.raises(ValueError, match="not contained in a live allocation"):
            w.copy_to(_dev_handle(0x203F, wid=0), src)  # last byte at 0x2040 == base + size
        nw.copy_to.assert_not_called()

    def test_copy_to_domain_window_chunk_not_narrowed_by_aliasing_buffer(self):
        # A carved buffer at offset 0 aliases the window base under the same
        # allocation id; recording its smaller extent must not shrink the
        # window's copy range (a large chunk into the window must still pass).
        w, nw = _l3_ready()
        with w._child_prov_lock:
            w._child_prov_record_domain(0, 0x4000, allocation_id=7, extent=2 << 20)
            w._child_prov_record_domain(0, 0x4000, allocation_id=7, extent=4)  # buffer #0 aliases the base
        # Minted one byte wider than the window so the handle's own length check never fires and
        # the recorded provenance extent is the guard under test.
        window = _dev_handle(0x4000, wid=0, nbytes=(2 << 20) + 1)
        with _host_buf(1 << 20) as chunk:
            w.copy_to(window, chunk)  # 1 MiB into a 2 MiB window
        nw.copy_to.assert_called_once()
        assert nw.copy_to.call_args.args[3] == 1 << 20
        with (
            _host_buf((2 << 20) + 1) as oversized,
            pytest.raises(ValueError, match="not contained in a live allocation"),
        ):
            w.copy_to(window, oversized)  # one byte past the window
        assert nw.copy_to.call_count == 1

    def test_copy_from_requires_live_device_src(self):
        w, nw = _l3_ready(0x3000)
        with pytest.raises(ValueError, match="not contained in a live allocation"):
            w.copy_from(_HOSTSRC, _dev_handle(0x3000, wid=0))
        nw.copy_from.assert_not_called()

    def test_orch_delegates_to_worker(self):
        # orch.alloc_child_tensor / free are thin wrappers over the bound Worker.
        w, nw = _l3_ready(0x1000)
        o = Orchestrator(MagicMock(), w)
        h = o.alloc_child_tensor(0, (64,), DataType.UINT8)
        assert (0, 0x1000) in w._child_alloc_prov
        o.free(h)
        nw.free.assert_called_once_with(0, 0x1000)
        assert (0, 0x1000) not in w._child_alloc_prov

    def test_orch_without_worker_rejects_memory_ops(self):
        # A worker-less Orchestrator can't allocate/free/copy — the impl lives on the Worker.
        o = Orchestrator(MagicMock(), None)
        with pytest.raises(RuntimeError, match="requires a Worker context"):
            o.free(_dev_handle(0x1000))


# ----------------------------------------------------------------------------
# submit_next_level / group dispatch guard
# ----------------------------------------------------------------------------


@pytest.fixture
def _fake_handle(monkeypatch):
    """Patch _require_handle so submit_* can run device-free with a chosen
    eligible set; returns a setter for the eligible worker ids."""
    state = {"eligible": (0,)}

    def _fake(callable_handle, **_kwargs):
        return (b"d" * 32, "NEXT_LEVEL", "LOCAL_CHIP", state["eligible"])

    monkeypatch.setattr(orch_mod, "_require_handle", _fake)
    return state


class TestSubmitDispatchGuard:
    def test_child_arg_to_correct_worker_passes(self, _fake_handle):
        w = _l3()
        _record_malloc(w, 0, 0x1000)
        fake = MagicMock()
        o = Orchestrator(fake, w)
        o.submit_next_level(object(), _child_args(0x1000), None, worker=0)
        fake.submit_next_level.assert_called_once()

    def test_child_arg_to_wrong_worker_rejected(self, _fake_handle):
        w = _l3()
        w._chip_shms = [object(), object()]
        _record_malloc(w, 0, 0x1000)
        fake = MagicMock()
        o = Orchestrator(fake, w)
        with pytest.raises(ValueError, match="not a live allocation on target worker 1"):
            o.submit_next_level(object(), _child_args(0x1000), None, worker=1)
        fake.submit_next_level.assert_not_called()

    def test_host_only_args_are_not_guarded(self, _fake_handle):
        # A submit with no device (DEVICE_MALLOC) ref never touches provenance.
        w = _l3()
        fake = MagicMock()
        o = Orchestrator(fake, w)
        args = TaskArgs()
        host = wrap_fork_inherited(0x9000, 64, _OID, buffer_id=0x9000)
        args.add_tensor(host.tensor(shapes=(16,), dtype=_F32), TensorArgType.INPUT)
        o.submit_next_level(object(), args, None, worker=0)
        fake.submit_next_level.assert_called_once()

    def test_group_member_child_arg_wrong_worker_rejected(self, _fake_handle):
        w = _l3()
        w._chip_shms = [object(), object()]
        _record_malloc(w, 0, 0x1000)
        fake = MagicMock()
        o = Orchestrator(fake, w)
        # member 0 carries the child ptr live on worker 0, but is pinned to worker 1
        with pytest.raises(ValueError, match="not a live allocation on target worker 1"):
            o.submit_next_level_group(object(), [_child_args(0x1000), TaskArgs()], None, workers=[1, 0])
        fake.submit_next_level_group.assert_not_called()

    def test_group_member_child_arg_correct_worker_passes(self, _fake_handle):
        w = _l3()
        w._chip_shms = [object(), object()]
        _record_malloc(w, 1, 0x1000)
        fake = MagicMock()
        o = Orchestrator(fake, w)
        o.submit_next_level_group(object(), [TaskArgs(), _child_args(0x1000)], None, workers=[0, 1])
        fake.submit_next_level_group.assert_called_once()

    def test_group_child_member_rejects_mismatched_workers_length(self, _fake_handle):
        # A non-empty workers list must be one-per-member; a short list must NOT
        # be silently padded (that would bypass the C++ length check).
        w = _l3()
        w._chip_shms = [object(), object()]
        _record_malloc(w, 0, 0x1000)
        fake = MagicMock()
        o = Orchestrator(fake, w)
        with pytest.raises(ValueError, match="workers length must match"):
            o.submit_next_level_group(object(), [_child_args(0x1000), TaskArgs()], None, workers=[0])
        fake.submit_next_level_group.assert_not_called()

    def test_local_callable_rejects_remote_worker_target(self, _fake_handle):
        # A LOCAL_CHIP callable pinned to a remote worker id would enqueue on the
        # remote endpoint, whose manifest lacks the local digest -> async unknown
        # hashid. The Python guard rejects it up front; a local chip id passes.
        w = _l3()
        w._chip_shms = [object()]
        w._remote_worker_ids = [7]
        fake = MagicMock()
        o = Orchestrator(fake, w)
        with pytest.raises(ValueError, match="remote NEXT_LEVEL worker"):
            o.submit_next_level(object(), TaskArgs(), None, worker=7)
        fake.submit_next_level.assert_not_called()
        o.submit_next_level(object(), TaskArgs(), None, worker=0)
        fake.submit_next_level.assert_called_once()

    def test_local_group_rejects_remote_worker_target(self, _fake_handle):
        w = _l3()
        w._chip_shms = [object()]
        w._remote_worker_ids = [7]
        fake = MagicMock()
        o = Orchestrator(fake, w)
        with pytest.raises(ValueError, match="remote NEXT_LEVEL worker"):
            o.submit_next_level_group(object(), [TaskArgs(), TaskArgs()], None, workers=[0, 7])
        fake.submit_next_level_group.assert_not_called()

    def test_domain_pointer_dispatch_to_owner_then_rejected_after_release(self, _fake_handle):
        # CommDomain window pointers enter provenance so they dispatch as kind4
        # to their owning chip; a release revokes them.
        w = _l3()
        w._chip_shms = [object(), object()]
        with w._child_prov_lock:
            w._child_prov_record_domain(0, 0x5000, allocation_id=42, extent=64)
        fake = MagicMock()
        o = Orchestrator(fake, w)
        o.submit_next_level(object(), _child_args(0x5000), None, worker=0)
        fake.submit_next_level.assert_called_once()
        # wrong chip is rejected
        with pytest.raises(ValueError, match="target worker 1"):
            o.submit_next_level(object(), _child_args(0x5000), None, worker=1)
        # after release the pointer is dead everywhere
        with w._child_prov_lock:
            w._child_prov_drop_domain(42)
        with pytest.raises(ValueError, match="not a live allocation"):
            o.submit_next_level(object(), _child_args(0x5000), None, worker=0)


# ----------------------------------------------------------------------------
# L2 Worker.malloc/free/copy path (direct to the single chip)
# ----------------------------------------------------------------------------


class TestL2WorkerPath:
    def _l2(self) -> tuple[Worker, MagicMock]:
        w = Worker(level=2, platform="a2a3sim", runtime="tensormap_and_ringbuffer", device_id=0)
        chip = MagicMock()
        chip.malloc.return_value = 0x2000
        w._chip_worker = chip
        w._lifecycle = _Lifecycle.READY
        return w, chip

    def test_l2_malloc_records_and_free_clears(self):
        w, chip = self._l2()
        h = w.malloc(64)
        assert h.base == 0x2000
        assert (0, 0x2000) in w._child_alloc_prov
        w.free(h)
        chip.free.assert_called_once_with(0x2000)
        assert (0, 0x2000) not in w._child_alloc_prov

    def test_l2_free_stale_rejected_without_native_free(self):
        w, chip = self._l2()
        h = w.malloc(64)
        w.free(h)
        chip.free.reset_mock()
        with pytest.raises(ValueError, match="already-freed/stale"):
            w.free(h)
        chip.free.assert_not_called()

    def test_l2_free_revokes_before_native_free(self):
        # L2 mirrors the L3 commit barrier: revoke before the native free.
        w, chip = self._l2()
        h = w.malloc(64)
        seen = {}
        chip.free.side_effect = lambda p: seen.__setitem__("live_at_native", (0, 0x2000) in w._child_alloc_prov)
        w.free(h)
        assert seen["live_at_native"] is False

    def test_l2_copy_to_requires_live_dst(self):
        w, chip = self._l2()
        with pytest.raises(ValueError, match="not contained in a live allocation"):
            w.copy_to(_dev_handle(0x2000, wid=0), _HOSTSRC)
        chip.copy_to.assert_not_called()
        w.malloc(64)
        w.copy_to(_dev_handle(0x2000, wid=0), _HOSTSRC)
        chip.copy_to.assert_called_once()

    def test_l2_copy_to_interior_range_passes(self):
        # base + offset partial update on the single chip (issue #1537).
        w, chip = self._l2()
        w.malloc(64)  # chip.malloc returns 0x2000
        w.copy_to(_dev_handle(0x2020, wid=0), bytearray(16))
        chip.copy_to.assert_called_once()
        assert chip.copy_to.call_args.args[0] == 0x2020
        assert chip.copy_to.call_args.args[2] == 16

    def test_l2_free_of_interior_pointer_rejected(self):
        w, chip = self._l2()
        w.malloc(64)
        with pytest.raises(ValueError, match="not a live malloc base"):
            w.free(_dev_handle(0x2020, wid=0))
        chip.free.assert_not_called()


# ----------------------------------------------------------------------------
# Provenance transaction failures (record after alloc success; free revokes
# before the native free — safety-first, terminal-leak on failure)
# ----------------------------------------------------------------------------


class TestProvenanceTransactions:
    def test_alloc_child_native_error_records_nothing(self):
        # Provenance is recorded only after the backend malloc succeeds.
        w, nw = _l3_ready()
        nw.malloc.side_effect = RuntimeError("device OOM")
        with pytest.raises(RuntimeError, match="device OOM"):
            w.alloc_child_tensor(0, (64,), DataType.UINT8)
        assert w._child_alloc_prov == {}

    def test_free_revokes_before_native_free(self):
        # Safety-first commit barrier: provenance is revoked BEFORE the native
        # free, so an async unwind after a successful free cannot leave a freed
        # address live.
        w, nw = _l3_ready(0x1000)
        h = w.alloc_child_tensor(0, (64,), DataType.UINT8)
        seen = {}
        nw.free.side_effect = lambda wid, p: seen.__setitem__("live_at_native", (0, 0x1000) in w._child_alloc_prov)
        w.free(h)
        assert seen["live_at_native"] is False  # already revoked when native free runs

    def test_free_native_error_revokes_provenance_safe_first(self):
        # A native free that fails becomes a terminal leak — provenance is
        # revoked, never re-authorized. No retry (the address is no longer a
        # live malloc base).
        w, nw = _l3_ready(0x1000)
        h = w.alloc_child_tensor(0, (64,), DataType.UINT8)
        nw.free.side_effect = RuntimeError("free failed")
        with pytest.raises(RuntimeError, match="free failed"):
            w.free(h)
        assert (0, 0x1000) not in w._child_alloc_prov  # revoked (terminal leak)
        nw.free.side_effect = None
        with pytest.raises(ValueError, match="not a live malloc base"):
            w.free(h)

    def test_free_holds_lock_across_native_free(self):
        # Deterministic mutual-exclusion check, in the narrower form this exclusion
        # now takes: the native free runs under *that worker's* lock, so a
        # concurrent free/copy/dispatch on the same chip still cannot interleave
        # with a half-completed free — but `_child_prov_lock` is released, so a
        # slow free on one chip no longer blocks provenance work for another.
        # Safe because provenance is keyed by (worker_id, ptr) and the revoke
        # commits first: a concurrent dispatch reading the table under
        # `_child_prov_lock` finds this address already gone, or is about a
        # different chip entirely.
        w, nw = _l3_ready(0x1000)
        h = w.alloc_child_tensor(0, (64,), DataType.UINT8)
        seen = {}

        def _sf(wid, p):
            worker_lock = w._child_prov_worker_lock(wid)
            free_to_take = worker_lock.acquire(blocking=False)
            seen["worker_lock_held"] = not free_to_take
            if free_to_take:
                worker_lock.release()
            shared_free = w._child_prov_lock.acquire(blocking=False)
            seen["shared_lock_released"] = shared_free
            if shared_free:
                w._child_prov_lock.release()
            seen["revoke_committed"] = (wid, p) not in w._child_alloc_prov

        nw.free.side_effect = _sf
        w.free(h)
        assert seen["worker_lock_held"] is True
        assert seen["shared_lock_released"] is True
        assert seen["revoke_committed"] is True

    def test_capture_refs_after_provenance_analysis(self, _fake_handle, monkeypatch):
        # blocker: the kind4 provenance analysis must run BEFORE remote slot refs
        # are captured, so an analysis failure cannot strand captured refs outside
        # the rollback try (deferring a remote free forever).
        w = _l3()
        o = Orchestrator(MagicMock(), w)
        monkeypatch.setattr(w, "_child_ptrs_in_args", MagicMock(side_effect=RuntimeError("boom")))
        cap = MagicMock(return_value=[])
        monkeypatch.setattr(w, "_capture_remote_sidecar_refs", cap)
        with pytest.raises(RuntimeError, match="boom"):
            o.submit_next_level(object(), _child_args(0x1000), None, worker=0)
        cap.assert_not_called()

    def test_l2_malloc_native_error_records_nothing(self):
        w = Worker(level=2, platform="a2a3sim", runtime="tensormap_and_ringbuffer", device_id=0)
        chip = MagicMock()
        chip.malloc.side_effect = RuntimeError("device OOM")
        w._chip_worker = chip
        w._lifecycle = _Lifecycle.READY
        with pytest.raises(RuntimeError, match="device OOM"):
            w.malloc(64)
        assert w._child_alloc_prov == {}


# ----------------------------------------------------------------------------
# CommDomain physical release — revoke before backend free (commit barrier)
# ----------------------------------------------------------------------------


class TestDomainReleaseOrdering:
    def _domain_worker(self):
        """A level-3 Worker with a recorded domain and a native worker present."""
        w = _l3()
        w._worker = MagicMock()  # non-None so _release_domain_now proceeds
        with w._child_prov_lock:
            w._child_prov_record_domain(0, 0x5000, allocation_id=9, extent=64)
            w._child_prov_record_domain(1, 0x6000, allocation_id=9, extent=64)
        handle = SimpleNamespace(
            name="d",
            workers=(0, 1),
            allocation_id=9,
            _domain_size=2,
            _domain_ranks={0: 0, 1: 1},
        )
        return w, handle

    def test_release_revokes_provenance_before_backend_free(self, monkeypatch):
        w, handle = self._domain_worker()
        seen_live_at_dispatch = {}

        def _fake_dispatch(**kwargs):
            # At physical-free time the pointers must already be revoked.
            seen_live_at_dispatch["still_live"] = (0, 0x5000) in w._child_alloc_prov

        monkeypatch.setattr(w, "_dispatch_control_domain", _fake_dispatch)
        w._release_domain_now(handle)  # type: ignore[arg-type]
        assert seen_live_at_dispatch["still_live"] is False
        assert (0, 0x5000) not in w._child_alloc_prov
        assert (1, 0x6000) not in w._child_alloc_prov

    def test_release_vs_dispatch_rejects_during_native_release(self, monkeypatch):
        # Deterministic domain-release-vs-dispatch: by the time the backend free
        # runs, the pointer is already revoked, so a dispatch that lands during
        # the native release is rejected — no freed-but-live window.
        w, handle = self._domain_worker()
        outcome = {}

        def _fake_dispatch(**kwargs):
            try:
                with w._child_prov_lock:
                    w._child_prov_check_dispatch([(0x5000, 0)], 0, api="submit_next_level")
                outcome["dispatch"] = "allowed"
            except ValueError:
                outcome["dispatch"] = "rejected"

        monkeypatch.setattr(w, "_dispatch_control_domain", _fake_dispatch)
        w._release_domain_now(handle)  # type: ignore[arg-type]
        assert outcome["dispatch"] == "rejected"

    def test_release_backend_failure_leaves_provenance_dropped(self, monkeypatch):
        # A partial/failed backend release must not restore the pointers to live
        # (a leak is safe; a use-after-free is not).
        w, handle = self._domain_worker()

        def _boom(**kwargs):
            raise RuntimeError("release failed on one chip")

        monkeypatch.setattr(w, "_dispatch_control_domain", _boom)
        with pytest.raises(RuntimeError, match="release failed"):
            w._release_domain_now(handle)  # type: ignore[arg-type]
        assert (0, 0x5000) not in w._child_alloc_prov
        assert (1, 0x6000) not in w._child_alloc_prov


# ----------------------------------------------------------------------------
# copy_to / copy_from handle validation
#
# The transfer length comes from the *host* object, so without these checks a host buffer larger
# than the device backing writes past it, and a READ-only backing accepts a write.
# ----------------------------------------------------------------------------


class TestCopyHandleValidation:
    def test_rejects_host_handle(self):
        w, nw = _l3_ready(0x1000)
        host = create_host_shared_buffer(64, _OID, buffer_id=1)
        try:
            with pytest.raises(ValueError, match="expected a DEVICE handle"):
                w.copy_to(host, _HOSTSRC)
            nw.copy_to.assert_not_called()
        finally:
            host.close()

    def test_rejects_transfer_larger_than_the_backing(self):
        w, nw = _l3_ready(0x2000)
        w.alloc_child_tensor(0, (32,), DataType.UINT8)  # a 32-byte backing
        with pytest.raises(ValueError, match="exceeds the 32-byte backing"):
            w.copy_to(_dev_handle(0x2000, wid=0, nbytes=32), _HOSTSRC)  # _HOSTSRC is 64 B
        nw.copy_to.assert_not_called()

    def test_rejects_write_to_a_read_only_backing(self):
        w, nw = _l3_ready(0x2000)
        w.alloc_child_tensor(0, (64,), DataType.UINT8)
        ro = wrap_device_malloc(0x2000, 64, _OID, buffer_id=0x2000, owner_worker_id=0, access=AccessMode.READ)
        with pytest.raises(ValueError, match="needs WRITE"):
            w.copy_to(ro, _HOSTSRC)
        nw.copy_to.assert_not_called()
        w.copy_from(_HOSTSRC, ro)  # reading it is fine
        nw.copy_from.assert_called_once()

    def test_l3_copy_sends_both_ends_as_handles(self):
        # The owner's mapped address means nothing in the child, which resolves each descriptor
        # through its own ImportRegistry — so an L3 copy sends handles, never an address.
        w, nw = _l3_ready(0x2000)
        w.alloc_child_tensor(0, (64,), DataType.UINT8)
        w.copy_to(_dev_handle(0x2000, wid=0), _HOSTSRC)
        (wid, dst_desc, src_desc, size) = nw.copy_to.call_args.args
        assert (wid, size) == (0, _HOSTSRC.nbytes)
        assert _device_ptr(dst_desc) == 0x2000
        assert src_desc.identity == _HOSTSRC.identity
        assert src_desc.address_space == AddressSpace.HOST


class TestCopyHandleTransport:
    """Both ends of a copy travel as descriptors the child resolves for itself."""

    def test_shm_handle_src_carries_its_own_backing(self):
        w, nw = _l3_ready(0x2000)
        w.alloc_child_tensor(0, (64,), DataType.UINT8)
        host = create_host_shared_buffer(64, _OID, buffer_id=1)
        host_shm = host.shm
        assert host_shm is not None
        try:
            w.copy_to(_dev_handle(0x2000, wid=0), host)
            (_wid, dst_desc, src_desc, size) = nw.copy_to.call_args.args
            assert src_desc.body.decode() == host_shm.name  # the caller's own shm, not a duplicate
            assert src_desc.backend_kind == BackendKind.POSIX_SHM
            assert (_device_ptr(dst_desc), size) == (0x2000, 64)
        finally:
            host.close()

    def test_rejects_raw_host_memory_at_l3(self):
        # Raw host memory has only an address, and an L3 copy runs in a forked child where that
        # address means nothing. There is no fallback that copies it somewhere reachable.
        w, nw = _l3_ready(0x2000)
        w.alloc_child_tensor(0, (64,), DataType.UINT8)
        with pytest.raises(TypeError, match="must be a Buffer from create_buffer"):
            w.copy_to(_dev_handle(0x2000, wid=0), bytearray(64))
        with pytest.raises(TypeError, match="must be a Buffer from create_buffer"):
            w.copy_from(bytearray(64), _dev_handle(0x2000, wid=0))
        nw.copy_to.assert_not_called()
        nw.copy_from.assert_not_called()

    def test_rejects_a_device_handle_on_the_host_side(self):
        w, _nw = _l3_ready(0x2000)
        w.alloc_child_tensor(0, (64,), DataType.UINT8)
        with pytest.raises(ValueError, match="host side must be a HOST handle"):
            w.copy_to(_dev_handle(0x2000, wid=0), _dev_handle(0x2000, wid=0))

    def test_rejects_writing_into_a_read_only_host_handle(self):
        w, _nw = _l3_ready(0x3000)
        w.alloc_child_tensor(0, (64,), DataType.UINT8)
        ro = create_host_shared_buffer(64, _OID, buffer_id=2, access=AccessMode.READ)
        try:
            with pytest.raises(ValueError, match="host backing grants READ"):
                w.copy_from(ro, _dev_handle(0x3000, wid=0))
        finally:
            ro.close()
