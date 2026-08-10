# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``Worker.release_buffer()``: rejects release while an in-flight run still references the
buffer's identity, and how that identity gets tracked in the first place.

L2 never needs this: a run completes synchronously inside ``submit()`` (``RunHandle._completed``),
so there is never an in-flight window. Tracking hooks the two L3+ async dispatch entry points,
``Orchestrator.submit_next_level`` / ``.submit_next_level_group``, device-free via the same
fake-C++-orchestrator harness ``test_child_addr_guard.py`` already established.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest
from _task_interface import DataType, TensorArgType
from simpler.buffer import create_host_shared_buffer, mint_owner_instance_id
from simpler.orchestrator import Orchestrator
from simpler.task_interface import TaskArgs
from simpler.worker import RunHandle, Worker, _RunResources

_F32 = 0  # DataType.FLOAT32 value
_OID = mint_owner_instance_id()


def _l3() -> Worker:
    return Worker(level=3, num_sub_workers=0, platform="a2a3sim", runtime="tensormap_and_ringbuffer")


def _buffer_args(buf) -> TaskArgs:
    args = TaskArgs()
    args.add_tensor(buf.tensor(shapes=(16,), dtype=DataType.FLOAT32), TensorArgType.INPUT)
    return args


def _fake_orchestrator(w: Worker, monkeypatch: pytest.MonkeyPatch) -> Orchestrator:
    import simpler.orchestrator as orch_mod  # noqa: PLC0415

    def _fake_require_handle(callable_handle, **_kw):
        return (b"d" * 32, "NEXT_LEVEL", "LOCAL_CHIP", (0,))

    monkeypatch.setattr(orch_mod, "_require_handle", _fake_require_handle)
    return Orchestrator(MagicMock(), w)


class TestTouchedIdentityTracking:
    def test_submit_next_level_records_touched_identity(self, monkeypatch):
        w = _l3()
        o = _fake_orchestrator(w, monkeypatch)
        w._building_run_resources = _RunResources()
        buf = create_host_shared_buffer(64, _OID, buffer_id=1)
        try:
            o.submit_next_level(object(), _buffer_args(buf), None, worker=0)
            assert buf.identity in w._building_run_resources.touched_identities
        finally:
            buf.close()

    def test_submit_next_level_group_records_touched_identity_per_member(self, monkeypatch):
        w = _l3()
        w._chip_shms = [object(), object()]
        o = _fake_orchestrator(w, monkeypatch)
        w._building_run_resources = _RunResources()
        buf_a = create_host_shared_buffer(64, _OID, buffer_id=2)
        buf_b = create_host_shared_buffer(64, _OID, buffer_id=3)
        try:
            o.submit_next_level_group(object(), [_buffer_args(buf_a), _buffer_args(buf_b)], None, workers=[0, 1])
            touched = w._building_run_resources.touched_identities
            assert buf_a.identity in touched
            assert buf_b.identity in touched
        finally:
            buf_a.close()
            buf_b.close()

    def test_no_current_run_is_a_no_op(self, monkeypatch):
        # submit_next_level runs fine with no _building_run_resources open (e.g. a direct call
        # outside a run's orchestration callback) -- tracking is opportunistic, not required.
        w = _l3()
        o = _fake_orchestrator(w, monkeypatch)
        assert w._building_run_resources is None
        buf = create_host_shared_buffer(64, _OID, buffer_id=4)
        try:
            o.submit_next_level(object(), _buffer_args(buf), None, worker=0)  # must not raise
        finally:
            buf.close()


def _in_flight_handle(w: Worker, identity, *, done: bool) -> RunHandle:
    resources = _RunResources()
    resources.touched_identities.add(identity)
    handle = RunHandle(w, run_id=1, keepalive=())
    handle._resources = resources
    handle._cleanup_published = done
    return handle


def _l3_with_registered_buffer(nbytes: int = 64):
    """An L3 Worker (one fake chip child, so create_buffer's topology gate passes) plus one
    buffer already registered in w._buffers, the way Worker.create_buffer() would leave it."""
    w = _l3()
    w._chip_shms = [object()]
    buf = w._create_buffer_locked(nbytes)
    return w, buf


class TestReleaseBuffer:
    def test_rejects_while_in_flight(self):
        w, buf = _l3_with_registered_buffer()
        w._accepted_run_handles.add(_in_flight_handle(w, buf.identity, done=False))
        with pytest.raises(RuntimeError, match="in-flight run"):
            w.release_buffer(buf)
        assert not buf.closed
        buffer_id = int(buf.identity.buffer_id)
        assert w._buffers.get(buffer_id) is buf  # rejection must not half-apply the registry drop
        w._accepted_run_handles.clear()
        w.release_buffer(buf)  # cleanup

    def test_succeeds_once_run_completed(self):
        w, buf = _l3_with_registered_buffer()
        w._accepted_run_handles.add(_in_flight_handle(w, buf.identity, done=True))
        w.release_buffer(buf)
        assert buf.closed
        assert int(buf.identity.buffer_id) not in w._buffers

    def test_succeeds_with_no_in_flight_runs(self):
        w, buf = _l3_with_registered_buffer()
        w.release_buffer(buf)
        assert buf.closed
        assert int(buf.identity.buffer_id) not in w._buffers

    def test_is_idempotent(self):
        w, buf = _l3_with_registered_buffer()
        w.release_buffer(buf)
        w.release_buffer(buf)  # must not raise
        assert buf.closed

    def test_serializes_with_a_racing_orchestration_callback(self):
        # _submit_l3_locked adds a run's handle to _accepted_run_handles (worker.py:9910) BEFORE
        # its orchestration callback runs -- the callback is what eventually records
        # touched_identities via submit_next_level. Without taking _submit_mu (the same lock that
        # already serializes graph construction, worker.py:9741), release_buffer could observe the
        # handle with an empty touched_identities mid-callback and release a Buffer the callback is
        # about to dispatch a Tensor over. This drives that exact window directly.
        w, buf = _l3_with_registered_buffer()
        resources = _RunResources()
        handle = RunHandle(w, run_id=1, keepalive=())
        handle._resources = resources
        handle._cleanup_published = False

        entered_callback = threading.Event()
        proceed = threading.Event()
        release_returned = threading.Event()

        def fake_orchestration_callback():
            with w._submit_mu:
                with w._hierarchical_start_cv:
                    w._accepted_run_handles.add(handle)
                entered_callback.set()
                proceed.wait(timeout=5)  # simulates the callback still building its graph
                # simulates submit_next_level's _record_touched_identities -- _submit_mu held
                # continuously from before the handle is visible until this point, exactly as
                # _submit_l3_locked holds it across the whole orchestration callback.
                resources.touched_identities.add(buf.identity)

        callback_thread = threading.Thread(target=fake_orchestration_callback)
        callback_thread.start()
        assert entered_callback.wait(timeout=5)

        def try_release():
            try:
                w.release_buffer(buf)
            except RuntimeError:
                pass
            finally:
                release_returned.set()

        releaser = threading.Thread(target=try_release)
        releaser.start()
        # release_buffer must block behind _submit_mu, not race past it while touched_identities
        # is still empty.
        assert not release_returned.wait(timeout=0.3)

        proceed.set()
        callback_thread.join(timeout=5)
        releaser.join(timeout=5)
        assert release_returned.is_set()
        assert not buf.closed  # correctly rejected once it could finally check -- not released mid-race
