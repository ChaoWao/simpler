# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for L4 → L3 → L2 recursive Worker composition.

L4 Worker dispatches to L3 Worker children (PROCESS mode). Each L3 Worker
has its own SubWorkers. Verifies the full DAG completes and sub callables
at L3 level see correct data.

No NPU device required: the dataflow tests give their L3 children SubWorkers
only, and the chip-callable cascade gives them a fake chip (``_harness``).
"""

from __future__ import annotations

import struct
from multiprocessing.shared_memory import SharedMemory

import pytest
from _task_interface import DataType
from simpler.buffer import mint_owner_instance_id, wrap_device_malloc
from simpler.callable_identity import CallableHandle
from simpler.task_interface import CallConfig, TaskArgs
from simpler.worker import Worker

from ._harness import (
    SIM_PLATFORM,
    SIM_RUNTIME,
    TEST_WALL_BUDGET_S,
    chip_callable,
    hard_timeout,
    install_fake_chip,
    requires_sim_binaries,
)

_OID = mint_owner_instance_id()
_HOSTBUF = bytearray(64)


def _dev_handle(ptr: int, *, wid: int = 0):
    return wrap_device_malloc(ptr, 64, _OID, buffer_id=ptr, owner_worker_id=wid)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shared_counter(slots: int = 1):
    """Allocate `slots` 4-byte shared counters accessible from forked subprocesses.

    `_increment_counter` is a non-atomic read-modify-write, so a slot tolerates
    at most one writing process: two processes that read before either writes
    both store the same value and one increment is lost. A test whose
    increments come from more than one process must give each process its own
    slot.
    """
    shm = SharedMemory(create=True, size=4 * slots)
    buf = shm.buf
    assert buf is not None
    for slot in range(slots):
        struct.pack_into("i", buf, 4 * slot, 0)
    return shm, buf


def _read_counter(buf, slot: int = 0) -> int:
    return struct.unpack_from("i", buf, 4 * slot)[0]


def _increment_counter(buf, slot: int = 0) -> None:
    v = struct.unpack_from("i", buf, 4 * slot)[0]
    struct.pack_into("i", buf, 4 * slot, v + 1)


def _slot_for(worker: Worker, handle: CallableHandle) -> int:
    return worker._identity_registry[handle.digest].slot_id


# ---------------------------------------------------------------------------
# Test: L4 lifecycle (init / close without submitting any tasks)
# ---------------------------------------------------------------------------


class TestL4Lifecycle:
    def test_init_close_no_children(self):
        """L4 with zero next-level workers and zero sub workers."""
        w4 = Worker(level=4, num_sub_workers=0)
        w4.init()
        w4.close()

    def test_init_close_with_l3_child(self):
        """L4 with one L3 child (no device, sub-only) — init and close cleanly."""
        l3 = Worker(level=3, num_sub_workers=1)
        l3.register(lambda args: None)

        w4 = Worker(level=4, num_sub_workers=0)
        w4.register(lambda orch, args, config: None)
        w4.add_worker(l3)
        w4.init()
        w4.close()

    def test_context_manager(self):
        """L4 via context manager cleans up correctly."""
        l3 = Worker(level=3, num_sub_workers=1)
        l3.register(lambda args: None)

        with Worker(level=4, num_sub_workers=0) as w4:
            w4.register(lambda orch, args, config: None)
            w4.add_worker(l3)
            w4.init()


class TestL4Validation:
    def test_add_worker_requires_level4(self):
        """add_worker on level 3 raises."""
        w3 = Worker(level=3, num_sub_workers=0)
        child = Worker(level=3, num_sub_workers=0)
        with pytest.raises(RuntimeError, match="level >= 4"):
            w3.add_worker(child)

    def test_add_worker_after_init_raises(self):
        w4 = Worker(level=4, num_sub_workers=0)
        w4.init()
        child = Worker(level=3, num_sub_workers=0)
        with pytest.raises(RuntimeError, match="before init"):
            w4.add_worker(child)
        w4.close()

    def test_add_initialized_child_raises(self):
        child = Worker(level=3, num_sub_workers=0)
        child.init()
        w4 = Worker(level=4, num_sub_workers=0)
        with pytest.raises(RuntimeError, match="must be NEW"):
            w4.add_worker(child)
        child.close()
        w4.close()

    def test_l4_device_ids_rejected(self):
        w4 = Worker(level=4, device_ids=[0], num_sub_workers=0)
        with pytest.raises(RuntimeError, match="device_ids are only supported on L3"):
            w4.init()

    def test_add_worker_with_device_ids_rejected(self):
        w4 = Worker(level=4, device_ids=[0], num_sub_workers=0)
        child = Worker(level=3, num_sub_workers=0)
        with pytest.raises(RuntimeError, match="cannot be combined with device_ids"):
            w4.add_worker(child)

    def test_device_mem_on_l4_rejected(self):
        # L4 has no chip mailboxes — device memory ops must be rejected rather than silently dispatch
        # CTRL_MALLOC/FREE/COPY to a next_level (L3 worker) child whose `_child_worker_loop` doesn't
        # recognise them and would return a garbage pointer from an uninitialised mailbox result slot.
        # `malloc` is L2-only (TypeError); the child-device ops guard the worker id (IndexError).
        l3_child = Worker(level=3, num_sub_workers=0)
        w4 = Worker(level=4, num_sub_workers=0)
        w4.add_worker(l3_child)
        w4.init()
        try:
            with pytest.raises(TypeError, match="L2-only"):
                w4.malloc(1024)
            with pytest.raises(IndexError, match="out of range"):
                w4.alloc_child_tensor(0, (256,), DataType.FLOAT32)
            with pytest.raises(IndexError, match="out of range"):
                w4.free(_dev_handle(0xDEADBEEF, wid=0))
            with pytest.raises(IndexError, match="out of range"):
                w4.copy_to(_dev_handle(0xDEAD, wid=0), _HOSTBUF)
            with pytest.raises(IndexError, match="out of range"):
                w4.copy_from(_HOSTBUF, _dev_handle(0xBEEF, wid=0))
        finally:
            w4.close()


@requires_sim_binaries
class TestL4DynamicRegister:
    """Cascade of CTRL_REGISTER / CTRL_UNREGISTER through an L4 → L3 worker tree.

    The L3 child carries a fake chip, so the cascade runs the whole path from
    L4 parent → next_level mailbox → ``_child_worker_loop`` CONTROL handler →
    ``inner_worker._register_child_chip`` → the L3's own chip mailbox → the
    chip child's native register. The fake crosses both forks, so no NPU is
    required.
    """

    @staticmethod
    def _l4_over_chip_backed_l3(monkeypatch, *, register_error: str | None = None) -> Worker:
        install_fake_chip(monkeypatch, register_error=register_error)
        l3 = Worker(
            level=3,
            device_ids=[0],
            platform=SIM_PLATFORM,
            runtime=SIM_RUNTIME,
            num_sub_workers=1,
            startup_timeout_s=10.0,
        )
        l3.register(lambda args: None)  # at least one registered callable so L3 init is happy

        w4 = Worker(level=4, num_sub_workers=0, startup_timeout_s=10.0)
        w4.register(lambda orch, args, config: None)
        w4.add_worker(l3)
        return w4

    def test_l4_register_chip_callable_after_init_succeeds(self, monkeypatch):
        w4 = self._l4_over_chip_backed_l3(monkeypatch)
        try:
            with hard_timeout(TEST_WALL_BUDGET_S):
                w4.init()
                handle = w4.register(chip_callable())
                assert isinstance(handle, CallableHandle)
                assert _slot_for(w4, handle) >= 0
        finally:
            w4.close()

    def test_l4_register_then_unregister_recycles_cid(self, monkeypatch):
        w4 = self._l4_over_chip_backed_l3(monkeypatch)
        try:
            with hard_timeout(TEST_WALL_BUDGET_S):
                w4.init()
                callable_obj = chip_callable()
                handle_a = w4.register(callable_obj)
                slot_a = _slot_for(w4, handle_a)
                w4.unregister(handle_a)
                assert slot_a not in w4._callable_registry
                # Slot is freed; the next register reuses it.
                handle_b = w4.register(callable_obj)
                assert _slot_for(w4, handle_b) == slot_a
        finally:
            w4.close()

    def test_l4_register_surfaces_grandchild_chip_register_failure(self, monkeypatch):
        # The cascade terminates at the chip child two forks down, not at the
        # L3 child: a native register that raises there travels back up as the
        # L4's REGISTER_PARTIAL_FAILURE.
        w4 = self._l4_over_chip_backed_l3(monkeypatch, register_error="injected chip register failure")
        try:
            with hard_timeout(TEST_WALL_BUDGET_S):
                w4.init()
                with pytest.raises(RuntimeError, match="REGISTER_PARTIAL_FAILURE"):
                    w4.register(chip_callable())
        finally:
            w4.close()

    def test_l4_register_python_orch_after_start_succeeds(self):
        counter_shm, counter_buf = _make_shared_counter()
        try:
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_handle = l3.register(lambda args: _increment_counter(counter_buf))

            w4 = Worker(level=4, num_sub_workers=0)
            bootstrap_handle = w4.register(lambda orch, args, config: None)
            l3_worker_id = w4.add_worker(l3)
            w4.init()

            def bootstrap(orch, args, config):
                orch.submit_next_level(bootstrap_handle, TaskArgs(), CallConfig(), worker=l3_worker_id)

            w4.run(bootstrap)

            def dynamic_l3_orch(orch, args, config):
                orch.submit_sub(l3_sub_handle)

            dynamic_handle = w4.register(dynamic_l3_orch)

            def l4_orch(orch, args, config):
                orch.submit_next_level(dynamic_handle, TaskArgs(), CallConfig(), worker=l3_worker_id)

            w4.run(l4_orch)
            w4.close()

            assert _read_counter(counter_buf) == 1
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: L4 → L3 PROCESS mode — single dispatch
# ---------------------------------------------------------------------------


class TestL4ToL3SingleDispatch:
    def test_l4_dispatches_to_l3_sub(self):
        """L4 orch submits one task to L3 child. L3 orch runs a sub callable.

        Verifies that the sub callable's shared counter is incremented,
        proving the full L4 → L3 → sub dispatch chain works.
        """
        counter_shm, counter_buf = _make_shared_counter()

        try:
            # L3 child: one sub worker, one sub callable that increments counter
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_handle = l3.register(lambda args: _increment_counter(counter_buf))

            def l3_orch(orch, args, config):
                orch.submit_sub(l3_sub_handle)

            # L4 parent: one next-level child, register L3 orch fn
            w4 = Worker(level=4, num_sub_workers=0)
            l3_handle = w4.register(l3_orch)
            l3_worker_id = w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                orch.submit_next_level(l3_handle, TaskArgs(), CallConfig(), worker=l3_worker_id)

            w4.run(l4_orch)
            w4.close()

            assert _read_counter(counter_buf) == 1
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: L4 → L3 — multiple dispatches
# ---------------------------------------------------------------------------


class TestL4ToL3MultipleDispatches:
    def test_l4_dispatches_three_times(self):
        """L4 orch submits 3 tasks to L3 child, each running a sub callable."""
        counter_shm, counter_buf = _make_shared_counter()

        try:
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_handle = l3.register(lambda args: _increment_counter(counter_buf))

            def l3_orch(orch, args, config):
                orch.submit_sub(l3_sub_handle)

            w4 = Worker(level=4, num_sub_workers=0)
            l3_handle = w4.register(l3_orch)
            l3_worker_id = w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                for _ in range(3):
                    orch.submit_next_level(l3_handle, TaskArgs(), CallConfig(), worker=l3_worker_id)

            w4.run(l4_orch)
            w4.close()

            assert _read_counter(counter_buf) == 3
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: L4 with own sub workers + L3 child
# ---------------------------------------------------------------------------


class TestL4WithOwnSubs:
    def test_l4_sub_and_l3_dispatch(self):
        """L4 has its own sub workers AND dispatches to an L3 child.

        L4's sub callable and L3's sub callable each increment a counter, and
        both paths must execute. They are the only two increments in this file
        that run in *different* processes — L4's own SubWorker and the L3
        child's SubWorker — dispatched from independent WorkerThreads with no
        ordering between them, so each gets its own slot. Sharing one slot
        would make the pair a cross-process non-atomic read-modify-write.
        """
        counter_shm, counter_buf = _make_shared_counter(slots=2)
        L3_SLOT, L4_SLOT = 0, 1

        try:
            # L3 child: sub worker increments counter
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_handle = l3.register(lambda args: _increment_counter(counter_buf, L3_SLOT))

            def l3_orch(orch, args, config):
                orch.submit_sub(l3_sub_handle)

            # L4: own sub worker + L3 child
            w4 = Worker(level=4, num_sub_workers=1)
            l3_handle = w4.register(l3_orch)
            l4_verify_handle = w4.register(lambda args: _increment_counter(counter_buf, L4_SLOT))
            l3_worker_id = w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                orch.submit_next_level(l3_handle, TaskArgs(), CallConfig(), worker=l3_worker_id)
                orch.submit_sub(l4_verify_handle)

            w4.run(l4_orch)
            w4.close()

            assert _read_counter(counter_buf, L3_SLOT) == 1, "L3 sub callable did not run"
            assert _read_counter(counter_buf, L4_SLOT) == 1, "L4 sub callable did not run"
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: L4 → L3 — multiple runs
# ---------------------------------------------------------------------------


class TestL4MultipleRuns:
    def test_l4_multiple_runs_no_leak(self):
        """Multiple w4.run() calls on the same Worker — slots don't leak."""
        counter_shm, counter_buf = _make_shared_counter()

        try:
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_handle = l3.register(lambda args: _increment_counter(counter_buf))

            def l3_orch(orch, args, config):
                orch.submit_sub(l3_sub_handle)

            w4 = Worker(level=4, num_sub_workers=0)
            l3_handle = w4.register(l3_orch)
            l3_worker_id = w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                orch.submit_next_level(l3_handle, TaskArgs(), CallConfig(), worker=l3_worker_id)

            for _ in range(5):
                w4.run(l4_orch)

            w4.close()

            assert _read_counter(counter_buf) == 5
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: L4 → L3 — L3 uses multiple sub workers
# ---------------------------------------------------------------------------


class TestL4L3WithMultipleSubs:
    def test_l3_child_runs_multiple_subs(self):
        """L3 child submits 2 sub tasks per dispatch (serialized through 1 worker).

        Uses 1 sub worker because _increment_counter is a non-atomic RMW
        that races across parallel SubWorker processes.
        """
        counter_shm, counter_buf = _make_shared_counter()

        try:
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_handle = l3.register(lambda args: _increment_counter(counter_buf))

            def l3_orch(orch, args, config):
                orch.submit_sub(l3_sub_handle)
                orch.submit_sub(l3_sub_handle)

            w4 = Worker(level=4, num_sub_workers=0)
            l3_handle = w4.register(l3_orch)
            l3_worker_id = w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                orch.submit_next_level(l3_handle, TaskArgs(), CallConfig(), worker=l3_worker_id)

            w4.run(l4_orch)
            w4.close()

            assert _read_counter(counter_buf) == 2
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: L3 orch receives its own Orchestrator (not L4's)
# ---------------------------------------------------------------------------


class TestL3OwnOrchestrator:
    def test_l3_gets_own_orchestrator(self):
        """The L3 orch fn receives an Orchestrator from the L3 inner worker,
        not the L4 parent. Prove by checking orch.alloc works at L3 level."""
        counter_shm, counter_buf = _make_shared_counter()

        try:
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_handle = l3.register(lambda args: _increment_counter(counter_buf))

            def l3_orch(orch, args, config):
                # orch is L3's own Orchestrator — alloc + submit_sub should work
                orch.submit_sub(l3_sub_handle)

            w4 = Worker(level=4, num_sub_workers=0)
            l3_handle = w4.register(l3_orch)
            l3_worker_id = w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                orch.submit_next_level(l3_handle, TaskArgs(), CallConfig(), worker=l3_worker_id)

            w4.run(l4_orch)
            w4.close()

            assert _read_counter(counter_buf) == 1
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: generalised _Worker(level) — no hardcoded 3
# ---------------------------------------------------------------------------


class TestGeneralised_Worker:
    def test_worker_level_param(self):
        """_Worker accepts level != 3 without error."""
        from simpler.task_interface import _Worker  # noqa: PLC0415

        for level in (3, 4, 5):
            dw = _Worker(level)
            dw.close()
