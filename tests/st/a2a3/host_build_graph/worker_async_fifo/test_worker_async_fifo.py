#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Onboard validation for bounded whole-run FIFO admission.

The first run completes real NPU work but remains active behind a SubTask
fence.  The second run builds its graph into the other pipeline slot and must
not dispatch until the first run becomes terminal.  A third submission must
block before its graph callback while both slots are admitted.
"""

import atexit
import tempfile
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.scene_test import _build_l3_task_args

_VECTOR_KERNELS = "../vector_example/kernels"
_SIZE = 128 * 128


class _FileSignal:
    """Pickle-safe parent/child signal for the standalone scene runner."""

    def __init__(self, label: str):
        token = uuid.uuid4().hex
        self._path = Path(tempfile.gettempdir()) / f"simpler-worker-async-fifo-{label}-{token}"

    def clear(self) -> None:
        with suppress(FileNotFoundError):
            self._path.unlink()

    def set(self) -> None:
        self._path.touch(exist_ok=True)

    def wait(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._path.exists():
                return True
            time.sleep(0.001)
        return self._path.exists()


_SUB_ENTERED = _FileSignal("entered")
_SUB_RELEASE = _FileSignal("release")


def _clear_signals() -> None:
    _SUB_ENTERED.clear()
    _SUB_RELEASE.clear()


atexit.register(_clear_signals)


def _wait_for_release(_args):
    _SUB_ENTERED.set()
    if not _SUB_RELEASE.wait(30.0):
        raise RuntimeError("whole-run FIFO test timed out waiting for the release fence")


@scene_test(level=3, runtime="host_build_graph")
class TestWorkerAsyncWholeRunFifo(SceneTestCase):
    """A prepared run may build ahead but cannot dispatch ahead."""

    CALLABLE = {
        "callables": [
            {
                "name": "vector",
                "orchestration": {
                    "source": f"{_VECTOR_KERNELS}/orchestration/example_orch.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.IN, D.IN, D.OUT],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": f"{_VECTOR_KERNELS}/aiv/kernel_add.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT],
                    },
                    {
                        "func_id": 1,
                        "source": f"{_VECTOR_KERNELS}/aiv/kernel_add_scalar.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT],
                    },
                    {
                        "func_id": 2,
                        "source": f"{_VECTOR_KERNELS}/aiv/kernel_mul.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT],
                    },
                ],
            },
            {"name": "wait_for_release", "callable": _wait_for_release},
        ],
    }

    CASES = [
        {
            "name": "whole_run_fifo",
            "platforms": ["a2a3"],
            "config": {"device_count": 1, "num_sub_workers": 1, "aicpu_thread_num": 4},
            "params": {},
        },
    ]

    @staticmethod
    def _tensor_from_host_buffer(worker, value):
        buffer = worker.create_host_buffer(_SIZE * torch.float32.itemsize)
        tensor = torch.frombuffer(buffer.buffer, dtype=torch.float32, count=_SIZE)
        tensor.fill_(value)
        return buffer, tensor

    def _run_and_validate_l3(  # noqa: PLR0913 -- mirror the scene-test runner hook
        self,
        worker,
        compiled_callables,
        sub_handles,
        case,
        rounds=1,
        skip_golden=False,
        enable_l2_swimlane=0,
        enable_dump_args=False,
        enable_pmu=0,
        enable_dep_gen=False,
        enable_scope_stats=False,
        output_prefix="",
    ):
        """Run the custom concurrency checks from the standalone scene runner.

        Unlike a regular scene case, this test has no single orchestration
        callback or generated argument set.  Pytest collects the three methods
        below directly; the standalone runner reaches them through this hook.
        """
        del (
            rounds,
            skip_golden,
            enable_l2_swimlane,
            enable_dump_args,
            enable_pmu,
            enable_dep_gen,
            enable_scope_stats,
            output_prefix,
        )
        # The pytest fixture publishes these maps on the class.  Standalone
        # execution passes the same registered handles into this hook instead.
        type(self)._st_chip_handles = compiled_callables
        type(self)._st_sub_handles = sub_handles
        platform = str(worker._config["platform"])  # noqa: SLF001 -- scene-test white-box validation
        assert platform in case["platforms"]
        self.test_prepared_run_device_control_waits_for_the_active_run(platform, worker)
        self.test_a_run_whose_cleanup_touches_the_device_degrades_to_depth_one(platform, worker)
        self.test_run(platform, worker)

    def test_prepared_run_device_control_waits_for_the_active_run(self, st_platform, st_worker):
        """A prepared run's malloc/free/copy must not overtake the active run.

        These reach a child directly rather than through a TaskSlot, so the
        ready-queue FIFO does not order them and the mailbox mutex only
        serialises one command at a time. Without an admission gate a prepared
        successor could free or overwrite child memory the active run is still
        reading, which is the ordering the whole-run FIFO exists to provide.
        """
        if st_platform != "a2a3":
            pytest.skip("whole-run FIFO leases require an a2a3 onboard worker")

        _SUB_ENTERED.clear()
        _SUB_RELEASE.clear()
        entered_callback = threading.Event()
        control_returned = threading.Event()
        result = {}
        buffers = []
        tensors = []
        submitter = None
        first = None
        try:
            for value in (2.0, 3.0, 0.0):
                buffer, tensor = self._tensor_from_host_buffer(st_worker, value)
                buffers.append(buffer)
                tensors.append(tensor)
            first_a, first_b, first_out = tensors

            vector_handle = type(self)._st_chip_handles["vector"]
            vector_signature = type(self)._st_chip_handles["vector_sig"]
            sub_handle = type(self)._st_sub_handles["wait_for_release"]

            def first_graph(orch, _args, _cfg):
                builder = TaskArgsBuilder(Tensor("a", first_a), Tensor("b", first_b), Tensor("f", first_out))
                chip_args, _ = _build_l3_task_args(builder, vector_signature)
                orch.submit_next_level(vector_handle, chip_args, self._build_config(self.CASES[0]["config"]), worker=0)
                orch.submit_sub(sub_handle)

            first = st_worker.submit(first_graph)
            assert _SUB_ENTERED.wait(30.0), "the first run never reached its SUB fence"

            # The successor is admitted into the second slot, so its callback
            # runs — but its first device-control call must block until the
            # first run releases the FIFO head.
            def second_graph(orch, _args, _cfg):
                entered_callback.set()
                ptr = orch.malloc(0, 4096)
                control_returned.set()
                orch.free(0, ptr)

            submitter = threading.Thread(
                target=lambda: result.setdefault("handle", st_worker.submit(second_graph)), daemon=True
            )
            submitter.start()

            assert entered_callback.wait(10.0), "the prepared run's graph callback never entered"
            assert not control_returned.wait(1.0), (
                "a prepared run's device control returned while the active run still held the FIFO head"
            )

            _SUB_RELEASE.set()
            first.wait(30.0)
            assert control_returned.wait(30.0), "device control stayed blocked after the active run became terminal"
            submitter.join(30.0)
            assert not submitter.is_alive()
            result["handle"].wait(30.0)
        finally:
            _SUB_RELEASE.set()
            if submitter is not None:
                submitter.join(30.0)
            handles = [first, result.get("handle")]
            for handle in handles:
                if handle is not None:
                    with suppress(Exception):
                        handle.wait(30.0)
            tensors.clear()
            tensor = None
            first_a = first_b = first_out = None
            if all(handle is None or handle.done for handle in handles):
                for buffer in buffers:
                    st_worker.free_host_buffer(buffer)

    def test_a_run_whose_cleanup_touches_the_device_degrades_to_depth_one(self, st_platform, st_worker):
        """A CommDomain release is mailbox control the whole-run FIFO cannot order.

        The FIFO orders tasks. A domain's teardown happens after the native
        fence and reaches a child through the mailbox, so N+1 allocating a
        domain while N is still releasing one can leave two collectives each
        holding a different chip's mailbox. A run that takes such a resource
        therefore degrades this worker to depth one: the successor's graph
        callback does not start at all until that teardown has run.

        The sibling test above pins the other half — a run that only dispatches
        tasks keeps the full depth, and its successor's callback does run while
        it is still fenced.
        """
        if st_platform != "a2a3":
            pytest.skip("whole-run FIFO leases require an a2a3 onboard worker")

        _SUB_ENTERED.clear()
        _SUB_RELEASE.clear()
        entered_callback = threading.Event()
        result = {}
        submitter = None
        first = None
        try:
            sub_handle = type(self)._st_sub_handles["wait_for_release"]

            def first_graph(orch, _args, _cfg):
                # Control before any submit: a task travels the ready queue and
                # this travels the mailbox, so the two are not ordered.
                with orch.allocate_domain(name="degrade", workers=[0], window_size=4096, buffers=[]):
                    pass
                orch.submit_sub(sub_handle)

            first = st_worker.submit(first_graph)
            assert _SUB_ENTERED.wait(30.0), "the first run never reached its SUB fence"

            def second_graph(_orch, _args, _cfg):
                entered_callback.set()

            submitter = threading.Thread(
                target=lambda: result.setdefault("handle", st_worker.submit(second_graph)), daemon=True
            )
            submitter.start()

            assert not entered_callback.wait(2.0), (
                "a successor's graph callback started while a cleanup-bearing run was still outstanding"
            )

            _SUB_RELEASE.set()
            first.wait(30.0)
            assert entered_callback.wait(30.0), "the successor stayed blocked after its predecessor's cleanup ran"
            submitter.join(30.0)
            assert not submitter.is_alive()
            result["handle"].wait(30.0)
        finally:
            _SUB_RELEASE.set()
            if submitter is not None:
                submitter.join(30.0)
            for handle in (first, result.get("handle")):
                if handle is not None:
                    with suppress(Exception):
                        handle.wait(30.0)

    def test_run(self, st_platform, st_worker):
        if st_platform != "a2a3":
            pytest.skip("whole-run FIFO leases require an a2a3 onboard worker")

        _SUB_ENTERED.clear()
        _SUB_RELEASE.clear()
        third_callback = threading.Event()
        third_result = {}
        buffers = []
        tensors = []
        submitter = None
        first = None
        second = None
        try:
            for value in (2.0, 3.0, 0.0, 5.0, 7.0, 0.0):
                buffer, tensor = self._tensor_from_host_buffer(st_worker, value)
                buffers.append(buffer)
                tensors.append(tensor)
            first_a, first_b, first_out, second_a, second_b, second_out = tensors

            vector_handle = type(self)._st_chip_handles["vector"]
            vector_signature = type(self)._st_chip_handles["vector_sig"]
            sub_handle = type(self)._st_sub_handles["wait_for_release"]

            def submit_vector(orch, a, b, out, *, hold_open=False):
                builder = TaskArgsBuilder(Tensor("a", a), Tensor("b", b), Tensor("f", out))
                chip_args, _ = _build_l3_task_args(builder, vector_signature)
                orch.submit_next_level(vector_handle, chip_args, self._build_config(self.CASES[0]["config"]), worker=0)
                if hold_open:
                    orch.submit_sub(sub_handle)

            first = st_worker.submit(
                lambda orch, _args, _cfg: submit_vector(orch, first_a, first_b, first_out, hold_open=True)
            )
            assert _SUB_ENTERED.wait(10.0), "the first run's SubTask did not start"

            first_expected = (first_a + first_b + 1) * (first_a + first_b + 2)
            deadline = time.monotonic() + 10.0
            while not torch.allclose(first_out, first_expected) and time.monotonic() < deadline:
                time.sleep(0.001)
            assert torch.allclose(first_out, first_expected), "the first run's NPU task did not complete"

            second_graph_done = threading.Event()

            def second_graph(orch, _args, _cfg):
                submit_vector(orch, second_a, second_b, second_out)
                second_graph_done.set()

            second = st_worker.submit(second_graph)
            assert second_graph_done.is_set(), "the second run did not build ahead"
            assert torch.count_nonzero(second_out).item() == 0, (
                "the prepared run dispatched before the active run ended"
            )

            def third_graph(_orch, _args, _cfg):
                third_callback.set()

            submitter = threading.Thread(
                target=lambda: third_result.setdefault("handle", st_worker.submit(third_graph)), daemon=True
            )
            submitter.start()
            assert not third_callback.wait(0.1), "the third graph callback entered before admission capacity was free"

            # The check above only proves the prepared run had not dispatched at
            # one instant. Re-check after the hold: the active run is still
            # blocked in its SUB task for the whole window, so a prepared run
            # that leaks past the FIFO gate at any point during it is caught
            # here rather than passing on timing.
            assert torch.count_nonzero(second_out).item() == 0, (
                "the prepared run dispatched while the active run was still holding the FIFO head"
            )

            _SUB_RELEASE.set()
            first.wait(10.0)
            assert third_callback.wait(10.0), "the third submission did not enter after the first run freed its slot"
            second.wait(10.0)
            second_expected = (second_a + second_b + 1) * (second_a + second_b + 2)
            assert torch.allclose(second_out, second_expected), "the prepared run did not execute correctly on the NPU"

            submitter.join(10.0)
            assert not submitter.is_alive()
            third_result["handle"].wait(10.0)
        finally:
            _SUB_RELEASE.set()
            if submitter is not None:
                submitter.join(10.0)
            handles = [first, second, third_result.get("handle")]
            for handle in handles:
                if handle is not None:
                    with suppress(Exception):
                        handle.wait(10.0)
            tensors.clear()
            tensor = None
            first_a = first_b = first_out = second_a = second_b = second_out = None
            if all(handle is None or handle.done for handle in handles):
                for buffer in buffers:
                    st_worker.free_host_buffer(buffer)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
