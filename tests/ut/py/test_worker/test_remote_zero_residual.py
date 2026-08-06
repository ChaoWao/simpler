# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full-hierarchy Remote L3 failure cleanup acceptance.

Each case starts a real L4 with a READY local SUB child, then fails remote
activation at a different boundary.  The assertions retain the exact PIDs and
SHM names observed for that epoch, prove they disappear, and reuse the daemon
for a subsequent healthy session.
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import simpler.worker as worker_mod
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker

from tests.ut.py.test_callable_identity import _free_tcp_port, _RemoteL3Daemon  # noqa: PLC2701
from tests.ut.py.test_worker._harness import hard_timeout

_SHM_ROOT = Path("/dev/shm")
_TEST_WALL_BUDGET_S = 30.0
_CLEANUP_OBSERVE_S = 8.0

pytestmark = pytest.mark.skipif(
    not _SHM_ROOT.is_dir() or not Path("/proc").is_dir(),
    reason="exact PID/process-group/POSIX-SHM probes require Linux /proc and /dev/shm",
)


def _direct_child_pids(pid: int) -> set[int]:
    children = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        return {int(value) for value in children.read_text().split()}
    except (FileNotFoundError, ProcessLookupError):
        return set()


def _process_shm_names(pid: int) -> set[str]:
    try:
        lines = Path(f"/proc/{pid}/maps").read_text().splitlines()
    except (FileNotFoundError, ProcessLookupError):
        return set()
    names = set()
    for line in lines:
        marker = "/dev/shm/"
        marker_at = line.find(marker)
        if marker_at < 0:
            continue
        name = line[marker_at + len(marker) :].split(maxsplit=1)[0]
        if name.startswith(("sp-", "psm_")):
            names.add(name)
    return names


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_until(predicate, message: str, timeout_s: float = _CLEANUP_OBSERVE_S) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(message)


@dataclass
class _ObservedEpoch:
    local_pids: set[int] = field(default_factory=set)
    remote_pgids: set[int] = field(default_factory=set)
    shm_names: set[str] = field(default_factory=set)

    def capture_local_tree(self, worker: Worker) -> None:
        self.local_pids.update(worker._sub_pids)
        self.local_pids.update(worker._chip_pids)
        self.local_pids.update(worker._next_level_pids)
        for shm in (*worker._sub_shms, *worker._chip_shms, *worker._next_level_shms):
            self.shm_names.add(shm.name)

    def capture_remote_session(self, session: worker_mod._RemoteSession) -> set[str]:
        remote_pids = {session.pid, *_direct_child_pids(session.pid)}
        shm_names = set().union(*(_process_shm_names(pid) for pid in remote_pids))
        if session.pid > 0:
            self.remote_pgids.add(session.pid)
        self.shm_names.update(shm_names)
        return shm_names

    def assert_reclaimed(self, worker: Worker) -> None:
        _wait_until(
            lambda: all(not _pid_exists(pid) for pid in self.local_pids),
            f"local child PID(s) survived rollback: {sorted(pid for pid in self.local_pids if _pid_exists(pid))}",
        )
        _wait_until(
            lambda: all(not _process_group_exists(pgid) for pgid in self.remote_pgids),
            "remote runner process group(s) survived rollback: "
            f"{sorted(pgid for pgid in self.remote_pgids if _process_group_exists(pgid))}",
        )
        _wait_until(
            lambda: all(not (_SHM_ROOT / name).exists() for name in self.shm_names),
            "SHM segment(s) survived rollback: "
            f"{sorted(name for name in self.shm_names if (_SHM_ROOT / name).exists())}",
        )
        assert worker._sub_pids == []
        assert worker._chip_pids == []
        assert worker._next_level_pids == []
        assert worker._sub_shms == []
        assert worker._chip_shms == []
        assert worker._next_level_shms == []
        assert worker._remote_sessions == []
        assert worker._worker is None
        assert not worker._has_live_resources()


@contextlib.contextmanager
def _unlistening_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    try:
        yield int(sock.getsockname()[1])
    finally:
        sock.close()


def _remote_spec(port: int, *, num_sub_workers: int = 0) -> RemoteWorkerSpec:
    return RemoteWorkerSpec(
        endpoint=f"127.0.0.1:{port}",
        platform="a2a3sim",
        transport="host_tcp",
        num_sub_workers=num_sub_workers,
    )


def _start_delayed_daemon(monkeypatch, port: int, import_delay_s: float) -> _RemoteL3Daemon:
    monkeypatch.setenv("SIMPLER_TEST_REMOTE_IMPORT_DELAY_S", str(import_delay_s))
    try:
        # Construction starts the daemon process, which inherits the delay
        # before this test-only environment value is removed from the parent.
        return _RemoteL3Daemon(port)
    finally:
        monkeypatch.delenv("SIMPLER_TEST_REMOTE_IMPORT_DELAY_S")


def _assert_daemon_accepts_next_session(port: int) -> None:
    worker = Worker(level=4, num_sub_workers=0, startup_timeout_s=10.0, remote_session_timeout_s=5.0)
    try:
        worker.add_remote_worker(_remote_spec(port))
        worker.init()
    finally:
        worker.close()


def _start_runner_observer(
    daemon: _RemoteL3Daemon, observed: _ObservedEpoch
) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()

    def observe() -> None:
        while not stop.is_set():
            runner_pids = _direct_child_pids(daemon._proc.pid)
            observed.remote_pgids.update(runner_pids)
            for pid in runner_pids:
                observed.shm_names.update(_process_shm_names(pid))
            stop.wait(0.005)

    thread = threading.Thread(target=observe, name="remote-runner-observer", daemon=True)
    thread.start()
    return stop, thread


def test_remote_open_timeout_reclaims_full_local_tree_and_runner(monkeypatch):
    port = _free_tcp_port()
    daemon = _start_delayed_daemon(monkeypatch, port, import_delay_s=3.0)
    observed = _ObservedEpoch()
    real_open = Worker._open_remote_session

    def observed_open(self: Worker, **kwargs: Any):
        if kwargs["spec"].num_sub_workers == 0:
            return real_open(self, **kwargs)
        observed.capture_local_tree(self)
        stop, thread = _start_runner_observer(daemon, observed)
        try:
            return real_open(self, **kwargs)
        finally:
            stop.set()
            thread.join(timeout=1.0)
            assert not thread.is_alive(), "remote runner observer did not stop"

    monkeypatch.setattr(Worker, "_open_remote_session", observed_open)
    worker = Worker(level=4, num_sub_workers=1, startup_timeout_s=1.0, remote_session_timeout_s=5.0)
    try:
        daemon.await_ready()
        worker_id = worker.add_remote_worker(_remote_spec(port, num_sub_workers=1))
        worker.register(
            RemoteCallable("tests.ut.py.test_worker._remote_zero_residual_target:noop"),
            workers=[worker_id],
        )
        with hard_timeout(_TEST_WALL_BUDGET_S), pytest.raises(TimeoutError) as failure:
            worker.init()
        assert f"worker {worker_id}" in str(failure.value)
        assert observed.local_pids
        assert observed.remote_pgids
        assert observed.shm_names
        observed.assert_reclaimed(worker)
        _assert_daemon_accepts_next_session(port)
    finally:
        worker.close()
        daemon.stop()


def test_later_remote_failure_reclaims_ready_session_and_full_local_tree(monkeypatch):
    port = _free_tcp_port()
    daemon = _RemoteL3Daemon(port)
    observed = _ObservedEpoch()
    real_open = Worker._open_remote_session

    def observed_open(self: Worker, **kwargs: Any):
        observed.capture_local_tree(self)
        session = real_open(self, **kwargs)
        if kwargs["spec"].num_sub_workers > 0:
            assert observed.capture_remote_session(session)
        return session

    monkeypatch.setattr(Worker, "_open_remote_session", observed_open)
    with _unlistening_port() as failing_port:
        worker = Worker(level=4, num_sub_workers=1, startup_timeout_s=10.0, remote_session_timeout_s=5.0)
        try:
            daemon.await_ready()
            first_worker_id = worker.add_remote_worker(_remote_spec(port, num_sub_workers=1))
            failing_worker_id = worker.add_remote_worker(_remote_spec(failing_port))
            with hard_timeout(_TEST_WALL_BUDGET_S), pytest.raises(RuntimeError) as failure:
                worker.init()
            assert first_worker_id != failing_worker_id
            assert f"worker {failing_worker_id}" in str(failure.value)
            assert observed.local_pids
            assert observed.remote_pgids
            observed.assert_reclaimed(worker)
            _assert_daemon_accepts_next_session(port)
        finally:
            worker.close()
            daemon.stop()


def test_attach_failure_reclaims_open_session_and_full_local_tree(monkeypatch):
    port = _free_tcp_port()
    daemon = _RemoteL3Daemon(port)
    observed = _ObservedEpoch()
    real_worker_ctor = worker_mod._Worker
    real_open = Worker._open_remote_session

    class AttachFailingWorker:
        def __init__(self, *args: Any):
            self._inner = real_worker_ctor(*args)

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

        def add_remote_l3_socket(self, worker_id: int, *_args: Any) -> None:
            raise RuntimeError(f"injected remote attach failure for worker {worker_id}")

    def observed_open(self: Worker, **kwargs: Any):
        observed.capture_local_tree(self)
        session = real_open(self, **kwargs)
        if kwargs["spec"].num_sub_workers > 0:
            assert observed.capture_remote_session(session)
        return session

    worker: Worker | None = None
    try:
        daemon.await_ready()
        with monkeypatch.context() as patch:
            patch.setattr(worker_mod, "_Worker", AttachFailingWorker)
            patch.setattr(Worker, "_open_remote_session", observed_open)
            worker = Worker(level=4, num_sub_workers=1, startup_timeout_s=10.0, remote_session_timeout_s=5.0)
            worker_id = worker.add_remote_worker(_remote_spec(port, num_sub_workers=1))
            with (
                hard_timeout(_TEST_WALL_BUDGET_S),
                pytest.raises(RuntimeError, match=rf"endpoint attach failed for worker {worker_id}"),
            ):
                worker.init()
            assert observed.local_pids
            assert observed.remote_pgids
            observed.assert_reclaimed(worker)
            worker.close()
        _assert_daemon_accepts_next_session(port)
    finally:
        if worker is not None:
            worker.close()
        daemon.stop()
