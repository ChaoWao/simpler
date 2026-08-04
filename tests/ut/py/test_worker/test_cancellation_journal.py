# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""P0.2-c cooperative init cancellation and retryable resource cleanup journal."""

import threading
import time

import pytest
import simpler.worker as worker_mod
from simpler.worker import CleanupJournal, Worker, _Lifecycle, _shm_name

from ._harness import TEST_WALL_BUDGET_S, hard_timeout

_TEST_WALL_BUDGET_S = TEST_WALL_BUDGET_S
_hard_timeout = hard_timeout


def _run_catch(fn):
    try:
        fn()
        return None
    except BaseException as e:
        return e


def _init_hangs(*_a, **_k):
    time.sleep(3600)


def _init_raises(*_a, **_k):
    raise RuntimeError("injected inner init failure")


def _l3_child(sub_fn=None, num_sub_workers=1):
    l3 = Worker(level=3, num_sub_workers=num_sub_workers)
    l3.register(sub_fn if sub_fn is not None else (lambda args: None))
    return l3


def _trivial_orch(orch, args, config):
    return None


class TestCleanupJournal:
    def test_empty_drive_returns_none(self):
        j = CleanupJournal()
        assert j.empty
        assert j.drive() is None

    def test_success_removes_entry(self):
        j = CleanupJournal()
        called = []
        j.add("child", "test", lambda: called.append(1))
        assert not j.empty
        assert j.drive() is None
        assert j.empty
        assert called == [1]

    def test_failure_keeps_entry(self):
        j = CleanupJournal()
        j.add("child", "test", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        err = j.drive()
        assert err is not None
        assert "boom" in str(err)
        assert not j.empty

    def test_failed_entry_retryable(self):
        j = CleanupJournal()
        attempts = []

        def fail_then_ok():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("first attempt fails")

        j.add("child", "test", fail_then_ok)
        assert j.drive() is not None
        assert not j.empty
        assert j.drive() is None
        assert j.empty
        assert attempts == [1, 1]

    def test_all_entries_attempted_on_failure(self):
        j = CleanupJournal()
        called = []
        j.add("child", "a", lambda: called.append("ok"))
        j.add("child", "b", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        j.add("child", "c", lambda: called.append("also_ok"))
        err = j.drive()
        assert err is not None
        assert called == ["ok", "also_ok"]
        assert not j.empty
        assert len(j._entries) == 1

    def test_errors_accumulated(self):
        j = CleanupJournal()
        j.add("child", "a", lambda: (_ for _ in ()).throw(ValueError("a")))
        j.add("child", "b", lambda: (_ for _ in ()).throw(TypeError("b")))
        j.drive()
        assert len(j.errors) == 2

    def test_extend(self):
        j = CleanupJournal()
        called = []
        j.extend([("child", "a", lambda: called.append("a")), ("child", "b", lambda: called.append("b"))])
        j.drive()
        assert j.empty
        assert called == ["a", "b"]


class TestCloseDuringInitializing:
    def test_close_cancels_init_reaches_closed(self, monkeypatch):
        entered = threading.Event()
        release = threading.Event()
        orig = Worker._start_hierarchical

        def paused_start(self):
            entered.set()
            assert release.wait(10.0)
            return orig(self)

        monkeypatch.setattr(Worker, "_start_hierarchical", paused_start)
        w = Worker(level=3, num_sub_workers=1, startup_timeout_s=30.0)
        w.register(lambda args: None)

        def owner_body():
            _run_catch(w.init)

        it = threading.Thread(target=owner_body)
        it.start()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                assert entered.wait(3.0)
                close_result: list = []
                ct = threading.Thread(target=lambda: close_result.append(_run_catch(w.close)))
                ct.start()
                while not w._cancel_token:
                    time.sleep(0.001)
                release.set()
                ct.join(10.0)
                it.join(10.0)
                assert close_result == [None]
                assert w._lifecycle is worker_mod._Lifecycle.CLOSED
        finally:
            release.set()
            it.join(10.0)

    def test_uncooperative_init_bounds_the_close_wait(self, monkeypatch):
        """An init blocked past every cooperative point makes close() raise on
        its unwind deadline instead of blocking forever."""
        entered = threading.Event()
        release = threading.Event()
        orig = Worker._start_hierarchical

        def paused_start(self):
            entered.set()
            assert release.wait(30.0)
            return orig(self)

        monkeypatch.setattr(Worker, "_start_hierarchical", paused_start)
        monkeypatch.setattr(worker_mod, "_CLOSE_CANCEL_UNWIND_TIMEOUT_S", 0.5)
        w = Worker(level=3, num_sub_workers=1, startup_timeout_s=30.0)
        w.register(lambda args: None)

        def owner_body():
            _run_catch(w.init)

        it = threading.Thread(target=owner_body)
        it.start()
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                assert entered.wait(3.0)
                err = _run_catch(w.close)
                assert isinstance(err, RuntimeError)
                assert "did not unwind" in str(err)
                assert w._lifecycle is worker_mod._Lifecycle.INITIALIZING
        finally:
            release.set()
            it.join(10.0)

    def test_same_thread_close_during_init_rejected(self, monkeypatch):
        entered = threading.Event()
        release = threading.Event()
        orig = Worker._start_hierarchical

        def paused_start(self):
            entered.set()
            err = _run_catch(self.close)
            assert isinstance(err, RuntimeError)
            assert "init-owner" in str(err)
            assert release.wait(10.0)
            return orig(self)

        monkeypatch.setattr(Worker, "_start_hierarchical", paused_start)
        w = Worker(level=3, num_sub_workers=1, startup_timeout_s=30.0)
        w.register(lambda args: None)
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                release.set()
                w.init()
                assert w._lifecycle is worker_mod._Lifecycle.READY
        finally:
            w.close()


class TestClosedAbsorption:
    def test_closed_absorbing(self):
        w = Worker(level=3, num_sub_workers=1)
        w.register(lambda args: None)
        with w._hierarchical_start_cv:
            w._lifecycle = _Lifecycle.INITIALIZING
        with w._hierarchical_start_cv:
            w._lifecycle = _Lifecycle.CLOSED
        with w._hierarchical_start_cv:
            if w._lifecycle is _Lifecycle.INITIALIZING:
                w._lifecycle = _Lifecycle.FAILED
        assert w._lifecycle is worker_mod._Lifecycle.CLOSED
        w.close()


class TestJournalInAbortHierarchical:
    def test_journal_persists_after_close_failure(self):
        """A journal entry added before close() persists through a failed
        close() and is retried by a second close()."""
        w = Worker(level=3, num_sub_workers=1)
        w.register(lambda args: None)
        attempts = []

        def fail_then_ok():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("transient journal failure")

        w._cleanup_journal.add("child", "test", fail_then_ok)
        # First close: journal drive fails, entry stays.
        err1 = _run_catch(w.close)
        assert err1 is not None
        assert "transient journal failure" in str(err1)
        assert not w._cleanup_journal.empty
        assert w._lifecycle is worker_mod._Lifecycle.CLOSED
        # The attempt is incomplete — journal still has entries.
        assert w._close_completion is not None
        assert w._close_completion.incomplete
        assert w._close_completion.done
        # Second close: retries the journal, succeeds.
        err2 = _run_catch(w.close)
        assert err2 is None
        assert w._cleanup_journal.empty
        assert attempts == [1, 1]

    def test_fully_reaped_abort_leaves_empty_journal(self, monkeypatch):
        l3 = _l3_child()
        l3.init = _init_raises
        w4 = Worker(level=4, num_sub_workers=0, startup_timeout_s=10.0)
        w4.register(_trivial_orch)
        w4.add_worker(l3)
        try:
            with _hard_timeout(_TEST_WALL_BUDGET_S):
                with pytest.raises(RuntimeError, match="injected inner init failure"):
                    w4.init()
            assert w4._cleanup_journal.empty
            assert w4._next_level_pids == []
        finally:
            w4.close()


class TestJournalRetryOnClose:
    def test_journal_driven_before_teardown(self):
        """When the journal has entries, _teardown_ready_tree drives them
        and they are removed on success."""
        w = Worker(level=3, num_sub_workers=1)
        w.register(lambda args: None)
        driven = []
        w._cleanup_journal.add("child", "test", lambda: driven.append("journal"))
        # close() invokes _teardown_ready_tree which drives the journal.
        w.close()
        assert driven == ["journal"], f"journal entry was not driven: {driven}"
        assert w._cleanup_journal.empty

    def test_journal_retry_on_close(self):
        """A journal entry that fails on first close stays in the journal
        and is retried by a second close()."""
        w = Worker(level=3, num_sub_workers=1)
        w.register(lambda args: None)
        attempts = []

        def fail_then_ok():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("transient failure")

        w._cleanup_journal.add("child", "test", fail_then_ok)
        # First close: journal drive fails, entry stays, error surfaces.
        err1 = _run_catch(w.close)
        assert err1 is not None
        assert "transient failure" in str(err1)
        assert not w._cleanup_journal.empty
        assert w._lifecycle is worker_mod._Lifecycle.CLOSED
        # Second close: retries the journal entry, succeeds.
        err2 = _run_catch(w.close)
        assert err2 is None
        assert w._cleanup_journal.empty
        assert attempts == [1, 1]


class TestShmNames:
    def test_shm_name_with_token(self):
        assert _shm_name("abc123456789", "sub-0") == "sp-abc12345-sub-0"

    def test_shm_name_without_token(self):
        assert _shm_name("", "sub-0") is None


class TestNewWorkerClose:
    def test_new_worker_close_journal_empty(self):
        w = Worker(level=3, num_sub_workers=1)
        w.close()
        assert w._cleanup_journal.empty
        assert w._lifecycle is worker_mod._Lifecycle.CLOSED
