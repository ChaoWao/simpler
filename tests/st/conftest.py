# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Conftest for scene tests (tests/st/).

sys.path is handled by pyproject.toml [tool.pytest.ini_options] pythonpath.
"""

import time

import pytest


@pytest.fixture
def drain_host_log():
    """Read captured output with the host-log writer drained first.

    A `[STRACE]` record reaches captured stderr through the process writer
    thread, so a bare `capfd.readouterr()` races it: the reader can return before
    the last records are written, and the failure then looks like a missing span
    rather than a timing problem. Every test that counts or matches spans in
    captured output needs this.

    The wait is bounded but it is not the verdict. Producers are quiescent by the
    time a test reads — the run has completed — so `pending_record_count`
    reaching zero is a real drain rather than a deadline standing in for
    correctness, and the caller's own assertion stays the thing that decides.
    Exhausting the bound means the writer is genuinely stuck, and the message
    reports the drop counter so a queue loss is not mistaken for a slow drain.
    """
    from simpler.task_interface import (  # noqa: PLC0415
        _flush_host_log,
        _host_log_dropped_records,
        _host_log_pending_records,
    )

    def _drain(capfd, timeout_s: float = 5.0) -> str:
        dropped_before = _host_log_dropped_records()
        chunks: list[str] = []
        deadline = time.monotonic() + timeout_s
        while True:
            flushed = _flush_host_log(100)
            captured = capfd.readouterr()
            chunks.extend((captured.err, captured.out))
            if flushed and _host_log_pending_records() == 0:
                return "".join(chunks)
            if time.monotonic() >= deadline:
                pending = _host_log_pending_records()
                dropped = _host_log_dropped_records() - dropped_before
                raise AssertionError(
                    f"host-log writer did not drain within {timeout_s:.1f}s: "
                    f"pending={pending}, dropped_delta={dropped}, last_flush={flushed}"
                )
            time.sleep(0.01)

    return _drain
