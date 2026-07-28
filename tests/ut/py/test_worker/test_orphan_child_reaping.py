# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""No forked worker child outlives the Worker that forked it.

A parent killed without `Worker.close()` — a `timeout`, an OOM kill, a
cancelled CI job — cannot tell its children to shut down, so the children have
to notice on their own. Before they did, every survivor sat in its mailbox poll
holding a core for the lifetime of the machine (#1493).

The parent is a subprocess rather than the pytest process itself, because the
test has to SIGKILL it: a parent that exits cleanly reaps its own children and
would prove nothing.

The subprocess writes to a file rather than a pipe, and is never waited on with
`communicate()`. Orphans inherit the parent's stdout, so on regression they hold
a pipe's write end open indefinitely — reading one would hang this test for as
long as the bug survives, instead of failing it.
"""

import os
import signal
import subprocess
import sys
import textwrap
import time

PARENT_EXIT_TIMEOUT_S = 60.0
REAP_TIMEOUT_S = 20.0
NUM_SUB_WORKERS = 3

# Prints the pids it forked, then dies by SIGKILL — no close(), no SHUTDOWN.
#
# The pids come from the Worker's own bookkeeping, never from `pgrep -P`:
# enumerating children externally also catches the transient shell that runs
# pgrep, and inside a container's shallow pid namespace it picks up unrelated
# low pids — which the test would then wait on and, worse, SIGKILL.
#
# The pids go to a dedicated file named on the command line, not to stdout:
# the parent's stdout and stderr carry log lines and warnings, and scraping
# integers out of that stream picks up any bare number they happen to contain.
_PARENT_SRC = textwrap.dedent(
    """
    import os, sys
    from simpler.worker import Worker

    worker = Worker(level=3, num_sub_workers=%d)
    worker.register(lambda args: None)
    worker.init()
    with open(sys.argv[1], "w") as f:
        f.write(" ".join(str(p) for p in sorted(worker._sub_pids)))
        f.flush()
        os.fsync(f.fileno())
    os.kill(os.getpid(), 9)
    """
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_children_exit_when_parent_is_killed(tmp_path):
    pid_path = tmp_path / "child_pids.txt"
    log_path = tmp_path / "parent_output.txt"
    reported: list[int] = []
    try:
        with log_path.open("w") as log:
            parent = subprocess.Popen(
                [sys.executable, "-c", _PARENT_SRC % NUM_SUB_WORKERS, str(pid_path)], stdout=log, stderr=log
            )
        returncode = parent.wait(timeout=PARENT_EXIT_TIMEOUT_S)

        # The parent is supposed to die from its own SIGKILL. Any other exit
        # means it failed before forking, and its own output says why —
        # without this the symptom surfaces as an unexplained "expected 3 pids,
        # got []" that reads like a defect in the code under test.
        assert returncode == -signal.SIGKILL, (
            f"parent exited {returncode}, expected -{int(signal.SIGKILL)} (its own SIGKILL): "
            f"it failed before it could fork\nparent output:\n{log_path.read_text()}"
        )

        reported = [int(tok) for tok in pid_path.read_text().split()]
        assert len(reported) == NUM_SUB_WORKERS, (
            f"expected exactly {NUM_SUB_WORKERS} forked sub-worker pids, parent reported {reported}\n"
            f"parent output:\n{log_path.read_text()}"
        )

        deadline = time.monotonic() + REAP_TIMEOUT_S
        survivors = reported
        while survivors and time.monotonic() < deadline:
            time.sleep(0.1)
            survivors = [p for p in survivors if _alive(p)]

        assert not survivors, (
            f"{len(survivors)} of {len(reported)} children outlived their SIGKILLed parent by "
            f"{REAP_TIMEOUT_S:.0f}s: {survivors} — a child is not noticing that its parent is gone"
        )
    finally:
        # A regression leaves spinning processes behind; never hand them on to
        # the next test, whether the assertion passed, failed, or never ran.
        for pid in reported:
            try:
                os.kill(pid, 9)
            except OSError:
                pass
