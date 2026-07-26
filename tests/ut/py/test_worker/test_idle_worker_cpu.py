# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Idle forked workers must not hold a core.

A child whose mailbox poll never pauses burns a full core for its entire
lifetime, so once the live sub/child workers outnumber the available cores
they starve each other and parallel dispatch degrades toward serial.

The assertion is on **CPU time consumed**, not wall time: it measures the
spin directly instead of inferring it from how long some parallel workload
took, so it does not depend on the scheduler's choices and cannot flake under
load. Reads `/proc`, so it is Linux-only; the loops under test are
platform-independent.
"""

import os
import subprocess
import sys
import time

import pytest
from simpler.worker import Worker

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc for per-child CPU time")

NUM_SUB_WORKERS = 4
IDLE_WINDOW_S = 1.0
# The paced loop measures ~1% of a core per child; an unpaced one measures
# 80-100%. Anything under this is decisively "not spinning" without depending
# on how loaded the machine is.
MAX_IDLE_CORE_FRACTION = 0.25


def _child_cpu_seconds(pids: list[int]) -> float:
    """Summed utime+stime of `pids`, in seconds."""
    ticks = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat") as f:
                # Fields after the (possibly space-containing) comm field:
                # utime and stime are indices 11 and 12.
                fields = f.read().rsplit(") ", 1)[1].split()
        except OSError:
            continue
        ticks += int(fields[11]) + int(fields[12])
    return ticks / os.sysconf("SC_CLK_TCK")


def test_idle_sub_workers_do_not_spin():
    worker = Worker(level=3, num_sub_workers=NUM_SUB_WORKERS)
    worker.register(lambda args: None)
    worker.init()
    try:
        children = [int(p) for p in subprocess.check_output(["pgrep", "-P", str(os.getpid())]).split()]
        assert len(children) >= NUM_SUB_WORKERS, f"expected at least {NUM_SUB_WORKERS} forked children, saw {children}"

        before = _child_cpu_seconds(children)
        time.sleep(IDLE_WINDOW_S)
        burned = _child_cpu_seconds(children) - before
    finally:
        worker.close()

    per_child = burned / (IDLE_WINDOW_S * len(children))
    assert per_child < MAX_IDLE_CORE_FRACTION, (
        f"{len(children)} idle children burned {burned:.2f} CPU-seconds over {IDLE_WINDOW_S}s "
        f"({per_child:.0%} of a core each, limit {MAX_IDLE_CORE_FRACTION:.0%}) — "
        f"the mailbox poll is spinning instead of pacing itself"
    )
