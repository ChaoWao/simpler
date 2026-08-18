# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""One word per level of the worker hierarchy, and the trace prefix derived from it.

`[STRACE]` span names lead with the level that emitted them, so a reader can tell
a chip-runtime span from a host-scheduler one without consulting the tree. The
words live here because the emitting code cannot supply them: `Orchestrator` and
`WorkerThread` are level-agnostic — the same code runs at every level above the
chip — so the word is a property of the process, resolved once from its Worker's
level.

Above `host` the physical layer is not fixed: sometimes a host sits under a pod,
sometimes directly under a supernode. Naming those levels after a specific
interconnect does not work either (mostly SU, sometimes RoCE). So they are named
by how many network hops separate them from the host — `network1` is one hop up,
whatever hardware implements it. The digit counts hops, not the level number:
`network1` is L4.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = ["WorkerLevel", "span_prefix"]


class WorkerLevel(IntEnum):
    """A level in the worker hierarchy, named by the role it plays.

    `IntEnum` because `level` is an integer everywhere it is compared
    (`worker.py` gates on `level >= 3` and `level >= 4`) and is serialized as
    one. Membership is therefore additive: naming a level here does not
    constrain any of those comparisons.

    `L1` is absent because no such level exists; the ladder skips from the
    in-core `core` to the whole-chip `chip`.
    """

    core = 0
    chip = 2
    host = 3
    network1 = 4
    network2 = 5
    network3 = 6


def span_prefix(level: int) -> str:
    """Return the `[STRACE]` name prefix for `level`, e.g. 3 -> ``"host"``.

    Raises `ValueError` for a level the ladder does not name, rather than
    inventing a word. L5 and L6 run what
    `docs/hierarchical-level-runtime.md` calls "the local L4 code path" and are
    untested; when one is first exercised, a missing word should surface as a
    failure here rather than as spans quietly labelled with the wrong level —
    which is the defect this prefix exists to fix.

    Note `WorkerLevel(level).name`, not `str(WorkerLevel(level))`: this project
    targets Python 3.10, where `enum.StrEnum` does not exist and `str()` on any
    `Enum` yields ``"WorkerLevel.host"``.
    """
    try:
        return WorkerLevel(level).name
    except ValueError:
        named = ", ".join(f"{member.value}={member.name}" for member in WorkerLevel)
        raise ValueError(f"no span prefix for worker level {level}; the ladder names {named}") from None
