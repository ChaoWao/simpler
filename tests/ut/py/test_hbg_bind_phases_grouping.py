# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Grouping of `bind phase=` lines into binds."""

from __future__ import annotations

import sys

from simpler_setup.tools import hbg_bind_phases

# A bind emits its segments in one contiguous burst, in this order. `arena_h2d`
# is second to last, so it does not close a bind; `host_view_close` does.
EMISSION_ORDER = (
    "args",
    "arena_build",
    "static_arena",
    "gm_heap",
    "shared_mem",
    "runtime_init",
    "host_orch",
    "graph_upload",
    "arena_h2d",
    "host_view_close",
)


def _write_binds(path, count: int, *, order=EMISSION_ORDER, omit: tuple[str, ...] = ()) -> None:
    """One line per segment per bind, each duration encoding its own bind index."""
    lines = ["[stamp] command commit=abc"]
    clock = 0
    for bind_index in range(count):
        for phase in order:
            if phase in omit:
                continue
            clock += 1
            # dur_ns = (bind_index + 1) ms, so a misattributed segment is visible.
            lines.append(f"bind phase={phase} start_ns={clock} dur_ns={(bind_index + 1) * 1000000}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_each_segment_lands_in_its_own_bind(tmp_path):
    log = tmp_path / "run.log"
    _write_binds(log, 3)

    binds = hbg_bind_phases.parse_binds(str(log))

    assert len(binds) == 3
    for index, bind in enumerate(binds):
        assert sorted(bind) == sorted(EMISSION_ORDER), f"bind {index} is not whole"
        for phase, duration in bind.items():
            assert duration == index + 1, f"bind {index} carries another bind's {phase}"


def test_a_bind_missing_its_closing_segment_does_not_swallow_the_next(tmp_path):
    """An interrupted bind stays one bind rather than merging with its successor."""
    log = tmp_path / "partial.log"
    _write_binds(log, 1)
    with log.open("a", encoding="utf-8") as handle:
        for phase in ("args", "host_orch", "arena_h2d"):
            handle.write(f"bind phase={phase} start_ns=999 dur_ns=2000000\n")

    binds = hbg_bind_phases.parse_binds(str(log))

    assert len(binds) == 2
    assert binds[0]["host_view_close"] == 1
    assert binds[1]["host_orch"] == 2
    assert "host_view_close" not in binds[1]


def test_a_bind_omitting_a_leading_segment_is_reported(tmp_path, monkeypatch, capsys):
    """A bind short of a leading segment absorbs the next bind's copy of it.

    Grouping closes on a repeated name, so a segment the previous bind lacks and
    the next one emits before any they share lands in the previous bind. `args`
    is emitted unconditionally and first, so no current bind can omit it — this
    pins what happens if that ever changes, and that the reader is told.
    """
    log = tmp_path / "no_args.log"
    _write_binds(log, 1, order=EMISSION_ORDER[1:])
    with log.open("a", encoding="utf-8") as handle:
        for phase in EMISSION_ORDER:
            handle.write(f"bind phase={phase} start_ns=999 dur_ns=2000000\n")

    binds = hbg_bind_phases.parse_binds(str(log))

    assert [len(bind) for bind in binds] == [10, 9]
    assert binds[0]["args"] == 2, "the second bind's args landed in the first"
    assert "args" not in binds[1]

    monkeypatch.setattr(sys, "argv", ["hbg_bind_phases", str(log), "--keep-first"])
    assert hbg_bind_phases.main() == 0
    out = capsys.readouterr().out
    assert "not every bind has the same segments; args" in out


def test_a_split_burst_is_reported_rather_than_passed_off_as_a_bind(tmp_path, monkeypatch, capsys):
    """Grouping assumes uninterrupted bursts, and nothing enforces that.

    Ranks share one stream through no lock and the line prefix carries no pid, so
    one rank's burst can land inside another's. The result is binds whose segment
    sets differ, which has to reach the reader.
    """
    log = tmp_path / "split_burst.log"
    lines = ["[stamp] command commit=abc"]

    def emit(phase: str, rank: int) -> None:
        lines.append(f"bind phase={phase} start_ns={len(lines)} dur_ns={rank * 1000000}")

    for phase in EMISSION_ORDER[:4]:  # rank 1 starts
        emit(phase, 1)
    for phase in EMISSION_ORDER:  # rank 2 lands whole, inside it
        emit(phase, 2)
    for phase in EMISSION_ORDER[4:]:  # rank 1 finishes
        emit(phase, 1)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    binds = hbg_bind_phases.parse_binds(str(log))
    assert [len(bind) for bind in binds] == [10, 6], "the split burst is not silently made whole"

    # --keep-first so both binds are reported; dropping the cold one would leave a
    # single bind, whose segment set is trivially uniform with itself.
    monkeypatch.setattr(sys, "argv", ["hbg_bind_phases", str(log), "--keep-first"])
    assert hbg_bind_phases.main() == 0
    out = capsys.readouterr().out
    assert "not every bind has the same segments" in out
    for phase in ("args", "arena_build", "static_arena", "gm_heap"):
        assert phase in out
