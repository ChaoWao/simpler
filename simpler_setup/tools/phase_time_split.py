#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Split a bind segment's wall time into running and waiting, per segment and per thread.

Reads the `chip.run.bind.*` spans a run emits under
`SIMPLER_HBG_BIND_BREAKDOWN_ENABLE=1`. A segment's duration alone cannot say whether it
was computing or blocked, and that decides where to look next:

  on-CPU dominates   -> it is running. Split it by the fault count and by the syscalls
                        inside the phase's window, neither of which quantises.
  off-CPU dominates  -> it is waiting. `nvcsw` counts blocking and `nivcsw` counts being
                        preempted, which is a loaded box rather than the code.

Times come from per-thread CPU clocks, in nanoseconds, so a phase of a millisecond
resolves. rusage times are deliberately not used: `ru_utime`/`ru_stime` are accounted per
scheduler tick, 10 ms at `CLK_TCK=100`, so on such a phase they quantise to either zero
or a whole tick. rusage still supplies the three counters, which are event counts and do
not quantise.

`cpu` is the bind thread's own, so `dur - cpu` is the time it spent off CPU. `reccpu` is
every recording worker's summed, so `rec/dur` is how many threads' worth of work ran
alongside it — a ratio, not something to subtract from the wall.

Cold and warm binds are reported separately, because they answer different questions: a
cold bind pays the one-off cost of standing recorder storage and the arenas up, and a
warm one does not.
"""

import argparse
import re
import statistics
import sys

from simpler_setup.tools.strace_timing import expand_log_source, parse_spans

FIELDS = ("minflt", "tminflt", "nivcsw", "nvcsw", "cpu_ns", "rec_cpu_ns")
# The stage the segments subdivide. A name one level below it is a segment; two
# levels below is an orchestrator operation, which carries none of these counters.
BIND_SPAN = "chip.run.bind"


def parse(paths):
    """One row per bind segment, with the counters its attributes carry.

    A counter is `None` when the attribute string was truncated before it — the
    logger marks such a field with `~` — so a caller can report the row as
    incomplete instead of computing over a missing number.
    """
    prefix = f"{BIND_SPAN}."
    rows = []
    for path in paths:
        with open(path, errors="replace") as handle:
            for span in parse_spans(handle):
                if span.is_device or not span.name.startswith(prefix):
                    continue
                phase = span.name[len(prefix) :]
                if "." in phase:
                    continue
                row = {
                    "phase": phase,
                    "pid": span.pid,
                    "inv": span.inv,
                    "ts": span.ts,
                    "dur_us": span.dur / 1000.0,
                }
                for field in FIELDS:
                    hit = re.search(rf"\b{field}=(\d+)", span.attrs)
                    row[field] = int(hit.group(1)) if hit else None
                rows.append(row)
    return rows


def cold_keys(rows):
    """The `(pid, inv)` of each rank's first bind, which is warm-up.

    A span carries the run epoch its bind was allocated, so the cold bind is the
    earliest one that pid ran rather than a count of leading rows the caller has
    to supply.
    """
    earliest = {}
    for row in rows:
        key = (row["pid"], row["inv"])
        current = earliest.get(row["pid"])
        if current is None or row["ts"] < current[1]:
            earliest[row["pid"]] = (key, row["ts"])
    return {key for key, _ in earliest.values()}


def summarise(group):
    """Medians of the per-bind figures, so one outlying bind cannot set the row."""

    def med(pick):
        return statistics.median(pick(row) for row in group)

    return {
        "n": len(group),
        "dur": med(lambda r: r["dur_us"]),
        "cpu": med(lambda r: r["cpu_ns"] / 1000.0),
        # Per bind, so the median of the differences rather than the difference of medians.
        "offcpu": med(lambda r: max(0.0, r["dur_us"] - r["cpu_ns"] / 1000.0)),
        "reccpu": med(lambda r: r["rec_cpu_ns"] / 1000.0),
        "rec_ratio": med(lambda r: r["rec_cpu_ns"] / 1000.0 / r["dur_us"] if r["dur_us"] else 0.0),
        "minflt": med(lambda r: r["minflt"]),
        "tminflt": med(lambda r: r["tminflt"]),
        "nvcsw": med(lambda r: r["nvcsw"]),
        "nivcsw": med(lambda r: r["nivcsw"]),
        # One thread cannot spend more CPU than the phase's own wall, so a row like this
        # means the counter mark belongs to a different span than the duration. Each
        # phase now carries its own mark in the frame that opened it, so no other phase
        # — nested, or on a concurrently preparing thread — can take it; this stays as a
        # backstop, and a non-zero count is a defect in the runtime rather than a known
        # limitation. The affected row's cpu, offcpu and off% describe neither span and
        # are flagged rather than clamped.
        "unmarked": sum(1 for row in group if row["cpu_ns"] / 1000.0 > row["dur_us"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "log", nargs="+", help="run log(s) carrying `chip.run.bind.*` spans, or a directory of host.*.log files"
    )
    parser.add_argument("--phase", default=None, help="only this segment")
    args = parser.parse_args()

    paths = [path for source in args.log for path in expand_log_source(source)]
    rows = parse(paths)
    if not rows:
        sys.exit(
            f"{' '.join(args.log)}: no `chip.run.bind.<segment>` spans — was SIMPLER_HBG_BIND_BREAKDOWN_ENABLE=1 set?"
        )
    if all(row["cpu_ns"] is None for row in rows):
        sys.exit(f"{' '.join(args.log)}: spans carry no cpu_ns — this log predates the per-thread CPU clocks")

    # Warm-up is decided over every bind the log holds, before any row is
    # dropped: a bind whose attributes were truncated is still a bind that ran,
    # and dropping it first would promote its successor to cold and move a warm
    # bind's numbers into the cold row.
    # Warm-up is decided over every bind the log holds, before any row is
    # dropped: a bind whose attributes were truncated is still a bind that ran,
    # and dropping it first would promote its successor to cold and move a warm
    # bind's numbers into the cold row.
    cold = cold_keys(rows)

    # A row missing a counter had its attribute string truncated, so every figure
    # derived from that counter would be a guess. Report how many rather than
    # compute over them or abort the whole table.
    incomplete = [row for row in rows if any(row[field] is None for field in FIELDS)]
    rows = [row for row in rows if row not in incomplete]

    phases = {}
    for row in rows:
        phases.setdefault(row["phase"], []).append(row)

    header = (
        f"{'phase':<16}{'':>6}{'n':>3}{'dur':>13}{'cpu':>13}{'offcpu':>13}{'off%':>6}"
        f"{'reccpu':>13}{'rec/dur':>9}{'minflt':>8}{'tminflt':>8}{'nvcsw':>8}{'nivcsw':>7}"
    )
    print(header)
    print("-" * len(header))
    flagged = False
    for phase, group in phases.items():
        if args.phase and phase != args.phase:
            continue
        by_warmth = {
            "cold": [row for row in group if (row["pid"], row["inv"]) in cold],
            "warm": [row for row in group if (row["pid"], row["inv"]) not in cold],
        }
        for label, part in by_warmth.items():
            if not part:
                continue
            stats = summarise(part)
            off_pct = 100.0 * stats["offcpu"] / stats["dur"] if stats["dur"] else 0.0
            mark = " !" if stats["unmarked"] else ""
            flagged = flagged or bool(mark)
            print(
                f"{phase:<16}{label:>6}{stats['n']:>3}{stats['dur']:>13.1f}{stats['cpu']:>13.1f}"
                f"{stats['offcpu']:>13.1f}{off_pct:>5.0f}%{stats['reccpu']:>13.1f}{stats['rec_ratio']:>9.2f}"
                f"{stats['minflt']:>8.0f}{stats['tminflt']:>8.0f}{stats['nvcsw']:>8.0f}{stats['nivcsw']:>7.0f}{mark}"
            )
    print("\nmedians over the binds in each group; times in us. cpu/offcpu are the bind")
    print("thread's own, reccpu is every recording worker's, so rec/dur is concurrency.")
    if incomplete:
        print(f"{len(incomplete)} span(s) had a truncated attribute string and are excluded: a")
        print("  counter the logger cut is missing entirely, and every figure over it would")
        print("  be a guess. The attribute field is 192 bytes; see docs/dfx/hbg-bind-phases.md.")
    if flagged:
        print("! at least one bind reported more CPU than the segment's own wall, so its")
        print("  counter mark covers a different span than its duration: that row's")
        print("  cpu/offcpu/off% are unusable. Each phase carries its own mark, so this is")
        print("  a runtime defect worth reporting rather than a known limitation.")


if __name__ == "__main__":
    main()
