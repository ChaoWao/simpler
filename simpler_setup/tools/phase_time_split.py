#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Split a bind phase's wall time into running and waiting, per phase and per thread.

Reads the `bind phase=` markers a run emits under
`SIMPLER_HBG_BIND_BREAKDOWN_ENABLE=1`. A phase's duration alone cannot say whether it
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

FIELDS = ("minflt", "tminflt", "nivcsw", "nvcsw", "cpu_ns", "rec_cpu_ns")
MARKER = re.compile(r"bind phase=(?P<phase>[a-z_]+) start_ns=\d+ dur_ns=(?P<dur>\d+)(?P<rest>.*)")


def parse(path):
    rows = []
    for line in open(path, errors="replace"):
        marker = MARKER.search(line)
        if not marker:
            continue
        row = {"phase": marker.group("phase"), "dur_us": int(marker.group("dur")) / 1000.0}
        for field in FIELDS:
            hit = re.search(rf"\b{field}=(\d+)", marker.group("rest"))
            row[field] = int(hit.group(1)) if hit else None
        rows.append(row)
    return rows


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
    parser.add_argument("log", help="run log carrying `bind phase=` markers")
    parser.add_argument("--ranks", type=int, default=2, help="binds to treat as cold, one per rank (default 2)")
    parser.add_argument("--phase", default=None, help="only this phase")
    args = parser.parse_args()

    rows = parse(args.log)
    if not rows:
        sys.exit(f"{args.log}: no `bind phase=` markers — was SIMPLER_HBG_BIND_BREAKDOWN_ENABLE=1 set?")
    if rows[0]["cpu_ns"] is None:
        sys.exit(f"{args.log}: markers carry no cpu_ns — this log predates the per-thread CPU clocks")

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
        for label, part in (("cold", group[: args.ranks]), ("warm", group[args.ranks :])):
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
    if flagged:
        print("! at least one bind reported more CPU than the segment's own wall, so its")
        print("  counter mark covers a different span than its duration: that row's")
        print("  cpu/offcpu/off% are unusable. Each phase carries its own mark, so this is")
        print("  a runtime defect worth reporting rather than a known limitation.")


if __name__ == "__main__":
    main()
