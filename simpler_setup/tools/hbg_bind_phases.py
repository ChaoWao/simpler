#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-phase statistics from a `bind phase=` LOG_TIMING breakdown.

Reads the log a `SIMPLER_HBG_BIND_BREAKDOWN_ENABLE=1 --rounds N` run leaves
behind and reports each bind phase's minimum, median and maximum across the warm
binds, plus the control-plane total.

What the phases are, and how to compare two runs:
docs/dfx/hbg-bind-phases.md.
"""

import argparse
import re
import statistics
import sys

PHASE_LINE = re.compile(r"bind phase=(\w+) start_ns=(\d+) dur_ns=(\d+)")
# The recipe's first log line, recording the command and the commit the run used.
# Two runs are comparable only if it matches, so it is echoed above the table.
STAMP_LINE = re.compile(r"^\[stamp\] (.*)$")

# The segments between "the caller's data is in place" and "the device can run".
# `args` and `host_view_close` are per-byte costs over the weights the case
# stages, so they belong to getting the weights resident, not to dispatch.
CONTROL_PLANE = ("host_orch", "graph_upload", "relocate", "sm_h2d", "arena_h2d")

# Display order: the bind stage's own sequence, so a reader can follow it down.
PHASE_ORDER = (
    "args",
    "arena_build",
    "static_arena",
    "gm_heap",
    "shared_mem",
    "runtime_init",
    "host_orch",
    "graph_upload",
    "relocate",
    "sm_h2d",
    "arena_h2d",
    "host_view_close",
)

# The last segment of a bind, and so what closes one: the segments are not
# contiguous in time, so timestamp order does not group them.
BIND_CLOSING_PHASE = "arena_h2d"


def parse_binds(path: str) -> list[dict[str, float]]:
    """Group `bind phase=` lines into binds, in milliseconds."""
    binds: list[dict[str, float]] = []
    current: dict[str, float] = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = PHASE_LINE.search(line)
            if match is None:
                continue
            phase, _start_ns, dur_ns = match.group(1), int(match.group(2)), int(match.group(3))
            current[phase] = dur_ns / 1e6
            if phase == BIND_CLOSING_PHASE:
                binds.append(current)
                current = {}
    if current:
        binds.append(current)
    # A group without `host_orch` is not a bind.
    return [b for b in binds if "host_orch" in b]


def parse_stamp(path: str) -> str:
    """The run's environment stamp, or "" for a log this script did not produce."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = STAMP_LINE.match(line.rstrip("\n"))
            if match is not None:
                return match.group(1)
    return ""


def spread(values: list[float]) -> tuple[float, float, float]:
    """Minimum, median and maximum. The median averages the two central values."""
    return min(values), statistics.median(values), max(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", help="run log carrying the `bind phase=` lines")
    parser.add_argument(
        "--ranks",
        type=int,
        default=None,
        help="ranks in the run; the first bind of each is warm-up and is dropped. "
        "Default: inferred as the number of binds divided by the round count when "
        "--rounds is given, else 1.",
    )
    parser.add_argument("--rounds", type=int, default=None, help="rounds the run was given, to infer --ranks")
    parser.add_argument("--keep-first", action="store_true", help="keep the cold bind in the statistics")
    args = parser.parse_args()

    binds = parse_binds(args.log)
    if not binds:
        print(
            f"{args.log}: no `bind phase=` lines. SIMPLER_HBG_BIND_BREAKDOWN_ENABLE=1 and "
            "SIMPLER_LOG_LEVEL=TIMING must both be set, and a multi-device case must be "
            "invoked as its own child command -- see docs/dfx/hbg-bind-phases.md.",
            file=sys.stderr,
        )
        return 1

    ranks = args.ranks
    if ranks is None:
        ranks = max(1, len(binds) // args.rounds) if args.rounds else 1
    dropped = 0 if args.keep_first else min(ranks, len(binds))
    warm = binds[dropped:]
    if not warm:
        print(
            f"{args.log}: {len(binds)} bind(s) over {ranks} rank(s), all of them cold. The first "
            "bind of each rank is warm-up, so a warm bind needs --rounds 2 or more; --keep-first "
            "reports the cold binds instead.",
            file=sys.stderr,
        )
        return 1

    print(f"{args.log}")
    stamp = parse_stamp(args.log)
    if stamp:
        print(f"  {stamp}")
    else:
        print("  (no `[stamp]` line: the command and commit behind these numbers have")
        print("   to be established by hand before comparing them to anything)")
    print(f"  {len(binds)} binds, {len(warm)} warm ({dropped} cold dropped, ranks={ranks})\n")
    print(f"  {'phase':<18}{'min':>10}{'median':>10}{'max':>10}   n")
    for phase in PHASE_ORDER:
        values = [b[phase] for b in warm if phase in b]
        if not values:
            continue
        low, mid, high = spread(values)
        mark = "*" if phase in CONTROL_PLANE else " "
        print(f" {mark}{phase:<18}{low:>10.3f}{mid:>10.3f}{high:>10.3f}  {len(values):3d}")

    unknown = sorted({k for b in warm for k in b} - set(PHASE_ORDER))
    if unknown:
        print(f"\n  phases this tool does not know about: {', '.join(unknown)}")
        print("  (add them to PHASE_ORDER, and to CONTROL_PLANE if a dispatch change can move them)")

    # The control-plane set is not fixed: a change can retire a phase outright, so
    # the total covers the phases this run has and names the ones absent from every
    # bind. Absent from only some binds is a truncated log rather than a retired
    # phase, and would understate a bind, so those binds are excluded and warned
    # about.
    present = [k for k in CONTROL_PLANE if any(k in b for b in warm)]
    partial = [k for k in present if not all(k in b for b in warm)]
    retired = [k for k in CONTROL_PLANE if k not in present]
    complete = [b for b in warm if all(k in b for k in present)]
    if complete:
        totals = [sum(b[k] for k in present) for b in complete]
        low, mid, high = spread(totals)
        print("\n  * = control plane, summed within each bind and then:")
        print(f"    total{'':<13}{low:>10.3f}{mid:>10.3f}{high:>10.3f}  {len(totals):3d}   (ms)")
        if retired:
            print(f"    over {len(present)} of {len(CONTROL_PLANE)} phases; absent from every")
            print(f"    bind: {', '.join(retired)}")
        if partial:
            print(f"    WARNING: {', '.join(partial)} is missing from some binds but not all;")
            print("    those binds are excluded, and the total may not describe the run")
        print("\n  Compare by min, over the same phase set, against a log with the same")
        print("  stamp; see docs/dfx/hbg-bind-phases.md 'Comparing two branches'.")
    else:
        print(f"\n  no bind carries any of {CONTROL_PLANE}; the control-plane total is not computable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
