#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Export aligned Simpler host and device STRACE spans as a Chrome trace."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from trace_effective import Span, parse_spans


def _event(
    span: Span,
    *,
    category: str,
    pid: int,
    timestamp_ns: int,
    args: dict[str, object],
) -> dict[str, object]:
    return {
        "name": span.name,
        "cat": category,
        "ph": "X",
        "pid": pid,
        "tid": span.tid,
        "ts": timestamp_ns / 1000.0,
        "dur": span.duration_ms * 1000.0,
        "args": args,
    }


def export(log_path: Path, output_dir: Path) -> dict[str, object]:
    spans = parse_spans(log_path)
    by_inv: dict[tuple[int, int], list[Span]] = {}
    for span in spans:
        by_inv.setdefault((span.pid, span.inv), []).append(span)

    events: list[dict[str, object]] = []
    for pid in sorted({span.pid for span in spans}):
        for output_pid, name in (
            (pid, f"Simpler Host pid={pid}"),
            (pid + 20_000_000, f"AICPU device phases pid={pid}"),
            (pid + 30_000_000, f"AICore scheduler pid={pid}"),
        ):
            events.append(
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": output_pid,
                    "tid": 0,
                    "args": {"name": name},
                }
            )

    phase_counts: Counter[str] = Counter()
    unaligned = 0
    host_count = 0
    for (pid, inv), group in by_inv.items():
        host_runner = next(
            (
                span
                for span in group
                if span.name.endswith(".runner_run") and span.attrs.get("clk") != "dev"
            ),
            None,
        )
        for span in group:
            attrs: dict[str, object] = {"inv": inv, "hid": span.hid, **span.attrs}
            if span.attrs.get("clk") != "dev":
                host_count += 1
                events.append(
                    _event(
                        span,
                        category="simpler_host",
                        pid=pid,
                        timestamp_ns=span.timestamp_ns,
                        args=attrs,
                    )
                )
                continue
            if host_runner is None:
                unaligned += 1
                continue
            phase_counts[span.name] += 1
            is_aicpu = span.name.endswith(".orch")
            attrs["alignment"] = "chip.run.runner_run start"
            events.append(
                _event(
                    span,
                    category="aicpu" if is_aicpu else "aicore",
                    pid=pid + (20_000_000 if is_aicpu else 30_000_000),
                    timestamp_ns=host_runner.timestamp_ns + span.timestamp_ns,
                    args=attrs,
                )
            )

    required = (
        "chip.run.runner_run.device_wall",
        "chip.run.runner_run.device_wall.orch",
        "chip.run.runner_run.device_wall.sched",
    )
    missing = [name for name in required if phase_counts[name] == 0]
    if missing:
        raise RuntimeError(f"missing native device STRACE phases: {missing}")
    if unaligned:
        raise RuntimeError(f"native device STRACE has {unaligned} unaligned spans")

    events.sort(
        key=lambda event: (
            float(event.get("ts", 0)),
            int(event["pid"]),
            int(event["tid"]),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = output_dir / "simpler_strace_timeline.json"
    timeline_path.write_text(
        json.dumps(
            {
                "traceEvents": events,
                "displayTimeUnit": "ms",
                "metadata": {
                    "schema": "simpler-strace-chrome-timeline-v1",
                    "device_clock_alignment": "per-invocation chip.run.runner_run start",
                },
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": "simpler-strace-timeline-summary-v1",
        "passed": True,
        "timeline": timeline_path.name,
        "trace_event_count": len(events),
        "host_span_count": host_count,
        "device_span_count": sum(phase_counts.values()),
        "unaligned_device_span_count": unaligned,
        "device_phase_counts": dict(sorted(phase_counts.items())),
        "aicpu_phase": "chip.run.runner_run.device_wall.orch",
        "aicore_phase": "chip.run.runner_run.device_wall.sched",
    }
    (output_dir / "simpler_strace_timeline_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export(args.log.resolve(), args.output_dir.resolve()),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
