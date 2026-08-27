#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Analyze Simpler chip.run STRACE spans and emit a Perfetto swimlane."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

READY_RE = re.compile(r"\[chip_process pid=(\d+) dev=(\d+)\] ready")
SPAN_RE = re.compile(r"\[STRACE\].*\bpid=(\d+).*\binv=(\d+).*\bname=(\S+).*\bts=(\d+)\s+dur=(\d+)")
FIELDS_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)=([^ ]+)")
TRACKED = {
    "chip.run",
    "chip.run.pre_bind",
    "chip.run.bind",
    "chip.run.post_bind",
    "chip.run.runner_run",
    "chip.run.runner_run.device_wall",
    "chip.run.validate",
    "chip.run.claim_release",
}
REQUIRED = {"chip.run", "chip.run.runner_run", "chip.run.runner_run.device_wall", "chip.run.validate"}


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--threshold-ms", type=float, default=5.0)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--trace-out", type=Path, required=True)
    args = parser.parse_args()

    text = args.log.read_text(errors="replace")
    pid_to_device = {int(pid): int(device) for pid, device in READY_RE.findall(text)}
    spans = defaultdict(dict)
    trace_events = []
    for line in text.splitlines():
        match = SPAN_RE.search(line)
        if not match:
            continue
        pid, invocation, name, ts, dur = match.groups()
        pid = int(pid)
        if pid not in pid_to_device or name not in TRACKED:
            continue
        device = pid_to_device[pid]
        invocation = int(invocation)
        ts_ns = int(ts)
        dur_ns = int(dur)
        fields = {key: value for key, value in FIELDS_RE.findall(line)}
        spans[(invocation, device)][name] = (ts_ns, dur_ns)
        if name == "chip.run.runner_run.device_wall":
            continue
        trace_events.append(
            {
                "name": name.removeprefix("chip.run."),
                "cat": "simpler.host_strace",
                "ph": "X",
                "pid": 1000 + device,
                "tid": device,
                "ts": ts_ns / 1000.0,
                "dur": dur_ns / 1000.0,
                "args": {"round": invocation, **fields},
            }
        )

    devices = sorted(pid_to_device.values())
    first = args.warmup + 1
    last = args.warmup + args.rounds
    rounds = []
    outliers = []
    for invocation in range(first, last + 1):
        rows = {}
        for device in devices:
            row = spans.get((invocation, device), {})
            missing = REQUIRED - row.keys()
            if missing:
                raise RuntimeError(f"round {invocation} device {device} missing spans: {sorted(missing)}")
            root_ts, root_dur = row["chip.run"]
            runner_ts, runner_dur = row["chip.run.runner_run"]
            _device_ts, device_wall_dur = row["chip.run.runner_run.device_wall"]
            validate_ts, validate_dur = row["chip.run.validate"]
            rows[device] = {
                "root_start_ms": root_ts / 1e6,
                "step_ms": root_dur / 1e6,
                "runner_start_ms": runner_ts / 1e6,
                "runner_ms": runner_dur / 1e6,
                "device_wall_ms": device_wall_dur / 1e6,
                "runner_host_excess_ms": (runner_dur - device_wall_dur) / 1e6,
                "runner_end_ms": (runner_ts + runner_dur) / 1e6,
                "runner_to_validate_gap_ms": (validate_ts - runner_ts - runner_dur) / 1e6,
                "validate_ms": validate_dur / 1e6,
                "step_end_ms": (root_ts + root_dur) / 1e6,
            }
        starts = [row["runner_start_ms"] for row in rows.values()]
        ends = [row["runner_end_ms"] for row in rows.values()]
        step_ends = [row["step_end_ms"] for row in rows.values()]
        record = {
            "round": invocation - args.warmup,
            "invocation": invocation,
            "runner_start_skew_ms": max(starts) - min(starts),
            "runner_end_skew_ms": max(ends) - min(ends),
            "step_end_skew_ms": max(step_ends) - min(step_ends),
            "rows": rows,
        }
        rounds.append(record)

        median_start = statistics.median(starts)
        median_end = statistics.median(ends)
        for device, row in rows.items():
            if row["runner_start_ms"] - median_start > args.threshold_ms:
                outliers.append(
                    {
                        "round": record["round"],
                        "device": device,
                        "category": "runner_late_start",
                        "above_median_ms": row["runner_start_ms"] - median_start,
                    }
                )
            if row["runner_end_ms"] - median_end > args.threshold_ms:
                outliers.append(
                    {
                        "round": record["round"],
                        "device": device,
                        "category": "runner_long_tail",
                        "above_median_ms": row["runner_end_ms"] - median_end,
                    }
                )

    metrics = {}
    for field in ("runner_start_skew_ms", "runner_end_skew_ms", "step_end_skew_ms"):
        metrics[field] = _summary([record[field] for record in rounds])
    for field in (
        "runner_ms",
        "device_wall_ms",
        "runner_host_excess_ms",
        "runner_to_validate_gap_ms",
        "validate_ms",
        "step_ms",
    ):
        metrics[field] = _summary([row[field] for record in rounds for row in record["rows"].values()])

    result = {
        "source": str(args.log),
        "devices": devices,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "threshold_ms": args.threshold_ms,
        "metrics_ms": metrics,
        "outlier_counts": {
            category: sum(item["category"] == category for item in outliers)
            for category in ("runner_late_start", "runner_long_tail")
        },
        "outliers": outliers,
        "round_data": rounds,
    }
    args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    for device in devices:
        trace_events.extend(
            [
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": 1000 + device,
                    "tid": 0,
                    "args": {"name": f"Device {device}"},
                },
                {"name": "thread_name", "ph": "M", "pid": 1000 + device, "tid": device, "args": {"name": "chip.run"}},
            ]
        )
    trace = {
        "displayTimeUnit": "ms",
        "metadata": {"source": str(args.log), "devices": devices, "simpler_only": True},
        "traceEvents": trace_events,
    }
    args.trace_out.write_text(json.dumps(trace, separators=(",", ":")))
    print(json.dumps({"devices": devices, "metrics_ms": metrics, "outlier_counts": result["outlier_counts"]}))


if __name__ == "__main__":
    main()
