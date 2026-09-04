#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Extract Serving-compatible runtime timing from a pure Simpler STRACE log."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


_STRACE_RE = re.compile(
    r"\[STRACE\]\s+v=(?P<version>\d+)\s+pid=(?P<pid>\d+)\s+tid=(?P<tid>\d+)\s+"
    r"inv=(?P<inv>\d+)\s+hid=(?P<hid>[0-9a-fA-F]+)\s+depth=(?P<depth>\d+)\s+"
    r"name=(?P<name>\S+)\s+ts=(?P<ts>\d+)\s+dur=(?P<dur>\d+)(?:\s+(?P<attrs>.*))?"
)
_ATTR_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>\S+)")


@dataclass(frozen=True)
class Span:
    pid: int
    tid: int
    inv: int
    hid: str
    name: str
    timestamp_ns: int
    duration_ms: float
    attrs: dict[str, str]


def _parse_attrs(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    return {
        match.group("key"): match.group("value") for match in _ATTR_RE.finditer(text)
    }


def parse_spans(log_path: Path) -> list[Span]:
    spans = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _STRACE_RE.search(line)
        if match is None:
            continue
        spans.append(
            Span(
                pid=int(match.group("pid")),
                tid=int(match.group("tid")),
                inv=int(match.group("inv")),
                hid=match.group("hid"),
                name=match.group("name"),
                timestamp_ns=int(match.group("ts")),
                duration_ms=int(match.group("dur")) / 1_000_000.0,
                attrs=_parse_attrs(match.group("attrs")),
            )
        )
    return spans


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty metric")
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def _span_by_suffix(spans: list[Span], suffix: str) -> Span | None:
    matches = [span for span in spans if span.name.endswith(suffix)]
    if len(matches) > 1:
        raise ValueError(f"invocation has multiple spans ending in {suffix!r}")
    return matches[0] if matches else None


def _required_duration(spans: list[Span], suffix: str) -> float:
    span = _span_by_suffix(spans, suffix)
    if span is None:
        raise ValueError(f"invocation is missing span ending in {suffix!r}")
    return span.duration_ms


def invocation_rows(spans: list[Span]) -> list[dict[str, float | int | bool]]:
    roots = [span for span in spans if span.name in {"chip.run", "simpler_run"}]
    if not roots:
        raise ValueError("no chip.run or simpler_run root spans found")

    spans_by_inv: dict[int, list[Span]] = {}
    for span in spans:
        spans_by_inv.setdefault(span.inv, []).append(span)

    rows = []
    seen_dispatches = set()
    for root in roots:
        dispatch_id = int(root.attrs.get("dispatch_id", root.inv))
        if dispatch_id in seen_dispatches:
            raise ValueError(f"duplicate root span for dispatch {dispatch_id}")
        seen_dispatches.add(dispatch_id)
        invocation = spans_by_inv[root.inv]
        node_dispatch = _span_by_suffix(invocation, "node.dispatch")
        attrs = (
            root.attrs
            if node_dispatch is None
            else {**node_dispatch.attrs, **root.attrs}
        )
        effective = _required_duration(invocation, ".runner_run.device_wall.sched")
        rows.append(
            {
                "dispatch_id": dispatch_id,
                "slot_id": int(attrs.get("slot_id", -1)),
                "generation": int(attrs.get("generation", -1)),
                "prepare_only": attrs.get("prepare_only", "0") == "1",
                "root_start_ns": root.timestamp_ns,
                "root_completion_ns": root.timestamp_ns
                + round(root.duration_ms * 1_000_000),
                "chip_run_lifecycle_ms": root.duration_ms,
                "runner_run_ms": _required_duration(invocation, ".runner_run"),
                "effective_ms": effective,
                "device_wall_ms": _required_duration(
                    invocation, ".runner_run.device_wall"
                ),
                "args_ms": _required_duration(invocation, ".bind.args"),
                "validate_ms": _required_duration(invocation, ".validate"),
            }
        )
    return sorted(rows, key=lambda row: int(row["dispatch_id"]))


def summarize_campaign(
    rows: list[dict[str, float | int | bool]],
    *,
    warmup_runs: int,
    measured_runs: int,
    steps: int,
    steady_skip: int,
) -> dict[str, object]:
    expected = (warmup_runs + measured_runs) * steps
    if len(rows) != expected:
        raise ValueError(f"expected {expected} dispatches, found {len(rows)}")
    if steps <= steady_skip:
        raise ValueError("steady_skip must be smaller than steps")

    measured = rows[warmup_runs * steps :]
    latency_metrics = (
        "runner_run_ms",
        "effective_ms",
        "device_wall_ms",
        "args_ms",
        "validate_ms",
    )
    run_summaries = []
    pooled = {metric: [] for metric in ("rts_completion_interval_ms", *latency_metrics)}
    diagnostic_lifecycles = []
    for run_index in range(measured_runs):
        start = run_index * steps
        run_rows = measured[start : start + steps]
        steady_rows = run_rows[steady_skip:]
        dispatches = [int(row["dispatch_id"]) for row in run_rows]
        completions = [int(row["root_completion_ns"]) for row in run_rows]
        if any(
            current <= previous for previous, current in zip(dispatches, dispatches[1:])
        ):
            raise ValueError(
                f"run {run_index + 1} dispatch ids are not strictly increasing"
            )
        if any(
            current <= previous
            for previous, current in zip(completions, completions[1:])
        ):
            raise ValueError(
                f"run {run_index + 1} root completions are not strictly increasing"
            )

        interval_values = [
            (completions[index] - completions[index - 1]) / 1_000_000.0
            for index in range(steady_skip, steps)
        ]
        metric_stats = {"rts_completion_interval_ms": _stats(interval_values)}
        pooled["rts_completion_interval_ms"].extend(interval_values)
        for metric in latency_metrics:
            values = [float(row[metric]) for row in steady_rows]
            metric_stats[metric] = _stats(values)
            pooled[metric].extend(values)
        lifecycle_values = [float(row["chip_run_lifecycle_ms"]) for row in steady_rows]
        diagnostic_lifecycles.extend(lifecycle_values)
        run_summaries.append(
            {
                "run": run_index + 1,
                "dispatch_first": int(run_rows[0]["dispatch_id"]),
                "dispatch_last": int(run_rows[-1]["dispatch_id"]),
                "steady_dispatch_count": len(steady_rows),
                "metrics": metric_stats,
                "diagnostic_metrics": {
                    "chip_run_lifecycle_ms": _stats(lifecycle_values),
                },
            }
        )

    official = {}
    for metric in pooled:
        run_means = [float(run["metrics"][metric]["mean"]) for run in run_summaries]
        official[metric] = {
            "run_means": run_means,
            "run_mean_stats": _stats(run_means),
            "pooled_steady_stats": _stats(pooled[metric]),
        }

    return {
        "schema": 2,
        "aggregation": {
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "retained_run_numbers": list(range(1, measured_runs + 1)),
            "drop_high_low_run": False,
            "steps_per_run": steps,
            "steady_skip_dispatches": steady_skip,
            "steady_dispatches_per_run": steps - steady_skip,
        },
        "dispatch_contract": {
            "total": len(rows),
            "warmup": warmup_runs * steps,
            "measured": measured_runs * steps,
        },
        "runs": run_summaries,
        "official_metrics": official,
        "diagnostic_metrics": {
            "chip_run_lifecycle_ms": {
                "status": "diagnostic_only",
                "reason": "async-overlapped invocation lifetime is not frame latency or throughput",
                "pooled_steady_stats": _stats(diagnostic_lifecycles),
            }
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=127)
    parser.add_argument("--steady-skip", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = summarize_campaign(
        invocation_rows(parse_spans(args.log)),
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
        steps=args.steps,
        steady_skip=args.steady_skip,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
