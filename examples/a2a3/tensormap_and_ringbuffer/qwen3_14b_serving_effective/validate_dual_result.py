#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate one standalone dual-slot decode result."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from safetensors.torch import load_file

from trace_effective import invocation_rows, parse_spans


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _full_output_sha(first_tokens: list[int], token_rows: list[list[int]]) -> str:
    per_request = [
        [int(first_tokens[batch])] + [int(row[batch]) for row in token_rows]
        for batch in range(len(first_tokens))
    ]
    payload = json.dumps(per_request, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--golden-manifest", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()

    benchmark = json.loads((args.result / "benchmark.json").read_text())
    trace = json.loads((args.result / "trace_summary.json").read_text())
    run = benchmark["runs"][0]
    scenario = benchmark["scenario"]
    _require(benchmark["correctness"]["all_runs_valid"], "benchmark correctness failed")
    _require(scenario["mode"] == "dual", "result is not dual-slot")
    _require(
        scenario["warmup_runs"] == 0 and scenario["measured_runs"] == 1, "expected 0+1"
    )
    _require(scenario["decode_dispatches"] == args.steps, "decode step count mismatch")
    _require(
        run["valid"] and len(run["token_rows"]) == args.steps, "incomplete token rows"
    )

    rows = invocation_rows(parse_spans(args.result / "run.log"))
    _require(len(rows) == args.steps, "native dispatch count mismatch")
    expected_slots = [index % 2 for index in range(args.steps)]
    actual_slots = [int(row["slot_id"]) for row in rows]
    _require(actual_slots == expected_slots, "native slots do not strictly alternate")

    generations = [0, 0]
    for row in rows:
        slot = int(row["slot_id"])
        generations[slot] += 1
        _require(
            int(row["generation"]) == generations[slot],
            "slot generation is not monotonic",
        )
    _require(not bool(rows[0]["prepare_only"]), "first dispatch must start immediately")
    _require(
        all(bool(row["prepare_only"]) for row in rows[1:]),
        "later dispatch was not prepared",
    )

    host_runners = sorted(
        (
            span
            for span in parse_spans(args.result / "run.log")
            if span.name.endswith(".runner_run") and span.attrs.get("clk") != "dev"
        ),
        key=lambda span: span.timestamp_ns,
    )
    _require(len(host_runners) == args.steps, "runner span count mismatch")
    for previous, current in zip(host_runners, host_runners[1:]):
        previous_end = previous.timestamp_ns + round(previous.duration_ms * 1_000_000)
        _require(current.timestamp_ns >= previous_end, "device runner spans overlap")

    aggregation = trace["aggregation"]
    steady_skip = int(aggregation["steady_skip_dispatches"])
    expected_steady = args.steps - steady_skip
    effective = trace["official_metrics"]["effective_ms"]["pooled_steady_stats"]
    _require(effective["count"] == expected_steady, "steady Effective count mismatch")

    full_output_sha = None
    if args.steps == 127:
        fixture_manifest = json.loads((args.fixture / "manifest.json").read_text())
        metadata = load_file(str(args.fixture / fixture_manifest["metadata"]["path"]))
        golden_manifest = json.loads(args.golden_manifest.read_text())
        full_output_sha = _full_output_sha(
            metadata["first_generated_token_ids"].tolist(), run["token_rows"]
        )
        _require(
            full_output_sha == golden_manifest["full_128_output_token_ids_sha256"],
            "full output token SHA differs from golden",
        )

    slots = Counter(actual_slots)
    result = {
        "schema": "simpler-tmr-dual-qualification-v1",
        "passed": True,
        "steps": args.steps,
        "warmup_runs": 0,
        "measured_runs": 1,
        "runtime_slots": {str(slot): count for slot, count in sorted(slots.items())},
        "final_slot_generations": {"0": generations[0], "1": generations[1]},
        "prepare_only_after_first": True,
        "device_runner_serial": True,
        "full_128_output_token_ids_sha256": full_output_sha,
        "rts_completion_interval_ms": trace["official_metrics"][
            "rts_completion_interval_ms"
        ]["pooled_steady_stats"],
        "runner_run_ms": trace["official_metrics"]["runner_run_ms"][
            "pooled_steady_stats"
        ],
        "effective_ms": effective,
        "effective_over_50ms_count": sum(
            float(row["effective_ms"]) > 50.0 for row in rows[steady_skip:]
        ),
        "effective_over_70ms_count": sum(
            float(row["effective_ms"]) > 70.0 for row in rows[steady_skip:]
        ),
    }
    output = args.result / "dual_validation.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
