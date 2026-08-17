# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cross-runtime assertions for level-4 chip-swimlane scene tests."""

import json
from pathlib import Path

from simpler_setup.scene_test import _match_selectors, _outputs_dir, _parse_case_selector, _sanitize_for_filename
from simpler_setup.tools.swimlane_converter import read_perf_data


def scene_case_selected(request, class_name: str, case_name: str) -> bool:
    """Return whether the current ``--case`` selectors include one scene case."""
    raw_selectors = request.config.getoption("--case", default=None) or []
    selectors = [_parse_case_selector(value) for value in raw_selectors]
    return _match_selectors(class_name, case_name, selectors)


def locate_swimlane_artifact(case_label: str, *, since: float) -> Path:
    """Find the profiling artifact created for one case after ``since``."""
    safe_label = _sanitize_for_filename(case_label)
    matches = [path for path in _outputs_dir().glob(f"{safe_label}_*") if path.stat().st_mtime >= since]
    assert matches, f"no output dir for {safe_label!r} created this run — swimlane capture failed?"
    out_dir = max(matches, key=lambda path: path.stat().st_mtime)
    artifact = out_dir / "chip_swimlane_records.json"
    assert artifact.exists(), f"chip_swimlane_records.json missing under {out_dir} — swimlane capture failed?"
    return artifact


def validate_host_orchestrator_capture(case_label: str, *, since: float, expected_records: int) -> None:
    """Validate HBG host capture, sim clock anchors, and reader alignment."""
    artifact = locate_swimlane_artifact(case_label, since=since)
    with artifact.open() as stream:
        raw = json.load(stream)

    metadata = raw.get("metadata")
    assert isinstance(metadata, dict), f"metadata missing or not an object under {artifact}"
    assert metadata.get("orchestrator_source") == "host", f"host orchestrator source missing under {artifact}"

    capture = metadata.get("host_capture")
    assert isinstance(capture, dict), f"host_capture missing or not an object under {artifact}"
    host_records = sum(len(lane) for lane in raw.get("host_orchestrator_phases", []))
    assert capture.get("status") == "complete", f"incomplete host capture under {artifact}: {capture}"
    assert capture.get("expected_records") == expected_records, (
        f"unexpected expected_records under {artifact}: {capture}"
    )
    assert capture.get("recorded_records") == expected_records, (
        f"unexpected recorded_records under {artifact}: {capture}"
    )
    assert capture.get("dropped_records") == 0, f"host capture dropped records under {artifact}: {capture}"
    assert capture.get("error") is None, f"host capture reported an error under {artifact}: {capture}"
    assert host_records == expected_records, (
        f"got {host_records} host records, expected {expected_records} under {artifact}"
    )

    anchors = metadata.get("clock_anchors")
    assert isinstance(anchors, dict), f"clock_anchors missing or not an object under {artifact}"
    assert anchors.get("provider") == "sim_syscnt", f"unexpected clock provider under {artifact}: {anchors}"
    assert anchors.get("samples_per_position") == 3, f"unexpected anchor group size under {artifact}: {anchors}"
    samples = anchors.get("samples")
    assert isinstance(samples, list), f"clock anchor samples missing under {artifact}"
    for position in ("pre_host_orchestration", "post_device_execution"):
        position_samples = [sample for sample in samples if sample.get("position") == position]
        assert len(position_samples) == 3, f"expected three {position} anchors under {artifact}: {samples}"
        assert all(sample.get("error") is None for sample in position_samples), (
            f"invalid {position} anchor under {artifact}: {position_samples}"
        )

    data = read_perf_data(artifact)
    assert data.get("orchestrator_source") == "host", f"reader lost host orchestrator source under {artifact}"
    assert "aicpu_orchestrator_phases" not in data, f"reader mislabeled host records as AICPU under {artifact}"
    reader_records = sum(len(lane) for lane in data.get("host_orchestrator_phases", []))
    assert reader_records == expected_records, (
        f"reader returned {reader_records} host records, expected {expected_records} under {artifact}"
    )
    timeline = data.get("timeline_metadata")
    assert isinstance(timeline, dict), f"timeline_metadata missing or not an object under {artifact}"
    assert timeline.get("layout") == "clock_aligned", (
        f"host/device clocks were not aligned under {artifact}: {timeline}"
    )
    assert timeline.get("trace_status") == "complete", f"reader reported a partial trace under {artifact}: {timeline}"
    assert timeline.get("host_records_complete") is True, f"reader rejected host records under {artifact}: {timeline}"
    assert timeline.get("cross_domain_latency_available") is True, (
        f"cross-domain latency unavailable under {artifact}: {timeline}"
    )


def validate_aicpu_orchestrator_capture(data: dict, artifact: Path) -> None:
    """Validate the TMR device-orchestrator stream at level 4."""
    records = data.get("aicpu_orchestrator_phases")
    assert isinstance(records, list) and any(records), f"level-4 AICPU orchestrator phases missing under {artifact}"
    assert any(record for lane in records for record in lane), (
        f"level-4 AICPU orchestrator records empty under {artifact}"
    )
    assert data.get("orchestrator_source") == "aicpu", f"unexpected orchestrator source under {artifact}: {data}"
    assert "timeline_metadata" not in data, (
        f"AICPU-only capture unexpectedly carries cross-domain metadata under {artifact}"
    )
