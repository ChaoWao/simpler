#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from simpler_setup.tools.clock_correlation import build_clock_alignment


def _sample(position, sample_idx, before, cycles, after, error=None):
    return {
        "position": position,
        "sample_idx": sample_idx,
        "host_before_ns": before,
        "device_cycles": cycles,
        "host_after_ns": after,
        "error": error,
    }


def _anchors(samples, raw_unit="syscnt_cycles"):
    return {
        "device_timestamp_unit": "syscnt_cycles",
        "raw_device_timestamp_unit": raw_unit,
        "samples": samples,
    }


def test_alignment_selects_minimum_rtt_and_interpolates_offset_with_integer_math():
    anchors = _anchors(
        [
            _sample("pre_host_orchestration", 0, 900, 100, 1_100),
            _sample("pre_host_orchestration", 1, 990, 100, 1_010),
            _sample("post_device_execution", 0, 4_900, 4_100, 5_300),
            _sample("post_device_execution", 1, 5_080, 4_100, 5_120),
        ]
    )

    alignment = build_clock_alignment(anchors, 1_000_000_000, [100, 2_100, 4_100])

    assert alignment.status == "calibrated"
    assert alignment.metadata()["selected_sample_idx"] == {
        "pre_host_orchestration": 1,
        "post_device_execution": 1,
    }
    assert alignment.max_uncertainty_ns == 20
    assert alignment.map_cycles_to_host_ns(100) == 1_000
    assert alignment.map_cycles_to_host_ns(2_100) == 3_050
    assert alignment.map_cycles_to_host_ns(4_100) == 5_100


def test_alignment_preserves_low_bits_for_large_cycle_values():
    ref = 10**18
    anchors = _anchors(
        [
            _sample("pre_host_orchestration", 0, 9_999_999_990, ref, 10_000_000_010),
            _sample("post_device_execution", 0, 10_000_003_990, ref + 4_000, 10_000_004_010),
        ]
    )

    alignment = build_clock_alignment(anchors, 1_000_000_000, [ref + 1])

    assert alignment.map_cycles_to_host_ns(ref + 1) == 10_000_000_001


def test_acl_event_microsecond_quantization_is_included_in_uncertainty():
    anchors = _anchors(
        [
            _sample("pre_host_orchestration", 0, 990, 100, 1_010),
            _sample("post_device_execution", 0, 4_990, 4_100, 5_010),
        ],
        raw_unit="device_uptime_us",
    )

    alignment = build_clock_alignment(anchors, 1_000_000_000, [100, 4_100])

    assert alignment.max_uncertainty_ns == 1_010


def test_host_record_quantization_is_included_in_cross_domain_uncertainty():
    anchors = _anchors(
        [
            _sample("pre_host_orchestration", 0, 990, 100, 1_010),
            _sample("post_device_execution", 0, 4_980, 4_100, 5_020),
        ]
    )

    alignment = build_clock_alignment(
        anchors,
        50_000_000,
        [100, 4_100],
        host_timestamp_quantization_ns=20,
    )

    assert alignment.anchor_uncertainty_ns == 20
    assert alignment.max_uncertainty_ns == 40
    assert alignment.metadata()["host_timestamp_quantization_ns"] == 20


@pytest.mark.parametrize(
    ("anchors", "timestamps", "reason"),
    [
        ({}, [], "missing_clock_anchor_samples"),
        (
            _anchors(
                [
                    _sample("pre_host_orchestration", 0, 990, 100, 1_010, {"stage": "record", "code": 1}),
                    _sample("post_device_execution", 0, 4_990, 4_100, 5_010),
                ]
            ),
            [],
            "no_valid_pre_host_orchestration_anchor",
        ),
        (
            _anchors(
                [
                    _sample("pre_host_orchestration", 0, 990, 100, 1_010),
                    _sample("post_device_execution", 0, 4_990, 4_100, 5_010),
                ]
            ),
            [99],
            "device_timestamps_outside_anchor_coverage",
        ),
    ],
)
def test_alignment_fails_closed(anchors, timestamps, reason):
    alignment = build_clock_alignment(anchors, 1_000_000_000, timestamps)

    assert alignment.status == "unaligned"
    assert alignment.reason == reason
    assert alignment.max_uncertainty_ns is None
    with pytest.raises(ValueError):
        alignment.map_cycles_to_host_ns(100)
