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


def test_alignment_selects_minimum_rtt_and_offsets_at_the_nominal_frequency():
    anchors = _anchors(
        [
            _sample("pre_host_orchestration", 0, 900, 100, 1_100),
            _sample("pre_host_orchestration", 1, 990, 100, 1_010),
            _sample("post_device_execution", 0, 4_900, 4_100, 5_300),
            _sample("post_device_execution", 1, 5_080, 4_100, 5_120),
        ]
    )

    alignment = build_clock_alignment(anchors, 1_000_000_000)

    assert alignment.status == "calibrated"
    assert alignment.metadata()["selected_sample_idx"] == {"pre_host_orchestration": 1}
    assert alignment.max_uncertainty_ns == 10
    assert alignment.metadata()["anchor_group_duration_ns"] == {"pre_host_orchestration": 200}
    # A closing anchor from an older capture is ignored.
    assert alignment.map_cycles_to_host_ns(100) == 1_000
    assert alignment.map_cycles_to_host_ns(2_100) == 3_000
    assert alignment.map_cycles_to_host_ns(4_100) == 5_000


def test_mapping_needs_only_one_anchor():
    anchors = _anchors([_sample("pre_host_orchestration", 0, 990, 100, 1_010)])

    alignment = build_clock_alignment(anchors, 1_000_000_000)

    assert alignment.status == "calibrated"
    assert alignment.map_cycles_to_host_ns(9_100) == 10_000


def test_the_anchor_is_a_pin_and_extrapolates_equally_either_side():
    # The single reading supplies an offset, not the bound of a calibrated
    # interval, so a cycle count below the anchor maps by the same arithmetic
    # as one above it. host_mid_ns = 999_900 + 200 // 2.
    anchors = _anchors([_sample("pre_host_orchestration", 0, 999_900, 5_000, 1_000_100)])

    alignment = build_clock_alignment(anchors, 1_000_000_000)

    assert alignment.map_cycles_to_host_ns(5_000) == 1_000_000
    assert alignment.map_cycles_to_host_ns(6_500) == 1_001_500
    assert alignment.map_cycles_to_host_ns(3_500) == 998_500


def test_mapping_rounds_toward_zero_on_both_sides_of_the_anchor():
    # 3 GHz: one cycle is 1/3 ns, so a 5-cycle delta is 1.67 ns and has to be
    # truncated. Truncation is toward zero, which keeps the two directions
    # symmetric; floor division alone would bias the pre-anchor side away.
    anchors = _anchors([_sample("pre_host_orchestration", 0, 999_900, 5_000, 1_000_100)])

    alignment = build_clock_alignment(anchors, 3_000_000_000)

    assert alignment.map_cycles_to_host_ns(5_005) == 1_000_001
    assert alignment.map_cycles_to_host_ns(4_995) == 999_999


def test_alignment_preserves_low_bits_for_large_cycle_values():
    ref = 10**18
    anchors = _anchors([_sample("pre_host_orchestration", 0, 9_999_999_990, ref, 10_000_000_010)])

    alignment = build_clock_alignment(anchors, 1_000_000_000)

    assert alignment.map_cycles_to_host_ns(ref + 1) == 10_000_000_001


def test_acl_event_microsecond_quantization_is_included_in_uncertainty():
    anchors = _anchors(
        [_sample("pre_host_orchestration", 0, 990, 100, 1_010)],
        raw_unit="device_uptime_us",
    )

    alignment = build_clock_alignment(anchors, 1_000_000_000)

    assert alignment.max_uncertainty_ns == 1_010


def test_host_record_quantization_is_included_in_cross_domain_uncertainty():
    anchors = _anchors([_sample("pre_host_orchestration", 0, 990, 100, 1_010)])

    alignment = build_clock_alignment(
        anchors,
        50_000_000,
        host_timestamp_quantization_ns=20,
    )

    assert alignment.anchor_uncertainty_ns == 10
    assert alignment.max_uncertainty_ns == 30
    assert alignment.metadata()["host_timestamp_quantization_ns"] == 20


@pytest.mark.parametrize(
    ("anchors", "reason"),
    [
        ({}, "missing_clock_anchor_samples"),
        (
            _anchors(
                [
                    _sample("pre_host_orchestration", 0, 990, 100, 1_010, {"stage": "record", "code": 1}),
                    _sample("post_device_execution", 0, 4_990, 4_100, 5_010),
                ]
            ),
            "no_valid_anchor",
        ),
    ],
)
def test_alignment_fails_closed(anchors, reason):
    alignment = build_clock_alignment(anchors, 1_000_000_000)

    assert alignment.status == "unaligned"
    assert alignment.reason == reason
    assert alignment.max_uncertainty_ns is None
    with pytest.raises(ValueError):
        alignment.map_cycles_to_host_ns(100)
