# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for simpler's single-axis log-level configuration."""

import logging

import pytest
from simpler._log import TIMING, _normalize_threshold

from simpler_setup.log_config import (
    DEFAULT_LOG_LEVEL,
    LOG_LEVEL_CHOICES,
    configure_logging,
    parse_level,
)


def test_default_level_is_timing():
    assert DEFAULT_LOG_LEVEL == "timing"
    assert TIMING == 25
    assert parse_level("timing") == TIMING
    assert parse_level("info") == 20


def test_choices_are_single_axis():
    assert LOG_LEVEL_CHOICES == ["debug", "info", "timing", "warn", "error", "null"]
    for v in range(10):
        assert f"v{v}" not in LOG_LEVEL_CHOICES


def test_null_mutes_severity():
    configure_logging("null")
    sim = logging.getLogger("simpler")
    assert sim.getEffectiveLevel() >= 60


def test_timing_hides_info_but_keeps_timing():
    configure_logging("timing")
    sim = logging.getLogger("simpler")
    assert not sim.isEnabledFor(logging.INFO)
    assert sim.isEnabledFor(TIMING)


@pytest.mark.parametrize(
    "threshold, expected",
    [
        (10, 10),
        (11, 20),
        (20, 20),
        (21, 25),
        (25, 25),
        (26, 30),
        (30, 30),
        (31, 40),
        (40, 40),
        (41, 60),
        (60, 60),
    ],
)
def test_normalize_threshold(threshold, expected):
    assert _normalize_threshold(threshold) == expected
