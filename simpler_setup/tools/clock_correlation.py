#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional

_A0_POSITION = "pre_host_orchestration"
_A2_POSITION = "post_device_execution"
_METHOD = "nominal_frequency_offset_interp_v1"


@dataclass(frozen=True)
class ClockAnchor:
    position: str
    sample_idx: int
    host_mid_ns: int
    device_cycles: int
    rtt_ns: int
    uncertainty_ns: int


@dataclass(frozen=True)
class ClockAlignment:
    status: str
    reason: Optional[str]
    frequency_hz: int
    host_timestamp_quantization_ns: int = 0
    start: Optional[ClockAnchor] = None
    end: Optional[ClockAnchor] = None

    @property
    def anchor_uncertainty_ns(self):
        if self.start is None or self.end is None:
            return None
        return max(self.start.uncertainty_ns, self.end.uncertainty_ns)

    @property
    def max_uncertainty_ns(self):
        if self.anchor_uncertainty_ns is None:
            return None
        return self.anchor_uncertainty_ns + self.host_timestamp_quantization_ns

    def metadata(self):
        out = {
            "status": self.status,
            "method": _METHOD,
            "anchor_uncertainty_ns": self.anchor_uncertainty_ns,
            "host_timestamp_quantization_ns": self.host_timestamp_quantization_ns,
            "max_uncertainty_ns": self.max_uncertainty_ns,
        }
        if self.reason is not None:
            out["reason"] = self.reason
        if self.start is not None and self.end is not None:
            out["selected_sample_idx"] = {
                self.start.position: self.start.sample_idx,
                self.end.position: self.end.sample_idx,
            }
        return out

    def contains(self, device_cycles):
        return (
            self.status == "calibrated"
            and self.start is not None
            and self.end is not None
            and self.start.device_cycles <= device_cycles <= self.end.device_cycles
        )

    def map_cycles_to_host_ns(self, device_cycles):
        if self.start is None or self.end is None or not self.contains(device_cycles):
            raise ValueError(f"device timestamp {device_cycles} is outside calibrated anchor coverage")

        start = self.start
        end = self.end
        delta_cycles = device_cycles - start.device_cycles
        span_cycles = end.device_cycles - start.device_cycles
        nominal_delta_ns = _mul_div_toward_zero(delta_cycles, 1_000_000_000, self.frequency_hz)
        nominal_end_ns = start.host_mid_ns + _mul_div_toward_zero(span_cycles, 1_000_000_000, self.frequency_hz)
        end_correction_ns = end.host_mid_ns - nominal_end_ns
        correction_ns = _mul_div_toward_zero(end_correction_ns, delta_cycles, span_cycles)
        return start.host_mid_ns + nominal_delta_ns + correction_ns


def _mul_div_toward_zero(value, multiplier, divisor):
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    product = value * multiplier
    if product >= 0:
        return product // divisor
    return -((-product) // divisor)


def _unaligned(frequency_hz, reason, host_timestamp_quantization_ns=0):
    return ClockAlignment(
        status="unaligned",
        reason=reason,
        frequency_hz=frequency_hz,
        host_timestamp_quantization_ns=host_timestamp_quantization_ns,
    )


def _parse_anchor(sample, position, device_quantization_ns):
    error = sample.get("error")
    if error not in (None, 0, ""):
        return None
    try:
        host_before_ns = int(sample["host_before_ns"])
        device_cycles = int(sample["device_cycles"])
        host_after_ns = int(sample["host_after_ns"])
        sample_idx = int(sample["sample_idx"])
    except (KeyError, TypeError, ValueError):
        return None
    if host_before_ns <= 0 or device_cycles <= 0 or host_after_ns < host_before_ns or sample_idx < 0:
        return None
    rtt_ns = host_after_ns - host_before_ns
    return ClockAnchor(
        position=position,
        sample_idx=sample_idx,
        host_mid_ns=host_before_ns + rtt_ns // 2,
        device_cycles=device_cycles,
        rtt_ns=rtt_ns,
        uncertainty_ns=(rtt_ns + 1) // 2 + device_quantization_ns,
    )


def build_clock_alignment(  # noqa: PLR0912
    clock_anchors, frequency_hz, device_timestamps=(), host_timestamp_quantization_ns=0
):
    host_timestamp_quantization_ns = int(host_timestamp_quantization_ns)
    if host_timestamp_quantization_ns < 0:
        return _unaligned(frequency_hz, "invalid_host_timestamp_quantization")
    if frequency_hz <= 0:
        return _unaligned(frequency_hz, "invalid_device_frequency", host_timestamp_quantization_ns)
    if not isinstance(clock_anchors, dict):
        return _unaligned(frequency_hz, "missing_clock_anchors", host_timestamp_quantization_ns)
    samples = clock_anchors.get("samples")
    if not isinstance(samples, list):
        return _unaligned(frequency_hz, "missing_clock_anchor_samples", host_timestamp_quantization_ns)
    if clock_anchors.get("device_timestamp_unit") != "syscnt_cycles":
        return _unaligned(frequency_hz, "unsupported_device_timestamp_unit", host_timestamp_quantization_ns)
    raw_unit = clock_anchors.get("raw_device_timestamp_unit", "syscnt_cycles")
    if raw_unit == "device_uptime_us":
        # aclrtEventGetTimestamp has microsecond resolution. Normalizing it to
        # cycles does not recover the discarded sub-microsecond portion.
        device_quantization_ns = 1_000
    elif raw_unit == "syscnt_cycles":
        device_quantization_ns = 0
    else:
        return _unaligned(frequency_hz, "unsupported_raw_device_timestamp_unit", host_timestamp_quantization_ns)

    valid_by_position = {_A0_POSITION: [], _A2_POSITION: []}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        position = sample.get("position")
        if position not in valid_by_position:
            continue
        anchor = _parse_anchor(sample, position, device_quantization_ns)
        if anchor is not None:
            valid_by_position[position].append(anchor)

    if not valid_by_position[_A0_POSITION]:
        return _unaligned(frequency_hz, "no_valid_pre_host_orchestration_anchor", host_timestamp_quantization_ns)
    if not valid_by_position[_A2_POSITION]:
        return _unaligned(frequency_hz, "no_valid_post_device_execution_anchor", host_timestamp_quantization_ns)

    start = min(valid_by_position[_A0_POSITION], key=lambda anchor: (anchor.rtt_ns, anchor.sample_idx))
    end = min(valid_by_position[_A2_POSITION], key=lambda anchor: (anchor.rtt_ns, anchor.sample_idx))
    if end.host_mid_ns <= start.host_mid_ns:
        return _unaligned(frequency_hz, "host_anchor_order_invalid", host_timestamp_quantization_ns)
    if end.device_cycles <= start.device_cycles:
        return _unaligned(frequency_hz, "device_anchor_order_invalid", host_timestamp_quantization_ns)

    alignment = ClockAlignment(
        status="calibrated",
        reason=None,
        frequency_hz=frequency_hz,
        host_timestamp_quantization_ns=host_timestamp_quantization_ns,
        start=start,
        end=end,
    )
    for timestamp in device_timestamps:
        timestamp = int(timestamp)
        if timestamp > 0 and not alignment.contains(timestamp):
            return _unaligned(
                frequency_hz,
                "device_timestamps_outside_anchor_coverage",
                host_timestamp_quantization_ns,
            )
    return alignment
