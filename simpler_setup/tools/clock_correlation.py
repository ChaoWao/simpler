#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Map device timestamps onto the Host CLOCK_MONOTONIC axis.

The platform counter frequency supplies the scale. One paired Host/Device
reading supplies the offset, and the same mapping applies on either side of
that reading.
"""

from dataclasses import dataclass
from typing import Optional

_ANCHOR_POSITION = "pre_host_orchestration"
_METHOD = "nominal_frequency_offset_v1"


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
    anchor: Optional[ClockAnchor] = None
    anchor_group_duration_ns: Optional[int] = None

    @property
    def anchor_uncertainty_ns(self):
        if self.anchor is None:
            return None
        return self.anchor.uncertainty_ns

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
        if self.anchor is not None:
            out["selected_sample_idx"] = {self.anchor.position: self.anchor.sample_idx}
        if self.anchor_group_duration_ns is not None:
            out["anchor_group_duration_ns"] = {_ANCHOR_POSITION: self.anchor_group_duration_ns}
        return out

    def map_cycles_to_host_ns(self, device_cycles):
        if self.anchor is None:
            raise ValueError(f"device timestamp {device_cycles} has no calibration anchor")
        delta_cycles = device_cycles - self.anchor.device_cycles
        return self.anchor.host_mid_ns + _mul_div_toward_zero(delta_cycles, 1_000_000_000, self.frequency_hz)


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


def _anchor_group_duration_ns(samples):
    bounds = []
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("position") != _ANCHOR_POSITION:
            continue
        try:
            before = int(sample["host_before_ns"])
            after = int(sample["host_after_ns"])
        except (KeyError, TypeError, ValueError):
            continue
        if before > 0 and after >= before:
            bounds.append((before, after))
    if not bounds:
        return None
    return max(after for _, after in bounds) - min(before for before, _ in bounds)


def _resolve_samples(clock_anchors, frequency_hz, host_timestamp_quantization_ns):
    """Return `(samples, device_quantization_ns, reason)`; `reason` names the refusal."""
    if host_timestamp_quantization_ns < 0:
        return None, 0, "invalid_host_timestamp_quantization"
    if frequency_hz <= 0:
        return None, 0, "invalid_device_frequency"
    if not isinstance(clock_anchors, dict):
        return None, 0, "missing_clock_anchors"
    samples = clock_anchors.get("samples")
    if not isinstance(samples, list):
        return None, 0, "missing_clock_anchor_samples"
    if clock_anchors.get("device_timestamp_unit") != "syscnt_cycles":
        return None, 0, "unsupported_device_timestamp_unit"
    raw_unit = clock_anchors.get("raw_device_timestamp_unit", "syscnt_cycles")
    if raw_unit == "device_uptime_us":
        # aclrtEventGetTimestamp has microsecond resolution. Normalizing it to
        # cycles does not recover the discarded sub-microsecond portion.
        return samples, 1_000, None
    if raw_unit == "syscnt_cycles":
        return samples, 0, None
    return None, 0, "unsupported_raw_device_timestamp_unit"


def build_clock_alignment(clock_anchors, frequency_hz, host_timestamp_quantization_ns=0):
    host_timestamp_quantization_ns = int(host_timestamp_quantization_ns)
    samples, device_quantization_ns, reason = _resolve_samples(
        clock_anchors, frequency_hz, host_timestamp_quantization_ns
    )
    if samples is None:
        # A negative host quantization is not a usable value to report back.
        return _unaligned(frequency_hz, reason, max(host_timestamp_quantization_ns, 0))

    candidates = []
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("position") != _ANCHOR_POSITION:
            continue
        parsed = _parse_anchor(sample, _ANCHOR_POSITION, device_quantization_ns)
        if parsed is not None:
            candidates.append(parsed)

    if not candidates:
        return _unaligned(frequency_hz, "no_valid_anchor", host_timestamp_quantization_ns)

    # A wider bracket means a less certain instant, so the narrowest round trip
    # is the sharpest reading of the pair; sample_idx breaks ties reproducibly.
    anchor = min(candidates, key=lambda candidate: (candidate.rtt_ns, candidate.sample_idx))

    return ClockAlignment(
        status="calibrated",
        reason=None,
        frequency_hz=frequency_hz,
        host_timestamp_quantization_ns=host_timestamp_quantization_ns,
        anchor=anchor,
        anchor_group_duration_ns=_anchor_group_duration_ns(samples),
    )
