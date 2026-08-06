#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from simpler_setup.tools.strace_timing import count_record_heads, parse_spans


def _record(pid, inv, name, attrs=""):
    """One host-log record in the shape `HostLogger::emit` writes it.

    `LOG_TIMING` prepends `[<file>:<line>] ` to the caller's format string, so
    the marker never sits flush against the `<func>: ` separator on stderr.
    """
    return (
        f"[2026-08-04 10:00:00.00000{pid}][T0x{pid}][TIMING] emit_span: [strace.h:132] "
        f"[STRACE] v=1 pid={pid} tid={pid} inv={inv} hid=abc depth=0 name={name} ts=100 dur=20 {attrs}"
    )


def test_parse_spans_finds_adjacent_records_on_one_physical_line():
    line = (
        _record(1, 1, "simpler_run", "rank=0")
        + _record(2, 2, "simpler_run.runner_run.device_wall", "clk=dev rank=1")
        + "\n"
    )

    spans = list(parse_spans([line]))

    assert [(span.pid, span.inv, span.name) for span in spans] == [
        (1, 1, "simpler_run"),
        (2, 2, "simpler_run.runner_run.device_wall"),
    ]
    assert spans[0].attrs == "rank=0"
    assert spans[1].attrs == "clk=dev rank=1"


def test_parse_spans_keeps_every_record_of_a_multi_line_blob():
    blob = _record(1, 1, "simpler_run", "rank=0") + "\n" + _record(2, 2, "simpler_run.bind", "rank=1") + "\n"

    spans = list(parse_spans([blob]))

    assert [(span.pid, span.inv, span.name) for span in spans] == [
        (1, 1, "simpler_run"),
        (2, 2, "simpler_run.bind"),
    ]
    assert spans[0].attrs == "rank=0"
    assert spans[1].attrs == "rank=1"


def test_count_record_heads_sees_a_torn_record_that_parse_spans_drops():
    intact = _record(1, 1, "simpler_run", "rank=0")
    torn = intact[: intact.index(" ts=")]
    lines = [intact + "\n", torn + "\n"]

    assert count_record_heads(lines) == 2
    assert len(list(parse_spans(lines))) == 1
