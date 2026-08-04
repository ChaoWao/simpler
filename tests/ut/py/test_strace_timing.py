#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from simpler_setup.tools.strace_timing import parse_spans


def _marker(pid, inv, name, attrs=""):
    return f"[STRACE] v=1 pid={pid} tid={pid} inv={inv} hid=abc depth=0 name={name} ts=100 dur=20 {attrs}"


def test_parse_spans_finds_adjacent_records_on_one_physical_line():
    line = (
        "[2026-08-04 10:00:00.000001][T0x1][TIMING] emit: "
        + _marker(11, 1, "simpler_run", "rank=0")
        + "[2026-08-04 10:00:00.000002][T0x2][TIMING] emit: "
        + _marker(22, 2, "simpler_run.runner_run.device_wall", "clk=dev rank=1")
        + "\n"
    )

    spans = list(parse_spans([line]))

    assert [(span.pid, span.inv, span.name) for span in spans] == [
        (11, 1, "simpler_run"),
        (22, 2, "simpler_run.runner_run.device_wall"),
    ]
    assert spans[0].attrs == "rank=0"
    assert spans[1].attrs == "clk=dev rank=1"
