# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for torch backend autoload state in bind-phase reports."""

from __future__ import annotations

import sys

from simpler_setup.tools import hbg_bind_phases


def _write_log(path, autoload_records: tuple[str, ...]) -> None:
    lines = ["[stamp] command commit=abc"]
    lines.extend(f"TIMING simpler: {record}" for record in autoload_records)
    lines.extend(
        [
            "bind phase=host_orch start_ns=1 dur_ns=1000000",
            "bind phase=arena_h2d start_ns=2 dur_ns=1000000",
            "bind phase=host_orch start_ns=3 dur_ns=500000",
            "bind phase=arena_h2d start_ns=4 dur_ns=500000",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_report_shows_each_torch_autoload_state(tmp_path, monkeypatch, capsys):
    log = tmp_path / "run.log"
    first = (
        'torch_backend_autoload setting=invalid raw="a b\\"c" raw_truncated=false '
        "effective=disabled torch_imported=true torch_npu_loaded=false"
    )
    second = (
        'torch_backend_autoload setting=1 raw="1" raw_truncated=false '
        "effective=enabled torch_imported=true torch_npu_loaded=true"
    )
    _write_log(log, (first, first, second))
    monkeypatch.setattr(sys, "argv", ["hbg_bind_phases", str(log), "--rounds", "2"])

    assert hbg_bind_phases.main() == 0

    output = capsys.readouterr().out
    assert output.count(first) == 1
    assert output.count(second) == 1


def test_report_warns_when_torch_autoload_state_is_missing(tmp_path, monkeypatch, capsys):
    log = tmp_path / "run.log"
    _write_log(log, ())
    monkeypatch.setattr(sys, "argv", ["hbg_bind_phases", str(log), "--rounds", "2"])

    assert hbg_bind_phases.main() == 0

    output = capsys.readouterr().out
    assert "no `torch_backend_autoload` record" in output
    assert "must be established before comparing" in output


def test_parser_accepts_record_without_raw_fields(tmp_path):
    log = tmp_path / "run.log"
    record = "torch_backend_autoload setting=0 effective=disabled torch_imported=true torch_npu_loaded=false"
    _write_log(log, (record,))

    assert hbg_bind_phases.parse_torch_autoload(str(log)) == [record]
