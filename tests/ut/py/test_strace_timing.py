#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
from collections import defaultdict
from io import StringIO

import pytest

from simpler_setup.tools.strace_timing import (
    NativeOverlapError,
    assert_native_overlap,
    bucket_by_hid,
    count_record_heads,
    group_invocations,
    host_record_spans,
    legacy_spans,
    load_host_phase_records,
    main,
    parse_clock_anchors,
    parse_spans,
    print_rounds_table,
    to_chrome_trace,
    to_host_swimlane,
)


def _record(pid, inv, name, attrs="", *, depth=0, ts=100, dur=20):
    """One current host-log record in the shape `HostLogger::emit` writes it."""
    return (
        f"[mono_ns={1_000_000 + pid}][T0x{pid}][TIMING] emit_host_span: "
        f"[STRACE] v=1 pid={pid} tid={pid} inv={inv} hid=abc depth={depth} "
        f"name={name} ts={ts} dur={dur} {attrs}"
    )


def _legacy_record(pid, inv, name, attrs=""):
    return (
        f"[2026-08-04 10:00:00.00000{pid}][T0x{pid}][TIMING] emit_span: [strace.h:132] "
        f"[STRACE] v=1 pid={pid} tid={pid} inv={inv} hid=abc depth=0 name={name} ts=100 dur=20 {attrs}"
    )


def _anchor_record(pid, mono_ns, wall_ns):
    return (
        f"[mono_ns={mono_ns}][T0x{pid}][TIMING] clock_anchor: "
        f"[CLOCK_ANCHOR] v=1 pid={pid} mono_ns={mono_ns} wall_ns={wall_ns}\n"
    )


def _span_record(
    *,
    pid: int,
    tid: int,
    inv: int,
    name: str,
    ts: int,
    dur: int,
    attrs: str = "",
    hid: str = "abc",
    depth: int = 0,
) -> str:
    return f"[STRACE] v=1 pid={pid} tid={tid} inv={inv} hid={hid} depth={depth} name={name} ts={ts} dur={dur} {attrs}\n"


def test_parse_spans_finds_adjacent_records_on_one_physical_line():
    line = (
        _record(1, 1, "chip.run", "rank=0") + _record(2, 2, "chip.run.runner_run.device_wall", "clk=dev rank=1") + "\n"
    )

    spans = list(parse_spans([line]))

    assert [(span.pid, span.inv, span.name) for span in spans] == [
        (1, 1, "chip.run"),
        (2, 2, "chip.run.runner_run.device_wall"),
    ]
    assert spans[0].attrs == "rank=0"
    assert spans[1].attrs == "clk=dev rank=1"


def test_parse_spans_keeps_every_record_of_a_multi_line_blob():
    blob = _legacy_record(1, 1, "chip.run", "rank=0") + "\n" + _record(2, 2, "chip.run.bind", "rank=1") + "\n"

    spans = list(parse_spans([blob]))

    assert [(span.pid, span.inv, span.name) for span in spans] == [
        (1, 1, "chip.run"),
        (2, 2, "chip.run.bind"),
    ]
    assert spans[0].attrs == "rank=0"
    assert spans[1].attrs == "rank=1"


def test_parse_spans_preserves_64_bit_invocation_id():
    invocation_id = 2**32 + 7

    spans = list(parse_spans([_record(41, invocation_id, "chip.run")]))

    assert len(spans) == 1
    assert spans[0].inv == invocation_id


def test_parse_clock_anchor_maps_monotonic_to_wall_time():
    lines = [
        "worker-3: [mono_ns=1005][T0x1][TIMING] clock_anchor: "
        "[CLOCK_ANCHOR] v=1 pid=41 mono_ns=1000 wall_ns=1700000000000000000\n"
    ]

    anchors = list(parse_clock_anchors(lines))

    assert len(anchors) == 1
    assert anchors[0].pid == 41
    assert anchors[0].mono_ns == 1000
    assert anchors[0].wall_ns == 1_700_000_000_000_000_000
    assert anchors[0].to_wall_ns(1250) == 1_700_000_000_000_000_250


def test_parse_clock_anchor_rejects_payloads_outside_complete_anchor_records():
    payload = "[CLOCK_ANCHOR] v=1 pid=41 mono_ns=1000 wall_ns=1700000000000000000"
    lines = [
        payload + "\n",
        f"[mono_ns=1005][T0x1][INFO] message: copied {payload}\n",
        f"[mono_ns=1005][T0x1][TIMING] clock_anchor: {payload} trailing\n",
    ]

    assert list(parse_clock_anchors(lines)) == []


def test_trace_renderers_add_wall_time_without_replacing_monotonic_timestamps():
    wall_ns = 1_700_000_000_000_000_000
    anchors = list(parse_clock_anchors([_anchor_record(41, 1_000, wall_ns)]))
    spans = list(
        parse_spans(
            [
                _span_record(pid=41, tid=410, inv=7, name="chip.run", ts=1_250, dur=20),
                _span_record(
                    pid=41,
                    tid=410,
                    inv=7,
                    name="chip.run.runner_run.device_wall",
                    ts=300,
                    dur=40,
                    attrs="clk=dev",
                    depth=1,
                ),
            ]
        )
    )

    invocations = group_invocations(spans)
    chrome_trace = to_chrome_trace(invocations, bucket_by_hid(invocations), anchors=anchors)
    host_swimlane = to_host_swimlane(spans, anchors=anchors)

    for trace in (chrome_trace, host_swimlane):
        host_event = next(event for event in trace["traceEvents"] if event.get("name") == "chip.run")
        assert host_event["ts"] == 1.25
        assert host_event["args"]["wall_ts_ns"] == "1700000000000000250"
        assert host_event["args"]["wall_time"] == "2023-11-14T22:13:20.000000250Z"
        assert trace["clockAnchors"] == [{"pid": 41, "mono_ns": "1000", "wall_ns": "1700000000000000000"}]

    device_event = next(event for event in chrome_trace["traceEvents"] if event.get("name", "").endswith("device_wall"))
    assert "wall_ts_ns" not in device_event["args"]
    assert "wall_time" not in device_event["args"]


def test_wall_time_uses_the_matching_anchor_for_each_pid():
    anchors = list(
        parse_clock_anchors(
            [
                _anchor_record(41, 1_000, 1_700_000_000_000_000_000),
                _anchor_record(52, 1_000, 1_900_000_000_000_000_000),
                _anchor_record(64, 2_000, 2_000_000_000_000_000_000),
            ]
        )
    )
    spans = list(
        parse_spans(
            [
                _span_record(pid=41, tid=410, inv=1, name="host.submit", ts=1_500, dur=10),
                _span_record(pid=52, tid=520, inv=1, name="host.submit", ts=1_500, dur=10),
                _span_record(pid=63, tid=630, inv=1, name="host.submit", ts=1_500, dur=10),
                _span_record(pid=64, tid=640, inv=1, name="host.submit", ts=1_500, dur=10),
            ]
        )
    )

    trace = to_host_swimlane(spans, anchors=anchors)
    events = [event for event in trace["traceEvents"] if event.get("ph") == "X"]

    assert [event["args"].get("wall_ts_ns") for event in events] == [
        "1700000000000000500",
        "1900000000000000500",
        None,
        None,
    ]


def test_count_record_heads_sees_a_torn_record_that_parse_spans_drops():
    intact = _record(1, 1, "chip.run", "rank=0")
    torn = intact[: intact.index(" ts=")]
    lines = [intact + "\n", torn + "\n"]

    assert count_record_heads(lines) == 2
    assert len(list(parse_spans(lines))) == 1


def test_host_swimlane_keeps_real_host_lanes_and_builds_dispatch_flow():
    lines = [
        _span_record(
            pid=41,
            tid=410,
            inv=7,
            name="host.graph_build",
            ts=1_000,
            dur=900,
            attrs="run_id=7 role=facade",
        ),
        _span_record(
            pid=41,
            tid=410,
            inv=7,
            name="host.submit",
            ts=1_100,
            dur=100,
            attrs="run_id=7 task_slot=12 group_index=0 worker_id=3 role=facade",
        ),
        _span_record(
            pid=41,
            tid=411,
            inv=7,
            name="host.dispatch",
            ts=1_400,
            dur=80,
            attrs=(
                "run_id=7 task_slot=12 group_index=0 worker_id=3 dispatch_id=99 "
                "endpoint_kind=local_mailbox role=scheduler"
            ),
        ),
    ]

    trace = to_host_swimlane(list(parse_spans(lines)))
    events = trace["traceEvents"]
    slices = [event for event in events if event["ph"] == "X"]
    flows = [event for event in events if event["ph"] in {"s", "f"}]

    assert {(event["pid"], event["tid"]) for event in slices} == {(41, 410), (41, 411)}
    assert [(event["ph"], event["pid"], event["tid"]) for event in flows] == [
        ("s", 41, 410),
        ("f", 41, 411),
    ]
    assert flows[0]["id"] == flows[1]["id"]
    assert isinstance(flows[0]["id"], int)
    assert slices[-1]["args"]["dispatch_id"] == 99
    thread_names = {
        (event["pid"], event["tid"]): event["args"]["name"]
        for event in events
        if event["ph"] == "M" and event["name"] == "thread_name"
    }
    assert thread_names == {(41, 410): "orchestrator / facade", (41, 411): "scheduler"}


def test_host_swimlane_names_the_scheduler_lane_from_every_role_it_emits():
    """The scheduler thread emits `role=worker` spans before its first `role=scheduler` one.

    `scheduler.cpp`'s loop is the only caller of both `dispatch_ready()` and
    `manager->progress()`, and inside `submit_dispatch` the `host.frame_submit`
    scope closes within `submit_progress` — before the `host.dispatch` record
    emitted after the admission lock is released. Naming the lane from the
    first span it emitted therefore labels the scheduler `worker 3`.
    """
    lines = [
        _span_record(
            pid=41,
            tid=411,
            inv=7,
            name="host.frame_submit",
            ts=1_410,
            dur=40,
            attrs="run_id=7 task_slot=12 group_index=0 worker_id=3 dispatch_id=1 role=worker",
        ),
        _span_record(
            pid=41,
            tid=411,
            inv=7,
            name="host.dispatch",
            ts=1_400,
            dur=80,
            attrs="run_id=7 task_slot=12 group_index=0 worker_id=3 dispatch_id=1 role=scheduler",
        ),
        _span_record(
            pid=41,
            tid=411,
            inv=7,
            name="host.complete",
            ts=2_000,
            dur=30,
            attrs="run_id=7 task_slot=12 group_index=0 worker_id=3 dispatch_id=1 role=worker",
        ),
    ]

    trace = to_host_swimlane(list(parse_spans(lines)))
    thread_names = {
        (event["pid"], event["tid"]): event["args"]["name"]
        for event in trace["traceEvents"]
        if event["ph"] == "M" and event["name"] == "thread_name"
    }

    assert thread_names == {(41, 411): "scheduler"}


def test_parse_spans_decodes_percent_escaped_name_and_attribute_values():
    """`encode_host_span_field` escapes whatever would otherwise be record grammar."""
    lines = [
        _span_record(
            pid=41,
            tid=410,
            inv=7,
            name="host.odd%20name",
            ts=100,
            dur=10,
            attrs="run_id=7 reason=submit%20failed%3A%20%5Bfatal%5D role=facade",
        )
    ]

    span = next(iter(parse_spans(lines)))
    assert span.name == "host.odd name"

    trace = to_host_swimlane([span])
    args = next(event["args"] for event in trace["traceEvents"] if event["ph"] == "X")
    assert args["reason"] == "submit failed: [fatal]"
    # The raw field stays verbatim: it is the record as written.
    assert "%5Bfatal%5D" in args["attrs"]


def test_host_swimlane_pairs_dispatch_with_latest_preceding_submit():
    lines = [
        _span_record(
            pid=41,
            tid=410,
            inv=7,
            name="host.submit",
            ts=100,
            dur=20,
            attrs="run_id=7 task_slot=12 group_index=0 worker_id=3 role=facade",
        ),
        _span_record(
            pid=41,
            tid=411,
            inv=7,
            name="host.dispatch",
            ts=200,
            dur=10,
            attrs="run_id=7 task_slot=12 group_index=0 worker_id=3 dispatch_id=1 role=scheduler",
        ),
        _span_record(
            pid=41,
            tid=410,
            inv=7,
            name="host.submit",
            ts=300,
            dur=20,
            attrs="run_id=7 task_slot=12 group_index=0 worker_id=3 role=facade",
        ),
        _span_record(
            pid=41,
            tid=411,
            inv=7,
            name="host.dispatch",
            ts=400,
            dur=10,
            attrs="run_id=7 task_slot=12 group_index=0 worker_id=3 dispatch_id=2 role=scheduler",
        ),
    ]

    trace = to_host_swimlane(list(parse_spans(lines)))
    flows = [event for event in trace["traceEvents"] if event["ph"] in {"s", "f"}]

    assert [event["id"] for event in flows] == [1, 1, 2, 2]
    assert [event["ts"] for event in flows if event["ph"] == "s"] == [0.12, 0.32]
    assert [event["args"]["dispatch_key"] for event in flows] == [
        "dispatch:41:7:12:0:3:1",
        "dispatch:41:7:12:0:3:1",
        "dispatch:41:7:12:0:3:2",
        "dispatch:41:7:12:0:3:2",
    ]


def test_host_swimlane_keeps_unaligned_device_clock_out_of_visible_timeline():
    spans = list(
        parse_spans(
            [
                _span_record(
                    pid=41,
                    tid=410,
                    inv=7,
                    name="host.graph_build",
                    ts=1_000_000_000,
                    dur=900,
                    attrs="run_id=7 role=facade",
                ),
                _span_record(
                    pid=52,
                    tid=520,
                    inv=8,
                    name="chip.run.runner_run.device_wall",
                    ts=300,
                    dur=40,
                    attrs="clk=dev rank=1",
                ),
            ]
        )
    )

    trace = to_host_swimlane(spans)
    visible_slices = [event for event in trace["traceEvents"] if event.get("ph") == "X"]

    assert [(event["name"], event["ts"], event["dur"]) for event in visible_slices] == [
        ("host.graph_build", 1_000_000.0, 0.9)
    ]
    assert trace["unalignedDeviceSpans"] == [
        {
            "name": "chip.run.runner_run.device_wall",
            "ts_ns": 300,
            "dur_ns": 40,
            "pid": 52,
            "tid": 520,
            "inv": 8,
            "hid": "abc",
            "depth": 0,
            "attrs": {"raw": "clk=dev rank=1", "clk": "dev", "rank": 1},
        }
    ]


def test_legacy_trace_output_ignores_host_swimlane_markers():
    old = list(
        parse_spans(
            [
                _span_record(pid=61, tid=610, inv=3, name="chip.run", ts=1_000, dur=500),
                _span_record(pid=61, tid=610, inv=3, name="chip.run.bind", ts=1_100, dur=50, depth=1),
                _span_record(pid=61, tid=610, inv=4, name="chip.prewarm.build", ts=1_200, dur=75),
            ]
        )
    )
    mixed = old + list(
        parse_spans(
            [
                _span_record(
                    pid=61,
                    tid=611,
                    inv=9,
                    name="host.dispatch",
                    ts=900,
                    dur=20,
                    attrs="run_id=9 task_slot=4 group_index=0 worker_id=0 dispatch_id=1",
                )
            ]
        )
    )

    old_invocations = group_invocations(legacy_spans(old))
    mixed_invocations = group_invocations(legacy_spans(mixed))

    assert [span.name for span in legacy_spans(mixed)] == [
        "chip.run",
        "chip.run.bind",
        "chip.prewarm.build",
    ]
    assert to_chrome_trace(old_invocations, bucket_by_hid(old_invocations)) == to_chrome_trace(
        mixed_invocations, bucket_by_hid(mixed_invocations)
    )


def test_invocation_by_name_uses_earliest_timestamp_not_input_order():
    spans = list(
        parse_spans(
            [
                _span_record(pid=61, tid=610, inv=3, name="chip.run.bind", ts=1_200, dur=20, depth=1),
                _span_record(pid=61, tid=611, inv=3, name="chip.run.bind", ts=1_100, dur=30, depth=1),
            ]
        )
    )

    invocation = group_invocations(spans)[0]

    assert invocation.by_name()["chip.run.bind"].ts == 1_100


def test_swimlane_cli_writes_trace(tmp_path):
    log_path = tmp_path / "run.log"
    output_path = tmp_path / "host_swimlane.json"
    log_path.write_text(
        _anchor_record(71, 50, 1_700_000_000_000_000_000)
        + _span_record(pid=71, tid=710, inv=2, name="host.graph_build", ts=100, dur=25, attrs="run_id=2 role=facade"),
        encoding="utf-8",
    )

    assert main([str(log_path), "--swimlane", str(output_path)]) == 0

    trace = json.loads(output_path.read_text(encoding="utf-8"))
    event = next(event for event in trace["traceEvents"] if event.get("name") == "host.graph_build")
    assert event["args"]["wall_ts_ns"] == "1700000000000000050"


def test_cli_warns_when_one_pid_has_multiple_clock_anchors(tmp_path, capsys):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        _anchor_record(71, 50, 1_700_000_000_000_000_000)
        + _anchor_record(71, 75, 1_700_000_000_000_000_025)
        + _span_record(pid=71, tid=710, inv=2, name="chip.run", ts=100, dur=25),
        encoding="utf-8",
    )

    assert main([str(log_path)]) == 0

    assert "warning: multiple [CLOCK_ANCHOR] records found for pid 71" in capsys.readouterr().err


def test_rounds_table_omits_tmr_only_columns_when_only_host_and_device_exist():
    lines = []
    for inv, host_dur, device_dur in ((1, 100_000, 20_000), (2, 120_000, 24_000)):
        lines.append(
            _record(1, inv, "chip.run", dur=host_dur)
            + _record(
                1,
                inv,
                "chip.run.runner_run.device_wall",
                "clk=dev",
                depth=2,
                dur=device_dur,
            )
            + "\n"
        )
    buckets = bucket_by_hid(group_invocations(parse_spans(lines)))
    output = StringIO()

    print_rounds_table(buckets, stream=output)

    rendered = output.getvalue()
    assert "Host (us)" in rendered
    assert "Device (us)" in rendered
    assert "Effective (us)" not in rendered
    assert "Orch (us)" not in rendered
    assert "Sched (us)" not in rendered
    assert "Avg Host: 110.0 us" in rendered
    assert "Avg Device: 22.0 us [2/2]" in rendered


def _run_records(*, run_epoch, prepare, device, release, run_id=0, dispatch_id=0, slot_id=0, pid=7, inv=None):
    """One phased native run's spans, as the `chip.run` tree carries them.

    `prepare` and `device` are (ts, dur). The root span carries the identity and
    spans the whole lifetime, so it must enclose the children.
    """
    invocation = run_epoch if inv is None else inv
    identity = f"run_id={run_id} dispatch_id={dispatch_id} slot_id={slot_id} generation=1 run_epoch={run_epoch}"
    root_end = release + 10
    return [
        _record(pid, invocation, "chip.run.bind", depth=1, ts=prepare[0], dur=prepare[1]),
        _record(pid, invocation, "chip.run.runner_run", depth=1, ts=device[0], dur=device[1]),
        _record(pid, invocation, "chip.run.claim_release", depth=1, ts=release, dur=5),
        _record(pid, invocation, "chip.run", identity, depth=0, ts=prepare[0], dur=root_end - prepare[0]),
    ]


def test_native_overlap_reads_identity_from_the_invocation_root():
    """The identity is on the root; the windows are on its children.

    `(pid, inv)` is what joins them — the same grouping the TPOT views use.
    """
    lines = _run_records(run_epoch=1, prepare=(50, 20), device=(100, 100), release=200)
    lines += _run_records(run_epoch=2, prepare=(150, 30), device=(240, 100), release=340)

    checks = assert_native_overlap(parse_spans(lines))

    assert len(checks) == 1
    assert (checks[0].predecessor.sequence, checks[0].successor.sequence) == (1, 2)


def test_native_overlap_accepts_a_prepare_that_outlasts_the_device_window():
    """Overlap means the intervals intersect, not that prepare is contained.

    A prepare that starts inside the predecessor's device window and finishes
    after it still ran concurrently with device work, which is the property being
    proved. Demanding containment turns a slow host into a failure whose message
    blames the wrong thing.
    """
    lines = _run_records(run_epoch=1, prepare=(50, 20), device=(100, 100), release=200)
    lines += _run_records(run_epoch=2, prepare=(190, 40), device=(240, 100), release=340)

    assert len(assert_native_overlap(parse_spans(lines))) == 1

    with pytest.raises(NativeOverlapError, match="not fully hidden"):
        assert_native_overlap(parse_spans(lines), require_hidden=True)


def test_native_overlap_rejects_a_prepare_after_the_device_window():
    lines = _run_records(run_epoch=1, prepare=(50, 20), device=(100, 100), release=200)
    lines += _run_records(run_epoch=2, prepare=(210, 20), device=(240, 100), release=340)

    with pytest.raises(NativeOverlapError, match="did not overlap"):
        assert_native_overlap(parse_spans(lines))


def test_native_overlap_rejects_device_work_before_the_predecessor_release():
    lines = _run_records(run_epoch=1, prepare=(50, 20), device=(100, 100), release=250)
    lines += _run_records(run_epoch=2, prepare=(150, 30), device=(240, 100), release=400)

    with pytest.raises(NativeOverlapError, match="reordered before the claim release"):
        assert_native_overlap(parse_spans(lines))


def test_native_overlap_orders_by_run_epoch_when_dispatch_id_is_absent():
    """`_submit_chip_run_direct` defaults run_id and dispatch_id to 0.

    `run_epoch` is the only identity those runs carry, so it is what orders them.
    """
    lines = _run_records(run_epoch=1, prepare=(50, 20), device=(100, 100), release=200)
    lines += _run_records(run_epoch=2, prepare=(150, 30), device=(240, 100), release=340)

    identities = [check.predecessor for check in assert_native_overlap(parse_spans(lines))]

    assert identities[0].dispatch_id == 0
    assert identities[0].run_epoch == 1
    assert identities[0].sequence == 1


def test_native_overlap_orders_by_dispatch_id_when_the_scheduler_allocates_one():
    lines = _run_records(run_epoch=1, dispatch_id=10, run_id=5, prepare=(50, 20), device=(100, 100), release=200)
    lines += _run_records(run_epoch=2, dispatch_id=11, run_id=6, prepare=(150, 30), device=(240, 100), release=340)

    checks = assert_native_overlap(parse_spans(lines))

    assert (checks[0].predecessor.sequence, checks[0].successor.sequence) == (10, 11)


def test_native_overlap_skips_an_invocation_that_is_not_a_phased_run():
    """A lexical `chip.run` carries no identity attrs, so it orders nothing."""
    lines = _run_records(run_epoch=1, prepare=(50, 20), device=(100, 100), release=200)
    lines += _run_records(run_epoch=2, prepare=(150, 30), device=(240, 100), release=340)
    lines += [_record(7, 99, "chip.run", "", depth=0, ts=10, dur=5)]

    assert len(assert_native_overlap(parse_spans(lines))) == 1


def test_native_overlap_needs_two_runs_on_one_lane():
    lines = _run_records(run_epoch=1, prepare=(50, 20), device=(100, 100), release=200)

    with pytest.raises(NativeOverlapError, match="at least two complete native runs"):
        assert_native_overlap(parse_spans(lines))


def test_claim_release_span_stays_inside_the_invocation_tree():
    """It is a `chip.run` child, so the existing views absorb it.

    A separate family would have to be filtered out of every view; a depth-1
    child of the root is what the tree already knows how to render — and it
    cannot displace the root the way a second depth-0 span would.
    """
    lines = _run_records(run_epoch=1, prepare=(50, 20), device=(100, 100), release=200)

    invocation = group_invocations(legacy_spans(list(parse_spans(lines))))[0]

    assert invocation.root().name == "chip.run"
    assert invocation.by_name()["chip.run.claim_release"].depth == 1
    # The root still reports the whole lifetime, not a sub-stage's duration.
    assert invocation.root().dur == 210 - 50


def test_swimlane_splits_an_interleaved_thread_into_one_lane_per_pipeline_slot():
    """A K-deep pipeline runs on one OS thread, so tid is not the unit of concurrency.

    Flattening overlapping runs onto one lane makes Perfetto nest run N+1's spans
    inside run N's root by timestamp containment — false nesting that hides the
    very overlap the lane is meant to show. The pipeline slot is what a run holds
    exclusively, so runs sharing one cannot overlap.
    """
    lines = _run_records(run_epoch=1, slot_id=0, prepare=(50, 20), device=(100, 100), release=200)
    lines += _run_records(run_epoch=2, slot_id=1, prepare=(150, 30), device=(240, 100), release=340)
    lines += _run_records(run_epoch=3, slot_id=0, prepare=(300, 30), device=(380, 100), release=480)

    trace = to_host_swimlane(list(parse_spans(lines)))
    lanes = {
        event["tid"]: event["args"]["name"]
        for event in trace["traceEvents"]
        if event["ph"] == "M" and event["name"] == "thread_name"
    }

    assert sorted(lanes.values()) == ["pipeline slot 0 (tid 7)", "pipeline slot 1 (tid 7)"]
    # Two runs shared slot 0, so that lane holds both — and neither overlaps.
    roots = defaultdict(list)
    for event in trace["traceEvents"]:
        if event["ph"] == "X" and event["args"]["depth"] == 0:
            roots[event["tid"]].append((event["ts"], event["ts"] + event["dur"]))
    assert sorted(len(windows) for windows in roots.values()) == [1, 2]
    for windows in roots.values():
        windows.sort()
        assert all(a[1] <= b[0] for a, b in zip(windows, windows[1:]))
    # The real thread stays recoverable from the slice.
    slices = [event for event in trace["traceEvents"] if event["ph"] == "X"]
    assert {event["args"]["os_tid"] for event in slices} == {7}


def test_swimlane_keeps_a_sequential_thread_on_its_own_tid():
    """Splitting a thread whose runs never overlapped would only fragment it.

    The L3 scheduler thread carries a pipeline slot but runs strictly
    sequentially, so it must keep its real tid.
    """
    lines = _run_records(run_epoch=1, slot_id=0, prepare=(50, 20), device=(100, 100), release=200)
    lines += _run_records(run_epoch=2, slot_id=1, prepare=(300, 20), device=(350, 100), release=460)

    trace = to_host_swimlane(list(parse_spans(lines)))
    lanes = {
        event["tid"]: event["args"]["name"]
        for event in trace["traceEvents"]
        if event["ph"] == "M" and event["name"] == "thread_name"
    }

    assert list(lanes) == [7]


def test_chrome_trace_cli_writes_wall_time(tmp_path):
    log_path = tmp_path / "run.log"
    output_path = tmp_path / "strace.json"
    log_path.write_text(
        _anchor_record(71, 50, 1_700_000_000_000_000_000)
        + _span_record(pid=71, tid=710, inv=2, name="chip.run", ts=100, dur=25),
        encoding="utf-8",
    )

    assert main([str(log_path), "--trace-out", str(output_path)]) == 0

    trace = json.loads(output_path.read_text(encoding="utf-8"))
    event = next(event for event in trace["traceEvents"] if event.get("name") == "chip.run")
    assert event["ts"] == 0.1
    assert event["args"]["wall_ts_ns"] == "1700000000000000050"


def test_host_phase_records_loader_keeps_only_well_formed_passes(tmp_path):
    """Every line a consumer would choke on is dropped by the loader.

    The artifact is appended to while a run is in flight, so a truncated tail is
    ordinary. A line that parses but is not an object is malformed in the same
    sense — the consumer indexes it as a mapping — so it is dropped here rather
    than left for each consumer to re-check.
    """
    artifact = tmp_path / "host_phase_records.jsonl"
    artifact.write_text(
        "\n".join(
            [
                json.dumps({"pid": 7, "inv": 1, "records": [{"phase": "args", "start_ns": 10, "end_ns": 20}]}),
                "null",
                "[1, 2, 3]",
                "42",
                '"a string"',
                json.dumps({"pid": 7, "inv": 2, "records": "not-a-list"}),
                '{"pid": 7, "inv": 3, "records": [',
                "",
                json.dumps({"pid": 7, "inv": 4, "records": []}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    passes = load_host_phase_records([str(artifact)])

    assert [one_pass["inv"] for one_pass in passes] == [1, 4]


def test_host_record_spans_nest_bind_segments_and_orchestrator_operations(tmp_path):
    """A record's kind decides its depth: a bind segment sits under the stage, an
    orchestrator operation one level further in, under the segment it ran inside."""
    spans = list(parse_spans([_span_record(pid=9, tid=9, inv=5, name="chip.run.bind", ts=1_000, dur=500)]))
    passes = [
        {
            "pid": 9,
            "inv": 5,
            "records": [
                {"phase": "args", "start_ns": 1_000, "end_ns": 1_100, "detail": 4096},
                {"phase": "graph_submit", "start_ns": 1_200, "end_ns": 1_250, "detail": 77},
            ],
        }
    ]

    out, orphaned, covered = host_record_spans(spans, passes)

    assert orphaned == 0
    assert covered == frozenset({(9, 5)})
    by_name = {span.name: span for span in out}
    bind_depth = spans[0].depth
    assert by_name["chip.run.bind.args"].depth == bind_depth + 1
    assert by_name["chip.run.bind.host_orch.graph_submit"].depth == bind_depth + 2
    assert by_name["chip.run.bind.args"].ts == 1_000
    assert by_name["chip.run.bind.args"].dur == 100


def test_host_record_spans_drop_passes_with_no_matching_bind(tmp_path):
    spans = list(parse_spans([_span_record(pid=9, tid=9, inv=5, name="chip.run.bind", ts=1_000, dur=500)]))
    passes = [{"pid": 9, "inv": 999, "records": [{"phase": "args", "start_ns": 1_000, "end_ns": 1_100}]}]

    out, orphaned, covered = host_record_spans(spans, passes)

    assert out == []
    assert orphaned == 1
    assert covered == frozenset()
