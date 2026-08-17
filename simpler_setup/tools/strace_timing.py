#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Parse simpler host-side trace markers (``[STRACE]``) into per-stage timing.

The host runtime emits one ``[STRACE]`` line per span on scope exit (RAII
markers in ``src/common/log/include/common/strace.h``), gated by the
compile-time ``SIMPLER_HOST_STRACE`` macro (on by default) and emitted at
``LOG_TIMING``. Device-domain phases (AICPU subdivision of the on-NPU wall)
are emitted by the host after readback as ``clk=dev`` spans nested under
``simpler_run.runner_run.device_wall``.

Runtimes emit only the device spans they implement. Both current runtimes emit
``device_wall``; the finer orch/sched phase subdivision is TMR-specific.

Marker grammar (matched anywhere on the line, so the CANN/host log prefix is
ignored)::

    [STRACE] v=1 pid=<n> tid=<n> inv=<n> hid=<hex> depth=<n> name=<dotted> ts=<ns> dur=<ns> [k=v ...]

Grouping:
    * ``(pid, inv)`` identifies one ``simpler_run`` invocation — all its spans
      share these. ``inv`` is a process-wide id (atomic-allocated, so unique even
      across concurrent calls), NOT a token index.
    * ``hid`` is the callable's content hash (stable across slot reuse / runs).
      The most-frequently-seen hid bucket is the decode callable (one
      invocation per token); a once-seen hid is prefill.
    * ``depth`` rebuilds the call tree per invocation (no timestamp-containment
      guessing): a span at depth d is a child of the most recent span at d-1.

Outputs:
    * a per-callable TPOT table (each invocation's simpler_run dur + the mean
      of each sub-stage across invocations), and
    * optionally a Chrome-trace / Perfetto JSON (``--trace-out``): one ``ph:"X"``
      event per span on a synthetic per-invocation lane, so each host call tree
      renders as nested slices, or
    * a host scheduler swimlane (``--swimlane``) whose lanes are the real OS
      pid/tid, except that a thread which interleaved runs is split into one lane
      per pipeline slot so each lane reads as a sequence; cross-thread handoffs
      are Chrome flow events.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

# The monotonic prefix is current; the wall-clock form keeps archived logs
# parseable. The func segment excludes ':' because `LOG_TIMING` passes
# `__FUNCTION__`, an unqualified name. A qualified name containing '::' stops
# this alternative from matching, leaving `[STRACE]` to bound the record.
_HOST_LOG_TIME = r"(?:\[mono_ns=\d+\]|\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}\])"
_HOST_LOG_PREFIX = _HOST_LOG_TIME + r"\[T0x[0-9a-fA-F]+\]\[[A-Z]+\]\s+[^:\r\n]+:\s+"
# MULTILINE anchors the `$` alternative at every line end, so a caller passing a
# multi-line blob rather than one line per item keeps every record but the last.
_STRACE_RE = re.compile(
    r"\[STRACE\]\s+v=(?P<v>\d+)\s+pid=(?P<pid>\d+)\s+tid=(?P<tid>\d+)\s+"
    r"inv=(?P<inv>\d+)\s+hid=(?P<hid>[0-9a-fA-F]+)\s+depth=(?P<depth>\d+)\s+"
    r"name=(?P<name>\S+)\s+ts=(?P<ts>\d+)\s+dur=(?P<dur>\d+)(?P<attrs>.*?)"
    rf"(?={_HOST_LOG_PREFIX}|\[STRACE\]|\r?$)",
    re.MULTILINE,
)
# A record start, matched independently of whether the rest of that record
# survived the write that emitted it.
_STRACE_HEAD_RE = re.compile(r"\[STRACE\]\s+v=\d+")
_CLOCK_ANCHOR_RE = re.compile(
    r"\[mono_ns=\d+\]\[T0x[0-9a-fA-F]+\]\[TIMING\]\s+clock_anchor:\s+"
    r"\[CLOCK_ANCHOR\]\s+v=(?P<v>\d+)\s+pid=(?P<pid>\d+)\s+"
    r"mono_ns=(?P<mono_ns>\d+)\s+wall_ns=(?P<wall_ns>\d+)[ \t]*\r?$",
    re.MULTILINE,
)
# The emitter percent-encodes any byte that would otherwise be record grammar —
# see `encode_host_span_field` in src/common/log/host_log.cpp.
_PERCENT_ESCAPE_RE = re.compile(r"%([0-9A-Fa-f]{2})")


def decode_field(text):
    """Reverse the emitter's percent-encoding of a name or attribute value.

    A field the emitter truncated ends in ``~``, which is left in place: it is a
    marker that the value is incomplete, not an encoded byte.
    """
    return _PERCENT_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), text)


@dataclass
class ClockAnchor:
    pid: int
    mono_ns: int
    wall_ns: int

    def to_wall_ns(self, monotonic_ns):
        return self.wall_ns + monotonic_ns - self.mono_ns


@dataclass
class Span:
    pid: int
    tid: int
    inv: int
    hid: str
    depth: int
    name: str
    ts: int
    dur: int
    attrs: str

    @property
    def is_device(self) -> bool:
        return "clk=dev" in self.attrs


class NativeOverlapError(ValueError):
    """Raised when pipeline markers do not prove native preparation overlap."""


@dataclass(frozen=True)
class NativeDispatchIdentity:
    pid: int
    run_id: int
    dispatch_id: int
    run_epoch: int
    slot_id: int
    generation: int

    @property
    def sequence(self):
        """Submission order within a process.

        ``dispatch_id`` is the scheduler's, and is zero on the direct-chip lane,
        which allocates none; ``run_epoch`` is a per-process monotonic counter
        that is always set. Neither field stands in for the other in the record —
        the choice is made here, where it is visible.
        """
        return self.dispatch_id or self.run_epoch


@dataclass(frozen=True)
class NativeOverlapCheck:
    predecessor: NativeDispatchIdentity
    successor: NativeDispatchIdentity


@dataclass
class Invocation:
    """All spans emitted by one simpler_run call (one (pid, inv) group)."""

    pid: int
    inv: int
    hid: str
    spans: list = field(default_factory=list)

    def root(self):
        """The depth-0 span (simpler_run), or None if absent."""
        for s in self.spans:
            if s.depth == 0:
                return s
        return None

    def by_name(self):
        m = {}
        for s in self.spans:
            previous = m.get(s.name)
            if previous is None or s.ts < previous.ts:
                m[s.name] = s
        return m


def count_record_heads(lines):
    """Count ``[STRACE]`` record starts, torn ones included.

    Pairs with :func:`parse_spans`, which yields only records that survived
    intact. A shortfall between the two counts is instrumentation loss, and
    without it a torn record is indistinguishable from a real measurement.
    """
    return sum(len(_STRACE_HEAD_RE.findall(line)) for line in lines)


def parse_clock_anchors(lines):
    """Yield the per-process monotonic-to-wall mappings in a log."""
    for line in lines:
        for match in _CLOCK_ANCHOR_RE.finditer(line):
            if int(match["v"]) != 1:
                continue
            yield ClockAnchor(
                pid=int(match["pid"]),
                mono_ns=int(match["mono_ns"]),
                wall_ns=int(match["wall_ns"]),
            )


def parse_spans(lines):
    """Yield every complete span, including adjacent records on one line."""
    for line in lines:
        for m in _STRACE_RE.finditer(line):
            yield Span(
                pid=int(m["pid"]),
                tid=int(m["tid"]),
                inv=int(m["inv"]),
                hid=m["hid"].lower(),
                depth=int(m["depth"]),
                name=decode_field(m["name"]),
                ts=int(m["ts"]),
                dur=int(m["dur"]),
                attrs=m["attrs"].strip(),
            )


def legacy_spans(spans):
    """Return spans belonging to the established ``simpler_run`` views.

    Other marker families share the STRACE grammar but answer a different
    question, and the invocation views are keyed on ``(pid, inv)`` — a family
    that carries no invocation id would group into one bogus lane. Keeping them
    out here preserves the TPOT, rounds, tree, and ``--trace-out`` contracts for
    every consumer at once; filtering downstream covers only the view that
    filters. Existing families such as ``simpler_prewarm`` keep their old
    behavior.
    """
    return [span for span in spans if not span.name.startswith("l3.")]


def group_invocations(spans):
    """Group spans into Invocation objects keyed by (pid, inv)."""
    groups: dict = {}
    for s in spans:
        key = (s.pid, s.inv)
        inv = groups.get(key)
        if inv is None:
            inv = Invocation(pid=s.pid, inv=s.inv, hid=s.hid)
            groups[key] = inv
        inv.spans.append(s)
    # Stable order: by pid then inv.
    return [groups[k] for k in sorted(groups)]


def bucket_by_hid(invocations):
    """Map hid -> [Invocation], ordered by inv within each bucket."""
    buckets: dict = defaultdict(list)
    for inv in invocations:
        buckets[inv.hid].append(inv)
    for bucket in buckets.values():
        bucket.sort(key=lambda i: i.inv)
    return buckets


# The spans one native run contributes to the overlap proof. All three already
# exist in the `simpler_run` tree; only `claim_release` was added for it.
_PREPARE_SPAN = "simpler_run.bind"
_DEVICE_SPAN = "simpler_run.runner_run"
_RELEASE_SPAN = "simpler_run.claim_release"
_NATIVE_REQUIRED_SPANS = (_PREPARE_SPAN, _DEVICE_SPAN, _RELEASE_SPAN)
_PIPELINE_IDENTITY_FIELDS = ("run_id", "dispatch_id", "run_epoch", "slot_id", "generation")


def _native_dispatches(invocations):
    """Map each native run's identity to its ``{span name: span}``.

    The identity rides on the root ``simpler_run`` span, and every sub-span of
    that run shares its ``(pid, inv)`` — so grouping by invocation is what joins
    the windows to the identity. An invocation whose root carries no identity is
    not a phased native run and is skipped.
    """
    dispatches = {}
    for invocation in invocations:
        root = invocation.root()
        if root is None:
            continue
        attrs = _parsed_attrs(root)
        if not any(field in attrs for field in _PIPELINE_IDENTITY_FIELDS):
            continue
        missing = [field for field in _PIPELINE_IDENTITY_FIELDS if field not in attrs]
        if missing:
            raise NativeOverlapError(
                f"{root.name} (pid={root.pid} inv={root.inv}) is missing identity field(s): {', '.join(missing)}"
            )
        try:
            identity = NativeDispatchIdentity(
                pid=root.pid,
                run_id=int(attrs["run_id"]),
                dispatch_id=int(attrs["dispatch_id"]),
                run_epoch=int(attrs["run_epoch"]),
                slot_id=int(attrs["slot_id"]),
                generation=int(attrs["generation"]),
            )
        except (TypeError, ValueError) as exc:
            raise NativeOverlapError(f"{root.name} has a non-integer identity") from exc
        if identity.sequence == 0:
            continue
        if identity in dispatches:
            raise NativeOverlapError(f"duplicate identity for pid={identity.pid} sequence={identity.sequence}")
        dispatches[identity] = invocation.by_name()
    return dispatches


def assert_native_overlap(spans, *, require_hidden=False):
    """Prove native prepare overlap and FIFO launch ordering for adjacent runs.

    Two properties per adjacent pair on one process lane:

    * ``bind(N+1)`` overlaps ``runner_run(N)`` — the intervals intersect, which
      is what makes the successor's preparation concurrent with the
      predecessor's device work.
    * ``runner_run(N+1)`` does not start before ``claim_release(N)``.

    ``bind`` is the successor's own arena build and host orchestration and sits
    inside its prepare, so reading it is conservative: an overlap it reports is
    one the prepare certainly had.

    ``require_hidden`` additionally demands that the preparation *finish* inside
    the predecessor's device window — fully hidden rather than merely
    overlapping. That is a claim about pipeline depth and is sensitive to host
    scheduling, so it is opt-in: a bind that runs long on a contended machine
    still overlaps.
    """
    dispatches = _native_dispatches(group_invocations(legacy_spans(spans)))
    by_pid = defaultdict(list)
    for identity, by_name in dispatches.items():
        missing = [name for name in _NATIVE_REQUIRED_SPANS if name not in by_name]
        if missing:
            raise NativeOverlapError(f"sequence={identity.sequence} is missing span(s): {', '.join(missing)}")
        by_pid[identity.pid].append((identity, by_name))

    checks = []
    for lane in by_pid.values():
        lane.sort(key=lambda item: item[1][_DEVICE_SPAN].ts)
        for (predecessor, pred_spans), (successor, succ_spans) in zip(lane, lane[1:]):
            if successor.sequence <= predecessor.sequence:
                raise NativeOverlapError(
                    f"launch order is not monotonic: predecessor={predecessor.sequence} successor={successor.sequence}"
                )
            pred_device = pred_spans[_DEVICE_SPAN]
            pred_release = pred_spans[_RELEASE_SPAN]
            succ_prepare = succ_spans[_PREPARE_SPAN]
            succ_device = succ_spans[_DEVICE_SPAN]
            pred_device_end = pred_device.ts + pred_device.dur
            succ_prepare_end = succ_prepare.ts + succ_prepare.dur
            if succ_prepare.ts >= pred_device_end or succ_prepare_end <= pred_device.ts:
                raise NativeOverlapError(
                    f"preparation did not overlap: sequence={successor.sequence} prepare="
                    f"[{succ_prepare.ts},{succ_prepare_end}] predecessor_device="
                    f"[{pred_device.ts},{pred_device_end}]"
                )
            if require_hidden and succ_prepare_end > pred_device_end:
                raise NativeOverlapError(
                    f"preparation was not fully hidden: sequence={successor.sequence} prepare_end="
                    f"{succ_prepare_end} predecessor_device_end={pred_device_end}"
                )
            if succ_device.ts < pred_release.ts:
                raise NativeOverlapError(
                    f"device execution reordered before the claim release: sequence={successor.sequence} "
                    f"device_start={succ_device.ts} predecessor_release={pred_release.ts}"
                )
            checks.append(NativeOverlapCheck(predecessor=predecessor, successor=successor))
    if not checks:
        raise NativeOverlapError("need at least two complete native runs on one process lane")
    return checks


def _fmt_us(ns: int) -> str:
    return f"{ns / 1000.0:.1f}"


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _median(values):
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def print_tpot_table(buckets, label_for_hid=None, stream=sys.stdout):
    """Print a per-callable TPOT table. The most-invoked bucket is decode."""
    if not buckets:
        print("No [STRACE] markers found.", file=stream)
        return

    ordered = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
    for hid, invs in ordered:
        label = (label_for_hid or {}).get(hid, "")
        header = f"callable hid={hid}"
        if label:
            header += f" ({label})"
        header += f" — {len(invs)} invocation(s)"
        print(header, file=stream)

        roots = [i.root() for i in invs if i.root() is not None]
        durs = [r.dur for r in roots]
        if durs:
            print(
                f"  simpler_run: mean={_fmt_us(int(_mean(durs)))}us "
                f"min={_fmt_us(min(durs))}us max={_fmt_us(max(durs))}us",
                file=stream,
            )

        # Mean of each sub-stage across invocations (by span name).
        stage_durs: dict = defaultdict(list)
        for inv in invs:
            for name, span in inv.by_name().items():
                if span.depth == 0:
                    continue
                stage_durs[name].append(span.dur)
        for name in sorted(stage_durs, key=lambda n: (-len(stage_durs[n]), n)):
            ds = stage_durs[name]
            indent = "  " + "  " * name.count(".")
            print(f"{indent}{name}: mean={_fmt_us(int(_mean(ds)))}us (n={len(ds)})", file=stream)
        print(file=stream)


_ROUNDS_TABLE_NAMES = {
    "host": "simpler_run",
    "device": "simpler_run.runner_run.device_wall",
    "orch": "simpler_run.runner_run.device_wall.orch",
    "sched": "simpler_run.runner_run.device_wall.sched",
}

# Per-round table columns, in print order. "Effective" is the orch∪sched merged
# window (the old device-log "Total"), recomputed here purely from the orch/sched
# markers' device-domain ts+dur — no device log needed. label is the column
# header / "Avg <label>".
_ROUNDS_TABLE_COLUMNS = ("Host", "Device", "Effective", "Orch", "Sched")


def _round_metrics(inv):
    """Return one round's (Host, Device, Effective, Orch, Sched) in µs from spans.

    Host/Device/Orch/Sched are span durations; Effective =
    ``max(orch_end, sched_end) - min(orch_start, sched_start)`` from the orch/sched
    spans' device-domain ``ts``/``dur`` (0 when neither is present). All values in
    µs. Column order matches ``_ROUNDS_TABLE_COLUMNS``.
    """
    names = inv.by_name()

    def _dur(key):
        span = names.get(_ROUNDS_TABLE_NAMES[key])
        return span.dur / 1000.0 if span is not None else 0.0

    orch = names.get(_ROUNDS_TABLE_NAMES["orch"])
    sched = names.get(_ROUNDS_TABLE_NAMES["sched"])
    windows = [s for s in (orch, sched) if s is not None]
    if windows:
        start = min(s.ts for s in windows)
        end = max(s.ts + s.dur for s in windows)
        effective = (end - start) / 1000.0
    else:
        effective = 0.0

    return (_dur("host"), _dur("device"), effective, _dur("orch"), _dur("sched"))


def print_rounds_table(buckets, stream=sys.stdout):
    """Print a per-round Host/Device/Effective/Orch/Sched table (µs) for the busiest hid.

    This renders the per-round benchmark table that ``scene_test`` used to print
    inline. The most-invoked hid bucket is treated as the rounds (one row per
    invocation, ordered by ``inv``); each row's metrics come from
    :func:`_round_metrics`. A column is hidden when every row read 0 (e.g.
    device/orch/sched/effective are 0 when their marker is absent; for example,
    HBG emits device wall but has no device-side orch/sched windows).

    The output format is consumed by ``tools/benchmark_rounds.sh``'s
    framework-table parser (header ``Round  Host (us) …``, ``Avg Host:``
    terminator).
    """
    if not buckets:
        print("No [STRACE] markers found.", file=stream)
        return

    # Busiest hid = the rounds (decode emits one invocation per token; a static
    # L2 example emits one per --rounds repetition).
    _, invs = max(buckets.items(), key=lambda kv: len(kv[1]))
    invs = sorted(invs, key=lambda i: i.inv)
    rows = [_round_metrics(inv) for inv in invs]

    if not rows:
        print("No [STRACE] markers found.", file=stream)
        return

    n = len(rows)
    # Host (col 0) is always captured → averaged over all rounds. Every other
    # column is shown only if some round captured it, and averaged over nonzero.
    host_vals = sorted(r[0] for r in rows)
    host_avg = sum(host_vals) / n

    nz = {}  # col idx -> sorted nonzero values (cols 1..N)
    for idx in range(1, len(_ROUNDS_TABLE_COLUMNS)):
        vals = sorted(r[idx] for r in rows if r[idx] > 0.0)
        if vals:
            nz[idx] = vals
    shown = [0] + [idx for idx in range(1, len(_ROUNDS_TABLE_COLUMNS)) if idx in nz]

    def _avg(idx):
        return sum(nz[idx]) / len(nz[idx])

    header = f"  {'Round':<6}"
    for idx in shown:
        header += f"  {_ROUNDS_TABLE_COLUMNS[idx] + ' (us)':>12}"
    print(header, file=stream)
    print("  " + "-" * (len(header) - 2), file=stream)
    for i, r in enumerate(rows):
        line = f"  {i:<6d}"
        for idx in shown:
            line += f"  {r[idx]:>12.1f}"
        print(line, file=stream)

    summary = f"  Avg Host: {host_avg:.1f} us"
    for idx in shown[1:]:
        summary += f"  |  Avg {_ROUNDS_TABLE_COLUMNS[idx]}: {_avg(idx):.1f} us"
        if idx == 1:  # device gets a capture-count annotation
            summary += f" [{len(nz[1])}/{n}]"
    summary += f"  ({n} rounds)"
    print(summary, file=stream)

    trim = 10
    if n > 2 * trim:
        tc = n - 2 * trim
        host_trim = sum(host_vals[trim:-trim]) / tc
        msg = f"  Trimmed Avg Host: {host_trim:.1f} us"
        if 1 in nz and len(nz[1]) > 2 * trim:
            dev = nz[1]
            msg += f"  |  Trimmed Avg Device: {sum(dev[trim:-trim]) / (len(dev) - 2 * trim):.1f} us"
        msg += f"  (dropped {trim} low + {trim} high, {tc} rounds used)"
        print(msg, file=stream)


def _bucket_label(buckets, hid):
    """Short human label for an hid: 'decode' (busiest bucket) / 'prefill' (once) / hid prefix."""
    if not buckets:
        return hid[:8]
    ordered = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
    if hid == ordered[0][0] and len(ordered[0][1]) > 1:
        return "decode"
    if len(buckets.get(hid, [])) == 1:
        return "prefill"
    return hid[:8]


def to_chrome_trace(invocations, buckets=None):
    """Build a Chrome-trace / Perfetto event list with readable nested tracks.

    Each invocation gets its own named process lane ("decode inv=3" /
    "prefill inv=1"), and within it host spans and device (``clk=dev``) spans go
    to two separate threads — because host ``ts`` is steady_clock while device
    ``ts`` is a device-clock offset, the two are NOT on a common timeline and
    must not share a track. Within each track the spans nest by their own
    ``ts``/``dur`` (Perfetto renders containment as nested slices), and ``depth``
    is carried so the structure is unambiguous.
    """
    events = []
    lane_map = {}
    for inv in invocations:
        label = _bucket_label(buckets, inv.hid) if buckets else inv.hid[:8]
        # One process lane per invocation; host vs device on separate tracks.
        # Key by (pid, inv): `inv` is only unique within a pid, so distinct
        # processes (L3 parent + L2 children) can share inv values — mapping the
        # pair to a dense lane id keeps their lanes from merging in Perfetto.
        key = (inv.pid, inv.inv)
        if key not in lane_map:
            lane_map[key] = len(lane_map) + 1
        lane = lane_map[key]
        host_tid, dev_tid = 0, 1
        events.append(
            {
                "ph": "M",
                "name": "process_name",
                "pid": lane,
                "tid": host_tid,
                "args": {"name": f"{label} inv={inv.inv} (pid={inv.pid})"},
            }
        )
        events.append({"ph": "M", "name": "thread_name", "pid": lane, "tid": host_tid, "args": {"name": "host"}})
        events.append(
            {"ph": "M", "name": "thread_name", "pid": lane, "tid": dev_tid, "args": {"name": "device (clk=dev)"}}
        )
        for s in inv.spans:
            events.append(
                {
                    "name": s.name,
                    "ph": "X",
                    "ts": s.ts / 1000.0,  # Chrome trace ts is microseconds
                    "dur": s.dur / 1000.0,
                    "pid": lane,
                    "tid": dev_tid if s.is_device else host_tid,
                    "args": {"inv": s.inv, "hid": s.hid, "depth": s.depth, "attrs": s.attrs},
                }
            )
    return {"traceEvents": events, "displayTimeUnit": "ms"}


def _parsed_attrs(span):
    attrs = {}
    for attribute in span.attrs.split():
        key, separator, value = attribute.partition("=")
        if not separator:
            continue
        if re.fullmatch(r"-?\d+", value):
            attrs[key] = int(value)
        else:
            attrs[key] = decode_field(value)
    return attrs


# Highest-precedence match wins. One OS thread emits spans of several roles: the
# scheduler loop is the sole caller of both `dispatch_ready` and
# `manager->progress`, so it emits `l3.dispatch` (role=scheduler) alongside
# `l3.frame_submit` / `l3.activate` / `l3.complete`, whose `role=worker` names
# the worker a dispatch targets rather than the thread doing the work.
_HOST_THREAD_ROLES = ("facade", "scheduler", "worker")


def _roots_overlap(entries):
    """Whether two runs were in flight at once on this thread.

    A depth-0 span is one run's whole lifetime, so two of them overlapping means
    the thread interleaved runs. Nesting *within* a run is wanted; nesting one
    run's spans inside another's is an artifact of flattening them onto one lane.
    """
    roots = sorted((span.ts, span.ts + span.dur) for span, _ in entries if span.depth == 0)
    return any(a[1] > b[0] for a, b in zip(roots, roots[1:]))


def _slot_by_invocation(entries):
    """Map ``(pid, inv)`` to the pipeline slot that run held.

    Only the root span carries the identity; its children carry no attributes at
    all. They share the root's ``(pid, inv)``, so that is the join — the same one
    :func:`assert_native_overlap` uses.
    """
    slots = {}
    for span, attrs in entries:
        if "slot_id" in attrs:
            slots[(span.pid, span.inv)] = attrs["slot_id"]
    return slots


def _slot_lanes(entries, slot_by_invocation):
    """Split one thread's spans by pipeline slot, or None to leave it on its tid.

    The pipeline slot is what a run holds exclusively, so runs sharing a slot
    cannot overlap — which is what makes a per-slot lane render as a plain
    sequence instead of false containment. Returning None keeps the real-tid
    lane: splitting a thread whose runs never overlapped would only fragment it,
    and the L3 scheduler thread carries a slot while running strictly
    sequentially.
    """
    if not _roots_overlap(entries):
        return None
    by_slot = defaultdict(list)
    for span, attrs in entries:
        slot_id = slot_by_invocation.get((span.pid, span.inv))
        if slot_id is None:
            return None
        by_slot[slot_id].append((span, attrs))
    if len(by_slot) < 2 or any(_roots_overlap(group) for group in by_slot.values()):
        return None
    return dict(sorted(by_slot.items()))


def _host_thread_name(entries):
    """Name one OS thread's lane from every span it emitted.

    `entries` are that thread's (span, parsed attributes) pairs.
    """
    roles = set()
    worker_ids = set()
    for span, attrs in entries:
        role = attrs.get("role")
        if role == "facade" or span.name in {"l3.graph_build", "l3.submit"}:
            roles.add("facade")
        elif role in ("scheduler", "worker"):
            roles.add(role)
        elif span.name.startswith("l3."):
            roles.add("worker")
        if role == "worker":
            worker_ids.add(attrs.get("worker_id"))

    for role in _HOST_THREAD_ROLES:
        if role not in roles:
            continue
        if role == "facade":
            return "orchestrator / facade"
        if role == "scheduler":
            return "scheduler"
        worker_id = worker_ids.pop() if len(worker_ids) == 1 else None
        return f"worker {worker_id}" if worker_id is not None else "worker"

    if any(span.name == "simpler_run" or span.name.startswith("simpler_run.") for span, _ in entries):
        return "chip child"
    return f"tid {entries[0][0].tid}"


def _flow_key(span, attrs):
    run_id = attrs.get("run_id")
    task_slot = attrs.get("task_slot", attrs.get("slot"))
    if run_id is None or task_slot is None:
        return None
    return span.pid, run_id, task_slot


def _assign_lanes(host_entries, host_threads):
    """Choose a lane per span, and a name per lane.

    A thread that interleaved runs is split by pipeline slot (see
    :func:`_slot_lanes`); every other thread keeps its real tid. Synthetic lane
    ids start past the observed tid space so a split lane cannot collide with a
    real thread's.
    """
    lane_of = {}
    lane_names = {}
    next_synthetic_tid = max(tid for _, tid in host_threads) + 1
    slot_by_invocation = _slot_by_invocation(host_entries)
    for pid, tid in host_threads:
        on_thread = [entry for entry in host_entries if entry[0].pid == pid and entry[0].tid == tid]
        slot_lanes = _slot_lanes(on_thread, slot_by_invocation)
        if slot_lanes is None:
            lane_names[(pid, tid)] = _host_thread_name(on_thread)
            for span, _ in on_thread:
                lane_of[id(span)] = tid
            continue
        for slot_id, group in slot_lanes.items():
            lane_tid = next_synthetic_tid
            next_synthetic_tid += 1
            lane_names[(pid, lane_tid)] = f"pipeline slot {slot_id} (tid {tid})"
            for span, _ in group:
                lane_of[id(span)] = lane_tid
    return lane_of, lane_names


def to_host_swimlane(spans):
    """Build a real-pid/tid host scheduling timeline for Perfetto.

    Host timestamps remain on their shared CLOCK_MONOTONIC axis. Chrome Trace
    JSON has one timestamp axis, so raw ``clk=dev`` events cannot be rendered
    alongside host events without either a false clock alignment or a huge
    empty interval. Keep those raw events in ``unalignedDeviceSpans`` for
    inspection, but do not add them to Perfetto's visible ``traceEvents``.
    """
    # (span, parsed attributes) pairs, so the attributes travel with their span
    # through every partition below. `Span` is an unhashable dataclass, so a
    # side table would have to be keyed on identity.
    entries = [(span, _parsed_attrs(span)) for span in spans]
    events = []

    host_entries = [entry for entry in entries if not entry[0].is_device]
    device_entries = [entry for entry in entries if entry[0].is_device]
    host_pids = sorted({span.pid for span, _ in host_entries})
    host_threads = sorted({(span.pid, span.tid) for span, _ in host_entries})

    lane_of, lane_names = _assign_lanes(host_entries, host_threads)

    for pid in host_pids:
        process_spans = [span for span, _ in host_entries if span.pid == pid]
        role = "host" if any(span.name.startswith("l3.") for span in process_spans) else "chip child"
        events.append(
            {
                "ph": "M",
                "name": "process_name",
                "pid": pid,
                "tid": 0,
                "args": {"name": f"simpler {role} (pid={pid})"},
            }
        )
    for (pid, lane_tid), name in sorted(lane_names.items()):
        events.append(
            {
                "ph": "M",
                "name": "thread_name",
                "pid": pid,
                "tid": lane_tid,
                "args": {"name": name},
            }
        )
    for span, parsed in sorted(host_entries, key=lambda item: (item[0].ts, item[0].pid, item[0].tid, item[0].name)):
        event_args = {
            "inv": span.inv,
            "hid": span.hid,
            "depth": span.depth,
            "attrs": span.attrs,
            "os_tid": span.tid,
            **parsed,
        }
        events.append(
            {
                "name": span.name,
                "ph": "X",
                "ts": span.ts / 1000.0,
                "dur": span.dur / 1000.0,
                "pid": span.pid,
                "tid": lane_of[id(span)],
                "args": event_args,
            }
        )

    submits = defaultdict(list)
    for span, attrs in host_entries:
        if span.name != "l3.submit":
            continue
        key = _flow_key(span, attrs)
        if key is not None:
            submits[key].append(span)
    for candidates in submits.values():
        candidates.sort(key=lambda item: item.ts)

    dispatches = []
    for span, attrs in host_entries:
        if span.name != "l3.dispatch":
            continue
        key = _flow_key(span, attrs)
        source = None
        if key is not None:
            for candidate in submits.get(key, []):
                if candidate.ts > span.ts:
                    break
                source = candidate
        if source is None:
            continue
        dispatches.append((source, span, attrs))

    for flow_id, (source, destination, attrs) in enumerate(sorted(dispatches, key=lambda item: item[1].ts), start=1):
        dispatch_key = (
            f"dispatch:{source.pid}:{attrs['run_id']}:{attrs.get('task_slot', attrs.get('slot'))}:"
            f"{attrs.get('group_index', -1)}:{attrs.get('worker_id', -1)}:{attrs.get('dispatch_id', 0)}"
        )
        flow_args = {"dispatch_key": dispatch_key}
        events.append(
            {
                "name": "task dispatch",
                "cat": "host.scheduler",
                "ph": "s",
                "id": flow_id,
                "ts": min(source.ts + source.dur, destination.ts) / 1000.0,
                "pid": source.pid,
                "tid": lane_of[id(source)],
                "args": flow_args,
            }
        )
        events.append(
            {
                "name": "task dispatch",
                "cat": "host.scheduler",
                "ph": "f",
                "id": flow_id,
                "ts": destination.ts / 1000.0,
                "pid": destination.pid,
                "tid": lane_of[id(destination)],
                "args": flow_args,
            }
        )

    unaligned_device_spans = []
    for span, attrs in sorted(
        device_entries, key=lambda item: (item[0].pid, item[0].inv, item[0].ts, item[0].tid, item[0].name)
    ):
        unaligned_device_spans.append(
            {
                "name": span.name,
                "ts_ns": span.ts,
                "dur_ns": span.dur,
                "pid": span.pid,
                "tid": span.tid,
                "inv": span.inv,
                "hid": span.hid,
                "depth": span.depth,
                "attrs": {"raw": span.attrs, **attrs},
            }
        )

    return {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "unalignedDeviceSpans": unaligned_device_spans,
    }


def _print_agg_tree(invs, stream=sys.stdout):
    """Print a callable's spans as a nested tree built from the dotted span
    names (so e.g. ``simpler_run.bind.args`` nests under ``simpler_run.bind``),
    NOT from depth+ts — host (steady_clock) and device (``clk=dev``) spans live
    on different clocks, so timestamp containment across domains is meaningless;
    the dotted name is the unambiguous parent link. Device spans are tagged
    ``[dev]``; durations are µs.

    Each node's duration is the **median across every invocation** of this
    callable, not one invocation's value. A single-invocation tree would mislead
    on a callable whose invocations differ in cost — e.g. qwen3 decode, where the
    pypto-serving profile warmup dispatches a tiny-KV decode step (seq_len≈257)
    before the real steps: its Effective (~28 ms) is far below the steady-state
    (~40 ms at 3.5k context). The median is robust to that warmup outlier."""
    # Per-span-name median duration across all invocations. by_name() rebuilds
    # its dict per call, so materialize each invocation's map once and reuse it
    # for both the medians here and the per-inv Effective loop below.
    by_names = [inv.by_name() for inv in invs]
    dur_samples: dict = defaultdict(list)
    for bn in by_names:
        for name, span in bn.items():
            dur_samples[name].append(span.dur)
    med = {name: _median(ds) for name, ds in dur_samples.items()}

    # Structure + ts ordering from the LAST invocation — for qwen decode the
    # warmup step is inv 0, so the last is a steady-state one; either way the
    # dotted-name tree shape is identical across invocations.
    ref = invs[-1]
    by_name = {s.name: s for s in ref.spans}
    children = {}
    roots = []
    for s in ref.spans:
        parent = s.name.rsplit(".", 1)[0] if "." in s.name else None
        if parent is not None and parent in by_name:
            children.setdefault(parent, []).append(s)
        else:
            roots.append(s)

    def emit(s, indent):
        tag = " [dev]" if s.is_device else ""
        leaf = s.name.rsplit(".", 1)[-1] if "." in s.name else s.name
        stream.write(f"{'  ' * indent}{leaf:<22}{tag:>6}  {med[s.name] / 1000.0:>12.1f} us\n")
        kids = sorted(children.get(s.name, []), key=lambda x: x.ts)
        # orch and sched run concurrently (see docs/dfx/device-phases.md): render
        # them on ONE line, left = orch, right = sched, under their merged window
        # `Effective = orch ∪ sched`, instead of as two sequential-looking rows.
        has_sched = any(k.name.rsplit(".", 1)[-1] == "sched" for k in kids)
        has_orch = any(k.name.rsplit(".", 1)[-1] == "orch" for k in kids)
        for c in kids:
            cleaf = c.name.rsplit(".", 1)[-1]
            if cleaf == "orch" and has_sched:
                sched = next(k for k in kids if k.name.rsplit(".", 1)[-1] == "sched")
                # Effective is per-invocation (orch ∪ sched depends on both
                # markers' overlap in that inv), so take the median of the
                # per-inv Effective values rather than combining the two medians.
                effs = []
                for bn in by_names:
                    o, sc = bn.get(c.name), bn.get(sched.name)
                    if o is not None and sc is not None:
                        effs.append(max(o.ts + o.dur, sc.ts + sc.dur) - min(o.ts, sc.ts))
                eff = _median(effs) / 1000.0
                base = "  " * (indent + 1)
                # Effective = the merged orch ∪ sched window, with the two
                # concurrent children shown side by side on the indented line
                # below it (see docs/dfx/device-phases.md).
                stream.write(f"{base}{'Effective':<22} [dev]  {eff:>12.1f} us\n")
                stream.write(
                    f"{base}  orch {med[c.name] / 1000.0:.1f}  ∥  sched {med[sched.name] / 1000.0:.1f}   (concurrent)\n"
                )
            elif cleaf == "sched" and has_orch:
                continue  # shown beside orch on the Effective line above
            else:
                emit(c, indent + 1)

    for r in sorted(roots, key=lambda x: x.ts):
        emit(r, 0)


def print_tree(buckets, stream=sys.stdout):
    """Per-callable, per-invocation indented tree of spans (the nested view)."""
    if not buckets:
        print("No [STRACE] markers found.", file=stream)
        return
    ordered = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
    for hid, invs in ordered:
        label = _bucket_label(buckets, hid)
        n = len(invs)
        suffix = f" — median of {n} invocation(s)" if n > 1 else " — 1 invocation"
        print(f"callable hid={hid} ({label}){suffix}", file=stream)
        _print_agg_tree(invs, stream=stream)
        print(file=stream)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="path to a host/CANN log containing [STRACE] lines (or '-' for stdin)")
    ap.add_argument(
        "--trace-out", help="write a Chrome-trace/Perfetto JSON here (load in chrome://tracing or perfetto)"
    )
    ap.add_argument(
        "--swimlane",
        help="write a real-pid/tid L3/L4 host swimlane JSON here (load in chrome://tracing or perfetto)",
    )
    ap.add_argument(
        "--rounds-table",
        action="store_true",
        help="print a per-round Host/Device/Orch/Sched table (the format tools/benchmark_rounds.sh "
        "parses) instead of the per-callable TPOT table",
    )
    ap.add_argument(
        "--tree",
        action="store_true",
        help="print an indented nested span tree per callable (device_wall → sub-phases), "
        "instead of the per-callable TPOT table",
    )
    ap.add_argument(
        "--assert-native-overlap",
        action="store_true",
        help="fail unless adjacent dispatches prove prepare(N+1) overlaps device(N) and preserve ordered launch",
    )
    ap.add_argument(
        "--require-hidden-prepare",
        action="store_true",
        help="with --assert-native-overlap, also require prepare(N+1) to finish inside device(N) "
        "(fully hidden, not merely overlapping — sensitive to host scheduling)",
    )
    args = ap.parse_args(argv)

    if args.log == "-":
        lines = sys.stdin.readlines()
    else:
        with open(args.log, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

    spans = list(parse_spans(lines))
    heads = count_record_heads(lines)
    if heads > len(spans):
        print(
            f"warning: {heads - len(spans)} of {heads} [STRACE] records are incomplete and are "
            "excluded from the timing below",
            file=sys.stderr,
        )
    legacy = legacy_spans(spans)
    invocations = group_invocations(legacy)
    buckets = bucket_by_hid(invocations)

    if args.assert_native_overlap:
        try:
            checks = assert_native_overlap(spans, require_hidden=args.require_hidden_prepare)
        except NativeOverlapError as exc:
            print(f"native overlap assertion failed: {exc}", file=sys.stderr)
            return 2
        print(f"Native overlap verified for {len(checks)} adjacent dispatch pair(s).")
    elif args.rounds_table:
        print_rounds_table(buckets)
    elif args.tree:
        print_tree(buckets)
    else:
        print_tpot_table(buckets)

    if args.trace_out:
        with open(args.trace_out, "w", encoding="utf-8") as f:
            json.dump(to_chrome_trace(invocations, buckets), f)
        print(f"Wrote Chrome trace: {args.trace_out} ({len(legacy)} spans)")

    if args.swimlane:
        with open(args.swimlane, "w", encoding="utf-8") as f:
            json.dump(to_host_swimlane(spans), f)
        host_count = sum(not span.is_device for span in spans)
        print(
            f"Wrote host swimlane: {args.swimlane} "
            f"({host_count} host spans, {len(spans) - host_count} unaligned device spans)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
