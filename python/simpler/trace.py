# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Put a caller's own spans on the host timeline.

One producer, one span, one gate — the whole surface:

```python
from simpler import trace

with trace.span("my_phase", batch=16, layer=3):
    ...

@trace.span("my_func")
def f(): ...

with trace.span("checkpoint", token=17):   # a marker is a span of no work
    pass

if trace.enabled():                        # only then build costly attributes
    with trace.span("expensive", **compute_attrs()):
        ...

tracer = trace.producer("pypto")           # a library names itself
```

There is no separate instant/marker call: a zero-duration event would be a second
event concept, and `dur=0` already means "this phase was never stamped" on the
emitting side of our own spans. A marker is a span whose body does no work, which
records a real short interval instead of a value that means something else.

Three properties hold by construction rather than by documentation, which is why
this surface exists at all next to the seven-argument `_emit_host_span`:

- **One clock.** A caller never handles a timestamp; the wrapper reads
  `_monotonic_now_ns`, the same clock every host record is stamped with.
- **One namespace.** Every name emitted here is prefixed `ext.<producer>.`, so a
  caller cannot land a span in one of ours whatever it passes.
- **One gate.** `enabled()` is the runtime host-span query the C++ emit sites
  read, not a second notion of "on".

This is not a side channel for callers. `[STRACE]` is the host's one timeline
format, and a caller's interval is a span for the same reason a runtime's own
bind segment is; what the reserved namespace separates is *whose* span it is, not
which format it uses.

See [docs/dfx/host-trace.md](../../docs/dfx/host-trace.md) for the record grammar
and what the `ext.` namespace guarantees in each view.
"""

import functools
import inspect
import re
from typing import Any, Callable, Optional

from _task_interface import (  # pyright: ignore[reportMissingImports]
    _emit_host_span,
    _host_spans_active,
    _monotonic_now_ns,
)

__all__ = ["Producer", "enabled", "producer", "span"]

# The leading word reserved for producers outside simpler: `ext.<producer>.<span>`.
# `simpler_setup.tools.strace_timing.span_family` reads it.
_RESERVED_WORD = "ext"

# A producer segment is one dot-free word, because attribution splits the name on
# dots and reads the second segment. Everything else is percent-encoded by the
# record writer rather than rejected, so this is the only spelling constraint.
_PRODUCER_PATTERN = re.compile(r"\A[A-Za-z0-9_-]+\Z")

# The producer a caller that names none writes under. Deliberately a constant
# rather than a name derived from the running program: every derivation is a
# heuristic with its own special cases, and one call to `producer()` names an
# application better than any of them.
_UNNAMED_APPLICATION = "app"


def _emit(name: str, start_ns: int, duration_ns: int, attributes: dict) -> None:
    """Write one span record.

    `inv` and `hid` are our correlation keys — a native run epoch and a callable
    digest — and `depth` is our scheduler's nesting counter. An external span
    reaches no view that reads them, and a caller has no value to put in them, so
    all three are 0.
    """
    payload = " ".join(f"{key}={value}" for key, value in attributes.items())
    _emit_host_span(name, 0, 0, 0, start_ns, duration_ns, payload)


class Span:
    """One named region of a caller's code, times itself, emits on exit.

    Both uses re-read the gate at the moment they run, never at the moment the
    span was created: a `with` block on `__enter__`, a decorated call on every
    call. A module decorated while the threshold was still at its default
    therefore starts producing spans as soon as a worker seeds a lower one.

    A decorated `async def` gets a coroutine wrapper, so its span covers the
    body's execution rather than the coroutine's creation.

    One instance times one scope, since it holds the one start timestamp:
    entering the same instance twice before its first exit mispairs the two.
    `span()` returns a fresh instance per call, and the decorator times each call
    without touching the instance, so only a stored-and-shared instance can reach
    that.
    """

    __slots__ = ("_attributes", "_name", "_start_ns")

    def __init__(self, name: str, attributes: dict) -> None:
        self._name = name
        self._attributes = attributes
        self._start_ns: Optional[int] = None

    def __enter__(self) -> "Span":
        self._start_ns = _monotonic_now_ns() if _host_spans_active() else None
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        start_ns = self._start_ns
        self._start_ns = None
        if start_ns is not None:
            _emit(self._name, start_ns, _monotonic_now_ns() - start_ns, self._attributes)
        return False

    def __call__(self, function: Callable[..., Any]) -> Callable[..., Any]:
        name = self._name
        attributes = self._attributes

        if inspect.iscoroutinefunction(function):
            # A sync wrapper around a coroutine function would time the coroutine's
            # *creation* — a few hundred nanoseconds — and close the span before the
            # body ran at all. The choice is made here, at decoration, so the call
            # path carries no extra test.
            @functools.wraps(function)
            async def async_wrapper(*args, **kwargs):
                if not _host_spans_active():
                    return await function(*args, **kwargs)
                start_ns = _monotonic_now_ns()
                try:
                    return await function(*args, **kwargs)
                finally:
                    _emit(name, start_ns, _monotonic_now_ns() - start_ns, attributes)

            return async_wrapper

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if not _host_spans_active():
                return function(*args, **kwargs)
            start_ns = _monotonic_now_ns()
            try:
                return function(*args, **kwargs)
            finally:
                _emit(name, start_ns, _monotonic_now_ns() - start_ns, attributes)

        return wrapper

    def __repr__(self) -> str:
        return f"Span({self._name!r})"


class Producer:
    """One named external producer, owning the `ext.<name>.` segment of the namespace."""

    __slots__ = ("_prefix", "name")

    def __init__(self, name: str) -> None:
        if not isinstance(name, str) or not _PRODUCER_PATTERN.match(name):
            raise ValueError(
                f"producer name must be one word of letters, digits, '_' or '-' — got {name!r}. "
                "Attribution splits ext.<producer>.<span> on dots, so a dotted name names no producer."
            )
        self.name = name
        self._prefix = f"{_RESERVED_WORD}.{name}."

    def span(self, name: str, **attributes: Any) -> Span:
        """A span named `ext.<producer>.<name>`, for use as a context manager or a decorator.

        The name is refused when empty whether or not the gate is open, so an
        unnamed span — `ext.<producer>.`, which attributes to nobody — fails the
        same way in a run with spans off as in one with them on.
        """
        if not name:
            raise ValueError("span name must be non-empty: ext.<producer>. names a producer but no span")
        return Span(self._prefix + name, attributes)

    def enabled(self) -> bool:
        """Whether this process is emitting host spans right now.

        Worth asking before building attributes that cost more than the span. Every
        producer asks the same gate: this is the runtime query the C++ emit sites
        read, and it is false in a build with host spans compiled out, so there is
        one answer rather than two.
        """
        return _host_spans_active()

    def __repr__(self) -> str:
        return f"Producer({self.name!r})"


def producer(name: str) -> Producer:
    """A producer that names itself, for a library instrumenting its own code.

    A library takes one of these rather than the module-level `span`, whose
    producer is the unnamed default shared by whatever else in the process did not
    name itself either.
    """
    return Producer(name)


# The module-level surface is one producer's own methods, the shape `random`
# exposes a hidden `Random` instance's: one call layer rather than a wrapper per
# function, and no second implementation to drift from this one.
_APPLICATION = Producer(_UNNAMED_APPLICATION)

span = _APPLICATION.span
enabled = _APPLICATION.enabled
