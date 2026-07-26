# Host worker dispatch latency: where the remaining ~50 us goes

**Date**: 2026-07-26
**Verdict**: dropped — the cost is fixed per `run()`, not per task, and
amortizes to 2% at realistic graph sizes. Also closes the case for replacing
the mailbox poll with a blocking wakeup primitive: the poll is not where the
latency is.

## Question

[#1499](https://github.com/hw-native-sys/simpler/pull/1499) removed the parent's
50 us `TASK_DONE` sleep and took a single `Worker.run()` from 150 us to ~51 us.
The obvious follow-up: is there another factor of three in the remaining 51 us,
and is a blocking wakeup primitive ([#1498](https://github.com/hw-native-sys/simpler/issues/1498))
the way to get it?

## What was tried

Four measurements on aarch64 Linux, 1 sub worker, no-op callables, medians over
200-500 iterations.

**1. The IPC floor.** `tests/ut/py/test_hostsub_fork_shm.py::_SubWorkerPool` is a
minimal fork + shared-memory mailbox with no simpler runtime in it at all, so a
round trip through it is what the mechanism costs before any of our code runs.

**2. A full `Worker.run()`** with one sub task, for the difference.

**3. `cProfile` over 500 dispatches** to see the shape of the runtime's own work.

**4. Tasks per `run()` swept 1 → 256**, to separate fixed from marginal cost.
This is the measurement that decides the question; the first three only frame it.

## Result

```text
bare mailbox round trip (fork + shm, no runtime):    3.5 us
Worker.run(), one sub task:                         53.8 us
  => runtime overhead per dispatch:                 50.3 us  (94%)
```

94% sitting in our code rather than the IPC looks damning, and taken alone it
argues for optimising the runtime and against #1498. The sweep shows why that
reading is wrong:

| tasks per `run()` | total | per task |
| ----------------- | ----- | -------- |
| 1 | 51.4 us | 51.38 us |
| 2 | 62.7 us | 31.35 us |
| 4 | 80.2 us | 20.04 us |
| 8 | 114.4 us | 14.30 us |
| 16 | 177.0 us | 11.06 us |
| 64 | 538.7 us | 8.42 us |
| 256 | 2113.0 us | 8.25 us |

**Fixed: 43.3 us per `run()`. Marginal: 8.08 us per task.**

The 50 us is almost entirely the cost of *entering and leaving an
orchestration*, not of dispatching work. `cProfile` agrees on the shape: a
no-op dispatch runs ~113 Python calls and ~15 threading lock/notify operations,
spread thin across the submit → wait → drain plumbing with no single hot frame.

## Why not (now)

**The fixed cost amortizes to nothing at realistic sizes.** A production run
submits a graph per `run()`, not one task — at 256 tasks the fixed 43 us is 2%
of the invocation. Optimising it would pay off only for a workload that makes
many tiny `run()` calls, and we do not have one. Spending effort on the thin,
call-count-bound Python plumbing to win 43 us that a real graph already hides is
the wrong trade.

**The marginal cost is already close to the floor.** 8.08 us per task against a
3.5 us bare fork+shm round trip is ~2.3x, and that 2.3x buys arg marshalling,
callable-identity resolution and per-task error reporting that the bare pool
does not do.

**And it settles #1498 against itself.** A pipe wake measures **4.4 us one way**
on this box (8.85 us round trip, p99 9.62 us). Replacing the poll with a
blocking wait would therefore add more than the entire 3.5 us IPC floor to every
dispatch, in exchange for CPU, while 94% of the latency is somewhere else
entirely. The polling is not the problem, so removing the polling is not the
fix.

What survives from #1498 is narrower and unrelated to latency: a child spinning
while *nothing* is outstanding holds a core for as long as its parent lives.
That is a CPU-efficiency question for narrow-core hosts, and it shrank further
once [#1495](https://github.com/hw-native-sys/simpler/pull/1495) made orphans
exit with their parent.

## When to reconsider

- A workload appears that issues many small `run()` calls — deep L3/L4
  hierarchies, or per-token orchestration — where 43 us fixed stops amortizing.
  Measure that workload's tasks-per-`run()` first; if it is above ~16 the fixed
  cost is already under 25%.
- The marginal 8.08 us becomes the bottleneck for a SubWorker-heavy graph. Note
  this is the Python-callable path; NPU tasks do not go through it.
- Someone proposes a blocking primitive again for **latency** reasons. The
  numbers above say it costs latency. Only the CPU argument is open, and only
  in a spin-then-block hybrid that never reaches the blocking path while work
  is in flight.

## References

- [#1499](https://github.com/hw-native-sys/simpler/pull/1499) — removed the
  parent's dispatch-path sleep; the measurements here start from its result.
- [#1498](https://github.com/hw-native-sys/simpler/issues/1498) — blocking
  wakeup primitive; re-scoped to CPU-only on the strength of these numbers.
- [`.claude/rules/codestyle.md`](../../.claude/rules/codestyle.md) rule 5 — the
  dispatch path may spin or block, but never sleep.
