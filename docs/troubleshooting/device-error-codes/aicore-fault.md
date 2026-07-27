# Chasing down an AICore fault (UB out of bounds, VEC/CUBE instruction error)

← [Device Error Codes](../device-error-codes.md)

An AICore addressing fault reaches the host as the same generic `507018` a stall
does, usually with a `SCHEDULER_TIMEOUT` tail, because the faulting core stops
answering and a watchdog reaps the op afterwards. The device log is the only
place the fault itself appears:

```text
VEC instruction error: the ub address out of bounds
  core id 15, blk:1, errcode 0x4000000000000000
→ aicpu exception → 507018 → sched_error_code=100 SCHEDULER_TIMEOUT
```

**Redirect the device log before the run that you expect to fail.** The harness
does not do this for you: `outputs/<case>_<ts>/` is only created when a DFX flag
is on, so a plain onboard run leaves its device log in the shared
`~/ascend/log/debug/device-<id>/` among every other user's. A fault you cannot
attribute to a file is a fault you cannot diagnose, and the evidence is gone
once the run ends:

```bash
LOGDIR="$PWD/outputs/<label>/ascend"; mkdir -p "$LOGDIR"   # driver fopens; it will not mkdir
export ASCEND_PROCESS_LOG_PATH="$LOGDIR"
```

See the "Device logs" section of
[`running-onboard.md`](../../../.claude/rules/running-onboard.md).

## F1: is it the kernel's own addressing, or did the core jump somewhere else?

These two look identical from the host and have nothing in common as bugs, so
settle it before reading any kernel arithmetic:

- **`PrintErrorInfoForDavinciTask`** — `fault kernel_name`, `hash`, and
  `GetBinAndKernelNameExceptionArgs: binSize`. If `binSize` matches the
  runtime's own `aicore_kernel.o` rather than your kernel, the report is naming
  the **polling-dispatch executor**, and the faulting code is whatever it jumped
  to — a dispatch-payload / `function_bin_addr` problem, not kernel arithmetic.
- **`PrintCoreInfo`** — `pc current`. Whether that offset lands inside the
  executor's binary size or far outside it tells you the same thing
  independently.

[#1036](https://github.com/hw-native-sys/simpler/issues/1036) has the worked
example of making this distinction.

## F2: rule the kernel's own addressing in or out statically

Before instrumenting, check whether the kernel *can* compute an out-of-range UB
address at all. Compile-time `TASSIGN` offsets and template-derived tile extents
do not make that impossible — the constants themselves can overrun — but they
make it **decidable on paper**, because runtime values that only shrink a tile
(a per-rank chunk count, a shorter row) move no base and grow no extent.

So compute it. Sum the highest byte each `TASSIGN` base plus its tile extent
reaches and compare against the AIV UB size:

- **Fits for every input the guards admit** → the address did not come from
  here, and you are in F1's second case.
- **Does not fit** → you have found the bug without touching the device.

The allreduce collectives were cleared this way in
[#1489](https://github.com/hw-native-sys/simpler/issues/1489): all five modes
bind every tile to a constant base, `nranks` only divides the chunk count, and
the whole footprint stays under 132 KiB of the 192 KiB UB at every rank count.

That case is worth following to its end, because F2's verdict held. The fault
was not in the kernels; it was almost certainly
[#1477](https://github.com/hw-native-sys/simpler/pull/1477), a `CoreTracker`
whose `uint64_t` state overran past 21 clusters while those cases ran at the
device's full 24. Corrupted dispatch state, F1's second case — reached by
believing F2 rather than re-reading the kernel arithmetic a second time.

## F3: separate a real fault from a post-mortem register dump

A core killed while spinning can produce a fault-looking line that describes the
state it was reaped in rather than an instruction it executed. Count the
detectors, per [`running-onboard.md`](../../../.claude/rules/running-onboard.md):
if `FATAL: Task Allocator Deadlock` and `Timeout (N cycles):
producer/consumers` are both **zero** and only `HandleTaskTimeout` fired, then
nothing on the device proved a fault or a deadlock — treat the wait that never
completed as the primary object of the investigation, not the addressing line.
