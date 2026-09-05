# A2/A3 worker retirement: CLOSE must precede return

The A2/A3 persistent worker must not return after merely acknowledging EXIT.
It must wait until the AICPU has closed the group's fast-path register windows
and explicitly released the workers through global memory (GM).

This applies to both `tensormap_and_ringbuffer` and `host_build_graph`.
It does not add borrowed-stream/L1 support or change the A5 exit protocol.

## What fast path means here

The AICPU accesses each AICore's register window through MMIO. It writes
`DATA_MAIN_BASE` to dispatch work and reads `COND` for ACK/FIN status. The
AICore reads its own dispatch SPR and writes its own COND SPR.
`FAST_PATH_ENABLE` (offset `0x18`) is written with `0xE` to open and `0xF`
to close the window. This is a device-side dispatch mechanism, not a torch
allocator setting, host task-queue option, or ACL event.

## The missing ordering edge

The old protocol established these two chains:

```text
AICore: observe EXIT -> write COND=EXITED -> return
AICPU:                 observe EXITED   -> write IDLE -> CLOSE
```

They have a common predecessor, not an ordering between their tails. The
following execution was therefore legal in the software protocol:

1. AICore observes EXIT and publishes EXITED.
2. AICore returns while its fast-path window is still open.
3. AICPU observes EXITED, writes IDLE, and closes the window.

EXITED proves that the worker has stopped accepting tasks. It cannot also
prove that the AICPU has finished managing the window: that work occurs
*after* the ACK. The missing edge is `CLOSE -> worker return`.

A host mutex, an event between streams, or waiting for an entire launch to
complete cannot retroactively order these two operations inside that launch.
Multiple in-flight graph executions are not required for this race. Nor does
an asynchronous error reported at a later native operator prove that the
native operator caused it.

## Two-phase retirement

Normal shutdown uses the existing all-AICPU-thread completion rendezvous:
the last participant retires the entire worker group. Retirement proceeds in
separate passes:

1. Send EXIT to every initialized core before waiting for any ACK.
2. Collect EXITED from the group, using one shared deadline.
3. Reset dispatch to IDLE and close every acknowledged window. Read back the
   MMIO window and complete that read before publishing a GM return gate.
4. Release-store `AICORE_POST_CLOSE_RELEASE` to each closed core's control word.
5. AICore observes that word and only then returns.

The A2/A3 onboard worker uses `ld_dev` for an uncached/bypass read and
`dsb(DSB_DDR)` before return. The simulation uses an atomic acquire load.
`ld_dev` is not being treated as a general-purpose atomic RMW or as a C++
acquire operation; this is a single-writer, per-core handoff. The platform's
MMIO completion ordering and visibility of the GM publication are distinct
parts of the protocol. A release store alone does not flush posted MMIO.

Emergency shutdown uses the same retirement helper and an ownership latch,
so normal finalization does not subsequently write to already released
workers' registers. A core that misses the deadline is not closed or released;
responsive peers still retire. The existing host recovery path remains
responsible for unresponsive cores and cores whose windows never opened.
This patch does not introduce a new reset operation or recovery policy.

## Cache-line and generation ownership

`Handshake` contains two separately aligned 64-byte lines:

| Line | Writer/access | Purpose |
| ---- | ------------- | ------- |
| First | Existing cached report/task publication protocol | Startup identity, ready report, dispatch-payload pointer |
| Second | AICPU atomic stores; AICore bypass reads | Post-close permission to return |

The return gate must not share the line flushed by the AICore's startup or
exit report. Otherwise a stale cached report writeback could overwrite the
AICPU's release. No cached writes or cache maintenance may target the gate
line while it is active. Compile-time layout and trivial-copy assertions
guard the shared host/AICPU/AICore image.

The boot leader atomically resets the gate before publishing handshake setup
and before any register window opens. Thus the preceding launch's value of
1 cannot release a new launch. Reuse assumes the existing runtime lifecycle:
the previous launch has completed before the same storage is re-armed. This
is not a claim that one runtime image supports concurrent independent runs.

HBG shares its `Handshake` definition with A5, so that internal image also
grows from 64 to 128 bytes per worker on A5; the second line remains unused
there. Host and device binaries must be rebuilt together. The public Python
API and the A5 execution protocol are unchanged.

## Evidence and reproduction boundaries

The executor regression links each **production** A2/A3 AICore execution loop
against simulated register storage. With CLOSE deliberately withheld, all
four AIC/AIV checks fail on the unmodified main implementation at
`fab1a41e2fd5bbefb9eb18e59609876e63297e98`: the worker returns immediately.
With the handoff, those checks pass. Companion tests check eventual return,
32 launches reusing the same gate, group ACK ordering, partial timeout, and
invalid-target rejection.

```bash
cmake -S tests/ut/cpp -B tests/ut/cpp/build
cmake --build tests/ut/cpp/build --parallel 2 \
  --target test_a2a3_tensormap_and_ringbuffer_retirement test_a2a3_host_build_graph_retirement
ctest --test-dir tests/ut/cpp/build -R '^test_a2a3_.*_retirement$' --output-on-failure
```

These are software-ordering tests; they do not emulate silicon register
retirement hazards. The motivating downstream L1 ACLGraph investigation
separately found that removing only the post-close wait changed a passing
1280-replay run into a matching vector timeout at replay 2. The fault PC was
at the persistent AIV wrapper's end, not inside the later native operator.
That is downstream integration evidence, not a claim that upstream main
already supports or reproduces that L1 graph workload. No undocumented
internal hardware state-machine failure is asserted here.

The PR's validation record distinguishes new upstream tests from this
historical downstream evidence and lists any unavailable toolchains.
