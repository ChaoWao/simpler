# sdma_async_completion_demo — deferred completion over SDMA

Two ranks, one transfer, one dependency:

```text
producer:  TGET_ASYNC the peer rank's input from the HCCL window into local
           `out`, then register the PTO AsyncEvent via defer_pto_async_event
consumer:  depends on the producer's output, writes result = out + 1
```

Checking `out` and `result` tests two separate things: `out` proves SDMA
completion polling saw the transfer land, `result` proves the deferred-release
dependency held the consumer until it had. A consumer that ran early would
still produce a plausible `result` from a partially written `out`, which is why
both are checked.

The remote address is plain symmetric-window arithmetic — take the local
pointer's offset from `windowsIn[rankId]` and add it to `windowsIn[peer_rank]`.
Every rank's window is laid out identically, so an offset is rank-independent.

## Gated behind an overlay that is off by default

Unlike its a2a3 namesake, the a5 demo needs the PTO async-SDMA workspace
compiled into the host runtime:

| Gate | Effect |
| ---- | ------ |
| `@pytest.mark.platforms(["a5"])` | deselected on any other `--platform` |
| `@pytest.mark.device_count(2)` | needs two dies |
| `@pytest.mark.skipif(...)` | skipped unless `SIMPLER_ENABLE_PTO_SDMA_WORKSPACE` is `1` / `ON` / `TRUE` / `YES` |

The CMake option defaults `OFF`, so a stock build skips this test even on a5
hardware — **a green CI run says nothing about SDMA completion on a5.**

```bash
SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=ON pip install --no-build-isolation -e .
SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=ON \
  pytest examples/a5/tensormap_and_ringbuffer/sdma_async_completion_demo \
  --platform a5 --device 0-1
```

The variable is read twice: `simpler_setup/runtime_builder.py` forwards it to
CMake so the overlay is compiled in, and the test reads it from the environment
to decide whether to skip.

Wrap the hardware run in `task-submit` on a shared box.

## Compare with

- [`../urma_deferred_completion_demo/`](../urma_deferred_completion_demo/) — the same protocol over URMA. `kernel_consumer.cpp` is byte-identical; only the transfer kernel, its completion header, and the build flag differ. **The two overlays are mutually exclusive in one build**, so comparing them means rebuilding — that README has the detail.
- [`examples/a2a3/tensormap_and_ringbuffer/sdma_async_completion_demo/`](../../../a2a3/tensormap_and_ringbuffer/sdma_async_completion_demo/) — the a2a3 port of this demo, which needs no overlay flag.
- [`docs/a5-sdma-overlay.md`](../../../../docs/a5-sdma-overlay.md) — why the overlays are gated off and the checklist for re-enabling them.
