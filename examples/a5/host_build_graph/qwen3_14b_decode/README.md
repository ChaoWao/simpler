# Qwen3-14B 40-layer decode with Graph Execution

This A5 scene migrates the complete Qwen3-14B decoder from the existing
[`tensormap_and_ringbuffer` case](../../tensormap_and_ringbuffer/qwen3_14b_decode/README.md)
to `host_build_graph`. Its `main.py` delegates to the TMR standalone driver and
changes only the runtime and orchestration source, reusing the 37 incore
sources, streaming fixture, and golden implementation.

The orchestration submits a short preparation prefix, records the first decoder
layer as a Graph, replays the cached definition for layers 1 through 39, and
then copies the final hidden state to the output. Each replay binds the current
layer's weights, KV-cache view, hidden state, and scratch addresses.

Two hidden/normalized-state pairs provide ping-pong storage. All layers share
one BF16 and one FP32 scratch arena; Graph dependencies serialize their use, so
temporary memory stays constant as the layer count grows.

Run on A5 hardware through the shared-device queue:

```bash
task-submit --device auto --device-num 1 --run \
  ".claude/skills/onboard-arch-precheck/check.sh a5 && \
   .venv/bin/python examples/a5/host_build_graph/qwen3_14b_decode/main.py \
   -p a5 -d \"\$TASK_DEVICE\" --rounds 100 --skip-golden"
```

The driver allocates all entry parameters once in device memory, streams one
valid fixture upload before the first round, and reuses the same addresses for
every Graph replay. `--skip-golden` skips only torch computation and explicit
D2H comparison. The thin pytest wrapper remains manual, so the case runs in the
daily full scene-test sweep and is excluded from per-PR CI.
