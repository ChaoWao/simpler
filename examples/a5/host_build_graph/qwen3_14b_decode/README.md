# Qwen3-14B 40-layer decode with Graph Execution

This A5 scene migrates the complete Qwen3-14B decoder from the existing
[`tensormap_and_ringbuffer` case](../../tensormap_and_ringbuffer/qwen3_14b_decode/README.md)
to `host_build_graph`. It directly reuses that case's 37 incore sources, input
generator, and golden implementation.

The orchestration submits a short preparation prefix, records the first decoder
layer as a Graph, replays the cached definition for layers 1 through 39, and
then copies the final hidden state to the output. Each replay binds the current
layer's weights, KV-cache view, hidden state, and scratch addresses.

Two hidden/normalized-state pairs provide ping-pong storage. All layers share
one BF16 and one FP32 scratch arena; Graph dependencies serialize their use, so
temporary memory stays constant as the layer count grows.

Run on A5 hardware through the shared-device queue:

```bash
.claude/skills/onboard-arch-precheck/check.sh a5 || exit 1
task-submit --device auto --device-num 1 --run \
  ".venv/bin/python -m pytest \
  examples/a5/host_build_graph/qwen3_14b_decode \
  --platform a5 --device \$TASK_DEVICE --manual include"
```

The case runs in the daily full scene-test sweep and is excluded from per-PR
CI.
