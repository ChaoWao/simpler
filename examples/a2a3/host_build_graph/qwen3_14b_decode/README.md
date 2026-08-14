# Qwen3-14B 40-layer decode with Graph Execution

This scene runs the complete Qwen3-14B decoder stack with the
`host_build_graph` runtime. It reuses the kernels, fixture, and golden from the
[`tensormap_and_ringbuffer` case](../../tensormap_and_ringbuffer/qwen3_14b_decode/README.md),
but records one decoder layer as a Graph and replays it for the remaining 39.

## Graph shape

The invocation contains a short ordinary prefix, 40 Graph submissions, and a
short ordinary suffix:

```text
paged-attention tiling + input preparation
  -> layer 0: record decoder DAG off the ring, submit one GRAPH task
  -> layers 1..39: reuse the cached Definition, submit one GRAPH task each
  -> copy final hidden state to output
```

Every replay updates the current layer's weights, KV cache, hidden state, and
scratch addresses while preserving the recorded boundary contract. See the
[`Graph Execution` documentation](../../../../src/a2a3/runtime/host_build_graph/docs/GRAPH_EXECUTION.md)
for that contract.

## Temporary storage

The orchestration allocates Graph boundary storage before recording:

- two hidden-state and normalized-state slots are used as a ping-pong pair;
- one BF16 and one FP32 scratch arena are shared by every layer;
- the decoder dependency chain and shared inout workspace serialize Graph
  execution, so a later layer cannot overwrite storage still used by an
  earlier layer.

This keeps the temporary live set flat in layer count and fits the default ring
configuration.

## Run

```bash
.claude/skills/onboard-arch-precheck/check.sh a2a3 || exit 1
task-submit --device auto --device-num 1 --run \
  ".venv/bin/python -m pytest \
  examples/a2a3/host_build_graph/qwen3_14b_decode \
  --platform a2a3 --device \$TASK_DEVICE --manual include"
```

The case runs in the daily full scene-test sweep and is excluded from per-PR
CI.
