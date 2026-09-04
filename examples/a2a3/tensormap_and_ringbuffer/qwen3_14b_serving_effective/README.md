# Qwen3-14B Standalone TMR Decode

This example runs the post-prefill Qwen3-14B decode workload directly through
Simpler's `tensormap_and_ringbuffer` runtime. It never imports or starts
Serving. A fixture contains the KV snapshot and golden metadata produced by
one prefill; each run restores that snapshot and executes the consumed 127
decode dispatches. The `single` and `dual` modes are separate Python entry
points so the dual-slot pipeline cannot change the single-slot execution
layout.

## Running

`run_standalone_0p1.sh` is environment-driven so it can be used with a local
checkout, a CI worker, or a `task-submit` command. Supply the device and
output directory as positional arguments, and select `single` or `dual` as the
optional third argument:

```bash
PYTHON_BIN=/path/to/python \
PYPTO_ROOT=/path/to/pypto \
PYPTO_LIB_ROOT=/path/to/pypto-lib \
PTOAS_ROOT=/path/to/ptoas \
FIXTURE_ROOT=/path/to/fixture/qualification \
MODEL_DIR=/path/to/Qwen3-14B \
ARTIFACT_SOURCE=/path/to/artifact/qualification \
bash run_standalone_0p1.sh DEVICE_ID OUTPUT_DIR [single|dual] [STEPS]
```

The default is a zero-warmup, one-measured-run qualification. `STEPS` defaults
to 127 and may be reduced for a fast validation run. The caller owns output
and input locations; no private absolute paths are embedded in this case.

Required environment variables are `PYTHON_BIN`, `PYPTO_ROOT`,
`PYPTO_LIB_ROOT`, `PTOAS_ROOT`, `FIXTURE_ROOT`, `MODEL_DIR`, and
`ARTIFACT_SOURCE`. No `pypto-serving` path is required at runtime, and the
large KV/artifact payloads are deliberately kept outside the source repository.

## Reproducibility

`stack_manifest.json` records the exact software and input contract used to
produce the reference fixture and artifact. The reference stack is Simpler
`56511f80`, PyPTO `f1a2f35f`, pypto-lib `7f8d7ea8`, Python 3.10, torch/torch-npu
2.6, CANN 9.0 and PTOAS 0.57. Reproduce those versions when comparing against
the frozen numbers; regenerate the artifact when using a different PyPTO or
pypto-lib ABI.

The case fixes batch size 16, 127 consumed decode dispatches, steady skip 5,
ring task/dependency capacity 131072, and ring heaps 1/1/1/8 GiB. Every output
token is checked against the fixture golden before the run succeeds.

Dual mode keeps immutable weights, RoPE and the autoregressive KV cache shared.
It uses separate metadata, logits and next-hidden buffers for the two in-flight
frames. The dual validator requires strict native slot 0/1 alternation,
independent per-slot generations, prepared dispatches after the first frame,
serialized device execution and an exact full-output golden SHA.

## Metrics

- `effective_ms`: `chip.run.runner_run.device_wall.sched` for each decode
  dispatch. The steady set contains 122 rows after skipping the first five.
- `rts_completion_interval_ms`: consecutive `chip.run` completion timestamps
  within the decode sequence. This is the standalone decode TPOT proxy, not a
  full Serving request TPOT.
- `runner_run_ms`: host child span around the runtime call.

`trace_summary.json` contains aggregate statistics. The native Simpler STRACE
log is exported to `profile/simpler_strace_timeline.json`, which can be opened
by Perfetto.

## Correctness Contract

A successful run requires:

- 127 decode dispatches on the selected slot(s);
- monotonically increasing generation and completion order;
- every sampled token matching the 127-step golden;
- the first decode input matching the fixture's prefill token;
- fixture, model and artifact checksum validation before device execution;
- native device STRACE phases present and alignable to their host invocation.

The scripts are a hardware benchmark entry point rather than a default CI
scene test. Preserve the generated `benchmark.json`, `trace_summary.json`,
timeline and `SHA256SUMS` files with each result.
