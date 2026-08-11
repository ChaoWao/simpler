# Compute then TLOAD mixed L3 example

This example keeps one L4/L3/L2 process tree alive across two ordered task
rounds — a local compute round, then a cross-machine communication round
through the same Global CommDomain:

```text
L4 parent on machine A -> local L3 on machine A  -> local NPU, rank 0
                       -> remote L3 on machine B -> remote NPU, rank 1

round 1 (compute):        each rank runs input = lhs + rhs on its own L2
round 2 (communication):  each rank peer-TLOADs every rank's input, sums, stores result
```

Both rounds run in the same Worker lifetime and the same committed domain. The
parent verifies the compute outputs before issuing any communication task —
`Worker.run()` drains the DAG before returning, so the round boundary is also
the completion barrier — then verifies the communication outputs. Compute and
communication are checked separately, so a communication failure cannot hide
behind a correct compute result or vice versa.

`global_tload_mixed_l3` covers the domain build and the peer `TLOAD` alone;
this example exists for the phase ordering — device results produced by one
round feeding the next round's cross-machine reads.

## Prepare both machines

Use the same source revision on machine A and machine B:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --no-build-isolation -e .
```

## Start the remote L3 daemon

Start this on machine B. `192.0.2.20` is a documentation placeholder, not a
usable address:

```bash
source .venv/bin/activate
python -m simpler.remote_l3_worker --host 192.0.2.20 --port 19073
```

## Run on the L4 parent

Run this on machine A. Only the first id of each device list is used — each L3
owns exactly one L2:

```bash
source .venv/bin/activate
export SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_REMOTE=192.0.2.20:19073
export SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_LOCAL_DEVICES=0
export SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_REMOTE_DEVICES=0
bash examples/workers/l4/compute_then_tload_mixed_l3/run_parent.sh
```

Expected output ends with:

```text
[compute-then-tload-mixed-l3] compute local rank=0 max_diff=0.000e+00
[compute-then-tload-mixed-l3] compute remote rank=1 max_diff=0.000e+00
[compute-then-tload-mixed-l3] communication local rank=0 max_diff=0.000e+00
[compute-then-tload-mixed-l3] communication remote rank=1 max_diff=0.000e+00
compute_then_tload_mixed_l3 passed
```

Any `max_diff` above its tolerance exits non-zero.

## Configuration

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_REMOTE` | (required) | Remote daemon endpoint, `HOST:PORT` |
| `SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_LOCAL_DEVICES` | `0` | Device list; the first id is the local L3's L2 |
| `SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_REMOTE_DEVICES` | `0` | Device list; the first id is the remote L3's L2 |
| `SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_PLATFORM` | `a2a3` | Target platform |
| `SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_RUNTIME` | `tensormap_and_ringbuffer` | Runtime |
| `SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_COMM_PROFILE` | `a3-fabric-v1` | Global CommDomain backend profile |
| `SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_SESSION_LISTEN_HOST` | `0.0.0.0` | Interface the parent's session runner binds |
| `SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_SESSION_TIMEOUT` | `120` | Seconds to wait on the remote session |

The peer `TLOAD` needs real cross-device windows, so the default profile is
`a3-fabric-v1` and requires real A3 devices on both machines. In CI the pod
job runs the `test_compute_then_tload_mixed_l3.py` wrapper through
`pod-run-pytest`. CI supplies `POD_REMOTE_ENDPOINT`, `POD_REMOTE_DEVICES`,
`POD_L3_SESSION_TIMEOUT_S`, `POD_L3_SESSION_LISTEN_HOST`, and pytest's
`--platform` / `--device` options instead of the `SIMPLER_*` variables above;
`run_parent.sh` remains the manual entry point.
