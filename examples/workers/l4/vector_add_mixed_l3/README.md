# Vector add mixed L3 example

This example validates the mixed-L3 vector-add path for an L4 parent that combines one
forked local L3 worker and one TCP-connected remote L3 worker:

```text
L4 parent on machine A -> local L3 on machine A  -> local NPU 0 + NPU 1 vector group
                       -> remote L3 on machine B -> remote NPU 0 + NPU 1 vector group
```

The parent stages local inputs through owner Buffers it allocates with
`create_buffer` — the forked local L3 maps each one lazily from the descriptor
embedded in the tensor — and remote inputs through `remote_malloc` /
`remote_copy_to`. It dispatches both L3
tasks in one L4 run, downloads the remote outputs with `remote_copy_from`, and
checks the golden result on both sides. No `mpirun` is used.

This baseline proves remote startup, remote buffers, `RemoteTensorRef`, L4 task
dispatch to mixed local/remote L3 workers, local L3 group scheduling, NPU
execution, and result copy-back. It does not read peer-machine memory or
exercise Global CommDomain; those stay out of this mixed-L3 vector-add path.

## Prepare both machines

Use the same source revision on machine A and machine B:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --no-build-isolation -e .

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PATH="$ASCEND_HOME_PATH/bin:$PATH"
```

## Start the remote L3 daemon

Start this on machine B:

The daemon is the generic session server — nothing about it is specific to this
example, so it is started directly rather than through a wrapper:

`PEER_HOST` is machine B's address as machine A reaches it — `192.0.2.20` below
is a documentation placeholder, not a usable address.

```bash
export ASCEND_PROCESS_LOG_PATH=/tmp/simpler-vector-add-mixed-l3-machine-b
source .venv/bin/activate

PEER_HOST=192.0.2.20            # machine B's address
python -m simpler.remote_l3_worker --host "$PEER_HOST" --port 19073
```

The daemon starts a session runner on random TCP command and health ports.
Firewalls between the parent and remote machine must allow those returned ports
in addition to the daemon port.

## Run on the L4 parent

Run this on machine A. The local device ids are owned by the forked local L3
on machine A; the remote device ids are owned by the daemon-started L3 on
machine B.

```bash
export ASCEND_PROCESS_LOG_PATH=/tmp/simpler-vector-add-mixed-l3-machine-a
PEER_HOST=192.0.2.20            # the same machine B address used above
export SIMPLER_VECTOR_ADD_MIXED_L3_REMOTE="$PEER_HOST:19073"
export SIMPLER_VECTOR_ADD_MIXED_L3_LOCAL_DEVICES=0,1
export SIMPLER_VECTOR_ADD_MIXED_L3_REMOTE_DEVICES=0,1

bash examples/workers/l4/vector_add_mixed_l3/run_parent.sh
```

Each side must provide exactly two free A3 device ids. Success requires both
local outputs and both remote outputs to match the golden result.
