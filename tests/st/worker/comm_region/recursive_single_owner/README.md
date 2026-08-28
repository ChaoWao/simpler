# Recursive single-owner scene tests

Cross-architecture private ST for the delegated-region path. This directory
is not an example, does not add public `create_region()`, and does not live
under architecture-specific `comm_region` or `worker/comm_domain` trees.

| Test | Topologies | Platforms |
| ---- | ---------- | --------- |
| [`test_l3_single_hop.py`](test_l3_single_hop.py) | L3 → L2 | `a2a3sim`, `a2a3`, `a5sim`, `a5` |
| [`test_l4_two_hop.py`](test_l4_two_hop.py) | L4 → L3 → L2 | `a2a3sim`, `a2a3`, `a5sim`, `a5` |
| [`test_fault_injection.py`](test_fault_injection.py) | L3/L4 fatal edges | `a2a3sim`, `a5sim` |

Each L3 success case uses `Orchestrator.create_worker_chip_region`. Each L4
success case uses `Worker._materialize_region_instance`. Both enter the same
DRCT v1 materializer, which is the only region control path in the tree:
control commands 16 and 17 are retired and their numbers stay unassigned.

## Contract under test

- PAYLOAD and COUNTER are independent allocations; host import/close ordering is
  `import/import/close/close/release`.
- Host payload write/read and counter notify/wait.
- Provider-local views reach a real AICPU task through existing TaskArgs
  descriptor scalars.
- Release/backend cleanup is one-shot.
- The same Worker session runs two complete lifecycles; transaction IDs are
  monotonic and clean retirement leaves only the allocator position.

A2/A3 onboard covers the cache-maintenance call path. A5 is CI-only. Fault
cases stay on sim so they can inject reply loss, malformed replies, timeout,
child death, and fatal tree teardown without sharing a physical device.

## Run

```bash
source .venv/bin/activate
python -m pytest tests/st/worker/comm_region/recursive_single_owner --platform a2a3sim
```
