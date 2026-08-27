# urma_deferred_completion_demo — the same protocol, over URMA

The a5-only twin of
[`../sdma_async_completion_demo/`](../sdma_async_completion_demo/). Both run the
identical two-rank protocol; they differ in which transport moves the bytes and
which completion path reports it done.

```text
producer:  TGET_ASYNC the peer's input from the window into local `out`,
           register the AsyncEvent through the deferred-completion path
consumer:  depends on that producer output, writes result = out + 1
```

Checking both `out` and `result` is what makes it a test of two things at once:
`out` proves completion polling saw the transfer land, and `result` proves the
deferred-release dependency held the consumer back until it had.

## What actually differs from the SDMA demo

`kernels/aiv/kernel_consumer.cpp` is **byte-identical** between the two
directories. Only the transfer kernel and the orchestration change:

| What | URMA | SDMA |
| ---- | ---- | ---- |
| Transfer kernel | `kernel_urma_tget_async.cpp` | `kernel_sdma_tget_async.cpp` |
| Completion header | `backend/urma/urma_completion_kernel.h` | `backend/sdma/sdma_completion_kernel.h` |
| Workspace field | `urmaWorkSpace` | `sdmaWorkSpace` |

Read them side by side and the transport is the only variable — which is
exactly what you want when deciding which one a workload should use.

## SDMA and URMA are both available

The A5 host runtime provisions both workspaces when it creates a communication
domain. `CommContext` keeps the SDMA workspace in the leading `sdmaWorkSpace`
pair and the URMA workspace in the appended `urmaWorkSpace` pair, so both
engines are usable from the same build and domain.

Provisioning is best-effort on both: a transport that fails to initialize
leaves its pair zero and the transfer kernel takes its `urmaWorkSpace == 0`
self-skip. This demo then **fails**, and loudly — `compute_golden` expects
`out_rank == in_{1-rank}`, so a skipped TGET leaves `out` all zeros against a
non-zero golden. A green run of this test is therefore real evidence that URMA
moved data. When it fails, check the host log for
`URMA workspace unavailable on at least one rank` to tell an absent transport
apart from a wrong one.

URMA metadata follows communicator-rank order, while each derived context
carries an explicit domain-rank-to-communicator-rank map. Subsets and reordered
domains therefore use the same communicator-scoped registration safely.

The A5 worker registers one per-rank arena for the communicator (200 MiB by
default, `Worker(comm_arena_size=…)` to change it, and the backend reports back
the size it granted) and carves ordinary dynamic domains from it. The first 256
bytes are reserved, so this demo's normal `allocate_domain` path performs a
real URMA TGET through a non-zero derived offset. Its domain worker order is
also reversed to `[1, 0]`, covering domain-rank-to-communicator-rank remapping,
peer MR selection, and `UrmaTget` in one transfer. Releasing a domain frees its
context and returns its slice; the HCCL registration and channels live until
communicator teardown.

The test has only the ordinary scene constraints:

| Gate | Effect |
| ---- | ------ |
| `CASES[*]["platforms"] = ["a5"]` | deselected on any other `--platform` |
| `CASES[*]["config"]["device_count"] = 2` | needs two dies |

## Run

```bash
pytest examples/a5/tensormap_and_ringbuffer/urma_deferred_completion_demo \
  --platform a5 --device 0-1
```

Wrap the hardware run in `task-submit` on a shared box.

## See also

[`../sdma_async_completion_demo/`](../sdma_async_completion_demo/) — the
SDMA variant of the same protocol. It can run immediately before this test
without rebuilding or changing the environment.
