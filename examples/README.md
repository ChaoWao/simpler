# Examples

Two ways in, split by what you are trying to learn.

| Directory | Entry point | Read it to learn |
| --------- | ----------- | ---------------- |
| [`workers/`](workers/README.md) | the raw `Worker` class | **How the runtime works.** Every step is explicit — build a `ChipCallable` from `.cpp` sources, pack `TaskArgs`, compose chips and sub-workers into a DAG. Organized by level: [`l2/`](workers/l2/README.md) is one chip, [`l3/`](workers/l3/README.md) is a host driving several. |
| [`a2a3/`](a2a3/tensormap_and_ringbuffer/README.md), [`a5/`](a5/tensormap_and_ringbuffer/README.md) | the `@scene_test` framework | **How to write and validate a kernel.** Case parametrization, golden comparison, and pytest integration are handled for you. Organized by architecture, then runtime. |

Start with `workers/l2/hello_worker/` if the runtime is new to you; start with
your architecture's `vector_example/` if kernels are new to you.

## Running them

Everything under `examples/` is collected by pytest, and CI runs the whole tree
on both simulators:

```bash
pytest examples --platform a2a3sim
pytest examples -m "not sdma" --platform a2a3 --exclude-level 4 --device 0-1        # hardware (SDMA/network1 quarantined; network1 runs in the two-machine job)
```

A single example:

```bash
pytest examples/a2a3/tensormap_and_ringbuffer/vector_example --platform a2a3sim --manual include
```

Most `workers/` examples are also plain scripts, which is the quickest way to
read one end to end:

```bash
python examples/workers/l2/hello_worker/main.py -p a2a3sim -d 0
```

Each example declares the platforms it supports and the number of devices it
needs; pytest deselects the ones that do not apply to your `--platform`. On a
shared box, wrap hardware runs in `task-submit` — see
[`.claude/rules/running-onboard.md`](../.claude/rules/running-onboard.md).

Requires the venv and `pip install .` described in
[`docs/getting-started.md`](../docs/getting-started.md).

## Layout

```text
examples/
├── workers/                      # raw Worker API, organized by level
│   ├── l2/                       #   one chip
│   └── l3/                       #   one host, several chips (+ SubWorkers)
├── a2a3/
│   ├── host_build_graph/            # @scene_test host-built graphs, a2a3
│   └── tensormap_and_ringbuffer/    # @scene_test device orchestration, a2a3
└── a5/tensormap_and_ringbuffer/     # @scene_test device orchestration, a5
```

An example directory holds its test file, a `kernels/` tree (`aic/`, `aiv/`,
`orchestration/`), and a README:

```text
my_example/
├── kernels/
│   ├── aic/                # AICore-CUBE sources        (optional)
│   ├── aiv/                # AICore-VECTOR sources      (optional)
│   └── orchestration/      # orchestration C++
├── test_my_example.py
└── README.md
```

## Related

- [`docs/user/how-to/write-and-run-a-kernel.md`](../docs/user/how-to/write-and-run-a-kernel.md) — the guided version of what these examples show
- [`docs/testing.md`](../docs/testing.md) — the `@scene_test` framework and the wider test suite
- [`tests/st/`](../tests/st/) — scene tests proper, including the full collective-algorithm corpus
