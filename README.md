# simpler — Simple Runtime

**Make writing runtime simpler.**

Modular runtime for building and executing task dependency graphs on Ascend devices with coordinated AICPU and AICore execution. Three independently compiled programs (Host `.so`, AICPU `.so`, AICore `.o`) work together through clearly defined APIs.

## Where simpler sits in the stack

simpler is one layer of the [hw-native-sys](https://github.com/hw-native-sys) stack. Each repo below imports the ones to its left:

```text
pto-isa ──▶ simpler ──▶ pypto ──▶ pypto-lib ──▶ pypto-serving
(tile ISA)  (runtime)  (compiler)  (kernels)    (serving)
                ▲
             PTOAS (assembler, installed globally)
```

**simpler owns** task orchestration and dispatch at runtime: DAG submission, the AICPU/host scheduler, worker and device lifecycle, the host↔device handshake, and the DFX/profiling tooling that reads it back. **simpler does not own** kernel bodies, the tile ISA, graph compilation, or model/serving logic — those belong to the repos below.

| Repository | Owns | Relation to simpler |
| ---------- | ---- | ------------------- |
| [pto-isa](https://github.com/hw-native-sys/pto-isa) | PTO tile virtual ISA and its C++ tile-instruction library | Build-time dependency: AICore kernels in `examples/` include its headers; revision pinned by [`pto_isa.pin`](pto_isa.pin) |
| [PTOAS](https://github.com/hw-native-sys/PTOAS) | `ptoas` — MLIR-based assembler/optimizer lowering PTO bytecode to device code | Provided as a global binary on dev boxes and CI; simpler neither builds nor vendors it |
| [pypto](https://github.com/hw-native-sys/pypto) | Tensor/Tile programming framework and compiler (Python frontend → PTO IR → device code) | Vendors simpler as a submodule at `runtime/` and drives it through simpler's Python API |
| [pypto-lib](https://github.com/hw-native-sys/pypto-lib) | Tensor-level kernels and end-to-end model implementations (Qwen3, DeepSeek) | Consumer of pypto + simpler; source of the cross-repo performance and regression workloads |
| [pypto-serving](https://github.com/hw-native-sys/pypto-serving) | LLM inference service and standalone model runners | Top of the stack — exercises simpler through the full serving path |
| [pypto_top_level_documents](https://github.com/hw-native-sys/pypto_top_level_documents) | Cross-repo design documents and proposals | Where runtime designs spanning more than one repo are written down |

To run a workload from any of these repos against this checkout of simpler, see [`.claude/skills/multi-repo-setup/SKILL.md`](.claude/skills/multi-repo-setup/SKILL.md).

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd simpler

# Install (venv recommended — see `.claude/rules/venv-isolation.md`)
pip install --no-build-isolation -e '.[test]'

# Run the vector example (simulation, no hardware required)
python examples/a2a3/tensormap_and_ringbuffer/vector_example/test_vector_example.py -p a2a3sim
```

PTO ISA headers are automatically cloned on first run. See [Getting Started](docs/getting-started.md) for manual setup and troubleshooting.

## Platforms

| Platform | Description | Requirements |
| -------- | ----------- | ------------ |
| `a2a3` | Real Ascend A2/A3 hardware | CANN toolkit (ccec, aarch64 cross-compiler) |
| `a2a3sim` | Thread-based A2/A3 simulation | gcc/g++ only (no Ascend SDK needed) |
| `a5` | Real Ascend A5 hardware | CANN toolkit (ccec, aarch64 cross-compiler) |
| `a5sim` | Thread-based A5 simulation | gcc/g++ only (no Ascend SDK needed) |

## Runtime Variants

Two runtimes under `src/{arch}/runtime/`, each with a different graph-building strategy:

| Runtime | Graph built on | Use case |
| ------- | -------------- | -------- |
| `host_build_graph` | Host CPU | Development, debugging |
| `tensormap_and_ringbuffer` | AICPU (device) | Production workloads |

See runtime docs per arch: [a2a3](src/a2a3/docs/runtimes.md), [a5](src/a5/docs/runtimes.md).

## Testing

```bash
# Simulation scene tests (no hardware)
pytest examples tests/st --platform a2a3sim

# Hardware scene tests (requires Ascend device)
pytest examples tests/st --platform a2a3 --device 4-7

# Python unit tests
pytest tests/ut -m "not requires_hardware" -v

# C++ unit tests
cmake -B tests/ut/cpp/build -S tests/ut/cpp && cmake --build tests/ut/cpp/build && ctest --test-dir tests/ut/cpp/build --output-on-failure
```

See [Testing Guide](docs/testing.md) for details.

## Environment Setup

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

## Documentation

Building **on** simpler? Start at **[docs/user/](docs/user/README.md)** — how-to guides plus the Python and CLI reference. **[docs/README.md](docs/README.md) indexes every document**, grouped by task. The entry points:

| Document | Description |
| -------- | ----------- |
| [Capability Survey](docs/capability-survey.md) | Status snapshot: what is shipped, gated, or design-only across topology (L2→L4), CANN launch, and comm engines |
| [Chip-Level Architecture](docs/chip-level-arch.md) | L2 single-chip: three-program model (host/AICPU/AICore), API layers, handshake protocol |
| [Hierarchical Level Runtime](docs/hierarchical-level-runtime.md) | L0–L6 level model, component composition (Orchestrator / Scheduler / Worker) |
| [Task Flow](docs/task-flow.md) | End-to-end data flow: Callable / TaskArgs / CallConfig handles, IWorker interface |
| [Orchestrator](docs/orchestrator.md) | DAG submission internals: submit flow, TensorMap, Scope, Ring, task state machine |
| [Scheduler](docs/scheduler.md) | DAG dispatch internals: wiring/ready/completion queues, dispatch loop |
| [Worker Manager](docs/worker-manager.md) | Worker pool, WorkerThread, THREAD/PROCESS modes, fork + mailbox mechanics |
| [Getting Started](docs/getting-started.md) | Setup, prerequisites, build process, configuration |
| [Developer Guide](docs/developer-guide.md) | Directory structure, role ownership, conventions |
| [Testing Guide](docs/testing.md) | CI pipeline, test types, writing new tests |

### Per-arch docs

| Document | a2a3 arch | a5 arch |
| -------- | --------- | ------- |
| Runtimes | [a2a3/docs/runtimes.md](src/a2a3/docs/runtimes.md) | [a5/docs/runtimes.md](src/a5/docs/runtimes.md) |
| Platform | [a2a3/docs/platform.md](src/a2a3/docs/platform.md) | [a5/docs/platform.md](src/a5/docs/platform.md) |

## License

This project is licensed under the **CANN Open Software License Agreement Version 2.0**. See the [LICENSE](LICENSE) file for the full license text.

## References

- [src/a2a3/platform/](src/a2a3/platform/) - Platform implementations
- [src/a2a3/runtime/](src/a2a3/runtime/) - Runtime implementations
- [examples/](examples/README.md) - Runnable examples, indexed: `workers/` for the raw `Worker` API, `a2a3/` and `a5/` for `@scene_test` kernels
- [simpler_setup/](simpler_setup/) - SceneTestCase framework, runtime builder, kernel compiler
- [python/](python/) - Python bindings and user-facing runtime API
