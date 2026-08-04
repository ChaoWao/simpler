# CI Pipeline

## Overview

The CI pipeline maps test categories (st, ut-py, ut-cpp) × hardware tiers to GitHub Actions jobs. See [testing.md](testing.md) for full test organization and hardware classification.

Design principles:

1. **Merge by runner, not by language** — Python and C++ unit tests share setup cost and run as steps within a single job per runner tier (`ut`, `ut-a2a3`, `ut-a5`).
2. **Runner matches hardware tier** — no-hardware tests run on `ubuntu-latest`; platform-specific tests run on self-hosted runners with the matching label (`a2a3`, `a5`).
3. **`--platform` is the only filter** — pytest uses `--platform` + the `requires_hardware` marker; ctest uses label `-LE` exclusion. No `-m st`, no `-m "not requires_hardware"`.
4. **sim = no hardware** — `a2a3sim`/`a5sim` jobs run on github-hosted runners alongside unit tests.
5. **Skip irrelevant platforms, and irrelevant suites** — `detect-changes` gates `st-sim-*` and `st-onboard-*` by platform, so pure-a5 PRs skip a2a3 scene-test runs and vice versa. **UT jobs are still not gated by platform** — unit tests cover shared contracts and the cost of a falsely-skipped regression outweighs the savings. They *are* gated by test category: a diff confined to `tests/st/` or `examples/` cannot break a unit test, and one confined to `tests/ut/` cannot break a scene test, because the two suites execute disjoint trees (`pytest examples tests/st` vs `pytest tests/ut` plus the C++ ctest) and no unit test reads `examples/` or `tests/st/`. Shared test infrastructure — the root `conftest.py`, `pyproject.toml`, `simpler_setup/`, `tests/lint/` — belongs to neither category and runs both.
6. **Non-code PRs run pre-commit and docs, and nothing else** — `detect-changes` sets `non_code_only` when *no* changed file falls outside the `NON_CODE` set. Skipping the UT jobs is riskless there because nothing in that set can change what the code does, and each member already has its own gate: markdownlint inside `pre-commit` reads the markdown, `docs.yml` (unconditional on every PR) builds the site with `--strict`, and `pre-commit` itself is ungated so a `.pre-commit-config.yaml` change is fully exercised by it.

## Full Job Matrix

The complete test-type × hardware-tier matrix. Empty cells have no tests yet; only non-empty jobs exist in `ci.yml`.

| Category | github-hosted (no hardware) | a2a3 runner | a5 runner |
| -------- | --------------------------- | ----------- | --------- |
| **ut** (py + cpp) | `ut` | `ut-a2a3` | `ut-a5` |
| **st** | `st-sim-a2a3`, `st-sim-a5` | `st-onboard-a2a3` | `st-onboard-a5` |

## GitHub Actions Jobs

`ci.yml` and `ci-self-cpu.yml` are now thin topology callers: they own
triggers, `needs:`, gate `if:` expressions, runner/setup inputs, and matrix
shape. The executable job bodies live in reusable workflows:
`_detect-changes.yml`, `_pre-commit.yml`, `_ut-no-hardware.yml`,
`_packaging.yml`, `_profiling-flags-smoke.yml`, `_st-sim-a2a3.yml`,
`_st-sim-a5.yml`, `_ut-npu-a2a3.yml`, `_ut-npu-a5.yml`,
`_st-npu-a2a3.yml`, and `_st-npu-a5.yml`. The scene-test and NPU unit-test
bodies are split one workflow per architecture so each job renders only its own
steps. Shared step scaffolding that is safe to run
after checkout lives in composite actions under `.github/actions/`
(`cache-pip`, `setup-venv`).

```text
PullRequest
  ├── pre-commit             (ubuntu-latest)
  ├── packaging-matrix       (ubuntu + macOS)        — [needs !examples_only && !tests_only]
  ├── profiling-flags-smoke  (ubuntu-latest)         — (a2a3_changed || a5_changed) && !examples_only && !tests_only
  ├── ut                     (ubuntu + macOS)        — Python + C++ UT, no hardware [needs ut_affected]
  ├── detect-changes         (reusable; ubuntu-latest in ci.yml) — outputs non_code_only, a{2a3,5}_changed, {st,ut}_affected, examples_only, tests_only
  ├── st-sim-a2a3            (ubuntu + macOS)        — a2a3_changed && st_affected
  ├── st-sim-a5              (ubuntu + macOS)        — a5_changed && st_affected
  ├── ut-a2a3                (a2a3 self-hosted)      — Python + C++ UT, a2a3 hardware [needs ut_affected]
  ├── st-onboard-a2a3        (a2a3 self-hosted)      — a2a3_changed && st_affected
  ├── ut-a5                  (a5 self-hosted)        — Python + C++ UT, a5 hardware [needs ut_affected]
  └── st-onboard-a5          (a5 self-hosted)        — a5_changed && st_affected
```

| Job | Runner | What it runs |
| --- | ------ | ------------ |
| `ut` | `ubuntu-latest`, `macos-latest` | `pytest tests/ut` + `ctest -LE requires_hardware` |
| `st-sim-a2a3` | `ubuntu-latest`, `macos-latest` | `pytest examples tests/st --platform a2a3sim` |
| `st-sim-a5` | `ubuntu-latest`, `macos-latest` | `pytest examples tests/st --platform a5sim` |
| `ut-a2a3` | a2a3 self-hosted | `pytest tests/ut --platform a2a3` + `ctest -L "^requires_hardware(_a2a3)?$" --resource-spec-file ...` + build `tools/cann-examples/query` and run `query version` (no device) + build `tools/cann-examples/aicpu-device-query` and `tools/cann-examples/aicpu-kernel-launch` (host + cross-compiled device SO, link smoke only) |
| `st-onboard-a2a3` | a2a3 self-hosted | `pytest examples tests/st -m "not sdma" --platform a2a3 --device ...`, then a separate `-m sdma` step, then the DFX per-feature smokes |
| `ut-a5` | a5 self-hosted | `pytest tests/ut --platform a5` + `ctest -L "^requires_hardware(_a5)?$"` + build `tools/cann-examples/query` and run `query version` (no device) + build `tools/cann-examples/aicpu-device-query` and `tools/cann-examples/aicpu-kernel-launch` (link smoke only) |
| `st-onboard-a5` | a5 self-hosted | `pytest examples tests/st --platform a5 --device ...` |

### Nightly sanitizer sweep

A **separate** workflow, [`sanitizers.yml`](../.github/workflows/sanitizers.yml),
runs on a nightly `schedule` — kept out of `ci.yml` so the cron fires only the
sanitizer jobs, never the PR/self-hosted pipeline. Its
`sanitizer-sim` job builds the sim runtime + kernels with ASAN or TSAN
(`pip install --config-settings=cmake.define.SIMPLER_SANITIZER=...`) and runs a
**scoped** subset under the matching `LD_PRELOAD` (a2a3sim/a5sim, ubuntu-only).
`dlopen_count` tests are excluded everywhere (they assert exact dlopen accounting
that the sanitizers perturb by interposing `dlopen`). The full suite is avoided
because ASAN/TSAN slow the sim enough that oversubscription-heavy cases livelock
on a 4-vCPU runner — so the scope is parallelism-limited per sanitizer:

- **ASAN** (~1.7x): `prepared_callable` + `dynamic_register` (where present),
  `--max-parallel 2`, skipping `parallel_broadcast`.
- **TSAN** (~5-15x): livelocks the chip-fork L3 cases even when run serially, so it
  runs only the light `prepared_callable` L2 tests, `--max-parallel 1`, with
  `TSAN_OPTIONS=halt_on_error=0:exitcode=0` (report races without aborting *or*
  failing the job — TSAN's default `exitcode=66` would otherwise redden the cell on
  every race; the job gates on hang/crash, triaging the reported races into a
  suppressions file is a follow-up).

Both sanitizer jobs gate (no `continue-on-error`). Not a PR gate; see
[sanitizers.md](sanitizers.md) for the design + usage.

### Parallel ST runs on hardware

For self-hosted jobs with multiple NPUs, pass a `--device` range (and
optionally pytest's `-x` for fail-fast) to get the full dispatcher
benefit — device bin-packing for L3, xdist fanout for L2, and a shared
`ChipWorker` per `(runtime, device)`:

```bash
# Recommended CI invocation — a2a3 deselects the SDMA marker, as the job does,
# and runs it as a second pass afterwards
pytest examples tests/st -m "not sdma" --platform a2a3 --device 4-7 -x
pytest examples tests/st -m sdma --platform a2a3 --device 4-5 -x

# Same for a5, which has no marker filter
pytest examples tests/st --platform a5 --device 0-7 -x
```

`-x` (`--exitfirst`) is appropriate for CI, where aborting on first
failure saves runner minutes. Local development usually wants the opposite
(let every failure surface) — just drop the flag. The short form is the
same in both pytest and standalone on purpose; see
[testing.md §CLI Design Principles](testing.md#cli-design-principles).

`pytest-xdist` is pulled in via the `test` extra. See
[testing.md §Parallel Test Execution](testing.md#parallel-test-execution-and-resource-reuse)
for the full hierarchy, fail-fast semantics, and the
profiling-vs-parallelism trade-off.

### Sim jobs on CPU-constrained runners

Sim jobs (`st-sim-a2a3`, `st-sim-a5`) run on `ubuntu-latest`, whose standard
GitHub-hosted runner currently has **4 vCPUs**. `--device 0-15` is still the
right choice for the **pool size** (some L3 cases need several virtual ids), but
the default `--max-parallel auto` caps the in-flight subprocess count to
`min(nproc, len(--device))` — on a 4-core runner that becomes `4`. Note
`os.cpu_count()` reports the host's logical CPUs and ignores any cgroup CPU
quota, so this is the true core count, not a container limit.

```bash
# Sim: --max-parallel auto resolves to 4 on a standard ubuntu-latest runner
pytest examples tests/st --platform a2a3sim --device 0-15

# Throttle further on a CPU-starved runner: 4 concurrent cases (each forking
# several chip subprocesses with many threads) can oversubscribe 4 cores and
# trigger the sim handshake/deinit failures in
# troubleshooting/sim-oversubscription-hang.md. --max-parallel 2 trades
# throughput for stability.
pytest examples tests/st --platform a2a3sim --device 0-15 --max-parallel 2
```

On hardware jobs the `auto` default is `len(--device)` because each subprocess
is device-bound (host CPU mostly waits on the NPU), so hardware runners do
not need `--max-parallel` manually.

### Scheduling constraints

- Sim scene tests and no-hardware unit tests run on github-hosted runners (no hardware).
- `detect-changes` is implemented once in [`.github/workflows/_detect-changes.yml`](../.github/workflows/_detect-changes.yml) and computes four axes from the PR diff — non-code (`non_code_only`), architecture (`a2a3_changed` / `a5_changed`), test category (`st_affected` / `ut_affected`), and corpus (`examples_only` / `tests_only`) — **all of them derived from one `NON_CODE` set**: `docs/`, `.docs/`, `.claude/`, `mkdocs.yml`, `.github/workflows/docs.yml`, `.gitignore`, `.pre-commit-config.yaml`, and any `*.md` file anywhere. Membership follows a file's *effect*, not its path — `mkdocs.yml` and `docs.yml` are docs tooling that happens to live outside `docs/`. An arch flag is `false` only when every changed file is in the opposite platform's tree (`src/{arch}/`, `examples/{arch}/`, `tests/{st,ut/cpp}/{arch}/`) or in `NON_CODE`. Anything else — shared C++ (`src/common/`), Python (`python/`, `simpler_setup/`), build files (`CMakeLists.txt`, `pyproject.toml`), shared test infra (`tests/ut/py/`, `tests/lint/`), tooling (`tools/`), or any CI implementation workflow (`.github/workflows/ci.yml`, `.github/workflows/ci-self-cpu.yml`, `.github/workflows/_*.yml`) — flips both flags to `true`. CI implementation workflows are deliberately excluded from `NON_CODE`: a change to the gates or reusable job bodies must run everything, including whatever it just switched off.
- **Test-category axis:** `ST_ONLY='^(tests/st/|examples/)'` and `UT_ONLY='^tests/ut/'`, applied in the same shape as the arch patterns — a category is unaffected only when *every* changed file is exclusively the other's. `st_affected` gates the four scene-test jobs; `ut_affected` gates `ut`, `ut-a2a3`, `ut-a5`. Anything belonging to neither (root `conftest.py`, `pyproject.toml`, `simpler_setup/`, `tests/lint/`) flips both, so shared infrastructure always runs both suites.
- **Corpus axis:** `EXAMPLES_ONLY='^examples/'` and `TESTS_ONLY='^(tests/)'`, same shape again, one per side of the product jobs' payload. `packaging-matrix` and `profiling-flags-smoke` build and install the product and then exercise it with a fixed, tiny payload — one entry-point script (a `tests/st/` file) and one `vector_example` — so neither reads either suite as a corpus, and a diff confined to `examples/` or to `tests/` cannot reach them: `wheel.packages` is `["simpler_setup", "python/simpler"]`, so both partitions are provably absent from the product. A payload file changed under either is still exercised by the scene-test job that reads the same corpus.
- **Gated jobs (scene tests):** `st-sim-{a2a3,a5}`, `st-onboard-{a2a3,a5}` run iff their platform's flag **and** `st_affected` are `true`.
- **Platform-independent jobs (all UT + packaging):** `ut`, `ut-a2a3`, `ut-a5`, `packaging-matrix` ignore the *platform* flags — unit tests exercise shared contracts (nanobind bindings, RuntimeBuilder, ring buffers, etc.) and the risk of silently skipping a regression outweighs the CI minutes saved. The `tests/ut/cpp/{arch}/` entry in the gating regex only *attributes* an arch-specific C++ UT change to that platform (so it does not spuriously flip the other arch's scene-test flag); it does not gate the UT jobs themselves. The three UT jobs do respect `ut_affected`, which is a statement about test category rather than silicon. `packaging-matrix` respects the corpus axis, which is a statement about which corpus a job reads: neither `examples/` nor `tests/` is in `wheel.packages`, so a diff confined to either partition cannot change what the packaging job builds.
- **`non_code_only` is the same `NON_CODE` set, not a narrower one.** It is `true` when no changed file falls outside it. Nothing in the set can change what the code does, and no workflow consumes any of it beyond its own gate: `pre-commit` is ungated so it always exercises `.pre-commit-config.yaml`, `docs.yml` is unconditional on every PR so it always exercises `mkdocs.yml` / `docs/`, and **no workflow invokes anything under `.claude/`** (`grep -rn '\.claude' .github/workflows/` finds only comments). An **empty diff short-circuits the whole step**: attribution is impossible, so a single guard sets `non_code_only=false`, every arch and category flag `true`, and both corpus flags (`examples_only`, `tests_only`) `false`, then returns — running the full matrix, packaging and the profiling smoke included. That guard is deliberately one place; testing emptiness per flag is what previously left `non_code_only` false while both arch flags also came out false, running UT and packaging but skipping every scene test.

  The arch flags subtract `NON_CODE` before deciding, so a non-code-only change already makes both `false`. An arch-gated job therefore needs no separate non-code check. See [`.claude/rules/ci-change-detection.md`](../.claude/rules/ci-change-detection.md) for the invariants these gates must keep.

- **SDMA tests run as their own step inside `st-onboard-a2a3`.** The sweep deselects them with `-m "not sdma"` and a later step runs `-m sdma`. Ordering is what the two paths share: the SDMA step is always second, so no fault-injection case can land on a device that has already provisioned SDMA. Device acquisition differs by host arch — on aarch64 the SDMA step takes its own `task-submit --device auto --device-num 2`, so the two steps are disjoint in devices as well; on x86_64 there is no `task-submit` and both steps use the same `${DEVICE_RANGE}`, leaving ordering as the only separation. Provisioning the SDMA workspace creates device-only STARS streams that live in the device fault domain, so an AICore fault on a device that has provisioned SDMA costs minutes instead of milliseconds — the sweep's `aicore_op_timeout` fault injection must therefore never share a device with them ([#1425](https://github.com/hw-native-sys/simpler/issues/1425)). Selection is by marker on both sides, so the two cannot drift apart; the split can be dropped once #1425 is fixed. Nothing outside `st-onboard-a2a3` filters on the marker, so a local `pytest examples tests/st` still runs everything.

### CPU emergency lane (`ci-self-cpu.yml`) and the `/run-cpu` button

When GitHub-hosted runners are congested, a repo admin can validate a PR on the
repo-level self-hosted runners via the emergency lane, bypassing GitHub-hosted
queueing entirely:

- **Trigger**: comment `/run-cpu` on the PR (`ci-self-cpu-button.yml`), or
  `gh workflow run ci-self-cpu.yml -f repository=<repo> -f ref=<sha>` manually
  (covers forks and arbitrary SHAs). The button gate is
  `permission(commenter) == 'admin'` only (`getCollaboratorPermissionLevel`).
  `issue_comment` from fork PRs runs with a read-only token, so the permission
  check must work under it — verify on first fork-PR use.
- **What it runs**: checks out `repository@ref` (the PR head), then T1 — the
  no-hardware Linux jobs (`pre-commit`, `ut`, `packaging`, `profiling-flags-smoke`,
  `st-sim-{a2a3,a5}`) on `[self-hosted, cpu]` — and T3 — the NPU jobs
  (`ut-a2a3`, `st-onboard-a2a3`, `ut-a5`, `st-onboard-a5`) on
  `[self-hosted, a2a3/a5]`. T2 (macOS) is intentionally absent. The lane calls
  the same reusable job-body workflows as `ci.yml`, passing
  `setup_variant=self-cpu`, `repository`, `ref`, and self-hosted runner labels
  where the main CI passes `setup_variant=github` and GitHub-hosted runners.
  Gate outputs still come from the canonical `detect-changes` workflow, executed
  on `[self-hosted, cpu]` in this lane.
- **cpu runner contract**: dnf-installed `cmake ninja-build gcc-c++ clang-tools-extra graphviz gtest-devel python3-devel`, plus a pip-installable torch aarch64 CPU wheel; `g++-15` is a symlink stand-in for the ubuntu-toolchain ppa g++. On the agents `g++` resolves to a conda GCC 15 prefix rather than `/usr/bin/g++`, so the lane's sim artifacts are built with GCC 15 and `compile_commands.json` names that prefix's `<triple>-g++`; `tests/lint/clang_tidy.py` drops the triple before replaying a command, without which clang-tidy adopts it as a target and resolves no C++ standard library at all. `ci.yml` lints with clang-tidy 18 and HCE 2.0 packages only LLVM 12, so an agent additionally provides 18 on `PATH` as `clang-tidy-18`, installed together with its clang builtin headers — a clang-tidy whose prefix carries no `lib/clang/<major>/include` resolves no resource dir and fails every `#include <stddef.h>`. The `pre-commit` job shadows the distro `clang-tidy` with it when present.
- The lane run is standalone — it attaches no checks to the PR; results are read from the run.

## Hardware Classification

Three hardware tiers, applied to all test categories. See [testing.md](testing.md#hardware-classification) for the full table including per-category mechanisms (pytest markers, ctest labels, folder structure).

| Tier | CI Runner | Job examples |
| ---- | --------- | ------------ |
| No hardware | `ubuntu-latest` | `ut`, `st-sim-*` |
| Platform-specific (a2a3) | `[self-hosted, a2a3]` | `ut-a2a3`, `st-onboard-a2a3` |
| Platform-specific (a5) | `[self-hosted, a5]` | `ut-a5`, `st-onboard-a5` |

On a self-hosted runner, every step that touches an NPU — pytest and ctest
alike — must hold its devices exclusively while it runs. There are two a2a3
runner pools, branched at run time on the host arch (`uname -m`):

- **ARM64 a2a3 runners** share the host with interactive users, so the step
  runs through `task-submit --device <list> --run "..."`, whose per-device
  lock keeps a CI job from colliding with someone's local run (and vice
  versa).
- **X64 a2a3 runners** do not use `task-submit` — their cards are exclusive to
  the runner — so the step runs `pytest`/`ctest` directly with
  `--device ${DEVICE_RANGE}`.

a5 runners are ARM64-only and always use `task-submit`. Steps that only build
(cmake, `RuntimeBuilder`, the `cann-examples` smokes) take no lock on either
arch. The same device-lock rule applies to local onboard work — see
[.claude/rules/running-onboard.md](../.claude/rules/running-onboard.md).

## Test Sources

### `tests/ut/` — Python unit tests (ut-py)

Python unit tests. Run via pytest, filtered by `--platform` + `requires_hardware` marker.

| File | Content | Hardware? |
| ---- | ------- | --------- |
| `test_task_interface.py` | nanobind extension API tests | No |
| `test_runtime_builder.py` (mocked classes) | RuntimeBuilder discovery, error handling, build logic | No |
| `test_runtime_builder.py::TestRuntimeBuilderIntegration` | Real compilation across platform × runtime | Yes (`@pytest.mark.requires_hardware`) |

### `tests/ut/cpp/` — C++ unit tests (ut-cpp)

GoogleTest-based tests for pure C++ modules. Run via ctest, filtered by label `-LE` exclusion.

| Runner | Command |
| ------ | ------- |
| No hardware | `ctest --test-dir tests/ut/cpp/build -LE requires_hardware` |
| a2a3 | `ctest --test-dir tests/ut/cpp/build -L "^requires_hardware(_a2a3)?$"` |
| a5 | `ctest --test-dir tests/ut/cpp/build -L "^requires_hardware(_a5)?$"` |

### `examples/` — Small examples (sim + onboard)

Small, fast examples that run on both simulation and real hardware. Organized as `examples/{arch}/{runtime}/{name}/`. Discovered and executed by pytest via each example's `test_*.py` (`@scene_test` format).

### `tests/st/` — Scene tests (onboard-biased)

Large-scale, feature-rich hardware tests. Too slow or using instructions unsupported by the simulator. Organized as `tests/st/{arch}/{runtime}/{name}/`. Platform compatibility is declared per test via `@scene_test(platforms=[...])`.

### Shared structure

Both `examples/` and `tests/st/` cases follow the same layout:

```text
{name}/
  test_{name}.py                 # @scene_test class (generate_args, compute_golden)
  kernels/
    orchestration/*.cpp
    aic/*.cpp                    # optional
    aiv/*.cpp                    # optional
```

Cases are discovered by pytest via `test_*.py` files. Each test module ends with `if __name__ == "__main__": SceneTestCase.run_module(__name__)` so it can also run standalone as `python test_*.py -p <platform>`.

## Selection Scheme

A single `--platform` flag controls hardware/non-hardware splitting across all three categories.

### ut-py (pytest marker)

```python
@pytest.mark.requires_hardware                  # any hardware
class TestRuntimeBuilderIntegration:
    ...

@pytest.mark.requires_hardware("a2a3")          # a2a3 specifically
class TestA2A3Feature:
    ...
```

Selection:

```bash
# No hardware (no-hw tests run, requires_hardware tests skip)
pytest tests/ut

# Hardware (no-hw tests skip, hw + platform-specific tests run)
pytest tests/ut --platform a2a3
```

### ut-cpp (ctest label)

```cmake
# any hardware
set_tests_properties(test_runtime_integration PROPERTIES LABELS "requires_hardware")
# a2a3-specific
set_tests_properties(test_a2a3_feature PROPERTIES LABELS "requires_hardware_a2a3")
```

Selection uses `-LE` (label exclude) on no-hw runner and `-L` (label include) on device runners:

```bash
ctest -LE requires_hardware                 # no-hardware runner: only unlabeled
ctest -L "^requires_hardware(_a2a3)?$"      # a2a3 runner: hw + a2a3-specific
ctest -L "^requires_hardware(_a5)?$"        # a5 runner: hw + a5-specific
```

### st (`@scene_test`)

```python
@scene_test(level=2, platforms=["a2a3sim", "a2a3"], runtime="tensormap_and_ringbuffer")
class TestVectorExample(SceneTestCase):
    ...
```

| `--platform` | Behavior |
| ------------ | -------- |
| `a2a3sim` | Run if `"a2a3sim"` in `platforms` |
| `a2a3` | Run if `"a2a3"` in `platforms` |
| *(none)* | Auto-parametrize over all `*sim` entries in `platforms` |

No `--platform` means "run all sims" — tests with no sim in their `platforms` list are skipped. No additional markers are used.

## Platform notes

- **macOS libomp collision**: on macOS, the root `conftest.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` before `import pytest` to work around a duplicate-libomp abort triggered by homebrew numpy and pip torch coexisting in one Python process (see [troubleshooting/macos-libomp-collision.md](troubleshooting/macos-libomp-collision.md)). Standalone `python test_*.py` bypasses conftest — rely on the env var being exported by the shell or `tools/verify_packaging.sh`.
- **sim hangs / `rc=-1` under CPU oversubscription**: on a few-vCPU runner, high `--max-parallel` (or many concurrent sim cases) oversubscribes the host CPUs, where sim's busy-spin handshake can livelock (hang → `rc=124`) or the deinit timeout can false-trip (`simpler_run failed with code -1`). Mitigate with `--max-parallel 2`; onboard is unaffected (see [troubleshooting/sim-oversubscription-hang.md](troubleshooting/sim-oversubscription-hang.md)).
- **local runs time out more slowly than CI**: compiled defaults are lenient for serving workloads, while CI sets tighter `SIMPLER_*_TIMEOUT_*` env values to fail fast. Use the same env values locally when debugging suspected hangs (see [troubleshooting/local-timeout-defaults.md](troubleshooting/local-timeout-defaults.md)).
- **`st-onboard-a2a3` mass 507899 is not OOM**: a whole-suite collapse of `507899`/`507018`/`register_callable -1` is an AICPU device-fault cascade (`simpler_aicpu_exec` exception), not memory exhaustion. Diagnosis recipe and the per-device preinstall-name fix are in [troubleshooting/a2a3-507899-aicpu-shared-so-fault.md](troubleshooting/a2a3-507899-aicpu-shared-so-fault.md).
