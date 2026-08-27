#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ast
import os
import shlex
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
HARNESS = REPO_ROOT / "tools/benchmark_rounds.sh"


def _fake_python_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    invocations = tmp_path / "invocations"
    python3 = bin_dir / "python3"
    python3.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            if [[ "${1:-}" == "-m" ]]; then
                cat <<'EOF'
              Round      Host (us)   Device (us)
              ----------------------------------
              0              100.0          20.0
              Avg Host: 100.0 us  |  Avg Device: 20.0 us [1/1]  (1 rounds)
            EOF
                exit 0
            fi
            printf '%s\\n' "$*" >> "$FAKE_BENCH_INVOCATIONS"
            """
        )
    )
    python3.chmod(0o755)

    env = os.environ.copy()
    env["FAKE_BENCH_INVOCATIONS"] = str(invocations)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    return env, invocations


def _run_harness_from_root(
    tmp_path: Path, repo_root: Path, *args: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    env, invocations = _fake_python_env(tmp_path)
    result = subprocess.run(  # noqa: S603 -- fixed repository script
        [str(repo_root / "tools/benchmark_rounds.sh"), *args],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, invocations


def _run_harness(tmp_path: Path, *args: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    return _run_harness_from_root(tmp_path, REPO_ROOT, *args)


def _invoked_cases(invocations: Path) -> list[tuple[str, str, str]]:
    result = []
    for line in invocations.read_text().splitlines():
        args = shlex.split(line)
        test_file = Path(args[0])
        case_idx = args.index("--case")
        result.append((test_file.parent.parent.name, test_file.parent.name, args[case_idx + 1]))
        assert "--skip-golden" in args
        assert args[case_idx + 2 : case_idx + 4] == ["--manual", "include"]
    return result


@pytest.mark.parametrize(
    ("platform", "runtime", "expected"),
    [
        (
            "a2a3",
            "tensormap_and_ringbuffer",
            [
                ("tensormap_and_ringbuffer", "alternating_matmul_add", "Case1"),
                ("tensormap_and_ringbuffer", "benchmark_bgemm", "Case0"),
                ("tensormap_and_ringbuffer", "paged_attention_unroll", "Case1"),
                ("tensormap_and_ringbuffer", "paged_attention_unroll", "Case2"),
                ("tensormap_and_ringbuffer", "paged_attention_unroll_manual_scope", "Case1"),
                ("tensormap_and_ringbuffer", "paged_attention_unroll_manual_scope", "Case2"),
                ("tensormap_and_ringbuffer", "batch_paged_attention", "Case1"),
                ("tensormap_and_ringbuffer", "qwen3_14b_decode", "StressBatch16Seq3500"),
            ],
        ),
        (
            "a2a3",
            "host_build_graph",
            [
                ("host_build_graph", "alternating_matmul_add", "Case1"),
                ("host_build_graph", "benchmark_bgemm", "Case0"),
                ("host_build_graph", "paged_attention_unroll", "Case1"),
                ("host_build_graph", "paged_attention_unroll", "Case2"),
                ("host_build_graph", "paged_attention_unroll_manual_scope", "Case1"),
                ("host_build_graph", "paged_attention_unroll_manual_scope", "Case2"),
                ("host_build_graph", "batch_paged_attention", "Case1"),
                ("host_build_graph", "qwen3_14b_decode", "GraphExecutionBatch16Seq3500"),
            ],
        ),
        (
            "a5",
            "tensormap_and_ringbuffer",
            [
                ("tensormap_and_ringbuffer", "alternating_matmul_add", "Case1"),
                ("tensormap_and_ringbuffer", "benchmark_bgemm", "Case0"),
                ("tensormap_and_ringbuffer", "paged_attention_unroll", "Case1"),
                ("tensormap_and_ringbuffer", "paged_attention_unroll", "Case2"),
                ("tensormap_and_ringbuffer", "paged_attention_unroll_manual_scope", "Case1"),
                ("tensormap_and_ringbuffer", "paged_attention_unroll_manual_scope", "Case2"),
                ("tensormap_and_ringbuffer", "batch_paged_attention", "Case1"),
                ("tensormap_and_ringbuffer", "qwen3_14b_decode", "StressBatch16Seq3500"),
            ],
        ),
        (
            "a5",
            "host_build_graph",
            [
                ("host_build_graph", "alternating_matmul_add", "Case1"),
                ("host_build_graph", "benchmark_bgemm", "Case0"),
                ("host_build_graph", "paged_attention_unroll", "Case1"),
                ("host_build_graph", "paged_attention_unroll", "Case2"),
                ("host_build_graph", "paged_attention_unroll_manual_scope", "Case1"),
                ("host_build_graph", "paged_attention_unroll_manual_scope", "Case2"),
                ("host_build_graph", "batch_paged_attention", "Case1"),
                ("host_build_graph", "qwen3_14b_decode", "GraphExecutionBatch16Seq3500"),
            ],
        ),
    ],
    ids=["a2a3-tmr", "a2a3-hbg", "a5-tmr", "a5-hbg"],
)
def test_arch_runtime_quadrants_route_to_independent_corpora(
    tmp_path: Path, platform: str, runtime: str, expected: list[tuple[str, str, str]]
) -> None:
    result, invocations = _run_harness(
        tmp_path,
        "--runtime",
        runtime,
        "--platform",
        platform,
        "--device",
        "7",
        "--rounds",
        "1",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    invoked_cases = _invoked_cases(invocations)
    assert invoked_cases == expected
    expected_qwen_cases = {
        ("a2a3", "tensormap_and_ringbuffer"): ["StressBatch16Seq3500"],
        ("a2a3", "host_build_graph"): ["GraphExecutionBatch16Seq3500"],
        ("a5", "tensormap_and_ringbuffer"): ["StressBatch16Seq3500"],
        ("a5", "host_build_graph"): ["GraphExecutionBatch16Seq3500"],
    }
    qwen_cases = [case for _, example, case in invoked_cases if example == "qwen3_14b_decode"]
    assert qwen_cases == expected_qwen_cases[platform, runtime]
    assert all(example != "spmd_paged_attention" for _, example, _ in expected)
    assert f"Runtime: {runtime}" in result.stdout
    expected_mode = "default" if runtime == "host_build_graph" else "parallel"
    assert f"[{expected_mode}]" in result.stdout
    assert f"Performance Summary ({runtime})" in result.stdout
    assert "Host (us)" in result.stdout
    assert "Device (us)" in result.stdout
    if runtime == "host_build_graph":
        assert "Effective (us)" not in result.stdout
        assert "Orch (us)" not in result.stdout
        assert "Sched (us)" not in result.stdout
    assert f"Benchmark complete ({runtime}): {len(expected)} passed, 0 failed" in result.stdout


@pytest.mark.parametrize(
    ("relative_path", "case_name"),
    [
        (
            "examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_decode/test_qwen3_14b_decode.py",
            "StressBatch16Seq3500",
        ),
        (
            "examples/a2a3/host_build_graph/qwen3_14b_decode/test_qwen3_14b_decode.py",
            "GraphExecutionBatch16Seq3500",
        ),
        (
            "examples/a5/tensormap_and_ringbuffer/qwen3_14b_decode/test_qwen3_14b_decode.py",
            "StressBatch16Seq3500",
        ),
        (
            "examples/a5/host_build_graph/qwen3_14b_decode/test_qwen3_14b_decode.py",
            "GraphExecutionBatch16Seq3500",
        ),
    ],
)
def test_qwen_benchmark_cases_are_manual(relative_path: str, case_name: str) -> None:
    wrapper_path = REPO_ROOT / relative_path
    wrapper_tree = ast.parse(wrapper_path.read_text())
    manual_wrappers = [
        node
        for node in ast.walk(wrapper_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(decorator, ast.Attribute)
            and decorator.attr == "manual"
            and isinstance(decorator.value, ast.Attribute)
            and decorator.value.attr == "mark"
            and isinstance(decorator.value.value, ast.Name)
            and decorator.value.value.id == "pytest"
            for decorator in node.decorator_list
        )
    ]
    assert manual_wrappers

    # The wrapper must forward the global pytest CLI contract rather than
    # silently running the standalone driver with all defaults.
    assert any(
        keyword.arg is None
        and isinstance(keyword.value, ast.Call)
        and isinstance(keyword.value.func, ast.Name)
        and keyword.value.func.id == "standalone_pytest_options"
        for wrapper in manual_wrappers
        for call in ast.walk(wrapper)
        if isinstance(call, ast.Call)
        for keyword in call.keywords
    )

    driver_tree = ast.parse((wrapper_path.parent / "main.py").read_text())
    driver_case_names = []
    for node in ast.walk(driver_tree):
        if not isinstance(node, ast.Assign):
            continue
        target_names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if "CASES" in target_names:
            driver_case_names.extend(case["name"] for case in ast.literal_eval(node.value))
        if "CASE_NAME" in target_names:
            driver_case_names.append(ast.literal_eval(node.value))
    assert case_name in driver_case_names


def test_default_runtime_remains_tensormap_and_ringbuffer(tmp_path: Path) -> None:
    result, invocations = _run_harness(tmp_path, "--platform", "a2a3", "--device", "7", "--rounds", "1")

    assert result.returncode == 0, result.stdout + result.stderr
    invoked_cases = _invoked_cases(invocations)
    assert len(invoked_cases) == 8
    assert all(runtime == "tensormap_and_ringbuffer" for runtime, _, _ in invoked_cases)
    assert "Runtime: tensormap_and_ringbuffer" in result.stdout
    assert "[parallel]" in result.stdout
    assert "host_build_graph" not in invocations.read_text()


def test_extra_arguments_are_forwarded_to_every_benchmark(tmp_path: Path) -> None:
    result, invocations = _run_harness(
        tmp_path,
        "--runtime",
        "host_build_graph",
        "--rounds",
        "1",
        "--custom-flag",
        "custom-value",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    for line in invocations.read_text().splitlines():
        assert shlex.split(line)[-2:] == ["--custom-flag", "custom-value"]


def test_host_build_graph_rejects_serial_orch_sched_before_running(tmp_path: Path) -> None:
    result, invocations = _run_harness(
        tmp_path,
        "--runtime",
        "host_build_graph",
        "--serial-orch-sched",
    )

    assert result.returncode == 1
    assert result.stdout.strip() == "ERROR: --serial-orch-sched is only supported by tensormap_and_ringbuffer."
    assert not invocations.exists()


def test_missing_workload_is_an_error_not_a_skip(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    tools_dir = repo_root / "tools"
    tools_dir.mkdir(parents=True)
    harness = tools_dir / "benchmark_rounds.sh"
    harness.write_text(HARNESS.read_text())
    harness.chmod(0o755)

    result, invocations = _run_harness_from_root(
        tmp_path,
        repo_root,
        "--runtime",
        "host_build_graph",
        "--platform",
        "a5",
    )

    assert result.returncode == 1
    assert "ERROR: benchmark workload 'alternating_matmul_add' not found" in result.stdout
    assert "SKIP" not in result.stdout
    assert not invocations.exists()


def test_help_lists_both_runtime_metric_shapes(tmp_path: Path) -> None:
    result, invocations = _run_harness(tmp_path, "--help")

    assert result.returncode == 0
    assert "tensormap_and_ringbuffer (default)" in result.stdout
    assert "host_build_graph" in result.stdout
    assert "TMR: Host, Device, Effective, Orch, Sched." in result.stdout
    assert "HBG: Host, Device." in result.stdout
    assert not invocations.exists()
