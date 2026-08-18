# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
WORKFLOW = REPO_ROOT / ".github/workflows/_pre-commit.yml"


def _selection_script() -> str:
    workflow = WORKFLOW.read_text()
    start = "      - name: Select lint build target\n"
    end = "\n      - name: Install build and lint tools\n"
    assert workflow.count(start) == 1
    block = workflow.split(start, 1)[1].split(end, 1)[0]
    run = block.split("        run: |\n", 1)[1]
    return textwrap.dedent(run)


def _commit(repo: Path, relative_path: str, contents: str) -> str:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    subprocess.run(["git", "add", relative_path], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=CI", "-c", "user.email=ci@example.com", "commit", "-m", relative_path],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _run_selection(repo: Path, base: str, head: str, output: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({"BASE_SHA": base, "HEAD_SHA": head, "GITHUB_OUTPUT": str(output)})
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _selection_script()],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    diagnostics = result.stdout + result.stderr
    assert result.returncode == 0, diagnostics
    assert output.is_file(), diagnostics or "selection script did not write GITHUB_OUTPUT"
    return dict(line.split("=", 1) for line in output.read_text().splitlines())


def test_selection_script_uses_macos_bash_compatible_case_matching() -> None:
    script = _selection_script()

    assert "${path,,}" not in script
    assert "shopt -s nocasematch" in script
    assert "shopt -u nocasematch" in script


def test_self_cpu_uses_the_project_venv_for_lint() -> None:
    workflow = WORKFLOW.read_text()
    setup = "      - name: venv + deps\n"
    lint_only = "      - name: Lint-only venv\n"
    activate = "      - name: Use lint venv\n"
    run = "      - name: Run pre-commit\n"

    assert workflow.count(setup) == workflow.count(lint_only) == workflow.count(activate) == workflow.count(run) == 1
    assert workflow.index(setup) < workflow.index(lint_only) < workflow.index(activate) < workflow.index(run)
    lint_only_block = workflow.split(lint_only, 1)[1].split(activate, 1)[0]
    assert "steps.lint-build.outputs.needs_build != 'true'" in lint_only_block
    assert "run: python3 -m venv .venv" in lint_only_block
    block = workflow.split(activate, 1)[1].split(run, 1)[0]
    assert "if: inputs.setup_variant == 'self-cpu'" in block
    assert 'run: echo "$PWD/.venv/bin" >> "$GITHUB_PATH"' in block


def test_only_selected_build_paths_install_the_project() -> None:
    workflow = WORKFLOW.read_text()
    cache = "      - name: Cache pip packages\n"
    install = "      - name: Install package (for clang-tidy compile databases)\n"
    setup = "      - name: venv + deps\n"
    lint_only = "      - name: Lint-only venv\n"

    assert "pip install torch" not in workflow
    cache_block = workflow.split(cache, 1)[1].split(install, 1)[0]
    assert "inputs.setup_variant == 'self-cpu' || steps.lint-build.outputs.needs_build == 'true'" in cache_block
    install_block = workflow.split(install, 1)[1].split(setup, 1)[0]
    assert "steps.lint-build.outputs.needs_build == 'true'" in install_block
    assert "build.targets=build_package_sim" in install_block
    setup_block = workflow.split(setup, 1)[1].split(lint_only, 1)[0]
    assert 'install-torch: "false"' in setup_block


@pytest.mark.parametrize(
    ("changed_files", "expected"),
    [
        (["docs/readme.md"], {"needs_build": "false"}),
        ([".github/actions/setup-gcc-15/ubuntu-toolchain-r-test.asc"], {"needs_build": "false"}),
        ([".pre-commit-config.yaml"], {"needs_build": "true"}),
        (["python/simpler/api.py"], {"needs_build": "false"}),
        (["python/simpler/api.PYI"], {"needs_build": "false"}),
        (["tools/server.wsgi"], {"needs_build": "false"}),
        (["3rdparty/tool.py"], {"needs_build": "false"}),
        (["src/common/api.cpp"], {"needs_build": "true"}),
        (["src/common/api.inl"], {"needs_build": "true"}),
        (["src/common/api.HPP"], {"needs_build": "true"}),
        (["vendor/dependency.cpp"], {"needs_build": "true"}),
        (["3rdparty/dependency.cpp"], {"needs_build": "false"}),
        (["3rdparty/dependency.CPP"], {"needs_build": "false"}),
        (["3RDPARTY/dependency.cpp"], {"needs_build": "true"}),
        (["python/bindings/module.cpp"], {"needs_build": "false"}),
        (["examples/a2a3/demo/kernels/aic/kernel.cpp"], {"needs_build": "false"}),
        (["src/a2a3/runtime/demo/aicore/kernel.cpp"], {"needs_build": "false"}),
        (["src/common/future.newcpp"], {"needs_build": "true"}),
        (["python/simpler/api.py", "include/simpler/api.h"], {"needs_build": "true"}),
    ],
)
def test_select_lint_build_target(tmp_path: Path, changed_files: list[str], expected: dict[str, str]) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    base = _commit(repo, "README.md", "base\n")
    for index, changed_file in enumerate(changed_files):
        head = _commit(repo, changed_file, f"change {index}\n")

    actual = _run_selection(repo, base, head, tmp_path / "github-output")
    assert actual == expected


def test_deleted_source_does_not_require_a_build(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    base = _commit(repo, "src/common/deleted.cpp", "int deleted;\n")
    (repo / "src/common/deleted.cpp").unlink()
    subprocess.run(["git", "add", "src/common/deleted.cpp"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=CI", "-c", "user.email=ci@example.com", "commit", "-m", "delete"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

    assert _run_selection(repo, base, head, tmp_path / "github-output") == {"needs_build": "false"}


def test_special_characters_in_source_name_require_a_build(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    base = _commit(repo, "README.md", "base\n")
    head = _commit(repo, "src/common/api name\npart.cpp", "int special_name;\n")

    assert _run_selection(repo, base, head, tmp_path / "github-output") == {"needs_build": "true"}


def test_unrecognized_path_with_trailing_newline_requires_a_build(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    base = _commit(repo, "README.md", "base\n")
    head = _commit(repo, "src/common/not-source.CPP\n", "not source\n")

    assert _run_selection(repo, base, head, tmp_path / "github-output") == {"needs_build": "true"}


def test_unrecognized_executable_script_requires_a_build(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    base = _commit(repo, "README.md", "base\n")
    script = repo / "tools/run-check"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env python3\n")
    script.chmod(0o755)
    subprocess.run(["git", "add", "tools/run-check"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=CI", "-c", "user.email=ci@example.com", "commit", "-m", "script"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

    assert _run_selection(repo, base, head, tmp_path / "github-output") == {"needs_build": "true"}


def test_selection_uses_merge_base_diff(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    common = _commit(repo, "README.md", "base\n")
    head = _commit(repo, "docs/readme.md", "docs\n")
    subprocess.run(["git", "checkout", "-q", "--detach", common], cwd=repo, check=True)
    base = _commit(repo, "src/common/base_only.cpp", "int base_only;\n")

    assert _run_selection(repo, base, head, tmp_path / "github-output") == {"needs_build": "false"}


def test_selection_falls_back_when_refs_have_no_merge_base(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    base = _commit(repo, "README.md", "base\n")
    subprocess.run(["git", "checkout", "-q", "--orphan", "unrelated"], cwd=repo, check=True)
    (repo / "README.md").unlink()
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    head = _commit(repo, "src/common/unrelated.cpp", "int unrelated;\n")

    assert _run_selection(repo, base, head, tmp_path / "github-output") == {"needs_build": "true"}
