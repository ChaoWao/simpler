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


@pytest.mark.parametrize(
    ("changed_files", "expected"),
    [
        (["docs/readme.md"], {"needs_package": "false", "needs_cpp": "false", "target": "_task_interface"}),
        (["python/simpler/api.py"], {"needs_package": "true", "needs_cpp": "false", "target": "_task_interface"}),
        (["src/common/api.cpp"], {"needs_package": "true", "needs_cpp": "true", "target": "build_package_sim"}),
        (["vendor/dependency.cpp"], {"needs_package": "false", "needs_cpp": "false", "target": "_task_interface"}),
        (
            ["python/simpler/api.py", "include/simpler/api.h"],
            {"needs_package": "true", "needs_cpp": "true", "target": "build_package_sim"},
        ),
    ],
)
def test_select_lint_build_target(tmp_path: Path, changed_files: list[str], expected: dict[str, str]) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    base = _commit(repo, "README.md", "base\n")
    for index, changed_file in enumerate(changed_files):
        head = _commit(repo, changed_file, f"change {index}\n")

    output = tmp_path / "github-output"
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

    assert result.returncode == 0, result.stdout + result.stderr
    actual = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert actual == expected
