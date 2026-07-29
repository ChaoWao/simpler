# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Script to check that test functions never hardcode a platform name.

A test under ``examples/`` or ``tests/st/`` must take its platform from the
``st_platform`` fixture and declare what it supports with
``@pytest.mark.platforms([...])``. A literal platform passed as a call argument
makes the test ignore ``--platform``: it builds a callable for the wrong arch on
whatever silicon the run was scoped to.
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import Optional

# Roots whose ``test_*.py`` files are driven by the ST fixtures in conftest.py.
CHECKED_ROOTS = ("examples", "tests/st")

# Single source of truth for platform names; parsed, not imported, so this hook
# runs in pre-commit's isolated env without installing the project.
PLATFORM_SSOT = Path("simpler_setup/platform_info.py")
PLATFORM_SSOT_SYMBOL = "PLATFORM_MAP"

# Used only when the SSOT cannot be read (e.g. hook invoked outside the repo).
FALLBACK_PLATFORMS = frozenset({"a2a3", "a2a3sim", "a5", "a5sim"})

# `str` methods that inspect a platform rather than consume it. A test guarding
# on the fixture (``st_platform.startswith("a2a3")``) is the pattern this check
# steers people to, so a literal handed to one of these is not a finding.
STRING_INSPECTION_METHODS = frozenset(
    {
        "startswith",
        "endswith",
        "count",
        "find",
        "rfind",
        "index",
        "rindex",
        "split",
        "rsplit",
        "partition",
        "rpartition",
        "replace",
        "strip",
        "lstrip",
        "rstrip",
        "removeprefix",
        "removesuffix",
    }
)


def load_platform_names(root_dir: Path) -> frozenset[str]:
    """Read the platform names off ``PLATFORM_MAP`` in the SSOT module."""
    ssot = root_dir / PLATFORM_SSOT
    try:
        tree = ast.parse(ssot.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return FALLBACK_PLATFORMS

    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if not any(isinstance(t, ast.Name) and t.id == PLATFORM_SSOT_SYMBOL for t in targets):
            continue
        if isinstance(node.value, ast.Dict):
            names = {k.value for k in node.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if names:
                return frozenset(names)

    return FALLBACK_PLATFORMS


def _call_argument_literals(node: ast.Call) -> list[ast.Constant]:
    """String constants passed to ``node`` as a positional or keyword argument.

    Only arguments to a call that *consumes* a platform count. Inspecting the
    fixture is correct usage and is skipped in both its forms: a comparison
    (``if st_platform != "a2a3":``) is not an argument position at all, and a
    string predicate (``if not st_platform.startswith("a2a3"):``) is filtered
    by ``STRING_INSPECTION_METHODS``.
    """
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in STRING_INSPECTION_METHODS:
        return []

    literals = []
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            literals.append(arg)
    for kw in node.keywords:
        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            literals.append(kw.value)
    return literals


def check_file(file_path: Path, platforms: frozenset[str]) -> list[tuple[int, str]]:
    """Return ``(line, platform)`` for every hardcoded platform in a test body."""
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return []

    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
            continue
        # Walk the body only — ``@pytest.mark.platforms(["a2a3"])`` lives in
        # ``decorator_list`` and is the pattern this check steers people to.
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if not isinstance(sub, ast.Call):
                    continue
                for literal in _call_argument_literals(sub):
                    if literal.value in platforms:
                        violations.append((literal.lineno, literal.value))

    return sorted(set(violations))


def _is_checked_test_file(path: Path, root_dir: Path) -> bool:
    if not path.name.startswith("test_") or path.suffix != ".py":
        return False
    try:
        rel = path.resolve().relative_to(root_dir)
    except ValueError:
        return False
    return any(rel.is_relative_to(root) for root in CHECKED_ROOTS)


def _collect_files(files: list[str], root_dir: Path) -> Optional[list[Path]]:
    """Collect test files to check, from an explicit list or a full scan."""
    if files:
        return [p for f in files if _is_checked_test_file(p := Path(f).resolve(), root_dir)]

    if not (root_dir / ".git").exists():
        print(f"Error: '{root_dir}' is not a git repository", file=sys.stderr)
        return None

    collected: list[Path] = []
    for root in CHECKED_ROOTS:
        collected.extend(sorted((root_dir / root).rglob("test_*.py")))
    return collected


def main() -> int:
    """Main function."""
    parser = argparse.ArgumentParser(description="Check that test functions never hardcode a platform name")
    parser.add_argument("files", nargs="*", help=f"Files to check (default: all test_*.py under {CHECKED_ROOTS})")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    root_dir = Path(".").resolve()
    files_to_check = _collect_files(args.files, root_dir)
    if files_to_check is None:
        return 1

    if not files_to_check:
        print("No test files found to check")
        return 0

    platforms = load_platform_names(root_dir)
    print(f"Checking {len(files_to_check)} test file(s) for hardcoded platforms ({', '.join(sorted(platforms))})...")

    failed = 0
    for file_path in files_to_check:
        violations = check_file(file_path, platforms)
        display = file_path.relative_to(root_dir) if file_path.is_relative_to(root_dir) else file_path
        if not violations:
            if args.verbose:
                print(f"✓ {display}")
            continue
        failed += 1
        for line_num, platform in violations:
            print(f"✗ {display}:{line_num} hardcoded platform {platform!r}")

    if failed:
        print(
            f"\n⚠ {failed} test file(s) hardcode a platform, so they ignore --platform and run on the wrong "
            "silicon.\n"
            "  Take the platform from the fixture and declare support with a marker:\n\n"
            '      @pytest.mark.platforms(["a2a3"])\n'
            '      @pytest.mark.runtime("tensormap_and_ringbuffer")\n'
            "      @pytest.mark.device_count(2)\n"
            "      def test_my_demo(st_device_ids, st_platform) -> None:\n"
            "          assert run(st_platform, [int(d) for d in st_device_ids]) == 0\n"
        )
        return 1

    print("\n✓ No test hardcodes a platform!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
