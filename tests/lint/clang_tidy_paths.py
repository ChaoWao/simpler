#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Classify changed paths for the clang-tidy hook and its CI preparation."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_CPP_SUFFIXES = frozenset(
    {
        ".c",
        ".c++",
        ".c++m",
        ".cc",
        ".ccm",
        ".cpp",
        ".cppm",
        ".cxx",
        ".cxxm",
        ".h",
        ".hh",
        ".hpp",
        ".hxx",
        ".inl",
        ".ino",
        ".ipp",
        ".ixx",
        ".mm",
        ".tpp",
    }
)
_RECOGNIZED_NON_CPP_SUFFIXES = frozenset(
    {
        ".asc",
        ".cce",
        ".gyp",
        ".gypi",
        ".py",
        ".pyi",
        ".pyt",
        ".tac",
        ".wsgi",
        ".md",
        ".rst",
        ".txt",
        ".yml",
        ".yaml",
        ".json",
        ".toml",
        ".pin",
        ".cmake",
        ".in",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".gitignore",
        ".clang-format",
        ".clang-tidy",
    }
)


def _relative_path(path: str) -> str:
    """Normalize a repository-relative or repository-local absolute path."""
    candidate = Path(path.replace("\\", "/"))
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(_ROOT)
        except ValueError:
            return candidate.as_posix()
    return candidate.as_posix().removeprefix("./")


def is_cpp_path(path: str) -> bool:
    """Return whether pre-commit treats a path as C or C++ source."""
    return Path(_relative_path(path)).suffix.lower() in _CPP_SUFFIXES


def is_clang_tidy_excluded(path: str) -> bool:
    """Return whether the clang-tidy policy excludes a C or C++ path."""
    normalized = _relative_path(path)
    return (
        normalized.startswith("3rdparty/")
        or normalized.startswith("python/bindings/")
        or "/kernels/" in normalized
        or "/aicore/" in normalized
    )


def should_run_clang_tidy(path: str) -> bool:
    """Return whether clang-tidy should inspect a changed path."""
    return is_cpp_path(path) and not is_clang_tidy_excluded(path)


def needs_pre_commit_build(path: str) -> bool:
    """Return whether a changed, existing path needs the simulator build."""
    normalized = _relative_path(path)
    if normalized == ".pre-commit-config.yaml":
        return True
    if should_run_clang_tidy(normalized):
        return True
    if is_cpp_path(normalized):
        return False

    lower_path = normalized.lower()
    path = Path(lower_path)
    if path.suffix in _RECOGNIZED_NON_CPP_SUFFIXES or path.name in _RECOGNIZED_NON_CPP_SUFFIXES:
        return False
    if path.name == "license" or path.name.startswith("license."):
        return False
    return True


def _existing_paths(paths: Iterable[str]) -> Iterable[str]:
    for path in paths:
        if os.path.exists(path) or os.path.islink(path):
            yield path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--needs-build",
        action="store_true",
        help="print whether any changed path needs a build",
    )
    parser.add_argument(
        "--null",
        action="store_true",
        help="read NUL-delimited paths from standard input",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.needs_build or not args.null:
        raise SystemExit("--needs-build and --null are required")

    paths = (os.fsdecode(path) for path in sys.stdin.buffer.read().split(b"\0") if path)
    print(str(any(needs_pre_commit_build(path) for path in _existing_paths(paths))).lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
