# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile selected scene-test callables without creating a device worker."""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

try:
    from _pytest.skipping import evaluate_skip_marks
except ImportError:
    evaluate_skip_marks = None

logger = logging.getLogger(__name__)

_DEFAULT_COMPILE_WORKERS = 1


def _is_skipped(item) -> bool:
    """Report whether pytest already knows it will skip ``item``.

    ``evaluate_skip_marks`` is pytest-internal and absent on a pytest whose
    layout has moved; the warm-up then compiles the skipped class too, which
    costs compile time but cannot change what the later pytest run does.
    """
    if evaluate_skip_marks is None:
        return False
    return evaluate_skip_marks(item) is not None


def _compile_scene_test_class(cls, platform: str) -> None:
    if cls._st_level == 2:
        cls.compile_chip_callable(platform)
    elif cls._st_level == 3:
        cls._compile_l3_callables(platform)


def compile_collected_scene_tests(
    items, platform: str, *, max_workers: int = _DEFAULT_COMPILE_WORKERS
) -> tuple[int, list[tuple[str, Exception]]]:
    """Compile each non-skipped scene-test class represented by ``items``.

    Returns ``(compiled_count, failures)`` with one ``(class_name, error)`` per
    class that raised. A compilation error is reported rather than propagated
    because this pass only populates a cache: the pytest run that follows
    recompiles what is missing and attributes the error to the case it belongs
    to, so one unbuildable kernel does not cost the whole batch its results.
    """
    selected_classes = []
    seen_classes = set()
    for item in items:
        cls = getattr(item, "cls", None)
        if cls is None or cls in seen_classes or not hasattr(cls, "_st_level"):
            continue
        if _is_skipped(item):
            continue
        if cls._st_level not in (2, 3):
            continue
        seen_classes.add(cls)
        selected_classes.append(cls)

    def compile_one(cls) -> tuple[str, Exception] | None:
        try:
            _compile_scene_test_class(cls, platform)
        except Exception as error:
            logger.warning("[SceneTestCompile] %s failed to compile: %r", cls.__name__, error)
            return (cls.__name__, error)
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(compile_one, selected_classes))
    failures = [result for result in results if result is not None]
    return len(selected_classes) - len(failures), failures


class _CompilePlugin:
    def pytest_addoption(self, parser):
        group = parser.getgroup("scene-test compilation")
        group.addoption(
            "--compile-workers",
            action="store",
            type=int,
            default=_DEFAULT_COMPILE_WORKERS,
            help="Maximum concurrent SceneTestCase compilations (default: 1)",
        )

    def pytest_collection_finish(self, session):
        platform = session.config.getoption("--platform")
        if not platform:
            import pytest  # noqa: PLC0415

            raise pytest.UsageError("--platform is required for scene-test compilation")
        max_workers = session.config.getoption("--compile-workers")
        if max_workers < 1:
            import pytest  # noqa: PLC0415

            raise pytest.UsageError("--compile-workers must be at least 1")
        count, failures = compile_collected_scene_tests(session.items, platform, max_workers=max_workers)
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(f"compiled scene-test cache for {count} class(es)")
            for name, error in failures:
                reporter.write_line(f"::warning::scene-test cache warm-up failed for {name}: {error!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run pytest collection and populate the persistent scene-test cache."""
    import pytest  # noqa: PLC0415

    args = list(sys.argv[1:] if argv is None else argv)
    if "--collect-only" not in args:
        args.append("--collect-only")
    return int(pytest.main(args, plugins=[_CompilePlugin()]))


if __name__ == "__main__":
    raise SystemExit(main())
