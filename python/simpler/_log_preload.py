# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Put ``libsimpler_log.so`` in the process-global symbol scope.

The logger has to be loaded before host runtimes and before workers fork. The
package's earliest ``_task_interface`` import also passes its host-span entry
point into the extension, where a nullable function pointer provides the
cross-platform optional-sink contract.

A missing library leaves that pointer null, which disables host spans and fails
nothing. The path is resolved with ``find_spec`` rather than by importing
``simpler_setup.environment``: locating the package does not execute it, so
``import simpler`` does not pull in the compiler-side surface for a diagnostic.
"""

from __future__ import annotations

import ctypes
import functools
import importlib.util
import os
from pathlib import Path

_LIBRARY_NAME = "libsimpler_log.so"

# RTLD_GLOBAL dlopen registry, keyed by path. host_runtime.so and the sim context
# resolve against these globals, so each must be loaded exactly once before any
# host_runtime.so dlopen; a second dlopen of a path already here is skipped
# rather than repeated. Never closed. Lives here rather than in task_interface so
# the import-time preload below and ChipWorker.init share one registry.
_preloaded_globals: dict[str, ctypes.CDLL] = {}


def preload_global(path: str) -> ctypes.CDLL:
    """dlopen `path` with RTLD_NOW | RTLD_GLOBAL, idempotently (one CDLL per path).

    Eager resolution (RTLD_NOW) surfaces a missing-symbol problem at load time
    rather than at first use.
    """
    handle = _preloaded_globals.get(path)
    if handle is None:
        handle = ctypes.CDLL(path, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
        _preloaded_globals[path] = handle
    return handle


def _resolve_library() -> Path | None:
    """Locate libsimpler_log.so under either install layout, or None.

    Mirrors ``simpler_setup.environment._resolve_project_root``: a wheel keeps
    the built libraries under ``simpler_setup/_assets``, a source tree or
    editable install keeps them at the repo root.
    """
    try:
        spec = importlib.util.find_spec("simpler_setup")
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None

    package = Path(spec.origin).resolve().parent
    assets = package / "_assets"
    project_root = assets if (assets / "src").is_dir() else package.parent
    library = project_root / "build" / "lib" / _LIBRARY_NAME
    return library if library.is_file() else None


@functools.cache
def preload() -> ctypes.CDLL | None:
    """dlopen the process-global logger RTLD_GLOBAL, or return None quietly.

    Cached, so the repeated call from ``ChipWorker.init``'s own preload of this
    library does not bump the loader refcount again.
    """
    library = _resolve_library()
    if library is None:
        return None
    try:
        return preload_global(str(library))
    except OSError:
        return None


def host_span_sink_address(handle: ctypes.CDLL | None) -> int:
    """Return the logger's host-span entry-point address, or zero if absent."""
    if handle is None:
        return 0
    try:
        symbol = handle.simpler_log_emit_host_span
    except AttributeError:
        return 0
    return int(ctypes.cast(symbol, ctypes.c_void_p).value or 0)
