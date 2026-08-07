# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Simpler unified logging — one Python threshold propagated to C++.

The public ladder is DEBUG / INFO / TIMING / WARN / ERROR. TIMING sits between
INFO and WARN and is the default so stable performance markers remain visible
without enabling ordinary INFO traffic. NUL is a suppression sentinel.

`Worker.init()` snapshots the effective ``simpler`` logger threshold. A
hierarchical worker seeds the parent process before its first fork, and every
`ChipWorker.init()` seeds its child before loading host_runtime.so; onboard
setup maps the same threshold onto CANN's coarser severity ladder.
"""

import logging

from ._log_preload import host_span_sink_address as _host_span_sink_address
from ._log_preload import preload as _preload_simpler_log

# Load the logger mapping before probing the extension below. ``simpler``
# imports this module eagerly; once the extension is available, its local sink
# slot is therefore bound before Worker initialization can fork.
_host_log_handle = _preload_simpler_log()

# DEFAULT_LOG_THRESHOLD is exposed by the _task_interface nanobind module so
# Python and C++ share one constant. During a fresh `pip install -e .` the
# pre-existing .so may be stale or absent, so fall back to the hardcoded
# value (kept in sync manually with src/common/log/include/common/log_level.h).
try:
    from _task_interface import (  # pyright: ignore[reportMissingImports]
        DEFAULT_LOG_THRESHOLD as _NATIVE_DEFAULT,
    )
except (ImportError, AttributeError):
    _NATIVE_DEFAULT = 25

try:
    from _task_interface import _bind_host_span_sink  # pyright: ignore[reportMissingImports]
except (ImportError, AttributeError):
    pass
else:
    _bind_host_span_sink(_host_span_sink_address(_host_log_handle))

# Public verbosity constants (Python integer levels).
TIMING = 25
NUL = 60

DEFAULT_THRESHOLD = _NATIVE_DEFAULT

# pytest validates --log-level before importing simpler, so conftest mirrors
# these registrations at module load.
logging.addLevelName(TIMING, "TIMING")
setattr(logging, "TIMING", TIMING)
logging.addLevelName(NUL, "NUL")
setattr(logging, "NUL", NUL)
setattr(logging, "NULL", NUL)  # pytest upcases user's `--log-level null` → NULL

_LOGGER_NAME = "simpler"
_logger = logging.getLogger(_LOGGER_NAME)
if _logger.level == logging.NOTSET:
    _logger.setLevel(DEFAULT_THRESHOLD)


def get_logger() -> logging.Logger:
    """Return the simpler-namespaced Python logger."""
    return _logger


def _normalize_threshold(threshold: int) -> int:
    """Round an arbitrary Python logging threshold up to the public ladder."""
    if threshold <= logging.DEBUG:
        return logging.DEBUG
    if threshold <= logging.INFO:
        return logging.INFO
    if threshold <= TIMING:
        return TIMING
    if threshold <= logging.WARNING:
        return logging.WARNING
    if threshold <= logging.ERROR:
        return logging.ERROR
    return NUL


def get_current_config() -> int:
    """Return the current threshold for forwarding to ChipWorker.init().

    Reads the simpler logger's effective level — which respects user
    setLevel() calls and falls back to DEFAULT_THRESHOLD when unconfigured
    (we set that at module import).
    """
    return _normalize_threshold(_logger.getEffectiveLevel())
