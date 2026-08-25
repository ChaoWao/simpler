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

`Worker.init()` snapshots the effective ``simpler`` logger threshold into the
state owned by the native extension. A hierarchical worker seeds the parent
process before its first fork, and every loaded host module receives a pointer
to that state; onboard setup maps the same threshold onto CANN's coarser
severity ladder.
"""

import logging
import threading

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

# Two Worker or ChipWorker instances can initialize concurrently from different
# threads, and their locks are per instance, so nothing above serializes the
# handler installation below. Without this, both callers can pass the "already
# installed?" check and add a handler, and every record is then forwarded twice.
_attach_mutex = threading.Lock()

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


def _ladder_level(levelno: int) -> int:
    """Map an arbitrary record level down onto the public ladder.

    Rounds toward the less severe name, unlike `_normalize_threshold`, which
    rounds a *threshold* up: a record at 35 is not an error, it is a warning that
    someone gave a custom number.
    """
    if levelno >= logging.ERROR:
        return logging.ERROR
    if levelno >= logging.WARNING:
        return logging.WARNING
    if levelno >= TIMING:
        return TIMING
    if levelno >= logging.INFO:
        return logging.INFO
    return logging.DEBUG


class _UnifiedLogHandler(logging.Handler):
    """Forwards `simpler` records into the host logger the C++ side writes with.

    The two were separate logging systems that shared only a threshold. A Python
    record carried no timestamp and no thread id, so it could not be ordered
    against a C++ record wherever either was written, and it went to whatever
    handler happened to be on the root logger rather than to the host logger's
    configured output. Forwarding gives every record in the process one envelope,
    one clock and one destination.

    `propagate` is deliberately left alone, so a record still reaches the root
    logger as well: that is what keeps pytest's capture and an interactive
    console working. The host log is the complete copy; the console is a view of
    it.

    Which is also why forwarding is conditional. The host logger writes to stderr
    until it is given a directory, and the root logger's handler already prints
    there — so forwarding unconditionally would put the same record on the same
    stream twice, once in each envelope, for every run that does not ask for
    diagnostics. A record is forwarded only once the host log has a destination
    of its own. That is checked per record because the directory arrives with a
    run's config, after this handler is installed.
    """

    def __init__(self, emit_record, host_log_directory):
        super().__init__()
        self._emit_record = emit_record
        self._host_log_directory = host_log_directory

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if not self._host_log_directory():
                return
            self._emit_record(_ladder_level(record.levelno), record.funcName or "python", self.format(record))
        except Exception:  # noqa: BLE001 -- a logging sink must never raise into its caller
            self.handleError(record)


def attach_unified_log_handler(emit_record, host_log_directory) -> bool:
    """Send `simpler` records to the host logger as well as to the root logger.

    Called from the same place that seeds the native log threshold, which is
    after the extension is loaded — installing this at import time would put the
    extension on the import path of the logging surface, which is the one thing
    that has to keep working when the extension is missing or stale.

    Idempotent, and atomically so: returns True only for the call that installs
    the handler, even when several callers race.
    """
    with _attach_mutex:
        if any(isinstance(handler, _UnifiedLogHandler) for handler in _logger.handlers):
            return False
        _logger.addHandler(_UnifiedLogHandler(emit_record, host_log_directory))
        return True


def get_current_config() -> int:
    """Return the current threshold for forwarding to ChipWorker.init().

    Reads the simpler logger's effective level — which respects user
    setLevel() calls and falls back to DEFAULT_THRESHOLD when unconfigured
    (we set that at module import).
    """
    return _normalize_threshold(_logger.getEffectiveLevel())
