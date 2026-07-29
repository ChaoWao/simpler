# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Simpler runtime — public Python surface.

Host-side log filter setup happens in `ChipWorker.init` (see
`simpler.task_interface`): it `ctypes.CDLL`s libsimpler_log.so RTLD_GLOBAL,
calls its `simpler_log_init` C entry to seed the process-wide HostLogger, then
hands off to the C++ `_ChipWorker.init` which dlopens host_runtime.so (whose
`simpler_init` reads CANN dlog config off that same HostLogger, onboard only).
The level forwarded is a one-shot snapshot of the `simpler` Python logger.
Nothing log-related needs to happen at import time here.

`Worker` and the `task_interface` submodule resolve on first attribute access
rather than at import time: both pull in the `_task_interface` extension, so
eager imports here would make `import simpler` fail wherever that extension is
missing or stale, including for callers that only want the logging helpers.
"""

import importlib
from typing import Any

# Importing _log auto-configures the simpler logger to TIMING if unset.
from ._log import (
    DEFAULT_THRESHOLD,
    NUL,
    TIMING,
    get_current_config,
    get_logger,
)

__all__ = [
    "DEFAULT_THRESHOLD",
    "Worker",
    "NUL",
    "TIMING",
    "get_current_config",
    "get_logger",
    "task_interface",
]

# name -> (module, attribute). Resolved by __getattr__ on first access.
_LAZY_ATTRS = {"Worker": (f"{__name__}.worker", "Worker")}
_LAZY_SUBMODULES = ("task_interface",)


def __getattr__(name: str) -> Any:
    """Resolve the extension-backed surface on first access (PEP 562)."""
    if name in _LAZY_ATTRS:
        module_name, attribute = _LAZY_ATTRS[name]
        return getattr(importlib.import_module(module_name), attribute)
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
