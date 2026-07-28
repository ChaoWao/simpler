# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stop griffe's missing-annotation notices from failing ``--strict``.

``--strict`` is what makes the build fail on a dead link, a page missing from
``nav``, or an anchor that does not exist — the checks worth gating on. It counts
warnings through ``mkdocs.utils.CountHandler`` on the root logger, which also
picks up griffe's "No type or annotation for parameter". That one reports a
*source* typing gap — a parameter the docstring describes but the signature does
not annotate — not a documentation defect, and it cannot be fixed from here
without inventing a type for a parameter whose accepted values have to be
established first.

The filter is attached to the counting handler only, so the warning still prints
and stays visible; it just does not abort the build. Every other griffe or MkDocs
warning is still counted and still fails ``--strict``. Remove this hook once the
annotations land.
"""

from __future__ import annotations

import logging

from mkdocs.utils import CountHandler

_SUPPRESSED = "No type or annotation for parameter"


class _UncountMissingAnnotation(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _SUPPRESSED not in record.getMessage()


def on_pre_build(config) -> None:  # noqa: ANN001, ARG001 -- MkDocs hook signature
    # Not on_startup: `build()` attaches the counter to the `mkdocs` logger at its
    # own entry, which is after startup but before this event fires.
    for handler in logging.getLogger("mkdocs").handlers:
        if isinstance(handler, CountHandler):
            handler.addFilter(_UncountMissingAnnotation())
