# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Rewrite doc links that point outside ``docs/`` to absolute repository URLs.

The docs are written to be read on GitHub, where a relative link like
``../../src/common/hierarchical/worker.cpp`` resolves. MkDocs only serves
``docs_dir``, so those links would 404 on the built site — roughly a quarter of
all relative links in ``docs/`` leave the tree, most of them citing source files
as evidence for a claim.

This hook resolves each relative link against the page's own directory and, when
the target escapes ``docs/``, replaces it with a permalink into the repository at
the ref being built. Links that stay inside ``docs/`` are untouched, so MkDocs
still validates them and ``--strict`` still catches a genuinely dead one.
"""

from __future__ import annotations

import os
import posixpath
import re

REPO_URL = "https://github.com/hw-native-sys/simpler"

# Ref to point at. CI sets DOCS_REF; a local build falls back to main.
REF = os.environ.get("DOCS_REF", "main")

# ](target) or ](target#anchor) — skips images, absolute URLs and pure anchors.
_LINK = re.compile(r"(?<!!)\]\((?!https?://|mailto:|#)([^)\s]+?)(#[^)]*)?\)")

# Directories serve as tree/ URLs; files as blob/. A trailing slash means a dir.
_TREE_SUFFIX = ("/",)


def _repo_url(target: str) -> str:
    kind = "tree" if target.endswith(_TREE_SUFFIX) else "blob"
    return f"{REPO_URL}/{kind}/{REF}/{target.rstrip('/')}"


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ARG001 -- MkDocs hook signature
    page_dir = posixpath.dirname(page.file.src_uri)

    def replace(match: re.Match[str]) -> str:
        target, anchor = match.group(1), match.group(2) or ""
        resolved = posixpath.normpath(posixpath.join(page_dir, target))
        # normpath drops a trailing slash; keep it so directories stay tree/ URLs.
        if target.endswith("/"):
            resolved += "/"
        if not resolved.startswith(".."):
            return match.group(0)  # still inside docs/ — leave for MkDocs to check
        # Strip only the leading ../ hops that took us out of docs_dir. Not
        # lstrip("./"), which would also eat the dot of a dotfile path such as
        # ../.claude/rules/codestyle.md and yield claude/rules/....
        repo_path = resolved
        while repo_path.startswith("../"):
            repo_path = repo_path[3:]
        return f"]({_repo_url(repo_path)}{anchor})"

    return _LINK.sub(replace, markdown)
