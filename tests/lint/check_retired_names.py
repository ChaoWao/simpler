# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Script to check that the retired brand names do not come back.

`.claude/rules/codestyle.md` rule 10 says "PTO Runtime", "PTO Runtime2" and a bare
"Runtime2" are not names of anything, and that the count only goes down. That was a
promise with nothing keeping it: the rule's prose claimed the surface was empty
while its own grep found 38 banners across 36 files, because the number in the
prose had been measured with a narrower pattern (`PTO2|pto2_`) than the rule
prescribes. A count in prose cannot stay true; a hook can.

Scope, and why it is this narrow:

- Only the **brand** spellings are checked. `PTO2_*` still appears legitimately in
  prose that records the retirement of a specific name ("the now-retired
  `PTO2_RING_*` variables"), and in `PTO2_MANUAL_MAX_SEQ`, a pypto-lib knob this
  repository never reads. Banning those would fail the sentences that explain them.
- The lowercase `pto_*` identifier namespace is **not** checked, and must not be.
  `pto_cpu_sim_bind_device`, `pto_sim_get_subblock_id`,
  `pto_sim_get_pipe_shared_state` and `pto_sim_register_hooks` are live symbols at
  the pto-isa boundary, resolved by `dlsym(RTLD_DEFAULT)`. Renaming one produces no
  compile error — just a hook that is never found at run time.
"""

import argparse
import re
import sys
from pathlib import Path

# "PTO Runtime", "PTO Runtime2", "Runtime2" as a standalone word.
RETIRED_BRAND = re.compile(r"PTO[ _-]Runtime2?\b|\bRuntime2\b")

# The rule has to spell the names it bans, and this hook has to quote them to
# explain itself. Paths are repo-relative and matched exactly.
EXEMPT = frozenset(
    {
        ".claude/rules/codestyle.md",
        "tests/lint/check_retired_names.py",
    }
)


def offending_lines(path: Path) -> list[tuple[int, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return [(lineno, line.strip()) for lineno, line in enumerate(text.splitlines(), 1) if RETIRED_BRAND.search(line)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="files to check (pre-commit passes the staged set)")
    args = parser.parse_args()

    failures = []
    for name in args.files:
        if name in EXEMPT:
            continue
        for lineno, line in offending_lines(Path(name)):
            failures.append(f"{name}:{lineno}: {line}")

    if not failures:
        return 0

    print("Retired brand name reintroduced. The project is `simpler`; its runtimes are")
    print("`host_build_graph` and `tensormap_and_ringbuffer`. Name the runtime, or the")
    print("component the sentence actually means — see .claude/rules/codestyle.md rule 10.")
    print()
    for failure in failures:
        print(f"  {failure}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
