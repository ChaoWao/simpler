#!/usr/bin/env bash
#
# Assert this package still corresponds to the runtime it measures.
#
# The engine is compiled straight out of ../src, so there is no copy to drift
# and nothing to diff. What can still go wrong is the package's own description
# of what it builds:
#   1. the TU list in CMakeLists.txt matches build_config.py's `host` target
#      (minus host/, which is runtime_maker and its CANN dependencies);
#   2. every include directory and engine TU CMake names actually exists;
#   3. the qwen3_dynamic_tensormap case is present, in exactly one place, and
#      the build still pins it to QWEN3_SPMD_TIER=0.
#
# Usage: scripts/check_extraction.sh [path-to-simpler-checkout]

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIMPLER="${1:-$ROOT/..}"

GREEN='\033[32m'; RED='\033[31m'; YELLOW='\033[33m'; RESET='\033[0m'
ok()   { printf "${GREEN}%s${RESET}\n" "$*"; }
warn() { printf "${YELLOW}WARN: %s${RESET}\n" "$*"; }
FAIL=0
fail() { printf "${RED}FAIL: %s${RESET}\n" "$*"; FAIL=$((FAIL + 1)); }

if [[ ! -d "$SIMPLER/src/a2a3/runtime/host_build_graph" ]]; then
    warn "simpler tree not found at $SIMPLER — pass its path as \$1. Skipping."
    exit 0
fi

printf '\n\033[1m### 1/3  TU list matches build_config.py host target\033[0m\n'
# The engine TUs CMake compiles, relative to the runtime dir.
CMAKE_TUS=$(grep -oE '\$\{R\}/[a-z0-9_/]+\.cpp' "$ROOT/CMakeLists.txt" | sed 's|${R}/||' | sort)
# What build_config.py's host target says, minus host/ (runtime_maker + CANN).
EXPECTED=$(cd "$SIMPLER/src/a2a3/runtime/host_build_graph" \
    && find runtime/orchestrator_core runtime/shared orchestration -name '*.cpp' | sort)
if [[ "$CMAKE_TUS" == "$EXPECTED" ]]; then
    ok "$(echo "$CMAKE_TUS" | wc -l) TUs match"
else
    fail "TU list disagrees with the source repo"
    diff <(echo "$CMAKE_TUS") <(echo "$EXPECTED") | sed 's/^/    /'
fi

printf '\n\033[1m### 2/3  every path CMake names exists\033[0m\n'
# A moved or renamed engine header fails the build anyway, but it fails as a
# wall of missing-include errors. Naming the dead path is faster to act on.
MISSING=0
R="$SIMPLER/src/a2a3/runtime/host_build_graph"
while IFS= read -r tu; do
    [[ -f "$R/$tu" ]] || { fail "engine TU not found: $tu"; MISSING=$((MISSING + 1)); }
done <<< "$CMAKE_TUS"
# Include dirs are written against ${R} or ${S}. Read them only from inside the
# target_include_directories block — matching ${R}/... anywhere in the file
# would also pick up the TU list with its .cpp suffix stripped.
while IFS= read -r inc; do
    resolved="${inc/\$\{R\}/$R}"
    resolved="${resolved/\$\{S\}/$SIMPLER/src}"
    [[ -d "$resolved" ]] || { fail "include dir not found: $inc"; MISSING=$((MISSING + 1)); }
done < <(awk '/^target_include_directories\(l2_engine/,/^\)/' "$ROOT/CMakeLists.txt" \
    | grep -oE '\$\{[RS]\}(/[a-zA-Z0-9_/]+)?' | sort -u)
[[ $MISSING -eq 0 ]] && ok "all engine TUs and include dirs resolve"

printf '\n\033[1m### 3/3  qwen3_dynamic_tensormap case: present once, pinned to tier 0\033[0m\n'
CASES=$(find "$ROOT" -name 'qwen3_dynamic_tensormap.h' -not -path "*/build*" 2>/dev/null)
CASE_N=$(echo "$CASES" | grep -c . )
if [[ $CASE_N -eq 0 ]]; then
    fail "qwen3_dynamic_tensormap.h not found in the package"
elif [[ $CASE_N -gt 1 ]]; then
    fail "several copies of the case — the build may pick up the wrong one:"
    echo "$CASES" | sed 's/^/    /'
else
    ok "case present at ${CASES#"$ROOT"/} ($(wc -l < "$CASES") lines)"
fi
# The tier changes the DAG, so a build that lost the pin measures a different
# graph while still printing plausible numbers.
if grep -q 'QWEN3_SPMD_TIER=0' "$ROOT/CMakeLists.txt"; then
    ok "build pins QWEN3_SPMD_TIER=0"
else
    fail "CMakeLists.txt no longer pins QWEN3_SPMD_TIER=0"
fi

echo
if [[ $FAIL -eq 0 ]]; then
    ok "harness corresponds to the repo"
    exit 0
fi
printf "${RED}%d failure(s)${RESET}\n" "$FAIL"
exit 1
