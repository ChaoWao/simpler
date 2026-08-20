#!/usr/bin/env bash
#
# Full bench + profile of the L2 orchestration sequence, driven by
# qwen3_dynamic_tensormap.h at QWEN3_SPMD_TIER=0.
#
#   1. Throughput, uninstrumented — the numbers to quote.
#   2. Three-level profile (four lines / intercepted ops table / engine's own
#      per-STEP counters).
#   3. Cold/warm attribution: how much of each STEP is first-touch page cost
#      rather than engine work. Without this, STEP 5 reads as 66% of submit cost
#      and the real bottleneck (STEP 3) is invisible. Needs the Level-4 build
#      for the TensorMap chain stats.
#
# No engine source is modified; see the header of bench/l2_profile.h for which
# seams the timestamps come from.
#
# Usage: scripts/profile_l2.sh [--repeat N] [--jobs N]

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REPEAT=5
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repeat) REPEAT="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

GREEN='\033[32m'; RED='\033[31m'; YELLOW='\033[33m'; RESET='\033[0m'
step()  { printf '\n\033[1m### %s\033[0m\n' "$*"; }
green() { printf "${GREEN}%s${RESET}\n" "$*"; }
warn()  { printf "${YELLOW}WARN: %s${RESET}\n" "$*"; }
FAILURES=0
fail()  { printf "${RED}FAIL: %s${RESET}\n" "$*"; FAILURES=$((FAILURES + 1)); }

LIBSTDCXX_DIR="$(dirname "$(readlink -f "$(g++ -print-file-name=libstdc++.so)")")"
export LD_LIBRARY_PATH="${LIBSTDCXX_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

REPORTS="$ROOT/reports"
LOGS="$ROOT/build/logs"
mkdir -p "$REPORTS" "$LOGS"

# Pin to a small CPU set, not one CPU. Everything here is single-threaded (the
# orchestrator is single-writer by design), so a 4-CPU set costs nothing and
# keeps the run off whichever core the OS is busy with.
PIN=()
if command -v taskset > /dev/null 2>&1 && taskset -c 0-3 true 2>/dev/null; then
    PIN=(taskset -c 0-3)
else
    warn "taskset unavailable — measurements will be noisier"
fi

# NOTHING is ever reclaimed here (no scheduler, no completions), so both the
# task window and the heap must hold the entire graph at once. These are the
# binary's own defaults, restated so a change to either is visible in one place.
ARGS=(--task-window=8192 --heap-mb=2048)

# ---------------------------------------------------------------------------
step "Build: default, SIMPLER_ORCH_PROFILING=1, and +TENSORMAP"
# ---------------------------------------------------------------------------
cmake -B build -S . > "$LOGS/build-default.log" 2>&1 \
    && cmake --build build --parallel "$JOBS" >> "$LOGS/build-default.log" 2>&1 \
    || fail "default build (see $LOGS/build-default.log)"

cmake -B build-prof -S . -DL2_ORCH_PROFILING=1 > "$LOGS/build-prof.log" 2>&1 \
    && cmake --build build-prof --parallel "$JOBS" >> "$LOGS/build-prof.log" 2>&1 \
    || fail "profiling build (see $LOGS/build-prof.log)"

# Level 4 adds the TensorMap chain counters, which is what makes STEP 3's cost
# attributable to a root cause rather than just large.
cmake -B build-tm -S . -DL2_TENSORMAP_PROFILING=1 > "$LOGS/build-tm.log" 2>&1 \
    && cmake --build build-tm --parallel "$JOBS" >> "$LOGS/build-tm.log" 2>&1 \
    || fail "tensormap-profiling build (see $LOGS/build-tm.log)"
green "built build/ (quote from this), build-prof/ (L3), build-tm/ (L3+L4)"

# ---------------------------------------------------------------------------
step "1/3  Throughput (uninstrumented)"
# ---------------------------------------------------------------------------
"${PIN[@]}" ./build/l2_bench --mode=throughput --repeat="$REPEAT" "${ARGS[@]}" || fail "throughput"

# ---------------------------------------------------------------------------
step "2/3  Per-step profile"
# ---------------------------------------------------------------------------
"${PIN[@]}" ./build-prof/l2_bench --mode=profile "${ARGS[@]}" \
    --out="$REPORTS/profile-qwen3-dyn.json" | tee "$REPORTS/profile-qwen3-dyn.txt" \
    || fail "profile"

# ---------------------------------------------------------------------------
step "3/3  Cold/warm attribution (which STEPs are really first-touch cost)"
# ---------------------------------------------------------------------------
# Prefaulting is a DIAGNOSTIC, not an optimization: it moves first-touch cost out
# of the measured window, it does not avoid it. The production fix for the SM
# half is to pool the host mirror across runs (runtime_maker.cpp:487 allocates it
# fresh every time).
echo
"${PIN[@]}" ./build-tm/l2_bench --mode=profile "${ARGS[@]}" \
    > "$LOGS/attr-cold.txt" 2>/dev/null || fail "attribution (cold)"
"${PIN[@]}" ./build-tm/l2_bench --mode=profile "${ARGS[@]}" --prefault-all \
    > "$LOGS/attr-warm.txt" 2>/dev/null || fail "attribution (warm)"
python3 "$ROOT/scripts/cold_warm_table.py" "$LOGS/attr-cold.txt" "$LOGS/attr-warm.txt" \
    || fail "attribution table"
sed -n '/LEVEL 4/,/wasted/p' "$LOGS/attr-warm.txt" | sed 's/^/  /'

echo
echo "Reports: $REPORTS/profile-qwen3-dyn.txt (human), profile-qwen3-dyn.json"
if [[ $FAILURES -eq 0 ]]; then
    green "profile complete"
    exit 0
fi
printf "${RED}%d failure(s)${RESET}\n" "$FAILURES"
exit 1
