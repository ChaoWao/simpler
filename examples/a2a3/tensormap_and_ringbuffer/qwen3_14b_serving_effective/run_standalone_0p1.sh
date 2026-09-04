#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -eo pipefail

DEVICE_ID="${1:?usage: run_standalone_0p1.sh DEVICE_ID OUTPUT_ROOT [MODE] [STEPS]}"
OUTPUT_ROOT="${2:?usage: run_standalone_0p1.sh DEVICE_ID OUTPUT_ROOT [MODE] [STEPS]}"
MODE="${3:-single}"
STEPS="${4:-127}"
: "${PYTHON_BIN:?set PYTHON_BIN to the isolated Python interpreter}"
: "${PYPTO_ROOT:?set PYPTO_ROOT to the PyPTO checkout}"
: "${PYPTO_LIB_ROOT:?set PYPTO_LIB_ROOT to the matching pypto-lib checkout}"
: "${PTOAS_ROOT:?set PTOAS_ROOT to the matching PTOAS installation}"
: "${FIXTURE_ROOT:?set FIXTURE_ROOT to the prefill KV snapshot}"
: "${MODEL_DIR:?set MODEL_DIR to the Qwen3-14B checkpoint}"
: "${ARTIFACT_SOURCE:?set ARTIFACT_SOURCE to the decode artifact}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
case "$MODE" in
  single) BENCHMARK="$SCRIPT_DIR/benchmark.py" ;;
  dual) BENCHMARK="$SCRIPT_DIR/benchmark_dual.py" ;;
  *) echo "mode must be single or dual: $MODE" >&2; exit 2 ;;
esac
if (( STEPS < 1 || STEPS > 127 )); then
  echo "steps must be in [1, 127]: $STEPS" >&2
  exit 2
fi
STEADY_SKIP=5
if (( STEPS <= STEADY_SKIP )); then
  STEADY_SKIP=0
fi
RESULT="$OUTPUT_ROOT/results/${MODE}_0warmup_1measured"
ARTIFACT_WORK="$OUTPUT_ROOT/artifact_work/${MODE}_0warmup_1measured"
RUNLOG="$OUTPUT_ROOT/logs/${MODE}_0warmup_1measured"

test ! -e "$RESULT"
test ! -e "$ARTIFACT_WORK"
eval "$(pypto-setup --export)"
test -f "$ASCEND_HOME_PATH/set_env.sh"
source "$ASCEND_HOME_PATH/set_env.sh"
set -u

mkdir -p "$OUTPUT_ROOT/results" "$OUTPUT_ROOT/artifact_work" "$RUNLOG"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PYPTO_ROOT/python:$PYPTO_LIB_ROOT:$REPO_ROOT/python:$SCRIPT_DIR"
export PATH="$PTOAS_ROOT/bin:$PATH"
export PYPTO_PROG_BUILD_DIR="$OUTPUT_ROOT/compile_cache"
export ASCEND_PROCESS_LOG_PATH="$RUNLOG/ascend"
export SIMPLER_DEVICE_STRACE_ENABLE=1
unset PTO2_OP_EXECUTE_TIMEOUT_US PTO2_STREAM_SYNC_TIMEOUT_MS
unset PTO2_RING_DEP_POOL PTO2_RING_TASK_WINDOW PTO2_RING_HEAP
unset PYPTO_RING_DEP_POOL PYPTO_RING_TASK_WINDOW PYPTO_RING_HEAP

mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$PYPTO_PROG_BUILD_DIR"
npu-smi info -t board -i "$DEVICE_ID" -c 0 > "$RUNLOG/device_before.txt" 2>&1 || true
capture_after() {
  npu-smi info -t board -i "$DEVICE_ID" -c 0 > "$RUNLOG/device_after.txt" 2>&1 || true
}
trap capture_after EXIT

"$PYTHON_BIN" "$BENCHMARK" \
  --fixture "$FIXTURE_ROOT" \
  --fixture-module "$SCRIPT_DIR/fixture.py" \
  --weights-module "$SCRIPT_DIR/weights.py" \
  --model-dir "$MODEL_DIR" \
  --artifact-source "$ARTIFACT_SOURCE" \
  --artifact-work-dir "$ARTIFACT_WORK" \
  --output-dir "$RESULT" \
  --mode "$MODE" \
  --device-id "$DEVICE_ID" \
  --warmup-runs 0 \
  --measured-runs 1 \
  --steady-skip "$STEADY_SKIP" \
  --inter-run-wait 0 \
  --qualification-steps "$STEPS" \
  2>&1 | tee "$RUNLOG/run.log"

cp "$RUNLOG/run.log" "$RESULT/run.log"
cp "$RUNLOG/device_before.txt" "$RESULT/device_before.txt"
"$PYTHON_BIN" "$SCRIPT_DIR/trace_effective.py" \
  --log "$RESULT/run.log" \
  --output "$RESULT/trace_summary.json" \
  --warmup-runs 0 \
  --measured-runs 1 \
  --steps "$STEPS" \
  --steady-skip "$STEADY_SKIP"
"$PYTHON_BIN" "$SCRIPT_DIR/export_strace_timeline.py" \
  --log "$RESULT/run.log" \
  --output-dir "$RESULT/profile"

if [[ "$MODE" == "dual" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/validate_dual_result.py" \
    --result "$RESULT" \
    --fixture "$FIXTURE_ROOT" \
    --golden-manifest "$FIXTURE_ROOT/../../golden/qualification/manifest.json" \
    --steps "$STEPS"
fi

checksum_files=(
  "$RESULT/benchmark.json"
  "$RESULT/trace_summary.json"
  "$RESULT/profile/simpler_strace_timeline.json"
)
if [[ "$MODE" == "dual" ]]; then
  checksum_files+=("$RESULT/dual_validation.json")
fi
sha256sum "${checksum_files[@]}" > "$RESULT/SHA256SUMS"
