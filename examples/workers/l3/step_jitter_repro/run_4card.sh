#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
output_dir=${1:?usage: run_4card.sh OUTPUT_DIR [ROUNDS] [KERNEL_REPEATS]}
rounds=${2:-1000}
kernel_repeats=${3:-4096}
python_bin=${PYTHON:-python}
devices=${TASK_DEVICE:?TASK_DEVICE must contain four allocated device IDs}

cd "$repo_root"
mkdir -p "$output_dir"

SIMPLER_LOG_LEVEL=TIMING "$python_bin" \
  examples/workers/l3/step_jitter_repro/main.py \
  -p a2a3 -d "$devices" --warmup 5 --rounds "$rounds" --depth 2 \
  --kernel-repeats "$kernel_repeats" >"$output_dir/run.log" 2>&1

"$python_bin" examples/workers/l3/step_jitter_repro/analyze_strace.py \
  "$output_dir/run.log" --warmup 5 --rounds "$rounds" \
  --json-out "$output_dir/analysis.json" \
  --trace-out "$output_dir/swimlane.json"
