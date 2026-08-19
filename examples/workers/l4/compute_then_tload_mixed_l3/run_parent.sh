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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

: "${SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_REMOTE:?set remote L3 daemon as HOST:PORT}"
: "${SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_LOCAL_DEVICES:=0}"
: "${SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_REMOTE_DEVICES:=0}"
: "${SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_PLATFORM:=a2a3}"
: "${SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_RUNTIME:=tensormap_and_ringbuffer}"
: "${SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_COMM_PROFILE:=a3-fabric-v1}"
: "${SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_SESSION_LISTEN_HOST:=0.0.0.0}"
: "${SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_SESSION_TIMEOUT:=120}"

cd "${ROOT_DIR}"
# The network1 job builds the venv in an earlier step, so a staging failure reaches
# this script before it reaches python — name it here rather than let `set -e`
# abort on a bare "No such file or directory".
if [[ ! -f .venv/bin/activate ]]; then
  echo "error: ${ROOT_DIR}/.venv/bin/activate not found; create the virtual environment first" >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate

exec python -m examples.workers.l4.compute_then_tload_mixed_l3.main \
  --remote "${SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_REMOTE}" \
  --local-devices "${SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_LOCAL_DEVICES}" \
  --remote-devices "${SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_REMOTE_DEVICES}" \
  --platform "${SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_PLATFORM}" \
  --runtime "${SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_RUNTIME}" \
  --comm-profile "${SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_COMM_PROFILE}" \
  --session-listen-host "${SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_SESSION_LISTEN_HOST}" \
  --session-timeout "${SIMPLER_COMPUTE_THEN_TLOAD_MIXED_L3_SESSION_TIMEOUT}"
