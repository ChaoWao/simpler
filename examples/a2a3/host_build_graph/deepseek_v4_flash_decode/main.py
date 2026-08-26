#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 FLASH decode on host_build_graph: same program, host-run orchestration.

The 43-layer network, its 368 kernels and its parameter table come from the
``tensormap_and_ringbuffer`` case; only the runtime changes. HBG compiles the
orchestration with the host g++, runs it on the host CPU instead of the AICPU,
and ships the built SM image to the device, which boots scheduler-only.

``kernels/orchestration/decode_fwd_graph.cpp`` is that case's orchestration
with the runtime untouched, recast as a Graph: the whole forward pass is cut
into Graph blocks covering all 43 layers, so the host records eight
Definitions and submits 129 tasks itself, the rest being built on the recording
threads. The ten ``recv_count_out`` reads that drove the MoE per-expert tile
loops (a task-produced tensor, unreadable while the host builds the graph) are
now dispatch predicates the scheduler evaluates on device. Tensor initialization
is shared with the TMR case and runs in AIV kernels, and the attention/FFN scale
factors travel as kernel tensor inputs read from GM, so the host never writes a
GM-heap device address and never copies tensor data into task scalars. The
orchestration source is therefore the TMR one recast as a Graph, with no
runtime-specific rewrite.

The one ``get_tensor_data`` read left (``num_tokens_per_owner``) drives
``set_block_num``, which a dispatch predicate cannot express. That is why the
shared driver keeps this one parameter host-backed while every other parameter
lives in child memory: a child-memory tensor reaches the device without a host
view, and this runtime's orchestrator has to read it on the host.

Run (2 dies; the pytest wrapper next to this file is the same thing under
``--manual only``):

    python examples/a2a3/host_build_graph/deepseek_v4_flash_decode/main.py -p a2a3 -d <d0>,<d1>

See README.md for the measurements and for the fixes that got the replay running.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TMR_CASE_DIR = HERE.parents[1] / "tensormap_and_ringbuffer/deepseek_v4_flash_decode"
RUNTIME = "host_build_graph"
ORCHESTRATION_SOURCE = HERE / "kernels/orchestration/decode_fwd_graph.cpp"


def _load_tmr_driver():
    """The TMR case's driver module — same kernels, same parameter table, same flags."""
    module_name = "_dsv4_flash_tmr_driver"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, TMR_CASE_DIR / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the TMR deepseek_v4_flash_decode driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run(device_ids, platform: str, **kwargs) -> int:
    """Run the TMR case's program with the Graph-form orchestration on this runtime."""
    driver = _load_tmr_driver()
    kwargs.setdefault("runtime", RUNTIME)
    kwargs.setdefault("orchestration_source", ORCHESTRATION_SOURCE)
    return driver.run(device_ids, platform, **kwargs)


def main(argv=None) -> int:
    driver = _load_tmr_driver()
    return driver.main(argv, runtime=RUNTIME, orchestration_source=ORCHESTRATION_SOURCE)


if __name__ == "__main__":
    sys.exit(main())
