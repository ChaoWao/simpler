# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 FLASH decode on host_build_graph: same program, host-run orchestration.

The 43-layer network, its 367 kernels and its fixture come from the
``tensormap_and_ringbuffer`` case; only the runtime changes. HBG compiles the
orchestration with the host g++, runs it on the host CPU instead of the AICPU,
and ships the built SM image to the device, which boots scheduler-only.

``kernels/orchestration/decode_fwd_graph.cpp`` is that case's orchestration
with the runtime untouched, recast as a Graph: the 20-iteration decoder layer
loop (40 of the 43 layers) submits one ``rt_submit_graph`` per iteration whose
body is the layer's full task set, so the host records a 744-node Definition
once and the 15991-task network collapses to 1131 host-submitted tasks. The ten
``get_tensor_data`` reads of ``recv_count_out`` (a task-produced tensor,
unreadable while the host builds the graph) stand in a constant, and the six
``set_initial_value`` calls are dropped because their target is a GM-heap
device address. The graph therefore keeps the size and shape of the real one
but not the fixture's routing — hence ``skip_golden``. ``manual`` because the
367-kernel compile takes minutes.

Host construction and Graph recording complete (1131 tasks on host); device
execution of the non-graph remainder currently stalls 12 tasks from the end.
See README.md for the measurement, what has been ruled out, and what is still
open.

    python examples/a2a3/host_build_graph/deepseek_v4_flash_decode/\\
test_deepseek_v4_flash_decode.py -p a2a3 -d <d0>,<d1> --manual only
"""

import copy
import importlib.util
import sys
from pathlib import Path

from simpler_setup import SceneTestCase, scene_test
from simpler_setup.goldens.deepseek_v4_flash_decode import N_RANKS, generate_inputs

HERE = Path(__file__).resolve().parent
TMR_CASE_DIR = HERE.parents[1] / "tensormap_and_ringbuffer/deepseek_v4_flash_decode"


def _load_tmr_case():
    module_name = "_dsv4_flash_tmr_base"
    spec = importlib.util.spec_from_file_location(module_name, TMR_CASE_DIR / "test_deepseek_v4_flash_decode.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the TMR deepseek_v4_flash_decode case")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_TMR = _load_tmr_case()


def _host_build_graph_callable():
    """Same CALLABLE as the TMR case, with kernel sources re-pointed at the TMR dir
    and the orchestration swapped for the predicated-dispatch variant.

    ``decode_fwd_hostbuild.cpp`` is the tensormap_and_ringbuffer orchestration
    with two edits, 51 lines in all, and the runtime untouched. HBG runs the
    orchestrator on the host before the device executes anything, so the ten
    ``get_tensor_data(recv_count_out)`` reads (a task-produced tensor driving the
    MoE per-expert tile loops) have no value to return; they are replaced by
    ``HBG_RECV_ROWS_PER_EXPERT``, which holds the tile loops at their real trip
    count. The six ``set_initial_value`` calls are dropped because the host
    cannot store to the GM-heap address they target. The other 31
    ``get_tensor_data`` reads are external tensors the runtime stages with a host
    view, and are left alone.

    The graph therefore keeps the size and shape of the real one but not the
    fixture's routing, which is why the case is ``skip_golden``: it measures
    host-side graph construction and device execution, not numerics.
    """
    callable_config = copy.deepcopy(_TMR.TestDeepseekV4FlashDecode.CALLABLE)
    chip = callable_config["callables"][0]
    chip["orchestration"]["source"] = str(HERE / "kernels/orchestration/decode_fwd_graph.cpp")
    for incore in chip["incores"]:
        incore["source"] = str(TMR_CASE_DIR / incore["source"])
    return callable_config


@scene_test(level=3, runtime="host_build_graph")
class TestDeepseekV4FlashDecodeHostBuildGraph(SceneTestCase):
    """DSv4 FLASH EP2/TP2 decode with the orchestration built on the host."""

    CALLABLE = _host_build_graph_callable()
    CASES = [
        {
            "name": "DecodeFwdEP2TP2",
            "platforms": ["a2a3"],
            "manual": True,
            "skip_golden": True,
            "config": {
                "device_count": N_RANKS,
                "num_sub_workers": 0,
                # Ring task window matches the TMR case (the predicated rewrite
                # materializes the same per-expert tile count the dynamic loops
                # produced in-regime). The heap grows 2x: the static grid
                # allocates tile scratch for all 32 experts per MoE layer, while
                # the dynamic loops allocated only for experts that received rows.
                "runtime_env": {
                    "ring_task_window": 16384,
                    "ring_heap": 2 << 30,
                    "ring_dep_pool": 16384,
                },
            },
            "params": {"seed": 1234},
        }
    ]

    def generate_args(self, params):
        return generate_inputs(params.get("seed", 1234))

    def compute_golden(self, args, params):
        raise NotImplementedError(
            "deepseek_v4_flash_decode is a completion/smoke case (skip_golden): no "
            "full-network torch reference exists upstream either. Component-level "
            "goldens live with the standalone kernels in pypto-lib."
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
