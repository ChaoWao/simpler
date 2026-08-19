#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Manual HBG coverage for the disabled A5 SPMD paged-attention workload."""

from copy import deepcopy

from simpler_setup import scene_test
from tests.st.a5.tensormap_and_ringbuffer.spmd_paged_attention.test_spmd_paged_attention import (
    TestSpmdPagedAttentionA5 as _TmrBase,
)


@scene_test(level=2, runtime="host_build_graph")
class TestSpmdPagedAttentionHbgA5(_TmrBase):
    # The TMR shapes, run through host_build_graph. Deep-copied so the two
    # classes do not share a params dict. Onboard only: the MIX kernel's
    # TPUSH/TPOP path does not build for the CPU simulator (PTO-ISA rejects the
    # ColMajor TMULS in the softmax tail), so a5sim is out of scope. See #1832.
    # Every HBG case stays manual; the Per-PR sweep covers these shapes on the
    # TMR side.
    CASES = [{**deepcopy(case), "platforms": ["a5"], "manual": True} for case in _TmrBase.CASES]


if __name__ == "__main__":
    TestSpmdPagedAttentionHbgA5.run_module(__name__)
