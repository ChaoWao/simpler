#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU checks for standalone single/dual buffer selection and slot metadata."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

import benchmark


DUAL_PATH = Path(__file__).with_name("benchmark_dual.py")
DUAL_SPEC = importlib.util.spec_from_file_location("benchmark_dual", DUAL_PATH)
assert DUAL_SPEC is not None and DUAL_SPEC.loader is not None
benchmark_dual = importlib.util.module_from_spec(DUAL_SPEC)
sys.modules[DUAL_SPEC.name] = benchmark_dual
DUAL_SPEC.loader.exec_module(benchmark_dual)


def main() -> int:
    assert callable(benchmark._run_sequence)
    assert callable(benchmark_dual._run_sequence)

    initial = torch.full((16 * 32,), -1, dtype=torch.int32)
    initial.view(16, 32)[:, :27] = torch.arange(432, dtype=torch.int32).view(16, 27)
    slots = []
    for _ in range(2):
        slots.append(
            {
                "seq_lens": torch.zeros(16, dtype=torch.int32),
                "slot_mapping": torch.zeros(16, dtype=torch.int32),
                "block_table": initial.clone(),
                "sampled_ids_host": torch.zeros((16, 8), dtype=torch.int32),
            }
        )
    golden = {
        "seq_lens": torch.full((2, 16), 3457, dtype=torch.int32),
        "slot_mapping": torch.stack(
            (torch.arange(432, 448, dtype=torch.int32) * 128,) * 2
        ),
    }
    benchmark._update_slot(slots[0], golden, 0)
    assert torch.all(slots[1]["block_table"].view(16, 32)[:, 27] == -1)
    benchmark._update_slot(slots[1], golden, 1)
    assert torch.equal(
        slots[0]["block_table"].view(16, 32)[:, 27],
        torch.arange(432, 448, dtype=torch.int32),
    )
    assert torch.equal(
        slots[1]["block_table"].view(16, 32)[:, 27],
        torch.arange(432, 448, dtype=torch.int32),
    )
    print("standalone single/dual CPU checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
