#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3 post-fork zero-copy host buffers (issue #1027 #1).

A host tensor created *after* the chip children are forked (lazily on the
first ``Worker.run()``) is not visible to those children: the orch fn runs in
the parent and carries a raw parent VA that is unmapped (or stale) in the child.
``Worker.create_buffer`` allocates a POSIX-shm backing and hands back a
``Buffer``; naming it in a task arg with ``handle.tensor(...)`` carries a
self-describing descriptor that the forked chip child materializes lazily on
first receipt (map-once), so a tensor built over ``handle.shm.buf`` round-trips
with **no per-run copy** — the child reads and writes the same physical pages the
parent sees.

Covers the mechanism end-to-end (allocate a post-fork buffer, fill it in place,
run, read the result back).

a2a3sim: ``create_buffer`` is pure host-side (POSIX shm; the descriptor rides in
the mailbox, no eager broadcast) with no platform branching, so the sim backend
exercises the full mechanism without needing a device. The vector_example
orchestration kernels exist only for a2a3.
"""

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType

from simpler_setup import SceneTestCase, scene_test

KERNELS_BASE = "../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"

SIZE = 128 * 128
DTYPE = torch.float32
_F32 = DataType.FLOAT32


def _golden(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    s = a + b
    return (s + 1) * (s + 2) + s


def _one_task_orch(chip_handle, ba, bb, bout):
    def orch_fn(orch, _args, cfg):
        ta = TaskArgs()
        ta.add_tensor(ba.tensor((SIZE,), _F32), TensorArgType.INPUT)
        ta.add_tensor(bb.tensor((SIZE,), _F32), TensorArgType.INPUT)
        ta.add_tensor(bout.tensor((SIZE,), _F32), TensorArgType.OUTPUT_EXISTING)
        orch.submit_next_level(chip_handle, ta, cfg, worker=0)

    return orch_fn


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestPostForkHostBufferZeroCopy(SceneTestCase):
    """Post-fork zero-copy host buffers on a single L3 worker (issue #1027 #1)."""

    CALLABLE = {
        "callables": [
            {
                "name": "vector",
                "orchestration": {
                    "source": f"{KERNELS_BASE}/orchestration/example_orchestration.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.IN, D.IN, D.OUT],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": f"{KERNELS_BASE}/aiv/kernel_add.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT],
                    },
                    {
                        "func_id": 1,
                        "source": f"{KERNELS_BASE}/aiv/kernel_add_scalar.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT],
                    },
                    {
                        "func_id": 2,
                        "source": f"{KERNELS_BASE}/aiv/kernel_mul.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT],
                    },
                ],
            },
        ],
    }

    CASES = [
        {"name": "post_fork_zero_copy", "platforms": ["a2a3sim"]},
    ]

    def test_run(self, st_worker):
        """Zero-copy: buffers allocated AFTER the fork via ``create_buffer``,
        filled in place, run, and read back — all without a per-run copy.

        ``Worker.init()`` is eager, so the chip child is already forked when the
        buffers are created; a post-init ``create_buffer`` reaches that child by
        naming the handle as a ``Tensor`` the child materializes lazily.
        """
        worker = st_worker
        chip_handle = type(self)._st_chip_handles["vector"]

        nbytes = SIZE * DTYPE.itemsize  # element count × dtype size, not a magic 4
        ba = worker.create_buffer(nbytes)
        bb = worker.create_buffer(nbytes)
        bout = worker.create_buffer(nbytes)
        a = b = out = None
        result = False
        try:
            a = torch.frombuffer(ba.shm.buf, dtype=DTYPE, count=SIZE)
            b = torch.frombuffer(bb.shm.buf, dtype=DTYPE, count=SIZE)
            out = torch.frombuffer(bout.shm.buf, dtype=DTYPE, count=SIZE)
            a.fill_(5.0)  # in place → lands directly in the child-visible shm
            b.fill_(7.0)
            out.zero_()
            worker.run(_one_task_orch(chip_handle, ba, bb, bout), args=None, config=CallConfig())
            result = torch.allclose(out, _golden(a, b), rtol=self.RTOL, atol=self.ATOL)
        finally:
            # Drop the views before freeing so the shm releases promptly, and do it
            # in finally so a run()/assert failure above still cleans up all three
            # buffers instead of leaving live views warning on the first free.
            del a, b, out
            ba.close()
            bb.close()
            bout.close()
        assert result


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
