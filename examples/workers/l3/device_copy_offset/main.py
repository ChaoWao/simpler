#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3 partial-copy demo — ``copy_to`` / ``copy_from`` with offsets, no kernels.

A sub-range of a device allocation is named by **the allocation plus an offset**, never by a handle
rebuilt at an interior address: an interior handle carries a canonical identity that names no
allocation at all, and is refused. This example drives the offset form across the fork boundary and
checks the bytes landed where they were addressed.

Why L3 rather than L2 (``examples/workers/l2/worker_malloc``, which covers the whole-buffer form on
one process): at L3 the chip runs in a **forked child**, so neither end of a copy travels as an
address. The parent sends two ``BufferDescriptor``s plus the span, and the child resolves each
descriptor through its own ``ImportRegistry`` and adds the offset to *the base it resolved itself*.
That last step has no other coverage — a parent-side unit test can only prove the descriptor still
names the allocation's own base and that the span went out intact. Applying an offset twice, on the
wrong side, or in the wrong process all produce a valid-looking copy that lands at the wrong address,
which is exactly what the untouched-neighbour assertions below catch.

Kernel-free and does not call ``worker.run()``: the copies are control-plane operations, so the
sequence is malloc -> H2D -> D2H -> free on the Worker API directly.

Run:
    python examples/workers/l3/device_copy_offset/main.py -p a2a3sim -d 0
    python examples/workers/l3/device_copy_offset/main.py -p a2a3   -d 0
"""

import argparse
import sys

from simpler.task_interface import DataType
from simpler.worker import Worker

# Deliberately not a multiple of 4/8/64: an offset is a byte count, and a stride- or element-scaled
# offset anywhere on the path would land the payload somewhere these bounds do not expect.
NBYTES = 4096
DST_OFFSET = 1000
SRC_OFFSET = 24
COPY_NBYTES = 700
FILL = 0xA5


def parse_args() -> argparse.Namespace:
    """Same CLI shape as every example under ``examples/workers/``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-p",
        "--platform",
        required=True,
        choices=["a2a3sim", "a2a3", "a5sim", "a5"],
        help="Target platform.",
    )
    parser.add_argument("-d", "--device", type=int, default=0, help="Device id whose chip child runs the copies.")
    return parser.parse_args()


def _pattern(nbytes: int, seed: int) -> bytes:
    """A deterministic byte pattern; every byte differs from ``FILL`` so a miss is visible."""
    return bytes(((seed + i) * 131 + i * i) % 0xFF for i in range(nbytes))


def _view(handle) -> memoryview:
    """The ``handle.nbytes`` host-visible bytes of a ``create_buffer`` backing.

    Sliced to the Buffer's own extent rather than handed over whole: a POSIX shm mapping is
    page-rounded, and a page is 4 KiB on x86-64 and aarch64 Linux but 16 KiB on Apple silicon, so
    the mapping is a multiple of the requested size on one platform and larger than it on another.
    The copies themselves are unaffected -- they take their length from ``Buffer.nbytes`` -- but a
    whole-mapping view would make every bound below relative to the page size instead of the
    backing.
    """
    shm = handle.shm
    assert shm is not None
    buf = shm.buf
    assert buf is not None
    view = buf[: int(handle.nbytes)]
    assert len(view) == int(handle.nbytes)
    return view


def _first_diff(got: bytes, want: bytes, base: int) -> str:
    for i, (a, b) in enumerate(zip(got, want)):
        if a != b:
            return f"byte {base + i}: got 0x{a:02x}, want 0x{b:02x}"
    return "no differing byte"


def _check_h2d_offsets(worker: Worker, dev_h, host_src, host_dst) -> None:
    """H2D into a sub-range: only ``[DST_OFFSET, DST_OFFSET + COPY_NBYTES)`` may change."""
    src, dst = _view(host_src), _view(host_dst)
    src[:] = _pattern(NBYTES, seed=1)

    # Fill the whole allocation first, so anything the partial copy touches outside its window shows
    # up as a byte that is no longer FILL.
    filler = worker.create_buffer(NBYTES)
    try:
        _view(filler)[:] = bytes([FILL]) * NBYTES
        worker.copy_to(dev_h, filler)
    finally:
        filler.close()

    worker.copy_to(dev_h, host_src, dst_offset=DST_OFFSET, src_offset=SRC_OFFSET, nbytes=COPY_NBYTES)

    dst[:] = bytes(NBYTES)
    worker.copy_from(host_dst, dev_h)
    got = bytes(dst)

    head, want_head = got[:DST_OFFSET], bytes([FILL]) * DST_OFFSET
    assert head == want_head, f"H2D wrote before its window — {_first_diff(head, want_head, 0)}"

    window = got[DST_OFFSET : DST_OFFSET + COPY_NBYTES]
    want_window = bytes(src[SRC_OFFSET : SRC_OFFSET + COPY_NBYTES])
    assert window == want_window, f"H2D window mismatch — {_first_diff(window, want_window, DST_OFFSET)}"

    tail_at = DST_OFFSET + COPY_NBYTES
    tail, want_tail = got[tail_at:], bytes([FILL]) * (NBYTES - tail_at)
    assert tail == want_tail, f"H2D wrote past its window — {_first_diff(tail, want_tail, tail_at)}"
    print(f"[device_copy_offset]   H2D [{DST_OFFSET}, {tail_at}) <- src[{SRC_OFFSET}:] OK, neighbours intact")


def _check_d2h_offsets(worker: Worker, dev_h, host_dst) -> bytes:
    """D2H into a sub-range of the host side: only the addressed window may change."""
    dst = _view(host_dst)
    dst[:] = bytes([FILL]) * NBYTES

    worker.copy_from(host_dst, dev_h, dst_offset=SRC_OFFSET, src_offset=DST_OFFSET, nbytes=COPY_NBYTES)
    got = bytes(dst)

    head, want_head = got[:SRC_OFFSET], bytes([FILL]) * SRC_OFFSET
    assert head == want_head, f"D2H wrote before its window — {_first_diff(head, want_head, 0)}"

    # The device range read back is the one H2D just wrote, so the expected bytes are the same
    # source slice — proving both directions agree about which byte an offset names.
    window = got[SRC_OFFSET : SRC_OFFSET + COPY_NBYTES]
    tail_at = SRC_OFFSET + COPY_NBYTES
    tail, want_tail = got[tail_at:], bytes([FILL]) * (NBYTES - tail_at)
    assert tail == want_tail, f"D2H wrote past its window — {_first_diff(tail, want_tail, tail_at)}"
    print(f"[device_copy_offset]   D2H host[{SRC_OFFSET}, {tail_at}) <- dev[{DST_OFFSET}:] OK, neighbours intact")
    return window


def _check_bounds_are_enforced(worker: Worker, dev_h, host_src) -> None:
    """An offset must not walk a legal-looking length past the registered allocation."""
    cases = {
        "offset + length past the device backing": {"dst_offset": NBYTES - 8, "nbytes": 9},
        "offset past the device backing": {"dst_offset": NBYTES + 1, "nbytes": 0},
        "negative device offset": {"dst_offset": -1, "nbytes": 1},
        "offset + length past the host backing": {"src_offset": NBYTES - 8, "nbytes": 9},
    }
    for what, kwargs in cases.items():
        try:
            worker.copy_to(dev_h, host_src, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"copy_to accepted a range it must refuse: {what} ({kwargs})")
    print(f"[device_copy_offset]   {len(cases)} out-of-range spans refused")


def run(platform: str, device_id: int) -> int:
    """Core logic — callable from both CLI and pytest."""
    print(f"[device_copy_offset] platform={platform} device={device_id}")
    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=[device_id],
        num_sub_workers=0,
    )
    worker.init()
    host_src = host_dst = None
    try:
        # The host end of an L3 copy is a Buffer, never a raw address: the chip child maps it by name
        # in its own process, where a parent address means nothing.
        host_src = worker.create_buffer(NBYTES)
        host_dst = worker.create_buffer(NBYTES)
        dev_h = worker.alloc_child_tensor(worker_id=0, shapes=(NBYTES,), dtype=DataType.UINT8)
        try:
            _check_h2d_offsets(worker, dev_h, host_src, host_dst)
            window = _check_d2h_offsets(worker, dev_h, host_dst)
            want = bytes(_view(host_src)[SRC_OFFSET : SRC_OFFSET + COPY_NBYTES])
            assert window == want, f"D2H window mismatch — {_first_diff(window, want, SRC_OFFSET)}"
            _check_bounds_are_enforced(worker, dev_h, host_src)
        finally:
            worker.free(dev_h)
        print("[device_copy_offset] all offset checks PASSED")
    finally:
        for handle in (host_src, host_dst):
            if handle is not None:
                handle.close()
        worker.close()
    return 0


def main() -> int:
    args = parse_args()
    return run(args.platform, args.device)


if __name__ == "__main__":
    sys.exit(main())
