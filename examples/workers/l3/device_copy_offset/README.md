# L3 partial copy — `copy_to` / `copy_from` with offsets

Drives the offset form of the two control-plane copies across the fork boundary and checks the
bytes landed where they were addressed. No kernels, no `worker.run()`.

```bash
python examples/workers/l3/device_copy_offset/main.py -p a2a3sim -d 0
python examples/workers/l3/device_copy_offset/main.py -p a2a3   -d 0
```

## What it shows

A sub-range of a device allocation is named by **the allocation plus an offset**:

```python
dev = worker.alloc_child_tensor(worker_id=0, shapes=(4096,), dtype=DataType.UINT8)
worker.copy_to(dev, host_src, dst_offset=1000, src_offset=24, nbytes=700)
worker.copy_from(host_dst, dev, dst_offset=24, src_offset=1000, nbytes=700)
```

`nbytes` defaults to the rest of the **host** side after that side's offset, so a plain
`copy_to(dev, host_src)` still transfers the whole host backing.

There is deliberately no way to name a sub-range with a handle rebuilt at an interior address: such
a handle carries a canonical identity that names no allocation, and is refused. Both offsets are
bounded together with the length against the *registered* allocation, so an offset cannot walk a
legal-looking length past the end.

## Why this is an L3 example

[`../../l2/worker_malloc`](../../l2/worker_malloc) already covers the whole-buffer form, but its
chip runs in the calling process. At L3 the chip is a **forked child**, so neither end of a copy
travels as an address: the parent sends two `BufferDescriptor`s plus the span, and the child
resolves each descriptor through its own `ImportRegistry` and adds the offset to *the base it
resolved itself*.

That last step is what this example exists for. A parent-side unit test can only prove the
descriptor still names the allocation's own base and that the span went out intact — applying an
offset twice, on the wrong side, or in the wrong process all produce a valid-looking copy that
lands at the wrong address. So the checks here assert the untouched neighbours as well as the
window: the allocation is filled with a marker byte first, and every byte outside
`[offset, offset + nbytes)` must still carry it afterwards.
