# Group reservation scene test

This scene deterministically exercises overlapping `NEXT_LEVEL` group
reservations on workers 0, 1, and 2:

```text
group {0,1}: worker 1 waits for a device-side notification
group {1,2}: remains at the group queue head while worker 1 is busy
single {0}:  must run because worker 0 is unrelated; it releases worker 1
single {2}:  must wait behind group {1,2}
```

The second group's worker-2 task records a device-side marker before the final
single reads it. A result of `1` proves the single did not overtake the
reservation. Completion of the whole test proves the blocked group did not
reserve unrelated worker 0.

Run on the 3-device simulator:

```bash
python -m pytest tests/st/worker/collectives/group_reservation \
  --platform a2a3sim --device 0-2 -v
```
