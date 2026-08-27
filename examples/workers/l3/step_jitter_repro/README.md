# Simpler-only multi-chip step-jitter reproducer

This example depends only on the current Simpler checkout and its normal runtime dependencies. It does not import
PyPTO, pypto-lib, or any serving package.

One logical step submits a four-member chip group. Two top-level `Worker.submit` handles remain in flight, and the
oldest handle is retired only after its successor has been submitted. Each chip run submits repeated vector work so
the device remains busy long enough for host scheduling jitter to be observable.

```bash
export SIMPLER_LOG_LEVEL=TIMING
python examples/workers/l3/step_jitter_repro/main.py \
  -p a2a3 -d "$TASK_DEVICE" --warmup 5 --rounds 1000 --depth 2 \
  --kernel-repeats 4096 2>&1 | tee /tmp/simpler-step-jitter.log

python examples/workers/l3/step_jitter_repro/analyze_strace.py \
  /tmp/simpler-step-jitter.log --warmup 5 --rounds 1000 \
  --json-out /tmp/simpler-step-jitter-analysis.json \
  --trace-out /tmp/simpler-step-jitter-swimlane.json
```

The wrapper below runs the workload and writes the raw STRACE, analysis, and merged four-device swimlane to one
output directory. `TASK_DEVICE` must contain exactly four comma-separated device IDs:

```bash
TASK_DEVICE=0,1,2,3 PYTHON=.venv/bin/python \
  bash examples/workers/l3/step_jitter_repro/run_4card.sh /tmp/simpler-jitter
```

On shared hardware, run this wrapper inside a four-device allocation so all devices remain reserved for the complete
workload.
