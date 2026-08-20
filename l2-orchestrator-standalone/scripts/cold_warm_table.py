"""Per-STEP cold vs prefaulted table, from two --mode=profile reports.

The first-touch share is the whole point: a STEP that is large cold and small
warm was paging, not orchestrating, and attributing its cost to engine work
sends optimization at the wrong target.
"""

import re
import sys


def steps(path):
    out = []
    for line in open(path):
        m = re.match(r"\s{2}(STEP \d.*?)\s+(\d+)\s+([\d.]+)\s+([\d.]+)%\s*$", line)
        if m:
            out.append((m.group(1).strip(), float(m.group(3)), float(m.group(4))))
    return out


def main(cold_path, warm_path):
    cold, warm = steps(cold_path), steps(warm_path)
    if not cold or not warm:
        print("no STEP rows found — was this a Level-3 build?", file=sys.stderr)
        return 1
    print("qwen3-dyn — per-STEP totals, cold vs prefaulted (ms)")
    print(f"  {'step':<44}{'cold':>10}{'warm':>10}{'warm %':>9}  {'first-touch share':>18}")
    print("  " + "-" * 93)
    for (name, cms, _), (_, wms, wpct) in zip(cold, warm):
        ft = 100.0 * (cms - wms) / cms if cms > 0 else 0.0
        print(f"  {name:<44}{cms:>10.3f}{wms:>10.3f}{wpct:>8.1f}%  {ft:>17.0f}%")
    print("  " + "-" * 93)
    print(f"  {'TOTAL':<44}{sum(x[1] for x in cold):>10.3f}{sum(x[1] for x in warm):>10.3f}")
    print("  first-touch share = how much of the cold number was paging, not engine work.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
