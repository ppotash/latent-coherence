#!/usr/bin/env python3
"""Where a model's money and time are going, and what finishing will cost.

  python scripts/project_cost.py --run runs/main --model grok-4-7 --cap synthetic-full=5

Per (family, spec) group: answered vs remaining prompts, mean reasoning tokens, mean cost
and latency per call, how many calls needed retries (429s / transient errors show up
there), and the projected cost of the remaining prompts at that group's observed rate.
No API calls.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lc.evaluate import apply_caps, is_valid, parse_caps   # noqa: E402
from lc.schema import read_jsonl                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cap", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    rows = apply_caps(read_jsonl(run / "rows.jsonl"), parse_caps(a.cap))
    group = {r["row_id"]: f"{r['family']}/{r['spec_level']}" for r in rows}
    recs = [r for r in read_jsonl(run / f"{a.model}.jsonl") if r["row_id"] in group]
    done = {r["row_id"] for r in recs if is_valid(r)}

    stats = collections.defaultdict(lambda: {"n": 0, "cost": 0.0, "reas": 0, "lat": 0.0, "retried": 0})
    for r in recs:
        if r.get("error") is not None:
            continue
        s = stats[group[r["row_id"]]]
        s["n"] += 1
        s["cost"] += r.get("cost") or 0
        s["reas"] += r["usage"]["reasoning_tokens"]
        s["lat"] += r.get("latency_s") or 0
        s["retried"] += (r.get("attempts") or 1) > 1
    remaining = collections.Counter(group[r["row_id"]] for r in rows if r["row_id"] not in done)

    total_spent = sum(s["cost"] for s in stats.values())
    all_n = sum(s["n"] for s in stats.values()) or 1
    overall_rate = total_spent / all_n
    print(f"{'group':26s} {'done':>5s} {'left':>5s} {'reasoning':>9s} {'$/call':>7s} {'sec/call':>8s} "
          f"{'retried':>7s} {'projected $':>11s}")
    proj_total = 0.0
    for g in sorted(set(group.values())):
        s = stats.get(g, {"n": 0, "cost": 0, "reas": 0, "lat": 0, "retried": 0})
        n = s["n"]
        rate = s["cost"] / n if n else overall_rate
        proj = rate * remaining[g]
        proj_total += proj
        print(f"{g:26s} {n:5d} {remaining[g]:5d} {s['reas'] / n if n else 0:9.0f} {rate:7.4f} "
              f"{s['lat'] / n if n else 0:8.1f} {s['retried']:7d} {proj:11.2f}"
              + ("  (no data yet: overall rate)" if not n else ""))
    print(f"\nspent so far ${total_spent:.2f}; projected to finish ${proj_total:.2f}; "
          f"total ~${total_spent + proj_total:.2f}")


if __name__ == "__main__":
    main()
