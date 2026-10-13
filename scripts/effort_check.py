#!/usr/bin/env python3
"""Does a provider actually honour reasoning_effort? Asks the same prompts at low / medium /
high and prints the reasoning tokens and the provider-reported cost for each call.

  python scripts/effort_check.py --run runs/main --model grok-4-7 --rows 2

If the three levels use similar token counts, the parameter is being ignored. Reuses
prompts the model already answered in its probe, so results line up with that run.
Costs a few dozen cents for Grok (2 rows x 3 levels = 6 calls).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lc.keys import get_key                      # noqa: E402
from lc.prompts import prompt_for_row            # noqa: E402
from lc.providers.registry import load_models    # noqa: E402
from lc.schema import read_jsonl                 # noqa: E402

URLS = {"xai": "https://api.x.ai/v1/chat/completions",
        "openai": "https://api.openai.com/v1/chat/completions",
        "moonshot": "https://api.moonshot.ai/v1/chat/completions",
        "zai": "https://api.z.ai/api/paas/v4/chat/completions",
        "deepseek": "https://api.deepseek.com/chat/completions"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--levels", default="low,medium,high")
    a = ap.parse_args()

    cfg = load_models()[a.model]
    if cfg["provider"] not in URLS:
        raise SystemExit(f"{a.model}: provider {cfg['provider']} not supported by this check")
    run = Path(a.run)
    rows = {r["row_id"]: r for r in read_jsonl(run / "rows.jsonl")}
    prev = [r for r in read_jsonl(run / f"{a.model}.jsonl") if r.get("error") is None]
    if not prev:
        raise SystemExit(f"no answered prompts in {run / (a.model + '.jsonl')}; run the probe first")
    picked = prev[:a.rows]
    rates = cfg["rates"]

    headers = {"Authorization": f"Bearer {get_key(cfg['provider'])}", "Content-Type": "application/json"}
    total = 0.0
    with httpx.Client(timeout=900) as http:
        for rec in picked:
            row = rows[rec["row_id"]]
            p = prompt_for_row(row)
            print(f"\n{row['family']}/{row['spec_level']}  {row['qid']}  "
                  f"(earlier run at {rec.get('effort')}: {rec['usage']['reasoning_tokens']} reasoning tokens)")
            for level in a.levels.split(","):
                body = {"model": cfg["model_id"], "reasoning_effort": level, "max_tokens": 32000,
                        "messages": [{"role": "system", "content": p.system},
                                     {"role": "user", "content": p.as_text()}]}
                r = http.post(URLS[cfg["provider"]], headers=headers, json=body)
                if r.status_code != 200:
                    print(f"  {level:>6s}: HTTP {r.status_code} {r.text[:300]}")
                    continue
                d = r.json()
                u = d.get("usage", {}) or {}
                reas = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
                comp = u.get("completion_tokens", 0) or 0
                ticks = u.get("cost_in_usd_ticks")
                vendor_cost = ticks / 1e10 if ticks is not None else None
                ours = (u.get("prompt_tokens", 0) * rates["input"] + (comp + reas) * rates["output"]) / 1e6
                total += vendor_cost if vendor_cost is not None else ours
                ans = ((d.get("choices") or [{}])[0].get("message") or {}).get("content", "")
                print(f"  {level:>6s}: reasoning {reas:>6d}  visible {comp:>4d}  answer {ans.strip()[:20]!r:22s}"
                      f" cost ours ${ours:.4f}" + (f" | xAI-reported ${vendor_cost:.4f}" if vendor_cost is not None else ""))
            print("  raw usage of last call: " + json.dumps(u))
    print(f"\ntotal spent on this check: ~${total:.2f}")


if __name__ == "__main__":
    main()
