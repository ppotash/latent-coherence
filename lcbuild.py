#!/usr/bin/env python3
"""Build the latent-coherence eval dataset.

  python lcbuild.py synth           --out data/synth_items.jsonl
  python lcbuild.py hover-select    --hover-json path/to/hover_train_release_v1.1.json --out data/hover_candidates.jsonl
  python lcbuild.py hover-process   --candidates data/hover_candidates.jsonl --out data/hover_processed.jsonl
  python lcbuild.py hover-build     --processed data/hover_processed.jsonl --out data/hover_items.jsonl
  python lcbuild.py export-prompts  --items data/synth_items.jsonl data/hover_items.jsonl --out data/prompts.jsonl
  python lcbuild.py stats           --items data/*.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lc.schema import write_jsonl, read_jsonl, render_prompts, item_from_dict  # noqa: E402
from lc.env import load_env  # noqa: E402

load_env(Path(__file__).resolve().parent / ".env")


def cmd_synth(a):
    from lc.synth import FAMILIES
    rng = np.random.default_rng(a.seed)
    items = []
    for fam in a.families.split(","):
        specs = ["full"] if fam == "causal" else a.specs.split(",")
        for spec in specs:
            for i in range(a.n):
                items.append(FAMILIES[fam](rng, i, spec).to_dict())
    write_jsonl(a.out, items)
    print(f"wrote {len(items)} items -> {a.out}")


def cmd_hover_select(a):
    from lc.hover import select_claims
    rows = select_claims(a.hover_json, a.n, seed=a.seed)
    write_jsonl(a.out, rows)
    print(f"wrote {len(rows)} candidate claims -> {a.out}")


def cmd_hover_process(a):
    from lc.llm import LLM
    from lc.hover import run_decompose_verify, dec_key, ver_key
    from lc.hover_gold import GOLD
    claims = read_jsonl(a.candidates)
    if a.limit:
        claims = claims[: a.limit]
    fixture = None
    if a.backend == "fixture":   # offline test path: only claims with a gold decomposition succeed
        fixture = {}
        for c in claims:
            for g in GOLD.values():
                if g["claim"] == c["claim"]:
                    fixture[dec_key(c["uid"])] = g["output"]
                    fixture[ver_key(c["uid"], g["output"])] = {"conjunction_equivalent": True, "subclaims_independent": True,
                                                   "negations_exact": True, "fully_fictional_and_consistent": True,
                                                   "self_contained": True, "issues": "", "passed": True}
    cache = Path(a.out).with_suffix(".cache")
    cache.mkdir(parents=True, exist_ok=True)
    dec = LLM(a.backend, a.dec_model, cache / "decompose.jsonl", a.concurrency, fixture=fixture)
    ver = LLM(a.backend, a.ver_model, cache / "verify.jsonl", a.concurrency, temperature=0.0, fixture=fixture)
    recs = asyncio.run(run_decompose_verify(claims, dec, ver))
    write_jsonl(a.out, recs)
    c = collections.Counter("accepted" if r["accepted"] else
                            ("verify_failed" if r["structural_ok"] else r["structural_issue"].split(":")[0])
                            for r in recs)
    print(f"wrote {len(recs)} processed claims -> {a.out}\n  outcomes: {dict(c)}")


def cmd_hover_build(a):
    from lc.hover import build_hover_item
    from lc.hover_gold import GOLD
    gold_claims = {g["claim"] for g in GOLD.values()}   # few-shot claims: the model saw the answer
    recs = [r for r in read_jsonl(a.processed) if r.get("accepted") and r["claim"] not in gold_claims]
    if a.require_crossver:
        n0 = len(recs)
        recs = [r for r in recs if (r.get("crossver") or {}).get("passed")]
        print(f"dropped {n0 - len(recs)} claims without a passing cross-verification")
    if a.drop_similar_names:
        from lc.hover import similar_name_pairs
        n0 = len(recs)
        recs = [r for r in recs if not similar_name_pairs(r["decomposition"])]
        print(f"dropped {n0 - len(recs)} claims with near-identical entity names")
    rng = np.random.default_rng(a.seed)
    items, idx, skipped = [], 0, 0
    for spec in a.specs.split(","):
        for r in recs:
            for _ in range(a.variants):
                it = build_hover_item(rng, r, idx, spec)
                if it is None:
                    skipped += 1
                    continue
                items.append(it.to_dict())
                idx += 1
    write_jsonl(a.out, items)
    print(f"wrote {len(items)} hover items from {len(recs)} accepted claims (skipped {skipped}) -> {a.out}")


def cmd_hover_review(a):
    """Write a human-readable sample of accepted / verifier-rejected / unusable records (UTF-8)."""
    R = read_jsonl(a.processed)
    rng = np.random.default_rng(a.seed)

    def show(r):
        d = r.get("decomposition") or {}
        return {"uid": r["uid"], "claim": r["claim"], "explicit": d.get("fictional_claim_explicit"),
                "subs": [s["fictional"] for s in d.get("subclaims", [])],
                "negs": [s["fictional_negation"] for s in d.get("subclaims", [])],
                "bridge": d.get("bridge_index"),
                "why_rejected": r.get("structural_issue") or (r.get("verification") or {}).get("issues")}

    def sample(rows, k):
        idx = rng.permutation(len(rows))[:k]
        return [rows[i] for i in idx]

    groups = {"accepted": [r for r in R if r["accepted"]],
              "verify_failed": [r for r in R if r["structural_ok"] and not r["accepted"]],
              "not_usable": [r for r in R if not r["structural_ok"]]}
    out = {"counts": {k: len(v) for k, v in groups.items()},
           "accepted": [show(r) for r in sample(groups["accepted"], a.n)],
           "verify_failed": [show(r) for r in sample(groups["verify_failed"], a.n)],
           "not_usable": [{"claim": r["claim"], "reason": (r.get("decomposition") or {}).get("reason")}
                          for r in sample(groups["not_usable"], a.n)]}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"counts: {out['counts']} -> {a.out}")


def cmd_hover_crossver(a):
    from lc import crossver
    if a.action == "submit":
        crossver.submit(a.processed, a.model, a.effort, a.max_tokens, a.limit, a.dry_run, a.force)
    else:
        crossver.collect(a.processed)


def cmd_export(a):
    rows = []
    framings = set(a.framings.split(","))
    rng = np.random.default_rng(a.seed)
    for p in a.items:
        items = read_jsonl(p)
        # noise-floor subset: these items keep every paraphrase, the rest keep only p0
        full_grid = set(rng.choice(len(items), size=min(a.noise_floor_items, len(items)), replace=False).tolist())
        for j, d in enumerate(items):
            for r in render_prompts(d):
                if r["framing"] not in framings and r["framing"] not in ("none", "custom"):
                    continue
                if r["paraphrase"] >= a.paraphrases and j not in full_grid:
                    continue
                r["noise_floor_item"] = j in full_grid
                rows.append(r)
    write_jsonl(a.out, rows)
    print(f"wrote {len(rows)} prompt rows -> {a.out}")


def cmd_stats(a):
    for p in a.items:
        items = [item_from_dict(d) for d in read_jsonl(p)]
        by = collections.Counter((it.family, it.spec_level) for it in items)
        nq = sum(len(it.queries) for it in items)
        worst = max(abs(v) for it in items for v in it.identity_residuals({q.qid: q.truth for q in it.queries}).values())
        print(f"{p}: {len(items)} items, {nq} queries, max |identity residual on truth| = {worst:.2e}")
        flagged = sum(bool(it.meta.get("similar_name_pairs")) for it in items)
        if any(it.family == "hover" for it in items):
            print(f"   hover items with near-identical entity names: {flagged}")
        for k, v in sorted(by.items()):
            print(f"   {k}: {v}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("synth"); s.set_defaults(fn=cmd_synth)
    s.add_argument("--families", default="source,diagnostic,causal")
    s.add_argument("--specs", default="full,qualitative")
    s.add_argument("--n", type=int, default=300, help="items per family per spec level")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/synth_items.jsonl")

    s = sp.add_parser("hover-select"); s.set_defaults(fn=cmd_hover_select)
    s.add_argument("--hover-json", required=True)
    s.add_argument("--n", type=int, default=1500)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/hover_candidates.jsonl")

    s = sp.add_parser("hover-process"); s.set_defaults(fn=cmd_hover_process)
    s.add_argument("--candidates", default="data/hover_candidates.jsonl")
    s.add_argument("--backend", default="gemini", choices=["gemini", "fixture"])
    s.add_argument("--dec-model", default="gemini-3.8-flash")
    s.add_argument("--ver-model", default="gemini-3.8-flash", help="verifier; a different/stronger model is a plus")
    s.add_argument("--concurrency", type=int, default=16)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--out", default="data/hover_processed.jsonl")

    s = sp.add_parser("hover-build"); s.set_defaults(fn=cmd_hover_build)
    s.add_argument("--processed", default="data/hover_processed.jsonl")
    s.add_argument("--specs", default="full,qualitative")
    s.add_argument("--variants", type=int, default=1, help="independently sampled worlds per claim per spec")
    s.add_argument("--require-crossver", action="store_true",
                   help="keep only claims that also passed the Claude cross-verification")
    s.add_argument("--drop-similar-names", action="store_true",
                   help="exclude claims whose fictional entities have near-identical names (e.g. Vane/Vance)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/hover_items.jsonl")

    s = sp.add_parser("hover-crossver", help="re-verify accepted claims with Claude (Batch API)")
    s.set_defaults(fn=cmd_hover_crossver)
    s.add_argument("action", choices=["submit", "collect"])
    s.add_argument("--processed", default="data/hover_processed.jsonl")
    s.add_argument("--model", default="claude-opus-5-5")
    s.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    s.add_argument("--max-tokens", type=int, default=16000, help="ceiling on thinking + answer tokens")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--dry-run", action="store_true", help="write the requests to a preview file; send nothing")
    s.add_argument("--force", action="store_true", help="submit even if a previous batch is uncollected")

    s = sp.add_parser("hover-review"); s.set_defaults(fn=cmd_hover_review)
    s.add_argument("--processed", default="data/hover_processed.jsonl")
    s.add_argument("--n", type=int, default=8, help="records sampled per group")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="review.json")

    s = sp.add_parser("export-prompts"); s.set_defaults(fn=cmd_export)
    s.add_argument("--items", nargs="+", required=True)
    s.add_argument("--paraphrases", type=int, default=1, help="paraphrases per query for ordinary items (max 3)")
    s.add_argument("--framings", default="learn,assume", help="conditioning framings to keep")
    s.add_argument("--noise-floor-items", type=int, default=100,
                   help="items per file that keep all 3 paraphrases, for estimating the noise floor")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/prompts.jsonl")

    s = sp.add_parser("stats"); s.set_defaults(fn=cmd_stats)
    s.add_argument("--items", nargs="+", required=True)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
