"""Cross-provider re-verification of accepted HoVer decompositions via the Anthropic Message
Batches API.

  submit   build one request per accepted claim not yet cross-verified, upload as a batch,
           save the batch id to data/crossver_state.json (laptop can be closed afterwards)
  collect  if the batch has ended: download results, append them to the results cache,
           merge a `crossver` field into hover_processed.jsonl, print agreement + cost

The prompt is the same audit the Gemini verifier received (VERIFY_INSTRUCTIONS); the answer
is constrained to the Verification JSON schema with structured outputs.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .hover import VERIFY_INSTRUCTIONS, verify_prompt, _h
from .hover_gold import GOLD
from .schema import read_jsonl, write_jsonl

# batch prices, USD per million tokens (50% of standard); update if pricing changes
BATCH_PRICES = {
    "claude-opus-5-5": (2.0, 10.0),
    "claude-sonnet-5-5": (1.0, 5.0),
    "claude-haiku-5-5": (0.05, 0.25),
    "claude-fable-5-1": (5.0, 25.0),
}

FIELDS = ["conjunction_equivalent", "subclaims_independent", "negations_exact",
          "fully_fictional_and_consistent", "self_contained", "issues", "passed"]
SCHEMA = {
    "type": "object",
    "properties": {**{f: {"type": "boolean"} for f in FIELDS if f != "issues"}, "issues": {"type": "string"}},
    "required": FIELDS,
    "additionalProperties": False,
}
CROSSVER_VERSION = _h(VERIFY_INSTRUCTIONS + json.dumps(SCHEMA, sort_keys=True))


def _paths(processed: str):
    d = Path(processed).parent
    return d / "crossver_state.json", d / "crossver_results.jsonl"


def _custom_id(rec: dict) -> str:
    # 1-64 chars of [A-Za-z0-9_-]; HoVer uids are 36-char UUIDs
    return rec["uid"]


def _targets(recs: list[dict], done: set[str]) -> list[dict]:
    gold = {g["claim"] for g in GOLD.values()}
    return [r for r in recs if r.get("accepted") and r["claim"] not in gold and _custom_id(r) not in done]


def _load_results(results_path: Path) -> dict[str, dict]:
    if not results_path.exists():
        return {}
    out = {}
    for row in read_jsonl(results_path):
        if row.get("version") == CROSSVER_VERSION:
            out[row["custom_id"]] = row          # later rows win (e.g. a successful retry)
    return out


def build_request(rec: dict, model: str, effort: str, max_tokens: int) -> dict:
    return {
        "custom_id": _custom_id(rec),
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort,
                              "format": {"type": "json_schema", "schema": SCHEMA}},
            "messages": [{"role": "user", "content": verify_prompt(rec["decomposition"])}],
        },
    }


def submit(processed: str, model: str, effort: str, max_tokens: int, limit: int = 0,
           dry_run: bool = False, force: bool = False):
    state_path, results_path = _paths(processed)
    if state_path.exists() and not force:
        st = json.loads(state_path.read_text(encoding="utf-8"))
        if not st.get("collected"):
            raise SystemExit(f"Batch {st['batch_id']} has not been collected yet. Run "
                             f"`hover-crossver collect` first (or pass --force to submit another).")
    recs = read_jsonl(processed)
    done = {k for k, v in _load_results(results_path).items() if v["status"] == "succeeded"}
    targets = _targets(recs, done)
    if limit:
        targets = targets[:limit]
    if not targets:
        print("Nothing to submit: every accepted claim already has a successful cross-verification.")
        return
    reqs = [build_request(r, model, effort, max_tokens) for r in targets]
    est_in = sum(len(r["params"]["messages"][0]["content"]) for r in reqs) / 3.8
    pin, pout = BATCH_PRICES.get(model, (None, None))
    print(f"{len(reqs)} requests, ~{est_in / 1e6:.2f}M input tokens"
          + (f" (~${est_in * pin / 1e6:.2f} input; output depends on thinking, "
             f"~${len(reqs) * 1500 * pout / 1e6:.2f} at 1.5k tokens/request)" if pin else ""))
    if dry_run:
        out = Path(processed).with_name("crossver_requests_preview.jsonl")
        write_jsonl(out, reqs)
        print(f"dry run: wrote requests to {out}; nothing was sent.")
        return
    import anthropic
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=reqs)
    state = {"batch_id": batch.id, "model": model, "effort": effort, "n_requests": len(reqs),
             "version": CROSSVER_VERSION, "created_at": str(batch.created_at), "collected": False}
    state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    print(f"submitted batch {batch.id} ({len(reqs)} requests). Saved to {state_path}.\n"
          f"You can close the laptop. Later run: python lcbuild.py hover-crossver collect")


def parse_result(entry: dict, model: str) -> dict:
    """entry: one batch result as a plain dict (custom_id, result{type, message?, error?})."""
    cid, res = entry["custom_id"], entry["result"]
    row = {"custom_id": cid, "version": CROSSVER_VERSION, "model": model, "status": res["type"]}
    if res["type"] != "succeeded":
        row["error"] = json.dumps(res.get("error"))[:500]
        return row
    msg = res["message"]
    usage = msg.get("usage") or {}
    row["usage"] = {"input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens")}
    row["stop_reason"] = msg.get("stop_reason")
    text = next((b.get("text") for b in msg.get("content", []) if b.get("type") == "text"), None)
    try:
        out = json.loads(text)
        assert set(FIELDS) <= set(out)
        row["output"] = {f: out[f] for f in FIELDS}
    except Exception:
        row["status"] = "unparsable"
        row["error"] = f"stop_reason={msg.get('stop_reason')}; text={str(text)[:300]}"
    return row


def merge(processed: str, results: dict[str, dict]):
    recs = read_jsonl(processed)
    for r in recs:
        res = results.get(_custom_id(r))
        if res is None or not r.get("accepted"):
            continue
        cv = {"model": res["model"], "status": res["status"]}
        if res["status"] == "succeeded":
            cv.update(res["output"])
            cv["agrees_with_primary"] = bool(res["output"]["passed"]) == bool((r.get("verification") or {}).get("passed"))
        r["crossver"] = cv
    shutil.copyfile(processed, str(processed) + ".bak")
    write_jsonl(processed, recs)
    return recs


def summarize(recs: list[dict], results: dict[str, dict], model: str):
    cv = [r["crossver"] for r in recs if r.get("accepted") and "crossver" in r]
    ok = [c for c in cv if c["status"] == "succeeded"]
    failed_props = {f: sum(not c[f] for c in ok) for f in FIELDS if f not in ("issues", "passed")}
    print(f"cross-verified {len(ok)} accepted claims ({len(cv) - len(ok)} without a usable result)")
    if ok:
        npass = sum(c["passed"] for c in ok)
        print(f"  second verifier passed {npass} / {len(ok)} ({100 * npass / len(ok):.1f}%)")
        print(f"  failures by property: {failed_props}")
    rows = [v for v in results.values() if v["status"] == "succeeded"]
    tin = sum(v["usage"]["input_tokens"] for v in rows)
    tout = sum(v["usage"]["output_tokens"] for v in rows)
    tth = sum(v["usage"]["thinking_tokens"] or 0 for v in rows)
    pin, pout = BATCH_PRICES.get(model, (None, None))
    cost = f", ~${(tin * pin + tout * pout) / 1e6:.2f} at batch prices" if pin else ""
    print(f"  tokens: {tin:,} in / {tout:,} out (of which thinking {tth:,}){cost}")


def collect(processed: str):
    state_path, results_path = _paths(processed)
    if not state_path.exists():
        raise SystemExit("No submitted batch found (data/crossver_state.json missing). Run submit first.")
    st = json.loads(state_path.read_text(encoding="utf-8"))
    import anthropic
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(st["batch_id"])
    c = batch.request_counts
    print(f"batch {batch.id}: {batch.processing_status} | processing {c.processing}, succeeded {c.succeeded}, "
          f"errored {c.errored}, expired {c.expired}, canceled {c.canceled}")
    if batch.processing_status != "ended":
        print("Not finished yet; run collect again later.")
        return
    new = [parse_result(e.model_dump(mode="json"), st["model"]) for e in client.messages.batches.results(batch.id)]
    with results_path.open("a", encoding="utf-8") as f:
        for row in new:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    st["collected"] = True
    state_path.write_text(json.dumps(st, indent=1), encoding="utf-8")
    results = _load_results(results_path)
    recs = merge(processed, results)
    bad = sum(r["status"] != "succeeded" for r in new)
    print(f"downloaded {len(new)} results ({bad} errored/expired/unparsable; resubmitting retries only those).")
    summarize(recs, results, st["model"])
