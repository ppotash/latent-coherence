"""Batch path for `lceval.py batch`: half the price of synchronous calls, for Anthropic,
OpenAI (and any OpenAI-compatible provider with the same Batch API) and Gemini.

  submit   1. a few synchronous probe calls (full price), spread across groups, settle the
              API contract and measure real tokens per call, so the projection is grounded
           2. the remaining prompts are submitted as one or more batches; their ids are saved
              to runs/<run>/<model>.batch.json. The laptop can then be closed.
  collect  downloads finished batches and appends answers to runs/<run>/<model>.jsonl in the
           same format as synchronous runs (mode="batch", cost at batch prices). Errored or
           expired requests are recorded as errors and resubmitted by the next `submit`.

Request bodies come from each provider's synchronous adapter (`client._body`), after the
probe has settled any parameter adaptations, so batch and synchronous runs send identical
parameters and wording. Responses are parsed by the same functions the adapters use.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

import httpx

from .evaluate import make_record, _done_ids, is_truncated, check_effort, _rates, _results_path, apply_caps, parse_caps, split_probe
from .keys import get_key
from .prompts import prompt_for_row
from .providers.registry import load_models, make_client
from .schema import read_jsonl

BATCH_PRICE_FACTOR = 0.5


def _cid(row_id: str) -> str:
    # Anthropic custom_id must match ^[a-zA-Z0-9_-]{1,64}$; row ids contain ':' and '|'
    return hashlib.sha1(row_id.encode("utf-8")).hexdigest()[:40]


# ============================================================================ backends
#
# create(items) -> (batch_id, extra_state)   items: list of (custom_id, request_body)
# status(batch) -> (ended: bool, summary: str)
# results(batch) -> iterator of (custom_id, response_body | None, error | None)

class AnthropicBatches:
    max_per_batch = 10_000

    def __init__(self, cfg):
        import anthropic
        self.api = anthropic.Anthropic(api_key=get_key("anthropic"))

    def create(self, items):
        b = self.api.messages.batches.create(requests=[{"custom_id": c, "params": body} for c, body in items])
        return b.id, {}

    def status(self, batch):
        info = self.api.messages.batches.retrieve(batch["id"])
        c = info.request_counts
        return (info.processing_status == "ended",
                f"{info.processing_status} | processing {c.processing}, succeeded {c.succeeded}, "
                f"errored {c.errored}, expired {c.expired}, canceled {c.canceled}")

    def results(self, batch):
        for res in self.api.messages.batches.results(batch["id"]):
            d = res.model_dump(mode="json")
            r = d["result"]
            if r["type"] == "succeeded":
                yield d["custom_id"], r["message"], None
            else:
                yield d["custom_id"], None, f"batch {r['type']}: {json.dumps(r.get('error'))[:300]}"

    def parse(self, body, cfg):
        from .providers.anthropic import completion_from_message
        return completion_from_message(body, cfg["model_id"])


class OpenAIBatches:
    """Files API upload + /v1/batches over /v1/chat/completions (the endpoint the adapter uses)."""
    max_per_batch = 50_000
    root = "https://api.openai.com/v1"

    def __init__(self, cfg, http: httpx.Client | None = None):
        self.http = http or httpx.Client(timeout=300, headers={"Authorization": f"Bearer {get_key(cfg['provider'])}"})
        self.reasoning_in_completion = True
        if cfg["provider"] != "openai":
            from .providers import openai_compat as oc
            cls = {"xai": oc.XAIClient, "zai": oc.ZAIClient, "moonshot": oc.MoonshotClient,
                   "deepseek": oc.DeepSeekClient}[cfg["provider"]]
            self.root = cls.base_url.rsplit("/chat/completions", 1)[0]
            self.reasoning_in_completion = cls.reasoning_in_completion

    def _check(self, r):
        if r.status_code >= 300:
            raise SystemExit(f"OpenAI batch API error {r.status_code}: {r.text[:500]}")
        return r

    def create(self, items):
        lines = "".join(json.dumps({"custom_id": c, "method": "POST", "url": "/v1/chat/completions",
                                    "body": body}, ensure_ascii=False) + "\n" for c, body in items)
        f = self._check(self.http.post(f"{self.root}/files", data={"purpose": "batch"},
                                       files={"file": ("batch.jsonl", io.BytesIO(lines.encode("utf-8")),
                                                       "application/jsonl")})).json()
        b = self._check(self.http.post(f"{self.root}/batches", json={
            "input_file_id": f["id"], "endpoint": "/v1/chat/completions", "completion_window": "24h"})).json()
        return b["id"], {"input_file_id": f["id"]}

    def _get(self, batch):
        return self._check(self.http.get(f"{self.root}/batches/{batch['id']}")).json()

    def status(self, batch):
        b = self._get(batch)
        c = b.get("request_counts") or {}
        done = b["status"] in ("completed", "failed", "expired", "cancelled")
        return done, (f"{b['status']} | completed {c.get('completed', '?')}, failed {c.get('failed', '?')}, "
                      f"total {c.get('total', '?')}")

    def _lines(self, file_id):
        if not file_id:
            return []
        txt = self._check(self.http.get(f"{self.root}/files/{file_id}/content")).text
        return [json.loads(l) for l in txt.splitlines() if l.strip()]

    def results(self, batch):
        b = self._get(batch)
        seen = set()
        for d in self._lines(b.get("output_file_id")) + self._lines(b.get("error_file_id")):
            cid = d.get("custom_id")
            seen.add(cid)
            resp = d.get("response") or {}
            if resp.get("status_code") == 200 and resp.get("body"):
                yield cid, resp["body"], None
            else:
                err = d.get("error") or resp.get("body") or {"status_code": resp.get("status_code")}
                yield cid, None, f"batch error: {json.dumps(err)[:300]}"
        if b["status"] in ("failed", "expired", "cancelled"):
            for cid in batch.get("custom_ids", []):
                if cid not in seen:
                    yield cid, None, f"batch {b['status']}: {json.dumps(b.get('errors'))[:300]}"

    def parse(self, body, cfg):
        from .providers.openai_compat import completion_from_chat
        return completion_from_chat(body, cfg["model_id"], self.reasoning_in_completion)


class GeminiBatches:
    """Gemini Batch Mode with inline requests (results come back in input order).

    The docs disagree on where the job state and the inline results sit in the batch resource,
    so both are looked up in every documented location; a final resource with no recognizable
    results is saved next to the run for inspection instead of being silently dropped."""
    max_per_batch = 2_000            # keeps each inline request well under the 20 MB limit
    root = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, cfg, http: httpx.Client | None = None):
        self.model_id = cfg["model_id"]
        self.http = http or httpx.Client(timeout=300, headers={"x-goog-api-key": get_key("gemini")})

    def _check(self, r):
        if r.status_code >= 300:
            raise SystemExit(f"Gemini batch API error {r.status_code}: {r.text[:500]}")
        return r

    def create(self, items):
        body = {"batch": {"display_name": f"latent-coherence-{int(time.time())}",
                          "input_config": {"requests": {"requests": [
                              {"request": req, "metadata": {"key": c}} for c, req in items]}}}}
        op = self._check(self.http.post(f"{self.root}/models/{self.model_id}:batchGenerateContent",
                                        json=body)).json()
        return op["name"], {}

    def _get(self, batch):
        return self._check(self.http.get(f"{self.root}/{batch['id']}")).json()

    @staticmethod
    def _state(op):
        for d in (op, op.get("metadata") or {}, op.get("response") or {}):
            if isinstance(d.get("state"), str):
                return d["state"]
        return "JOB_STATE_SUCCEEDED" if op.get("done") and "error" not in op else "UNKNOWN"

    @staticmethod
    def _inlined(op):
        for d in (op.get("response") or {}, op.get("metadata") or {}, op):
            for path in (("inlinedResponses", "inlinedResponses"), ("output", "inlinedResponses", "inlinedResponses"),
                         ("inlinedResponses",), ("output", "inlinedResponses")):
                x = d
                for k in path:
                    x = x.get(k) if isinstance(x, dict) else None
                if isinstance(x, list):
                    return x
        return None

    def status(self, batch):
        op = self._get(batch)
        st = self._state(op)
        done = bool(op.get("done")) or st.endswith(("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED"))
        return done, st

    def results(self, batch):
        op = self._get(batch)
        st = self._state(op)
        cids = batch.get("custom_ids", [])
        inl = self._inlined(op)
        if inl is None:
            if st.endswith("SUCCEEDED"):
                dump = Path(batch["_run"]) / f"gemini_batch_{batch['id'].split('/')[-1]}.json"
                dump.write_text(json.dumps(op, indent=1), encoding="utf-8")
                raise SystemExit(f"batch succeeded but no inline results were found in the response; "
                                 f"saved it to {dump}. Send that file over and the parser can be fixed.")
            for cid in cids:
                yield cid, None, f"batch {st}: {json.dumps(op.get('error'))[:300]}"
            return
        for i, r in enumerate(inl):
            cid = ((r.get("metadata") or {}).get("key")) or (cids[i] if i < len(cids) else None)
            if r.get("response"):
                yield cid, r["response"], None
            else:
                yield cid, None, f"batch error: {json.dumps(r.get('error'))[:300]}"

    def parse(self, body, cfg):
        from .providers.gemini import completion_from_generate
        return completion_from_generate(body, cfg["model_id"])


def backend_for(cfg):
    p = cfg["provider"]
    if p == "anthropic":
        return AnthropicBatches(cfg)
    if p == "gemini":
        return GeminiBatches(cfg)
    if p in ("openai", "xai", "zai", "moonshot", "deepseek"):
        return OpenAIBatches(cfg)
    raise SystemExit(f"no batch path for provider {p!r}; use `run`")


# ============================================================================ state

def _state_path(run: Path, model: str) -> Path:
    return run / f"{model}.batch.json"


def _load_state(run: Path, model: str) -> dict:
    p = _state_path(run, model)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"batches": []}


def _save_state(run: Path, model: str, st: dict):
    _state_path(run, model).write_text(json.dumps(st, indent=1), encoding="utf-8")


# ============================================================================ submit / collect

def submit(run_dir: str, model_name: str, effort: str, thinking: bool, max_tokens: int,
           temperature: float, probe: int, yes: bool, limit: int, save_reasoning: bool,
           caps: str | None = None):
    run = Path(run_dir)
    cfg = load_models()[model_name]
    effort = effort or cfg.get("effort") or "medium"
    if cfg["provider"] not in ("anthropic", "gemini", "openai", "xai", "zai", "moonshot", "deepseek"):
        raise SystemExit(f"no batch path for provider {cfg['provider']!r}; use `run`")
    rates = _rates(cfg)
    st = _load_state(run, model_name)
    pending = [b for b in st["batches"] if not b.get("collected")]
    if pending:
        raise SystemExit(f"{len(pending)} batch(es) not yet collected for {model_name}; "
                         f"run `lceval.py batch collect` first.")
    all_rows = read_jsonl(run / "rows.jsonl")
    rows = apply_caps(all_rows, parse_caps(caps))
    if len(rows) < len(all_rows):
        print(f"--cap: {len(rows)} of {len(all_rows)} prompts in scope for this model")
    out_path = _results_path(run, model_name)
    check_effort(out_path, effort)
    todo = [r for r in rows if r["row_id"] not in _done_ids(out_path)]
    if limit:
        todo = todo[:limit]
    if not todo:
        print(f"{model_name}: all {len(rows)} prompts already answered.")
        return
    client = make_client(cfg, effort)

    # 1. synchronous probe, spread across groups
    n_probe = min(probe, len(todo))
    print(f"{model_name}: {len(todo)} prompts. Probing {n_probe} synchronously (full price)...")
    probe_rows, rest = split_probe(todo, n_probe)
    probe_recs = []
    for row in probe_rows:
        try:
            c = client.complete(prompt_for_row(row), max_tokens=max_tokens, temperature=temperature,
                                thinking=thinking, effort=effort)
            rec = make_record(row, model_name, effort, thinking, c, rates, save_reasoning)
        except Exception as e:
            rec = make_record(row, model_name, effort, thinking, None, rates, save_reasoning,
                              error=f"{type(e).__name__}: {str(e)[:400]}")
        probe_recs.append(rec)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    ok = [r for r in probe_recs if r.get("error") is None]
    if not ok:
        raise SystemExit("every probe call failed:\n  " + "\n  ".join(r["error"] for r in probe_recs))
    avg = lambda k: sum(r["usage"][k] for r in ok) / len(ok)
    print(f"  probe: {len(ok)}/{n_probe} ok | avg input {avg('input_tokens'):.0f} tok, output "
          f"{avg('output_tokens'):.0f} (reasoning {avg('reasoning_tokens'):.0f}) | "
          f"parsed {sum(r['prob'] is not None for r in ok)}/{len(ok)}"
          + (f" | {sum(map(is_truncated, ok))} cut off at --max-tokens {max_tokens}"
             if any(map(is_truncated, ok)) else ""))
    for r in ok[:5]:
        print(f"    {r['qid']:>10s} -> {r['text'][:60]!r}  (parsed {r['prob']})")
    dropped = getattr(client, "dropped_params", None)
    if dropped:
        print(f"  note: the API rejected and the adapter dropped: {dropped} (the batch requests omit them too)")
    if not rest:
        return
    if rates:
        per_call = sum(r["cost"] or 0 for r in ok) / len(ok) * BATCH_PRICE_FACTOR
        print(f"  batch price ~${per_call:.4f}/call -> ~${per_call * len(rest):.2f} projected for {len(rest)} prompts")
    if not yes and input("  Submit batch? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Not submitted. Probe answers are saved.")
        return

    # 2. submit; bodies are built after the probe so any contract adaptation applies
    backend = backend_for(cfg)
    for i in range(0, len(rest), backend.max_per_batch):
        chunk = rest[i:i + backend.max_per_batch]
        items = [(_cid(r["row_id"]), client._body(prompt_for_row(r), max_tokens, temperature, thinking, effort))
                 for r in chunk]
        bid, extra = backend.create(items)
        st["batches"].append({"id": bid, "provider": cfg["provider"], "n": len(items),
                              "custom_ids": [c for c, _ in items], "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                              "effort": effort, "thinking": thinking, "max_tokens": max_tokens,
                              "collected": False, **extra})
        _save_state(run, model_name, st)
        print(f"  submitted batch {bid} ({len(items)} requests)")
    print(f"Saved to {_state_path(run, model_name)}. You can close the laptop; later run:\n"
          f"  python lceval.py batch collect --run {run_dir} --model {model_name}")


def collect(run_dir: str, model_name: str, save_reasoning: bool):
    run = Path(run_dir)
    cfg = load_models()[model_name]
    rates = _rates(cfg)
    st = _load_state(run, model_name)
    pending = [b for b in st["batches"] if not b.get("collected")]
    if not pending:
        print(f"{model_name}: no uncollected batches.")
        return
    backend = backend_for(cfg)
    rows = {_cid(r["row_id"]): r for r in read_jsonl(run / "rows.jsonl")}
    out_path = _results_path(run, model_name)
    total_cost, n_ok, n_err = 0.0, 0, 0
    for b in pending:
        ended, summary = backend.status(b)
        print(f"batch {b['id']}: {summary}")
        if not ended:
            continue
        recs = []
        for cid, body, err in backend.results({**b, "_run": str(run)}):
            row = rows.get(cid)
            if row is None:
                continue
            if body is not None:
                try:
                    comp = backend.parse(body, cfg)
                except Exception as e:
                    body, err = None, f"unparsable batch response: {type(e).__name__}: {e}"
                else:
                    rec = make_record(row, model_name, b["effort"], b["thinking"], comp, rates,
                                      save_reasoning, mode="batch", price_factor=BATCH_PRICE_FACTOR)
                    total_cost += rec["cost"] or 0
                    n_ok += 1
            if body is None:
                rec = make_record(row, model_name, b["effort"], b["thinking"], None, rates, save_reasoning,
                                  error=err, mode="batch")
                n_err += 1
            recs.append(rec)
        with out_path.open("a", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        b["collected"] = True
        b["collected_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _save_state(run, model_name, st)
    still = sum(not b.get("collected") for b in st["batches"])
    print(f"{model_name}: collected {n_ok} answers ({n_err} errored/expired), ${total_cost:.2f} at batch prices"
          + (f"; {still} batch(es) still processing, run collect again later" if still else "")
          + (f"\n  run `batch submit` again to retry the {n_err} failed prompts" if n_err else ""))
