"""Evaluation-harness checks (no network). Run: python tests/test_eval.py

1. probability parsing on the answer formats models actually produce
2. provider adapters: request bodies and response parsing, against a mocked HTTP transport
3. end to end with simulated models: a coherent simulator scores exactly zero, and the
   noise decomposition attributes a noisy-but-coherent simulator's residuals to noise
"""
import collections
import json
import os
import shutil
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lc.parse import parse_probability                        # noqa: E402
from lc.prompts import Prompt                                   # noqa: E402

for k in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
    os.environ.setdefault(k, "test-key")


def test_parse():
    cases = {"0.35": 0.35, ".35": 0.35, "35%": 0.35, "Probability: **0.42**": 0.42,
             "0.35 (35%)": 0.35, "1": 1.0, "0": 0.0, "about 1 in 3": 1 / 3, "2/5": 0.4,
             "Given 3 sources, the probability is 0.27.": 0.27, "35": 0.35,
             "I cannot determine this.": None, "": None, "150%": None,
             # answer first, then an explanation with other numbers (GLM's layout)
             "**0.4**\n\nSince the fault is a worn bearing: 0.3 x 0.6 + ...": 0.4,
             "**≈ 0.342**\n\nReasoning summary:\n- **Zateim**: careless (0.2)": 0.342,
             "P(A)=0.3 and P(B)=0.5, so the final answer is 0.42.": 0.42,
             "The first factor is 0.3. Overall: **0.21**.": 0.21,
             # explanation first, answer last (Opus/Haiku's layout) must keep the last number
             "P = 0.3*0.5 + 0.7*0.2 = 0.15 + 0.14\n\n**0.29**": 0.29,
             "Source A is reliable (0.8), B is not (0.4).\nSo the probability is about 0.62.": 0.62,
             "0.3 for the first case and 0.5 for the second, giving 0.44": 0.44,
             "Weighted: 0.6*0.5 + 0.4*0.25\n= 0.4": 0.4,
             # real Haiku answers: labelled answer first, explanation with other numbers after
             "**Answer: 0.90**\n\nThe prior is 0.30 ... treated as an 80% prior.": 0.9,
             "**Estimate: 0.38**\n\nroughly 0.97 ... about 0.6 ... at about 0.38.\n\n3": 0.38,
             "**0.786**\n\n- P(shaded) = 175/600 ... 24/25 = 0.96\n\nSum: ... ≈ 0.7857": 0.786}
    for txt, want in cases.items():
        got, _ = parse_probability(txt)
        assert (got is None and want is None) or abs(got - want) < 1e-9, (txt, got, want)


def _mock(client, responses, seen):
    it = iter(responses)
    def handler(req):
        seen.append(json.loads(req.content))
        status, body = next(it)
        return httpx.Response(status, json=body)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))


P = Prompt(system="SYS", cacheable="CONTEXT\n\n", tail="QUESTION")


def test_anthropic_adapter():
    from lc.providers.anthropic import AnthropicClient
    c, seen = AnthropicClient("m", "claude-test", effort="medium"), []
    _mock(c, [(200, {"model": "claude-test", "stop_reason": "end_turn",
                     "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "0.42"}],
                     "usage": {"input_tokens": 100, "output_tokens": 50,
                               "output_tokens_details": {"thinking_tokens": 30}}})], seen)
    out = c.complete(P, max_tokens=1000, temperature=1.0, thinking=True, effort="medium")
    b = seen[0]
    assert b["thinking"] == {"type": "adaptive"} and b["output_config"] == {"effort": "medium"}
    assert b["system"] == "SYS" and b["messages"][0]["content"][0]["cache_control"]["type"] == "ephemeral"
    assert out.text == "0.42" and out.usage.output_tokens == 50 and out.usage.reasoning_tokens == 30


def test_gemini_adapter():
    from lc.providers.gemini import GeminiClient
    c, seen = GeminiClient("g", "gemini-test", effort="medium"), []
    _mock(c, [(200, {"candidates": [{"finishReason": "STOP", "content": {"parts": [
        {"text": "thinking...", "thought": True}, {"text": "0.3"}]}}],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 5, "thoughtsTokenCount": 200}})], seen)
    out = c.complete(P, max_tokens=1000, temperature=1.0, thinking=True, effort="medium")
    assert seen[0]["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "MEDIUM"
    assert out.text == "0.3" and out.usage.output_tokens == 205 and out.usage.reasoning_tokens == 200


def test_openai_adapter_drops_rejected_param():
    from lc.providers.openai_compat import OpenAIClient
    c, seen = OpenAIClient("o", "gpt-test", effort="medium"), []
    _mock(c, [(400, {"error": {"message": "Unsupported parameter: 'temperature' is not supported with this model."}}),
              (200, {"model": "gpt-test", "choices": [{"message": {"content": "0.7"}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 100, "completion_tokens": 300,
                               "completion_tokens_details": {"reasoning_tokens": 250}}})], seen)
    out = c.complete(P, max_tokens=1000, temperature=1.0, thinking=True, effort="medium")
    assert "temperature" in seen[0] and "temperature" not in seen[1]
    assert seen[1]["reasoning_effort"] == "medium" and "max_completion_tokens" in seen[1]
    assert out.text == "0.7" and c.dropped_params == ["temperature"]


def test_end_to_end_simulated():
    from lc.evaluate import select, run
    from lc.analyze import load_run, identity_table, noise_decomposition
    rd = ROOT / "runs" / "_test"
    shutil.rmtree(rd, ignore_errors=True)
    select([str(ROOT / "data/synth_items.jsonl"), str(ROOT / "data/sample_hover_items.jsonl")],
           str(rd), n_per_group=4, noise_items=10, framings=["learn", "assume"], seed=0,
           exclude_similar_names=False, force=True)
    for m in ("sim-coherent", "sim-noisy"):
        run(str(rd), m, "medium", True, 1000, 1.0, 4, None, 2, True, 0, True, False)
    items, noise, answers, meta, _ = load_run(rd)
    df = identity_table(items, noise, answers)
    coh = df[(df.model == "sim-coherent") & (df.kind == "total_probability")]
    assert len(coh) and coh.resid_logit.abs().max() < 1e-3, coh.resid_logit.abs().max()
    nd = noise_decomposition(df)
    noisy_all = nd[(nd.model == "sim-noisy") & (nd.category == "all")].iloc[0]
    assert noisy_all.noise_sd > 0.3 and noisy_all.systematic_share < 0.2, noisy_all.to_dict()
    shutil.rmtree(rd, ignore_errors=True)


def test_select_grows_nested():
    """Growing a run keeps every earlier item and row, so paid answers stay valid and only
    the new prompts get run."""
    from lc.evaluate import select, run
    rd = ROOT / "runs" / "_test_grow"
    shutil.rmtree(rd, ignore_errors=True)
    items = [str(ROOT / "data/synth_items.jsonl")]
    select(items, str(rd), 3, 2, ["learn"], 7, False, False)
    first = {json.loads(l)["row_id"] for l in open(rd / "rows.jsonl")}
    noise1 = json.loads((rd / "manifest.json").read_text())["noise_item_ids"]
    run(str(rd), "sim-noisy", "medium", True, 1000, 1.0, 2, None, 1, True, 0, True, False)
    select(items, str(rd), 6, 2, ["learn"], 7, False, False, grow=True)
    second = {json.loads(l)["row_id"] for l in open(rd / "rows.jsonl")}
    assert first < second
    assert json.loads((rd / "manifest.json").read_text())["noise_item_ids"] == noise1
    run(str(rd), "sim-noisy", "medium", True, 1000, 1.0, 2, None, 1, True, 0, True, False)
    answered = [json.loads(l)["row_id"] for l in open(rd / "sim-noisy.jsonl")]
    assert len(answered) == len(set(answered)) == len(second)     # nothing asked twice
    try:
        select(items, str(rd), 6, 2, ["learn", "assume"], 7, False, False, grow=True)
        raise AssertionError("grow with different framings should be refused")
    except SystemExit:
        pass
    shutil.rmtree(rd, ignore_errors=True)


def test_caps_and_targeted_noise():
    from lc.evaluate import select, apply_caps, parse_caps
    from lc.schema import read_jsonl
    rd = ROOT / "runs" / "_test_caps"
    shutil.rmtree(rd, ignore_errors=True)
    items = [str(ROOT / "data/synth_items.jsonl")]
    select(items, str(rd), 6, 3, ["learn"], 0, False, False)
    old = set(json.loads((rd / "manifest.json").read_text())["noise_item_ids"])
    select(items, str(rd), 6, 7, ["learn"], 0, False, False, grow=True, noise_groups=["source/qualitative"])
    new = set(json.loads((rd / "manifest.json").read_text())["noise_item_ids"])
    assert old < new and all(i.startswith("source-qualitative") for i in new - old) and len(new) == 7
    rows = read_jsonl(rd / "rows.jsonl")
    assert parse_caps("synthetic-full=2") == {"source/full": 2, "diagnostic/full": 2, "causal/full": 2}
    capped = apply_caps(rows, parse_caps("synthetic-full=2"))
    per = collections.Counter()
    for r in {(r["family"], r["spec_level"], r["item_id"]) for r in capped}:
        per[f"{r[0]}/{r[1]}"] += 1
    assert per["source/full"] == per["causal/full"] == 2 and per["source/qualitative"] == 6
    assert apply_caps(rows, parse_caps("synthetic-full=2")) == capped          # deterministic
    shutil.rmtree(rd, ignore_errors=True)


def test_anthropic_batch_roundtrip():
    """probe (mocked HTTP) -> submit (mocked SDK) -> collect: answers land in the same
    format as sync runs, at half price, with failed requests left for resubmission."""
    import types
    import anthropic
    from lc import batch, evaluate
    from lc.providers import registry
    from lc.providers.anthropic import AnthropicClient
    from lc.analyze import load_run
    rd = ROOT / "runs" / "_test_batch"
    shutil.rmtree(rd, ignore_errors=True)
    evaluate.select([str(ROOT / "data/synth_items.jsonl")], str(rd), n_per_group=1, noise_items=0,
                    framings=["learn"], seed=0, exclude_similar_names=False, force=True)
    msg = lambda txt: {"model": "claude-haiku-5-5", "stop_reason": "end_turn",
                       "content": [{"type": "text", "text": txt}],
                       "usage": {"input_tokens": 400, "output_tokens": 600}}

    def fake_client(cfg, effort, rows=None):
        c = AnthropicClient(cfg["name"], cfg["model_id"], effort=effort)
        c._client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=msg("0.5"))))
        return c
    submitted = {}

    class R:
        def __init__(s, d): s.d = d
        def model_dump(s, mode=None): return s.d

    class Batches:
        def create(s, requests):
            submitted["reqs"] = requests
            return types.SimpleNamespace(id="msgbatch_x", created_at="now")
        def retrieve(s, bid):
            return types.SimpleNamespace(processing_status="ended", request_counts=types.SimpleNamespace(
                processing=0, succeeded=len(submitted["reqs"]) - 1, errored=0, expired=1, canceled=0))
        def results(s, bid):
            reqs = submitted["reqs"]
            out = [R({"custom_id": r["custom_id"], "result": {"type": "succeeded", "message": msg("0.25")}})
                   for r in reqs[:-1]]
            return out + [R({"custom_id": reqs[-1]["custom_id"], "result": {"type": "expired"}})]

    real_make, real_sdk = registry.make_client, anthropic.Anthropic
    batch.make_client = fake_client
    anthropic.Anthropic = lambda **kw: types.SimpleNamespace(messages=types.SimpleNamespace(batches=Batches()))
    try:
        batch.submit(str(rd), "claude-haiku-5-5", "medium", True, 4000, 1.0, 2, True, 0, True)
        body = submitted["reqs"][0]["params"]
        assert body["thinking"] == {"type": "adaptive"} and body["output_config"] == {"effort": "medium"}
        assert all(len(r["custom_id"]) <= 64 for r in submitted["reqs"])
        try:
            batch.submit(str(rd), "claude-haiku-5-5", "medium", True, 4000, 1.0, 2, True, 0, True)
            raise AssertionError("second submit should refuse while a batch is uncollected")
        except SystemExit:
            pass
        batch.collect(str(rd), "claude-haiku-5-5", True)
        recs = read = [json.loads(l) for l in open(rd / "claude-haiku-5-5.jsonl")]
        b = [r for r in recs if r["mode"] == "batch" and r.get("error") is None]
        assert b and abs(b[0]["cost"] - (400 * 0.10 + 600 * 0.50) / 1e6 * 0.5) < 1e-12
        assert sum(r.get("error") is not None for r in recs) == 1          # the expired one
        n_rows = sum(1 for _ in open(rd / "rows.jsonl"))
        assert len(evaluate._done_ids(rd / "claude-haiku-5-5.jsonl")) == n_rows - 1
        _, _, answers, _, _ = load_run(rd)
        assert len(answers["claude-haiku-5-5"]) == n_rows - 1
    finally:
        batch.make_client, anthropic.Anthropic = real_make, real_sdk
        shutil.rmtree(rd, ignore_errors=True)


def _batch_run_dir(name):
    from lc import evaluate
    rd = ROOT / "runs" / name
    shutil.rmtree(rd, ignore_errors=True)
    evaluate.select([str(ROOT / "data/synth_items.jsonl")], str(rd), n_per_group=1, noise_items=0,
                    framings=["learn"], seed=0, exclude_similar_names=False, force=True)
    return rd


def test_openai_batch_roundtrip():
    """Files upload + /v1/batches + output and error files, against a mocked OpenAI API."""
    from lc import batch
    from lc.providers.openai_compat import OpenAIClient
    rd = _batch_run_dir("_test_oai_batch")
    chat = lambda txt: {"model": "gpt-6.1-sol", "choices": [{"message": {"content": txt}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 400, "completion_tokens": 600,
                                  "completion_tokens_details": {"reasoning_tokens": 500}}}
    state = {}

    def api(req):
        path = req.url.path
        if req.method == "POST" and path == "/v1/chat/completions":            # probe calls
            body = json.loads(req.content)
            if "temperature" in body:
                return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'temperature'"}})
            return httpx.Response(200, json=chat("0.5"))
        if req.method == "POST" and path == "/v1/files":
            raw = req.content.decode("utf-8", "ignore")
            state["lines"] = [json.loads(l) for l in raw.splitlines() if l.startswith('{"custom_id"')]
            return httpx.Response(200, json={"id": "file-in"})
        if req.method == "POST" and path == "/v1/batches":
            state["create"] = json.loads(req.content)
            return httpx.Response(200, json={"id": "batch_1", "status": "validating"})
        if req.method == "GET" and path == "/v1/batches/batch_1":
            n = len(state["lines"])
            return httpx.Response(200, json={"id": "batch_1", "status": "completed", "output_file_id": "file-out",
                                             "error_file_id": "file-err",
                                             "request_counts": {"total": n, "completed": n - 1, "failed": 1}})
        if path == "/v1/files/file-out/content":
            out = [{"custom_id": l["custom_id"], "response": {"status_code": 200, "body": chat("0.25")}, "error": None}
                   for l in state["lines"][:-1]]
            return httpx.Response(200, text="\n".join(json.dumps(o) for o in reversed(out)))   # order not guaranteed
        if path == "/v1/files/file-err/content":
            l = state["lines"][-1]
            return httpx.Response(200, text=json.dumps({"custom_id": l["custom_id"], "response": None,
                                                        "error": {"code": "batch_expired"}}))
        return httpx.Response(404, text=f"unmocked {req.method} {path}")

    http = httpx.Client(transport=httpx.MockTransport(api))

    def fake_client(cfg, effort, rows=None):
        c = OpenAIClient(cfg["name"], cfg["model_id"], effort=effort)
        c._client = httpx.Client(transport=httpx.MockTransport(api))
        return c
    real_make, real_backend = batch.make_client, batch.backend_for
    batch.make_client = fake_client
    batch.backend_for = lambda cfg: batch.OpenAIBatches(cfg, http=http)
    try:
        batch.submit(str(rd), "gpt-6-1-sol", None, True, 4000, 1.0, 2, True, 0, True)
        assert state["create"] == {"input_file_id": "file-in", "endpoint": "/v1/chat/completions",
                                   "completion_window": "24h"}
        b0 = state["lines"][0]["body"]
        assert state["lines"][0]["url"] == "/v1/chat/completions" and b0["model"] == "gpt-6.1-sol"
        assert b0["reasoning_effort"] == "medium" and "temperature" not in b0     # probe adaptation carried over
        batch.collect(str(rd), "gpt-6-1-sol", True)
        recs = [json.loads(l) for l in open(rd / "gpt-6-1-sol.jsonl")]
        ok = [r for r in recs if r["mode"] == "batch" and r.get("error") is None]
        assert ok and all(r["prob"] == 0.25 for r in ok)
        assert abs(ok[0]["cost"] - (400 * 2.0 + 600 * 10.0) / 1e6 * 0.5) < 1e-12
        assert sum(r.get("error") is not None for r in recs) == 1
    finally:
        batch.make_client, batch.backend_for = real_make, real_backend
        shutil.rmtree(rd, ignore_errors=True)


def test_gemini_batch_roundtrip():
    """Inline batchGenerateContent + polling, for both result layouts the docs describe."""
    from lc import batch
    from lc.providers.gemini import GeminiClient
    gen = lambda txt: {"modelVersion": "gemini-3.8-flash", "candidates": [{"finishReason": "STOP", "content": {
        "parts": [{"text": "thinking", "thought": True}, {"text": txt}]}}],
        "usageMetadata": {"promptTokenCount": 400, "candidatesTokenCount": 5, "thoughtsTokenCount": 595}}
    for layout in ("response", "metadata_output"):
        rd = _batch_run_dir("_test_gem_batch")
        state = {}

        def api(req):
            path = req.url.path
            if req.method == "POST" and path.endswith(":generateContent"):
                return httpx.Response(200, json=gen("0.5"))
            if req.method == "POST" and path.endswith(":batchGenerateContent"):
                state["reqs"] = json.loads(req.content)["batch"]["input_config"]["requests"]["requests"]
                return httpx.Response(200, json={"name": "batches/123", "metadata": {"state": "JOB_STATE_PENDING"}})
            if req.method == "GET" and path.endswith("/batches/123"):
                inl = [{"response": gen("0.25")} for _ in state["reqs"][:-1]] + \
                      [{"error": {"code": 400, "message": "bad"}}]
                if layout == "response":
                    return httpx.Response(200, json={"name": "batches/123", "done": True,
                                                     "metadata": {"state": "JOB_STATE_SUCCEEDED"},
                                                     "response": {"inlinedResponses": {"inlinedResponses": inl}}})
                return httpx.Response(200, json={"name": "batches/123", "done": True, "metadata": {
                    "state": "BATCH_STATE_SUCCEEDED", "output": {"inlinedResponses": {"inlinedResponses": inl}}}})
            return httpx.Response(404, text=f"unmocked {req.method} {path}")

        http = httpx.Client(transport=httpx.MockTransport(api))

        def fake_client(cfg, effort, rows=None):
            c = GeminiClient(cfg["name"], cfg["model_id"], effort=effort)
            c._client = httpx.Client(transport=httpx.MockTransport(api))
            return c
        real_make, real_backend = batch.make_client, batch.backend_for
        batch.make_client = fake_client
        batch.backend_for = lambda cfg: batch.GeminiBatches(cfg, http=http)
        try:
            batch.submit(str(rd), "gemini-3-8-flash", None, True, 4000, 1.0, 2, True, 0, True)
            r0 = state["reqs"][0]
            assert r0["request"]["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "MEDIUM"
            assert "systemInstruction" in r0["request"] and "key" in r0["metadata"]
            batch.collect(str(rd), "gemini-3-8-flash", True)
            recs = [json.loads(l) for l in open(rd / "gemini-3-8-flash.jsonl")]
            ok = [r for r in recs if r["mode"] == "batch" and r.get("error") is None]
            assert ok and all(r["prob"] == 0.25 for r in ok), layout
            assert ok[0]["usage"]["output_tokens"] == 600 and ok[0]["usage"]["reasoning_tokens"] == 595
            assert abs(ok[0]["cost"] - (400 * 0.75 + 600 * 3.75) / 1e6 * 0.5) < 1e-12
            assert sum(r.get("error") is not None for r in recs) == 1, layout
        finally:
            batch.make_client, batch.backend_for = real_make, real_backend
            shutil.rmtree(rd, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("all tests passed")


def test_truncated_responses_excluded_and_retried():
    """A response cut off by the token ceiling is not a final answer: analysis drops it, the
    next run retries it, and a complete retry replaces it."""
    from lc.evaluate import select, run, _done_ids
    from lc.analyze import load_run
    rd = ROOT / "runs" / "_test_trunc"
    shutil.rmtree(rd, ignore_errors=True)
    select([str(ROOT / "data/synth_items.jsonl")], str(rd), 2, 0, ["learn"], 0, False, True)
    run(str(rd), "sim-coherent", "medium", True, 1000, 1.0, 2, None, 1, True, 0, True, False)
    path = rd / "sim-coherent.jsonl"
    recs = [json.loads(l) for l in open(path)]
    victim = recs[0]
    victim.update(stop_reason="MAX_TOKENS", prob=0.123)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    key = (victim["item_id"], victim["qid"], victim["framing"], victim["paraphrase"])
    _, _, answers, meta, trunc = load_run(rd)
    assert key not in answers["sim-coherent"]
    assert meta.iloc[0].truncated == 1 and len(trunc) == 1
    assert victim["row_id"] not in _done_ids(path)
    run(str(rd), "sim-coherent", "medium", True, 1000, 1.0, 2, None, 1, True, 0, True, False)
    _, _, answers, meta, trunc = load_run(rd)
    assert key in answers["sim-coherent"] and answers["sim-coherent"][key] != 0.123
    assert meta.iloc[0].truncated == 0 and len(trunc) == 0
    shutil.rmtree(rd, ignore_errors=True)
