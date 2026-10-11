"""Minimal async JSON-mode LLM client with an on-disk cache.

Backends
--------
gemini   : google-genai SDK; needs GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment
fixture  : returns canned outputs from a dict {key: output}; used for offline tests

Every call is keyed; results are appended to a JSONL cache so a crashed or interrupted run
resumes without re-paying for finished calls.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path


class LLM:
    def __init__(self, backend: str, model: str, cache_path: str | Path, concurrency: int = 16,
                 temperature: float = 0.2, fixture: dict | None = None, max_retries: int = 6):
        self.backend, self.model, self.temperature = backend, model, temperature
        self.cache_path = Path(cache_path)
        self.sem = asyncio.Semaphore(concurrency)
        self.fixture = fixture or {}
        self.max_retries = max_retries
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec["output"]
        self._lock = asyncio.Lock()
        self._client = None
        if backend == "gemini":
            from google import genai
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise SystemExit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) to use the gemini backend.")
            self._client = genai.Client(api_key=key)

    async def _append(self, key, output):
        async with self._lock:
            self.cache[key] = output
            with self.cache_path.open("a") as f:
                f.write(json.dumps({"key": key, "model": self.model, "output": output}, ensure_ascii=False) + "\n")

    async def json_call(self, key: str, prompt: str, schema) -> dict | None:
        """schema: a pydantic model class describing the JSON object to return."""
        if key in self.cache:
            return self.cache[key]
        if self.backend == "fixture":
            out = self.fixture.get(key)
            if out is not None:
                await self._append(key, out)
            return out
        from google.genai import types
        cfg = types.GenerateContentConfig(temperature=self.temperature,
                                          response_mime_type="application/json",
                                          response_schema=schema)
        async with self.sem:
            for attempt in range(self.max_retries):
                try:
                    resp = await self._client.aio.models.generate_content(
                        model=self.model, contents=prompt, config=cfg)
                    out = json.loads(resp.text)
                    schema.model_validate(out)
                    await self._append(key, out)
                    return out
                except Exception as e:  # rate limits, transient 5xx, malformed JSON
                    wait = min(60, 2 ** attempt + random.random())
                    print(f"[llm] {key}: {type(e).__name__}: {str(e)[:120]} -- retry in {wait:.0f}s")
                    await asyncio.sleep(wait)
        print(f"[llm] {key}: giving up")
        return None
