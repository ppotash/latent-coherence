"""Model config (config/models.toml) -> client instance."""
from __future__ import annotations

import math
import random
import tomllib
from pathlib import Path

from .base import Completion, Usage

CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.toml"


def load_models(path: str | Path = CONFIG) -> dict[str, dict]:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    models = cfg.get("models", {})
    for name, m in models.items():
        m["name"] = name
    return models


def make_client(cfg: dict, effort: str, rows: list[dict] | None = None):
    p = cfg["provider"]
    if cfg.get("model_id", "TODO") == "TODO":
        raise SystemExit(f"{cfg['name']}: set model_id in config/models.toml first")
    if p == "anthropic":
        from .anthropic import AnthropicClient
        return AnthropicClient(cfg["name"], cfg["model_id"], effort=effort)
    if p == "gemini":
        from .gemini import GeminiClient
        return GeminiClient(cfg["name"], cfg["model_id"], effort=effort)
    if p in ("openai", "xai", "zai", "moonshot", "deepseek"):
        from . import openai_compat as oc
        cls = {"openai": oc.OpenAIClient, "xai": oc.XAIClient, "zai": oc.ZAIClient,
               "moonshot": oc.MoonshotClient, "deepseek": oc.DeepSeekClient}[p]
        return cls(cfg["name"], cfg["model_id"], effort=effort)
    if p == "sim":
        return SimClient(cfg["name"], rows or [], **cfg.get("sim", {}))
    raise SystemExit(f"{cfg['name']}: unknown provider {p!r}")


class SimClient:
    """Offline stand-in that answers from ground truth, for testing the harness.

    noise   : SD of Gaussian noise added in log-odds space to every answer
    shrink  : multiplies the log-odds of *conditional* answers (<1 = conditionals pulled
              toward 0.5, a simple mechanism that breaks total-probability coherence)
    garble  : fraction of answers returned in an unparsable form
    """
    provider = "sim"

    def __init__(self, name, rows, noise=0.0, shrink=1.0, garble=0.0, seed=0):
        self.name, self.model_id = name, name
        self.noise, self.shrink, self.garble = noise, shrink, garble
        self._seed = seed
        self._by_text = {r["prefix"] + r["question"]: r for r in rows}

    def complete(self, prompt, *, max_tokens, temperature, thinking=True, effort=None):
        row = self._by_text[prompt.as_text()]
        rng = random.Random(f"{self._seed}:{row['row_id']}")
        p = min(max(row["truth"], 1e-4), 1 - 1e-4)
        z = math.log(p / (1 - p))
        if row["qtype"] in ("conditional", "reverse", "adjustment"):
            z *= self.shrink
        z += rng.gauss(0, self.noise)
        v = 1 / (1 + math.exp(-z))
        if self.noise == 0 and self.shrink == 1:
            v = row["truth"]
        if rng.random() < self.garble:
            text = "I cannot determine this."
        else:
            text = rng.choice([f"{v:.4f}", f"Probability: **{v:.4f}**", f"{100 * v:.2f}%"])
        return Completion(text=text, model_id=self.model_id, stop_reason="end_turn",
                          usage=Usage(input_tokens=len(prompt.as_text()) // 4, output_tokens=40))

    def close(self):
        pass
