"""API keys, read from the environment or the repo's .env file. Never logged."""
from __future__ import annotations

import os
from pathlib import Path

from .env import load_env

ENV_VAR = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "xai": ["XAI_API_KEY"],
    "zai": ["ZAI_API_KEY"],
    "moonshot": ["MOONSHOT_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
}

_loaded = False


def get_key(provider: str) -> str:
    global _loaded
    if not _loaded:
        load_env(Path(__file__).resolve().parents[1] / ".env")
        _loaded = True
    names = ENV_VAR.get(provider)
    if not names:
        raise SystemExit(f"Unknown provider {provider!r}; expected one of {sorted(ENV_VAR)}")
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    raise SystemExit(f"No API key for {provider}: set {' or '.join(names)} in .env")
