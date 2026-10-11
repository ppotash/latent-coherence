"""Load KEY=VALUE pairs from a .env file into os.environ (no external dependency).

Real environment variables always win over the file. Lines starting with '#' and blank
lines are ignored; surrounding single or double quotes on values are stripped.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> list[str]:
    p = Path(path)
    if not p.is_file():
        return []
    loaded = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        if key and val and key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded
