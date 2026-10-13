"""Extract a probability from a model's free-text answer.

Accepted forms: an explicit fraction ("1/3", "1 in 3"), a number in [0, 1] ("0.35", ".35",
"0", "1"), a percentage ("35%"). A bare number in (1, 100] without a percent sign is read
as a percentage but flagged.

Most models answer with the number alone. Some (GLM, for one) put the answer first, in
bold, and then explain, and the explanation can contain other numbers in [0, 1]. So when
the text holds several different candidates, the answer is chosen in this order:

  1. the whole text is one probability                      -> flag ""
  2. the first non-empty line is just a probability         -> flag "first_line"
     (markdown, "~", "≈", "approximately", a trailing period are ignored)
  3. the last "answer is / answer: / probability:" marker   -> flag "marker"
  4. exactly one distinct probability in **bold**           -> flag "bold"
  5. otherwise the last candidate                           -> flag "ambiguous"

Flags other than "" and "assumed_percent" mark answers worth spot-checking; `analyze`
counts them per model.
"""
from __future__ import annotations

import re

_FRAC = re.compile(r"(?<![\d.])(\d+)\s*(?:/|in|out of)\s*(\d+)(?![\d.])", re.I)
_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?|\.\d+)\s*(%|percent)?", re.I)
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.S)
_MARKER = re.compile(r"(?:final\s+answer|answer|probability)\s*(?:is|:|=|≈|of)?\s*", re.I)
_DECOR = re.compile(r"[*_`#>~≈]|\$|\\\(|\\\)|\bapprox(?:imately)?\.?|\babout\b|\baround\b|\bP\s*=", re.I)


def _candidates(t: str) -> list[tuple[int, float, str]]:
    """All probability candidates in t as (end position, value, flag), in text order."""
    cands: list[tuple[int, float, str]] = []
    frac_spans = []
    for m in _FRAC.finditer(t):
        frac_spans.append((m.start(), m.end()))
        a, b = int(m.group(1)), int(m.group(2))
        if b > 0 and a <= b:
            cands.append((m.end(), a / b, ""))
    for m in _NUM.finditer(t):
        if any(s <= m.start() < e for s, e in frac_spans):
            continue
        v, pct = float(m.group(1)), bool(m.group(2))
        if pct:
            if 0 <= v <= 100:
                cands.append((m.end(), v / 100, ""))
        elif 0 <= v <= 1:
            cands.append((m.end(), v, ""))
        elif 1 < v <= 100 and len(t.strip()) <= 12:     # e.g. a bare "35"
            cands.append((m.end(), v / 100, "assumed_percent"))
    return sorted(cands)


def _only_probability(s: str) -> tuple[float, str] | None:
    """s is a probability and nothing else (ignoring decoration), e.g. '**≈ 0.342**'."""
    core = _DECOR.sub(" ", s).strip().rstrip(".").strip()
    if not core:
        return None
    c = _candidates(core)
    if not c or len({round(v, 9) for _, v, _ in c}) != 1:
        return None
    # the candidate(s) must account for the whole string ("0.35 (35%)" is one answer)
    rest = _NUM.sub(" ", _FRAC.sub(" ", core))
    rest = re.sub(r"[()%=:,\s]|\bpercent\b", "", rest, flags=re.I)
    return (c[-1][1], c[-1][2]) if not rest else None


def parse_probability(text: str | None) -> tuple[float | None, str]:
    """Return (probability or None, flag). flag is "" when unambiguous."""
    if not text or not text.strip():
        return None, "empty"
    t = text.replace(" ", " ").replace(",", "")
    cands = _candidates(t)
    if not cands:
        return None, "no_probability"
    values = {round(v, 9) for _, v, _ in cands}
    if len(values) == 1:                                      # 1. one answer, however often stated
        return cands[-1][1], cands[-1][2]

    lines = [ln for ln in t.splitlines() if ln.strip()]
    first = _only_probability(lines[0]) if lines else None
    if first is not None:                                     # 2. answer-first layout
        return first[0], "first_line"

    markers = list(_MARKER.finditer(t))
    for m in reversed(markers):                               # 3. explicit answer marker
        after = [c for c in cands if c[0] > m.end() and c[0] - m.end() <= 40]
        if after:
            return after[0][1], "marker"

    bold_vals = set()
    for m in _BOLD.finditer(t):                               # 4. a single bolded probability
        inner = m.group(1) or m.group(2)
        bold_vals |= {round(v, 9) for _, v, _ in _candidates(inner)}
    if len(bold_vals) == 1:
        return bold_vals.pop(), "bold"

    return cands[-1][1], "ambiguous"                          # 5. fall back to the last one
