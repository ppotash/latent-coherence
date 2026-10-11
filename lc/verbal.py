"""Shared text helpers: verbal probability phrases and source/track-record rendering."""
from __future__ import annotations

from .schema import pct


def freq_word(p: float) -> str:
    """Frequency adverb for a conditional probability (qualitative spec level)."""
    if p <= 0.1: return "almost never"
    if p <= 0.3: return "occasionally"
    if p <= 0.45: return "somewhat less often than not"
    if p <= 0.55: return "about half the time"
    if p <= 0.7: return "somewhat more often than not"
    if p <= 0.9: return "usually"
    return "almost always"


def base_rate_phrase(p: float, what: str) -> str:
    if p < 0.3: return f"{what} usually turn out to be false"
    if p < 0.45: return f"{what} turn out to be false somewhat more often than true"
    if p <= 0.55: return f"{what} turn out to be true about as often as false"
    if p <= 0.7: return f"{what} turn out to be true somewhat more often than false"
    return f"{what} usually turn out to be true"


def careful_share_phrase(p: float) -> str:
    if p < 0.4: return "Most sources are careless."
    if p <= 0.6: return "Careful and careless sources are about equally common."
    return "Most sources are careful."


def source_model_text(spec: str, pi_c: float, a_hi: float, a_lo: float) -> str:
    if spec == "full":
        return (f"Every source is one of two kinds. A careful source's reports are correct "
                f"{pct(a_hi)} of the time; a careless source's reports are correct {pct(a_lo)} of the "
                f"time. Overall, {pct(pi_c)} of sources are careful. A source's kind never changes, "
                f"and different reports are correct or incorrect independently of one another, "
                f"given each source's kind.")
    return ("Every source is one of two kinds. A careful source's reports are almost always "
            "correct; a careless source's reports are correct only somewhat more often than a coin "
            f"flip. {careful_share_phrase(pi_c)} A source's kind never changes, and different "
            "reports are correct or incorrect independently of one another, given each source's kind.")


def track_table(names: list[str], tracks: list[tuple[int, int]]) -> str:
    lines = ["| Source | Past reports later checked | Of those, correct |", "|---|---|---|"]
    for name, (n, k) in zip(names, tracks):
        lines.append(f"| {name[:1].upper() + name[1:]} | {n} | {k} |")
    return "\n".join(lines)
