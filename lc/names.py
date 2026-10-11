"""Deterministic generator of invented, pronounceable names (no real entities)."""
from __future__ import annotations

import numpy as np

_ON = ["b", "br", "d", "dr", "f", "g", "gr", "h", "k", "kl", "l", "m", "n", "p", "pr",
       "r", "s", "st", "t", "tr", "v", "vr", "z", "th", "sh"]
_NU = ["a", "e", "i", "o", "u", "ae", "ia", "ou", "ei"]
_CO = ["", "n", "r", "l", "s", "th", "rn", "nd", "x", "v", "m", "sk"]


class NameGen:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.used: set[str] = set()

    def _word(self, syl_lo=2, syl_hi=3):
        r = self.rng
        k = int(r.integers(syl_lo, syl_hi + 1))
        w = "".join(r.choice(_ON) + r.choice(_NU) for _ in range(k)) + r.choice(_CO)
        return w.capitalize()

    def unique(self, fn):
        for _ in range(1000):
            x = fn()
            if x not in self.used:
                self.used.add(x)
                return x
        raise RuntimeError("name space exhausted")

    def word(self):
        return self.unique(lambda: self._word())

    def person(self):
        return self.unique(lambda: f"{self._word(2, 2)} {self._word(2, 3)}")

    def outlet(self):
        kinds = ["Gazette", "Courier", "Ledger", "Dispatch", "Bulletin", "Monitor", "Review", "Observer"]
        return self.unique(lambda: f"the {self._word(2, 2)} {self.rng.choice(kinds)}")

    def source(self):
        return self.person() if self.rng.random() < 0.5 else self.outlet()
