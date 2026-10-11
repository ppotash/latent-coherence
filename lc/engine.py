"""Exact inference for 'source reliability' worlds.

Generative model (shared by the synthetic `source` family and the HoVer hybrid):

  * n statements S_1..S_n, each independently true with prior prior_true[i]
  * m sources, each independently 'careful' (prob pi_careful) or 'careless'
  * a careful source's reports are correct with prob a_hi, a careless one's with a_lo
  * each source has a track record: n_past past reports checked, k_past of them correct
    (Binomial given its type)
  * current reports: source s says about statement i either "S_i" or its contradiction
    "S_i'" (exactly one of the two is true); a report is correct with prob a_type

Joint over the n + m binary latent variables is enumerated exactly (2^(n+m) states;
we keep n + m <= 12).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Report:
    source: int
    statement: int
    asserts_true: bool   # True -> source asserts S_i ; False -> asserts the contradiction S_i'


@dataclass
class ReliabilityWorld:
    prior_true: list[float]
    pi_careful: float
    a_hi: float
    a_lo: float
    tracks: list[tuple[int, int]]           # per source (n_past, k_past)
    reports: list[Report]
    # hidden ground-truth realisation (kept for bookkeeping; not used by inference)
    realised_truths: list[bool] = field(default_factory=list)
    realised_types: list[bool] = field(default_factory=list)

    @property
    def n(self):
        return len(self.prior_true)

    @property
    def m(self):
        return len(self.tracks)

    def _joint(self):
        if hasattr(self, "_cache"):
            return self._cache
        n, m = self.n, self.m
        assert n + m <= 14, "enumeration too large"
        states = np.array(list(itertools.product([0, 1], repeat=n + m)), dtype=bool)
        truths, careful = states[:, :n], states[:, n:]
        logw = np.zeros(len(states))
        for i, p in enumerate(self.prior_true):
            logw += np.where(truths[:, i], math.log(p), math.log1p(-p))
        for s, (npast, kpast) in enumerate(self.tracks):
            logw += np.where(careful[:, s], math.log(self.pi_careful), math.log1p(-self.pi_careful))
            acc = np.where(careful[:, s], self.a_hi, self.a_lo)
            logw += kpast * np.log(acc) + (npast - kpast) * np.log1p(-acc)
        for r in self.reports:
            acc = np.where(careful[:, r.source], self.a_hi, self.a_lo)
            correct = truths[:, r.statement] == r.asserts_true
            logw += np.where(correct, np.log(acc), np.log1p(-acc))
        w = np.exp(logw - logw.max())
        self._cache = (truths, careful, w)
        return self._cache

    def prob(self, event, given=None) -> float:
        """event/given: functions (truths[N,n], careful[N,m]) -> bool[N]."""
        truths, careful, w = self._joint()
        mask = np.ones(len(w), dtype=bool) if given is None else given(truths, careful)
        denom = w[mask].sum()
        if denom <= 0:
            raise ZeroDivisionError("conditioning on a zero-probability event")
        return float(w[mask & event(truths, careful)].sum() / denom)


# convenient event constructors ------------------------------------------------

def stmt(i, value=True):
    return lambda t, c: t[:, i] == value


def all_true():
    return lambda t, c: t.all(axis=1)


def not_all_true():
    return lambda t, c: ~t.all(axis=1)


def is_careful(s, value=True):
    return lambda t, c: c[:, s] == value


def sample_world(rng: np.random.Generator, prior_true, pi_careful, a_hi, a_lo,
                 report_plan: list[tuple[int, int]], track_n: list[int]) -> ReliabilityWorld:
    """report_plan: list of (source, statement) pairs; track_n: n_past per source."""
    truths = [bool(rng.random() < p) for p in prior_true]
    types = [bool(rng.random() < pi_careful) for _ in track_n]
    tracks = []
    for s, npast in enumerate(track_n):
        acc = a_hi if types[s] else a_lo
        tracks.append((npast, int(rng.binomial(npast, acc))))
    reports = []
    for s, i in report_plan:
        acc = a_hi if types[s] else a_lo
        correct = rng.random() < acc
        reports.append(Report(s, i, truths[i] if correct else (not truths[i])))
    return ReliabilityWorld(list(prior_true), pi_careful, a_hi, a_lo, tracks, reports, truths, types)
