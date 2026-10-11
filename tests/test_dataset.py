"""Correctness checks. Run: python tests/test_dataset.py

1. The enumeration engine agrees with brute-force rejection sampling from the generative model.
2. Diagnostic and causal ground truth agree with brute-force simulation / direct counting.
3. Every identity holds exactly on ground truth, for every family and spec level.
4. Rendered prompts contain no unfilled template slots and every query is rendered.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lc.engine import ReliabilityWorld, Report, stmt, all_true, is_careful   # noqa: E402
from lc.synth import FAMILIES                                                  # noqa: E402
from lc.schema import render_prompts                                           # noqa: E402


def test_engine_vs_rejection_sampling():
    rng = np.random.default_rng(0)
    # 2 statements, 2 sources; small tracks so exact-match rejection sampling is feasible
    w = ReliabilityWorld(prior_true=[0.7, 0.6], pi_careful=0.5, a_hi=0.9, a_lo=0.55,
                         tracks=[(3, 3), (2, 1)],
                         reports=[Report(0, 0, True), Report(1, 0, False), Report(1, 1, True)])
    N = 3_000_000
    T = rng.random((N, 2)) < np.array([0.7, 0.6])
    C = rng.random((N, 2)) < 0.5
    acc = np.where(C, 0.9, 0.55)
    keep = (rng.binomial(3, acc[:, 0]) == 3) & (rng.binomial(2, acc[:, 1]) == 1)
    for r in w.reports:
        correct = rng.random(N) < acc[:, r.source]
        said_true = np.where(correct, T[:, r.statement], ~T[:, r.statement])
        keep &= said_true == r.asserts_true
    Tk, Ck = T[keep], C[keep]
    checks = {
        "P(S0)": (w.prob(stmt(0)), Tk[:, 0].mean()),
        "P(all)": (w.prob(all_true()), Tk.all(1).mean()),
        "P(c1)": (w.prob(is_careful(1)), Ck[:, 1].mean()),
        "P(all|c1)": (w.prob(all_true(), is_careful(1)), Tk[Ck[:, 1]].all(1).mean()),
        "P(c0|all)": (w.prob(is_careful(0), all_true()), Ck[Tk.all(1), 0].mean()),
    }
    print(f"  rejection sampling kept {keep.sum():,} of {N:,}")
    for k, (exact, mc) in checks.items():
        print(f"  {k:10s} exact={exact:.4f}  mc={mc:.4f}")
        assert abs(exact - mc) < 0.01, k


def test_diagnostic_vs_simulation():
    rng = np.random.default_rng(1)
    it = FAMILIES["diagnostic"](rng, 0, "full")
    p = it.meta["params"]
    prior, cpt, pout = np.array(p["prior"]), np.array(p["cpt"]), np.array(p["p_outcome"])
    from lc.synth import DIAG_SKINS  # noqa: F401
    # recover observed test indices from the rendered table order
    tests = [row.split("|")[1].strip() for row in it.context.splitlines() if row.startswith("| ") and "%" in row
             and not row.startswith("| Outcome")]
    obs = {tests.index(k[:1].upper() + k[1:]): v for k, v in p["observed"].items()}
    N = 2_000_000
    d = rng.choice(3, size=N, p=prior / prior.sum())
    keep = np.ones(N, bool)
    for t, v in obs.items():
        keep &= (rng.random(N) < cpt[t, d]) == v
    x = rng.random(N) < pout[d]
    truth = {q.qid: q.truth for q in it.queries}
    print(f"  P(X): exact={truth['X']:.4f} mc={x[keep].mean():.4f}")
    assert abs(truth["X"] - x[keep].mean()) < 0.01
    for k in range(3):
        assert abs(truth[f"Z{k}"] - (d[keep] == k).mean()) < 0.01


def test_causal_counts():
    it = FAMILIES["causal"](np.random.default_rng(2), 0, "full")
    cnt = np.array(it.meta["counts_zxy"])  # [z, x, y]
    t = {q.qid: q.truth for q in it.queries}
    N = cnt.sum()
    assert abs(t["z1"] - cnt[1].sum() / N) < 1e-12
    assert abs(t["Y|x1,z0"] - cnt[0, 1, 1] / cnt[0, 1].sum()) < 1e-12
    assert abs(t["Y|x1"] - cnt[:, 1, 1].sum() / cnt[:, 1].sum()) < 1e-12
    do1 = sum(cnt[z].sum() / N * cnt[z, 1, 1] / cnt[z, 1].sum() for z in (0, 1))
    assert abs(t["Y|do(x1)"] - do1) < 1e-12


def test_identities_and_rendering():
    rng = np.random.default_rng(3)
    for fam, fn in FAMILIES.items():
        for spec in (["full"] if fam == "causal" else ["full", "qualitative"]):
            for i in range(40):
                it = fn(rng, i, spec)
                res = it.identity_residuals({q.qid: q.truth for q in it.queries})
                assert max(abs(v) for v in res.values()) < 1e-9, (it.id, res)
                rows = render_prompts(it)
                assert {r["qid"] for r in rows} == {q.qid for q in it.queries}
                for r in rows:
                    for txt in (r["prompt_numeric"], r["prompt_yesno"] or ""):
                        assert "{" not in txt and "}" not in txt, txt[-300:]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("all tests passed")
