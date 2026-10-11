"""Fully programmatic families with exact ground truth.

  source      claim truth X, latent = kind (careful/careless) of one reporting source
  diagnostic  3-state hidden cause D inferred from an inspection log; targets = future outcome
              and an unperformed test
  causal      observational count table with a confounder Z; interventional P(Y|do(X)) vs the
              back-door adjustment, plus the observational total-probability identity
"""
from __future__ import annotations

import numpy as np

from .engine import sample_world, stmt, is_careful
from .names import NameGen
from .schema import Item, pct
from .verbal import (base_rate_phrase, source_model_text, track_table, freq_word)

MID = (0.10, 0.90)


def _mid(p, lo=MID[0], hi=MID[1]):
    return lo <= p <= hi


def _r05(x):
    return float(np.clip(round(x * 20) / 20, 0.05, 0.95))


def _cap(s):
    return s[:1].upper() + s[1:]


# =============================================================================
# 1. source family
# =============================================================================

CLAIMS = [
    ("the {W} bridge will reopen before the harvest festival", "the {W} bridge will not reopen before the harvest festival"),
    ("the {W} town council approved the new water tariff", "the {W} town council did not approve the new water tariff"),
    ("{P} won this year's {W} regional chess title", "{P} did not win this year's {W} regional chess title"),
    ("the {W} mine produced more copper this year than last year", "the {W} mine did not produce more copper this year than last year"),
    ("the {W} ferry service will resume next month", "the {W} ferry service will not resume next month"),
    ("{P} will be appointed director of the {W} museum", "{P} will not be appointed director of the {W} museum"),
    ("the {W} reservoir is above its seasonal average", "the {W} reservoir is not above its seasonal average"),
    ("the {W} cooperative signed a supply deal with {V} Mills", "the {W} cooperative did not sign a supply deal with {V} Mills"),
    ("the {W} textile plant will close by the end of the year", "the {W} textile plant will not close by the end of the year"),
    ("{P} wrote the anonymous pamphlet circulating in {W}", "{P} did not write the anonymous pamphlet circulating in {W}"),
    ("the {W} observatory detected a new comet last week", "the {W} observatory did not detect a new comet last week"),
    ("the {W} rail strike will end before Friday", "the {W} rail strike will not end before Friday"),
    ("the {W} orchard lost more than a third of its crop to frost", "the {W} orchard did not lose more than a third of its crop to frost"),
    ("{P} will run for mayor of {W}", "{P} will not run for mayor of {W}"),
]


def _fill(tpl, names):
    return tpl.format(**names)


def make_source_item(rng: np.random.Generator, idx: int, spec: str, max_tries=500) -> Item:
    ng = NameGen(rng)
    for _ in range(max_tries):
        m = int(rng.integers(1, 4))
        prior = _r05(rng.uniform(0.2, 0.8))
        pi_c = _r05(rng.uniform(0.3, 0.7))
        a_hi = float(rng.choice([0.85, 0.9, 0.95]))
        a_lo = float(rng.choice([0.5, 0.55, 0.6]))
        track_n = [int(rng.choice([0, 2, 3, 4, 5, 6, 8, 10, 12])) for _ in range(m)]
        w = sample_world(rng, [prior], pi_c, a_hi, a_lo, [(s, 0) for s in range(m)], track_n)
        z = int(rng.integers(0, m))   # latent: kind of source z
        X, nX = stmt(0, True), stmt(0, False)
        Zc, Zl = is_careful(z, True), is_careful(z, False)
        pX = w.prob(X)
        pZ = w.prob(Zc)
        pX_c, pX_l = w.prob(X, Zc), w.prob(X, Zl)
        if not (_mid(pX) and _mid(pZ) and abs(pX_c - pX_l) >= 0.15):
            continue
        # passed: render
        names = {"W": ng.word(), "V": ng.word(), "P": ng.person()}
        tpl = CLAIMS[int(rng.integers(len(CLAIMS)))]
        claim, neg = _fill(tpl[0], names), _fill(tpl[1], names)
        srcs = [ng.source() for _ in range(m)]
        reports = "\n".join(
            f"- {_cap(srcs[r.source])} reports that {claim if r.asserts_true else neg}." for r in w.reports)
        if spec == "full":
            base = f"Before any reports, claims of this kind are true {pct(prior)} of the time."
        else:
            base = _cap(base_rate_phrase(prior, "claims of this kind")) + "."
        ctx = (f"Situation: Some sources have reported on whether {claim}.\n\n"
               f"How sources work: {source_model_text(spec, pi_c, a_hi, a_lo)}\n\n"
               f"Base rate: {base}\n\n"
               f"Track records:\n{track_table(srcs, w.tracks)}\n\n"
               f"Current reports:\n{reports}")
        it = Item(f"source-{spec}-{idx:05d}", "source", spec, ctx, meta={
            "params": dict(prior=prior, pi_careful=pi_c, a_hi=a_hi, a_lo=a_lo, m=m,
                           tracks=w.tracks, latent_source=z),
            "realised": dict(claim_true=w.realised_truths[0], kinds=w.realised_types)})
        S = srcs[z]
        it.q("X", "direct", claim, pX)
        it.q("notX", "negation", neg, w.prob(nX))
        it.q("Zc", "latent", f"{S} is a careful source", pZ)
        it.q("Zl", "latent", f"{S} is a careless source", w.prob(Zl))
        it.q("X|Zc", "conditional", claim, pX_c, given=[f"{_cap(S)} is a careful source."])
        it.q("X|Zl", "conditional", claim, pX_l, given=[f"{_cap(S)} is a careless source."])
        it.q("Zc|X", "reverse", f"{S} is a careful source", w.prob(Zc, X), given=[f"In fact, {claim}."])
        it.q("Zc|notX", "reverse", f"{S} is a careful source", w.prob(Zc, nX), given=[f"In fact, {neg}."])
        it.total_prob("tp_latent", "X", [("Zc", "X|Zc"), ("Zl", "X|Zl")])
        it.total_prob("tp_reverse", "Zc", [("X", "Zc|X"), ("notX", "Zc|notX")])
        it.normalization("norm_latent", ["Zc", "Zl"])
        it.normalization("norm_target", ["X", "notX"])
        return it
    raise RuntimeError("could not sample a mid-range source item")


# =============================================================================
# 2. diagnostic family
# =============================================================================

DIAG_SKINS = [
    dict(
        unit="plant bed {W}", what="the cause of the leaf spotting in plant bed {W}",
        states=["a fungal infection", "a nutrient deficiency", "heat stress"],
        tests=["the leaf-tissue stain shows dark threads", "the soil nitrogen strip reads low",
               "the leaves curl upward at midday", "the spots have yellow halos",
               "the newest leaves are pale"],
        outcome="more than half of the plants in plant bed {W} will drop their leaves within two weeks",
        intro="A greenhouse technician is investigating leaf spotting in plant bed {W}. Exactly one of the following is the cause."),
    dict(
        unit="pump unit {W}", what="the fault in pump unit {W}",
        states=["a worn bearing", "a cracked impeller", "a clogged intake"],
        tests=["the vibration sensor shows a high-frequency spike", "the outlet pressure is below normal",
               "the motor housing is hotter than normal", "a grinding noise is heard at start-up",
               "the intake filter shows debris buildup"],
        outcome="pump unit {W} will fail completely within the next 30 days",
        intro="A maintenance engineer is diagnosing pump unit {W}. Exactly one of the following faults is present."),
    dict(
        unit="the {W} reservoir", what="the cause of the discoloration in the {W} reservoir",
        states=["an algal bloom", "runoff from upstream farms", "corroded intake pipes"],
        tests=["the water has a green tint", "nitrate levels are elevated", "iron levels are elevated",
               "the discoloration worsens after rain", "dissolved oxygen is low at dawn"],
        outcome="the {W} reservoir will be closed to swimmers before the end of the month",
        intro="A water inspector is investigating discoloration in the {W} reservoir. Exactly one of the following is the cause."),
]


def make_diagnostic_item(rng: np.random.Generator, idx: int, spec: str, max_tries=2000) -> Item:
    ng = NameGen(rng)
    skin = DIAG_SKINS[int(rng.integers(len(DIAG_SKINS)))]
    W = ng.word()
    K, T = 3, len(skin["tests"])
    for _ in range(max_tries):
        prior = rng.dirichlet([3, 3, 3])
        prior = np.array([_r05(p) for p in prior])
        prior[-1] = round(1 - prior[:-1].sum(), 2)
        if prior[-1] < 0.05:
            continue
        cpt = np.array([[_r05(rng.beta(1, 1)) for _ in range(K)] for _ in range(T)])  # P(test+|state)
        pout = np.array([_r05(rng.beta(1, 1)) for _ in range(K)])                     # P(outcome|state)
        if pout.max() - pout.min() < 0.3:
            continue
        d = int(rng.choice(K, p=prior / prior.sum()))
        order = rng.permutation(T)
        n_obs = int(rng.integers(2, 4))
        obs_idx, held = order[:n_obs], int(order[n_obs])
        obs = {int(t): bool(rng.random() < cpt[t, d]) for t in obs_idx}
        like = np.ones(K)
        for t, v in obs.items():
            like *= cpt[t] if v else 1 - cpt[t]
        post = prior * like
        post /= post.sum()
        pX_d, pY_d = pout, cpt[held]
        pX, pY = float(post @ pX_d), float(post @ pY_d)
        if not (_mid(pX) and _mid(pY) and post.max() <= 0.9):
            continue
        if float(post @ np.abs(pX_d - pX)) < 0.12:   # decomposition must matter
            continue
        break
    else:
        raise RuntimeError("could not sample a mid-range diagnostic item")

    fmt = lambda s: s.format(W=W)
    states, tests = skin["states"], [fmt(t) for t in skin["tests"]]
    outcome = fmt(skin["outcome"])
    lines = [fmt(skin["intro"])]
    if spec == "full":
        lines.append("\nHow common each cause is, before any inspection:")
        lines += [f"- {_cap(s)}: {pct(p)}" for s, p in zip(states, prior)]
        hdr = "| Finding | " + " | ".join(_cap(s) for s in states) + " |"
        lines += ["\nHow often each finding occurs under each cause (findings occur independently of one another once the cause is fixed):",
                  hdr, "|---|" + "---|" * K]
        lines += [f"| {_cap(tests[t])} | " + " | ".join(pct(cpt[t, k]) for k in range(K)) + " |" for t in range(T)]
        lines += [f"| Outcome: {outcome} | " + " | ".join(pct(pout[k]) for k in range(K)) + " |"]
    else:
        lines.append("\nWhat is known about each cause (findings occur independently of one another once the cause is fixed):")
        rare = lambda p: "rare" if p < 0.25 else ("common" if p > 0.45 else "fairly common")
        for k, s in enumerate(states):
            facts = "; ".join(f"{tests[t]} ({freq_word(cpt[t, k])})" for t in range(T))
            lines.append(f"- {_cap(s)}: {rare(prior[k])} as a cause. When it is the cause: {facts}. "
                         f"Outcome, {outcome}: {freq_word(pout[k])}.")
    lines.append("\nInspection log so far:")
    for t in obs_idx:
        lines.append(f"- Checked whether {tests[t]}: {'yes' if obs[int(t)] else 'no'}.")
    ctx = "\n".join(lines)

    it = Item(f"diagnostic-{spec}-{idx:05d}", "diagnostic", spec, ctx, meta={
        "params": dict(prior=prior.tolist(), cpt=cpt.tolist(), p_outcome=pout.tolist(),
                       observed={tests[t]: v for t, v in obs.items()}, held_out_test=tests[held]),
        "realised": dict(cause=states[d])})
    the_what = fmt(skin["what"])
    it.q("X", "direct", outcome, pX)
    it.q("notX", "negation", f"it is not the case that {outcome}", 1 - pX)
    it.q("Y", "direct", f"if the inspector next checks whether {tests[held]}, the answer will be yes", pY)
    zq = []
    for k, s in enumerate(states):
        zq.append(it.q(f"Z{k}", "latent", f"{the_what} is {s}", float(post[k])))
    for k, s in enumerate(states):
        g = [f"{_cap(the_what)} is {s}."]
        it.q(f"X|Z{k}", "conditional", outcome, float(pX_d[k]), given=g)
        it.q(f"Y|Z{k}", "conditional",
             f"if the inspector next checks whether {tests[held]}, the answer will be yes", float(pY_d[k]), given=g)
    it.total_prob("tp_latent_X", "X", [(f"Z{k}", f"X|Z{k}") for k in range(K)])
    it.total_prob("tp_latent_Y", "Y", [(f"Z{k}", f"Y|Z{k}") for k in range(K)])
    it.normalization("norm_latent", zq)
    it.normalization("norm_target", ["X", "notX"])
    return it


# =============================================================================
# 3. causal family
# =============================================================================

CAUSAL_SKINS = [
    dict(unit="plot", units="plots", ref="it", scene="A farm survey recorded {N} plots in the {W} valley.",
         z_col="Shaded", z_sg="is shaded", z_neg_sg="is not shaded", z_pl="are shaded", z_neg_pl="are not shaded",
         x_col="Fertilized", x_sg="received the fertilizer", x_neg_sg="did not receive the fertilizer",
         x_pl="received the fertilizer", x_neg_pl="did not receive the fertilizer",
         x_do_pos="the fertilizer is applied to it", x_do_neg="the fertilizer is withheld from it",
         y="has a high yield", y_fut="will have a high yield", y_col="High yield",
         decider="Farmers chose which plots to fertilize; the fertilizer was not assigned at random.",
         assume="Assume that whether a plot is shaded is the only factor that affected both which plots were fertilized and their yields."),
    dict(unit="worker", units="workers", ref="the worker", scene="A safety audit at the {W} works recorded {N} workers over one year.",
         z_col="Night shift", z_sg="works the night shift", z_neg_sg="works the day shift",
         z_pl="work the night shift", z_neg_pl="work the day shift",
         x_col="Attended course", x_sg="attended the safety course", x_neg_sg="did not attend the safety course",
         x_pl="attended the safety course", x_neg_pl="did not attend the safety course",
         x_do_pos="the worker is required to attend the safety course",
         x_do_neg="the worker is kept from attending the safety course",
         y="had no workplace incident during the year", y_fut="will have no workplace incident during the year", y_col="No incident",
         decider="Workers decided for themselves whether to attend the course.",
         assume="Assume that shift (night or day) is the only factor that affected both course attendance and incidents."),
    dict(unit="battery", units="batteries", ref="it", scene="A test lab at {W} recorded {N} batteries.",
         z_col="Older line", z_sg="came from the older production line", z_neg_sg="came from the newer production line",
         z_pl="came from the older production line", z_neg_pl="came from the newer production line",
         x_col="Slow protocol", x_sg="was charged with the slow protocol", x_neg_sg="was charged with the fast protocol",
         x_pl="were charged with the slow protocol", x_neg_pl="were charged with the fast protocol",
         x_do_pos="it is charged with the slow protocol", x_do_neg="it is charged with the fast protocol",
         y="kept more than 80% of its capacity after 500 cycles", y_fut="will keep more than 80% of its capacity after 500 cycles", y_col="Kept >80% capacity",
         decider="Technicians chose the charging protocol for each battery; it was not assigned at random.",
         assume="Assume that the production line is the only factor that affected both the protocol chosen and capacity retention."),
]


def make_causal_item(rng: np.random.Generator, idx: int, spec: str = "full", max_tries=2000) -> Item:
    ng = NameGen(rng)
    sk = CAUSAL_SKINS[int(rng.integers(len(CAUSAL_SKINS)))]
    W = ng.word()
    for _ in range(max_tries):
        N = int(rng.choice([200, 300, 400, 500, 600]))
        pz = rng.uniform(0.25, 0.75)
        px_z = np.array([rng.uniform(0.05, 0.5), rng.uniform(0.5, 0.95)])   # confounded assignment
        if rng.random() < 0.5:
            px_z = px_z[::-1]
        py = rng.uniform(0.05, 0.95, size=(2, 2))                         # py[x, z]
        z = rng.random(N) < pz
        x = rng.random(N) < np.where(z, px_z[1], px_z[0])
        y = rng.random(N) < py[x.astype(int), z.astype(int)]
        cnt = np.zeros((2, 2, 2), int)   # [z, x, y]
        np.add.at(cnt, (z.astype(int), x.astype(int), y.astype(int)), 1)
        if (cnt.sum(axis=2) < 15).any():
            continue
        P = cnt / N
        Pz = P.sum(axis=(1, 2))                                  # P(z)
        Pzx = P.sum(axis=2)                                      # P(z, x)
        Py_xz = cnt[:, :, 1] / cnt.sum(axis=2)                  # [z, x]
        Pz_x = Pzx / Pzx.sum(axis=0, keepdims=True)              # P(z | x) [z, x]
        do = [float(sum(Pz[zz] * Py_xz[zz, xx] for zz in (0, 1))) for xx in (0, 1)]
        obs = [float(sum(Pz_x[zz, xx] * Py_xz[zz, xx] for zz in (0, 1))) for xx in (0, 1)]
        if max(abs(do[0] - obs[0]), abs(do[1] - obs[1])) < 0.08:
            continue
        if not all(_mid(v, 0.05, 0.95) for v in do + obs):
            continue
        if not ((Py_xz >= 0.03) & (Py_xz <= 0.97)).all():
            continue
        break
    else:
        raise RuntimeError("could not sample causal item")

    u, us, ref = sk["unit"], sk["units"], sk["ref"]
    lines = [sk["scene"].format(N=N, W=W), sk["decider"], sk["assume"], "",
             f"Counts of {us}, by whether the {u} {sk['z_sg']} ('{sk['z_col']}'), whether it {sk['x_sg']} "
             f"('{sk['x_col']}'), and whether it {sk['y']} ('{sk['y_col']}'):", "",
             f"| {sk['z_col']} | {sk['x_col']} | {sk['y_col']}: yes | {sk['y_col']}: no |", "|---|---|---|---|"]
    for zz in (1, 0):
        for xx in (1, 0):
            lines.append(f"| {'yes' if zz else 'no'} | {'yes' if xx else 'no'} | {cnt[zz, xx, 1]} | {cnt[zz, xx, 0]} |")
    ctx = "\n".join(lines)

    picks = [f"A {u} is chosen at random from all {N} recorded {us}",
             f"Pick one of the {N} recorded {us} uniformly at random",
             f"Consider a {u} drawn at random from the records"]
    zsg = {1: sk["z_sg"], 0: sk["z_neg_sg"]}
    zpl = {1: sk["z_pl"], 0: sk["z_neg_pl"]}
    xpl = {1: sk["x_pl"], 0: sk["x_neg_pl"]}

    def among(cond):
        return [f"A {u} is chosen at random from among the recorded {us} that {cond}",
                f"Pick uniformly at random one of the recorded {us} that {cond}",
                f"Consider a {u} drawn at random from the recorded {us} that {cond}"]

    def ask(prefixes, tail):
        return [f"{p}. What is the probability that {ref} {tail}?" for p in prefixes]

    it = Item(f"causal-{spec}-{idx:05d}", "causal", spec, ctx, meta={
        "counts_zxy": cnt.tolist(), "skin": u,
        "simpson_reversal": bool(np.sign(do[1] - do[0]) != np.sign(obs[1] - obs[0]))})
    for xx in (1, 0):
        dox = sk["x_do_pos"] if xx else sk["x_do_neg"]
        it.q(f"Y|do(x{xx})", "interventional", sk["y"], do[xx], custom_questions=[
            f"{p}. Suppose that {dox}, regardless of whether {ref} {sk['z_sg']}. "
            f"What is the probability that {ref} {sk['y_fut']}?" for p in picks])
        it.q(f"Y|x{xx}", "observational", sk["y"], obs[xx], custom_questions=ask(among(xpl[xx]), sk["y"]))
        for zz in (1, 0):
            it.q(f"Y|x{xx},z{zz}", "adjustment", sk["y"], float(Py_xz[zz, xx]),
                 custom_questions=ask(among(f"{zpl[zz]} and {xpl[xx]}"), sk["y"]))
            it.q(f"z{zz}|x{xx}", "adjustment", zsg[zz], float(Pz_x[zz, xx]),
                 custom_questions=ask(among(xpl[xx]), zsg[zz]))
    for zz in (1, 0):
        it.q(f"z{zz}", "adjustment", zsg[zz], float(Pz[zz]), custom_questions=ask(picks, zsg[zz]))
    for xx in (1, 0):
        it.total_prob(f"backdoor_x{xx}", f"Y|do(x{xx})", [(f"z{zz}", f"Y|x{xx},z{zz}") for zz in (1, 0)])
        it.total_prob(f"tp_obs_x{xx}", f"Y|x{xx}", [(f"z{zz}|x{xx}", f"Y|x{xx},z{zz}") for zz in (1, 0)])
        it.normalization(f"norm_z|x{xx}", [f"z1|x{xx}", f"z0|x{xx}"])
    it.normalization("norm_z", ["z1", "z0"])
    return it


FAMILIES = {
    "source": make_source_item,
    "diagnostic": make_diagnostic_item,
    "causal": make_causal_item,
}
