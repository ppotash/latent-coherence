#!/usr/bin/env python3
"""Extra analyses for the write-up, computed from a finished run. No API calls.

  python scripts/paper_extras.py --run runs/main > runs/main/analysis_7models/paper_extras.txt

Sections:
  A. coherence vs accuracy: item-level correlation of |residual| with the direct answer's error
  B. error attribution: errors in P(z), P(X|z) and P(X) for coherent vs incoherent instances
  C. reasoning tokens vs |residual| within each model (across items, and within item across wordings)
  D. robustness: clipping frequency, other clip levels, probability-scale residuals, renormalization
  E. back-door anomalies: every causal instance with a visible residual, with the answer texts
Uses the same shared-item set and parsing as `lceval.py analyze`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lc.analyze import _getter, _item_framings, category, identity_table, load_run   # noqa: E402
from lc.evaluate import is_truncated, is_valid                                        # noqa: E402
from lc.parse import parse_probability                                                # noqa: E402
from lc.schema import read_jsonl                                                      # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

LATENT = "total prob. over latent"
BACKDOOR = "backdoor (do vs adjustment)"


def lg(p, eps=0.01):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def boot_ci(f, n, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [f(rng.integers(0, n, n)) for _ in range(reps)]
    vals = [v for v in vals if np.isfinite(v)]
    return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)


def spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def raw_records(run: Path, models: list[str]) -> dict:
    """(model, item_id, qid, framing, paraphrase) -> latest valid record (for tokens and text)."""
    out = {}
    for m in models:
        latest = {}
        for r in read_jsonl(run / f"{m}.jsonl"):
            rank = 2 if is_valid(r) else (1 if r.get("error") is None else 0)
            old = latest.get(r["row_id"])
            old_rank = -1 if old is None else (2 if is_valid(old) else (1 if old.get("error") is None else 0))
            if rank >= old_rank:
                latest[r["row_id"]] = r
        for r in latest.values():
            if is_valid(r):
                out[(m, r["item_id"], r["qid"], r["framing"], r["paraphrase"])] = r
    return out


def fmt(df):
    return df.to_string(index=False, float_format=lambda x: f"{x:.3f}") if len(df) else "  (none)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--sections", default="ABCDE", help="which sections to print, e.g. C")
    a = ap.parse_args()
    run = Path(a.run)

    items, noise_items, answers, meta, _ = load_run(run)
    if a.models:
        answers = {m: v for m, v in answers.items() if m in a.models}
    models = sorted(answers)
    df = identity_table(items, noise_items, answers)
    common = set.intersection(*df.groupby("model").item_id.agg(set))
    df = df[df.item_id.isin(common)]
    recs = raw_records(run, models)
    print(f"models: {', '.join(models)}\nshared items: {len(common)}\n")

    # one row per latent identity instance with its component queries
    inst = []
    for m in models:
        get_ans = answers[m]
        for iid in sorted(common):
            item = items[iid]
            truth = {q["qid"]: q["truth"] for q in item["queries"]}
            ks = [0, 1, 2] if iid in noise_items else [0]
            for fr in _item_framings(item):
                for k in ks:
                    get = _getter(get_ans, iid, fr, k)
                    for idn in item["identities"]:
                        if idn["kind"] != "total_probability":
                            continue
                        lhs = get(idn["lhs"])
                        terms = [(get(w), get(c)) for w, c in idn["terms"]]
                        if lhs is None or any(w is None or c is None for w, c in terms):
                            continue
                        dec = sum(w * c for w, c in terms)
                        qids = [idn["lhs"]] + [q for t in idn["terms"] for q in t]

                        def tok(q):
                            for f2 in (fr, "none", "custom"):
                                r = recs.get((m, iid, q, f2, k))
                                if r is not None:
                                    return r["usage"]["reasoning_tokens"]
                            return np.nan
                        inst.append({
                            "model": m, "item_id": iid, "identity": idn["name"], "framing": fr, "k": k,
                            "category": category(idn["name"]), "family": item["family"],
                            "spec": item["spec_level"],
                            "direct": lhs, "decomp": dec, "truth": truth[idn["lhs"]],
                            "resid": float(lg(lhs) - lg(dec)),
                            "err_x": float(abs(lg(lhs) - lg(truth[idn["lhs"]]))),
                            "err_decomp": float(abs(lg(dec) - lg(truth[idn["lhs"]]))),
                            "err_z": float(np.mean([abs(lg(w) - lg(truth[wq])) for (w, _), (wq, _) in
                                                    zip(terms, idn["terms"])])),
                            "err_cond": float(np.mean([abs(lg(c) - lg(truth[cq])) for (_, c), (_, cq) in
                                                       zip(terms, idn["terms"])])),
                            "tok_direct": tok(idn["lhs"]),
                            "tok_total": float(np.nansum([tok(q) for q in qids])),
                            "wsum": sum(w for w, _ in terms),
                        })
    I = pd.DataFrame(inst)
    I["abs_r"] = I.resid.abs()
    L0 = I[(I.category == LATENT) & (I.k == 0)]

    # ---------------------------------------------------------------- A
    if "A" in a.sections.upper():
        print("== A. coherence vs accuracy (latent instances, first wording) ==")
        print("   correlation of |residual| with the direct answer's error vs truth; 95% CI by bootstrap\n")
        rows = []
        for m, g in L0.groupby("model"):
            x, y = g.abs_r.to_numpy(), g.err_x.to_numpy()
            rho = spearman(x, y)
            lo, hi = boot_ci(lambda i: spearman(x[i], y[i]), len(x))
            rows.append({"model": m, "n": len(g), "spearman": rho, "ci_lo": lo, "ci_hi": hi,
                         "pearson": np.corrcoef(x, y)[0, 1],
                         "err_x|coherent(|r|<=0.1)": g[g.abs_r <= 0.1].err_x.mean(),
                         "err_x|incoherent(|r|>0.5)": g[g.abs_r > 0.5].err_x.mean(),
                         "n_incoh": int((g.abs_r > 0.5).sum())})
        print(fmt(pd.DataFrame(rows)))

    # ---------------------------------------------------------------- B
    if "B" in a.sections.upper():
        print("\n== B. error attribution: mean log-odds error of each estimate ==")
        print("   err_z: P(z) pieces | err_cond: P(X|z) pieces | err_x: direct P(X) | err_decomp: implied P(X)\n")
        rows = []
        for m, g in L0.groupby("model"):
            for label, h in (("all", g), ("|r|<=0.1", g[g.abs_r <= 0.1]), ("|r|>0.5", g[g.abs_r > 0.5])):
                rows.append({"model": m, "subset": label, "n": len(h), "err_z": h.err_z.mean(),
                             "err_cond": h.err_cond.mean(), "err_x": h.err_x.mean(),
                             "err_decomp": h.err_decomp.mean(),
                             "decomp_closer": (h.err_decomp < h.err_x).mean() if len(h) else np.nan})
        print(fmt(pd.DataFrame(rows)))
        print("\n   pooled over models, by spec level:")
        print(fmt(L0.groupby("spec")[["err_z", "err_cond", "err_x", "err_decomp", "abs_r"]].mean().reset_index()))

    # ---------------------------------------------------------------- C
    if "C" in a.sections.upper():
        print("\n== C. reasoning tokens vs |residual| within each model ==")
        print("   across items: Spearman of |r| with tokens on the direct call and summed over all calls of")
        print("   the identity. within item: on repeated items, |r| and log total tokens are centred within")
        print("   each instance (removing item difficulty), then correlated; its CI resamples whole instances.\n")
        rows = []
        for m, g in L0.groupby("model"):
            x = g.abs_r.to_numpy()
            td, tt = g.tok_direct.to_numpy(), g.tok_total.to_numpy()
            lo, hi = boot_ci(lambda i: spearman(x[i], tt[i]), len(x))
            R = I[(I.model == m) & (I.category == LATENT) & I.item_id.isin(noise_items)]
            R = R.groupby(["item_id", "identity", "framing"]).filter(lambda z: z.k.nunique() == 3).copy()
            R["logtok"] = np.log1p(R["tok_total"])
            key = ["item_id", "identity", "framing"]
            R["c_r"] = R.abs_r - R.groupby(key).abs_r.transform("mean")
            R["c_t"] = R["logtok"] - R.groupby(key)["logtok"].transform("mean")
            within, w_lo, w_hi = np.nan, np.nan, np.nan
            if len(R) > 3 and R.c_t.std() > 0:
                R = R.sort_values(key + ["k"])
                cr = R.c_r.to_numpy().reshape(-1, 3)      # one row per instance, one column per wording
                ct = R.c_t.to_numpy().reshape(-1, 3)

                def wcorr(idx):                           # centred values have mean 0, so this is Pearson r
                    a, b = cr[idx].ravel(), ct[idx].ravel()
                    den = np.sqrt((a * a).sum() * (b * b).sum())
                    return (a * b).sum() / den if den > 0 else np.nan
                within = wcorr(np.arange(len(cr)))
                # cluster bootstrap: resample whole instances, keeping each one's three wordings together
                w_lo, w_hi = boot_ci(wcorr, len(cr))
            rows.append({"model": m, "n": len(g), "rho_direct": spearman(x, td), "rho_total": spearman(x, tt),
                         "ci_lo": lo, "ci_hi": hi,
                         "tok_total|r<=0.1": np.median(tt[x <= 0.1]) if (x <= 0.1).any() else np.nan,
                         "tok_total|r>0.5": np.median(tt[x > 0.5]) if (x > 0.5).any() else np.nan,
                         "within_item_r": within, "w_ci_lo": w_lo, "w_ci_hi": w_hi,
                         "n_within": R.groupby(key).ngroups})
        print(fmt(pd.DataFrame(rows)))

    # ---------------------------------------------------------------- D
    if "D" in a.sections.upper():
        print("\n== D. robustness ==")
        rows = []
        for m in models:
            vals = [r["prob"] for (mm, iid, *_), r in recs.items() if mm == m and iid in common
                    and r.get("prob") is not None]
            v = np.array(vals, dtype=float)
            g = L0[L0.model == m]

            def mean_item(eps):
                rr = (lg(g.direct, eps) - lg(g.decomp, eps))
                return pd.Series(np.abs(rr), index=g.index).groupby(g.item_id).mean().mean()
            rows.append({"model": m, "answers": len(v), "pct_clipped": 100 * np.mean((v <= 0.01) | (v >= 0.99)),
                         "pct_exact_0_or_1": 100 * np.mean((v == 0) | (v == 1)),
                         "mean|r|_eps.001": mean_item(0.001), "mean|r|_eps.01": mean_item(0.01),
                         "mean|r|_eps.05": mean_item(0.05),
                         "mean|P-D|_pp": 100 * pd.Series((g.direct - g.decomp).abs().values)
                         .groupby(g.item_id.values).mean().mean(),
                         "median|P-D|_pp": 100 * (g.direct - g.decomp).abs().median(),
                         "mean|r|_renorm": pd.Series(np.abs(lg(g.direct) - lg(g.decomp / g.wsum)))
                         .groupby(g.item_id.values).mean().mean(),
                         "median|sum P(z)-1|": (g.wsum - 1).abs().median()})
        print(fmt(pd.DataFrame(rows)))
        print("\n   rank of models by mean |r| at each clip level:")
        D = pd.DataFrame(rows).set_index("model")
        for c in ("mean|r|_eps.001", "mean|r|_eps.01", "mean|r|_eps.05", "mean|P-D|_pp"):
            print(f"   {c:18s} " + " < ".join(D[c].sort_values().index))

    # ---------------------------------------------------------------- E
    if "E" in a.sections.upper():
        print("\n== E. back-door instances with |residual| > 0.01 ==")
        B = I[(I.category == BACKDOOR) & (I.k == 0) & (I.abs_r > 0.01)]
        if not len(B):
            print("  (none)")
        for _, b in B.iterrows():
            item = items[b.item_id]
            idn = next(x for x in item["identities"] if x["name"] == b.identity)
            print(f"\n-- {b.model} | {b.item_id} | {b.identity} | residual {b.resid:+.3f}")
            print(f"   direct P(y|do(x)) = {b.direct:.4f}   own adjustment = {b.decomp:.4f}   truth = {b.truth:.4f}")
            for q in [idn["lhs"]] + [q for t in idn["terms"] for q in t]:
                r = None
                for f2 in (b.framing, "none", "custom"):
                    r = recs.get((b.model, b.item_id, q, f2, 0)) or r
                tr = next(x["truth"] for x in item["queries"] if x["qid"] == q)
                if r is not None:
                    txt = (r.get("text") or "").strip().replace("\n", " ")
                    print(f"   {q:10s} answer {parse_probability(r.get('text'))[0]:.4f}  truth {tr:.4f}  "
                          f"| {txt[:240]}")


if __name__ == "__main__":
    main()
