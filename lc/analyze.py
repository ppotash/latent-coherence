"""Coherence analysis of a run directory.

For each model, item, identity and conditioning framing (learn / assume / custom) this
computes, from the model's OWN answers (paraphrase 0):

  total_probability   direct  = P^(lhs)
                      decomp  = sum_k P^(w_k) P^(c_k)
                      decomp_renorm: same with the weights P^(w_k) rescaled to sum to 1, which
                      separates "the latent's distribution doesn't normalize" from
                      "marginalization itself is off"
                      residuals: direct - decomp, in probability and log-odds
  normalization       sum_k P^(q_k) - 1

Noise floor (noise-floor items only, where every prompt was asked in 3 paraphrases): each
identity is recomputed with paraphrase k for every term (k = 0, 1, 2); the SD across k is
the residual you would see from wording alone.

Calibration (ground truth): |logit P^ - logit P_true| for the direct and decomposed
estimates of the same quantity, i.e. "would the model do better by marginalizing its own
answers?"

Pairwise model comparison: item-level mean |log-odds residual| per category, paired across
models; reports d_z and the number of items needed for 80% power after Bonferroni
correction over the model pairs.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from .evaluate import is_truncated, is_valid
from .parse import parse_probability
from .schema import read_jsonl

EPS = 0.01          # probabilities are clipped to [EPS, 1 - EPS] before taking log-odds


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def category(name: str) -> str:
    if name.startswith("norm"):
        return "normalization"
    if name.startswith("backdoor"):
        return "backdoor (do vs adjustment)"
    if name.startswith("tp_obs"):
        return "observational total prob."
    if name.startswith("tp_reverse"):
        return "reverse total prob."
    return "total prob. over latent"


# ----------------------------------------------------------------------------- load

def load_run(run_dir: str | Path):
    run = Path(run_dir)
    items = {d["id"]: d for d in read_jsonl(run / "items.jsonl")}
    noise_items = {r["item_id"] for r in read_jsonl(run / "rows.jsonl") if r.get("noise_floor_item")}
    answers, meta, truncated = {}, [], []
    for p in sorted(run.glob("*.jsonl")):
        if p.name in ("rows.jsonl", "items.jsonl"):
            continue
        model = p.stem
        recs = read_jsonl(p)
        latest = {}
        for r in recs:
            # a complete answer beats a truncated one beats an error; among equals, later wins
            rank = 2 if is_valid(r) else (1 if r.get("error") is None else 0)
            old = latest.get(r["row_id"])
            old_rank = -1 if old is None else (2 if is_valid(old) else (1 if old.get("error") is None else 0))
            if rank >= old_rank:
                latest[r["row_id"]] = r
        # Re-parse every answer with the current parser, so all models are scored by the same
        # rules even if their runs predate a parser fix; count how many answers that changed.
        n_reparsed, n_check = 0, 0
        for r in latest.values():
            if r.get("error") is None and r.get("text") is not None:
                prob, flag = parse_probability(r["text"])
                if (prob is None) != (r.get("prob") is None) or (
                        prob is not None and abs(prob - r["prob"]) > 1e-9):
                    n_reparsed += 1
                r["prob"], r["parse_flag"] = prob, flag
                n_check += flag in ("first_line", "marker", "bold", "ambiguous")
        a = {}
        for r in latest.values():
            if is_valid(r) and r.get("prob") is not None:
                a[(r["item_id"], r["qid"], r["framing"], r["paraphrase"])] = r["prob"]
        answers[model] = a
        ok = [r for r in latest.values() if r.get("error") is None]
        trunc = [r for r in ok if is_truncated(r)]
        for r in trunc:
            it = items.get(r["item_id"], {})
            text = (r.get("text") or "").strip().replace("\n", " ")
            truncated.append({"model": model, "group": f"{it.get('family')}/{it.get('spec_level')}",
                              "item_id": r["item_id"], "qid": r["qid"], "paraphrase": r["paraphrase"],
                              "parsed_prob": r.get("prob"),
                              "output_tokens": r["usage"]["output_tokens"],
                              "reasoning_tokens": r["usage"]["reasoning_tokens"],
                              "text_chars": len(text), "text_tail": text[-300:]})
        meta.append({
            "model": model, "effort": ",".join(sorted({str(r.get("effort")) for r in ok})) or None,
            "prompts": len(latest), "answered": len(ok) - len(trunc),
            "errors": len(latest) - len(ok),
            "truncated": len(trunc),
            "unparsed": sum(r.get("prob") is None for r in ok if not is_truncated(r)),
            "multi_number": n_check,          # answers holding several numbers (see lc/parse.py)
            "reparsed": n_reparsed,           # answers whose value changed under the current parser
            # every paid call counts, including superseded retries and truncated responses
            "cost_usd": round(sum(r.get("cost") or 0 for r in recs), 4),
            "mean_output_tokens": round(np.mean([r["usage"]["output_tokens"] for r in ok]), 1) if ok else None,
            "mean_reasoning_tokens": round(np.mean([r["usage"]["reasoning_tokens"] for r in ok]), 1) if ok else None,
        })
    return items, noise_items, answers, pd.DataFrame(meta), pd.DataFrame(truncated)


def _item_framings(item: dict) -> list[str]:
    fr = set()
    for q in item["queries"]:
        if q.get("custom_questions"):
            fr.add("custom")
        elif q.get("given"):
            fr.update(q.get("framings") or ["learn", "assume"])
    return sorted(fr) or ["none"]


def _getter(ans: dict, item_id: str, framing: str, k: int):
    def get(qid):
        for fr in (framing, "none", "custom"):
            v = ans.get((item_id, qid, fr, k))
            if v is not None:
                return v
        return None
    return get


def identity_table(items: dict, noise_items: set, answers: dict) -> pd.DataFrame:
    out = []
    for model, ans in answers.items():
        for iid, item in items.items():
            truth = {q["qid"]: q["truth"] for q in item["queries"]}
            ks = [0, 1, 2] if iid in noise_items else [0]
            for fr in _item_framings(item):
                for k in ks:
                    get = _getter(ans, iid, fr, k)
                    for idn in item["identities"]:
                        rec = {"model": model, "item_id": iid, "family": item["family"],
                               "spec": item["spec_level"], "identity": idn["name"],
                               "category": category(idn["name"]), "kind": idn["kind"],
                               "framing": fr, "paraphrase": k,
                               "similar_names": bool(item.get("meta", {}).get("similar_name_pairs"))}
                        if idn["kind"] == "total_probability":
                            lhs = get(idn["lhs"])
                            terms = [(get(w), get(c)) for w, c in idn["terms"]]
                            if lhs is None or any(w is None or c is None for w, c in terms):
                                continue
                            wsum = sum(w for w, _ in terms)
                            dec = sum(w * c for w, c in terms)
                            dec_rn = dec / wsum if wsum > 0 else np.nan
                            t_lhs = truth[idn["lhs"]]
                            rec.update(direct=lhs, decomp=dec, decomp_renorm=dec_rn, weight_sum=wsum,
                                       resid_prob=lhs - dec,
                                       resid_logit=float(logit(lhs) - logit(dec)),
                                       resid_logit_renorm=float(logit(lhs) - logit(dec_rn)),
                                       truth=t_lhs,
                                       err_direct=float(abs(logit(lhs) - logit(t_lhs))),
                                       err_decomp=float(abs(logit(dec) - logit(t_lhs))))
                        else:
                            vals = [get(q) for q in idn["terms"]]
                            if any(v is None for v in vals):
                                continue
                            rec.update(resid_prob=sum(vals) - 1.0)
                        out.append(rec)
    return pd.DataFrame(out)


def framing_table(items: dict, answers: dict) -> pd.DataFrame:
    """|logit P(learn) - logit P(assume)| for every conditional query, paraphrase 0."""
    out = []
    for model, ans in answers.items():
        for iid, item in items.items():
            for q in item["queries"]:
                a, b = ans.get((iid, q["qid"], "learn", 0)), ans.get((iid, q["qid"], "assume", 0))
                if a is not None and b is not None:
                    out.append({"model": model, "item_id": iid, "family": item["family"], "qid": q["qid"],
                                "abs_logit_diff": float(abs(logit(a) - logit(b)))})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- summaries

def _boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def summarize(df: pd.DataFrame, fr: pd.DataFrame) -> dict:
    p0 = df[df.paraphrase == 0]
    tp = p0[p0.kind == "total_probability"]
    rows = []
    for (model, cat), g in tp.groupby(["model", "category"]):
        item_level = g.assign(a=g.resid_logit.abs()).groupby("item_id").a.mean()
        lo, hi = _boot_ci(item_level.values)
        rows.append({"model": model, "category": cat, "n_items": len(item_level),
                     "mean_abs_logit": item_level.mean(), "ci_lo": lo, "ci_hi": hi,
                     "median_abs_logit": g.resid_logit.abs().median(),
                     "median_abs_logit_renorm": g.resid_logit_renorm.abs().median(),
                     "median_abs_prob": g.resid_prob.abs().median(),
                     "mean_signed_logit": g.resid_logit.mean()})
    coherence = pd.DataFrame(rows)

    by_family = (tp.assign(a=tp.resid_logit.abs())
                 .groupby(["model", "category", "family", "spec"])
                 .agg(n=("a", "size"), mean_abs_logit=("a", "mean"), median_abs_logit=("a", "median"),
                      p90_abs_logit=("a", lambda v: v.quantile(0.9)),
                      mean_err_direct=("err_direct", "mean"), mean_err_decomp=("err_decomp", "mean"))
                 .reset_index())

    norm = p0[p0.kind == "normalization"]
    normalization = (norm.assign(a=norm.resid_prob.abs()).groupby("model")
                     .agg(n=("a", "size"), median_abs=("a", "median"), mean_signed=("resid_prob", "mean"))
                     .reset_index())

    noise = noise_decomposition(df)

    calib = (tp.groupby(["model", "category"])
             .apply(lambda g: pd.Series({"n": int(len(g)), "mean_err_direct": g.err_direct.mean(),
                                         "mean_err_decomp": g.err_decomp.mean(),
                                         "frac_decomp_better": _frac_better(g)}),
                    include_groups=False)
             .reset_index())

    framing = (fr.groupby("model").abs_logit_diff.agg(["size", "median", "mean"]).reset_index()
               if len(fr) else pd.DataFrame())

    return {"coherence": coherence, "by_family": by_family, "normalization": normalization, "noise": noise,
            "calibration": calib, "framing": framing}


def _frac_better(g):
    d = g.err_direct - g.err_decomp
    return float(((d > 1e-9) + 0.5 * (d.abs() <= 1e-9)).mean())


def noise_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Split residual variance into wording noise and systematic incoherence.

    On noise-floor items every identity instance (item x identity x framing) has a residual
    for each of 3 paraphrases. A coherent-but-noisy model has residuals that average to zero
    for every instance; a systematically incoherent one has an instance-specific nonzero
    residual that persists across wordings. With r_ik = mu_i + e_ik:
        within  = mean_i var_k(r_ik)                 (wording noise)
        between = var_i(mean_k r_ik) - within / 3    (systematic, noise-corrected)
    `systematic_sd` = sqrt(max(between, 0)): incoherence that wording noise does not explain.
    Note: `between` measures spread around the mean residual; a constant bias shared by all
    instances shows up in `mean_resid` instead.
    """
    tp = df[df.kind == "total_probability"]
    keys = ["model", "item_id", "identity", "framing"]
    cols = ["model", "category", "n_instances", "noise_sd", "systematic_sd", "sys_sd_ci_lo", "sys_sd_ci_hi",
            "systematic_share", "mean_resid"]
    if not len(tp):
        return pd.DataFrame(columns=cols)
    g = tp.groupby(keys).filter(lambda x: x.paraphrase.nunique() == 3)
    if not len(g):
        return pd.DataFrame(columns=cols)
    inst = g.groupby(keys + ["category", "spec", "family"]).resid_logit.agg(["mean", "var"]).reset_index()
    groups = (list(inst.groupby(["model", "category"]))
              + [((m, "all"), y) for m, y in inst.groupby("model")]
              + [((m, f"  latent | {sp}"), y) for (m, sp), y in
                 inst[inst.category == "total prob. over latent"].groupby(["model", "spec"])]
              + [((m, f"  latent | {fa}/{sp}"), y) for (m, fa, sp), y in
                 inst[inst.category == "total prob. over latent"].groupby(["model", "family", "spec"])])
    out = []
    for (model, cat), x in groups:
        if len(x) < 3:
            continue
        within = x["var"].mean()
        between = x["mean"].var(ddof=1) - within / 3
        sys_var = max(between, 0.0)
        # bootstrap over identity instances (resampling instances keeps each one's 3 paraphrases)
        rng = np.random.default_rng(0)
        mu, va = x["mean"].to_numpy(), x["var"].to_numpy()
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(x), len(x))
            b = mu[idx].var(ddof=1) - va[idx].mean() / 3
            boots.append(math.sqrt(max(b, 0.0)))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append({"model": model, "category": cat, "n_instances": len(x),
                    "noise_sd": math.sqrt(within), "systematic_sd": math.sqrt(sys_var),
                    "sys_sd_ci_lo": lo, "sys_sd_ci_hi": hi,
                    "systematic_share": sys_var / (sys_var + within) if (sys_var + within) > 1e-12 else np.nan,
                    "mean_resid": x["mean"].mean()})
    return pd.DataFrame(out, columns=cols)


def _signflip_p(d: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation p-value for mean(d) = 0. Residual differences
    are heavy-tailed, so this avoids the t-test's normality assumption."""
    d = d[np.isfinite(d)]
    if len(d) == 0 or not np.any(d):
        return 1.0
    obs = abs(d.mean())
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(null >= obs - 1e-12)) / (n_perm + 1))


def pairwise_power(df: pd.DataFrame, alpha=0.05, power=0.8, min_effect=0.2,
                   planned_models: int | None = None) -> pd.DataFrame:
    """Sample-size planning from the SD of paired item-level differences.

    `items_needed_at_min_effect` uses only the observed SD and a pre-chosen smallest
    difference worth detecting (`min_effect`, in mean |log-odds residual|). Sizing a study
    from variance alone, not from the observed effect, is the standard internal-pilot rule:
    the first-stage items can then count toward the final analysis without meaningfully
    inflating false positives. `planned_models` sets the Bonferroni correction for the number
    of models you intend to compare, if more than are in the run so far."""
    p0 = df[(df.paraphrase == 0) & (df.kind == "total_probability")]
    models = sorted(p0.model.unique())
    pairs = list(itertools.combinations(models, 2))
    if not pairs:
        return pd.DataFrame()
    n_pairs = max(len(pairs), (planned_models * (planned_models - 1) // 2) if planned_models else 0)
    z_a = NormalDist().inv_cdf(1 - alpha / (2 * n_pairs))
    z_b = NormalDist().inv_cdf(power)
    out = []
    for cat, g in p0.groupby("category"):
        lvl = g.assign(a=g.resid_logit.abs()).groupby(["model", "item_id"]).a.mean().unstack(0)
        for m1, m2 in pairs:
            if m1 not in lvl or m2 not in lvl:
                continue
            d = (lvl[m1] - lvl[m2]).dropna()
            if len(d) < 3:
                continue
            sd = d.std(ddof=1)
            dz = d.mean() / sd if sd > 0 else np.nan
            n_obs = math.ceil(((z_a + z_b) / abs(dz)) ** 2) if dz and np.isfinite(dz) else None
            n_min = math.ceil(((z_a + z_b) * sd / min_effect) ** 2) if sd > 0 else None
            p = _signflip_p(d.to_numpy())
            out.append({"category": cat, "model_a": m1, "model_b": m2, "n_items": len(d),
                        "mean_diff": d.mean(), "sd_diff": sd, "d_z": dz,
                        "p": p, "p_bonf": min(1.0, p * n_pairs),
                        "items_needed_at_min_effect": n_min, "items_needed_at_observed_effect": n_obs})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- report

def analyze(run_dir: str, exclude_similar_names: bool = False, min_effect: float = 0.2,
            planned_models: int | None = None, all_items: bool = False,
            models: list[str] | None = None, out_name: str = "analysis"):
    run = Path(run_dir)
    items, noise_items, answers, meta, trunc = load_run(run)
    if models:
        unknown = sorted(set(models) - set(answers))
        if unknown:
            raise SystemExit(f"no results for {unknown} in {run}; have {sorted(answers)}")
        answers = {m: v for m, v in answers.items() if m in models}
        meta = meta[meta.model.isin(models)].reset_index(drop=True)
        if len(trunc):
            trunc = trunc[trunc.model.isin(models)]
    if not answers:
        raise SystemExit("no model results in this run yet")
    df = identity_table(items, noise_items, answers)
    if exclude_similar_names and len(df):
        df = df[~df.similar_names]
    note = ""
    if len(df) and df.model.nunique() > 1 and not all_items:
        # models capped to different item sets: summarize everyone on the shared items, so the
        # pooled numbers are comparable (per-group tables are unaffected in meaning)
        per_model = df.groupby("model").item_id.agg(set)
        common = set.intersection(*per_model)
        if any(len(v) != len(common) for v in per_model):
            df = df[df.item_id.isin(common)]
            note = (f"\nNOTE: models answered different item sets; all tables use the {len(common)} items "
                    f"answered by every model (--all-items to disable).")
    fr = framing_table(items, answers)
    s = summarize(df, fr)
    pw = pairwise_power(df, min_effect=min_effect, planned_models=planned_models)

    outd = run / out_name
    outd.mkdir(exist_ok=True)
    df.to_csv(outd / "identity_residuals.csv", index=False)
    fr.to_csv(outd / "framing_differences.csv", index=False)
    meta.to_csv(outd / "calls.csv", index=False)
    for k, v in s.items():
        v.to_csv(outd / f"{k}.csv", index=False)
    pw.to_csv(outd / "pairwise_power.csv", index=False)
    if len(trunc):
        trunc.to_csv(outd / "truncated.csv", index=False)

    if note:
        print(note)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    fmt = lambda d: d.to_string(index=False, float_format=lambda x: f"{x:.3f}") if len(d) else "  (none)"
    print("== calls ==\n" + fmt(meta))
    if len(trunc):
        by = trunc.groupby(["model", "group"]).size().reset_index(name="truncated")
        print("\n   truncated = cut off by the token ceiling before a final answer. These are excluded\n"
              "   from every table below (rerun `run`/`batch submit` with a larger --max-tokens to retry\n"
              "   them); details in analysis/truncated.csv\n" + fmt(by))
    print("\n== coherence: |direct - decomposed| in log-odds (paraphrase 0; CI = bootstrap over items) ==\n"
          + fmt(s["coherence"]))
    print("\n== coherence by family and spec level (total-probability identities, paraphrase 0) ==\n"
          + fmt(s["by_family"]))
    print("\n== noise vs systematic incoherence (noise-floor items; log-odds) ==\n"
          "   noise_sd: residual spread from rewording alone | systematic_sd: incoherence that persists\n"
          "   across rewordings (noise-corrected; CI = bootstrap over instances) | systematic_share: its\n"
          "   fraction of residual variance\n"
          + fmt(s["noise"]))
    print("\n== normalization: sum of P over a partition minus 1 ==\n" + fmt(s["normalization"]))
    print("\n== framing: |logit P(.|learn) - logit P(.|assume)| ==\n" + fmt(s["framing"]))
    print("\n== calibration vs ground truth: which estimate is closer? (log-odds error) ==\n" + fmt(s["calibration"]))
    if len(pw):
        print(f"\n== pairwise: paired item-level differences in mean |log-odds residual| ==\n"
              f"   p: sign-flip permutation test; p_bonf: Bonferroni over model pairs (significant if < 0.05)\n"
              f"   items_needed_at_min_effect: for 80% power to detect a {min_effect} log-odds difference\n"
              + fmt(pw))
    print(f"\nCSV files in {outd}")
    return df, s, pw
