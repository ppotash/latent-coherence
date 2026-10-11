"""HoVer hybrid family.

Pipeline
  1. select     stratified sample of HoVer claims (heuristic pre-filter drops comparisons etc.)
  2. decompose  LLM: keep only clean conjunctions; split into logically independent atomic
                sub-claims about *named* entities; fictionalize every entity consistently;
                write an exact natural-language negation for each sub-claim
  3. verify     independent LLM pass checks the four properties the ground truth relies on
  4. build      attach a sampled source-reliability world to the fictional sub-claims and
                compute exact probabilities (lc.engine)

Ground-truth semantics: in the generated world each sub-claim is independently true with the
stated base rate, the target claim is true iff all sub-claims are true, and each report
asserts either a sub-claim or its exact negation. Steps 2-3 exist to make those
semantics match the text.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from .engine import sample_world, stmt, all_true, not_all_true, is_careful
from .hover_gold import GOLD
from .names import NameGen
from .schema import Item, pct
from .verbal import source_model_text, track_table, base_rate_phrase

# ----------------------------------------------------------------------------- 1. select

_SKIP = re.compile(r"\b(not both|neither|than|same|older|younger|earlier|later|fewer|more|less|"
                   r"either|or not|both .* and .* not)\b", re.I)


def select_claims(hover_json: str | Path, n: int, seed: int = 0, hops=(2, 3, 4)) -> list[dict]:
    data = json.loads(Path(hover_json).read_text(encoding="utf-8"))
    gold_claims = {g["claim"] for g in GOLD.values()}
    pool = [x for x in data
            if x["num_hops"] in hops and len(x["claim"]) <= 320
            and not _SKIP.search(x["claim"]) and x["claim"] not in gold_claims]
    rng = np.random.default_rng(seed)
    out = []
    per = n // len(hops)
    for h in hops:
        cand = [x for x in pool if x["num_hops"] == h]
        idx = rng.choice(len(cand), size=min(per, len(cand)), replace=False)
        out += [{"uid": cand[i]["uid"], "claim": cand[i]["claim"], "num_hops": h,
                 "hover_label": cand[i]["label"]} for i in idx]
    return out


# ----------------------------------------------------------------------------- 2. decompose

class EntityMap(BaseModel):
    original: str
    fictional: str
    kind: str


class SubClaim(BaseModel):
    original: str
    fictional: str
    fictional_negation: str


class Decomposition(BaseModel):
    usable: bool
    reason: str
    entity_map: list[EntityMap]
    subclaims: list[SubClaim]
    fictional_claim_explicit: str
    fictional_claim_descriptive: str
    bridge_index: int


DECOMPOSE_INSTRUCTIONS = """You are preparing items for a research evaluation of probabilistic reasoning. You will receive one claim from a multi-hop fact-verification dataset. Your job is to turn it into a set of independent atomic statements about invented (fictional) entities.

First decide whether the claim is USABLE. It is usable only if it is true exactly when a list of atomic facts are ALL true (a conjunction). Mark it NOT usable if it is a comparison (earlier/older/more than/closer than), a negated conjunction ("not both", "neither"), a disjunction, a statement that hinges on counting or ranking, garbled or self-contradictory text, or if you cannot decompose it cleanly. When not usable, return empty lists, empty strings, and bridge_index -1.

If usable, produce:
1. entity_map: replace EVERY specific named entity (people, works, organizations, teams, products, buildings, towns, events) with an invented, plausible-sounding name that does not belong to any real, well-known entity. Use one fictional name per original entity, consistently. You may keep countries, large regions, languages, years, numbers, professions, genres, and generic categories. If the claim refers to an entity only by description (e.g. "the band whose album was X", "the director of film Y"), that is a bridge entity: give it an invented name and list it as "(unnamed ...)" in original.
2. subclaims: 2 to 5 atomic statements such that
   (a) the claim is true if and only if all subclaims are true;
   (b) the subclaims are LOGICALLY INDEPENDENT: no subclaim entails, presupposes, or contradicts another, so any combination of them being true or false is possible. Achieve this by naming bridge entities explicitly instead of using descriptions like "the film in which ...";
   (c) each is self-contained: no pronouns or references like "that draft" or "the same";
   (d) fictional_negation is a natural sentence that is true exactly when the subclaim is false (a precise logical negation, not a different specific alternative). Use "It is not the case that ..." when a simple negation would be ambiguous, e.g. for lists or compound descriptions.
   Fill "original" with the subclaim in terms of the real entities, and "fictional" with the version using invented names.
3. fictional_claim_explicit: one fluent sentence, with all entities named, that is exactly the conjunction of the fictional subclaims.
4. fictional_claim_descriptive: the original claim rewritten with fictional names but keeping its original multi-hop structure (descriptions instead of names for bridge entities).
5. bridge_index: index (0-based) of the subclaim that links the bridge entity to the rest of the claim (the 'hop'); if there are several, choose the one most central to the claim.

Do not judge whether the original claim is actually true; it does not matter.

Examples:
"""


def _few_shot() -> str:
    parts = []
    for g in GOLD.values():
        parts.append(f"CLAIM: {g['claim']}\nOUTPUT: {json.dumps(g['output'], ensure_ascii=False)}")
    return "\n\n".join(parts)


def decompose_prompt(claim: str) -> str:
    return f"{DECOMPOSE_INSTRUCTIONS}\n{_few_shot()}\n\nNow process this claim.\nCLAIM: {claim}\nOUTPUT:"


# ----------------------------------------------------------------------------- 3. verify

class Verification(BaseModel):
    conjunction_equivalent: bool
    subclaims_independent: bool
    negations_exact: bool
    fully_fictional_and_consistent: bool
    self_contained: bool
    issues: str
    passed: bool


VERIFY_INSTRUCTIONS = """You are auditing an item prepared for a probabilistic-reasoning evaluation. An item consists of a target claim and a list of atomic subclaims, each with a negation. Check each property strictly and report problems in "issues".

- conjunction_equivalent: the target claim is true if and only if ALL subclaims are true (nothing in the claim is missing from the subclaims, and the subclaims add nothing the claim does not say).
- subclaims_independent: no subclaim entails, presupposes, or contradicts another; every combination of true/false values across subclaims is logically possible.
- negations_exact: each negation is true exactly when its subclaim is false (not a different specific alternative, not ambiguous in scope).
- fully_fictional_and_consistent: the target claim, subclaims, and negations contain no names of real, well-known people, works, organizations, teams, products, buildings, or towns (countries, large regions, years and generic categories are fine), and each fictional name refers to the same entity everywhere.
- self_contained: each subclaim can be understood on its own (no pronouns or references like "that film").
- passed: true only if all five properties hold.

ITEM:
"""


def verify_prompt(dec: dict) -> str:
    body = {"target_claim": dec["fictional_claim_explicit"],
            "subclaims": [{"subclaim": s["fictional"], "negation": s["fictional_negation"]}
                          for s in dec["subclaims"]]}
    return VERIFY_INSTRUCTIONS + json.dumps(body, ensure_ascii=False, indent=1)


def _structural_ok(dec: dict) -> tuple[bool, str]:
    if not dec or not dec.get("usable"):
        return False, "not usable"
    subs = dec.get("subclaims") or []
    if not 2 <= len(subs) <= 5:
        return False, f"{len(subs)} subclaims"
    if not 0 <= dec.get("bridge_index", -1) < len(subs):
        return False, "bad bridge_index"
    if not dec.get("fictional_claim_explicit"):
        return False, "missing explicit claim"
    leaked = [e["original"] for e in dec.get("entity_map", [])
              if not e["original"].startswith("(") and len(e["original"]) > 3
              and e["original"].lower() in json.dumps(
                  [dec["fictional_claim_explicit"]] + [s["fictional"] + s["fictional_negation"] for s in subs]).lower()]
    if leaked:
        return False, f"original names leaked: {leaked}"
    return True, ""


async def run_decompose_verify(claims: list[dict], dec_llm, ver_llm) -> list[dict]:
    async def one(c):
        dec = await dec_llm.json_call(f"dec::{c['uid']}", decompose_prompt(c["claim"]), Decomposition)
        ok, why = _structural_ok(dec)
        rec = {**c, "decomposition": dec, "structural_ok": ok, "structural_issue": why}
        if ok:
            ver = await ver_llm.json_call(f"ver::{c['uid']}", verify_prompt(dec), Verification)
            rec["verification"] = ver
            rec["accepted"] = bool(ver and ver.get("passed"))
        else:
            rec["verification"], rec["accepted"] = None, False
        return rec
    return await asyncio.gather(*(one(c) for c in claims))


# ----------------------------------------------------------------------------- 4. build

def _cap(s):
    return s[:1].upper() + s[1:]


def build_hover_item(rng: np.random.Generator, rec: dict, idx: int, spec: str, max_tries=800) -> Item | None:
    dec = rec["decomposition"]
    subs = dec["subclaims"]
    n, b = len(subs), dec["bridge_index"]
    C = dec["fictional_claim_explicit"]
    best = None
    for t in range(max_tries):
        prior_a = float(rng.choice([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]))
        pi_c = float(rng.choice([0.3, 0.4, 0.5, 0.6, 0.7]))
        a_hi = float(rng.choice([0.85, 0.9, 0.95]))
        a_lo = float(rng.choice([0.5, 0.55, 0.6]))
        m = int(rng.integers(2, min(5, 12 - n) + 1))
        plan = []
        for i in range(n):                         # every statement gets 1-2 reports
            for s in rng.choice(m, size=int(rng.integers(1, 3)), replace=False):
                plan.append((int(s), i))
        for s in range(m):                         # every source reports at least once
            if not any(p[0] == s for p in plan):
                plan.append((s, int(rng.integers(n))))
        track_n = [int(rng.choice([0, 2, 3, 4, 5, 6, 8, 10])) for _ in range(m)]
        w = sample_world(rng, [prior_a] * n, pi_c, a_hi, a_lo, plan, track_n)
        X, nX, Ab = all_true(), not_all_true(), stmt(b, True)
        pX, pAb = w.prob(X), w.prob(Ab)
        if not (0.1 <= pX <= 0.9 and 0.1 <= pAb <= 0.9):
            continue
        # latent 2: kind of one source that reports on the bridge statement
        cands = sorted({p[0] for p in plan if p[1] == b})
        z = int(rng.choice(cands))
        pZ, pX_c, pX_l = w.prob(is_careful(z)), w.prob(X, is_careful(z)), w.prob(X, is_careful(z, False))
        src_ok = 0.1 <= pZ <= 0.9 and abs(pX_c - pX_l) >= 0.1
        cand = (w, z, src_ok, dict(prior_a=prior_a, pi_careful=pi_c, a_hi=a_hi, a_lo=a_lo, m=m,
                                     tracks=w.tracks, plan=plan, latent_source=z))
        if src_ok:
            best = cand
            break
        if best is None:
            best = cand
        if t > max_tries // 2 and best is not None:
            break
    if best is None:
        return None
    w, z, src_ok, params = best

    ng = NameGen(rng)
    srcs = [ng.source() for _ in range(w.m)]
    order = rng.permutation(len(w.reports))
    rep_lines = []
    for j in order:
        r = w.reports[j]
        txt = subs[r.statement]["fictional"] if r.asserts_true else subs[r.statement]["fictional_negation"]
        rep_lines.append(f'- {_cap(srcs[r.source])} reports: "{txt}"')
    if spec == "full":
        base = (f"Before any reports, each individual statement of the kind reported on below is true "
                f"{pct(params['prior_a'])} of the time, independently of the others.")
    else:
        base = (_cap(base_rate_phrase(params["prior_a"], "individual statements of the kind reported on below"))
                + ", independently of one another.")
    ctx = (
        "Situation: The reports below concern fictional people, places, organizations, and works. "
        "Use only the information given here; do not rely on outside knowledge.\n\n"
        f"How sources work: {source_model_text(spec, params['pi_careful'], params['a_hi'], params['a_lo'])}\n\n"
        f"Base rate: {base}\n\n"
        f"Track records:\n{track_table(srcs, w.tracks)}\n\n"
        "Current reports:\n" + "\n".join(rep_lines))

    it = Item(f"hover-{spec}-{idx:05d}", "hover", spec, ctx, meta={
        "hover_uid": rec["uid"], "num_hops": rec["num_hops"], "hover_label": rec.get("hover_label"),
        "original_claim": rec["claim"], "fictional_claim_descriptive": dec["fictional_claim_descriptive"],
        "subclaims": [s["fictional"] for s in subs], "bridge_index": b, "params": params,
        "has_source_latent": src_ok,
        "realised": dict(statements=w.realised_truths, kinds=w.realised_types)})

    ev_true = lambda s: f'the statement "{s}" is true'
    ev_false = lambda s: f'the statement "{s}" is false'
    giv_true = lambda s: f'The statement "{s}" is true.'
    giv_false = lambda s: f'The statement "{s}" is false.'
    Ab_txt = subs[b]["fictional"]

    it.q("X", "direct", ev_true(C), w.prob(all_true()))
    it.q("notX", "negation", ev_false(C), w.prob(not_all_true()))
    for i, s in enumerate(subs):
        it.q(f"A{i}", "latent" if i == b else "component", ev_true(s["fictional"]), w.prob(stmt(i)))
    it.q(f"notA{b}", "latent", ev_false(Ab_txt), w.prob(stmt(b, False)))
    it.q(f"X|A{b}", "conditional", ev_true(C), w.prob(all_true(), stmt(b)), given=[giv_true(Ab_txt)])
    it.q(f"X|notA{b}", "conditional", ev_true(C), w.prob(all_true(), stmt(b, False)), given=[giv_false(Ab_txt)])
    it.total_prob("tp_bridge", "X", [(f"A{b}", f"X|A{b}"), (f"notA{b}", f"X|notA{b}")])
    it.normalization("norm_bridge", [f"A{b}", f"notA{b}"])
    it.normalization("norm_target", ["X", "notX"])
    if src_ok:
        S = srcs[z]
        it.q("Zc", "latent", f"{S} is a careful source", w.prob(is_careful(z)))
        it.q("Zl", "latent", f"{S} is a careless source", w.prob(is_careful(z, False)))
        it.q("X|Zc", "conditional", ev_true(C), w.prob(all_true(), is_careful(z)), given=[f"{_cap(S)} is a careful source."])
        it.q("X|Zl", "conditional", ev_true(C), w.prob(all_true(), is_careful(z, False)), given=[f"{_cap(S)} is a careless source."])
        it.q("Zc|X", "reverse", f"{S} is a careful source", w.prob(is_careful(z), all_true()), given=[giv_true(C)])
        it.q("Zc|notX", "reverse", f"{S} is a careful source", w.prob(is_careful(z), not_all_true()), given=[giv_false(C)])
        it.total_prob("tp_source", "X", [("Zc", "X|Zc"), ("Zl", "X|Zl")])
        it.total_prob("tp_reverse", "Zc", [("X", "Zc|X"), ("notX", "Zc|notX")])
        it.normalization("norm_source", ["Zc", "Zl"])
    return it
