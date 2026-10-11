"""Item schema, coherence identities, and prompt rendering.

An *item* is one context (a situation description plus records) together with a set of
probability *queries* about it and a set of *identities* that the answers must satisfy
if the respondent is probabilistically coherent. Every query carries its exact
ground-truth probability under the generating model.

Identity kinds
--------------
total_probability : P(lhs) = sum_k P(w_k) * P(c_k)
                    e.g. P(X) = P(Z=z0) P(X|z0) + P(Z=z1) P(X|z1)
normalization     : sum_k P(q_k) = 1   (states of one variable, or an event and its negation)

The harness computes, for each identity, the residual between the model's direct answer
and the answer implied by its own other answers. Ground truth satisfies every identity
exactly (checked in tests/test_dataset.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Iterable

STEMS = [
    "What is the probability that {e}?",
    "How likely is it that {e}?",
    "Estimate the probability that {e}.",
]

FRAMINGS = {
    # evidential conditioning: new information arrives
    "learn": "Suppose you additionally learn, from a completely reliable check, the following: {g}",
    # hypothetical conditioning: same mathematical object, different wording
    "assume": "Assume hypothetically the following: {g} Taking this assumption as given:",
}

NUMERIC_SUFFIX = "Respond with only a probability between 0 and 1."
YESNO_TEMPLATE = "{given}Is it the case that {e}? Answer Yes or No."

PREAMBLE = (
    "Read the following situation and records carefully, then answer the question with "
    "your best probability estimate."
)


@dataclass
class Query:
    qid: str
    qtype: str                      # direct | negation | latent | conditional | reverse | interventional | observational | adjustment
    event: str                      # clause, e.g. "the Orvane bridge reopens before the festival"
    truth: float
    given: list[str] = field(default_factory=list)   # full sentences; conditions
    framings: list[str] | None = None                # defaults to ["learn","assume"] if given else ["none"]
    custom_questions: list[str] | None = None        # overrides stems + framing (used for do()-style questions)
    meta: dict = field(default_factory=dict)


@dataclass
class Identity:
    name: str
    kind: str                       # total_probability | normalization
    lhs: str | None                 # qid (total_probability only)
    terms: list                     # [[w_qid, c_qid], ...] or [qid, ...]


@dataclass
class Item:
    id: str
    family: str
    spec_level: str
    context: str
    queries: list[Query] = field(default_factory=list)
    identities: list[Identity] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # --- construction helpers -------------------------------------------------
    def q(self, qid, qtype, event, truth, given=(), **kw) -> str:
        assert qid not in {x.qid for x in self.queries}, qid
        assert 0.0 <= truth <= 1.0 + 1e-12, (qid, truth)
        self.queries.append(Query(qid, qtype, event, float(min(max(truth, 0.0), 1.0)), list(given), **kw))
        return qid

    def total_prob(self, name, lhs, terms):
        self.identities.append(Identity(name, "total_probability", lhs, [list(t) for t in terms]))

    def normalization(self, name, qids):
        self.identities.append(Identity(name, "normalization", None, list(qids)))

    def to_dict(self):
        return asdict(self)

    # --- checks ---------------------------------------------------------------
    def identity_residuals(self, values: dict[str, float]) -> dict[str, float]:
        """Residual of every identity under `values` (qid -> probability)."""
        out = {}
        for idn in self.identities:
            if idn.kind == "total_probability":
                implied = sum(values[w] * values[c] for w, c in idn.terms)
                out[idn.name] = values[idn.lhs] - implied
            elif idn.kind == "normalization":
                out[idn.name] = sum(values[q] for q in idn.terms) - 1.0
            else:
                raise ValueError(idn.kind)
        return out


# --- rendering ----------------------------------------------------------------

def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def render_prompts(item: Item | dict) -> list[dict]:
    """Expand an item into one flat row per (query, framing, paraphrase)."""
    if isinstance(item, dict):
        item = item_from_dict(item)
    rows = []
    head = f"{PREAMBLE}\n\n{item.context.strip()}\n\n"
    for q in item.queries:
        if q.custom_questions:
            variants = [("custom", i, txt, txt) for i, txt in enumerate(q.custom_questions)]
        else:
            framings = q.framings or (["learn", "assume"] if q.given else ["none"])
            variants = []
            for fr in framings:
                given_txt = "" if fr == "none" else FRAMINGS[fr].format(g=" ".join(q.given)) + " "
                for i, stem in enumerate(STEMS):
                    variants.append((fr, i, given_txt + stem.format(e=q.event), given_txt))
        for fr, i, question, given_txt in variants:
            if fr == "custom":
                yesno = None  # custom questions are already full questions; harness uses numeric form
            else:
                yesno = head + YESNO_TEMPLATE.format(given=given_txt, e=q.event)
            rows.append({
                "row_id": f"{item.id}::{q.qid}::{fr}::p{i}",
                "item_id": item.id,
                "family": item.family,
                "spec_level": item.spec_level,
                "qid": q.qid,
                "qtype": q.qtype,
                "framing": fr,
                "paraphrase": i,
                "prompt_numeric": head + question + "\n\n" + NUMERIC_SUFFIX,
                "prompt_yesno": yesno,
                "truth": q.truth,
            })
    return rows


def item_from_dict(d: dict) -> Item:
    it = Item(d["id"], d["family"], d["spec_level"], d["context"], meta=d.get("meta", {}))
    it.queries = [Query(**q) for q in d["queries"]]
    it.identities = [Identity(**x) for x in d["identities"]]
    return it


def write_jsonl(path, rows: Iterable[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(p: float) -> str:
    return f"{round(100 * p)}%"
