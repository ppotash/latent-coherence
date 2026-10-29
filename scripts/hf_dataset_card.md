---
license: cc-by-sa-4.0
language:
- en
pretty_name: Latent Coherence
size_categories:
- 1K<n<10K
task_categories:
- question-answering
tags:
- probabilistic-reasoning
- coherence
- law-of-total-probability
- causal-inference
- llm-evaluation
configs:
- config_name: items
  default: true
  data_files:
  - split: synthetic
    path: items/synthetic.parquet
  - split: hover
    path: items/hover.parquet
- config_name: prompts
  data_files:
  - split: synthetic
    path: prompts/synthetic.parquet
  - split: hover
    path: prompts/hover.parquet
---

# Latent Coherence

Do LLM probability judgments obey the law of total probability? This dataset tests whether a
model's direct answer P(X) agrees with its own decomposition Σ P(z)·P(X|z), where the latent z
must be inferred from records in the context. A causal family runs the same check on
P(y | do(x)) against the back-door adjustment. Every query has an exact ground-truth
probability, so coherence and accuracy can be measured separately.

- Paper: [arxiv.org/abs/2610.31833](https://arxiv.org/abs/2610.31833)
- Code, evaluation harness and results: [github.com/ppotash/latent-coherence](https://github.com/ppotash/latent-coherence)

## Loading

```python
from datasets import load_dataset

items = load_dataset("REPO_ID", "items", split="synthetic")      # or split="hover"
prompts = load_dataset("REPO_ID", "prompts", split="hover")
```

| config | split | rows |
|---|---|---|
| `items` | `synthetic` | {synthetic_items} |
| `items` | `hover` | {hover_items} |
| `prompts` | `synthetic` | {synthetic_prompts} |
| `prompts` | `hover` | {hover_prompts} |

## Item families

| family | records in context | latent z |
|---|---|---|
| `source` | reports on a claim from 1–3 sources, each with a track record | whether a source is careful or careless |
| `diagnostic` | a fault-diagnosis inspection log | which of three causes is present |
| `causal` | a count table over a confounder, a treatment and an outcome | back-door adjustment for P(y \| do(x)) |
| `hover` | a fictionalized multi-hop claim from HoVer, split into sub-claims reported by sources of varying reliability | truth of the bridge sub-claim; a source's kind |

`source`, `diagnostic` and `hover` come at two `spec_level`s: `full` states every
probability, and `qualitative` gives only verbal descriptions ("usually", "most sources are
careful"). `causal` is full only.

Ground truth: exact enumeration (`source`, `hover`), naive Bayes (`diagnostic`), and
empirical frequencies of the table, which is the whole population (`causal`). Items are
filtered so that targets and latent probabilities lie in [0.1, 0.9] and the latent shifts the
target by at least 0.10–0.15.

## `items` fields

- `id`, `family`, `spec_level`, `context`
- `prompt_prefix`: the exact text before each question (instructions + context)
- `queries`: list of `{qid, qtype, event, truth, given, questions}`. `questions` lists the
  exact question texts as `{framing, paraphrase, text}`: three wordings (`paraphrase` 0–2)
  and, for conditional queries, two framings (`learn`: "Suppose you additionally learn, from a
  completely reliable check…", `assume`: "Assume hypothetically…"). Causal queries use fixed
  wordings (`framing = "custom"`).
- `identities`: list of `{name, kind, lhs, terms}` that coherent answers satisfy:
  - `total_probability`: P(`lhs`) = Σ P(w)·P(c) over `terms = [[w, c], ...]`
  - `normalization`: Σ P(q) = 1 over `terms = [[q], ...]`
- `similar_names`: HoVer items whose invented names look alike (exclude them if you prefer)
- `hover_uid`, `num_hops`, `original_claim`: provenance of HoVer items (null otherwise)
- `meta_json`: generating parameters and other family-specific metadata, as JSON
- `in_study`, `repeated_wordings`: whether the item was selected for the published study, and
  whether it was asked in all three wordings there. In that study, all models but one answered
  only the first 5 selected items of each full-spec synthetic group (in item order), since
  models solve those essentially exactly.

## `prompts` fields

One row per prompt: `row_id`, `item_id`, `family`, `spec_level`, `qid`, `qtype`, `framing`,
`paraphrase`, `system` (system prompt), `prompt` (user message), `truth`, plus `core`, which
marks the prompts used in the published study (the queries the main identities need, in the
`learn` framing), and `similar_names`, `in_study`, `repeated_wordings` as above. Every query
should be its own single-turn call, so no answer is visible to another query.

## Scoring coherence

```python
import math

def logit(p, eps=0.01):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))

def residuals(item, answers):
    """answers: {qid: model probability} for one item (first wording)."""
    out = {}
    for idn in item["identities"]:
        if idn["kind"] != "total_probability":
            continue
        decomp = sum(answers[w] * answers[c] for w, c in idn["terms"])
        out[idn["name"]] = logit(answers[idn["lhs"]]) - logit(decomp)
    return out
```

The repository's `lceval.py analyze` adds the rest: separating persistent incoherence from
rewording noise using the three wordings, bootstrap CIs, accuracy against `truth`, and paired
model comparisons.

## Construction and caveats

Synthetic items are generated by code. For HoVer, Gemini 3.8 Flash decomposed each claim into
logically independent sub-claims and replaced every named entity with an invented name, a
second Gemini pass audited the decomposition, and Claude cross-checked each accepted claim
(849 of 908 passed). Truth values are then sampled from stated parameters, so HoVer items also
have exact ground truth. A decomposition error would affect the ground truth of that item's
target, but not coherence, which compares a model with itself.

## License and attribution

CC BY-SA 4.0. The HoVer-derived items adapt claims from HoVer (Jiang et al., 2020), released
under CC BY-SA 4.0.

## Citation

```bibtex
@misc{potash2026latentcoherence,
  title         = {Latent Coherence: Do Frontier Language Models' Probability Judgments Obey the Law of Total Probability?},
  author        = {Potash, Peter},
  year          = {2026},
  eprint        = {2610.31833},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2610.31833}
}

@inproceedings{jiang2020hover,
  title     = {{HoVer}: A Dataset for Many-Hop Fact Extraction and Claim Verification},
  author    = {Jiang, Yichen and Bordia, Shikha and Zhong, Zheng and Dognin, Charles and Singh, Maneesh and Bansal, Mohit},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
  year      = {2020}
}
```
