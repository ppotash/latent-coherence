# Latent-coherence eval dataset

Tests whether an LLM's probability judgments satisfy the law of total probability over a
**latent variable inferred from records in context**. The core check is whether P(X) asked
directly equals Σ_z P(z)·P(X | z) built from the model's own answers. A causal family adds
P(y | do(x)) vs the back-door adjustment. Every query has an **exact** ground-truth
probability, so incoherence and miscalibration can be measured separately.

## Families

| family | latent / structure | targets | identities per item |
|---|---|---|---|
| `source` | claim truth X; latent = kind (careful/careless) of one reporting source, inferred from its track record | X | total prob. over latent, reverse total prob. (Bayes direction), 2 normalizations |
| `diagnostic` | 3-state hidden cause inferred from an inspection log | future outcome X, unperformed test Y | 2 total prob. (X and Y over the same latent), 2 normalizations |
| `causal` | confounded observational count table (Z → X, Z → Y, X → Y) | P(y \| do(x)), P(y \| x) | back-door (×2), observational total prob. (×2), 3 normalizations |
| `hover` | HoVer multi-hop claims, decomposed and fictionalized, with reports from sources of varying reliability; latents = bridge sub-claim truth, and kind of a source reporting on it | X = full claim | total prob. over bridge fact (with P(X \| ¬bridge) = 0), over source kind, reverse, normalizations |

`source` and `diagnostic` come in two **spec levels**: `full` (all probabilities stated, so the
item is a well-posed calculation) and `qualitative` (only verbal descriptions like "usually"
or "most sources are careful"). Ground truth is always computed from the generating
numbers. Coherence needs no ground truth, so it is valid at both levels.

Every conditional query is rendered in two **framings**, `learn` ("Suppose you additionally
learn, from a completely reliable check…") and `assume` ("Assume hypothetically…"), with
three question **paraphrases**. Each prompt comes in a numeric form ("respond with a
probability") and a Yes/No form for logprob elicitation.

## Files

```
lcbuild.py              CLI (see below)
lc/engine.py            exact inference for source-reliability worlds (enumeration)
lc/synth.py             source / diagnostic / causal generators
lc/hover.py             HoVer selection, Gemini decompose + verify prompts, item builder
lc/hover_gold.py        hand-written gold decompositions (few-shot examples + offline fixture)
lc/llm.py               async Gemini JSON client with resumable on-disk cache
lc/schema.py            item / query / identity schema, prompt rendering
tests/test_dataset.py   engine vs. rejection sampling, identities exact on truth, rendering
data/synth_items.jsonl  1,500 synthetic items (300 per family × spec level)
data/synth_prompts.jsonl   (not committed) generate with: python lcbuild.py export-prompts --items data/synth_items.jsonl --out data/synth_prompts.jsonl
data/hover_candidates.jsonl 1,500 HoVer train claims (500 each of 2/3/4 hops), pre-filtered
data/sample_hover_items.jsonl 18 HoVer items built from the gold decompositions (preview only)
```

## Running it

```bash
pip install numpy google-genai pydantic
python tests/test_dataset.py

# synthetic (already built; rerun to change sizes/seed)
python lcbuild.py synth --n 300 --out data/synth_items.jsonl

# HoVer hybrid: needs your Gemini key
export GEMINI_API_KEY=...
python lcbuild.py hover-process --candidates data/hover_candidates.jsonl \
    --dec-model <your best Gemini model> --ver-model <a strong Gemini model> \
    --limit 50 --out data/hover_processed.jsonl      # pilot first; drop --limit for the full run
python lcbuild.py hover-build --processed data/hover_processed.jsonl --out data/hover_items.jsonl

python lcbuild.py stats --items data/synth_items.jsonl data/hover_items.jsonl
python lcbuild.py export-prompts --items data/synth_items.jsonl data/hover_items.jsonl --out data/prompts.jsonl
```

The default model names (`gemini-2.5-flash` / `gemini-2.5-pro`) are placeholders, so pass
whatever is current on your account. The Gemini calls are cached in
`data/hover_processed.cache/`, so an interrupted run resumes where it stopped. Expect about
2 calls per claim with roughly 3k input tokens each (the prompt includes 5 few-shot
examples).

After the pilot, read about 30 accepted records in `hover_processed.jsonl` before the full
run. The verifier is a second LLM, not a guarantee. Check three things:

1. the explicit claim equals the conjunction of the sub-claims;
2. no sub-claim entails another (the generator samples their truths independently);
3. each negation is an exact negation.

## Ground-truth semantics (what the numbers assume)

- **source / hover:** each statement is independently true with the stated base rate. The
  target claim is true iff all its sub-claims are true. Each source is careful or careless,
  with the stated prior. Past and current reports are correct with the kind-specific
  accuracy, independently given the kind. Each report asserts either a statement or its
  exact negation. The joint is enumerated exactly; tests confirm agreement with rejection
  sampling.
- **diagnostic:** naive Bayes over a 3-state cause with the shown table.
- **causal:** the count table *is* the population; probabilities are empirical frequencies.
  The context states that Z is the only confounder, so the back-door adjustment identifies
  P(y | do(x)).

All items are filtered so the targets and latents fall in [0.1, 0.9] and the decomposition
matters: the conditionals differ across latent states by at least 0.1–0.15. Without this
filter, coherence would be trivially satisfied near 0 or 1.

## Using it in a harness

For each item, collect the model's answer to every query (one call per prompt row; never
show one query's answer in another's context). Then for each identity:

- **total_probability:** residual = P̂(lhs) − Σ P̂(w)·P̂(c)
- **normalization:** residual = Σ P̂(q) − 1

Report residuals in both probability and log-odds space. Compare them against the noise
floor, estimated from the spread across paraphrases on the noise-floor items. Ground truth
lets you also compute calibration (Brier / log loss) for P̂(X) directly and for the
decomposed estimate Σ P̂(z)·P̂(X|z), which answers "is the decomposed estimate more
accurate?"

## License

Code: MIT (see `LICENSE`). Data in `data/`: CC BY-SA 4.0 (see `DATA_LICENSE`). The HoVer-derived
files adapt claims from HoVer (Jiang et al., 2020), which is distributed under CC BY-SA 4.0. HoVer
decompositions and fictionalizations were generated with Gemini.

## Citation

If you use the HoVer-derived items, please also cite:
Jiang et al. 2020. HoVer: A Dataset for Many-Hop Fact Extraction and Claim Verification. Findings of EMNLP.
