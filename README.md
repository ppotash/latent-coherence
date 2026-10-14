# latent-coherence

Do LLM probability judgments obey the law of total probability? This repo is an eval of
P(X) vs Σ P(z)·P(X|z) over latent variables inferred from in-context records.

Each item gives the model a set of records, such as source track records, an inspection log,
an observational count table or multi-hop claims with reports of varying reliability. The model
is then asked, in separate calls, for

- the probability of a target event X directly, and
- the probability of each state z of a latent variable, and of X given each state.

A coherent model's direct answer should equal its own decomposition Σ P̂(z)·P̂(X|z). A causal
family checks the same thing for P(y | do(x)) against the back-door adjustment. Every query
also has an exact ground-truth probability, so incoherence and miscalibration are measured
separately.

The motivation is LawZero's *Scientist AI* proposal (Bengio et al., 2025). It relies on a
predictor whose probabilistic judgments compose correctly, which raises the question of how
close current frontier models come.

## Results

There are 1,590 prompts per model, covering 95 shared items. Each prompt is an independent
single-turn call. Residuals are |direct − decomposed| in log-odds; brackets are 95% bootstrap
CIs over items.

| model | effort | mean residual | systematic SD | rewording noise | error vs truth | reasoning tok/call | cost |
|---|---|---|---|---|---|---|---|
| GPT-6.1 Sol | medium | **0.08** [0.06, 0.11] | 0.08 [0, 0.13] | **0.16** | **0.34** | ~400 | $3.96 |
| Claude Opus 5.5 | medium | 0.12 [0.07, 0.17] | 0.00 [0, 0.14] | 0.34 | 0.38 | ~500 | $17.61 |
| Grok 4.7 | low¹ | 0.21 [0.15, 0.27] | 0.20 [0, 0.30] | 0.51 | 0.47 | ~7,200 | $70.44 |
| Gemini 3.8 Flash | medium | 0.26 [0.19, 0.34] | **0.26 [0.09, 0.36]** | 0.50 | 0.73 | ~4,400 | $14.52 |
| GLM 5.3 | high² | 0.26 [0.20, 0.33] | 0.09 [0, 0.23] | 0.53 | 0.41 | ~6,700 | $53.57 |
| Claude Haiku 5.5 | medium | 0.37 [0.27, 0.49] | **0.32 [0.13, 0.44]** | 0.64 | 0.50 | ~1,100 | $0.62 |

*Systematic SD* is the part of the residual that persists across three rewordings of the same
question, with rewording noise subtracted out. This is the headline incoherence metric: a
model that is noisy but coherent has a systematic SD of about 0. *Error vs truth* is the
direct answer's mean log-odds error against ground truth.

1. Grok ran at low because medium was projected to cost about $94. It still reasoned the most
   of any model.
2. GLM's API offers only low, high and max, so it runs at high, the level nearest to medium.
   It needed a 65K-token limit: at 32K, 22% of its answers on multi-hop items with full
   probabilities were cut off.

What the results show:

- **Two tiers.** GPT and Opus are not distinguishable from each other. Both are significantly
  more coherent than Gemini, GLM and Haiku, and GPT also beats Grok. Within the lower tier,
  the only significant difference is Grok over Haiku. Significance is judged by a paired
  sign-flip permutation test with Bonferroni correction over the 15 model pairs.
- **Systematic incoherence is model-specific.** It is clearly non-zero only for Haiku and
  Gemini, and they err in opposite directions:
  - Haiku's direct answer is *above* its own decomposition (+0.14, and +0.22 on multi-hop
    claims with full probabilities).
  - Gemini's is *below* (−0.11, and −0.32 on qualitative multi-hop claims).

  For the other four models, the residual is mostly rewording noise.
- **More reasoning did not buy coherence.** The two most coherent models reasoned the least
  (~400–500 tokens per call). Grok, GLM and Gemini reasoned 9–18 times as much and landed in
  the lower tier. With six models this is a pattern, not a test.
- **Basic consistency holds everywhere.** Back-door adjustment matches P(y | do(x)) exactly,
  and probabilities over a partition sum to 1 (median deviation ≤ 0.01). The failures are
  specific to reasoning over a latent.
- **Decomposing rarely helps accuracy.** The model's own Σ P̂(z)·P̂(X|z) is about as far from
  the truth as its direct answer.

Methods details, caveats and every command are in [`docs/evaluation.md`](docs/evaluation.md).

## Quick start

```bash
pip install -r requirements.txt
python tests/test_dataset.py && python -m pytest tests/test_eval.py

# try the whole pipeline offline with a simulated model (no API calls)
python lceval.py select --run runs/try --n-per-group 4 --noise-items 8
python lceval.py run --run runs/try --model sim-incoherent --yes
python lceval.py analyze --run runs/try
```

To evaluate a real model, add its API key to `.env` (see `.env.example`) and its entry to
`config/models.toml`. Then:

```bash
python lceval.py select   --run runs/main --n-per-group 20 --noise-items 20
python lceval.py estimate --run runs/main --models claude-opus-5-5 --cap synthetic-full=5
python lceval.py batch submit  --run runs/main --model claude-opus-5-5 --cap synthetic-full=5  # Claude / GPT / Gemini, half price
python lceval.py batch collect --run runs/main --model claude-opus-5-5
python lceval.py run --run runs/main --model grok-4-7 --cap synthetic-full=5 --max-cost 20      # other providers
python lceval.py analyze  --run runs/main --planned-models 6
```

Every run starts with a few synchronous probe calls that measure the real cost per call, and
asks for confirmation before going further. Reasoning-heavy items cost far more than average,
so check progress with `python scripts/project_cost.py --run runs/main --model <name>`, which
projects the cost to finish per item group.

## Dataset

| family | latent / structure | spec levels |
|---|---|---|
| `source` | claim truth X; latent = kind (careful or careless) of one reporting source, inferred from its track record | full, qualitative |
| `diagnostic` | 3-state hidden cause inferred from an inspection log; targets are a future outcome and an unperformed test | full, qualitative |
| `causal` | confounded observational count table (Z → X, Z → Y, X → Y); P(y \| do(x)) vs back-door adjustment | full |
| `hover` | HoVer multi-hop claims, decomposed into sub-claims and fictionalized, with reports from sources of varying reliability; latents = the bridge sub-claim and a source's kind | full, qualitative |

The spec level decides how much the model is told:

- **full:** every probability is stated, so the item is a well-posed calculation.
- **qualitative:** only verbal descriptions are given ("usually", "most sources are
  careful"), so the model must form its own probabilities.

Ground truth always comes from the generating numbers. Coherence needs no ground truth, so it
is valid at both levels.

- **Synthetic items** (`data/synth_items.jsonl`): 1,500 items, 300 per family × spec level,
  generated with exact inference by enumeration.
- **HoVer items** (`data/hover_items.jsonl`): 1,698 items built from 849 claims from the HoVer
  training set (Jiang et al., 2020). Each claim was:
  1. decomposed into sub-claims and fictionalized by Gemini;
  2. checked by a second Gemini pass;
  3. cross-verified by Claude.

  Entity names are invented, so a model can't answer from memory. Items where the invented
  names look alike are flagged and can be excluded with `--exclude-similar-names`.
  `data/hover_processed.jsonl` and `data/crossver_results.jsonl` record how each claim was
  accepted.

Every query has three paraphrases, and conditional queries come in two framings ("suppose you
learn…", "assume…"). By default a run asks only the prompts the headline identities need,
in one framing.

### Ground-truth semantics

- **source / hover:**
  - Each statement is independently true with the stated base rate, and a target claim is
    true iff all its sub-claims are.
  - Each source is careful or careless with the stated prior.
  - Past and current reports are correct with the kind's accuracy, independently given the
    kind.
  - The joint is enumerated exactly; tests check it against rejection sampling.
- **diagnostic:** naive Bayes over a 3-state cause with the table shown.
- **causal:** the count table is the population. The context states that Z is the only
  confounder, so the back-door adjustment identifies P(y | do(x)).

Items are filtered so that targets and latents fall in [0.1, 0.9] and the conditionals differ
across latent states by at least 0.1–0.15. Without that filter, coherence would hold trivially.

### Rebuilding the dataset

```bash
python lcbuild.py synth --n 300 --out data/synth_items.jsonl
python lcbuild.py hover-process --candidates data/hover_candidates.jsonl --dec-model <gemini model> --ver-model <gemini model> --out data/hover_processed.jsonl
python lcbuild.py hover-crossver submit     # Claude Batch API; later:
python lcbuild.py hover-crossver collect
python lcbuild.py hover-build --processed data/hover_processed.jsonl --require-crossver --out data/hover_items.jsonl
```

Set the Gemini thinking level explicitly before a full HoVer run. Gemini 3 defaults to high
thinking, which makes generation several times more expensive.

## Repository layout

```
lceval.py, lcbuild.py   CLIs for evaluation and dataset construction
lc/evaluate.py          item selection, cost estimates, synchronous runs
lc/batch.py             Anthropic / OpenAI / Gemini batch APIs
lc/analyze.py           identities, noise-vs-systematic decomposition, calibration, pairwise tests
lc/parse.py             probability extraction (answer-first and answer-last layouts)
lc/providers/           provider adapters (reasoning controls, token accounting)
lc/engine.py, synth.py  exact-inference worlds and synthetic generators
lc/hover*.py, crossver.py  HoVer decomposition, gold examples, Claude cross-verification
config/models.toml      model names, reasoning effort, prices
scripts/                effort_check.py (does a provider honour reasoning effort?), project_cost.py
docs/evaluation.md      full evaluation guide
```

## License

Code: MIT. Data: CC BY-SA 4.0, because the HoVer items are derived from HoVer, which is
CC BY-SA.

## References

- Bengio et al. (2025). *Superintelligent Agents Pose Catastrophic Risks: Can Scientist AI
  Offer a Safer Path?* arXiv:2502.15657.
- Jiang et al. (2020). *HoVer: A Dataset for Many-Hop Fact Extraction and Claim Verification.*
  Findings of EMNLP.