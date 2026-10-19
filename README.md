# latent-coherence

Do LLM probability judgments obey the law of total probability? This repo is an eval of
P(X) vs Σ P(z)·P(X|z) over latent variables inferred from in-context records.

Each item gives a model some records (source track records, an inspection log, a count table,
or multi-hop claims with reports of varying reliability). In separate calls, it is asked for
P(X) directly, for P(z) for each state of a hidden variable z, and for P(X|z). If its judgments
are coherent, its direct answer equals its own decomposition Σ P̂(z)·P̂(X|z).

A causal family runs the same check on a cause-and-effect question. P(y | do(x)) is the chance
of outcome y if treatment x were *imposed*, as opposed to merely observed. When a confounder z
is the only thing linking x and y besides the effect itself, that chance equals the
*back-door adjustment* Σ_z P(z)·P(y | x, z). The model's direct answer is checked against the
adjustment built from its own answers; [the causal version](#the-causal-version) below explains
both terms with an example. Every query also has an exact true answer, so incoherence and
inaccuracy are measured separately.

The motivation is LawZero's *Scientist AI* proposal (Bengio et al., 2025), which answers
questions by averaging over hidden explanations of the evidence. That only works if the
underlying probability judgments fit together in this way.

## A worked example

A real item (`source-full-00000`). The model answers are made up for illustration.

```text
Situation: Some sources have reported on whether the Soustiakloun textile plant will close by
the end of the year.

How sources work: Every source is one of two kinds. A careful source's reports are correct 85%
of the time; a careless source's reports are correct 55% of the time. Overall, 65% of sources
are careful. A source's kind never changes, and different reports are correct or incorrect
independently of one another, given each source's kind.

Base rate: Before any reports, claims of this kind are true 30% of the time.

Track records:
| Source                  | Past reports later checked | Of those, correct |
|-------------------------|----------------------------|-------------------|
| Trokeil Staebras        | 0                          | 0                 |
| The Vigoum Gazette      | 3                          | 1                 |
| Braedrouth Triagiabraes | 5                          | 5                 |

Current reports:
- Trokeil Staebras reports that the Soustiakloun textile plant will not close by the end of the year.
- The Vigoum Gazette reports that the Soustiakloun textile plant will not close by the end of the year.
- Braedrouth Triagiabraes reports that the Soustiakloun textile plant will close by the end of the year.
```

The hidden variable z is the Gazette's kind, which the model has to infer from its track record
(1 correct out of 3). Each question is a separate call:

| query | question (after the context above) | truth | model |
|---|---|---|---|
| X | What is the probability that the plant will close by the end of the year? | 0.307 | 0.40 |
| Zc | What is the probability that the Vigoum Gazette is a careful source? | 0.258 | 0.25 |
| Zl | … that it is a careless source? | 0.742 | 0.75 |
| X\|Zc | Suppose you additionally learn, from a completely reliable check, that the Gazette is careful. What is the probability that the plant will close? | 0.114 | 0.12 |
| X\|Zl | (the same, given that it is careless) | 0.374 | 0.38 |

The true answers are coherent: 0.258 × 0.114 + 0.742 × 0.374 = 0.307.

**The residual.** The model's own answers imply 0.25 × 0.12 + 0.75 × 0.38 = 0.315, but it said 0.40
directly. The residual is measured in log-odds (logit), which treats a move from 1% to 2% as
seriously as one from 33% to 50%:

  residual = logit(0.40) − logit(0.315) = −0.405 − (−0.777) = **+0.37**

Near 50%, 0.1 in log-odds is about 2.5 percentage points, so +0.37 is about 9 points, with the
direct answer above the decomposition. The **mean residual** column in the results is the
average of |residual| over items.

**Rewording noise vs persistent residual.** A nonzero residual could just be the model's answers
jittering with the wording. So 20 items are asked in three wordings ("What is the probability that…",
"How likely is it that…", "Estimate the probability that…"):

| wording | P̂(X) | Σ P̂(z)·P̂(X\|z) | residual |
|---|---|---|---|
| 1 | 0.40 | 0.315 | +0.37 |
| 2 | 0.35 | 0.275 | +0.35 |
| 3 | 0.42 | 0.330 | +0.39 |

The answers move but the residual doesn't, so this is a real inconsistency. A merely noisy model
would give something like +0.40, −0.30, −0.05. Over all repeated items, the results report:

- **rewording noise**: the typical spread of the residual across wordings;
- **persistent residual** (the headline score): how large each item's residual is on average
  once you average over the three wordings, minus the part rewording noise alone would leave.
  It is about 0 for a model that is noisy but coherent, and it counts a model that is off by
  the same amount on every item;
- **direction**: the average signed residual, e.g. Haiku's +0.14 means its direct answers lean
  high.

The exact formulas, for each repeated item with residuals r₁, r₂, r₃ under the three wordings:

```text
within              = average over items of  variance(r₁, r₂, r₃)
rewording noise     = √within
persistent residual = √max(0, average over items of mean(r₁, r₂, r₃)²  −  within / 3)
direction           = average over items of mean(r₁, r₂, r₃)
```

Each 3-wording average still carries a third of the noise variance, which is why within ⁄ 3 is
subtracted. The 95% CI comes from a bootstrap over items. In `analyze` output these are
`noise_sd`, `systematic_sd` and `mean_resid`.

**Accuracy.** Error vs truth = |logit(0.40) − logit(0.307)| = 0.41, averaged over items.

**Comparing models.** Models are compared item by item with a sign-flip permutation test. Its
p-value is multiplied by 21, the number of model pairs (Bonferroni), and a difference counts if
the result is below 0.05.

### The causal version

One family of items asks a cause-and-effect question. In one item, a lab records 500
batteries:

| production line | charging | kept >80% capacity | did not |
|---|---|---|---|
| older | slow | 22 | 86 |
| older | fast | 20 | 93 |
| newer | slow | 182 | 51 |
| newer | fast | 31 | 15 |

Technicians chose the charging protocol themselves, and the text states that the production line
is the only factor that influenced both that choice and the outcome.

**Observing vs intervening.** Two different questions can be asked about slow charging:

- *Among batteries that happened to be charged slowly, how many kept their capacity?* Written
  P(y | slow), this is read straight off the table: (22 + 182) ⁄ (108 + 233) = **0.598**.
- *If we took a battery and charged it slowly, how likely is it to keep its capacity?* Written
  P(y | do(slow)). The `do( )` marks an intervention: we *set* the protocol ourselves, rather
  than looking at batteries where the technicians happened to choose it.

These differ because of the production line. Technicians used slow charging mostly on
newer-line batteries (233 of the 341 slow-charged), and newer-line batteries hold their
capacity much better whatever the protocol. So the slow-charged group is full of batteries that
would have done well anyway, and 0.598 flatters slow charging. A variable like this, which
influences both the "treatment" and the outcome, is a **confounder**.

**The back-door adjustment.** To get P(y | do(slow)), compare like with like. Within each
production line the confounding is gone, so take the slow-charging success rate inside each
line, then weight each line by its share of *all* batteries, not of the slow-charged ones:

| line | share of all batteries, P(z) | success rate if slow, P(y \| slow, z) |
|---|---|---|
| older | 221 ⁄ 500 = 0.442 | 22 ⁄ 108 = 0.204 |
| newer | 279 ⁄ 500 = 0.558 | 182 ⁄ 233 = 0.781 |

  P(y | do(slow)) = Σ_z P(z)·P(y | slow, z) = 0.442 × 0.204 + 0.558 × 0.781 = **0.526**

The name comes from causal diagrams. Besides the direct path charging → capacity, there is a
"back door" path charging ← production line → capacity, which creates a misleading
association. Adjusting for the production line closes that back door. The formula is valid
because the production line is stated to be the only confounder, and it isn't itself affected
by the charging protocol.

Each piece is asked separately: P(y | do(slow)) with the wording "Suppose that it is charged
with the slow protocol, regardless of whether it came from the older production line…", plus P(z) and P(y | slow, z). The
residual compares the model's direct interventional answer with the adjustment built from its own
pieces, exactly as for the textile plant. A model that answered the do-question with the raw
0.598 would show a large residual.

## Results

Seven models, the same 95 items (1,590 prompts) each, every prompt an independent call. Lower is
better in the first four columns; brackets are 95% CIs.

| model | effort | mean residual | persistent residual | rewording noise | error vs truth | reasoning tok/call | cost |
|---|---|---|---|---|---|---|---|
| GPT-6.1 Sol | medium | **0.08** [0.06, 0.11] | **0.08 [0.02, 0.13]** | **0.16** | **0.34** | ~400 | $3.96 |
| Claude Opus 5.5 | medium | 0.12 [0.07, 0.17] | 0.00 [0, 0.14] | 0.34 | 0.38 | ~500 | $17.61 |
| DeepSeek V4-Pro (0813) | high³ | 0.15 [0.10, 0.21] | **0.27 [0.10, 0.40]** | 0.30 | 0.37 | ~16,900 | ~$53⁴ |
| Grok 4.7 | low¹ | 0.21 [0.15, 0.27] | 0.21 [0, 0.32] | 0.51 | 0.47 | ~7,200 | $70.44 |
| Gemini 3.8 Flash | medium | 0.26 [0.19, 0.34] | **0.28 [0.10, 0.40]** | 0.50 | 0.73 | ~4,400 | $14.52 |
| GLM 5.3 | high² | 0.26 [0.20, 0.33] | 0.11 [0, 0.25] | 0.53 | 0.41 | ~6,700 | $53.57 |
| Claude Haiku 5.5 | medium | 0.37 [0.27, 0.49] | **0.35 [0.15, 0.48]** | 0.64 | 0.50 | ~1,100 | $0.62 |

In the persistent residual column, bold means the CI excludes 0; in the other columns it marks
the best value.

1. Medium was projected to cost about $94. Even at low, Grok reasoned more than every model
   except DeepSeek.
2. GLM has only low/high/max. It needed a 65K-token limit: at 32K, 22% of its answers on the
   hardest items were cut off.
3. DeepSeek has only none/high/max.
4. At DeepSeek's off-peak rates, half the tracked peak cost ($106.87).

**Mean vs persistent residual.** The mean residual measures how inconsistent a *single* answer
is: wording noise and real inconsistency together. The persistent residual keeps only the part
that survives re-asking. GLM and Gemini show the difference:

| | mean residual | persistent residual | rewording noise |
|---|---|---|---|
| GLM 5.3 | 0.26 | 0.11 [0, 0.25] | 0.53 |
| Gemini 3.8 Flash | 0.26 | **0.28 [0.10, 0.40]** | 0.50 |

A single answer from either model is equally inconsistent, but for different reasons. GLM's
residuals scatter around zero as the wording changes: its answers are unstable, and averaging
several samples would remove most of the inconsistency. Gemini's residuals stay put whatever
the wording, so its judgments themselves don't fit together, and re-asking won't fix that. For
a system like Scientist AI, the persistent kind is the more serious problem.

Compare each of the two numbers across models rather than against each other. The persistent
residual is a root-mean-square over the 20 repeated items, so it weights large residuals more
heavily than the mean residual, which averages sizes over all 95 items. That is why DeepSeek's
persistent residual (0.27) exceeds its mean residual (0.15): it is exact on most items (median
residual 0.013) but consistently off on a few.

**Findings**

- **GPT and Opus lead.** They aren't significantly different from each other, and both beat
  Gemini, GLM and Haiku. GPT also beats Grok. DeepSeek sits between the groups: it can't be
  separated from either, but it beats Haiku. Among the rest, only Grok vs Haiku is significant.
- **Persistent inconsistency is clearest in Haiku, Gemini and DeepSeek** (0.27–0.35), and they
  lean in different directions. Haiku and DeepSeek lean high: their direct answers are above
  their decompositions (+0.14 and +0.08, up to +0.22 and +0.19 on HoVer items). Gemini leans
  low (−0.11, and −0.32 on qualitative HoVer items). DeepSeek's answers barely change with
  rewording, but 44% of its residual is persistent, the highest share of any model. GPT's
  persistent residual also excludes 0, but it is small (0.08, about 2 points). For Opus, Grok
  and GLM it can't be distinguished from 0, so their residual is mostly rewording noise.
- **No clear link between reasoning length and coherence.** The two most coherent models
  reasoned least (~400–500 tokens). Grok, GLM and Gemini reasoned 9–18× more and landed lower,
  while DeepSeek reasoned most (~16,900) and came third.
- **Simpler checks pass.** P̂(careful) + P̂(careless) and similar pairs sum to 1 to within about
  0.02. Causal answers match the models' own back-door adjustment, except for one item each for
  GLM and DeepSeek.
- **Decomposing rarely helps accuracy.** Σ P̂(z)·P̂(X|z) is about as far from the truth as the
  direct answer.

Methods, caveats and every command are in [`docs/evaluation.md`](docs/evaluation.md).

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
python lceval.py analyze  --run runs/main --planned-models 7
```

Runs start with a few test calls to measure cost, then ask for confirmation. Some items take far
longer reasoning than others, so check progress with
`python scripts/project_cost.py --run runs/main --model <name>`, which projects the cost to
finish per item group.

## Dataset

| family | records | hidden variable z |
|---|---|---|
| `source` | reports plus each source's track record | whether a source is careful or careless |
| `diagnostic` | an inspection log | which of 3 possible causes is behind it |
| `causal` | a count table with one confounder | the check is the back-door adjustment |
| `hover` | multi-hop claims from HoVer, with reports from sources of varying reliability | whether a linking sub-claim is true; a source's kind |

Items are either **full**, with every probability stated, or **qualitative**, with only words
like "usually", so the model must form its own numbers. Coherence compares the model with
itself, so it can be measured at both levels.

- **Synthetic** (`data/synth_items.jsonl`): 1,500 items, 300 per family × level.
- **HoVer** (`data/hover_items.jsonl`): 1,698 items from 849 claims in the HoVer training set
  (Jiang et al., 2020). Gemini split each claim into sub-claims and replaced every real name
  with an invented one, so models can't answer from memory. A second Gemini pass checked the
  result and Claude cross-checked it. `data/hover_processed.jsonl` and
  `data/crossver_results.jsonl` record how each claim was accepted. Items with look-alike
  invented names are flagged; exclude them with `--exclude-similar-names`.

Items are filtered so that answers fall in [0.1, 0.9] and z shifts them by at least 0.1–0.15.
Otherwise coherence would be trivially easy.

**How the true answers are defined.**

- **source / hover:** statements are independently true at the stated base rate, and a
  multi-hop claim is true only if all its sub-claims are. Sources are careful or careless with
  the stated probability, and each report is correct with its kind's accuracy, independently
  given the kind. Probabilities are computed exactly by summing over all cases; tests check
  them against simulation.
- **diagnostic:** each cause produces each observation independently, with the probabilities
  shown.
- **causal:** the count table is the whole population and the stated confounder is the only
  one, so the back-door adjustment gives the exact P(y | do(x)).

### Rebuilding the dataset

```bash
python lcbuild.py synth --n 300 --out data/synth_items.jsonl
python lcbuild.py hover-process --candidates data/hover_candidates.jsonl --dec-model <gemini model> --ver-model <gemini model> --out data/hover_processed.jsonl
python lcbuild.py hover-crossver submit     # Claude Batch API; later:
python lcbuild.py hover-crossver collect
python lcbuild.py hover-build --processed data/hover_processed.jsonl --require-crossver --out data/hover_items.jsonl
```

Set Gemini's thinking level explicitly before a full HoVer run. Gemini 3 defaults to high,
which makes generation several times more expensive.

## Repository layout

```
lceval.py, lcbuild.py   CLIs for evaluation and dataset construction
lc/evaluate.py          item selection, cost estimates, synchronous runs
lc/batch.py             Anthropic / OpenAI / Gemini batch APIs
lc/analyze.py           residuals, rewording noise vs persistent residual, accuracy, pairwise tests
lc/parse.py             extracting the probability from each answer
lc/providers/           provider adapters (reasoning settings, token accounting)
lc/engine.py, synth.py  exact true answers and synthetic generators
lc/hover*.py, crossver.py  HoVer item construction and Claude cross-verification
config/models.toml      model names, reasoning effort, prices
scripts/                effort_check.py (does a provider honour the effort setting?), project_cost.py
docs/evaluation.md      full evaluation guide
```

## License

Code: MIT. Data: CC BY-SA 4.0, since the HoVer items derive from HoVer (CC BY-SA).

## References

- Bengio et al. (2025). *Superintelligent Agents Pose Catastrophic Risks: Can Scientist AI
  Offer a Safer Path?* arXiv:2502.15657.
- Jiang et al. (2020). *HoVer: A Dataset for Many-Hop Fact Extraction and Claim Verification.*
  Findings of EMNLP.