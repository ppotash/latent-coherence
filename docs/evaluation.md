# Running the evaluation

The study runs in stages, so you only pay for as many prompts as the statistics need:

```bash
# stage 1: 20 items per (family, spec) group = 140 items, ~1,300 prompts per model
python lceval.py select   --run runs/main --n-per-group 20 --noise-items 20
python lceval.py estimate --run runs/main --models claude-haiku-5-5 claude-sonnet-5-5
python lceval.py batch submit  --run runs/main --model claude-haiku-5-5
python lceval.py batch submit  --run runs/main --model gemini-3-8-flash
python lceval.py batch collect --run runs/main --model claude-haiku-5-5   # later
python lceval.py analyze  --run runs/main --planned-models 4

# only if the pairwise table says you need more items: grow the same run
python lceval.py select   --run runs/main --n-per-group 40 --noise-items 20 --grow
python lceval.py batch submit --run runs/main --model claude-haiku-5-5   # asks only the new prompts
```

Selection is nested: the items chosen at `--n-per-group 20` are always among those chosen at
40, so growing a run never re-asks or discards a paid answer. Stage 1 is sized from the
power analysis. `analyze` then reports, for each pair of models, how many items are needed
to detect a `--min-effect` difference (default 0.2 in mean |log-odds residual|). It uses
only the measured *variance*, not the observed effect. That is the standard internal-pilot
rule, which lets stage-1 items count toward the final analysis.

**Targeting and caps.** Noise-floor items can be added later, from chosen groups only:
`select --grow --noise-items 60 --noise-groups source/qualitative,hover/full` keeps the
existing noise-floor items and adds the rest from those groups. `--cap` on `estimate`,
`run` and `batch submit` limits how many items of a group one model answers. For example,
`--cap synthetic-full=5` keeps 5 items each of the full-spec synthetic groups, where models
compute exactly and score at ceiling. Capped models answer the first items in the run's
order, so models given the same cap answer the same items. When models answered different
item sets, `analyze` compares them on the shared items (`--all-items` disables this).

By default `select` keeps only the prompts the headline identities need (`--queries core`:
P(X), P(z) for each latent state, P(X|z)), in one conditioning framing (`--framings learn`).
That's about 6–9 prompts per item instead of 15–20. `--queries all --framings learn,assume`
adds the secondary analyses: reverse identities, negations, and the learn-vs-assume
framing effect.

## What each step does

**select** samples `--n-per-group` items from every (family, spec level) group and renders
the prompts. Every model in a run then answers exactly the same prompts, from
`runs/<name>/rows.jsonl`. Each query uses its first paraphrase, except on the
`--noise-items` items, where all three paraphrases are asked so the noise floor can be
measured. `--exclude-similar-names` drops HoVer items whose fictional entities have
near-identical names.

**estimate** prints the projected cost per model under three assumptions about output
length. For thinking models, hidden reasoning tokens dominate the cost, so the range is
wide.

**run** queries one model and appends results to `runs/<name>/<model>.jsonl`. Each
prompt is an independent single-turn request; no answer ever appears in another query's
context. The run starts with `--probe` serial calls (default 5), which settle each
provider's API contract and **measure the real cost per call**. It then asks for
confirmation before fanning out with `--concurrency` threads. `--max-cost` stops the run
once that many dollars are spent. Rerunning the same command resumes: answered prompts are
skipped, and errored ones are retried. A response cut off by `--max-tokens` (default 16,000,
reasoning included) is not the model's final answer, so `analyze` excludes it (listing it in
`analysis/truncated.csv`) and the next `run` or `batch submit` asks it again; raise
`--max-tokens` for that retry. Reasoning text is saved when the provider exposes
it (disable with `--no-save-reasoning`).

Each model's reasoning level comes from its `effort` in `config/models.toml`
(override with `--effort`). The nominal level is `medium` for every model: each prompt is a
single judgment over a short context. Vendors whose scales have no medium (GLM, Kimi,
DeepSeek: low/high/max only) run at `high`. Each adapter maps
effort to its provider's own control: Anthropic `effort`, Gemini `thinkingLevel`,
OpenAI-style `reasoning_effort`. Gemini 3 defaults to HIGH thinking when nothing is set,
which is why the adapter always sends a level. Every answer line records the effort used,
and `analyze` lists it per model. Most reasoning models ignore temperature, so runs cannot
be pinned to greedy decoding. Say so in the write-up.

**batch** (Claude, GPT and Gemini models): `batch submit` runs the same 5-call synchronous
probe, spread across groups, then submits the remaining prompts to the provider's batch API
at **half price**:
- Anthropic: Message Batches.
- OpenAI: a Files API upload plus `/v1/batches` over chat completions.
- Gemini: inline `batchGenerateContent`.

It saves the batch ids to `runs/<name>/<model>.batch.json`, after which you can close the
laptop. `batch collect` downloads finished results into the same `<model>.jsonl` format as
`run` (marked `mode: "batch"`, priced at batch rates). Turnaround is usually well under a
day: Anthropic and OpenAI expire unfinished requests after 24 hours, Gemini after 48, and
expired requests are not billed. Running `submit` again retries only prompts that errored
or expired. Request bodies come from the synchronous adapters, after the probe has settled
any parameter the API rejects, so batch and synchronous answers are directly comparable.
Use `batch` for every model it supports. `run` is for providers without a batch path
(Grok, GLM, Kimi, DeepSeek are untested with batch) and for quick checks.

**analyze** writes CSVs to `runs/<name>/analysis/` (`--models a b c` restricts it to some models, so the shared-item set is computed among them; `--out` names the CSV folder) and prints:

| table | question it answers |
|---|---|
| coherence | How far is the direct estimate P̂(X) from the model's own Σ P̂(z)·P̂(X\|z)? Log-odds, item-level bootstrap CIs, by identity category. `renorm` rescales the P̂(z) to sum to 1 first, which separates a normalization failure from a marginalization failure. |
| noise vs systematic | On noise-floor items, how much of that residual is rewording noise (`noise_sd`), and how much persists across rewordings (`systematic_sd`, `systematic_share`)? A noisy but coherent model has `systematic_sd ≈ 0`. **This is the headline incoherence metric.** |
| normalization | Do P̂(z) over a partition (or P̂(X) + P̂(¬X)) sum to 1? |
| framing | Does "suppose you learn z" give a different answer than "assume z"? |
| calibration | Against ground truth, is the direct estimate or the decomposed one closer? |
| pairwise | Paired item-level differences between models, d_z, and the number of items needed for 80% power (Bonferroni over model pairs). Run this after the pilot to size the full study. |

## Models

`config/models.toml` lists each model's provider, exact API model string and per-million
token rates. The rates drive the estimate, the running cost display and `--max-cost`. A
model with `model_id = "TODO"` won't run, and a model without rates needs
`--allow-unpriced`. API keys come from `.env` (see `.env.example`). The `sim-*` models
answer from ground truth offline, so you can exercise the whole pipeline at no cost:

```bash
python lceval.py select --run runs/try --n-per-group 4 --noise-items 8
python lceval.py run --run runs/try --model sim-incoherent --yes
python lceval.py analyze --run runs/try
```

The provider adapters (`lc/providers/`) handle each API's quirks: thinking contracts,
dropping parameters the API rejects (reported at the end of a run as `dropped_params`), and
normalizing token accounting. Check `dropped_params` on the first run of a new model.
