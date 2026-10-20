# Running the evaluation

The study runs in stages, so you only pay for as many prompts as the statistics need:

```bash
# stage 1: 20 items per (family, spec) group, 20 of them asked in all 3 wordings
python lceval.py select   --run runs/main --n-per-group 20 --noise-items 20
# then 40 more repeated items, from the groups where wording matters (full-spec synthetic
# items are calculations, so their wordings agree)
python lceval.py select   --run runs/main --n-per-group 20 --noise-items 60 --grow \
    --noise-groups diagnostic/qualitative,source/qualitative,hover/full,hover/qualitative
python lceval.py estimate --run runs/main --models claude-opus-5-5 --cap synthetic-full=5
python lceval.py batch submit  --run runs/main --model claude-opus-5-5 --cap synthetic-full=5
python lceval.py batch collect --run runs/main --model claude-opus-5-5     # later
python lceval.py analyze  --run runs/main --planned-models 7

# only if the pairwise table says you need more items: grow the same run
python lceval.py select   --run runs/main --n-per-group 40 --noise-items 20 --grow
python lceval.py batch submit --run runs/main --model claude-opus-5-5 --cap synthetic-full=5   # asks only the new prompts
```

The published results used exactly these two `select` commands: 140 items, 60 of them asked in
all three wordings (3 from each full-spec synthetic group, 12–13 from each of the other four).
That is 1,890 prompts per model, or 1,590 with `--cap synthetic-full=5`, which every model
except Haiku used.

Selection is nested: the items chosen at `--n-per-group 20` are always among those chosen at
40, so growing a run never re-asks or discards a paid answer. `analyze` reports, for each pair
of models, how many items would be needed to detect a `--min-effect` difference (default 0.2 in
mean |residual|). It uses only the measured variance, not the observed effect, so stage-1 items
can count toward the final analysis.

**Targeting and caps.**

- **More repeated items, from chosen groups only.** For example,
  `select --grow --noise-items 60 --noise-groups source/qualitative,hover/full` keeps the
  existing ones and adds the rest from those groups.
- **Caps.** `--cap` on `estimate`, `run` and `batch submit` limits how many items of a group
  one model answers. `--cap synthetic-full=5` keeps 5 items of each full-spec synthetic group,
  where models compute exactly and score at ceiling. Capped models answer the first items in
  the run's order, so models given the same cap answer the same items.
- **Shared items.** When models answered different item sets, `analyze` compares them only on
  the items all of them answered (`--all-items` disables this). So one unfinished model
  shrinks the comparison for everyone. Leave it out with `--models` until it's done.

By default `select` keeps only the prompts the headline checks need (`--queries core`: P(X),
P(z) for each latent state, P(X|z)), in one conditioning framing (`--framings learn`). That is
about 6–9 prompts per item instead of 15–20. `--queries all --framings learn,assume` adds the
secondary analyses: reverse identities, negations, and the learn-vs-assume framing effect.

## What each step does

**select** samples `--n-per-group` items from every (family, spec level) group and renders the
prompts. Every model in a run then answers exactly the same prompts, from
`runs/<name>/rows.jsonl`. Each query uses its first wording, except on the `--noise-items`
items, where all three wordings are asked. `--exclude-similar-names` drops HoVer items whose
invented names look alike.

**estimate** prints the projected cost per model under three assumptions about output length.
Hidden reasoning dominates the cost, so the range is wide.

**run** queries one model and appends each answer to `runs/<name>/<model>.jsonl` as soon as it
arrives.

- **Independent calls.** Each prompt is a separate single-turn request, so no answer ever
  appears in another query's context.
- **Probe first.** The run starts with `--probe` serial calls (default 5), spread across item
  groups. They settle each provider's API contract and measure the cost per call. The run then
  prints any parameter the API rejected and asks for confirmation before fanning out with
  `--concurrency` threads. If the *reasoning setting* itself was rejected, it warns that the
  model would run at its default level and always asks, even with `--yes`.
- **Probe projections can be far off.** A few items can make a model reason 10–20× longer than
  average: for Grok, GLM and DeepSeek it was the multi-hop HoVer items. Five probe calls rarely
  catch that, so final costs came out 3–5× above the probe projection. Once a few hundred
  answers are in, run `python scripts/project_cost.py --run runs/main --model <name>` for a
  per-group projection. It makes no API calls.
- **`--max-cost`** stops the run once that many dollars are spent *in the current session*. A
  rerun starts a fresh budget, so lower it by what has already been spent.
- **Stopping and resuming.** Ctrl+C cancels queued calls and saves the in-flight ones.
  Rerunning the same command resumes: answered prompts are skipped, and errored ones are
  retried. A resume must use the same `--effort`; the run refuses to mix reasoning levels in
  one results file.
- **Cut-off answers.** A response cut off by `--max-tokens` (default 16,000, reasoning included)
  is not the model's final answer. `analyze` excludes it and lists it in
  `analysis/truncated.csv`, and the next `run` or `batch submit` asks it again. Raise
  `--max-tokens` for that retry. Heavy reasoners needed 32K (Gemini) or 65K (GLM, DeepSeek).
- **Reasoning text** is saved when the provider exposes it (disable with
  `--no-save-reasoning`).

**Reasoning effort.** Each model's level comes from `effort` in `config/models.toml` (override
with `--effort`). The nominal level is `medium`. Where a vendor has no medium, the nearest
substantial level is used, and the deviations should be stated in any write-up:

| vendor | levels offered | used |
|---|---|---|
| Anthropic, OpenAI, Gemini | include medium | medium |
| xAI (Grok 4.7) | low / medium / high / xhigh | low, because medium was projected at ~$94 |
| Z.ai (GLM 5.3) | low / high / max | high |
| DeepSeek (V4-Pro) | none / high / max | high |
| Moonshot (Kimi K3) | max only | not run |

Each adapter maps effort to its provider's control: Anthropic `effort`, Gemini `thinkingLevel`,
OpenAI-style `reasoning_effort`, and Z.ai and DeepSeek `thinking` + `reasoning_effort`. Gemini 3
defaults to high thinking when nothing is set, which is why the adapter always sends a level.
An API can accept a setting and still ignore it. To check, run
`python scripts/effort_check.py --run runs/main --model <name> --levels low,high` (a few cents):
it asks the same prompts at each level and shows the reasoning tokens and the
provider-reported cost. Most reasoning models ignore temperature, so runs can't be pinned to
greedy decoding.

**batch** (Claude, GPT and Gemini models): `batch submit` runs the same probe, then submits the
remaining prompts to the provider's batch API at **half price**:

- Anthropic: Message Batches.
- OpenAI: a Files API upload plus `/v1/batches` over chat completions.
- Gemini: inline `batchGenerateContent`.

It saves the batch ids to `runs/<name>/<model>.batch.json`, after which you can close the
laptop. `batch collect` downloads finished results into the same `<model>.jsonl` format as
`run`, marked `mode: "batch"` and priced at batch rates. Turnaround is usually well under a
day: Anthropic and OpenAI expire unfinished requests after 24 hours, Gemini after 48, and
expired requests are not billed. Running `submit` again retries only prompts that errored,
expired or were cut off. Request bodies come from the synchronous adapters, so batch and
synchronous answers are directly comparable. Grok, GLM and DeepSeek have no batch path here
and use `run`.

**Reading the answer.** `lc/parse.py` pulls one probability out of each response. Most models
reply with just the number, or explain and end with it, but some (GLM, and Haiku at times) put
the answer first and explain afterwards. When several numbers appear, the parser takes, in
order of preference:

1. a last line that is only a probability;
2. a first line that is only a probability, optionally labelled ("**Answer: 0.9**");
3. the number after "answer is" or "final answer";
4. otherwise the last number in the text.

`analyze` re-reads every saved answer with the current parser, so all models are scored by the
same rules. Its `calls` table shows `multi_number` (answers containing several numbers) and
`reparsed` (answers whose value changed from what was recorded at run time). Each changed
answer is written to `analysis/parse_changes.csv` for spot-checking. For the published
results, all 125 changed Haiku answers and 4 changed GLM answers were checked by hand.

**analyze** writes CSVs to `runs/<name>/analysis/`. `--models a b c` restricts it to some
models, and `--out` names the CSV folder. It prints:

| table | what it shows |
|---|---|
| calls | per model: answered, errors, truncated, unparsed, `multi_number`, `reparsed`, total cost, mean output and reasoning tokens |
| coherence | mean \|residual\| between P̂(X) and the model's own Σ P̂(z)·P̂(X\|z), in log-odds, with bootstrap CIs over items, by identity category. The `_renorm` column first rescales the P̂(z) to sum to 1. Since models' P̂(z) almost always do sum to 1, it rarely differs. |
| noise vs systematic | on the repeated items: `noise_sd` (the README's *rewording noise*), `systematic_sd` with its CI (the *persistent residual*, the headline metric: typical residual left after averaging over wordings, noise-corrected, including any lean shared by all items), `systematic_share`, `mean_resid` (the signed *direction*), and `systematic_spread_sd` (a spread-only variant that ignores a shared lean) |
| normalization | do P̂(z) over all states, or P̂(X) + P̂(not X), sum to 1? |
| framing | does "suppose you learn z" give a different answer than "assume z"? (only with `--framings learn,assume`) |
| calibration | error against ground truth for the direct and the decomposed estimate |
| pairwise | per pair of models: paired item-level differences, d_z, a sign-flip permutation p-value (`p`) and its Bonferroni-corrected version (`p_bonf`, significant below 0.05), and the items needed for 80% power |

Set `--planned-models` to the total number of models in the study, so the Bonferroni
correction stays the same as models are added.

**Extra analyses for the write-up.** `python scripts/paper_extras.py --run runs/main` (no API
calls, same shared items and parser as `analyze`) prints: the item-level correlation of
\|residual\| with error against truth, which estimate (P̂(z), P̂(X\|z), P̂(X)) is wrong on
incoherent items, reasoning tokens vs \|residual\| within each model (including a comparison
across the three wordings of the same item, which removes item difficulty), clipping rates
and robustness to the clip level, the probability scale and renormalization, and every causal
instance with a visible residual alongside its answer texts.

## Models

`config/models.toml` lists each model's provider, exact API model string, reasoning effort and
per-million-token rates. The rates drive the estimate, the running cost display and
`--max-cost`. A model with `model_id = "TODO"` won't run, and a model without rates needs
`--allow-unpriced`.

- **Time-of-day pricing.** DeepSeek's off-peak price is half its peak price. The config holds
  peak rates, so tracked costs and `--max-cost` err high.
- **Keys** come from `.env` (see `.env.example`).
- **Offline testing.** The `sim-*` models answer from ground truth without any API calls, so
  you can exercise the whole pipeline for free:

```bash
python lceval.py select --run runs/try --n-per-group 4 --noise-items 8
python lceval.py run --run runs/try --model sim-incoherent --yes
python lceval.py analyze --run runs/try
```

The provider adapters (`lc/providers/`) handle each API's quirks:

- thinking and reasoning contracts;
- dropping parameters the API rejects, which is reported after the probe and at the end of the
  run as `dropped_params`;
- normalizing token accounting. Some providers count reasoning inside the output tokens and
  some separately; `effort_check.py` shows the raw usage block so you can confirm which.
