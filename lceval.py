#!/usr/bin/env python3
"""Run the latent-coherence evaluation.

  python lceval.py select   --run runs/main --n-per-group 20 --noise-items 20
  python lceval.py estimate --run runs/main --models claude-haiku-5-5 gemini-3-8-flash
  python lceval.py batch submit  --run runs/main --model claude-haiku-5-5   (half price; Claude/GPT/Gemini)
  python lceval.py batch collect --run runs/main --model claude-haiku-5-5
  python lceval.py analyze  --run runs/main

Models are defined in config/models.toml; keys come from .env.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("select", help="choose items and render the prompts every model will see")
    s.add_argument("--run", required=True, help="run directory, e.g. runs/pilot")
    s.add_argument("--items", nargs="+", default=["data/synth_items.jsonl", "data/hover_items.jsonl"])
    s.add_argument("--n-per-group", type=int, default=20, help="items per (family, spec level)")
    s.add_argument("--noise-items", type=int, default=20,
                   help="total items asked in all 3 paraphrases (with --grow, existing ones are kept)")
    s.add_argument("--noise-groups", default=None,
                   help="comma list of groups new noise-floor items are drawn from, e.g. "
                        "source/qualitative,hover/full")
    s.add_argument("--framings", default="learn", help="comma list; 'learn,assume' adds the framing comparison")
    s.add_argument("--queries", default="core", choices=["core", "all"],
                   help="core: only prompts the headline identities need (~1/3 of all)")
    s.add_argument("--exclude-similar-names", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--force", action="store_true")
    s.add_argument("--grow", action="store_true",
                   help="add items to an existing run (larger --n-per-group, same other settings); "
                        "answers already collected stay valid")

    s = sp.add_parser("estimate", help="projected cost per model before running")
    s.add_argument("--run", required=True)
    s.add_argument("--models", nargs="+", required=True)
    s.add_argument("--out-tokens", type=int, nargs="+", default=[200, 1000, 3000],
                   help="assumed output tokens per call (answer + reasoning)")
    s.add_argument("--cap", default=None,
                   help="limit items per group for this model, e.g. 'synthetic-full=5' or "
                        "'source/full=5,causal/full=0'; first items in the run's order are kept")

    s = sp.add_parser("run", help="query one model on every prompt of the run")
    s.add_argument("--run", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
                   help="override the model's effort from config/models.toml")
    s.add_argument("--no-thinking", action="store_true", help="ask for reasoning off where the API allows")
    s.add_argument("--max-tokens", type=int, default=16000, help="ceiling on reasoning + answer tokens")
    s.add_argument("--temperature", type=float, default=1.0, help="ignored by most reasoning models")
    s.add_argument("--concurrency", type=int, default=8)
    s.add_argument("--max-cost", type=float, default=None, help="stop once this many USD are spent")
    s.add_argument("--probe", type=int, default=5, help="serial calls before fanning out")
    s.add_argument("--yes", action="store_true", help="don't ask for confirmation after the probe")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--no-save-reasoning", action="store_true")
    s.add_argument("--allow-unpriced", action="store_true")
    s.add_argument("--cap", default=None,
                   help="limit items per group for this model, e.g. 'synthetic-full=5' or "
                        "'source/full=5,causal/full=0'; first items in the run's order are kept")

    s = sp.add_parser("batch", help="any Claude, GPT or Gemini model via its provider's batch API (50% cheaper)")
    s.add_argument("action", choices=["submit", "collect"])
    s.add_argument("--run", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
                   help="override the model's effort from config/models.toml")
    s.add_argument("--no-thinking", action="store_true")
    s.add_argument("--max-tokens", type=int, default=16000)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--probe", type=int, default=5, help="synchronous calls before submitting")
    s.add_argument("--cap", default=None,
                   help="limit items per group for this model, e.g. 'synthetic-full=5' or "
                        "'source/full=5,causal/full=0'; first items in the run's order are kept")
    s.add_argument("--yes", action="store_true")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--no-save-reasoning", action="store_true")

    s = sp.add_parser("analyze", help="coherence, noise floor, calibration, power")
    s.add_argument("--run", required=True)
    s.add_argument("--exclude-similar-names", action="store_true")
    s.add_argument("--min-effect", type=float, default=0.2,
                   help="smallest difference in mean |log-odds residual| worth detecting (power planning)")
    s.add_argument("--planned-models", type=int, default=None,
                   help="number of models you plan to compare (sets the Bonferroni correction)")
    s.add_argument("--all-items", action="store_true",
                   help="summarize each model on all its items, even when models answered different sets")

    a = ap.parse_args()
    if a.cmd == "select":
        from lc.evaluate import select
        select(a.items, a.run, a.n_per_group, a.noise_items, a.framings.split(","), a.seed,
               a.exclude_similar_names, a.force, a.queries, a.grow,
               a.noise_groups.split(",") if a.noise_groups else None)
    elif a.cmd == "estimate":
        from lc.evaluate import estimate
        estimate(a.run, a.models, a.out_tokens, a.cap)
    elif a.cmd == "run":
        from lc.evaluate import run
        run(a.run, a.model, a.effort, not a.no_thinking, a.max_tokens, a.temperature, a.concurrency,
            a.max_cost, a.probe, a.yes, a.limit, not a.no_save_reasoning, a.allow_unpriced, a.cap)
    elif a.cmd == "batch":
        from lc import batch
        if a.action == "submit":
            batch.submit(a.run, a.model, a.effort, not a.no_thinking, a.max_tokens, a.temperature,
                         a.probe, a.yes, a.limit, not a.no_save_reasoning, a.cap)
        else:
            batch.collect(a.run, a.model, not a.no_save_reasoning)
    else:
        from lc.analyze import analyze
        analyze(a.run, a.exclude_similar_names, a.min_effect, a.planned_models, a.all_items)


if __name__ == "__main__":
    main()
