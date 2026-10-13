"""Run directory layout and the three pre-analysis steps: select, estimate, run.

runs/<name>/
  manifest.json     selection parameters and counts (written by `select`)
  items.jsonl       the selected items (self-contained copy, for analysis)
  rows.jsonl        one prompt per line; identical for every model in the run
  <model>.jsonl     one answer per line (append-only; a rerun resumes)

Every query is an independent, single-turn request. No answer is ever shown in the
context of another query.
"""
from __future__ import annotations

import collections
import hashlib
import datetime as dt
import json
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

from .parse import parse_probability
from .prompts import prompt_for_row, SYSTEM
from .providers.base import ProviderError, Usage
from .providers.registry import load_models, make_client
from .schema import read_jsonl, render_prompts, write_jsonl


# ----------------------------------------------------------------------------- select

# identities that carry the headline question: does P(X) equal sum_z P(z) P(X|z), with z a
# latent inferred from the records (plus the do() vs back-door check for the causal family)
CORE_IDENTITIES = {"tp_latent", "tp_latent_X", "tp_bridge", "tp_source", "backdoor_x1", "backdoor_x0"}


def core_qids(item: dict) -> set[str]:
    """Queries needed by the core identities; normalization checks whose terms are all
    included come along for free in the analysis."""
    need = set()
    for idn in item["identities"]:
        if idn["name"] in CORE_IDENTITIES:
            need.add(idn["lhs"])
            for w, c in idn["terms"]:
                need.update((w, c))
    return need


def _group_order(group: list[dict], seed: int, key: tuple) -> list[dict]:
    """A fixed random order per group. Taking the first n gives nested selections: the items
    chosen at n=20 are always among those chosen at n=40, so a run can grow without
    discarding answers already paid for."""
    g = sorted(group, key=lambda d: d["id"])
    h = int(hashlib.sha256(f"{seed}|{key[0]}|{key[1]}".encode()).hexdigest()[:12], 16)
    return [g[i] for i in np.random.default_rng(h).permutation(len(g))]


def select(item_paths: list[str], out_dir: str, n_per_group: int, noise_items: int,
           framings: list[str], seed: int, exclude_similar_names: bool, force: bool,
           queries: str = "core", grow: bool = False, noise_groups: list[str] | None = None):
    out = Path(out_dir)
    old = None
    if (out / "rows.jsonl").exists():
        if grow:
            old = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            for k, v in {"item_files": item_paths, "framings": framings, "queries": queries,
                         "seed": seed, "exclude_similar_names": exclude_similar_names}.items():
                if old.get(k) != v:
                    raise SystemExit(f"--grow must keep the original selection settings; {k} was "
                                     f"{old.get(k)!r}, now {v!r}")
            if n_per_group < old["n_per_group"]:
                raise SystemExit(f"--grow needs --n-per-group >= {old['n_per_group']}")
        elif not force:
            raise SystemExit(f"{out}/rows.jsonl exists. Every model in a run must see the same rows; "
                             f"use --grow to add items, a new run name, or --force.")
    items = [d for p in item_paths for d in read_jsonl(p)]
    if exclude_similar_names:
        items = [d for d in items if not d.get("meta", {}).get("similar_name_pairs")]
    groups = collections.defaultdict(list)
    for d in items:
        groups[(d["family"], d["spec_level"])].append(d)
    chosen = []
    for key in sorted(groups):
        chosen += _group_order(groups[key], seed, key)[:n_per_group]
    by_group = collections.defaultdict(list)
    for d in chosen:
        by_group[(d["family"], d["spec_level"])].append(d["id"])
    # noise-floor items: earlier choices are kept; new ones are added round-robin over the
    # eligible groups (all, or --noise-groups), earliest items in each group's order first
    noise = set(old["noise_item_ids"]) if old is not None else set()
    keys = sorted(by_group)
    if noise_groups:
        unknown = set(noise_groups) - {f"{k[0]}/{k[1]}" for k in keys}
        if unknown:
            raise SystemExit(f"unknown --noise-groups {sorted(unknown)}; groups are "
                             f"{[f'{k[0]}/{k[1]}' for k in keys]}")
        keys = [k for k in keys if f"{k[0]}/{k[1]}" in noise_groups]
    target = min(noise_items, len(noise) + sum(len(by_group[k]) for k in keys))
    depth = 0
    while len(noise) < target and depth < max((len(by_group[k]) for k in keys), default=0):
        for k in keys:
            if depth < len(by_group[k]) and len(noise) < target:
                noise.add(by_group[k][depth])
        depth += 1
    rows = []
    for d in chosen:
        keep = core_qids(d) if queries == "core" else None
        for r in render_prompts(d):
            if r["framing"] not in ("none", "custom") and r["framing"] not in framings:
                continue
            if keep is not None and r["qid"] not in keep:
                continue
            if r["paraphrase"] > 0 and d["id"] not in noise:
                continue
            r["noise_floor_item"] = d["id"] in noise
            rows.append(r)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "rows.jsonl", rows)
    write_jsonl(out / "items.jsonl", chosen)
    manifest = {
        "created": (old or {}).get("created") or dt.datetime.now().isoformat(timespec="seconds"),
        "updated": dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "item_files": item_paths, "n_per_group": n_per_group, "noise_items": len(noise),
        "noise_item_ids": sorted(noise),
        "framings": framings, "queries": queries, "seed": seed,
        "exclude_similar_names": exclude_similar_names,
        "system_prompt": SYSTEM,
        "n_items": len(chosen), "n_rows": len(rows),
        "groups": {f"{k[0]}/{k[1]}": len(v) for k, v in sorted(by_group.items())},
        "history": (old or {}).get("history", []) + [{"n_per_group": n_per_group, "noise_items": len(noise),
                                                      "noise_groups": noise_groups, "n_rows": len(rows)}],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    added = f" ({len(rows) - old['n_rows']} new)" if old else ""
    print(f"selected {len(chosen)} items ({len(noise)} noise-floor) -> {len(rows)} prompts{added} in {out}")
    for k, v in manifest["groups"].items():
        print(f"   {k}: {v} items")


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip() or None
    except Exception:
        return None


# ----------------------------------------------------------------------------- shared

FULL_SYNTHETIC = ["source/full", "diagnostic/full", "causal/full"]


def split_probe(todo: list[dict], n: int, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Pick n probe prompts spread across groups and items (round-robin over groups, random
    item within each), so the measured cost per call is representative; return (probe, rest)."""
    if n <= 0 or not todo:
        return [], todo
    rng = np.random.default_rng(seed)
    by_group = collections.defaultdict(list)
    for i, r in enumerate(todo):
        by_group[f"{r['family']}/{r['spec_level']}"].append(i)
    groups = sorted(by_group)
    rng.shuffle(groups)
    picked, used_items = [], set()
    while len(picked) < min(n, len(todo)):
        progressed = False
        for g in groups:
            cands = [i for i in by_group[g] if i not in picked and todo[i]["item_id"] not in used_items]
            cands = cands or [i for i in by_group[g] if i not in picked]
            if cands and len(picked) < n:
                j = cands[int(rng.integers(len(cands)))]
                picked.append(j)
                used_items.add(todo[j]["item_id"])
                progressed = True
        if not progressed:
            break
    chosen = set(picked)
    return [todo[i] for i in picked], [r for i, r in enumerate(todo) if i not in chosen]


def parse_caps(spec: str | None) -> dict[str, int]:
    """'source/full=5,diagnostic/full=5' -> {group: k}. 'synthetic-full=5' expands to the three
    full-spec synthetic groups."""
    caps = {}
    for part in (spec or "").split(","):
        if not part.strip():
            continue
        g, k = part.split("=")
        for grp in (FULL_SYNTHETIC if g.strip() == "synthetic-full" else [g.strip()]):
            caps[grp] = int(k)
    return caps


def apply_caps(rows: list[dict], caps: dict[str, int]) -> list[dict]:
    """Keep at most caps[group] items of each capped group: the first ones in the run's nested
    order, so every model given the same caps answers the same items."""
    if not caps:
        return rows
    kept, seen = set(), collections.defaultdict(list)
    for r in rows:
        g = f"{r['family']}/{r['spec_level']}"
        if r["item_id"] not in seen[g]:
            seen[g].append(r["item_id"])
        if g not in caps or seen[g].index(r["item_id"]) < caps[g]:
            kept.add(r["item_id"])
    return [r for r in rows if r["item_id"] in kept]


def _results_path(run_dir: Path, model: str) -> Path:
    return run_dir / f"{model}.jsonl"


TRUNCATED_STOPS = ("max_tokens", "length")     # Anthropic/Gemini, OpenAI-style


def is_truncated(rec: dict) -> bool:
    """The response was cut off by the token ceiling, so its text is not the model's final
    answer (any number parsed from it may be an intermediate value)."""
    return str(rec.get("stop_reason") or "").lower() in TRUNCATED_STOPS


def is_valid(rec: dict) -> bool:
    return rec.get("error") is None and not is_truncated(rec)


def _done_ids(path: Path) -> set[str]:
    """A row counts as done once the API returned a complete response (even if it did not
    parse): re-asking would cost money and change the sample. Rows that errored or were cut
    off by the token ceiling are retried (raise --max-tokens for the latter)."""
    if not path.exists():
        return set()
    return {r["row_id"] for r in read_jsonl(path) if is_valid(r)}


def check_effort(path: Path, effort: str):
    """Refuse to mix reasoning levels in one model's results: a resumed run without the
    original --effort would otherwise silently answer the rest at the config default."""
    if not path.exists():
        return
    prev = {r.get("effort") for r in read_jsonl(path) if r.get("error") is None}
    other = sorted(str(e) for e in prev if e != effort)
    if other:
        raise SystemExit(f"{path} already has answers at effort {', '.join(other)}, and this run would use "
                         f"{effort}. Pass --effort {other[0]} to continue that run, or move the file out of "
                         f"the run folder to start over at {effort}.")


def _rates(cfg: dict) -> dict | None:
    r = cfg.get("rates")
    return r if r and "input" in r and "output" in r else None


# ----------------------------------------------------------------------------- estimate

def estimate(run_dir: str, model_names: list[str], out_tokens: list[int], caps: str | None = None):
    run = Path(run_dir)
    rows = apply_caps(read_jsonl(run / "rows.jsonl"), parse_caps(caps))
    models = load_models()
    in_tok = np.mean([len(SYSTEM + r["prefix"] + r["question"]) for r in rows]) / 3.8
    print(f"{len(rows)} prompts per model, ~{in_tok:.0f} input tokens each "
          f"(below Anthropic's 1,024-token cache minimum, so no caching discount assumed)\n")
    hdr = f"{'model':24s} {'remaining':>9s}  " + "  ".join(f"{'@' + str(o) + ' out':>10s}" for o in out_tokens)
    print(hdr)
    for name in model_names:
        cfg = models[name]
        n = len(rows) - len(_done_ids(_results_path(run, name)) & {r["row_id"] for r in rows})
        rates = _rates(cfg)
        if not rates:
            cells = "  ".join(f"{'no rates':>10s}" for _ in out_tokens)
        else:
            cells = "  ".join(f"${n * (in_tok * rates['input'] + o * rates['output']) / 1e6:>9.2f}"
                              for o in out_tokens)
        note = ("  (sync; `batch` is half this)"
                if rates and cfg["provider"] in ("anthropic", "openai", "gemini") else "")
        print(f"{name:24s} {n:>9d}  {cells}  [effort {cfg.get('effort', 'medium')}]{note}")
    print("\nOutput tokens include hidden reasoning, which dominates cost for thinking models. "
          "`run` measures the real figure on its first calls before committing.")


# ----------------------------------------------------------------------------- run

def make_record(row: dict, model_name: str, effort: str, thinking: bool, c, rates: dict | None,
                save_reasoning: bool, error: str | None = None, mode: str = "sync",
                price_factor: float = 1.0) -> dict:
    """One answer line in runs/<run>/<model>.jsonl (same shape for sync and batch)."""
    rec = {"row_id": row["row_id"], "item_id": row["item_id"], "qid": row["qid"],
           "framing": row["framing"], "paraphrase": row["paraphrase"], "model": model_name,
           "effort": effort, "thinking": thinking, "mode": mode, "ts": time.time()}
    if error is not None or c is None:
        rec["error"] = error or "no completion"
        return rec
    prob, flag = parse_probability(c.text)
    rec.update(error=None, text=c.text, prob=prob, parse_flag=flag, model_id=c.model_id,
               stop_reason=c.stop_reason, latency_s=round(c.latency_s, 3), attempts=c.attempts,
               usage=c.to_dict()["usage"],
               cost=c.usage.cost(rates) * price_factor if rates else None)
    if save_reasoning and c.reasoning_text:
        rec["reasoning"] = c.reasoning_text
    return rec


def run(run_dir: str, model_name: str, effort: str, thinking: bool, max_tokens: int,
        temperature: float, concurrency: int, max_cost: float | None, probe: int,
        yes: bool, limit: int, save_reasoning: bool, allow_unpriced: bool, caps: str | None = None):
    run = Path(run_dir)
    all_rows = read_jsonl(run / "rows.jsonl")
    rows = apply_caps(all_rows, parse_caps(caps))
    if len(rows) < len(all_rows):
        print(f"--cap: {len(rows)} of {len(all_rows)} prompts in scope for this model")
    models = load_models()
    if model_name not in models:
        raise SystemExit(f"unknown model {model_name!r}; see config/models.toml")
    cfg = models[model_name]
    effort = effort or cfg.get("effort") or "medium"
    rates = _rates(cfg)
    if rates is None and not allow_unpriced:
        raise SystemExit(f"{model_name} has no rates in config/models.toml, so cost cannot be tracked. "
                         f"Add them, or pass --allow-unpriced.")
    out_path = _results_path(run, model_name)
    check_effort(out_path, effort)
    done = _done_ids(out_path)
    todo = [r for r in rows if r["row_id"] not in done]
    if limit:
        todo = todo[:limit]
    if not todo:
        print(f"{model_name}: all {len(rows)} prompts already answered.")
        return
    client = make_client(cfg, effort, rows=rows)
    lock = threading.Lock()
    spent = [0.0]
    usage_total = [Usage()]

    def call(row) -> dict:
        prompt = prompt_for_row(row)
        try:
            c = client.complete(prompt, max_tokens=max_tokens, temperature=temperature,
                                thinking=thinking, effort=effort)
        except (ProviderError, Exception) as e:          # recorded; retried on the next run
            return make_record(row, model_name, effort, thinking, None, rates, save_reasoning,
                               error=f"{type(e).__name__}: {str(e)[:400]}")
        rec = make_record(row, model_name, effort, thinking, c, rates, save_reasoning)
        with lock:
            spent[0] += rec["cost"] or 0.0
            usage_total[0] = usage_total[0] + c.usage
        return rec

    def write(rec):
        with lock, out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 1. probe: a few serial calls settle the API contract and measure real cost per call
    n_probe = min(probe, len(todo))
    print(f"{model_name}: {len(todo)} prompts to run (effort={effort}, thinking={thinking}). "
          f"Probing with {n_probe} calls...")
    probe_rows, rest = split_probe(todo, n_probe)
    probe_recs = [call(r) for r in probe_rows]
    for r in probe_recs:
        write(r)
    ok = [r for r in probe_recs if r.get("error") is None]
    if not ok:
        raise SystemExit("every probe call failed:\n  " + "\n  ".join(r["error"] for r in probe_recs))
    u = usage_total[0]
    per_call = spent[0] / len(ok)
    print(f"  probe: {len(ok)}/{n_probe} ok | avg input {u.input_tokens / len(ok):.0f} tok, "
          f"output {u.output_tokens / len(ok):.0f} tok (reasoning {u.reasoning_tokens / len(ok):.0f}) | "
          f"parsed {sum(r['prob'] is not None for r in ok)}/{len(ok)}"
          + (f" | {sum(map(is_truncated, ok))} cut off at --max-tokens {max_tokens}"
             if any(map(is_truncated, ok)) else ""))
    for r in ok[:3]:
        print(f"    {r['qid']:>10s} -> {r['text'][:60]!r}  (parsed {r['prob']})")

    if rates:
        print(f"  ~${per_call:.4f}/call -> ~${per_call * len(rest):.2f} projected for the remaining {len(rest)}"
              + (f" (stops at --max-cost ${max_cost:.2f})" if max_cost else ""))
    dropped = list(getattr(client, "dropped_params", None) or [])
    lost_effort = [p for p in dropped if p in ("reasoning_effort", "thinking", "thinkingConfig", "effort")]
    if dropped:
        print(f"  note: the API rejected and the adapter dropped: {dropped}")
    if lost_effort:
        # The model now runs at its own default reasoning level (often the highest), which
        # breaks comparability with the other models -- never fan out on that silently.
        print(f"  WARNING: the reasoning setting ({', '.join(lost_effort)}) was rejected, so the remaining "
              f"calls would run at the model's DEFAULT effort, not {effort}.")
        yes = False
    if rest and not yes:
        if input("  Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Stopped. Probe answers are saved; rerun to continue.")
            return

    # 2. fan out. A timed wait keeps Ctrl+C responsive (an untimed wait can't be interrupted
    # on Windows); on Ctrl+C, queued calls are cancelled and in-flight ones are saved.
    n_ok, n_err, t0 = len(ok), len(probe_recs) - len(ok), time.time()
    stopped, n_done = False, 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(call, r) for r in rest]

        def handle(f):
            nonlocal n_ok, n_err, n_done, stopped
            if f.cancelled():
                return
            rec = f.result()
            write(rec)
            n_done += 1
            n_ok += rec.get("error") is None
            n_err += rec.get("error") is not None
            if n_done % 50 == 0 or n_done == len(rest):
                print(f"  {n_done}/{len(rest)} done | errors {n_err} | spent ${spent[0]:.2f} | "
                      f"{(time.time() - t0) / n_done:.2f}s/call")
            if max_cost and spent[0] > max_cost and not stopped:
                stopped = True
                for g in futs:
                    g.cancel()
                print(f"  --max-cost ${max_cost:.2f} reached; cancelling the rest (in-flight calls finish).")

        pending = set(futs)
        try:
            while pending:
                finished, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for f in finished:
                    handle(f)
        except KeyboardInterrupt:
            for g in futs:
                g.cancel()
            print("\n  Ctrl+C: cancelled the queued calls; saving the in-flight ones "
                  "(press Ctrl+C again to abandon them)...")
            for f in pending:
                if not f.cancelled():
                    handle(f)
            print("  stopped. Rerun the same command (same --effort and --cap) to continue.")
    close = getattr(client, "close", None)
    if close:
        close()
    dropped = getattr(client, "dropped_params", None)
    print(f"{model_name}: {n_ok} answered, {n_err} errors, ${spent[0]:.2f} this session -> {out_path}"
          + (f"\n  note: the API rejected and the adapter dropped: {dropped}" if dropped else ""))
    if n_err:
        print("  rerun the same command (same --effort and --cap) to retry the errored prompts.")
