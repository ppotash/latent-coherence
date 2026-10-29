#!/usr/bin/env python3
"""Package the dataset for the Hugging Face Hub, loadable with `datasets.load_dataset`.

  python scripts/build_hf_dataset.py --run runs/main                        # build ./hf_dataset
  python scripts/build_hf_dataset.py --run runs/main --push <user>/latent-coherence

Two configs, each with a `synthetic` and a `hover` split:
  items    one row per item: context, queries with ground truth and exact question texts,
           and the identities that coherent answers satisfy
  prompts  one row per exact prompt sent to a model (system + user text), with ground truth

--run marks the items used in the published study (optional). --push needs a write token
(`huggingface-cli login`, or HF_TOKEN in the environment).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lc.evaluate import core_qids                        # noqa: E402
from lc.prompts import SYSTEM                             # noqa: E402
from lc.schema import render_prompts                      # noqa: E402

STUDY_FRAMINGS = {"none", "custom", "learn"}             # framings used in the published run

QUESTION = pa.struct([("framing", pa.string()), ("paraphrase", pa.int32()), ("text", pa.string())])
QUERY = pa.struct([("qid", pa.string()), ("qtype", pa.string()), ("event", pa.string()),
                   ("truth", pa.float64()), ("given", pa.list_(pa.string())),
                   ("questions", pa.list_(QUESTION))])
IDENTITY = pa.struct([("name", pa.string()), ("kind", pa.string()), ("lhs", pa.string()),
                      ("terms", pa.list_(pa.list_(pa.string())))])
STUDY = [("in_study", pa.bool_()), ("repeated_wordings", pa.bool_())]
ITEMS = pa.schema([("id", pa.string()), ("family", pa.string()), ("spec_level", pa.string()),
                   ("context", pa.string()), ("prompt_prefix", pa.string()),
                   ("queries", pa.list_(QUERY)), ("identities", pa.list_(IDENTITY)),
                   ("similar_names", pa.bool_()), ("hover_uid", pa.string()),
                   ("num_hops", pa.int32()), ("original_claim", pa.string()),
                   ("meta_json", pa.string())] + STUDY)
PROMPTS = pa.schema([("row_id", pa.string()), ("item_id", pa.string()), ("family", pa.string()),
                     ("spec_level", pa.string()), ("qid", pa.string()), ("qtype", pa.string()),
                     ("framing", pa.string()), ("paraphrase", pa.int32()), ("system", pa.string()),
                     ("prompt", pa.string()), ("truth", pa.float64()), ("core", pa.bool_()),
                     ("similar_names", pa.bool_())] + STUDY)


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def study_sets(run):
    if run is None:
        return None, None
    run = Path(run)
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    chosen = {d["id"] for d in read_jsonl(run / "items.jsonl")}
    return chosen, set(man["noise_item_ids"])


def convert(items, chosen, noise):
    item_rows, prompt_rows = [], []
    for d in items:
        meta = d.get("meta") or {}
        similar = bool(meta.get("similar_name_pairs"))
        in_study = None if chosen is None else d["id"] in chosen
        repeated = None if noise is None else d["id"] in noise
        rendered = render_prompts(d)
        by_q = {}
        for r in rendered:
            by_q.setdefault(r["qid"], []).append(
                {"framing": r["framing"], "paraphrase": r["paraphrase"], "text": r["question"]})
        prefix = rendered[0]["prefix"] if rendered else ""
        item_rows.append({
            "id": d["id"], "family": d["family"], "spec_level": d["spec_level"],
            "context": d["context"], "prompt_prefix": prefix,
            "queries": [{"qid": q["qid"], "qtype": q["qtype"], "event": q["event"],
                         "truth": float(q["truth"]), "given": list(q.get("given") or []),
                         "questions": by_q.get(q["qid"], [])} for q in d["queries"]],
            # normalization terms are a flat list of qids; wrap each so every identity has
            # the same nested shape: [[w, c], ...] or [[q], ...]
            "identities": [{"name": i["name"], "kind": i["kind"], "lhs": i.get("lhs"),
                            "terms": [list(t) if isinstance(t, list) else [t] for t in i["terms"]]}
                           for i in d["identities"]],
            "similar_names": similar,
            "hover_uid": None if meta.get("hover_uid") is None else str(meta["hover_uid"]),
            "num_hops": meta.get("num_hops"),
            "original_claim": meta.get("original_claim"),
            "meta_json": json.dumps(meta, ensure_ascii=False),
            "in_study": in_study, "repeated_wordings": repeated,
        })
        core = core_qids(d)
        for r in rendered:
            prompt_rows.append({
                "row_id": r["row_id"], "item_id": d["id"], "family": d["family"],
                "spec_level": d["spec_level"], "qid": r["qid"], "qtype": r["qtype"],
                "framing": r["framing"], "paraphrase": r["paraphrase"], "system": SYSTEM,
                "prompt": r["prefix"] + r["question"], "truth": float(r["truth"]),
                "core": r["qid"] in core and r["framing"] in STUDY_FRAMINGS,
                "similar_names": similar, "in_study": in_study, "repeated_wordings": repeated,
            })
    return item_rows, prompt_rows


def write(rows, schema, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="data/synth_items.jsonl")
    ap.add_argument("--hover", default="data/hover_items.jsonl")
    ap.add_argument("--run", default=None, help="run directory whose selection marks `in_study`")
    ap.add_argument("--out", default="hf_dataset")
    ap.add_argument("--push", default=None, metavar="REPO_ID", help="e.g. <user>/latent-coherence")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    chosen, noise = study_sets(a.run)
    counts = {}
    for split, path in (("synthetic", a.synth), ("hover", a.hover)):
        items, prompts = convert(read_jsonl(path), chosen, noise)
        write(items, ITEMS, out / "items" / f"{split}.parquet")
        write(prompts, PROMPTS, out / "prompts" / f"{split}.parquet")
        counts[split] = (len(items), len(prompts))
        print(f"{split}: {len(items)} items, {len(prompts)} prompts")

    card = (Path(__file__).resolve().parent / "hf_dataset_card.md").read_text(encoding="utf-8")
    for split, (n_items, n_prompts) in counts.items():
        card = card.replace(f"{{{split}_items}}", f"{n_items:,}").replace(f"{{{split}_prompts}}", f"{n_prompts:,}")
    card = card.replace("REPO_ID", a.push or "<user>/latent-coherence")
    (out / "README.md").write_text(card, encoding="utf-8")
    print(f"wrote {out}/")

    if a.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.push, repo_type="dataset", private=a.private, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=a.push, repo_type="dataset",
                          commit_message="Upload latent-coherence dataset")
        print(f"pushed to https://huggingface.co/datasets/{a.push}")


if __name__ == "__main__":
    main()
