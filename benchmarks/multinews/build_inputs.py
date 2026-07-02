"""Build benchmarks/multinews/inputs/multinews_s.json from LongBench `multi_news`.

Downloads the LongBench `multi_news` test split and converts each example into
the CacheBlend/compblend `{ctxs, question, answers}` schema (the same schema the
CacheBlend authors used for their preprocessed samsum/musique inputs), splitting
each cluster's `context` into one chunk per source news article.

Run on a machine with Hugging Face access (e.g. the GPU pod):

    python benchmarks/multinews/build_inputs.py                 # first 150 examples
    MULTINEWS_N=50 python benchmarks/multinews/build_inputs.py  # first 50
    MULTINEWS_MIN_DOCS=3 python benchmarks/multinews/build_inputs.py  # keep clusters with >=3 articles

Env vars:
    MULTINEWS_N          number of examples to keep (default 150, to match musique_s.json)
    MULTINEWS_MIN_DOCS   drop clusters with fewer than this many source articles (default 2)
    MULTINEWS_OUT        output path (default benchmarks/multinews/inputs/multinews_s.json)
    LONGBENCH_NAME       HF dataset id (default THUDM/LongBench)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import load_dataset as hf_load_dataset

# reuse the same splitter the benchmark/utils use so chunking is identical
import importlib.util

_HERE = Path(__file__).resolve().parent


def _load_utils():
    spec = importlib.util.spec_from_file_location("_mn_utils", _HERE / "utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    utils = _load_utils()
    n = int(os.environ.get("MULTINEWS_N", "150"))
    min_docs = int(os.environ.get("MULTINEWS_MIN_DOCS", "2"))
    lb_name = os.environ.get("LONGBENCH_NAME", "THUDM/LongBench")
    out = Path(os.environ.get("MULTINEWS_OUT", str(_HERE / "inputs" / "multinews_s.json")))

    print(f"[build] load_dataset({lb_name!r}, 'multi_news', split='test')", flush=True)
    ds = hf_load_dataset(lb_name, "multi_news", split="test")

    examples = []
    n_docs_hist: dict[int, int] = {}
    skipped_singleton = 0
    for row in ds:
        docs = utils.split_into_docs(row["context"])
        if len(docs) < min_docs:
            skipped_singleton += 1
            continue
        answers = row["answers"] if isinstance(row["answers"], list) else [row["answers"]]
        examples.append({
            "ctxs": [{"title": "", "text": d} for d in docs],
            # LongBench multi_news has an empty `input`; the summarization
            # instruction is supplied by the benchmark's prompts, not the data.
            "question": (row.get("input") or "").strip(),
            "answers": answers,
        })
        n_docs_hist[len(docs)] = n_docs_hist.get(len(docs), 0) + 1
        if len(examples) >= n:
            break

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(examples, indent=2, ensure_ascii=False))

    total_docs = sum(len(e["ctxs"]) for e in examples)
    print(f"[build] wrote {len(examples)} examples -> {out}", flush=True)
    print(f"[build] skipped {skipped_singleton} clusters with <{min_docs} articles", flush=True)
    if examples:
        print(f"[build] articles/cluster: min={min(len(e['ctxs']) for e in examples)} "
              f"max={max(len(e['ctxs']) for e in examples)} "
              f"mean={total_docs / len(examples):.1f}", flush=True)
        print(f"[build] #articles histogram: "
              f"{dict(sorted(n_docs_hist.items()))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
