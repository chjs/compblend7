"""Multi-News (LongBench) helpers — summarization analogue of benchmarks/musique/utils.py.

Multi-News is multi-document news summarization: each example is a cluster of
source news articles plus one human-written reference summary. It is a natural
CacheBlend workload because the *documents are already separate* — each article
becomes a reusable KV chunk, and the summary requires cross-chunk attention
(exactly what selective recompute has to repair).

Schema (matches benchmarks/musique/inputs/musique_s.json so the same chunking /
blending machinery is reused verbatim):

    {
      "ctxs":    [{"title": "", "text": <one source news article>}, ...],
      "question": "",            # empty — the summarization instruction lives in
                                 # the benchmark's PREFIX/QUERY prompts, not the data
      "answers": [<reference summary>]
    }

Metric is ROUGE-L (LongBench's metric for multi_news / summarization), not the
token-F1 used for the QA datasets.
"""
import json
import re

from rouge_score import rouge_scorer

# The Multi-News source separates the articles of a cluster with five pipes.
# LongBench's `multi_news` context keeps that separator between source articles.
MULTINEWS_DOC_SEP = "|||||"


def load_dataset(dataset_path):
    print("Loading dataset:", dataset_path)
    with open(dataset_path) as f:
        return json.load(f)


def split_into_docs(context: str, sep: str = MULTINEWS_DOC_SEP) -> list[str]:
    """Split a Multi-News `context` string into its per-article documents.

    Primary split is on the Multi-News document separator (``|||||``, tolerating
    surrounding whitespace). If the separator is absent (some processed dumps
    strip it), fall back to blank-line paragraph blocks so the caller still gets
    more than one chunk. Empty fragments are dropped.
    """
    parts = re.split(r"\s*\|\|\|\|\|\s*", context) if sep == MULTINEWS_DOC_SEP \
        else context.split(sep)
    docs = [p.strip() for p in parts if p.strip()]
    if len(docs) >= 2:
        return docs
    # fallback: no explicit separator — chunk on blank lines
    blocks = [b.strip() for b in re.split(r"\n\s*\n", context) if b.strip()]
    return blocks if blocks else [context.strip()]


def build_summ_prompt(example, query_prompt):
    """Return (doc_prompts, q_prompt) for a summarization example.

    Mirrors musique's build_qa_prompt signature/return so the benchmark wiring is
    identical. Each source article becomes one document chunk; `query_prompt` is
    the trailing summarization instruction (the "live suffix").
    """
    doc_prompts = []
    for ctx in example["ctxs"]:
        title = (ctx.get("title") or "").strip()
        text = ctx["text"]
        doc_prompts.append(f"{title}\n\n{text}\n\n" if title else f"{text}\n\n")
    return doc_prompts, query_prompt


_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def compute_rouge_l(pred: str, gold: str) -> float:
    """ROUGE-L F-measure — LongBench's metric for multi_news."""
    # First generated line only, matching how summaries are read back.
    pred = pred.strip()
    return _ROUGE.score(gold, pred)["rougeL"].fmeasure
