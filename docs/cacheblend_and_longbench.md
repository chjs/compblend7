# CacheBlend paper analysis + LongBench dataset review (for adding Multi-News)

This note (a) analyzes the CacheBlend paper as it relates to this repo, (b)
reviews the LongBench datasets as candidate CacheBlend workloads, and (c)
explains why **Multi-News** was chosen and how it is wired in
(`benchmarks/multinews/`).

## 1. CacheBlend — what it does and why it matters here

**Paper:** Yao, Li, Liu, Ray, Cheng, Zhang, Du, Lu, Jiang, *"CacheBlend: Fast
Large Language Model Serving for RAG with Cached Knowledge Fusion"* (arXiv
2405.16444; EuroSys '25).

### The problem
In RAG, a prompt is *several text chunks* (retrieved passages) + a query. Prefix
caching only reuses a KV cache when a chunk is the **exact prefix**; retrieved
chunks appear in arbitrary order and positions, so only the first chunk ever hits.
Recomputing every chunk's KV from scratch (full prefill) is what dominates
time-to-first-token (TTFT) for long RAG inputs.

The naive fix — precompute each chunk's KV independently and just concatenate
them — is fast but **wrong**: each chunk's KV was computed in isolation, so it
carries **no cross-chunk attention**. Quality collapses because the model never
"sees" the other chunks while encoding each one.

### The insight (selective KV recompute)
CacheBlend reuses the precomputed per-chunk KV and **recomputes only a small
subset of tokens** to restore the missing cross-attention. The tokens that matter
are the ones whose KV changes most when the true (full) context is present —
**High-KV-Deviation (HKVD)** tokens. Key findings the paper leverages:

- **Deviation is sparse.** Only ~15% of tokens have large KV deviation; recomputing
  just those recovers full-prefill quality. The recommended recompute ratio is
  **~15%** (they sweep 5–20%).
- **Deviation is predictable across layers.** The set of high-deviation tokens at
  one layer predicts the high-deviation set at the next, so you can pick the token
  set at an early layer (this repo uses `check_layer=1`) and reuse that selection
  for the remaining layers → one cheap selection pass, then sparse recompute.
- **Gradual filtering / pipelining.** Selection + recompute is fused into the
  layers and overlapped with KV loading, so the extra compute is largely hidden.

### Results the paper reports
TTFT reduced **2.2–3.3×** and throughput up **2.8–5×** vs full prefill, at
**negligible quality loss**, across Mistral-7B / Yi-34B / Llama-70B.

### How this connects to compblend7
`compblend7` starts from the CacheBlend core (vendored submodule) and asks a
follow-on question: *if the reused KV is also **compressed** (KVzip token
pruning), which tokens should we recompute?* HKVD asks "which tokens **changed**"
(deviation); compression importance asks "which tokens **matter**". The repo's
selectors (`only_hkvd`, `importance_only`, `gated_hkvd`, …) and the
paired-bootstrap harness exist to test whether combining the two signals beats
HKVD alone. Adding Multi-News gives a **second task family (summarization)** to
test that question beyond multi-hop QA (musique), which matters because the
importance/deviation relationship may differ between extract-a-fact QA and
synthesize-everything summarization.

## 2. LongBench datasets — which fit a CacheBlend / blending experiment

CacheBlend's workload requirement is **multi-chunk inputs**: the context must
decompose into several independently-cacheable documents. That maps cleanly onto
LongBench's **multi-document** and **summarization** tasks and poorly onto
single-passage tasks. The CacheBlend paper itself used **2WikiMQA, Musique,
SAMSum, and MultiNews**.

| LongBench task | category | multi-doc? | metric | fit for blending |
|---|---|---|---|---|
| **multi_news** | multi-doc summarization | **yes — native article clusters** | ROUGE-L | **excellent** (chosen) |
| 2wikimqa | multi-hop QA | yes (passages) | F1 | excellent (paper used it) |
| musique | multi-hop QA | yes (passages) | F1 | excellent (already in repo) |
| hotpotqa | multi-hop QA | yes (passages) | F1 | good |
| gov_report | single-doc summarization | no (one long report) | ROUGE-L | weak — one document, must be chopped artificially |
| qmsum | query-based summarization | no (one meeting) | ROUGE-L | weak |
| samsum | dialogue summarization | few-shot examples act as chunks | ROUGE-L | ok (paper used it) |
| multifieldqa | single-doc QA | no | F1 | weak |
| qasper / narrativeqa | single-doc QA | no | F1 | weak |
| passage_retrieval / count | synthetic | yes | acc/EM | narrow |
| trec / triviaqa / lsht | few-shot classification/QA | few-shot chunks | acc/F1 | ok-ish |

**Takeaway:** the multi-document tasks (multi_news, 2wikimqa, musique, hotpotqa)
are the honest fits because their chunks are *real, independent documents*.
Single-document summarization (gov_report, qmsum) would require slicing one
document into pseudo-chunks, which confounds the blending question.

## 3. Why Multi-News specifically

- **Native chunking.** Each example is a *cluster of source news articles* about
  one event, separated in the LongBench `context` by the Multi-News `|||||`
  document separator. Each article → one KV chunk with **no artificial splitting**.
- **Cross-chunk dependency is intrinsic.** A good multi-document summary must fuse
  facts spread across articles — precisely the cross-chunk attention that naive
  concatenation destroys and selective recompute must restore. So it is a strong
  stress test for the blending arms.
- **Different task shape than QA.** Musique rewards recomputing the few tokens that
  pin down a single answer span; summarization rewards broad coverage. Testing both
  guards against a selector that only wins on extractive QA.
- **Paper parity.** Multi-News is one of CacheBlend's own evaluation datasets, and
  its metric (ROUGE-L) and generation budget (512) are defined by LongBench.

## 4. How it is wired in (`benchmarks/multinews/`)

Same schema and machinery as `benchmarks/musique/`; see that directory's README
for the arm definitions. Summarization-specific differences:

- **Input schema** `{ctxs, question, answers}` — `ctxs` = per-article chunks,
  `question` = `""` (LongBench `multi_news` has an empty `input`; the instruction
  lives in the prompt), `answers` = `[reference summary]`.
- **Builder** `build_inputs.py` downloads `THUDM/LongBench` `multi_news` and
  splits each `context` on `|||||` (blank-line fallback). Run on a host with HF
  access (the GPU pod); HF is unreachable from the CI sandbox.
- **Prompt** adapted from LongBench's `multi_news` template into CacheBlend's
  chunked prefix + live-suffix form.
- **Metric** ROUGE-L (`rouge_score`), `max_new_tokens=512`.
- **Smoke fixture** `inputs/multinews_smoke.json` is **synthetic** (clearly
  labeled) for offline pipeline checks — not real Multi-News data.
