# Multi-News (LongBench) benchmark

A **multi-document summarization** workload for the same compressed-KV-blending
machinery used by `benchmarks/musique/`. Multi-News is one of the datasets in the
CacheBlend paper; it is a natural fit because each cluster's **source articles are
already separate documents** — every article becomes a reusable KV chunk with no
artificial chunking, and a faithful summary needs cross-article attention (the
exact signal selective recompute must repair).

The only differences from the musique benchmark are the **dataset** (news
clusters), the **prompt** (one-page summary), the **generation length**
(LongBench `multi_news` `max_gen` = 512), and the **metric** (ROUGE-L instead of
token-F1). The compression model, blending arms, kvzip×recompute grid, and
paired-bootstrap analysis are identical, so results are directly comparable.

## Files

| file | role |
|---|---|
| `build_inputs.py` | download LongBench `multi_news` → `inputs/multinews_s.json` (run on a host with HF access) |
| `utils.py` | `load_dataset`, `split_into_docs` (per-article chunking), `build_summ_prompt`, `compute_rouge_l` |
| `blend_multinews_generic_kvzip.py` | the benchmark (mirror of the musique script) |
| `inputs/multinews_smoke.json` | **synthetic** 2-example fixture for offline code-path smoke tests — **not real Multi-News data** |

## 1. Build the input JSON (needs Hugging Face access)

`build_inputs.py` pulls `THUDM/LongBench` config `multi_news` (test split) and
converts each example into the `{ctxs, question, answers}` schema — the same
schema the CacheBlend authors used for their preprocessed samsum/musique inputs —
splitting each cluster's `context` into one chunk per source article (the
Multi-News `|||||` document separator, with a blank-line fallback).

```bash
pip install datasets rouge_score
python benchmarks/multinews/build_inputs.py                 # first 150 (matches musique_s.json)
MULTINEWS_N=50 MULTINEWS_MIN_DOCS=3 python benchmarks/multinews/build_inputs.py
```

`question` is empty (LongBench `multi_news` has an empty `input`); the
summarization instruction lives in the benchmark's prompts, not the data —
exactly as in the musique benchmark.

## 2. Run the benchmark (GPU pod)

```bash
export PYTHONPATH=src/external/KVzip:$PYTHONPATH
CACHEBLEND_MODEL=meta-llama/Llama-3.1-8B-Instruct \
COMPBLEND_KVZIP_RATIOS=0.5,0.4,0.3,0.2,0.1 \
COMPBLEND_RECOMP_RATIOS=0.2,0.15,0.1,0.05 \
COMPBLEND_OUT=$PWD/logs/blend_multinews.json \
    python benchmarks/multinews/blend_multinews_generic_kvzip.py
```

Offline smoke test (no GPU quality expectation — just exercises the pipeline):

```bash
COMPBLEND_MULTINEWS_INPUT=inputs/multinews_smoke.json \
COMPBLEND_ARMS=only_hkvd,importance_only \
    python benchmarks/multinews/blend_multinews_generic_kvzip.py
```

Key env vars: `CACHEBLEND_MODEL`, `CACHEBLEND_MULTINEWS_N` (first N examples),
`COMPBLEND_MULTINEWS_INPUT`, `COMPBLEND_MULTINEWS_MAXTOK` (default 512),
`COMPBLEND_KVZIP_RATIOS`, `COMPBLEND_RECOMP_RATIOS`, `COMPBLEND_ARMS`,
`CACHEBLEND_CHECK_LAYER`, `COMPBLEND_OUT` (use an absolute path — the script
`chdir`s into its own dir).
