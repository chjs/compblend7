# compblend7

Compressed KV-cache **blending** for multi-document QA. Reuses `cacheblend-hf-v7`
as the CacheBlend core (vendored as a git submodule) and adds a **KVzip
token-pruning** compression scenario plus importance-guided recompute selectors.

This is a **token-prune-only** distillation of `compblend6` — all per-head /
pair-level (per-(layer,head) eviction) code has been removed. Compression here is
**head-uniform**: each doc chunk keeps the top `kvzip_ratio` fraction of *tokens*
by KVzip importance and drops the rest entirely, so the surviving tokens carry a
full (all-heads) KV and no per-head attention mask is ever needed.

## What it does

For each MuSiQue question, doc chunks are compressed in isolation (KVzip), then
recombined and evaluated under several blending arms that differ only in **which
tokens get recomputed** to repair cross-chunk attention:

| arm | meaning |
|---|---|
| `full_prefill` | uncompressed chunks → full prefill — **roofline** |
| `full_reuse` | uncompressed per-chunk KV → reuse, no recompute |
| `full_reuse_kvzip` | token-pruned per-chunk KV → reuse, no recompute |
| `only_hkvd` | HKVD top-k selective recompute (CacheBlend deviation selector) |
| `gated_all_hkvd` | importance gate (mean over all layers) then HKVD |
| `importance_only` | recompute top-k by KVzip importance alone (Full-Depth Importance) |
| `random` / `anti_importance` | controls — random / bottom-k-importance selection |
| `hkvd_hi_imp` / `hkvd_lo_imp` | split-test — HKVD pool kept by high/low importance |

Swept over `COMPBLEND_KVZIP_RATIOS` × `COMPBLEND_RECOMP_RATIOS`; reports per-arm
token-F1 with paired bootstrap CIs vs `only_hkvd`.

## Run (GPU pod)

```bash
export PYTHONPATH=src/external/KVzip:$PYTHONPATH
CACHEBLEND_MODEL=meta-llama/Llama-3.1-8B-Instruct \
COMPBLEND_KVZIP_RATIOS=0.5,0.4,0.3,0.2,0.1 \
COMPBLEND_RECOMP_RATIOS=0.2,0.15,0.1,0.05 \
COMPBLEND_OUT=$PWD/logs/blend.json \
    python benchmarks/musique/blend_musique_generic_kvzip.py
```

Key env vars: `CACHEBLEND_MODEL`, `CACHEBLEND_MUSIQUE_N` (first N examples),
`COMPBLEND_KVZIP_RATIOS`, `COMPBLEND_RECOMP_RATIOS`, `COMPBLEND_ARMS` (subset
filter), `CACHEBLEND_CHECK_LAYER` (default 1), `COMPBLEND_OUT` (use an absolute
path — the script `chdir`s into its own dir).

## Layout

```
src/
  compblend/                 — selectors, KVzip backend, config, fuse_selective_compblend (fusor)
  external/
    cacheblend-hf-v7/        — submodule: CacheBlend core (LayerwiseModel, KVStore, precompute)
    KVzip/                   — submodule: KVzip compression (importance scoring)
benchmarks/musique/          — blend_musique_generic_kvzip.py + utils.py + inputs/
```

## Install (GPU pod, e.g. vast.ai — `pytorch:2.4.1-cuda12.4-cudnn9-devel`)

```bash
git clone --recursive https://github.com/chjs/compblend7
cd compblend7
grep -v -E '^torch(\s|=|$)' src/external/cacheblend-hf-v7/requirements.txt > /tmp/reqs.txt
pip install -r /tmp/reqs.txt
pip install -e src/external/cacheblend-hf-v7 -e .
pip install "transformers==4.51.3"
pip install -e src/external/KVzip --no-deps
```

(Llama-3.1-8B-Instruct is gated — export `HF_TOKEN`.)

## Provenance

Derived from `chjs/compblend6` (token-prune-only subset; per-head/pair-level,
RAG pipeline, paper/results/scripts/tests removed). CacheBlend core and KVzip are
pinned submodule commits.
