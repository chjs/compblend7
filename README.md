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

Convenience runner (sets `PYTHONPATH`, creates `logs/`, picks defaults, tees output):

```bash
export HF_TOKEN=...                                   # Llama-3.1-8B is gated
bash scripts/run_musique_kvzip.sh                     # full grid, N=150
COMPBLEND_MUSIQUE_N=3 COMPBLEND_ARMS=only_hkvd,importance_only \
    bash scripts/run_musique_kvzip.sh                 # quick smoke
```

Or invoke the benchmark directly:

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

## Research context: goal → implementation → status (for reviewers)

This section is written for external review (e.g. an LLM asked to verify the
code). It states the paper's claimed goal, how this repo implements it, what is
**intentionally simplified** here, and — honestly — where our own measurements
diverge from the paper's headline claim.

### 1. What the paper claims (goal)

From *"CompBlend: Reusing Compressed KV with Gated HKVD"* (Choi, Seo, Kim — SAIT):

KV-cache **reuse** (blending) and **compression** are complementary but combining
them naively fails for two reasons: (a) the **decompression trap** — decompressing
before blending erases the compression benefit; (b) an **algorithmic mismatch** —
blending asks *which tokens changed under the new context* (deviation), compression
asks *which tokens matter for output quality* (importance), and these two signals
can be **negatively correlated**. CompBlend's goal is a **pluggable backend
framework** that reuses **compressed** KV caches *without full decompression*,
treating compressed KV as the native representation and decoupling the blending
core from compressor-specific formats via a backend interface that exposes
per-token **importance** + **position** metadata.

Two contributions:
1. **Gated HKVD** — a two-stage recompute selector: **Stage 1** gate the candidate
   set by compression importance `C = { i | Importance(i) ≥ τ }`; **Stage 2** pick
   `top-k` *within C* by High KV Deviation `‖K_fresh(i) − K_cached(i)‖²`. Two stages
   (not one fused score) precisely because importance and deviation can be
   negatively correlated.
2. **CompBlend** — the pluggable, algorithm-agnostic backend framework, plus
   **RoPE-aware realignment** (`K_repos = R(p_new)R(−p_old)K_cached`) so cached keys
   can be reused at a new position.

### 2. How this repo implements it (mechanism → code)

| Paper mechanism | Code |
|---|---|
| Backend interface (importance + position metadata, no decompression) | `src/compblend/backends/base.py` → `CompressedChunk` (pre-RoPE `key_cache`/`value_cache`, per-(layer,head,token) `importance`, `is_structural`, ids) |
| A concrete backend (KVzip reconstruction importance) | `src/compblend/backends/kvzip.py` → `KVzipBackend.compress` (importance = KVzip `kv.score`, shape `[L, H_kv, T]`) |
| **Gated HKVD** (Stage-1 importance gate → Stage-2 HKVD top-k) | `src/compblend/selectors.py::gated_top_k`, dispatched in `fuse_selective_compblend.py::_select_recompute_indices` |
| RoPE-aware realignment | `fuse_selective_compblend.py` applies `apply_rotary_pos_emb` to cached pre-RoPE K at the **fused** position before deviation/attention |
| Selective recompute core (recompute top-k, reuse rest, sparse forward) | `fuse_selective_compblend.py` (fork of cacheblend-hf-v7 `fuse_selective`) |
| Evaluation (MuSiQue, token-F1, arms, bootstrap CIs) | `benchmarks/musique/blend_musique_generic_kvzip.py` |

Selector arms in the benchmark include `only_hkvd` (HKVD-only baseline),
`gated_all_hkvd` (the paper's Gated HKVD), `importance_only` (recompute by
importance alone), and controls `random` / `anti_importance`, plus split-test arms.

### 3. Scope of THIS repo vs the paper (read before reviewing)

`compblend7` is the **token-prune (head-uniform)** subset. Compression here keeps
the top-`kvzip_ratio` fraction of *whole tokens* by importance and **drops the rest
from the sequence**; surviving tokens keep full all-head KV, so there is **no
per-head attention mask and no per-(layer,head) eviction**. The paper's fuller
"compressed-native, per-head eviction, no-decompression" form (closer to KVzip's
native pair-level eviction) was **removed here** and lives in `chjs/compblend6`.
So this repo demonstrates the backend interface + Gated HKVD selector + RoPE-aware
blending **under a simplified compression model**, not the full per-head vision.

### 4. Implementation maturity & honest caveats

- **Stage-1 research prototype.** Verified to run end-to-end (Llama-3.1-8B,
  MuSiQue) on an **80 GB** A100 — it loads two 8B models (the CacheBlend
  `LayerwiseModel` + KVzip's scoring model), so 40 GB is insufficient. Not
  perf-optimized: token-prune re-prefills the surviving tokens.
- **Empirical divergence from the paper's headline (must disclose).** In our
  controlled runs (N=150, paired bootstrap CIs), the paper's headline — *Gated
  HKVD > HKVD-only* — **did not robustly reproduce**: the importance **gate** was
  statistically indistinguishable from `only_hkvd` (gate ≈ noise). The selector
  that showed a small, **regime-dependent** edge was **`importance_only`** (top-k
  by importance alone), strongest at high compression. Mechanistically, HKVD's
  high-deviation picks are ~uncorrelated with importance/attention (Spearman ≈ 0),
  which is consistent with the paper's own "negatively/weakly correlated" premise
  but does **not** support the two-stage *gate* beating HKVD in this token-prune
  regime. Treat the Gated-HKVD-beats-HKVD claim as **not yet supported by this
  implementation's measurements**; the defensible claim is narrower.
- **Known simplifications:** `recompute_ratio=1.0` ≈ re-prefilling a token-depleted
  sequence (so it *underperforms* low recompute at high compression — expected, not
  a bug); the backend's `compression_rate` is `0.0` because eviction happens
  downstream (token-drop in the benchmark), not inside the backend.

### 5. Reviewer checklist (what to verify)

- `gated_top_k`: Stage-1 percentile gate over the *candidate pool*, then HKVD
  `top-(k − forced)` within the gated set; `forced`/`structural`/`eligible` handling.
- Fusor: RoPE applied at the fused position; cached-vs-fresh K mixing at recomputed
  positions; sparse forward from `check_layer` onward.
- Deviation `D = ‖K_fresh − K_cached‖²` computed at `check_layer` (default 1) — and
  why layer 0 is uninformative (no cross-token mixing yet).
- Benchmark: token-F1, paired bootstrap CI vs `only_hkvd`, the kvzip×recompute grid.

## Provenance

Derived from `chjs/compblend6` (token-prune-only subset; per-head/pair-level,
RAG pipeline, paper/results/scripts/tests removed). CacheBlend core and KVzip are
pinned submodule commits.
