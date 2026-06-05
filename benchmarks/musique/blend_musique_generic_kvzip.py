"""musique workload with KVzip token-pruned chunk reuse — extends blend_musique_generic.py.

This is a copy of `blend_musique_generic.py` (the shim-free CacheBlend musique
reproduction) with ONE addition: a KVzip *token-pruning* compression scenario on
top of the same chunked prompt, same dataset, same token-F1 metric.

Compression model (token pruning)
---------------------------------
Each doc chunk is prefilled in isolation (`KVzipBackend.compress(ratio=1.0)` → full
per-chunk KV + KVzip per-(layer,head,token) importance, NO eviction). We then KEEP
the top `kvzip_ratio` fraction of tokens by mean importance and DROP the rest from
the sequence entirely. Because pruning is head-uniform, the surviving tokens carry
a full (all-heads) KV → the recompute-layer attention mask is the plain
sparse-causal slice (no per-head attention mask is ever needed).

Arms compared (all share the dataset / prompt / chunking / token-F1 of the original)
------------------------------------------------------------------------------------
  full_prefill        uncompressed chunks recombined → full prefill. This is the
                      CacheBlend-paper "full prefill" — the best-score ROOFLINE.
  full_reuse          uncompressed per-chunk KV recombined → reuse, NO recompute.
  full_reuse_kvzip    KVzip-token-pruned per-chunk KV recombined → reuse, NO recompute.
  only_hkvd           pruned chunks → HKVD top-k selective recompute (CacheBlend).
  gated_all_hkvd      pruned chunks → importance gate (mean over ALL layers) then HKVD.
  gated_deep_hkvd     pruned chunks → importance gate (mean over layers 15..30) then HKVD.

full_prefill / full_reuse are kvzip-ratio independent (computed once). full_reuse_kvzip
is recompute-ratio independent (one per kvzip ratio). The 3 blending arms run over the
full kvzip_ratio × recompute_ratio grid. (A `prefill_kvzip` arm — full-prefill of the
token-pruned survivors — was intentionally dropped: full-prefilling a token-depleted
context is not a meaningful upper bound; `full_prefill` is the roofline.)

Env vars (superset of blend_musique_generic.py):
    CACHEBLEND_MODEL         HF model id (default meta-llama/Llama-3.1-8B-Instruct)
    CACHEBLEND_DTYPE         model dtype (default bfloat16)
    CACHEBLEND_ATTN_IMPL     attn_implementation (default sdpa)
    CACHEBLEND_CHECK_LAYER   check_layer for the blending arms (default 1)
    CACHEBLEND_MUSIQUE_N     run only the first N examples (default: all 150)
    COMPBLEND_KVZIP_RATIOS   csv keep-fractions     (default "0.5,0.4,0.3,0.2,0.1")
    COMPBLEND_RECOMP_RATIOS  csv recompute fractions (default "0.2,0.15,0.1,0.05")
    COMPBLEND_GATE_PCT       gate percentile for gated arms (default 0.5)
    COMPBLEND_DEEP_LO/HI     deep importance band, half-open (default 15 / 31)
    COMPBLEND_OUT            json summary path (default logs/blend_musique_kvzip.json)

Run (standalone):
    CACHEBLEND_MODEL=meta-llama/Llama-3.1-8B-Instruct \
        python benchmarks/musique/blend_musique_generic_kvzip.py
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Standalone bootstrap ──
# NOTE: this dir holds a musique `utils.py`, but KVzip also ships a top-level
# `utils` PACKAGE. Python auto-prepends the script dir to sys.path, which would
# shadow KVzip's `utils` and break `from utils.func import ...`. So we load the
# musique utils explicitly by file path, then scrub this dir from sys.path.
import importlib.util  # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]                       # benchmarks/musique → repo root


def _load_by_path(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mq = _load_by_path("_musique_utils", _HERE / "utils.py")
load_dataset, build_qa_prompt, compute_f1 = _mq.load_dataset, _mq.build_qa_prompt, _mq.compute_f1

# drop the script dir (and cwd) so KVzip's `utils`/`model` packages resolve cleanly
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _HERE]
for _p in (
    str(_REPO / "src"),
    str(_REPO / "src" / "external" / "cacheblend-hf-v7" / "src"),
    str(_REPO / "src" / "external" / "KVzip"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_HERE)                                 # for load_dataset("inputs/...")

from cacheblend import LayerwiseModel
from cacheblend.chunker import Chunk, _stable_id
from cacheblend.kv_store import KVStore
from cacheblend.fusor import fuse_full_recompute

from compblend.backends.base import CompressionBudget, CompressedChunk, to_kvstore_entry
from compblend.backends.kvzip import KVzipBackend, KVzipConfig
from compblend.config import CompBlendConfig
from compblend.fuse_selective_compblend import fuse_selective_compblend


MODEL = os.environ.get("CACHEBLEND_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DTYPE = os.environ.get("CACHEBLEND_DTYPE", "bfloat16")
ATTN_IMPL = os.environ.get("CACHEBLEND_ATTN_IMPL", "sdpa")
CHECK_LAYER = int(os.environ.get("CACHEBLEND_CHECK_LAYER", "1"))
MAX_NEW_TOKENS = 32

KVZIP_RATIOS = [float(x) for x in os.environ.get("COMPBLEND_KVZIP_RATIOS", "0.5,0.4,0.3,0.2,0.1").split(",") if x.strip()]
RECOMP_RATIOS = [float(x) for x in os.environ.get("COMPBLEND_RECOMP_RATIOS", "0.2,0.15,0.1,0.05").split(",") if x.strip()]
GATE_PCT = float(os.environ.get("COMPBLEND_GATE_PCT", "0.5"))
HKVD_REDUCE = os.environ.get("COMPBLEND_HKVD_REDUCE", "sum")         # sum | max (over-head HKVD reduce)
DEEP_LO = int(os.environ.get("COMPBLEND_DEEP_LO", "15"))
DEEP_HI = int(os.environ.get("COMPBLEND_DEEP_HI", "31"))
OUT = Path(os.environ.get("COMPBLEND_OUT", str(_REPO / "logs" / "blend_musique_kvzip.json")))

# Instruction prompts — VERBATIM from blend_musique.py (experiment definition).
PREFIX_PROMPT = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
QUERY_PROMPT = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"

_WRAPPERS = {
    "mistral": ("[INST]", "[/INST]"),
    "llama-3": ("<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama3":  ("<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "qwen":    ("<|im_start|>user\n",
                "<|im_end|>\n<|im_start|>assistant\n"),
}


def _resolve_wrapper(model_id: str, tokenizer):
    mid = model_id.lower()
    for key, wrap in _WRAPPERS.items():
        if key in mid:
            return wrap
    sentinel = "\x00CONTENT\x00"
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": sentinel}],
        tokenize=False, add_generation_prompt=True,
    )
    if sentinel not in templated:
        raise RuntimeError(f"could not derive instruction wrapper for {model_id!r}")
    pre, post = templated.split(sentinel, 1)
    bos = tokenizer.bos_token or ""
    if bos and pre.startswith(bos):
        pre = pre[len(bos):]
    return pre, post


def _build_chunks(tokenizer, chunk_texts: list[str]) -> list[Chunk]:
    bos = tokenizer.bos_token_id
    chunks: list[Chunk] = []
    for i, text in enumerate(chunk_texts):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if i == 0 and bos is not None:
            ids = [bos] + ids
        chunks.append(Chunk(text=text, token_ids=ids, chunk_id=_stable_id(text, ids)))
    return chunks


def _greedy_decode(model, tokenizer, prefill_logits, past_kv, device):
    eos = getattr(tokenizer, "eos_token_id", None)
    next_id = prefill_logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    generated = [int(next_id.item())]
    with torch.inference_mode():
        for _ in range(MAX_NEW_TOKENS - 1):
            if eos is not None and generated[-1] == eos:
                break
            out = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_id = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            generated.append(int(next_id.item()))
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── KVzip token pruning ──────────────────────────────────────────────────────
def _subset_chunk(cmp: CompressedChunk, keep) -> CompressedChunk:
    """Return a CompressedChunk restricted to the (sorted) token indices `keep`.
    Token-pruning is head-uniform → the surviving tokens carry full all-heads KV."""
    keep = torch.as_tensor(keep, dtype=torch.long)
    keep_list = keep.tolist()
    nl = cmp.num_layers
    return dataclasses.replace(
        cmp,
        chunk_id=cmp.chunk_id + f":keep{len(keep_list)}",
        token_ids=[cmp.token_ids[i] for i in keep_list],
        key_cache=[cmp.key_cache[li][:, keep, :].contiguous() for li in range(nl)],
        value_cache=[cmp.value_cache[li][:, keep, :].contiguous() for li in range(nl)],
        importance=cmp.importance[:, :, keep].contiguous(),
        is_structural=cmp.is_structural[keep].contiguous(),
    )


def _token_prune(cmp: CompressedChunk, keep_ratio: float) -> CompressedChunk:
    """PER-CHUNK budget: keep the top `keep_ratio` of THIS chunk's tokens by importance."""
    n = cmp.chunk_len
    k = max(1, min(n, int(round(n * keep_ratio))))
    imp_tok = cmp.importance.float().mean(dim=(0, 1))
    keep = torch.sort(torch.topk(imp_tok, k).indices).values
    return _subset_chunk(cmp, keep)


def _entry_chunk(cmp: CompressedChunk) -> Chunk:
    """A Chunk whose token_ids match the (possibly pruned) compressed entry."""
    return Chunk(text="", token_ids=list(cmp.token_ids),
                 chunk_id=cmp.chunk_id)


def _run_compblend(lw, chunks, kv_store, selector, recompute_ratio, *, agg="check_layer", prune=0.0):
    # for prune selectors, recompute_ratio is the HKVD PRE-SELECT (rr + prune); dropping the
    # lowest-importance `prune` fraction yields final recompute = rr. prune=0 → no change.
    cfg = CompBlendConfig(
        check_layer=CHECK_LAYER, recompute_ratio=recompute_ratio + prune, selector=selector,
        gate_percentile=GATE_PCT, importance_prune_ratio=prune, importance_aggregation=agg,
        deep_layer_lo=DEEP_LO, deep_layer_hi=DEEP_HI, hkvd_head_reduce=HKVD_REDUCE,
        chunk_normalization="rank")
    out = fuse_selective_compblend(lw, chunks, kv_store, cfg,
                                   return_layerwise_output=True, last_logits_only=True)
    return out


def main() -> int:
    print(f"[kvzip] model={MODEL} dtype={DTYPE} check_layer={CHECK_LAYER} "
          f"kvzip_ratios={KVZIP_RATIOS} recomp={RECOMP_RATIOS} deep=[{DEEP_LO},{DEEP_HI})", flush=True)
    lw = LayerwiseModel(MODEL, dtype=DTYPE, attn_implementation=ATTN_IMPL)
    tokenizer, model, device = lw.tokenizer, lw.model, lw.device
    user_open, assistant_open = _resolve_wrapper(MODEL, tokenizer)
    backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain"))
    os.environ["COMPBLEND_KVZIP_NO_SYS_PROMPT"] = "1"   # isolated per-chunk compress (sink=0)

    eval_dataset = load_dataset("inputs/musique_s.json")
    n_env = os.environ.get("CACHEBLEND_MUSIQUE_N")
    if n_env:
        eval_dataset = eval_dataset[:int(n_env)]
    print(f"[kvzip] {len(eval_dataset)} examples", flush=True)

    # score accumulators
    # blending arms: (name, selector, importance_aggregation, prune_ratio).
    # prune arms = HKVD pre-select (rr + prune) then drop the lowest-importance `prune`
    # fraction → final recompute = rr (matches the other arms for a fair comparison).
    PRUNE = 0.05
    BLEND_ARMS = [
        ("only_hkvd",           "hkvd_only",           "check_layer",   0.0),
        ("importance_only",     "importance_only",     "all_layer",     0.0),  # top-k by importance, NO HKVD
        ("importance_only_max", "importance_only",     "all_layer_max", 0.0),  # top-k by MAX-agg importance
        ("random",              "random",              "check_layer",   0.0),  # control: random top-k
        ("anti_importance",     "importance_only_low", "all_layer",     0.0),  # control: bottom-k importance
        ("hkvd_hi_imp",         "hkvd_then_imp_prune",      "all_layer", 0.15),  # split: HKVD-pool top-2k, keep HIGH-imp k
        ("hkvd_lo_imp",         "hkvd_then_imp_prune_high", "all_layer", 0.15),  # split: HKVD-pool top-2k, keep LOW-imp k
        ("gated_all_hkvd",      "gated_top_k",         "all_layer",     0.0),    # mean over (layer,head)
        ("gated_all_max_hkvd",  "gated_top_k",         "all_layer_max", 0.0),    # MAX over (layer,head)
        ("gated_deep_hkvd",     "gated_top_k",         "deep",          0.0),    # mean over deep,head
        ("gated_deep_max_hkvd", "gated_top_k",         "deep_max",      0.0),    # MAX over deep,head
        ("hkvd_prune",          "hkvd_then_imp_prune", "check_layer",   PRUNE),  # HKVD 20% → drop low-imp → 15%
        ("hkvd_prune_max",      "hkvd_then_imp_prune", "all_layer_max", PRUNE),  # same, MAX-importance prune
    ]
    # optional subset filter: COMPBLEND_ARMS="only_hkvd,gated_all_hkvd,importance_only"
    _arms_env = os.environ.get("COMPBLEND_ARMS", "").strip()
    if _arms_env:
        _want = {a.strip() for a in _arms_env.split(",") if a.strip()}
        BLEND_ARMS = [a for a in BLEND_ARMS if a[0] in _want]
    print(f"[kvzip] arms={[a[0] for a in BLEND_ARMS]} compress=token_prune hkvd_reduce={HKVD_REDUCE}", flush=True)
    f1 = {"full_prefill": [], "full_reuse": []}
    for r in KVZIP_RATIOS:
        f1[f"full_reuse_kvzip@{r}"] = []
        for rr in RECOMP_RATIOS:
            for arm, _s, _a, _p in BLEND_ARMS:
                f1[f"{arm}@kv{r}_rc{rr}"] = []

    for qi, ex in enumerate(eval_dataset):
        answers = ex["answers"]
        doc_prompts, q_prompt = build_qa_prompt(ex, QUERY_PROMPT)
        chunk_texts = [user_open + PREFIX_PROMPT] + list(doc_prompts) + [q_prompt + assistant_open]
        chunks = _build_chunks(tokenizer, chunk_texts)
        doc_slice = slice(1, 1 + len(doc_prompts))

        # ---- uncompressed references ----
        out = fuse_full_recompute(lw, chunks, return_layerwise_output=True)
        res = _greedy_decode(model, tokenizer, out.logits, out.past_key_values, device)
        f1["full_prefill"].append(max(compute_f1(res, a, tokenizer) for a in answers))
        del out
        if device.type == "cuda": torch.cuda.empty_cache()

        # isolated per-chunk compress at ratio=1.0 → full KV + importance (no eviction)
        cmp_full = {}
        for c in chunks:
            ids = torch.tensor([c.token_ids], dtype=torch.long, device=device)
            cmp_full[c.chunk_id] = backend.compress(ids, model=backend.hf_model,
                                                    budget=CompressionBudget(ratio=1.0)).to(device)
        # full_reuse (uncompressed): recombine isolated per-chunk KV, recompute=0
        kv_uncomp = KVStore()
        for c in chunks:
            kv_uncomp._cache[c.chunk_id] = to_kvstore_entry(cmp_full[c.chunk_id])
        out = _run_compblend(lw, chunks, kv_uncomp, "hkvd_only", 0.0)
        res = _greedy_decode(model, tokenizer, out.logits, out.past_key_values, device)
        f1["full_reuse"].append(max(compute_f1(res, a, tokenizer) for a in answers))
        del out
        if device.type == "cuda": torch.cuda.empty_cache()

        # ---- per kvzip ratio: token-prune doc chunks ----
        for r in KVZIP_RATIOS:
            # TOKEN-pruning compression: PER-CHUNK budget — keep the top `r`
            # fraction of each doc chunk's tokens by importance.
            doc_cmps = [cmp_full[chunks[ci].chunk_id] for ci in range(doc_slice.start, doc_slice.stop)]
            doc_pruned = [_token_prune(c, r) for c in doc_cmps]
            pchunks = []; kv_r = KVStore(); di = 0
            for ci in range(len(chunks)):
                if doc_slice.start <= ci < doc_slice.stop:
                    pr = doc_pruned[di]; di += 1
                else:
                    pr = cmp_full[chunks[ci].chunk_id]
                ch = _entry_chunk(pr)
                pchunks.append(ch)
                kv_r._cache[ch.chunk_id] = to_kvstore_entry(pr)

            # full_reuse_kvzip: reuse surviving tokens' KV, recompute=0
            out = _run_compblend(lw, pchunks, kv_r, "hkvd_only", 0.0)
            res = _greedy_decode(model, tokenizer, out.logits, out.past_key_values, device)
            f1[f"full_reuse_kvzip@{r}"].append(max(compute_f1(res, a, tokenizer) for a in answers))
            del out
            if device.type == "cuda": torch.cuda.empty_cache()

            # blending arms over recompute ratios
            for rr in RECOMP_RATIOS:
                for arm, sel, agg, prune in BLEND_ARMS:
                    out = _run_compblend(lw, pchunks, kv_r, sel, rr, agg=agg, prune=prune)
                    res = _greedy_decode(model, tokenizer, out.logits, out.past_key_values, device)
                    f1[f"{arm}@kv{r}_rc{rr}"].append(max(compute_f1(res, a, tokenizer) for a in answers))
                    del out
                    if device.type == "cuda": torch.cuda.empty_cache()
        if (qi + 1) % 10 == 0 or qi == 0:
            print(f"[{qi+1}/{len(eval_dataset)}] full_prefill={np.mean(f1['full_prefill']):.3f} "
                  f"full_reuse={np.mean(f1['full_reuse']):.3f}", flush=True)

    # ── aggregate ──
    means = {k: float(np.mean(v)) for k, v in f1.items() if v}

    # paired bootstrap 95% CI of (arm − only_hkvd), per-question delta averaged over all cells
    cells = [(r, rr) for r in KVZIP_RATIOS for rr in RECOMP_RATIOS]
    rng = np.random.default_rng(0)

    def _bootci(arm, ref="only_hkvd", iters=2000):
        A = np.array([f1[f"{arm}@kv{r}_rc{rr}"] for r, rr in cells])      # [cells, nq]
        B = np.array([f1[f"{ref}@kv{r}_rc{rr}"] for r, rr in cells])
        perq = (A - B).mean(0)                                            # [nq]
        n = len(perq)
        bs = np.array([perq[rng.integers(0, n, n)].mean() for _ in range(iters)])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        return {"mean": float(perq.mean()), "ci_95": [float(lo), float(hi)],
                "significant": bool(lo > 0 or hi < 0)}

    # bootstrap only if only_hkvd is among the run's arms (it's the reference)
    ref_present = all(f"only_hkvd@kv{r}_rc{rr}" in f1 for r, rr in cells)
    boot = ({arm: _bootci(arm) for arm, _s, _a, _p in BLEND_ARMS if arm != "only_hkvd"}
            if ref_present else {})

    summary = {
        "config": {"model": MODEL, "n": len(eval_dataset), "check_layer": CHECK_LAYER,
                   "kvzip_ratios": KVZIP_RATIOS, "recompute_ratios": RECOMP_RATIOS,
                   "gate_percentile": GATE_PCT, "deep_band": [DEEP_LO, DEEP_HI],
                   "compress_mode": "token_prune", "hkvd_reduce": HKVD_REDUCE,
                   "max_new_tokens": MAX_NEW_TOKENS, "metric": "token_f1"},
        "f1_mean": means,
        "bootstrap_vs_only_hkvd": boot,
        "f1_raw": {k: v for k, v in f1.items() if v},   # per-question scores → cross-run paired bootstrap
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print("\n──────── blend_musique_kvzip ────────", flush=True)
    print(f"ROOFLINE full_prefill = {means['full_prefill']:.4f}   full_reuse = {means['full_reuse']:.4f}", flush=True)
    for r in KVZIP_RATIOS:
        print(f"\n kvzip_ratio={r}:  reuse_kvzip={means[f'full_reuse_kvzip@{r}']:.4f}", flush=True)
        for rr in RECOMP_RATIOS:
            row = "  ".join(f"{arm}={means[f'{arm}@kv{r}_rc{rr}']:.3f}" for arm, _s, _a, _p in BLEND_ARMS)
            print(f"    rc={rr}:  {row}", flush=True)
    print("\n──────── bootstrap 95% CI of Δ vs only_hkvd (paired over questions) ────────", flush=True)
    for arm, v in boot.items():
        sig = "★" if v["significant"] else " "
        print(f"  {sig} {arm:22s} Δ={v['mean']:+.4f}  CI[{v['ci_95'][0]:+.4f},{v['ci_95'][1]:+.4f}]", flush=True)
    print(f"\n[kvzip] wrote {OUT}", flush=True)
    print("BLEND_MUSIQUE_KVZIP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
