"""Does HKVD spend its recompute budget on low-attention tokens? (MuSiQue)

Thesis under test:
  (1) KVzip importance IS attention received (reconstruction max-attention).
  (2) A limited recompute budget should go to high-attention tokens.
  (3) HKVD picks high-DEVIATION tokens, which may receive little attention.

Per question, on the SAME fused token sequence, we compute two SELECTOR-FAITHFUL
per-token scores:
  D_sel    = sum_{head,dim} (K_whole - K_chunk)^2  at check_layer=1   (HKVD reuse error)
  imp_sel  = mean over ALL (layer, head) of per-chunk KVzip importance  (FDI all-layer)
             = the reconstruction attention each token receives.

Then, at recompute budget rho=0.15:
  HKVD set = top-rho by D_sel ;  IMP set = top-rho by imp_sel.
We report:
  - mean importance-PERCENTILE of HKVD-selected tokens  (≈0.5 ⇒ HKVD picks ~random attention)
  - fraction of HKVD picks whose attention is below the median
  - ATTENTION MASS captured  = sum(imp_sel over set) / sum(imp_sel over all)
        for HKVD vs IMP vs RANDOM set  (the headline)
  - overlap(HKVD set, IMP set)
Plus pooled (D-percentile, imp-percentile, in_hkvd, in_imp) for a highlighted scatter.

Env: CACHEBLEND_MODEL, CACHEBLEND_MUSIQUE_N, COMPBLEND_OUT, COMPBLEND_RHO (default 0.15),
     COMPBLEND_CHECK_LAYER (default 1).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


_mq = _load_by_path("_musique_utils", _HERE / "utils.py")
load_dataset, build_qa_prompt = _mq.load_dataset, _mq.build_qa_prompt
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _HERE]
for _p in (str(_REPO / "src"), str(_REPO / "src/external/cacheblend-hf-v7/src"),
           str(_REPO / "src/external/KVzip")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_HERE)

from cacheblend import LayerwiseModel
from cacheblend.chunker import Chunk, _stable_id
from compblend.backends.base import CompressionBudget

MODEL = os.environ.get("CACHEBLEND_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DTYPE = os.environ.get("CACHEBLEND_DTYPE", "bfloat16")
OUT = Path(os.environ.get("COMPBLEND_OUT", str(_REPO / "logs" / "attention_vs_hkvd.json")))
RHO = float(os.environ.get("COMPBLEND_RHO", "0.15"))
CHECK_LAYER = int(os.environ.get("COMPBLEND_CHECK_LAYER", "1"))

PREFIX_PROMPT = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
QUERY_PROMPT = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"
_WRAP = {"llama-3": ("<|start_header_id|>user<|end_header_id|>\n\n",
                     "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
         "llama3": ("<|start_header_id|>user<|end_header_id|>\n\n",
                    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")}


def _wrapper(model_id, tokenizer):
    for k, w in _WRAP.items():
        if k in model_id.lower():
            return w
    s = "\x00C\x00"; t = tokenizer.apply_chat_template([{"role": "user", "content": s}], tokenize=False, add_generation_prompt=True)
    pre, post = t.split(s, 1); bos = tokenizer.bos_token or ""
    return (pre[len(bos):] if pre.startswith(bos) else pre), post


def _build_chunks(tokenizer, texts):
    bos = tokenizer.bos_token_id; out = []
    for i, t in enumerate(texts):
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        if i == 0 and bos is not None:
            ids = [bos] + ids
        out.append(Chunk(text=t, token_ids=ids, chunk_id=_stable_id(t, ids)))
    return out


def _pct(x):
    """percentile rank in [0,1] (higher value -> higher pct)."""
    n = len(x)
    return x.argsort().argsort().astype(np.float64) / max(n - 1, 1)


def main() -> int:
    print(f"[attn_vs_hkvd] model={MODEL} rho={RHO} check_layer={CHECK_LAYER}", flush=True)
    os.environ["COMPBLEND_KVZIP_NO_SYS_PROMPT"] = "1"
    lw = LayerwiseModel(MODEL, dtype=DTYPE, attn_implementation="sdpa")
    tokenizer = lw.tokenizer
    from compblend.backends.kvzip import KVzipBackend, KVzipConfig
    backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain"))
    hf_model = backend.hf_model
    device = next(hf_model.parameters()).device
    cfg = hf_model.config
    H = cfg.num_key_value_heads
    Dh = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    user_open, asst_open = _wrapper(MODEL, tokenizer)

    data = load_dataset("inputs/musique_s.json")
    n_env = os.environ.get("CACHEBLEND_MUSIQUE_N")
    if n_env:
        data = data[:int(n_env)]
    print(f"[attn_vs_hkvd] {len(data)} examples", flush=True)

    def compress(ids):
        t = torch.tensor([ids], dtype=torch.long, device=device)
        return backend.compress(t, model=hf_model, budget=CompressionBudget(ratio=1.0)).to("cpu")

    rng = np.random.default_rng(0)
    pq = {"imp_pct_of_hkvd": [], "frac_hkvd_below_med_imp": [],
          "mass_hkvd": [], "mass_imp": [], "mass_random": [],
          "overlap_hkvd_imp": [], "D_pct_of_imp": []}
    scatter = {"d_pct": [], "imp_pct": [], "in_hkvd": [], "in_imp": []}
    n_layers = None

    for qi, ex in enumerate(data):
        doc_prompts, q_prompt = build_qa_prompt(ex, QUERY_PROMPT)
        texts = [user_open + PREFIX_PROMPT] + list(doc_prompts) + [q_prompt + asst_open]
        chunks = _build_chunks(tokenizer, texts)
        fused_ids = [t for c in chunks for t in c.token_ids]
        seq = len(fused_ids)

        cw = compress(fused_ids)
        cc = [compress(c.token_ids) for c in chunks]
        if sum(c.chunk_len for c in cc) != seq or cw.chunk_len != seq:
            print(f"[Q{qi+1} skip] length mismatch", flush=True); continue
        if n_layers is None:
            n_layers = cw.num_layers

        # D_sel at check_layer: sum of squared (K_whole - K_chunk) over head*dim  (HKVD)
        K1 = cw.key_cache[CHECK_LAYER].view(seq, H, Dh).float().numpy()
        K2 = np.concatenate([cc[ci].key_cache[CHECK_LAYER].view(cc[ci].chunk_len, H, Dh).float().numpy()
                             for ci in range(len(chunks))], axis=0)
        D_sel = ((K1 - K2) ** 2).sum(axis=(1, 2))                    # [seq]

        # imp_sel = mean over ALL (layer, head) of per-chunk importance (FDI all-layer) = attention received
        imp_layers = []
        for L in range(n_layers):
            imp4mat = np.concatenate([cc[ci].importance[L].float().numpy()
                                      for ci in range(len(chunks))], axis=1)   # [H, seq]
            imp_layers.append(imp4mat.mean(0))                       # [seq] mean over heads
        imp_sel = np.mean(imp_layers, axis=0)                        # [seq] mean over layers

        k = max(1, int(round(seq * RHO)))
        hkvd_set = set(np.argsort(-D_sel)[:k].tolist())
        imp_set = set(np.argsort(-imp_sel)[:k].tolist())
        rand_set = set(rng.choice(seq, size=k, replace=False).tolist())

        imp_pct = _pct(imp_sel); d_pct = _pct(D_sel)
        hk = np.array(sorted(hkvd_set))
        pq["imp_pct_of_hkvd"].append(float(imp_pct[hk].mean()))
        pq["frac_hkvd_below_med_imp"].append(float((imp_pct[hk] < 0.5).mean()))
        pq["D_pct_of_imp"].append(float(d_pct[np.array(sorted(imp_set))].mean()))
        tot = imp_sel.sum()
        pq["mass_hkvd"].append(float(imp_sel[np.array(sorted(hkvd_set))].sum() / tot))
        pq["mass_imp"].append(float(imp_sel[np.array(sorted(imp_set))].sum() / tot))
        pq["mass_random"].append(float(imp_sel[np.array(sorted(rand_set))].sum() / tot))
        pq["overlap_hkvd_imp"].append(len(hkvd_set & imp_set) / k)

        if len(scatter["d_pct"]) < 40000:
            step = max(1, seq // 300)   # subsample per question
            for t in range(0, seq, step):
                scatter["d_pct"].append(float(d_pct[t])); scatter["imp_pct"].append(float(imp_pct[t]))
                scatter["in_hkvd"].append(int(t in hkvd_set)); scatter["in_imp"].append(int(t in imp_set))

        if (qi + 1) % 10 == 0 or qi == 0:
            print(f"[{qi+1}/{len(data)}] imp_pct_of_HKVD={np.mean(pq['imp_pct_of_hkvd']):.3f} "
                  f"mass HKVD={np.mean(pq['mass_hkvd']):.3f} IMP={np.mean(pq['mass_imp']):.3f} "
                  f"RAND={np.mean(pq['mass_random']):.3f} overlap={np.mean(pq['overlap_hkvd_imp']):.3f}", flush=True)

    summary = {
        "config": {"model": MODEL, "n": len(data), "n_layers": n_layers, "rho": RHO,
                   "check_layer": CHECK_LAYER,
                   "note": "imp_sel = reconstruction attention (KVzip importance, all-layer mean); "
                           "D_sel = HKVD deviation at check_layer"},
        "means": {k: float(np.mean(v)) for k, v in pq.items()},
        "stds": {k: float(np.std(v)) for k, v in pq.items()},
        "per_question": pq,
        "scatter": scatter,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary))
    print("\n──────── HKVD spends budget on low-attention tokens? ────────", flush=True)
    m = summary["means"]
    print(f"  mean importance(attention) percentile of HKVD-selected tokens = {m['imp_pct_of_hkvd']:.3f}  (0.5 = random)", flush=True)
    print(f"  fraction of HKVD picks below MEDIAN attention                  = {m['frac_hkvd_below_med_imp']:.3f}", flush=True)
    print(f"  attention MASS captured by recompute set (rho={RHO}):", flush=True)
    print(f"      HKVD (deviation) = {m['mass_hkvd']:.3f}   IMP (attention) = {m['mass_imp']:.3f}   RANDOM = {m['mass_random']:.3f}", flush=True)
    print(f"  overlap(HKVD set, IMP set)                                     = {m['overlap_hkvd_imp']:.3f}", flush=True)
    print(f"\n[attn_vs_hkvd] wrote {OUT}", flush=True)
    print("ATTN_VS_HKVD_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
