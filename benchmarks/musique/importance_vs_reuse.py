"""Does KVzip importance predict where chunk-KV reuse breaks? (MuSiQue)

Computes four objects per question, all aligned on the SAME fused token sequence:

  1. full-prefill pre-RoPE K   — compress the WHOLE prompt (full context).      [KV]
  2. per-chunk isolated K      — compress each chunk alone, concatenate.        [KV]
  3. importance_full           — KVzip importance from the whole-prompt compress. [score]
  4. importance_chunk          — KVzip importance from the per-chunk compress.    [score]

The reuse-error / "needs-recompute" ground truth is the per-token K deviation
    D(t) = mean_head || K1(t) - K2(t) ||      (per layer; pre-RoPE).
We then ask: does importance (3 or 4) line up with D, and which is more appropriate?

Metrics (per layer, averaged over questions):
  - spearman(D, imp3), spearman(D, imp4), spearman(imp3, imp4)
  - top-50% overlap(D, imp3/imp4)  — the 50% KVzip keep-set vs the top-50% high-D set
  - top-15% overlap                — the HKVD recompute regime
  - intra-chunk local-position means of D / imp3 / imp4 (sink artifact at pos 0)

IMPORTANT on the "50% ratio": KVzip prune(0.5) zero-fills evicted K and zeroes evicted
importance (sentinel) — that would corrupt the K1-vs-K2 deviation and the score ranking.
Importance/K are ratio-independent before thresholding, and prune(0.5) keeps exactly the
top-50% by score. So we compress WITHOUT eviction (ratio=1.0 → clean K + raw scores) and
apply 50% as the analysis threshold — identical keep-set, valid comparison.

Env: CACHEBLEND_MODEL (default Llama-3.1-8B-Instruct), CACHEBLEND_MUSIQUE_N (default all),
     COMPBLEND_OUT. Run: python benchmarks/musique/importance_vs_reuse.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ── bootstrap (load musique utils by path; keep KVzip's `utils` package unshadowed) ──
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
OUT = Path(os.environ.get("COMPBLEND_OUT", str(_REPO / "logs" / "importance_vs_reuse.json")))
VIZ_DIR = Path(os.environ.get("COMPBLEND_VIZ_DIR", str(_REPO / "logs" / "imp_vs_reuse_viz")))
N_VIZ = int(os.environ.get("COMPBLEND_N_VIZ", "4"))     # # questions to render overlay plots for
TOPK = [0.50, 0.15]   # 0.50 = the KVzip 50% keep-set; 0.15 = the HKVD recompute regime
# Per-head aggregation of importance/D into a per-token scalar. "none" = raw mean over
# heads (a high-magnitude head can dominate); "rank" = rank-normalize each head across
# tokens to [0,1] THEN mean (scale-fair, outlier-robust — e.g. tames imp4's sink spike).
# Rank metrics (Spearman/overlap) are invariant to monotone transforms of the final
# per-token vector, but DO depend on this head-combination choice — hence the option.
HEAD_NORM = os.environ.get("COMPBLEND_HEAD_NORM", "none")

PREFIX_PROMPT = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
QUERY_PROMPT = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"
_WRAP = {"llama-3": ("<|start_header_id|>user<|end_header_id|>\n\n",
                     "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
         "llama3": ("<|start_header_id|>user<|end_header_id|>\n\n",
                    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
         "mistral": ("[INST]", "[/INST]"), "qwen": ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n")}


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


def _spearman(x, y):
    n = len(x)
    if n < 3:
        return np.nan
    rx = x.argsort().argsort().astype(np.float64); ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum()) * np.sqrt((ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def _rank01(mat):
    """[seq, H] → per-head (column) rank across tokens, scaled to [0, 1]."""
    seq = mat.shape[0]
    return mat.argsort(0).argsort(0).astype(np.float64) / max(seq - 1, 1)


def _head_agg(mat, mode):
    """Combine a [seq, H] per-(token,head) matrix into a [seq] per-token vector.

    mode = [rank_]{mean|max|q90}:
      mean → average over heads (BLURS a token critical to a single head).
      max  → "important/needs-recompute in AT LEAST ONE head" (preserves single-head salience).
      q90  → 90th-percentile over heads (max-like but robust to one noisy head).
      rank_* → rank-normalize each head across tokens to [0,1] first (scale-fair), then combine.
    """
    base = mat
    if mode.startswith("rank"):
        seq = mat.shape[0]
        base = mat.argsort(0).argsort(0).astype(np.float64) / max(seq - 1, 1)
    if "max" in mode:
        return base.max(1)
    if "q90" in mode:
        return np.quantile(base, 0.9, axis=1)
    return base.mean(1)


def _overlap(a, b, frac):
    k = max(1, int(round(len(a) * frac)))
    sa = set(np.argsort(-a)[:k].tolist()); sb = set(np.argsort(-b)[:k].tolist())
    return len(sa & sb) / k


def _boundary_decomp(gpool, bdec):
    """Split pooled global rank-max points into chunk-FIRST tokens vs interior, and
    report Spearman(D, imp) on each + how much of the top-D set is boundary tokens."""
    D = np.array(gpool["D"]); g3 = np.array(gpool["imp3"]); g4 = np.array(gpool["imp4"])
    b = np.array(gpool["bound"], dtype=bool)
    out = {
        "frac_top15D_is_boundary": float(np.mean(bdec["frac_top15D_is_boundary"])) if bdec["frac_top15D_is_boundary"] else None,
        "frac_tokens_boundary": float(np.mean(bdec["frac_boundary"])) if bdec["frac_boundary"] else None,
        "n_boundary": int(b.sum()), "n_interior": int((~b).sum()),
    }
    if b.sum() > 3:
        out["sp_boundary_D_imp3"] = _spearman(D[b], g3[b])
        out["sp_boundary_D_imp4"] = _spearman(D[b], g4[b])
    if (~b).sum() > 3:
        out["sp_interior_D_imp3"] = _spearman(D[~b], g3[~b])
        out["sp_interior_D_imp4"] = _spearman(D[~b], g4[~b])
    return out


def main() -> int:
    print(f"[imp_vs_reuse] model={MODEL}", flush=True)
    os.environ["COMPBLEND_KVZIP_NO_SYS_PROMPT"] = "1"   # no injected sink; scope is the only variable
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
    print(f"[imp_vs_reuse] {len(data)} examples", flush=True)

    def compress(ids):
        t = torch.tensor([ids], dtype=torch.long, device=device)
        return backend.compress(t, model=hf_model, budget=CompressionBudget(ratio=1.0)).to("cpu")

    n_layers = None
    # per-layer accumulators (lists over questions)
    acc = {"sp_D_imp3": [], "sp_D_imp4": [], "sp_imp3_imp4": []}
    for f in TOPK:
        acc[f"ov{int(f*100)}_D_imp3"] = []
        acc[f"ov{int(f*100)}_D_imp4"] = []
    pos_buckets = {k: {p: [] for p in [0, 1, 2, 3, 4, "rest"]} for k in ("D", "imp3", "imp4")}
    viz_examples = []                 # per-question per-token arrays for overlay plots
    pool = {"D": [], "imp3": [], "imp4": []}   # mid-layer pooled points for scatter
    viz_layers = None
    # GLOBAL rank-MAX over (layer, head): per token, "salient/needs-recompute in ANY (layer,head)".
    gacc = {"sp_gD_g3": [], "sp_gD_g4": [], "sp_g3_g4": []}
    for f in TOPK:
        gacc[f"ov{int(f*100)}_gD_g3"] = []
        gacc[f"ov{int(f*100)}_gD_g4"] = []
    gpool = {"D": [], "imp3": [], "imp4": [], "bound": []}
    # boundary test: is the imp4↔D agreement concentrated at chunk-FIRST tokens?
    bdec = {"frac_top15D_is_boundary": [], "frac_boundary": []}

    for qi, ex in enumerate(data):
        doc_prompts, q_prompt = build_qa_prompt(ex, QUERY_PROMPT)
        texts = [user_open + PREFIX_PROMPT] + list(doc_prompts) + [q_prompt + asst_open]
        chunks = _build_chunks(tokenizer, texts)
        fused_ids = [t for c in chunks for t in c.token_ids]
        doc_idx = list(range(1, 1 + len(doc_prompts)))

        cw = compress(fused_ids)                                   # whole-prompt
        cc = [compress(c.token_ids) for c in chunks]               # per-chunk
        seq = len(fused_ids)
        if sum(c.chunk_len for c in cc) != seq or cw.chunk_len != seq:
            print(f"[Q{qi+1} skip] length mismatch", flush=True); continue
        if n_layers is None:
            n_layers = cw.num_layers
            for key in list(acc):
                acc[key] = [[] for _ in range(n_layers)]
            viz_layers = sorted({1, n_layers // 2, min(n_layers - 1, 24)})
        boundaries = list(np.cumsum([len(c.token_ids) for c in chunks])[:-1])
        viz_this = (qi < N_VIZ)
        if viz_this:
            viz_examples.append({"q": qi, "boundaries": boundaries, "layers": {}})
        gD = np.zeros(seq); g3 = np.zeros(seq); g4 = np.zeros(seq)   # running rank-max over (layer,head)

        # per-chunk local-position index for each fused token (for sink buckets); -1 = non-doc
        local_pos = []
        for ci, c in enumerate(chunks):
            for j in range(len(c.token_ids)):
                local_pos.append(j if ci in doc_idx else -1)
        local_pos = np.array(local_pos)

        for L in range(n_layers):
            # K1, K2: [seq, H, Dh] pre-RoPE → per-(token,head) deviation [seq, H]
            K1 = cw.key_cache[L].view(seq, H, Dh).float().numpy()
            K2 = np.concatenate([cc[ci].key_cache[L].view(cc[ci].chunk_len, H, Dh).float().numpy()
                                 for ci in range(len(chunks))], axis=0)
            Dmat = np.sqrt(((K1 - K2) ** 2).sum(-1))               # [seq, H]
            imp3mat = cw.importance[L].float().numpy().T           # [seq, H]
            imp4mat = np.concatenate([cc[ci].importance[L].float().numpy()
                                      for ci in range(len(chunks))], axis=1).T   # [seq, H]
            D = _head_agg(Dmat, HEAD_NORM)                         # [seq] per-token (head-combined)
            imp3 = _head_agg(imp3mat, HEAD_NORM)
            imp4 = _head_agg(imp4mat, HEAD_NORM)
            # global rank-MAX over (layer, head): max of this layer's per-head ranks into the running max
            gD = np.maximum(gD, _rank01(Dmat).max(1))
            g3 = np.maximum(g3, _rank01(imp3mat).max(1))
            g4 = np.maximum(g4, _rank01(imp4mat).max(1))
            acc["sp_D_imp3"][L].append(_spearman(D, imp3))
            acc["sp_D_imp4"][L].append(_spearman(D, imp4))
            acc["sp_imp3_imp4"][L].append(_spearman(imp3, imp4))
            for f in TOPK:
                acc[f"ov{int(f*100)}_D_imp3"][L].append(_overlap(D, imp3, f))
                acc[f"ov{int(f*100)}_D_imp4"][L].append(_overlap(D, imp4, f))
            if viz_this and L in viz_layers:
                viz_examples[-1]["layers"][L] = (D.copy(), imp3.copy(), imp4.copy())
            if L == n_layers // 2 and len(pool["D"]) < 80000:
                pool["D"].extend(D.tolist()); pool["imp3"].extend(imp3.tolist()); pool["imp4"].extend(imp4.tolist())
            # intra-chunk position buckets (use a mid-deep layer L≈n//2 only, to keep it cheap+representative)
            if L == n_layers // 2:
                for arr, key in ((D, "D"), (imp3, "imp3"), (imp4, "imp4")):
                    a = arr.copy()
                    if a.std() > 0:
                        a = (a - a.mean()) / a.std()               # z-score within question for comparability
                    for p in [0, 1, 2, 3, 4]:
                        v = a[local_pos == p]
                        if len(v):
                            pos_buckets[key][p].append(float(v.mean()))
                    v = a[local_pos >= 5]
                    if len(v):
                        pos_buckets[key]["rest"].append(float(v.mean()))
        # per-question global (layer+head rank-max) metrics
        gacc["sp_gD_g3"].append(_spearman(gD, g3)); gacc["sp_gD_g4"].append(_spearman(gD, g4))
        gacc["sp_g3_g4"].append(_spearman(g3, g4))
        for f in TOPK:
            gacc[f"ov{int(f*100)}_gD_g3"].append(_overlap(gD, g3, f))
            gacc[f"ov{int(f*100)}_gD_g4"].append(_overlap(gD, g4, f))
        # boundary mask = first token of EACH chunk (chunk-start positions)
        bmask = np.zeros(seq, dtype=bool)
        bmask[[0] + boundaries] = True
        kD = max(1, int(round(seq * 0.15)))
        topD = set(np.argsort(-gD)[:kD].tolist())
        bdec["frac_top15D_is_boundary"].append(len(topD & set(np.where(bmask)[0].tolist())) / kD)
        bdec["frac_boundary"].append(float(bmask.mean()))
        if len(gpool["D"]) < 80000:
            gpool["D"].extend(gD.tolist()); gpool["imp3"].extend(g3.tolist())
            gpool["imp4"].extend(g4.tolist()); gpool["bound"].extend(bmask.tolist())
        if (qi + 1) % 10 == 0 or qi == 0:
            print(f"[{qi+1}/{len(data)}] L{n_layers//2}: sp(D,imp3)={np.nanmean(acc['sp_D_imp3'][n_layers//2]):.3f} "
                  f"sp(D,imp4)={np.nanmean(acc['sp_D_imp4'][n_layers//2]):.3f} | "
                  f"GLOBAL sp(gD,g3)={np.nanmean(gacc['sp_gD_g3']):.3f} sp(gD,g4)={np.nanmean(gacc['sp_gD_g4']):.3f}", flush=True)

    def per_layer(key):
        return [float(np.nanmean(acc[key][L])) for L in range(n_layers)]

    summary = {"config": {"model": MODEL, "n": len(data), "n_layers": n_layers, "kvzip_keep": 0.5,
                          "head_norm": HEAD_NORM,
                          "note": "compressed at ratio=1.0 (clean K/scores); 50% applied as analysis threshold"},
               "per_layer": {k: per_layer(k) for k in acc},
               "layer_mean": {k: float(np.nanmean([np.nanmean(acc[k][L]) for L in range(n_layers)])) for k in acc},
               "global_rankmax": {k: float(np.nanmean(v)) for k, v in gacc.items()},
               "boundary_decomp": _boundary_decomp(gpool, bdec),
               "intra_chunk_pos_zscore": {k: {str(p): (float(np.nanmean(v)) if v else None)
                                              for p, v in pos_buckets[k].items()} for k in pos_buckets}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))

    # ── visualizations ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        VIZ_DIR.mkdir(parents=True, exist_ok=True)

        def _nz(a):
            r = a.max() - a.min()
            return (a - a.min()) / r if r > 0 else a * 0.0

        # 1) overlay: D(t) with imp3/imp4 on top, per example & layer
        for ex in viz_examples:
            for L, (D, i3, i4) in sorted(ex["layers"].items()):
                x = np.arange(len(D))
                fig, axp = plt.subplots(figsize=(15, 3.6))
                axp.fill_between(x, _nz(D), color="0.82", label="D = ‖K1−K2‖  (reuse error / needs-recompute)")
                axp.plot(x, _nz(i3), color="C0", lw=1.0, label="imp3  (full-context importance)")
                axp.plot(x, _nz(i4), color="C1", lw=1.0, label="imp4  (per-chunk importance)")
                for b in ex["boundaries"]:
                    axp.axvline(b, color="0.55", ls=":", lw=0.6)
                axp.set_title(f"Q{ex['q']}  layer {L}  — per-token (each series min-max normalized); dotted = chunk boundaries")
                axp.set_xlabel("token position"); axp.set_ylabel("normalized"); axp.set_ylim(-0.02, 1.05)
                axp.legend(loc="upper right", fontsize=8, ncol=3)
                fig.tight_layout(); fig.savefig(VIZ_DIR / f"overlay_q{ex['q']}_L{L}.png", dpi=110); plt.close(fig)

        # 2) scatter at mid layer: importance vs D
        Dp = np.array(pool["D"]); i3p = np.array(pool["imp3"]); i4p = np.array(pool["imp4"])
        fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
        for axx, imp, name in ((axs[0], i3p, "imp3 (full-context)"), (axs[1], i4p, "imp4 (per-chunk)")):
            axx.scatter(imp, Dp, s=2, alpha=0.12, color="C0", edgecolors="none")
            axx.set_xlabel(f"{name} importance"); axx.set_title(f"{name}\nSpearman(imp, D) = {_spearman(imp, Dp):+.3f}")
        axs[0].set_ylabel("D = ‖K1−K2‖ (reuse error)")
        fig.suptitle(f"mid layer (L{n_layers//2}), pooled over {len(data)} questions ({len(Dp)} tokens)")
        fig.tight_layout(); fig.savefig(VIZ_DIR / "scatter_midlayer.png", dpi=110); plt.close(fig)

        # 3) per-layer Spearman curve
        fig, axc = plt.subplots(figsize=(9, 4))
        axc.plot(range(n_layers), per_layer("sp_D_imp3"), "-o", ms=3, color="C0", label="Spearman(D, imp3 full-context)")
        axc.plot(range(n_layers), per_layer("sp_D_imp4"), "-s", ms=3, color="C1", label="Spearman(D, imp4 per-chunk)")
        axc.axhline(0, color="0.6", lw=0.6)
        axc.set_xlabel("layer"); axc.set_ylabel("Spearman ρ"); axc.legend()
        axc.set_title("Does KVzip importance predict reuse error D, by layer?")
        fig.tight_layout(); fig.savefig(VIZ_DIR / "per_layer_spearman.png", dpi=110); plt.close(fig)

        # 4) GLOBAL rank-max scatter: imp(any layer,head) vs D(any layer,head)
        gDp = np.array(gpool["D"]); g3p = np.array(gpool["imp3"]); g4p = np.array(gpool["imp4"])
        fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
        for axx, gg, name in ((axs[0], g3p, "imp3 (full-context)"), (axs[1], g4p, "imp4 (per-chunk)")):
            axx.scatter(gg, gDp, s=2, alpha=0.12, color="C2", edgecolors="none")
            axx.set_xlabel(f"{name} global rank-max importance"); axx.set_title(f"{name}\nSpearman = {_spearman(gg, gDp):+.3f}")
        axs[0].set_ylabel("D global rank-max (reuse error)")
        fig.suptitle(f"GLOBAL rank-MAX over (layer, head) — pooled over {len(data)} questions")
        fig.tight_layout(); fig.savefig(VIZ_DIR / "scatter_global_rankmax.png", dpi=110); plt.close(fig)
        print(f"[imp_vs_reuse] wrote plots to {VIZ_DIR}", flush=True)
    except Exception as exc:
        print(f"[imp_vs_reuse] viz skipped: {type(exc).__name__}: {exc}", flush=True)

    print("\n──────── importance vs reuse-error (D = ||K1-K2||) ────────", flush=True)
    print(f"GLOBAL rank-MAX over (layer,head)  [head_norm={HEAD_NORM} for per-layer below]:", flush=True)
    for k, v in summary["global_rankmax"].items():
        print(f"   {k:16s} {v:+.3f}", flush=True)
    print("BOUNDARY decomposition (chunk-first tokens vs interior):", flush=True)
    for k, v in summary["boundary_decomp"].items():
        print(f"   {k:26s} {v}", flush=True)
    print("layer-mean Spearman / overlap:", flush=True)
    for k, v in summary["layer_mean"].items():
        print(f"   {k:16s} {v:+.3f}", flush=True)
    print("\nper-layer Spearman(D, imp3 full / imp4 chunk):", flush=True)
    sp3, sp4 = per_layer("sp_D_imp3"), per_layer("sp_D_imp4")
    for L in range(0, n_layers, max(1, n_layers // 16)):
        print(f"   L{L:2d}: imp3 {sp3[L]:+.3f}   imp4 {sp4[L]:+.3f}", flush=True)
    print("\nintra-chunk position (z-scored, mid layer):", flush=True)
    for k in ("D", "imp3", "imp4"):
        row = summary["intra_chunk_pos_zscore"][k]
        print(f"   {k:5s} " + "  ".join(f"p{p}={row[str(p)]:+.2f}" if row.get(str(p)) is not None else f"p{p}=NA"
                                        for p in [0, 1, 2, 3, 4, "rest"]), flush=True)
    print(f"\n[imp_vs_reuse] wrote {OUT}", flush=True)
    print("IMP_VS_REUSE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
