"""WHY does HKVD fail as compression rises?  KV-space ground-truth diagnosis (MuSiQue).

Per token we compute THREE pre-RoPE keys (context variants), at a few layers:
  K_full   = key when the WHOLE fused prompt is prefilled uncompressed   → GROUND TRUTH
  K_cached = key when the token's chunk is prefilled in ISOLATION        → the reused value
  K_recomp = key when the TOKEN-PRUNED (survivors-only) fused prompt is prefilled
             → what "recompute in the compressed/blended context" produces

From these, per token (squared-L2 over head*dim, matching HKVD's reduce):
  reuse_error  = ||K_cached - K_full||^2     (how wrong reuse is)
  recomp_error = ||K_recomp - K_full||^2     (how wrong recompute is)
  deviation    = ||K_recomp - K_cached||^2   (HKVD's selection score at check_layer)
  Δ_benefit    = reuse_error - recomp_error  (>0 ⇒ recompute moves the key TOWARD truth = helps)

Swept over kvzip ratio, we report:
  A) mean Δ_benefit for HKVD-selected (top-15% deviation) vs random vs all  → does recompute help?
  B) corr(deviation, reuse_error)             → does HKVD score track TRUE reuse error?
  C) corr(deviation, recomp_error - reuse_error) → does high deviation predict recompute DAMAGE?

Hypothesis (depleted-sequence pathology): at low compression recompute helps & deviation tracks
reuse_error; as compression rises, Δ_benefit for HKVD-selected → ≤0 and C → positive (HKVD picks
exactly the tokens recompute damages most).

Env: CACHEBLEND_MODEL, CACHEBLEND_MUSIQUE_N (default 40), COMPBLEND_KVZIP_RATIOS, COMPBLEND_OUT.
"""
from __future__ import annotations

import importlib.util, json, os, sys
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

_mq = _load("_musique_utils", _HERE / "utils.py")
load_dataset, build_qa_prompt = _mq.load_dataset, _mq.build_qa_prompt
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _HERE]
for _p in (str(_REPO/"src"), str(_REPO/"src/external/cacheblend-hf-v7/src"), str(_REPO/"src/external/KVzip")):
    if _p not in sys.path: sys.path.insert(0, _p)
os.chdir(_HERE)

from cacheblend import LayerwiseModel
from cacheblend.chunker import Chunk, _stable_id
from compblend.backends.base import CompressionBudget

MODEL = os.environ.get("CACHEBLEND_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DTYPE = os.environ.get("CACHEBLEND_DTYPE", "bfloat16")
OUT = Path(os.environ.get("COMPBLEND_OUT", str(_REPO/"logs"/"hkvd_failure.json")))
N = int(os.environ.get("CACHEBLEND_MUSIQUE_N", "40"))
KVR = [float(x) for x in os.environ.get("COMPBLEND_KVZIP_RATIOS", "1.0,0.8,0.6,0.4,0.3,0.2,0.1").split(",") if x.strip()]
CHECK_LAYER = int(os.environ.get("CACHEBLEND_CHECK_LAYER", "1"))

PREFIX = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
QUERY = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"
WRAP = ("<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

def build_chunks(tok, texts):
    bos = tok.bos_token_id; out=[]
    for i,t in enumerate(texts):
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if i==0 and bos is not None: ids=[bos]+ids
        out.append(Chunk(text=t, token_ids=ids, chunk_id=_stable_id(t,ids)))
    return out

def spearman(x,y):
    if len(x)<5: return np.nan
    rx=x.argsort().argsort().astype(float); ry=y.argsort().argsort().astype(float)
    rx-=rx.mean(); ry-=ry.mean()
    d=np.sqrt((rx**2).sum()*(ry**2).sum())
    return float((rx*ry).sum()/d) if d>0 else np.nan

def main():
    print(f"[hkvd_fail] model={MODEL} N={N} kvr={KVR} check_layer={CHECK_LAYER}", flush=True)
    os.environ["COMPBLEND_KVZIP_NO_SYS_PROMPT"]="1"
    lw = LayerwiseModel(MODEL, dtype=DTYPE, attn_implementation="sdpa")
    tok = lw.tokenizer
    from compblend.backends.kvzip import KVzipBackend, KVzipConfig
    backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain"))
    hf = backend.hf_model; dev = next(hf.parameters()).device
    cfg = hf.config; H=cfg.num_key_value_heads; Dh=getattr(cfg,"head_dim",cfg.hidden_size//cfg.num_attention_heads)

    data = load_dataset("inputs/musique_s.json")[:N]
    def compress(ids):
        t=torch.tensor([ids],dtype=torch.long,device=dev)
        return backend.compress(t, model=hf, budget=CompressionBudget(ratio=1.0)).to("cpu")

    LAYERS=[CHECK_LAYER, None, None]  # filled after first
    # accumulators: per kv, per layer → lists of (deviation, reuse_err, recomp_err) per token (pooled over q)
    acc = {kv: {} for kv in KVR}
    nL=None

    for qi,ex in enumerate(data):
        docs,q = build_qa_prompt(ex, QUERY)
        texts=[WRAP[0]+PREFIX]+list(docs)+[q+WRAP[1]]
        chunks=build_chunks(tok,texts)
        fused=[t for c in chunks for t in c.token_ids]
        doc_idx=list(range(1,1+len(docs)))
        seq=len(fused)
        cw=compress(fused)                                  # K_full (whole prompt)
        cc=[compress(c.token_ids) for c in chunks]          # per-chunk (K_cached + importance)
        if cw.chunk_len!=seq or sum(c.chunk_len for c in cc)!=seq:
            print(f"[Q{qi+1} skip] len mismatch",flush=True); continue
        if nL is None:
            nL=cw.num_layers; LAYERS=[CHECK_LAYER, nL//2, min(nL-1,24)]
            for kv in KVR: acc[kv]={L:{"dev":[],"reuse":[],"recomp":[]} for L in LAYERS}
        # chunk offsets in fused seq
        offs=[]; o=0
        for c in chunks: offs.append(o); o+=len(c.token_ids)
        # per-doc-chunk per-token mean importance (over layers,heads) for token-prune
        chunk_imp=[]
        for ci,c in enumerate(chunks):
            imp=np.stack([cc[ci].importance[L].float().numpy() for L in range(nL)],0)  # [L,H,len]
            chunk_imp.append(imp.mean(axis=(0,1)))           # [len] per-token
        # pre-RoPE K tensors: [L][seq,H,Dh]
        Kfull={L:cw.key_cache[L].view(seq,H,Dh).float().numpy() for L in LAYERS}
        Kcached={L:np.concatenate([cc[ci].key_cache[L].view(cc[ci].chunk_len,H,Dh).float().numpy() for ci in range(len(chunks))],0) for L in LAYERS}

        for kv in KVR:
            # build depleted fused: structural chunks full; doc chunks keep top-kv% by importance
            dep_ids=[]; survivors=[]   # survivors: (full_seq_idx, cached_idx==full_seq_idx here)
            for ci,c in enumerate(chunks):
                base=offs[ci]
                if ci in doc_idx and kv<1.0:
                    k=max(1,int(round(len(c.token_ids)*kv)))
                    keep=np.sort(np.argsort(-chunk_imp[ci])[:k])
                else:
                    keep=np.arange(len(c.token_ids))
                for j in keep:
                    dep_ids.append(c.token_ids[j]); survivors.append(base+int(j))
            cr=compress(dep_ids)                              # K_recomp (depleted context)
            Krec={L:cr.key_cache[L].view(len(dep_ids),H,Dh).float().numpy() for L in LAYERS}
            surv=np.array(survivors)
            for L in LAYERS:
                Kf=Kfull[L][surv]; Kc=Kcached[L][surv]; Kr=Krec[L]   # [nsurv,H,Dh]
                reuse=((Kc-Kf)**2).sum(axis=(1,2))
                recomp=((Kr-Kf)**2).sum(axis=(1,2))
                deviation=((Kr-Kc)**2).sum(axis=(1,2))
                acc[kv][L]["reuse"].append(reuse); acc[kv][L]["recomp"].append(recomp); acc[kv][L]["dev"].append(deviation)
        if (qi+1)%10==0 or qi==0:
            print(f"[{qi+1}/{len(data)}] done",flush=True)

    # aggregate
    res={"config":{"model":MODEL,"n":len(data),"kvzip_ratios":KVR,"layers":LAYERS,"check_layer":CHECK_LAYER},"per_kv":{}}
    for kv in KVR:
        res["per_kv"][str(kv)]={}
        for L in LAYERS:
            dev=np.concatenate(acc[kv][L]["dev"]); reuse=np.concatenate(acc[kv][L]["reuse"]); recomp=np.concatenate(acc[kv][L]["recomp"])
            benefit=reuse-recomp
            n=len(dev); k=max(1,int(round(n*0.15)))
            hk=np.argsort(-dev)[:k]                          # HKVD top-15% by deviation
            rng=np.random.default_rng(0); rd=rng.choice(n,size=k,replace=False)
            res["per_kv"][str(kv)][str(L)]={
                "n_tokens":int(n),
                "A_benefit_hkvd":float(benefit[hk].mean()),
                "A_benefit_random":float(benefit[rd].mean()),
                "A_benefit_all":float(benefit.mean()),
                "frac_hkvd_recompute_helps":float((benefit[hk]>0).mean()),
                "B_corr_dev_reuse":spearman(dev,reuse),
                "C_corr_dev_recompDamage":spearman(dev,-benefit),
                "mean_reuse":float(reuse.mean()),"mean_recomp":float(recomp.mean()),
            }
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(res,indent=2))
    print("\n──── HKVD failure diagnosis (layer=check_layer) ────",flush=True)
    L=str(LAYERS[0])
    print(f"{'kv':>5} {'Abenefit_HKVD':>13} {'Abenefit_rand':>13} {'frac_helps':>10} {'B_corr':>7} {'C_corr':>7}",flush=True)
    for kv in KVR:
        r=res["per_kv"][str(kv)][L]
        print(f"{kv:>5} {r['A_benefit_hkvd']:>13.4f} {r['A_benefit_random']:>13.4f} {r['frac_hkvd_recompute_helps']:>10.2f} {r['B_corr_dev_reuse']:>7.3f} {r['C_corr_dev_recompDamage']:>7.3f}",flush=True)
    print(f"\n[hkvd_fail] wrote {OUT}\nHKVD_FAILURE_DONE",flush=True)
    return 0

if __name__=="__main__":
    sys.exit(main())
