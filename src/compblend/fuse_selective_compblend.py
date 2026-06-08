"""CompBlend's selective-recompute fusor.

Selection runs over per-token HKVD deviation + compression importance and is
dispatched through `selectors.select_recompute_indices(config, ...)`. The last
position (and any chunk not in the KVStore) is passed as `forced_mask`;
window/sink positions as `structural_mask`.

Token-prune only: compression is head-uniform (whole tokens are kept or
dropped before blending), so the SDPA `attn_mask` at the recompute layers is
the plain sparse-causal slice `causal_mask_full[:, :, top_indices, :total_seq]`
— no per-head eviction overlay.

When KVStore entries carry no importance, `gated_hkvd` degenerates to HKVD-only.
"""
from __future__ import annotations

import time
from typing import Any

import torch
from torch.nn.functional import scaled_dot_product_attention as _sdpa
from transformers.cache_utils import DynamicCache


class _Timer:
    """Lightweight per-section timer using torch.cuda.synchronize.

    `mark(label)` adds the elapsed time since the previous mark/start to
    a running total under that label. Sections that occur inside loops
    (e.g. per-layer sparse forward) get summed automatically.

    Zero overhead when `enabled=False` — both start() and mark() short-circuit.
    """

    def __init__(self, enabled: bool, device: torch.device) -> None:
        self.enabled = enabled
        self.device = device
        self.events: dict[str, float] = {}
        self.last: float | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.last = time.perf_counter()

    def mark(self, label: str) -> None:
        if not self.enabled or self.last is None:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.events[label] = self.events.get(label, 0.0) + (now - self.last) * 1000.0
        self.last = now

from cacheblend.chunker import Chunk, chunk_offsets, fused_input_ids
from cacheblend.hkvd import kv_deviation
from cacheblend.kv_store import KVStore
from cacheblend.model import LayerwiseOutput

from compblend.config import CompBlendConfig
from compblend.selectors import chunk_internal_rank, select_recompute_indices


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _get_apply_rope(hf_model: Any):
    """Locate `apply_rotary_pos_emb` for the model's family.

    Llama / Mistral / Qwen2 each export their own copy of the function in
    their `modeling_*.py`. The math is identical (real-pair rotation), but
    the symbol lives in different modules. We pick by model class name.
    """
    cls_name = type(hf_model).__name__.lower()
    if "llama" in cls_name:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    elif "mistral" in cls_name:
        from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
    elif "qwen" in cls_name:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    else:
        # Fallback to mistral — every RoPE-using HF model uses identical math.
        from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
    return apply_rotary_pos_emb


def _full_fresh_layers(
    inner: Any,
    hidden_states: torch.Tensor,
    causal_mask_full: torch.Tensor,
    position_ids_full: torch.Tensor,
    position_embeddings_full: tuple[torch.Tensor, torch.Tensor],
    cache_position_full: torch.Tensor,
    past_key_values: DynamicCache,
    layer_range: range,
) -> torch.Tensor:
    """Run the standard HF decoder layer call for layers in `layer_range`.

    Returns updated hidden_states; past_key_values is mutated in-place via
    DynamicCache.update inside each layer.
    """
    for li in layer_range:
        out = inner.layers[li](
            hidden_states=hidden_states,
            attention_mask=causal_mask_full,
            position_ids=position_ids_full,
            past_key_value=past_key_values,
            use_cache=True,
            cache_position=cache_position_full,
            position_embeddings=position_embeddings_full,
        )
        hidden_states = out if not isinstance(out, tuple) else out[0]
    return hidden_states


def _load_chunk_extensions(
    entry: dict[str, Any],
    chunk_len: int,
    n_layers: int,
    n_kv_heads: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read CompBlend extensions from a KVStore entry, defaulting safely.

    Returns:
        importance:    [num_layers, H_kv, chunk_len] fp32
        is_structural: [chunk_len] bool
        has_cache:     [chunk_len] bool — True everywhere (this entry covers
                       all positions of the chunk; gap-detection at fused
                       level is the caller's job)
    """
    if "importance" in entry and entry["importance"] is not None:
        importance = entry["importance"]
        if importance.shape != (n_layers, n_kv_heads, chunk_len):
            raise ValueError(
                f"entry['importance'] shape {tuple(importance.shape)} != "
                f"expected ({n_layers}, {n_kv_heads}, {chunk_len})"
            )
        importance = importance.to(device=device, dtype=torch.float32)
    else:
        importance = torch.ones(
            n_layers, n_kv_heads, chunk_len, dtype=torch.float32, device=device,
        )

    if "is_structural" in entry and entry["is_structural"] is not None:
        is_structural = entry["is_structural"].to(device=device, dtype=torch.bool)
        if int(is_structural.numel()) != chunk_len:
            raise ValueError(
                f"entry['is_structural'].numel()={int(is_structural.numel())} "
                f"!= chunk_len={chunk_len}"
            )
    else:
        is_structural = torch.zeros(chunk_len, dtype=torch.bool, device=device)

    has_cache = torch.ones(chunk_len, dtype=torch.bool, device=device)
    return importance, is_structural, has_cache


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────


def fuse_selective_compblend(
    layerwise_model,
    chunks: list[Chunk],
    kv_store: KVStore,
    config: CompBlendConfig,
    return_layerwise_output: bool = False,
    return_hkvd_indices: bool = False,
    timings: dict | None = None,
    last_logits_only: bool = False,
    flags: dict | None = None,
    selector_stats: dict | None = None,
):
    """Selective recompute + Gated HKVD (token-prune-only).

    Args:
        layerwise_model: `cacheblend.LayerwiseModel` (wraps the HF causal LM).
        chunks: list of `cacheblend.Chunk` covering the FUSED prompt in
            order. Every chunk_id MUST be present in `kv_store`. If you have
            uncached prefix/suffix (system prompt, query), precompute them
            into the store first with `precompute_chunk_kv`.
        kv_store: `cacheblend.KVStore`. Entries may carry CompBlend
            extensions (`importance`, `is_structural`) — if absent, defaults
            are used.
        config: `CompBlendConfig` with check_layer, recompute_ratio,
            selector, gate_percentile, exempt_structural.

    Returns:
        Logits tensor `[1, total_seq, vocab_size]` by default (non-top
        positions zero-filled — only the last position's logits are valid
        for greedy decoding). When `return_layerwise_output=True`, returns
        the `LayerwiseOutput(logits, past_key_values)` for autoregressive
        decode. When `return_hkvd_indices=True`, additionally returns the
        selected top_indices.

    Boundary safe-shortcuts:
      * recompute_ratio == 0   → `fuse_full_reuse`.
      * recompute_ratio >= 1   → `fuse_full_recompute` (no KVStore reads).
      * len(chunks) <= 1       → `fuse_full_recompute`.
    """
    from cacheblend.fusor import fuse_full_recompute, fuse_full_reuse

    # ── Boundary shortcuts ──────────────────────────────────────────────
    if config.recompute_ratio == 0:
        return fuse_full_reuse(
            layerwise_model, chunks, kv_store,
            return_layerwise_output=return_layerwise_output,
        )
    if config.recompute_ratio >= 1:
        return fuse_full_recompute(
            layerwise_model, chunks,
            return_layerwise_output=return_layerwise_output,
        )
    if len(chunks) <= 1:
        return fuse_full_recompute(
            layerwise_model, chunks,
            return_layerwise_output=return_layerwise_output,
        )

    # ── Model-family RoPE function ──────────────────────────────────────
    apply_rotary_pos_emb = _get_apply_rope(layerwise_model.model)

    # ── Setup ───────────────────────────────────────────────────────────
    inner = layerwise_model._inner
    n_layers = layerwise_model.num_layers
    device = layerwise_model.device
    dtype = layerwise_model.dtype
    attn0 = inner.layers[0].self_attn
    num_kv_heads = attn0.config.num_key_value_heads
    num_heads = attn0.config.num_attention_heads
    head_dim = attn0.head_dim
    hidden_kv = num_kv_heads * head_dim
    n_rep = num_heads // num_kv_heads

    # Instrumentation (zero overhead when timings is None)
    timer = _Timer(enabled=(timings is not None), device=device)
    timer.start()

    if not (0 <= config.check_layer < n_layers):
        raise ValueError(
            f"check_layer={config.check_layer} out of range [0, {n_layers})"
        )

    # ── Assemble full-length cached K_pre + V + CompBlend extensions ────
    offsets = chunk_offsets(chunks)
    total_seq = offsets[-1][1]

    K_stored_pre = [
        torch.zeros((1, total_seq, hidden_kv), dtype=dtype, device=device)
        for _ in range(n_layers)
    ]
    V_stored = [
        torch.zeros((1, total_seq, hidden_kv), dtype=dtype, device=device)
        for _ in range(n_layers)
    ]
    if flags is not None:
        bytes_per_elem = (
            dtype.itemsize if hasattr(dtype, "itemsize") else 2
        )
        flags["dense_workspace_bytes"] = (
            2 * n_layers * total_seq * hidden_kv * bytes_per_elem
        )
        flags["dense_workspace_note"] = (
            "Allocates full [1, total_seq, H_kv*D] dense K_stored and "
            "V_stored per layer over the (token-pruned) fused prompt."
        )
        flags["total_seq"] = int(total_seq)
        flags["n_layers"] = int(n_layers)
        flags["selector"] = config.selector
        flags["gate_percentile"] = (
            float(config.gate_percentile)
            if config.selector == "gated_hkvd"
            else None
        )
        flags["recompute_ratio"] = float(config.recompute_ratio)
        flags["check_layer"] = int(config.check_layer)
    # Per-token importance for the selector, reduced over all layers and heads
    # per config.importance_reduce (mean / max).
    importance_full = torch.zeros(total_seq, dtype=torch.float32, device=device)
    structural_full = torch.zeros(total_seq, dtype=torch.bool, device=device)
    # Eligibility for the gated path. Token-prune compression drops whole
    # tokens before blending, so every stored position is head-uniformly
    # valid → eligibility is all-True over the fused prompt.
    eligible_full = torch.ones(total_seq, dtype=torch.bool, device=device)

    for chunk, (start, end) in zip(chunks, offsets):
        if not kv_store.has(chunk.chunk_id):
            raise KeyError(
                f"fuse_selective_compblend: chunk {chunk.chunk_id!r} not in "
                f"KVStore. All chunks must be pre-cached. "
                f"Precompute sys/query chunks via precompute_chunk_kv before "
                f"calling fuse."
            )
        entry = kv_store.get(chunk.chunk_id)
        # K, V — required
        for li in range(n_layers):
            K_stored_pre[li][:, start:end, :] = entry["K"][li]
            V_stored[li][:, start:end, :] = entry["V"][li]
        # CompBlend extensions — optional with safe defaults
        chunk_imp, chunk_struct, _ = _load_chunk_extensions(
            entry, chunk_len=end - start, n_layers=n_layers,
            n_kv_heads=num_kv_heads, device=device,
        )
        # Per-token importance: reduce over ALL layers and heads (importance is
        # full-depth by construction).
        if config.importance_reduce == "max":
            # Rank-normalize each (layer,head) across tokens to [0,1] THEN max →
            # a token salient in ANY head/layer scores high (vs mean, which blurs
            # single-head salience). Per-(layer,head) rank makes it scale-fair.
            _T = chunk_imp.shape[-1]
            flat = chunk_imp.reshape(-1, _T).float()                # [L*H, T]
            ranks = flat.argsort(dim=1).argsort(dim=1).float() / max(_T - 1, 1)
            chunk_imp_1d = ranks.max(dim=0).values                  # [T] max over (layer,head)
        else:                                                       # "mean"
            chunk_imp_1d = chunk_imp.mean(dim=(0, 1))
        if config.chunk_normalization == "rank":
            chunk_imp_1d = chunk_internal_rank(chunk_imp_1d)
        importance_full[start:end] = chunk_imp_1d
        structural_full[start:end] = chunk_struct
    timer.mark("kv_load")

    # ── Forward setup ──────────────────────────────────────────────────
    input_ids = fused_input_ids(chunks, device=device)
    position_ids_full = torch.arange(total_seq, device=device).unsqueeze(0)
    cache_position_full = torch.arange(total_seq, device=device)

    past_key_values = DynamicCache()
    # Reset pre-RoPE K capture (LayerwiseModel's hooks will fire during the
    # check_layer projection — we read the captured tensor for diagnostics
    # if needed; the algorithm itself uses k_full_pre directly).
    layerwise_model._pre_rope_k = {}

    with torch.inference_mode():
        hidden_states = inner.embed_tokens(input_ids)
        cos_full, sin_full = inner.rotary_emb(hidden_states, position_ids_full)
        position_embeddings_full = (cos_full, sin_full)
        timer.mark("io_embed_rotary")

        causal_mask_full = inner._update_causal_mask(
            attention_mask=None,
            input_tensor=hidden_states,
            cache_position=cache_position_full,
            past_key_values=past_key_values,
            output_attentions=False,
        )
        if causal_mask_full is None:
            # SDPA / FlashAttention2 return None — they build their own
            # causal internally. Our manual attention call below needs an
            # explicit additive mask. Construct (1, 1, S, S) -inf-above-diag.
            mask = torch.zeros((1, 1, total_seq, total_seq), dtype=dtype, device=device)
            triu = torch.triu(
                torch.ones(total_seq, total_seq, dtype=torch.bool, device=device),
                diagonal=1,
            )
            mask.masked_fill_(triu, float("-inf"))
            causal_mask_full = mask

        # ── Layers 0..check_layer-1: full fresh forward ─────────────────
        hidden_states = _full_fresh_layers(
            inner, hidden_states, causal_mask_full,
            position_ids_full, position_embeddings_full, cache_position_full,
            past_key_values, layer_range=range(config.check_layer),
        )
        timer.mark("full_prefix")

        # ── Layer check_layer: HKVD + selector + sparse slice ───────────
        layer_ck = inner.layers[config.check_layer]
        attn_ck = layer_ck.self_attn

        residual_full = hidden_states
        h_normed = layer_ck.input_layernorm(hidden_states)
        timer.mark("check_input_ln")

        q_full = attn_ck.q_proj(h_normed)                # (1, S, num_heads*D)
        k_full_pre = attn_ck.k_proj(h_normed)            # (1, S, hidden_kv)
        v_full = attn_ck.v_proj(h_normed)
        timer.mark("check_qkv_proj")

        # HKVD on pre-RoPE K (RoPE preserves L2 — same result as post-RoPE).
        if getattr(config, "hkvd_head_reduce", "sum") == "max":
            _S = k_full_pre.shape[1]
            _kf = k_full_pre.squeeze(0).view(_S, num_kv_heads, head_dim).float()
            _ks = K_stored_pre[config.check_layer].squeeze(0).view(_S, num_kv_heads, head_dim).float()
            deviations = ((_kf - _ks) ** 2).sum(dim=-1).max(dim=1).values   # per-head SSE → MAX over heads
        else:
            deviations = kv_deviation(k_full_pre, K_stored_pre[config.check_layer])
        timer.mark("check_hkvd_score")
        # Phase 2 diagnostic: expose per-position check-layer HKVD deviation so the
        # ablation runner can bucket it by intra-chunk position (sink-artifact test).
        if flags is not None:
            flags["deviations"] = deviations.detach().to("cpu")

        # Forced positions: always the last position (greedy decode needs valid
        # logits). With force_last_chunk, the whole last chunk (the live query
        # suffix) is forced — prefilled fresh against the blended cached KV — and
        # recompute_ratio budgets only the cached (non-query) context.
        forced_mask = torch.zeros(total_seq, dtype=torch.bool, device=device)
        forced_mask[-1] = True
        if config.force_last_chunk and len(chunks) > 1:
            last_start = offsets[-1][0]
            forced_mask[last_start:] = True
            n_forced = int(forced_mask.sum().item())
            recompute_k = n_forced + int((total_seq - n_forced) * config.recompute_ratio)
        else:
            recompute_k = max(int(total_seq * config.recompute_ratio), 1)
        top_indices = select_recompute_indices(
            config,
            deviations,
            importance_full,
            recompute_k,
            structural_mask=structural_full,
            forced_mask=forced_mask,
            eligible_mask=eligible_full,
        )
        topk_num = int(top_indices.shape[0])
        if topk_num == 0:
            raise RuntimeError(
                "selector returned empty top_indices — should be unreachable "
                "since forced_mask always includes the last position."
            )

        is_top = torch.zeros(total_seq, dtype=torch.bool, device=device)
        is_top[top_indices] = True
        timer.mark("check_selector")
        # Backward-compat alias for old "hkvd_select" key — sum of hkvd_score + selector.
        if timings is not None:
            timer.events["hkvd_select"] = (
                timer.events.get("check_hkvd_score", 0.0)
                + timer.events.get("check_selector", 0.0)
            )

        # ─── Selector statistics ────────────────────────────────────────
        if selector_stats is not None:
            structural_mask_cpu = structural_full.to(torch.bool)
            eligible_mask_cpu = eligible_full.to(torch.bool)
            forced_mask_cpu = forced_mask.to(torch.bool)
            is_top_cpu = is_top.to(torch.bool)
            n_total = int(total_seq)
            n_struct = int(structural_mask_cpu.sum().item())
            n_eligible = int(eligible_mask_cpu.sum().item())
            n_forced = int(forced_mask_cpu.sum().item())
            n_selected = int(is_top_cpu.sum().item())
            # Overlap with structural (selected positions that are structural)
            n_sel_from_struct = int((is_top_cpu & structural_mask_cpu).sum().item())
            # Selection diagnostics — how does this selector distribute its
            # choices across eligible / ineligible positions, and how do its
            # picked HKVD / importance values compare to the unpicked rest?
            sel_eligible_count = int((is_top_cpu & eligible_mask_cpu).sum().item())
            sel_ineligible_count = n_selected - sel_eligible_count
            # HKVD / importance distributions at selected vs not-selected
            sel_dev = deviations[top_indices].float()
            not_top = ~is_top
            not_top[forced_mask] = False  # exclude forced (selected anyway)
            unsel_dev = deviations[not_top].float() if bool(not_top.any().item()) else None
            sel_imp = importance_full[top_indices].float()
            # By eligibility — HKVD/importance at eligible vs ineligible
            elig = eligible_full
            inelig = ~eligible_full
            elig_dev_mean = float(deviations[elig].float().mean().item()) if bool(elig.any().item()) else None
            inelig_dev_mean = float(deviations[inelig].float().mean().item()) if bool(inelig.any().item()) else None
            elig_imp_mean = float(importance_full[elig].float().mean().item()) if bool(elig.any().item()) else None
            inelig_imp_mean = float(importance_full[inelig].float().mean().item()) if bool(inelig.any().item()) else None
            selector_stats.update({
                "total_tokens": n_total,
                "structural_tokens": n_struct,
                "compressible_tokens": n_total - n_struct,
                "eligible_tokens": n_eligible,
                "ineligible_tokens": n_total - n_eligible,
                "forced_tokens": n_forced,
                "selected_recompute_tokens": n_selected,
                "selected_from_structural_tokens": n_sel_from_struct,
                "selected_from_compressible_tokens": n_selected - n_sel_from_struct,
                "selected_from_eligible_tokens": sel_eligible_count,
                "selected_from_ineligible_tokens": sel_ineligible_count,
                "recompute_ratio_requested": float(config.recompute_ratio),
                "selected_ratio_actual": (n_selected / n_total) if n_total else 0.0,
                "eligible_ratio": (n_eligible / n_total) if n_total else 0.0,
                "selector": str(config.selector),
                "gate_percentile": (
                    float(config.gate_percentile)
                    if config.selector == "gated_hkvd"
                    else None
                ),
                "gate_is_active": (
                    config.selector == "gated_hkvd"
                    and 0.0 < float(config.gate_percentile) < 1.0
                ),
                # Distribution diagnostics
                "selected_hkvd_mean": float(sel_dev.mean().item()),
                "selected_hkvd_std":  float(sel_dev.std().item()) if sel_dev.numel() > 1 else 0.0,
                "selected_hkvd_min":  float(sel_dev.min().item()),
                "selected_hkvd_max":  float(sel_dev.max().item()),
                "not_selected_hkvd_mean": float(unsel_dev.mean().item()) if unsel_dev is not None else None,
                "selected_importance_mean": float(sel_imp.mean().item()),
                "selected_importance_std":  float(sel_imp.std().item()) if sel_imp.numel() > 1 else 0.0,
                "eligible_hkvd_mean":   elig_dev_mean,
                "ineligible_hkvd_mean": inelig_dev_mean,
                "eligible_importance_mean":   elig_imp_mean,
                "ineligible_importance_mean": inelig_imp_mean,
                # Selected indices themselves (for cross-selector overlap analysis).
                # ~4-5K ints per call → ~30KB JSON. Manageable.
                "selected_indices": top_indices.detach().cpu().tolist(),
            })

        # Mixed K_pre, V: cached at non-top, fresh at top.
        k_mixed_pre = k_full_pre.clone()
        v_mixed = v_full.clone()
        non_top_mask = ~is_top
        k_mixed_pre[:, non_top_mask, :] = K_stored_pre[config.check_layer][:, non_top_mask, :]
        v_mixed[:, non_top_mask, :] = V_stored[config.check_layer][:, non_top_mask, :]
        timer.mark("check_kv_mix")

        # Reshape to heads.
        hidden_shape_full = (1, total_seq, -1, head_dim)
        q_full_heads = q_full.view(hidden_shape_full).transpose(1, 2)
        k_mixed_heads_pre = k_mixed_pre.view(hidden_shape_full).transpose(1, 2)
        v_mixed_heads = v_mixed.view(hidden_shape_full).transpose(1, 2)

        # Apply RoPE on full Q and full mixed K at the full positions.
        q_full_heads_post, k_full_post = apply_rotary_pos_emb(
            q_full_heads, k_mixed_heads_pre, cos_full, sin_full,
        )

        # Slice Q to top_indices.
        q_sparse_post = q_full_heads_post[:, :, top_indices, :]
        residual_sparse = residual_full[:, top_indices, :]
        timer.mark("check_rope")

        past_key_values.update(k_full_post, v_mixed_heads, config.check_layer)

        # GQA expansion of K/V.
        k_rep = k_full_post.repeat_interleave(n_rep, dim=1)
        v_rep = v_mixed_heads.repeat_interleave(n_rep, dim=1)

        # Token-prune compression is head-uniform → the sparse-causal slice is
        # the exact attention mask (no per-head eviction overlay).
        attn_mask_ck = causal_mask_full[:, :, top_indices, :total_seq]
        timer.mark("check_attn_mask_build")
        attn_out_ck = _sdpa(q_sparse_post, k_rep, v_rep, attn_mask=attn_mask_ck, scale=attn_ck.scaling)
        attn_out_ck = attn_out_ck.transpose(1, 2).reshape(
            1, topk_num, num_heads * head_dim,
        ).contiguous()
        timer.mark("check_attention_sdpa")

        attn_out_ck = attn_ck.o_proj(attn_out_ck)
        timer.mark("check_o_proj")

        # Residual + FFN, sparse.
        h_sparse = residual_sparse + attn_out_ck
        residual_sparse2 = h_sparse
        h_sparse_normed = layer_ck.post_attention_layernorm(h_sparse)
        h_sparse = layer_ck.mlp(h_sparse_normed)
        h_sparse = residual_sparse2 + h_sparse
        timer.mark("check_ffn")

        # Roll-up: backward-compatible "check_layer" key = sum of all check_* events
        # for callers that still read the coarse name.
        if timings is not None:
            timer.events["check_layer"] = sum(
                v for k, v in timer.events.items() if k.startswith("check_")
                and k not in ("check_layer",)
            )

        # ── Layers check_layer+1..n_layers-1: sparse hidden, mixed K/V ──
        cos_sparse = cos_full[:, top_indices, :]
        sin_sparse = sin_full[:, top_indices, :]

        for li in range(config.check_layer + 1, n_layers):
            layer_li = inner.layers[li]
            attn_li = layer_li.self_attn

            residual_sparse_in = h_sparse
            h_normed_sparse = layer_li.input_layernorm(h_sparse)
            timer.mark("sparse_input_ln")

            q_sparse = attn_li.q_proj(h_normed_sparse)
            k_sparse_pre = attn_li.k_proj(h_normed_sparse)
            v_sparse = attn_li.v_proj(h_normed_sparse)
            timer.mark("sparse_qkv_proj")

            sparse_shape = (1, topk_num, -1, head_dim)
            q_sparse_heads = q_sparse.view(sparse_shape).transpose(1, 2)
            k_sparse_heads_pre = k_sparse_pre.view(sparse_shape).transpose(1, 2)
            v_sparse_heads = v_sparse.view(sparse_shape).transpose(1, 2)

            q_sparse_post, k_sparse_post = apply_rotary_pos_emb(
                q_sparse_heads, k_sparse_heads_pre, cos_sparse, sin_sparse,
            )
            timer.mark("sparse_rope")

            # Build full K cache: cached(RoPE-shifted) at non-top, fresh at top.
            k_cached_pre_full = K_stored_pre[li]                              # (1, S, hidden_kv)
            k_cached_pre_heads = k_cached_pre_full.view(
                1, total_seq, num_kv_heads, head_dim,
            ).transpose(1, 2)                                                  # (1, H_kv, S, D)
            dummy_q = torch.zeros_like(k_cached_pre_heads)
            _qd, k_cached_post_heads = apply_rotary_pos_emb(
                dummy_q, k_cached_pre_heads, cos_full, sin_full,
            )
            k_full_li = k_cached_post_heads.clone()
            k_full_li[:, :, top_indices, :] = k_sparse_post

            v_cached_heads = V_stored[li].view(
                1, total_seq, num_kv_heads, head_dim,
            ).transpose(1, 2)
            v_full_li = v_cached_heads.clone()
            v_full_li[:, :, top_indices, :] = v_sparse_heads

            past_key_values.update(k_full_li, v_full_li, li)
            timer.mark("sparse_kv_mix")

            k_rep_li = k_full_li.repeat_interleave(n_rep, dim=1)
            v_rep_li = v_full_li.repeat_interleave(n_rep, dim=1)

            # Token-prune compression is head-uniform → sparse-causal slice
            # is the exact attention mask (mirrors check_layer).
            attn_mask_li = causal_mask_full[:, :, top_indices, :total_seq]
            timer.mark("sparse_attn_mask_build")
            attn_out_li = _sdpa(q_sparse_post, k_rep_li, v_rep_li, attn_mask=attn_mask_li, scale=attn_li.scaling)
            attn_out_li = attn_out_li.transpose(1, 2).reshape(
                1, topk_num, num_heads * head_dim,
            ).contiguous()
            timer.mark("sparse_attention_sdpa")

            attn_out_li = attn_li.o_proj(attn_out_li)
            timer.mark("sparse_o_proj")

            h_sparse = residual_sparse_in + attn_out_li
            residual_sparse_in2 = h_sparse
            h_sparse_normed = layer_li.post_attention_layernorm(h_sparse)
            h_sparse = layer_li.mlp(h_sparse_normed)
            h_sparse = residual_sparse_in2 + h_sparse
            timer.mark("sparse_ffn")

        # Roll-up: backward-compatible "sparse_layers" key = sum of all sparse_* events.
        if timings is not None:
            timer.events["sparse_layers"] = sum(
                v for k, v in timer.events.items() if k.startswith("sparse_")
                and k not in ("sparse_layers",)
            )

        # ── Final norm + lm_head on SPARSE hidden ──────────────────────
        h_sparse_normed = inner.norm(h_sparse)
        if last_logits_only:
            # top_indices is sorted ASC and total_seq-1 is forced via forced_mask,
            # so the last sparse row corresponds to the last sequence position.
            logits_full = layerwise_model.model.lm_head(h_sparse_normed[:, -1:, :])
        else:
            logits_sparse = layerwise_model.model.lm_head(h_sparse_normed)        # (1, Q, vocab)

            vocab_size = logits_sparse.shape[-1]
            logits_full = torch.zeros(
                (1, total_seq, vocab_size),
                dtype=logits_sparse.dtype, device=device,
            )
            logits_full[:, top_indices, :] = logits_sparse
        timer.mark("lmhead")

    # Stash timing events on the caller's dict (if provided).
    # _total_ms excludes coarse rollup keys ("check_layer", "sparse_layers",
    # "hkvd_select") to avoid double-counting when fine-grained keys are present.
    _ROLLUP_KEYS = {"check_layer", "sparse_layers", "hkvd_select"}
    if timings is not None:
        timings.update(timer.events)
        timings["_total_ms"] = sum(
            v for k, v in timer.events.items() if k not in _ROLLUP_KEYS
        )

    out_obj = LayerwiseOutput(logits=logits_full, past_key_values=past_key_values)
    result = out_obj if return_layerwise_output else logits_full
    if return_hkvd_indices:
        return result, top_indices
    return result
