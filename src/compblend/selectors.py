"""Token selectors for CompBlend.

Each selector answers one question: given per-token HKVD deviation and
per-token compression importance, which positions should we recompute?
They share a uniform signature and the same three masks, and all return
int64 indices sorted ascending by position (causal attention needs
monotonically increasing K positions).

Layout
──────
    Primitives      select_topk_sorted, chunk_internal_rank
    Mask helpers    _coerce_masks, _masked_topk
    Selectors       hkvd_only, importance_only, random_select, anti_importance,
                    gated_hkvd, hkvd_then_importance_prune, hkvd_importance_exclude
    Dispatch        SELECTOR_REGISTRY + select_recompute_indices(config, ...)

The fusor calls `select_recompute_indices(config, ...)`; all selection logic
lives here.

Three masks, shared by every selector
──────────────────────────────────────
    forced_mask      True ⇒ ALWAYS recomputed (score → +inf). Counts toward
                     the budget; overrides structural and eligibility. In the
                     token-prune path this is just the last position (decode
                     needs valid logits).
    structural_mask  True ⇒ EXEMPT, never selected (score → −inf). For
                     window/sink-protected positions. Gated by
                     `config.exempt_structural`.
    eligible_mask    The candidate pool — only these MAY be selected (others
                     → −inf): don't resurrect evicted tokens. All-True under
                     token-prune (whole tokens are kept/dropped). `hkvd_only`
                     ignores it by design.
"""
from __future__ import annotations

import torch

from compblend.config import CompBlendConfig


# ──────────────────────────────────────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────────────────────────────────────


def select_topk_sorted(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k by score, returned as int64 indices sorted ASCENDING by position.

    Causal attention requires K positions to be monotonically increasing.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k = min(k, int(scores.numel()))
    top = torch.topk(scores, k=k).indices
    top, _ = torch.sort(top)
    return top


def _rank_normalize(scores: torch.Tensor) -> torch.Tensor:
    """Percentile-rank transform. Returns values in [0, 1]; ties broken arbitrarily."""
    n = scores.numel()
    if n == 0:
        return scores
    order = torch.argsort(scores, descending=False)
    ranks = torch.empty_like(scores, dtype=torch.float32)
    ranks[order] = torch.arange(n, device=scores.device, dtype=torch.float32) / max(n - 1, 1)
    return ranks


def chunk_internal_rank(chunk_importance: torch.Tensor) -> torch.Tensor:
    """Per-chunk percentile rank — makes importance comparable across chunks.

    Each chunk is compressed in isolation, so its importance is computed
    against its own reconstruction and raw values are only meaningful within
    a chunk. Ranking within each chunk puts every chunk on the same [0, 1]
    scale, the basis for the cross-chunk global top-k the selectors run.
    """
    return _rank_normalize(chunk_importance)


# ──────────────────────────────────────────────────────────────────────────
# Shared mask handling — every selector goes through these
# ──────────────────────────────────────────────────────────────────────────


def _coerce_masks(
    n: int,
    device: torch.device,
    structural_mask: torch.Tensor | None,
    forced_mask: torch.Tensor | None,
    eligible_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coerce the three optional masks to bool[n] on `device`.

    Resolves the one conflict rule: forced wins over structural (a position
    that's window-protected but has no cached K still MUST recompute).
    Returns (forced, structural, eligible).
    """
    def _c(m: torch.Tensor | None, default: bool) -> torch.Tensor:
        if m is None:
            return torch.full((n,), default, dtype=torch.bool, device=device)
        if int(m.numel()) != n:
            raise ValueError(f"mask length mismatch: expected {n}, got {int(m.numel())}")
        return m.to(device=device, dtype=torch.bool)

    forced = _c(forced_mask, False)
    structural = _c(structural_mask, False) & ~forced     # forced wins
    eligible = _c(eligible_mask, True)
    return forced, structural, eligible


def _target_k(recompute_k: int, forced: torch.Tensor) -> int:
    """Budget is a LOWER bound: if forced exceeds it, the budget grows to fit."""
    return max(recompute_k, int(forced.sum().item()))


def _masked_topk(
    scores: torch.Tensor,
    recompute_k: int,
    *,
    forced: torch.Tensor,
    structural: torch.Tensor,
    eligible: torch.Tensor,
    honor_eligible: bool,
) -> torch.Tensor:
    """Force-in (+inf), exempt/ineligible-out (−inf), then top-k by score.

    The single scoring primitive behind the score-based selectors
    (hkvd_only, importance_only, random_select, anti_importance).
    """
    s = scores.detach().clone().float()
    s[forced] = float("inf")
    s[structural] = float("-inf")
    if honor_eligible:
        s[(~eligible) & (~forced)] = float("-inf")
    return select_topk_sorted(s, _target_k(recompute_k, forced))


# ──────────────────────────────────────────────────────────────────────────
# Score-based selectors (one signal, top-k)
# ──────────────────────────────────────────────────────────────────────────


def hkvd_only(
    deviations: torch.Tensor,
    importance: torch.Tensor,
    recompute_k: int,
    *,
    structural_mask: torch.Tensor | None = None,
    forced_mask: torch.Tensor | None = None,
    eligible_mask: torch.Tensor | None = None,
    honor_eligible: bool = False,
) -> torch.Tensor:
    """Top-k by HKVD deviation only (baseline). Ignores importance.

    Does NOT honor eligibility by default: this baseline may pick evicted
    positions.
    """
    forced, structural, eligible = _coerce_masks(
        int(deviations.numel()), deviations.device,
        structural_mask, forced_mask, eligible_mask,
    )
    return _masked_topk(deviations, recompute_k, forced=forced, structural=structural,
                        eligible=eligible, honor_eligible=honor_eligible)


def importance_only(
    deviations: torch.Tensor,
    importance: torch.Tensor,
    recompute_k: int,
    *,
    structural_mask: torch.Tensor | None = None,
    forced_mask: torch.Tensor | None = None,
    eligible_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Top-k by compression importance only (FDI). Ignores HKVD deviation.

    The key comparison: importance as the recompute SELECTOR. Honors
    eligibility (importance is ~0 at evicted positions anyway; the explicit
    guard prevents picking them on ties).
    """
    forced, structural, eligible = _coerce_masks(
        int(importance.numel()), importance.device,
        structural_mask, forced_mask, eligible_mask,
    )
    return _masked_topk(importance, recompute_k, forced=forced, structural=structural,
                        eligible=eligible, honor_eligible=True)


def random_select(
    deviations: torch.Tensor,
    importance: torch.Tensor,
    recompute_k: int,
    *,
    structural_mask: torch.Tensor | None = None,
    forced_mask: torch.Tensor | None = None,
    eligible_mask: torch.Tensor | None = None,
    seed: int = 1234,
) -> torch.Tensor:
    """Control: random top-k (fixed seed). Proves HKVD/importance carry signal."""
    n = int(deviations.numel())
    device = deviations.device
    forced, structural, eligible = _coerce_masks(
        n, device, structural_mask, forced_mask, eligible_mask,
    )
    g = torch.Generator(device=device).manual_seed(seed)
    scores = torch.rand(n, generator=g, device=device)
    return _masked_topk(scores, recompute_k, forced=forced, structural=structural,
                        eligible=eligible, honor_eligible=True)


def anti_importance(
    deviations: torch.Tensor,
    importance: torch.Tensor,
    recompute_k: int,
    *,
    structural_mask: torch.Tensor | None = None,
    forced_mask: torch.Tensor | None = None,
    eligible_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Control: BOTTOM-k by importance (recompute the LEAST important). Anti-signal."""
    forced, structural, eligible = _coerce_masks(
        int(importance.numel()), importance.device,
        structural_mask, forced_mask, eligible_mask,
    )
    return _masked_topk(-importance, recompute_k, forced=forced, structural=structural,
                        eligible=eligible, honor_eligible=True)


def position_select(
    deviations: torch.Tensor,
    importance: torch.Tensor,
    recompute_k: int,
    *,
    structural_mask: torch.Tensor | None = None,
    forced_mask: torch.Tensor | None = None,
    eligible_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Positional baseline: recompute the LAST-k positions (ignores both signals).

    Tests whether HKVD's advantage is just "recompute later tokens" (later chunks
    gain the most cross-chunk context, so deviation skews late). If this matches
    HKVD, position alone explains it; if it underperforms, deviation adds
    content-level signal beyond position.
    """
    n = int(deviations.numel())
    forced, structural, eligible = _coerce_masks(
        n, deviations.device, structural_mask, forced_mask, eligible_mask,
    )
    scores = torch.arange(n, dtype=torch.float32, device=deviations.device)
    return _masked_topk(scores, recompute_k, forced=forced, structural=structural,
                        eligible=eligible, honor_eligible=True)


def score_fuse(
    deviations: torch.Tensor,
    importance: torch.Tensor,
    recompute_k: int,
    *,
    alpha: float,
    structural_mask: torch.Tensor | None = None,
    forced_mask: torch.Tensor | None = None,
    eligible_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Top-k by a fused score: alpha*rank(deviation) + (1-alpha)*rank(importance).

    Direct score fusion of the two signals — the approach the paper avoids (it
    uses gating instead, arguing fusion is unreliable when the signals are
    negatively correlated). Rank-normalizing each to [0,1] makes them scale-fair.
    alpha=1 → pure HKVD; alpha=0 → pure importance.
    """
    n = int(deviations.numel())
    forced, structural, eligible = _coerce_masks(
        n, deviations.device, structural_mask, forced_mask, eligible_mask,
    )
    fused = alpha * _rank_normalize(deviations.float()) + (1.0 - alpha) * _rank_normalize(importance.float())
    return _masked_topk(fused, recompute_k, forced=forced, structural=structural,
                        eligible=eligible, honor_eligible=True)


# ──────────────────────────────────────────────────────────────────────────
# Gated HKVD — importance gates the pool, HKVD picks within
# ──────────────────────────────────────────────────────────────────────────


def gated_hkvd(
    deviations: torch.Tensor,
    importance: torch.Tensor,
    recompute_k: int,
    *,
    gate_percentile: float,
    structural_mask: torch.Tensor | None = None,
    forced_mask: torch.Tensor | None = None,
    eligible_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gated HKVD: importance gates the candidate set, HKVD picks within it.

    Algorithm:
      1. Include all `forced_mask` positions unconditionally.
      2. Exclude `structural_mask`; restrict to `eligible_mask`.
      3. Keep the top `gate_percentile` fraction of candidates by importance.
      4. From that gated set, pick `recompute_k − forced_count` by HKVD.

    Forced positions are included before the gate runs (they don't participate
    in the gate's percentile), so the gate cardinality stays interpretable.
    """
    n = int(deviations.numel())
    if int(importance.numel()) != n:
        raise ValueError(
            f"score length mismatch: hkvd={n}, importance={int(importance.numel())}"
        )
    forced, structural, eligible = _coerce_masks(
        n, deviations.device, structural_mask, forced_mask, eligible_mask,
    )
    forced_idx = forced.nonzero(as_tuple=False).squeeze(-1)
    forced_count = int(forced_idx.numel())

    target_k = max(recompute_k, forced_count)
    remaining_slots = max(target_k - forced_count, 0)
    if remaining_slots == 0:
        out, _ = torch.sort(forced_idx)
        return out

    # Candidate pool: not-forced AND not-structural AND eligible.
    cand = (~forced) & (~structural) & eligible
    if not bool(cand.any().item()):
        out, _ = torch.sort(forced_idx)
        return out

    # ── Importance gate, ranked WITHIN the candidate pool ────────────────
    if 0.0 < gate_percentile < 1.0:
        cand_idx = cand.nonzero(as_tuple=False).squeeze(-1)
        cand_imp = importance[cand_idx]
        n_cand = int(cand_idx.numel())
        n_keep_gate = max(int(round(n_cand * gate_percentile)), 1)
        sorted_imp, _ = torch.sort(cand_imp, descending=True)
        threshold = sorted_imp[min(n_keep_gate - 1, n_cand - 1)]
        passes_gate = cand & (importance >= threshold)
    else:
        passes_gate = cand     # gate_percentile outside (0, 1): no gating

    # ── HKVD top-k within the gated pool ─────────────────────────────────
    gated_idx = passes_gate.nonzero(as_tuple=False).squeeze(-1)
    if int(gated_idx.numel()) == 0:
        out, _ = torch.sort(forced_idx)
        return out
    take = min(remaining_slots, int(gated_idx.numel()))
    top_in_gate = torch.topk(deviations[gated_idx], k=take).indices
    picked = gated_idx[top_in_gate]

    union = torch.cat([forced_idx, picked], dim=0).unique()
    union, _ = torch.sort(union)
    return union


# ──────────────────────────────────────────────────────────────────────────
# HKVD-first variants (HKVD pre-selects, importance trims) — ablation/controls
# ──────────────────────────────────────────────────────────────────────────


def hkvd_then_importance_prune(
    deviations: torch.Tensor,
    importance: torch.Tensor,
    recompute_k: int,
    *,
    prune_k: int,
    structural_mask: torch.Tensor | None = None,
    forced_mask: torch.Tensor | None = None,
    eligible_mask: torch.Tensor | None = None,
    keep_low: bool = False,
) -> torch.Tensor:
    """HKVD-first, then importance-prune (REVERSE order of gated_hkvd).

    HKVD picks the candidate set FIRST (the top-deviation tokens are always in),
    then importance only TRIMS the `prune_k` lowest-importance from it. A
    conservative use of importance designed NOT to drop the high-deviation
    tokens HKVD considers essential.

    `keep_low=False` (default) drops the lowest-importance survivors (keep HIGH).
    `keep_low=True` keeps the LOWEST-importance ones — the split-test control
    "HKVD-pool ∩ low-importance".

    Algorithm: forced included; candidates = eligible & ~structural & ~forced;
    HKVD top-`recompute_k` of candidates; drop `prune_k` lowest-IMPORTANCE
    (or highest, if keep_low); union with forced, sorted.
    """
    n = int(deviations.numel())
    forced, structural, eligible = _coerce_masks(
        n, deviations.device, structural_mask, forced_mask, eligible_mask,
    )
    forced_idx = forced.nonzero(as_tuple=False).squeeze(-1)

    cand = (~forced) & (~structural) & eligible
    if not bool(cand.any().item()):
        out, _ = torch.sort(forced_idx)
        return out
    cand_idx = cand.nonzero(as_tuple=False).squeeze(-1)

    # HKVD pre-select over candidates.
    take = min(max(recompute_k, 1), int(cand_idx.numel()))
    top_in_cand = torch.topk(deviations[cand_idx], k=take).indices
    hkvd_sel = cand_idx[top_in_cand]                       # ~recompute_k positions

    # Drop the prune_k lowest-importance from the HKVD set (keep the rest).
    keep = hkvd_sel
    if prune_k > 0 and int(hkvd_sel.numel()) > prune_k:
        n_keep = int(hkvd_sel.numel()) - prune_k
        imp_sel = importance[hkvd_sel]
        keep_local = torch.topk(-imp_sel if keep_low else imp_sel, k=n_keep).indices
        keep = hkvd_sel[keep_local]

    union = torch.cat([forced_idx, keep], dim=0).unique()
    union, _ = torch.sort(union)
    return union


def importance_then_hkvd_prune(
    deviations: torch.Tensor,
    importance: torch.Tensor,
    recompute_k: int,
    *,
    prune_k: int,
    structural_mask: torch.Tensor | None = None,
    forced_mask: torch.Tensor | None = None,
    eligible_mask: torch.Tensor | None = None,
    keep_low: bool = False,
) -> torch.Tensor:
    """Mirror of hkvd_then_importance_prune: importance pre-selects, HKVD trims.

    Importance picks the candidate set FIRST (top-importance tokens always in),
    then HKVD only TRIMS the `prune_k` survivors. Used for the disagreement-
    quadrant test:
      `keep_low=False` → keep HIGH-HKVD survivors  = importance-high ∩ HKVD-high.
      `keep_low=True`  → keep LOW-HKVD survivors   = importance-high ∩ HKVD-low
                         (the "importance wants it, HKVD doesn't" quadrant).

    Algorithm: forced included; candidates = eligible & ~structural & ~forced;
    importance top-`recompute_k` of candidates; drop `prune_k` by HKVD (lowest,
    or highest if keep_low); union with forced, sorted.
    """
    n = int(deviations.numel())
    forced, structural, eligible = _coerce_masks(
        n, deviations.device, structural_mask, forced_mask, eligible_mask,
    )
    forced_idx = forced.nonzero(as_tuple=False).squeeze(-1)

    cand = (~forced) & (~structural) & eligible
    if not bool(cand.any().item()):
        out, _ = torch.sort(forced_idx)
        return out
    cand_idx = cand.nonzero(as_tuple=False).squeeze(-1)

    # Importance pre-select over candidates.
    take = min(max(recompute_k, 1), int(cand_idx.numel()))
    top_in_cand = torch.topk(importance[cand_idx], k=take).indices
    imp_sel = cand_idx[top_in_cand]                        # ~recompute_k positions

    # Drop the prune_k lowest-HKVD from the importance set (keep the rest).
    keep = imp_sel
    if prune_k > 0 and int(imp_sel.numel()) > prune_k:
        n_keep = int(imp_sel.numel()) - prune_k
        hkvd_sel = deviations[imp_sel]
        keep_local = torch.topk(-hkvd_sel if keep_low else hkvd_sel, k=n_keep).indices
        keep = imp_sel[keep_local]

    union = torch.cat([forced_idx, keep], dim=0).unique()
    union, _ = torch.sort(union)
    return union


def hkvd_importance_exclude(
    deviations: torch.Tensor,
    importance: torch.Tensor,
    recompute_k: int,
    *,
    exclude_percentile: float,
    structural_mask: torch.Tensor | None = None,
    forced_mask: torch.Tensor | None = None,
    eligible_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reuse-safe selector: EXCLUDE high-importance (stable) tokens from recompute.

    Motivation: importance is negatively correlated with deep-layer reuse-error —
    high-importance tokens are MORE stable across context, so they are safe to
    reuse and need no recompute. Drop the top `exclude_percentile` by importance,
    then HKVD top-k among the remaining lower-importance pool. The INVERSE of
    gated_hkvd (which keeps high-importance).
    """
    n = int(deviations.numel())
    forced, structural, eligible = _coerce_masks(
        n, deviations.device, structural_mask, forced_mask, eligible_mask,
    )
    forced_idx = forced.nonzero(as_tuple=False).squeeze(-1)
    forced_count = int(forced_idx.numel())
    target_k = max(recompute_k, forced_count)
    remaining = max(target_k - forced_count, 0)

    cand = (~forced) & (~structural) & eligible
    if remaining == 0 or not bool(cand.any().item()):
        out, _ = torch.sort(forced_idx)
        return out
    cand_idx = cand.nonzero(as_tuple=False).squeeze(-1)

    # Drop the top `exclude_percentile` by importance (reuse-safe high-imp tokens).
    if 0.0 < exclude_percentile < 1.0 and int(cand_idx.numel()) > 1:
        cand_imp = importance[cand_idx]
        n_cand = int(cand_idx.numel())
        n_drop = int(round(n_cand * exclude_percentile))
        n_keep = max(n_cand - n_drop, 1)
        low_local = torch.topk(cand_imp, k=n_keep, largest=False).indices
        pool_idx = cand_idx[low_local]
    else:
        pool_idx = cand_idx

    take = min(remaining, int(pool_idx.numel()))
    top_in_pool = torch.topk(deviations[pool_idx], k=take).indices
    picked = pool_idx[top_in_pool]
    union = torch.cat([forced_idx, picked], dim=0).unique()
    union, _ = torch.sort(union)
    return union


# ──────────────────────────────────────────────────────────────────────────
# Registry + dispatch — the fusor's single entry point
# ──────────────────────────────────────────────────────────────────────────
#
# Each handler adapts the uniform (config, deviations, importance, recompute_k,
# **masks) call into its selector's own argument list, pulling any extra
# parameters off `config`. Adding a selector = add a function above + one
# entry here.


def _h_hkvd_only(config, dev, imp, k, **m):
    return hkvd_only(dev, imp, k, honor_eligible=False, **m)


def _h_importance_only(config, dev, imp, k, **m):
    return importance_only(dev, imp, k, **m)


def _h_random(config, dev, imp, k, **m):
    return random_select(dev, imp, k, **m)


def _h_anti_importance(config, dev, imp, k, **m):
    return anti_importance(dev, imp, k, **m)


def _h_position(config, dev, imp, k, **m):
    return position_select(dev, imp, k, **m)


def _h_score_fuse(config, dev, imp, k, **m):
    return score_fuse(dev, imp, k, alpha=config.fuse_alpha, **m)


def _h_gated_hkvd(config, dev, imp, k, **m):
    return gated_hkvd(dev, imp, k, gate_percentile=config.gate_percentile, **m)


def _h_hkvd_then_imp_prune(config, dev, imp, k, *, keep_low, **m):
    # recompute_k is the HKVD PRE-SELECT count; drop the lowest-importance
    # `importance_prune_ratio` fraction OF TOTAL → final ≈ recompute_ratio − prune.
    prune_k = int(int(dev.numel()) * config.importance_prune_ratio)
    return hkvd_then_importance_prune(dev, imp, k, prune_k=prune_k, keep_low=keep_low, **m)


def _h_imp_then_hkvd_prune(config, dev, imp, k, *, keep_low, **m):
    # Mirror: importance pre-selects, HKVD trims by importance_prune_ratio OF TOTAL.
    prune_k = int(int(dev.numel()) * config.importance_prune_ratio)
    return importance_then_hkvd_prune(dev, imp, k, prune_k=prune_k, keep_low=keep_low, **m)


def _h_hkvd_imp_exclude(config, dev, imp, k, **m):
    return hkvd_importance_exclude(dev, imp, k, exclude_percentile=config.gate_percentile, **m)


SELECTOR_REGISTRY = {
    "hkvd_only":                _h_hkvd_only,
    "importance_only":          _h_importance_only,
    "random":                   _h_random,
    "anti_importance":          _h_anti_importance,
    "position":                 _h_position,
    "score_fuse":               _h_score_fuse,
    "gated_hkvd":               _h_gated_hkvd,
    "hkvd_then_imp_prune":      lambda c, d, i, k, **m: _h_hkvd_then_imp_prune(c, d, i, k, keep_low=False, **m),
    "hkvd_then_imp_prune_high": lambda c, d, i, k, **m: _h_hkvd_then_imp_prune(c, d, i, k, keep_low=True, **m),
    "imp_then_hkvd_prune":      lambda c, d, i, k, **m: _h_imp_then_hkvd_prune(c, d, i, k, keep_low=False, **m),
    "imp_then_hkvd_prune_low":  lambda c, d, i, k, **m: _h_imp_then_hkvd_prune(c, d, i, k, keep_low=True, **m),
    "hkvd_imp_exclude":         _h_hkvd_imp_exclude,
}


def select_recompute_indices(
    config: CompBlendConfig,
    deviations: torch.Tensor,
    importance: torch.Tensor,
    recompute_k: int,
    *,
    structural_mask: torch.Tensor | None = None,
    forced_mask: torch.Tensor | None = None,
    eligible_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch on `config.selector`. Returns sorted-ascending int64 indices.

    Structural exemption is gated by `config.exempt_structural` (when False,
    no position is exempt). `eligible_mask` is the compression-aware candidate
    pool — honored by every selector except `hkvd_only` (baseline, by design).
    """
    try:
        handler = SELECTOR_REGISTRY[config.selector]
    except KeyError:
        raise ValueError(
            f"unknown selector: {config.selector!r}. "
            f"known: {sorted(SELECTOR_REGISTRY)}"
        )
    masks = dict(
        structural_mask=structural_mask if config.exempt_structural else None,
        forced_mask=forced_mask,
        eligible_mask=eligible_mask,
    )
    return handler(config, deviations, importance, recompute_k, **masks)


__all__ = [
    "select_topk_sorted",
    "chunk_internal_rank",
    "hkvd_only",
    "importance_only",
    "random_select",
    "anti_importance",
    "position_select",
    "score_fuse",
    "gated_hkvd",
    "hkvd_then_importance_prune",
    "importance_then_hkvd_prune",
    "hkvd_importance_exclude",
    "SELECTOR_REGISTRY",
    "select_recompute_indices",
]
