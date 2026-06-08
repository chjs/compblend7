"""CompBlend runtime configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SelectorKind = Literal[
    "hkvd_only",          # baseline — top-k by deviation only
    "importance_only",    # FDI — top-k by importance only (no HKVD)
    "gated_hkvd",         # default — importance gate then HKVD top-k
    "hkvd_then_imp_prune",  # ablation — HKVD top-k FIRST, then drop lowest-importance (keep HIGH-imp)
    "hkvd_then_imp_prune_high",  # split-test control — HKVD pre-select, keep LOW-imp (drop high-imp)
    "hkvd_imp_exclude",   # reuse-safe — EXCLUDE high-importance (stable) from recompute, HKVD within the rest
    "random",             # control — recompute a RANDOM top-k (no signal); proves importance/HKVD carry signal
    "anti_importance",    # control (anti) — recompute the LOWEST-importance top-k
]


ImportanceReduce = Literal[
    "mean",   # mean over ALL layers AND heads
    "max",    # per-(layer,head) rank-normalized across tokens, then max over ALL
              # (layer, head) — "salient in ANY head/layer"
]


ChunkNormalization = Literal[
    "none",     # raw per-chunk importance, concatenated as-is
    "rank",     # per-chunk percentile rank in [0, 1]. Length-fair across chunks.
]


@dataclass
class CompBlendConfig:
    """Configuration consumed by `fuse_selective_compblend`."""

    # Layer at which HKVD selection runs (one full forward pass at layer 0;
    # selection at this layer; sparse from this layer onward).
    check_layer: int = 1

    # Fraction of tokens to recompute. 0 → full_reuse, 1 → full_recompute
    # (boundary shortcuts). When force_last_chunk is set, this fraction applies
    # to the cached (non-query) context only.
    recompute_ratio: float = 0.15

    # Realistic serving: treat the last chunk as the live query suffix —
    # prefill it fresh against the blended cached KV (force-recompute its whole
    # span) instead of reusing isolated KV. The query is then never stale, and
    # recompute_ratio budgets only the cached prefix+document context.
    force_last_chunk: bool = False

    # Token selector.
    selector: SelectorKind = "gated_hkvd"

    # For gated_hkvd: importance percentile threshold over non-forced,
    # non-structural candidates. Keeps the top `gate_percentile` fraction.
    # 1.0 → no gating; 0.0 → no gating (everything passes); typical 0.5.
    gate_percentile: float = 0.5

    # For hkvd_then_imp_prune: after HKVD selects the top `recompute_ratio` fraction
    # (the PRE-SELECT, e.g. 0.20), drop the `importance_prune_ratio` fraction OF TOTAL
    # (e.g. 0.05) with the lowest importance from that set. Final recomputed fraction =
    # recompute_ratio − importance_prune_ratio (e.g. 0.20 − 0.05 = 0.15). 0.0 → no prune.
    importance_prune_ratio: float = 0.0

    # How importance is reduced from the backend's [n_layers, H_kv, chunk_len]
    # tensor to a per-token score. Always over ALL layers and heads (importance
    # is full-depth by construction); "mean" or "max". Affects selectors that
    # use importance (importance_only, gated_hkvd, hkvd_then_imp_prune).
    importance_reduce: ImportanceReduce = "mean"

    # How HKVD deviation is reduced over heads into a per-token score. "sum"
    # (default) = sum of squared (K_fresh−K_cached) over all heads*dims. "max" =
    # per-head SSE then MAX over heads ("needs recompute in ANY head").
    hkvd_head_reduce: str = "sum"

    # Exempt structural (window / sink) tokens from selector pressure.
    # When True and a backend marks `is_structural[p] = True`, that position
    # is never returned as a selected token. Its cached K/V stays in use.
    exempt_structural: bool = True

    # How per-chunk importance is combined into the fused-prompt importance.
    # "rank" (default) applies within-chunk percentile rank before concat;
    # "none" concatenates raw. Default is "rank" because each chunk is
    # compressed in isolation, so raw importance is only comparable within a
    # chunk — ranking per chunk is the basis for the cross-chunk global top-k.
    chunk_normalization: ChunkNormalization = "rank"

    def __post_init__(self) -> None:
        if not (0.0 <= self.recompute_ratio <= 1.0):
            raise ValueError(
                f"recompute_ratio must be in [0, 1], got {self.recompute_ratio}"
            )
        if not (0.0 <= self.gate_percentile <= 1.0):
            raise ValueError(
                f"gate_percentile must be in [0, 1], got {self.gate_percentile}"
            )
        if not (0.0 <= self.importance_prune_ratio <= 1.0):
            raise ValueError(
                f"importance_prune_ratio must be in [0, 1], got {self.importance_prune_ratio}"
            )
        if self.check_layer < 0:
            raise ValueError(f"check_layer must be >= 0, got {self.check_layer}")
        if self.selector not in ("hkvd_only", "importance_only", "gated_hkvd",
                                 "hkvd_then_imp_prune", "hkvd_then_imp_prune_high",
                                 "hkvd_imp_exclude", "random", "anti_importance"):
            raise ValueError(f"unknown selector: {self.selector!r}")
        if self.chunk_normalization not in ("none", "rank"):
            raise ValueError(
                f"unknown chunk_normalization: {self.chunk_normalization!r}"
            )
        if self.importance_reduce not in ("mean", "max"):
            raise ValueError(
                f"unknown importance_reduce: {self.importance_reduce!r}"
            )
