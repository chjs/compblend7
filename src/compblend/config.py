"""CompBlend runtime configuration — slimmed to Stage-1 fields only.

Compared with CompBlend-old's CompBlendConfig, this drops:
    * `attention_kernel` / `varlen_threshold` — v7 fusor only uses SDPA;
      varlen support is Stage-2 work.
    * `check_layers` tuple + `recompute_ratios` cascade — Stage 1 uses
      a single check_layer. Cascade is Stage 2.
    * `hkvd_compute_dtype` — v7's `kv_deviation` already upcasts to fp32.
    * `boundary_force_count` — EPIC AttnLink is a research extension, not in
      paper §3. Re-add only when needed.
    * `alpha` — additive_rank selector deferred; not currently a default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SelectorKind = Literal[
    "hkvd_only",          # v7 baseline — top-k by deviation only
    "importance_only",    # ablation — top-k by importance only
    "gated_top_k",        # paper §3 default — importance gate then HKVD top-k
    "hkvd_then_imp_prune",  # ablation — HKVD top-k FIRST, then drop lowest-importance (keep HIGH-imp)
    "hkvd_then_imp_prune_high",  # split-test control — HKVD pre-select, keep LOW-imp (drop high-imp)
    "hkvd_imp_exclude",   # reuse-safe — EXCLUDE high-importance (stable) from recompute, HKVD within the rest
    "random",             # control — recompute a RANDOM top-k (no signal); proves importance/HKVD carry signal
    "importance_only_low",# control (anti) — recompute the LOWEST-importance top-k
]


ImportanceAggregation = Literal[
    "check_layer",   # current default — importance at check_layer only, mean over heads
    "all_layer",     # CompBlend-old style — mean over ALL layers AND heads (H3 ablation)
    "deep",          # mean over DEEP layers [L//4, 3L//4) and heads (where imp↔dev corr is strongest)
    "all_layer_max", # MAX over ALL (layer, head): per-(layer,head) rank-normalized then max —
                     # "salient in ANY head/layer" (avoids the mean's blur of single-head salience)
    "deep_max",      # same MAX reduction but over DEEP layers only
]


ChunkNormalization = Literal[
    "none",     # raw per-chunk importance, concatenated as-is (subject to
                # length bias for backends like SnapKV — see CompBlend-old's
                # 4a-1 finding of ~14× scale ratio between short/long docs).
    "rank",     # per-chunk percentile rank in [0, 1]. Length-fair across
                # chunks. Recommended when blending chunks of unequal length.
]


@dataclass
class CompBlendConfig:
    """Configuration consumed by `fuse_selective_compblend`."""

    # Layer at which HKVD selection runs. v7 default = 1 (one full forward
    # pass at layer 0; selection at layer 1; sparse from layer 1 onward).
    check_layer: int = 1

    # Fraction of fused-prompt tokens to recompute. 0 → fall back to
    # full_reuse, 1 → fall back to full_recompute (v7 boundary shortcuts).
    recompute_ratio: float = 0.15

    # Token selector. Default is paper §3 Gated HKVD.
    selector: SelectorKind = "gated_top_k"

    # For gated_top_k: importance percentile threshold over non-forced,
    # non-structural candidates. Keeps the top `gate_percentile` fraction.
    # 1.0 → no gating; 0.0 → no gating (everything passes); typical 0.5.
    gate_percentile: float = 0.5

    # For hkvd_then_imp_prune: after HKVD selects the top `recompute_ratio` fraction
    # (the PRE-SELECT, e.g. 0.20), drop the `importance_prune_ratio` fraction OF TOTAL
    # (e.g. 0.05) with the lowest importance from that set. Final recomputed fraction =
    # recompute_ratio − importance_prune_ratio (e.g. 0.20 − 0.05 = 0.15). 0.0 → no prune.
    importance_prune_ratio: float = 0.0

    # H3 ablation: how the gate's per-token importance is aggregated from the
    # backend's [n_layers, H_kv, chunk_len] importance. "check_layer" = layer
    # `check_layer` only, mean over heads (current). "all_layer" = mean over ALL
    # layers AND heads (CompBlend-old style — richer global salience). Only
    # affects selectors that use importance (gated_top_k, hkvd_then_imp_prune).
    importance_aggregation: ImportanceAggregation = "check_layer"

    # When importance_aggregation == "deep", aggregate importance over layers
    # [deep_layer_lo, deep_layer_hi) (half-open). If both are None, fall back to
    # the default middle band [L//4, 3L//4). E.g. lo=15, hi=31 → layers 15..30.
    deep_layer_lo: int | None = None
    deep_layer_hi: int | None = None

    # How HKVD deviation is reduced over heads into a per-token score. "sum" (default,
    # v7) = sum of squared (K_fresh−K_cached) over all heads*dims (additive — many
    # moderate heads can outrank one large head). "max" = per-head SSE then MAX over
    # heads ("needs recompute in ANY head"), catching single-head-critical reuse errors.
    hkvd_head_reduce: str = "sum"

    # Exempt structural (window / sink) tokens from selector pressure.
    # When True and a backend marks `is_structural[p] = True`, that position
    # is never returned as a selected token. Its cached K/V stays in use.
    exempt_structural: bool = True

    # How per-chunk importance vectors are combined into the fused-prompt
    # `importance_scores`. "none" concatenates raw; "rank" applies within-
    # chunk percentile rank before concat (length-fair).
    chunk_normalization: ChunkNormalization = "none"

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
        if self.selector not in ("hkvd_only", "importance_only", "gated_top_k",
                                 "hkvd_then_imp_prune", "hkvd_then_imp_prune_high",
                                 "hkvd_imp_exclude", "random", "importance_only_low"):
            raise ValueError(f"unknown selector: {self.selector!r}")
        if self.chunk_normalization not in ("none", "rank"):
            raise ValueError(
                f"unknown chunk_normalization: {self.chunk_normalization!r}"
            )
        if self.importance_aggregation not in ("check_layer", "all_layer", "deep",
                                               "all_layer_max", "deep_max"):
            raise ValueError(
                f"unknown importance_aggregation: {self.importance_aggregation!r}"
            )
