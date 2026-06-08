"""Backend contract — pre-RoPE K storage.

Stores PRE-RoPE K and applies RoPE on-the-fly in the fusor (rotating a
pre-RoPE K to any target position is the canonical operation). Positions are
implicit [0, chunk_len); each entry carries per-token `importance`.

Compression is token-prune-only: whole tokens are kept or dropped before
blending, so there is no per-head eviction mask. The dense
`[1, chunk_len, H_kv*D]` K/V storage covers exactly the surviving tokens.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass
class CompressionBudget:
    """How aggressively a backend should compress a chunk."""

    ratio: float | None = None             # fraction of tokens to keep (e.g. 0.30)
    absolute: int | None = None            # absolute tokens to keep (overrides ratio)
    min_kept: int = 1
    per_layer_budget: list[int] | None = None    # PyramidKV-style override

    def __post_init__(self) -> None:
        if self.ratio is None and self.absolute is None and self.per_layer_budget is None:
            raise ValueError(
                "CompressionBudget requires one of: ratio, absolute, per_layer_budget"
            )
        if self.ratio is not None and not (0.0 < self.ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {self.ratio}")
        if self.absolute is not None and self.absolute < 1:
            raise ValueError(f"absolute must be >= 1, got {self.absolute}")


@dataclass
class CompressedChunk:
    """A single text chunk's compressed KV cache.

    Storage convention:
      * `key_cache[layer]` and `value_cache[layer]` are dense at chunk length:
        shape `[1, chunk_len, num_kv_heads * head_dim]`. They store **pre-RoPE**
        K (k_proj output) and V (v_proj output) respectively. Both come straight
        from forward-hooks on the underlying HF model during the backend's
        compress pass.
      * `importance[L, H_kv, chunk_len]` is the backend's per-token salience
        score, in fp32. Used to decide which whole tokens to keep when
        token-pruning the chunk.
      * `is_structural[chunk_len]` flags positions exempt from selector
        pressure. For KVzip it is all False.

    Positions are implicit `[0, chunk_len)`; the fused-prompt position of a
    chunk is decided at blend time and RoPE is applied over those positions.
    """

    # ---- identity ----
    algo_id: str                            # backend that produced this chunk
    chunk_id: str                           # stable hash of (text, tokenizer, model)
    model_id: str
    tokenizer_id: str

    # ---- geometry ----
    token_ids: list[int]                    # chunk_len = len(token_ids)

    # ---- pre-RoPE K/V (one Tensor per layer) ----
    # Layout matches cacheblend.kv_store.KVStore entry exactly:
    #   shape: [1, chunk_len, num_kv_heads * head_dim]
    key_cache: list[torch.Tensor]
    value_cache: list[torch.Tensor]

    # ---- per-(layer, head, position) importance ----
    importance: torch.Tensor                # fp32   [num_layers, H_kv, chunk_len]

    # ---- structural protection ----
    is_structural: torch.Tensor             # bool   [chunk_len]

    # ---- diagnostics (no semantic effect) ----
    compression_rate: float = 0.0
    backend_state: dict[str, Any] = field(default_factory=dict)

    # ──────────────────────────────────────────────────────────────────────
    # Derived properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def chunk_len(self) -> int:
        return len(self.token_ids)

    @property
    def num_layers(self) -> int:
        return len(self.key_cache)

    # ──────────────────────────────────────────────────────────────────────
    # Disk serialization (offline RAG: precompute corpus once, load per query)
    # ──────────────────────────────────────────────────────────────────────
    #
    # Format v1 is a single `torch.save` payload (pickle under the hood). All
    # tensors are moved to CPU and made contiguous before writing so files
    # are portable across machines and devices. Load rehydrates onto CPU by
    # default; call `.to(device)` after loading to move to GPU.

    SERIALIZATION_VERSION: int = 1

    def save(self, path: Any) -> None:
        """Write this chunk to disk in the v1 format."""
        from pathlib import Path as _Path
        p = _Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": CompressedChunk.SERIALIZATION_VERSION,
            "algo_id": self.algo_id,
            "chunk_id": self.chunk_id,
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "token_ids": list(self.token_ids),
            "key_cache": [k.detach().cpu().contiguous() for k in self.key_cache],
            "value_cache": [v.detach().cpu().contiguous() for v in self.value_cache],
            "importance": self.importance.detach().cpu().contiguous(),
            "is_structural": self.is_structural.detach().cpu().contiguous(),
            "compression_rate": float(self.compression_rate),
            "backend_state": dict(self.backend_state),
        }
        torch.save(payload, p)

    @classmethod
    def load(cls, path: Any, map_location: Any = "cpu") -> "CompressedChunk":
        """Read a chunk written by `save`.

        Version mismatches raise. We do NOT silently upgrade old payloads —
        a bad cache lookup is loud, not silent.
        """
        # weights_only=False: our payload contains Python lists/dicts/strings
        # alongside tensors. The default safe-loader rejects this. The file
        # is a research artifact produced by ourselves; trust is implicit.
        payload = torch.load(path, map_location=map_location, weights_only=False)
        got = payload.get("version")
        if got != cls.SERIALIZATION_VERSION:
            raise ValueError(
                f"CompressedChunk serialization version mismatch: "
                f"file={got}, code={cls.SERIALIZATION_VERSION}. "
                f"Re-compress with the current code or migrate explicitly."
            )
        return cls(
            algo_id=payload["algo_id"],
            chunk_id=payload["chunk_id"],
            model_id=payload["model_id"],
            tokenizer_id=payload["tokenizer_id"],
            token_ids=list(payload["token_ids"]),
            key_cache=list(payload["key_cache"]),
            value_cache=list(payload["value_cache"]),
            importance=payload["importance"],
            is_structural=payload["is_structural"],
            compression_rate=float(payload["compression_rate"]),
            backend_state=dict(payload["backend_state"]),
        )

    def to(self, device: Any, dtype: Any = None) -> "CompressedChunk":
        """Return a copy of this chunk with all tensors moved to `device`.

        `dtype` (if given) is applied to K/V; importance stays fp32,
        is_structural stays bool.
        """
        def _move(t: torch.Tensor, target_dtype: Any = None) -> torch.Tensor:
            if target_dtype is not None and t.is_floating_point():
                return t.to(device=device, dtype=target_dtype).contiguous()
            return t.to(device=device).contiguous()

        return CompressedChunk(
            algo_id=self.algo_id,
            chunk_id=self.chunk_id,
            model_id=self.model_id,
            tokenizer_id=self.tokenizer_id,
            token_ids=list(self.token_ids),
            key_cache=[_move(k, dtype) for k in self.key_cache],
            value_cache=[_move(v, dtype) for v in self.value_cache],
            importance=_move(self.importance, torch.float32),
            is_structural=_move(self.is_structural),
            compression_rate=self.compression_rate,
            backend_state=dict(self.backend_state),
        )


def to_kvstore_entry(chunk: CompressedChunk) -> dict[str, Any]:
    """Convert a CompressedChunk into a KVStore-compatible dict.

    `KVStore` is a plain dict keyed by chunk_id, so `importance` and
    `is_structural` are carried as extra keys that `fuse_selective_compblend`
    reads.

    Layout:
        entry["K"][li] : Tensor (1, chunk_len, H_kv*D)   pre-RoPE
        entry["V"][li] : Tensor (1, chunk_len, H_kv*D)
        entry["importance"]    : Tensor (num_layers, H_kv, chunk_len)  fp32
        entry["is_structural"] : Tensor (chunk_len,)  bool
        entry["algo_id"]       : str
        entry["chunk_id"]      : str
    """
    return {
        "K": chunk.key_cache,
        "V": chunk.value_cache,
        "importance": chunk.importance,
        "is_structural": chunk.is_structural,
        "algo_id": chunk.algo_id,
        "chunk_id": chunk.chunk_id,
    }


# ──────────────────────────────────────────────────────────────────────────
# Backend protocol + abstract base
# ──────────────────────────────────────────────────────────────────────────


@runtime_checkable
class CompressionBackend(Protocol):
    """Every backend adapter implements this interface."""

    algo_id: str
    model_id: str

    def compress(
        self,
        input_ids: torch.Tensor,
        model: Any,
        budget: CompressionBudget,
    ) -> CompressedChunk: ...


class CompressionBackendBase(ABC):
    """ABC for backends. Subclasses MUST implement `compress`."""

    algo_id: str = "abstract"
    model_id: str = ""

    @abstractmethod
    def compress(
        self,
        input_ids: torch.Tensor,
        model: Any,
        budget: CompressionBudget,
    ) -> CompressedChunk: ...
