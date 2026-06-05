#!/usr/bin/env bash
# scripts/run_musique_kvzip.sh — run the MuSiQue KVzip token-prune benchmark
# and save results, on a GPU pod where the stack is already installed (see README
# "Install (GPU pod)"). Thin wrapper around benchmarks/musique/blend_musique_generic_kvzip.py.
#
# Usage:
#   export HF_TOKEN=...                      # Llama-3.1-8B-Instruct is gated
#   bash scripts/run_musique_kvzip.sh        # full default grid, N=150
#   COMPBLEND_MUSIQUE_N=3 COMPBLEND_ARMS=only_hkvd,importance_only \
#       bash scripts/run_musique_kvzip.sh    # quick smoke
#
# All knobs are env-overridable (defaults shown). Results: $OUT_JSON + a stdout log.
set -euo pipefail

# Repo root = parent of this script's dir.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

# KVzip submodule must be importable.
export PYTHONPATH="$REPO/src/external/KVzip:${PYTHONPATH:-}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

# ── Knobs (override via env) ────────────────────────────────────────────────
export CACHEBLEND_MODEL="${CACHEBLEND_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export COMPBLEND_KVZIP_RATIOS="${COMPBLEND_KVZIP_RATIOS:-0.5,0.4,0.3,0.2,0.1}"
export COMPBLEND_RECOMP_RATIOS="${COMPBLEND_RECOMP_RATIOS:-0.2,0.15,0.1,0.05}"
# COMPBLEND_ARMS unset → run all arms; set to a CSV subset to filter.
[ -n "${COMPBLEND_ARMS:-}" ] && export COMPBLEND_ARMS

STAMP="$(date +%Y%m%d_%H%M%S 2>/dev/null || echo run)"
OUT_DIR="$REPO/logs"
mkdir -p "$OUT_DIR"
export COMPBLEND_OUT="${COMPBLEND_OUT:-$OUT_DIR/musique_kvzip_${STAMP}.json}"
LOG="$OUT_DIR/musique_kvzip_${STAMP}.log"

echo "[run] model=$CACHEBLEND_MODEL"
echo "[run] kvzip_ratios=$COMPBLEND_KVZIP_RATIOS recomp_ratios=$COMPBLEND_RECOMP_RATIOS arms=${COMPBLEND_ARMS:-<all>}"
echo "[run] N=${CACHEBLEND_MUSIQUE_N:-150}  →  json=$COMPBLEND_OUT  log=$LOG"

# Run; tee stdout+stderr to the log so progress is captured.
python benchmarks/musique/blend_musique_generic_kvzip.py 2>&1 | tee "$LOG"

echo "[run] DONE → results: $COMPBLEND_OUT"
