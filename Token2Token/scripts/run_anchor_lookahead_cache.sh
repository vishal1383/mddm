#!/usr/bin/env bash
set -euo pipefail

EXAMPLES="${EXAMPLES:-7473}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-64}"
TARGETS_FILE="${TARGETS_FILE:-outputs/token2token/anchor_lookahead/cache/gsm8k_train_t095_gain_text_q07_max512.jsonl}"

mkdir -p "$(dirname "$TARGETS_FILE")"

PYTHONUNBUFFERED=1 python3 -m Token2Token.main.precompute_threshold_unlock_targets \
  --dataset gsm8k \
  --examples "$EXAMPLES" \
  --confidence-threshold .95 \
  --candidate-prob-ratio .70 \
  --candidate-batch-size "$CANDIDATE_BATCH_SIZE" \
  --max-completion-tokens 512 \
  --resume \
  --output "$TARGETS_FILE"
