#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

ROOT="${ROOT:-outputs/token2token/full_k123_1319}"
LIMIT="${LIMIT:-1319}"
BATCH="${BATCH:-8}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
COMPLETION_LENGTH="${COMPLETION_LENGTH:-128}"
LOOKAHEAD_ADAPTER="${LOOKAHEAD_ADAPTER:-outputs/token2token/online_lookahead_v6/k2_select1_train7473_step7473/train/checkpoint-004000}"
STANDARD_ADAPTER="${STANDARD_ADAPTER:-outputs/token2token/standard_lora_gsm8k_full_run1/adapter-final}"

mkdir -p "$ROOT"

seed_predictions() {
  local source="$1"
  local destination="$2"
  if [[ -f "$source" && ! -f "$destination/predictions_t0p95.jsonl" ]]; then
    mkdir -p "$destination"
    cp "$source" "$destination/predictions_t0p95.jsonl"
  fi
}

evaluate() {
  local family="$1"
  local label="$2"
  local adapter="$3"
  local k="$4"
  local output="$ROOT/$family/k$k"
  local adapter_args=()
  if [[ -n "$adapter" ]]; then
    adapter_args=(--adapter-path "$adapter")
  fi
  mkdir -p "$output"
  python3 -m Token2Token.main.eval_threshold_gsm8k \
    "${adapter_args[@]}" \
    --model-label "$label" \
    --decoder topk \
    --tokens-per-step "$k" \
    --block-length "$BLOCK_LENGTH" \
    --completion-length "$COMPLETION_LENGTH" \
    --batch-size "$BATCH" \
    --limit "$LIMIT" \
    --resume \
    --output-dir "$output" \
    2>&1 | tee -a "$output/eval.log"
}

# Reuse predictions from matched earlier runs. The evaluator resumes by
# example ID and recomputes the aggregate over old plus newly appended rows.
seed_predictions \
  outputs/token2token/online_lookahead_v6/validation500/base_k1/predictions_t0p95.jsonl \
  "$ROOT/base/k1"
seed_predictions \
  outputs/token2token/online_lookahead_v6/base50/block32_k2/predictions_t0p95.jsonl \
  "$ROOT/base/k2"
seed_predictions \
  outputs/token2token/online_lookahead_v6/validation200/base_k3/predictions_t0p95.jsonl \
  "$ROOT/base/k3"
seed_predictions \
  outputs/token2token/online_lookahead_v6/k2_select1_train7473_step7473/validation500/checkpoint-004000/predictions_t0p95.jsonl \
  "$ROOT/lookahead_lora/k2"

# Run the headline comparison first so it becomes available before the rest
# of the ablation matrix.
evaluate lookahead_lora lookahead_lora_checkpoint4000_block32_k2 "$LOOKAHEAD_ADAPTER" 2
evaluate base base_llada8b_block32_k1 "" 1

python3 -m Token2Token.main.paired_comparison \
  --baseline-predictions "$ROOT/base/k1/predictions_t0p95.jsonl" \
  --trained-predictions "$ROOT/lookahead_lora/k2/predictions_t0p95.jsonl" \
  --baseline-label base_k1 \
  --trained-label lookahead_lora_k2 \
  --output "$ROOT/lookahead_k2_vs_base_k1.md"

evaluate base base_llada8b_block32_k2 "" 2
evaluate base base_llada8b_block32_k3 "" 3
evaluate lookahead_lora lookahead_lora_checkpoint4000_block32_k1 "$LOOKAHEAD_ADAPTER" 1
evaluate lookahead_lora lookahead_lora_checkpoint4000_block32_k3 "$LOOKAHEAD_ADAPTER" 3
evaluate standard_lora standard_lora_block32_k1 "$STANDARD_ADAPTER" 1
evaluate standard_lora standard_lora_block32_k2 "$STANDARD_ADAPTER" 2
evaluate standard_lora standard_lora_block32_k3 "$STANDARD_ADAPTER" 3

python3 -m Token2Token.experiments.summarize_k_matrix \
  --root "$ROOT" \
  --output "$ROOT/results.md"
