#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

ROOT="${ROOT:-outputs/token2token/threshold_lookahead_v7/full_train7473_t090_num099/normal_adaptive_t095_1319}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LIMIT="${LIMIT:-1319}"

evaluate_model() {
  local label="$1"
  local adapter_path="$2"
  local output_dir="$3"

  if [[ -f "$output_dir/summary.json" ]]; then
    echo "already complete: $output_dir"
    return
  fi

  mkdir -p "$output_dir"
  local model_args=()
  if [[ -n "$adapter_path" ]]; then
    model_args+=(--adapter-path "$adapter_path" --merge-adapter)
  fi
  python3 -m Token2Token.eval_threshold_gsm8k \
    --model-label "$label" \
    "${model_args[@]}" \
    --output-dir "$output_dir" \
    --dataset-split test \
    --completion-length 128 \
    --batch-size "$BATCH_SIZE" \
    --limit "$LIMIT" \
    --resume \
    --thresholds 0.95 \
    --commit-threshold-on-first-forward \
    --no-unlock-forward \
    --catalyst-filter any \
    --force-catalyst when-empty \
    2>&1 | tee "$output_dir/eval.log"
}

evaluate_model \
  base_llada8b_normal_adaptive_t095 \
  "" \
  "$ROOT/base"

evaluate_model \
  standard_lora_merged_normal_adaptive_t095 \
  outputs/token2token/standard_lora_gsm8k_full_run1/adapter-final \
  "$ROOT/standard_lora_merged"

evaluate_model \
  threshold_lookahead_checkpoint6000_merged_normal_adaptive_t095 \
  outputs/token2token/threshold_lookahead_v7/full_train7473_t090_num099/train/checkpoint-006000 \
  "$ROOT/lookahead_checkpoint6000_merged"
