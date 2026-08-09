#!/usr/bin/env bash
# Run one bounded train/eval trial against a fixed paired calibration slice.
set -euo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

ROOT="${ROOT:-outputs/token2token/all_states_confidence_v2/micro_trials}"
NAME="${NAME:-tau70_margin1_sel025_kl5}"
TARGETS_FILE="${TARGETS_FILE:-outputs/token2token/threshold_unlock/gsm8k_train_t095_gain_text_q07_max512.jsonl}"
TRAIN_RECORDS="${TRAIN_RECORDS:-500}"
TRAIN_RECORD_START="${TRAIN_RECORD_START:-0}"
MAX_STEPS="${MAX_STEPS:-180}"
STATES_PER_UPDATE="${STATES_PER_UPDATE:-8}"
MIN_UNLOCKED_TARGETS="${MIN_UNLOCKED_TARGETS:-0}"
MIN_SELECTION_SCORE="${MIN_SELECTION_SCORE:-none}"
CATALYST_CORRECT_ONLY="${CATALYST_CORRECT_ONLY:-false}"
MAX_STATES_PER_RECORD="${MAX_STATES_PER_RECORD:-0}"
EPOCHS="${EPOCHS:-1}"
CALIBRATION_START="${CALIBRATION_START:-7000}"
CALIBRATION_LIMIT="${CALIBRATION_LIMIT:-32}"
CALIBRATION_SPLIT="${CALIBRATION_SPLIT:-train}"
THRESHOLD="${THRESHOLD:-0.70}"
BATCH="${BATCH:-16}"
BUDGET_SECONDS="${BUDGET_SECONDS:-1180}"
CATALYST_FILTER="${CATALYST_FILTER:-text}"
CATALYST_MIN_LENGTH="${CATALYST_MIN_LENGTH:-0}"
FORCE_CATALYST="${FORCE_CATALYST:-always}"
NUMERIC_THRESHOLD="${NUMERIC_THRESHOLD:-none}"

TARGET_CONFIDENCE="${TARGET_CONFIDENCE:-0.70}"
ANCHOR_CE_WEIGHT="${ANCHOR_CE_WEIGHT:-1.0}"
UNLOCK_CE_WEIGHT="${UNLOCK_CE_WEIGHT:-1.0}"
CORRECTION_CE_WEIGHT="${CORRECTION_CE_WEIGHT:-0.0}"
CONFIDENCE_MARGIN_WEIGHT="${CONFIDENCE_MARGIN_WEIGHT:-1.0}"
CONFIDENCE_MARGIN_KIND="${CONFIDENCE_MARGIN_KIND:-unlocked}"
SELECTION_LOSS_WEIGHT="${SELECTION_LOSS_WEIGHT:-0.25}"
SELECTION_TARGETS="${SELECTION_TARGETS:-catalyst}"
PRESERVE_KL_WEIGHT="${PRESERVE_KL_WEIGHT:-5.0}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,k_proj,v_proj,attn_out}"
export NAME TRAIN_RECORDS TRAIN_RECORD_START MAX_STEPS CALIBRATION_START CALIBRATION_LIMIT
export CALIBRATION_SPLIT
export STATES_PER_UPDATE MIN_UNLOCKED_TARGETS MAX_STATES_PER_RECORD
export MIN_SELECTION_SCORE EPOCHS
export CATALYST_CORRECT_ONLY
export THRESHOLD BUDGET_SECONDS TARGET_CONFIDENCE ANCHOR_CE_WEIGHT
export CATALYST_FILTER CATALYST_MIN_LENGTH FORCE_CATALYST
export NUMERIC_THRESHOLD
export UNLOCK_CE_WEIGHT CONFIDENCE_MARGIN_WEIGHT SELECTION_LOSS_WEIGHT
export CORRECTION_CE_WEIGHT
export CONFIDENCE_MARGIN_KIND SELECTION_TARGETS
export PRESERVE_KL_WEIGHT LEARNING_RATE
export LORA_RANK LORA_ALPHA LORA_DROPOUT LORA_TARGETS
export ADAPTER_PATH="${ADAPTER_PATH:-}"

threshold_tag="$(python3 -c "from Token2Token.eval_threshold_gsm8k import threshold_tag; print(threshold_tag(float('$THRESHOLD')))" )"
numeric_tag="${NUMERIC_THRESHOLD//./p}"
decoder_tag="${CATALYST_FILTER}_${FORCE_CATALYST}_m${CATALYST_MIN_LENGTH}_num${numeric_tag}"
BASE_DIR="${BASE_DIR:-$ROOT/base_${threshold_tag}_${decoder_tag}_${CALIBRATION_SPLIT}_s${CALIBRATION_START}_n${CALIBRATION_LIMIT}}"
BASE_PREDICTIONS="${BASE_PREDICTIONS:-$BASE_DIR/predictions_${threshold_tag}.jsonl}"

TRIAL_DIR="$ROOT/$NAME"
TRAIN_DIR="$TRIAL_DIR/train"
EVAL_DIR="$TRIAL_DIR/eval"
REPORT_DIR="$TRIAL_DIR/report"
mkdir -p "$BASE_DIR" "$TRAIN_DIR" "$EVAL_DIR" "$REPORT_DIR"

started=$SECONDS
run_bounded() {
  local remaining=$((BUDGET_SECONDS - (SECONDS - started)))
  if (( remaining <= 0 )); then
    echo "trial budget exhausted before: $*" >&2
    return 124
  fi
  timeout --signal=INT --kill-after=30s "${remaining}s" "$@"
}

DECODER_ARGS=(
  --thresholds "$THRESHOLD"
  --dataset-split "$CALIBRATION_SPLIT"
  --start-index "$CALIBRATION_START"
  --completion-length 128
  --batch-size "$BATCH"
  --limit "$CALIBRATION_LIMIT"
  --commit-threshold-on-first-forward
  --no-unlock-forward
  --catalyst-filter "$CATALYST_FILTER"
  --catalyst-min-length "$CATALYST_MIN_LENGTH"
  --force-catalyst "$FORCE_CATALYST"
)
if [[ "$NUMERIC_THRESHOLD" != "none" ]]; then
  DECODER_ARGS+=(--numeric-threshold "$NUMERIC_THRESHOLD")
fi

if [[ ! -f "$BASE_PREDICTIONS" ]]; then
  run_bounded python3 -m Token2Token.eval_threshold_gsm8k \
    --model-label "base_${threshold_tag}" \
    --output-dir "$BASE_DIR" \
    "${DECODER_ARGS[@]}" \
    2>&1 | tee "$BASE_DIR/eval.log"
fi

adapter_path="$ADAPTER_PATH"
if [[ -z "$adapter_path" ]]; then
  adapter_path="$TRAIN_DIR/adapter-final"
  if [[ ! -f "$adapter_path/adapter_config.json" ]]; then
    stage_args=(--min-unlocked-targets "$MIN_UNLOCKED_TARGETS")
    if [[ "$MIN_SELECTION_SCORE" != "none" ]]; then
      stage_args+=(--min-selection-score "$MIN_SELECTION_SCORE")
    fi
    if [[ "$CATALYST_CORRECT_ONLY" == "true" ]]; then
      stage_args+=(--catalyst-correct-only)
    fi
    if (( MAX_STATES_PER_RECORD > 0 )); then
      stage_args+=(--max-states-per-record "$MAX_STATES_PER_RECORD")
    fi
    run_bounded python3 -m Token2Token.train_all_states_confidence \
      --targets-file "$TARGETS_FILE" \
      --record-start "$TRAIN_RECORD_START" \
      --record-limit "$TRAIN_RECORDS" \
      --epochs "$EPOCHS" \
      --max-steps "$MAX_STEPS" \
      --completion-length 128 \
      --states-per-update "$STATES_PER_UPDATE" \
      "${stage_args[@]}" \
      --target-confidence "$TARGET_CONFIDENCE" \
      --anchor-ce-weight "$ANCHOR_CE_WEIGHT" \
      --unlock-ce-weight "$UNLOCK_CE_WEIGHT" \
      --correction-ce-weight "$CORRECTION_CE_WEIGHT" \
      --confidence-margin-weight "$CONFIDENCE_MARGIN_WEIGHT" \
      --confidence-margin-kind "$CONFIDENCE_MARGIN_KIND" \
      --selection-loss-weight "$SELECTION_LOSS_WEIGHT" \
      --selection-targets "$SELECTION_TARGETS" \
      --preserve-kl-weight "$PRESERVE_KL_WEIGHT" \
      --learning-rate "$LEARNING_RATE" \
      --lora-rank "$LORA_RANK" \
      --lora-alpha "$LORA_ALPHA" \
      --lora-dropout "$LORA_DROPOUT" \
      --lora-targets "$LORA_TARGETS" \
      --save-every 0 \
      --output-dir "$TRAIN_DIR" \
      2>&1 | tee "$TRIAL_DIR/train.log"
  fi
fi

predictions="$EVAL_DIR/predictions_${threshold_tag}.jsonl"
if [[ ! -f "$predictions" ]]; then
  run_bounded python3 -m Token2Token.eval_threshold_gsm8k \
    --adapter-path "$adapter_path" \
    --model-label "$NAME" \
    --output-dir "$EVAL_DIR" \
    "${DECODER_ARGS[@]}" \
    2>&1 | tee "$TRIAL_DIR/eval.log"
fi

gate_args=(
  --baseline-predictions "$BASE_PREDICTIONS" \
  --trained-predictions "$predictions" \
  --min-example-id "$CALIBRATION_START" \
  --max-example-id "$((CALIBRATION_START + CALIBRATION_LIMIT))" \
  --expected-threshold "$THRESHOLD" \
  --minimum-tokens-per-forward 4.0 \
  --minimum-tokens-per-forward-gain 0.1 \
  --max-correct-loss 1 \
  --output "$REPORT_DIR/gate.json"
)
if [[ "$NUMERIC_THRESHOLD" != "none" ]]; then
  gate_args+=(--expected-numeric-threshold "$NUMERIC_THRESHOLD")
fi
python3 -m Token2Token.micro_trial_gate "${gate_args[@]}" \
  | tee "$REPORT_DIR/gate.log"

python3 - "$TRIAL_DIR/manifest.json" <<'PY'
import json
import os
import sys
from pathlib import Path

keys = (
    "NAME", "TRAIN_RECORDS", "TRAIN_RECORD_START", "MAX_STEPS", "STATES_PER_UPDATE", "EPOCHS",
    "MIN_UNLOCKED_TARGETS", "MIN_SELECTION_SCORE", "MAX_STATES_PER_RECORD",
    "CATALYST_CORRECT_ONLY",
    "CALIBRATION_START", "CALIBRATION_SPLIT",
    "CALIBRATION_LIMIT", "THRESHOLD", "BUDGET_SECONDS", "ADAPTER_PATH",
    "CATALYST_FILTER", "CATALYST_MIN_LENGTH", "FORCE_CATALYST",
    "NUMERIC_THRESHOLD",
    "TARGET_CONFIDENCE", "ANCHOR_CE_WEIGHT", "UNLOCK_CE_WEIGHT",
    "CORRECTION_CE_WEIGHT",
    "CONFIDENCE_MARGIN_WEIGHT", "CONFIDENCE_MARGIN_KIND",
    "SELECTION_LOSS_WEIGHT", "SELECTION_TARGETS",
    "PRESERVE_KL_WEIGHT", "LEARNING_RATE",
    "LORA_RANK", "LORA_ALPHA", "LORA_DROPOUT", "LORA_TARGETS",
)
Path(sys.argv[1]).write_text(
    json.dumps({key: os.environ.get(key) for key in keys}, indent=2) + "\n",
    encoding="utf-8",
)
PY

echo "trial_seconds=$((SECONDS - started))"
