#!/usr/bin/env bash
# Train all cached states from 500 records, then calibrate on held-out train rows.
set -euo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

ROOT="${ROOT:-outputs/token2token/all_states_confidence_v2}"
NAME="${NAME:-smoke500_tau70}"
TARGETS_FILE="${TARGETS_FILE:-outputs/token2token/threshold_unlock/gsm8k_train_t095_gain_text_q07_max512.jsonl}"
TRAIN_RECORDS="${TRAIN_RECORDS:-500}"
CALIBRATION_START="${CALIBRATION_START:-7000}"
CALIBRATION_LIMIT="${CALIBRATION_LIMIT:-100}"
BATCH="${BATCH:-16}"

TRAIN_DIR="$ROOT/$NAME/train"
BASE_DIR="$ROOT/calibration/base_t095"
TRAINED_DIR="$ROOT/$NAME/calibration/adapter-final"
REPORT_DIR="$ROOT/$NAME/report"
mkdir -p "$TRAIN_DIR" "$BASE_DIR" "$TRAINED_DIR" "$REPORT_DIR"

if [[ ! -f "$TRAIN_DIR/adapter-final/adapter_config.json" ]]; then
  python3 -m Token2Token.experiments.train_all_states_confidence \
    --targets-file "$TARGETS_FILE" \
    --record-limit "$TRAIN_RECORDS" \
    --epochs 1 \
    --completion-length 128 \
    --states-per-update 8 \
    --target-confidence 0.70 \
    --anchor-ce-weight 1.0 \
    --unlock-ce-weight 1.0 \
    --confidence-margin-weight 1.0 \
    --selection-loss-weight 0.25 \
    --preserve-kl-weight 5.0 \
    --learning-rate 1e-5 \
    --save-every 250 \
    --output-dir "$TRAIN_DIR" \
    2>&1 | tee "$ROOT/$NAME/train.log"
fi

DECODER_ARGS=(
  --dataset-split train
  --start-index "$CALIBRATION_START"
  --completion-length 128
  --batch-size "$BATCH"
  --limit "$CALIBRATION_LIMIT"
  --commit-threshold-on-first-forward
  --no-unlock-forward
  --catalyst-filter text
  --force-catalyst always
)

if [[ ! -f "$BASE_DIR/summary.json" ]]; then
  python3 -m Token2Token.main.eval_threshold_gsm8k \
    --model-label base_llada8b_calibration_t095 \
    --thresholds 0.95 \
    --output-dir "$BASE_DIR" \
    "${DECODER_ARGS[@]}" \
    2>&1 | tee "$BASE_DIR/eval.log"
fi

if [[ ! -f "$TRAINED_DIR/summary.json" ]]; then
  python3 -m Token2Token.main.eval_threshold_gsm8k \
    --adapter-path "$TRAIN_DIR/adapter-final" \
    --model-label all_states_v2_smoke500 \
    --thresholds 0.70,0.80,0.90,0.95 \
    --output-dir "$TRAINED_DIR" \
    "${DECODER_ARGS[@]}" \
    2>&1 | tee "$TRAINED_DIR/eval.log"
fi

python3 -m Token2Token.experiments.select_threshold_operating_point \
  --baseline-summary "$BASE_DIR/summary.json" \
  --trained-summary "$TRAINED_DIR/summary.json" \
  --baseline-threshold 0.95 \
  --accuracy-tolerance 0.02 \
  --minimum-tokens-per-forward 4.0 \
  --output "$REPORT_DIR/selection.json" \
  2>&1 | tee "$REPORT_DIR/selection.md"
