#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-outputsig}"
OUT_ROOT="${OUT_ROOT:-$ROOT/full_probe_ig}"
REPORT="${REPORT:-$ROOT/reports/mdm_spread_anchor_probe_report_ig.docx}"
LIMIT="${LIMIT:-20}"
MAX_K="${MAX_K:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"

if [[ -e "$ROOT" && "${FORCE:-0}" != "1" ]]; then
  echo "$ROOT already exists. Use FORCE=1 to replace/add files there." >&2
  exit 1
fi

mkdir -p "$ROOT"

OUT_ROOT="$OUT_ROOT" \
LIMIT="$LIMIT" \
MAX_K="$MAX_K" \
BATCH_SIZE="$BATCH_SIZE" \
./run_full_probe.sh

for run in \
  "$OUT_ROOT/llada-8b_gsm8k" \
  "$OUT_ROOT/llada-8b_humaneval" \
  "$OUT_ROOT/dream-7b_gsm8k" \
  "$OUT_ROOT/dream-7b_humaneval"
do
  python3 plot_cached_layout.py "$run"
done

python3 make_report.py \
  --root "$OUT_ROOT" \
  --out "$REPORT"

echo "IG experiment saved under $ROOT"
echo "Report: $REPORT"
