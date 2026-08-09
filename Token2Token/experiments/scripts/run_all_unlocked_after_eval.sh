#!/usr/bin/env bash
# Wait for an existing evaluation runner, then start all-unlocked training.
set -euo pipefail

EVAL_RUNNER_PID="${1:?pass the container evaluation runner PID}"
NAME="${NAME:-t095_full_retry1}"

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

while kill -0 "$EVAL_RUNNER_PID" 2>/dev/null; do
  printf 'waiting_for_eval pid=%s time=%s\n' \
    "$EVAL_RUNNER_PID" "$(date --iso-8601=seconds)"
  sleep 60
done

printf 'evaluation_finished pid=%s time=%s; starting_training=%s\n' \
  "$EVAL_RUNNER_PID" "$(date --iso-8601=seconds)" "$NAME"
exec env NAME="$NAME" MAX_STEPS=7472 bash Token2Token/experiments/scripts/run_all_unlocked_v1.sh
