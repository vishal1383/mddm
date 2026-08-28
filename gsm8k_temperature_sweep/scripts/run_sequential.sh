#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_ROOT="${MDDM_SWEEP_STATE_ROOT:-${SCRATCH:-$EXPERIMENT_ROOT/.runtime}}"
export MDDM_SWEEP_VENV="${MDDM_SWEEP_VENV:-$STATE_ROOT/venv}"
export ML_RL_DLLM_REPO="${ML_RL_DLLM_REPO:-$STATE_ROOT/ml-rl-dllm}"
export MDDM_SWEEP_OUTPUT_ROOT="${MDDM_SWEEP_OUTPUT_ROOT:-$EXPERIMENT_ROOT/final_results}"
export DPO_POLICY_CHECKPOINT="$MDDM_SWEEP_OUTPUT_ROOT/checkpoints/dpo_policy/model.safetensors"
export HF_HOME="${HF_HOME:-$STATE_ROOT/huggingface}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -x "$MDDM_SWEEP_VENV/bin/python" ]]; then
  echo "Missing experiment environment. Run scripts/bootstrap_env.sh first." >&2
  exit 2
fi
mkdir -p "$MDDM_SWEEP_OUTPUT_ROOT/logs" "$MDDM_SWEEP_OUTPUT_ROOT/manifests" "$HF_HOME"

ACTIVE_PID=""
requeue_near_timeout() {
  echo "Received Slurm USR1; stopping the active resumable stage and requeueing job $SLURM_JOB_ID." >&2
  if [[ -n "$ACTIVE_PID" ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    kill -TERM "$ACTIVE_PID" 2>/dev/null || true
    wait "$ACTIVE_PID" 2>/dev/null || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  trap requeue_near_timeout USR1
fi

run_stage() {
  "$@" &
  ACTIVE_PID=$!
  local status=0
  if wait "$ACTIVE_PID"; then
    status=0
  else
    status=$?
  fi
  ACTIVE_PID=""
  return "$status"
}

read -r BASE_MODEL_REVISION DPARALLEL_MODEL_REVISION PAPER_POLICY_REVISION < <(
  "$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/seal_model_revisions.py" \
    --output "$MDDM_SWEEP_OUTPUT_ROOT/manifests/sealed_model_revisions.json"
)
export BASE_MODEL_REVISION DPARALLEL_MODEL_REVISION PAPER_POLICY_REVISION
run_stage "$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/preflight.py"

run_eval_cell() {
  local task_id="$1"
  local status=0
  if "$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/is_cell_complete.py" \
    --output-root "$MDDM_SWEEP_OUTPUT_ROOT" --task-id "$task_id"; then
    echo "Task $task_id already complete; skipping."
    return 0
  else
    status=$?
  fi
  if [[ "$status" -ne 1 ]]; then
    echo "Task $task_id has an invalid saved result; refusing to overwrite it." >&2
    return "$status"
  fi
  run_stage "$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/evaluate.py" \
    --task-id "$task_id" --output-root "$MDDM_SWEEP_OUTPUT_ROOT"
}

echo "Stage 1/5: original 60 full-test cells on one A100, sequentially."
for task_id in $(seq 0 59); do
  run_eval_cell "$task_id"
done

echo "Stage 2/5: preserve the original 60-row table."
run_stage "$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/aggregate.py" \
  --output-root "$MDDM_SWEEP_OUTPUT_ROOT" --allow-partial --table-stem baseline_60_table

echo "Stage 3/5: full 7,473-example offline-DPO collection and training."
run_stage "$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/train_dpo_policy.py" \
  --output-dir "$(dirname "$DPO_POLICY_CHECKPOINT")"
run_stage "$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/preflight.py" --require-dpo

echo "Stage 4/5: 12 DPO full-test cells on the same A100, sequentially."
for task_id in $(seq 60 71); do
  run_eval_cell "$task_id"
done

echo "Stage 5/5: fail-closed final 72-row table."
run_stage "$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/aggregate.py" \
  --output-root "$MDDM_SWEEP_OUTPUT_ROOT" --table-stem final_table

echo "Complete: $MDDM_SWEEP_OUTPUT_ROOT/tables/final_table.md"
