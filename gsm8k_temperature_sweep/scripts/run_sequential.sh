#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_ROOT="${MDDM_SWEEP_STATE_ROOT:-${SCRATCH:-$EXPERIMENT_ROOT/.runtime}}"
export MDDM_SWEEP_VENV="${MDDM_SWEEP_VENV:-$STATE_ROOT/venv}"
export ML_RL_DLLM_REPO="${ML_RL_DLLM_REPO:-$STATE_ROOT/ml-rl-dllm-35e4830485f1}"
export MDDM_SWEEP_OUTPUT_ROOT="${MDDM_SWEEP_OUTPUT_ROOT:-$EXPERIMENT_ROOT/final_results}"
export DPO_POLICY_CHECKPOINT="$MDDM_SWEEP_OUTPUT_ROOT/checkpoints/dpo_policy_v3/model.safetensors"
export APPLE_POLICY_CHECKPOINT="$MDDM_SWEEP_OUTPUT_ROOT/checkpoints/apple_policy_rl/checkpoint-best/model.safetensors"
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
  if [[ -n "${SLURM_JOB_ID:-}" && ( "$status" -eq 120 || "$status" -eq 143 ) ]]; then
    echo "Resumable stage exited with status $status; requeueing job $SLURM_JOB_ID." >&2
    if scontrol requeue "$SLURM_JOB_ID"; then
      exit 0
    fi
    echo "Failed to requeue job $SLURM_JOB_ID." >&2
  fi
  return "$status"
}

read -r BASE_MODEL_REVISION DPARALLEL_MODEL_REVISION JUSTGRPO_MODEL_REVISION < <(
  "$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/seal_model_revisions.py" \
    --output "$MDDM_SWEEP_OUTPUT_ROOT/manifests/sealed_model_revisions_v3.json"
)
export BASE_MODEL_REVISION DPARALLEL_MODEL_REVISION JUSTGRPO_MODEL_REVISION
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

echo "Stage 1/7: JustGRPO checkpoint first, all 4 full-test temperatures."
for task_id in $(seq 12 15); do
  run_eval_cell "$task_id"
done

echo "Stage 2/7: train Apple's official BL32 alpha=0.3 GRPO policy on the full paper mixture."
run_stage "$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/train_apple_policy.py" \
  --apple-root "$ML_RL_DLLM_REPO" \
  --config "$EXPERIMENT_ROOT/configs/apple_llada_bl32_alpha03.yaml" \
  --output-dir "$MDDM_SWEEP_OUTPUT_ROOT/checkpoints/apple_policy_rl" \
  --base-revision "$BASE_MODEL_REVISION"
run_stage "$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/preflight.py" --require-apple

echo "Stage 3/7: evaluate the saved Apple GRPO policy at all 4 full-test temperatures."
for task_id in $(seq 20 23); do
  run_eval_cell "$task_id"
done

echo "Stage 4/7: finish/reuse Base, JSD, dParallel, and LoRA cells."
for task_id in $(seq 0 11) $(seq 16 19); do
  run_eval_cell "$task_id"
done

echo "Stage 5/7: preserve the 24-row non-DPO table."
run_stage "$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/aggregate.py" \
  --output-root "$MDDM_SWEEP_OUTPUT_ROOT" --allow-partial --table-stem baseline_table

echo "Stage 6/7: full 7,473-example online hidden-state trajectory-DPO training and evaluation."
run_stage "$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/train_dpo_policy.py" \
  --output-dir "$(dirname "$DPO_POLICY_CHECKPOINT")"
run_stage "$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/preflight.py" --require-dpo
for task_id in $(seq 24 27); do
  run_eval_cell "$task_id"
done

echo "Stage 7/7: fail-closed final 28-row table."
run_stage "$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/aggregate.py" \
  --output-root "$MDDM_SWEEP_OUTPUT_ROOT" --table-stem final_table

echo "Complete: $MDDM_SWEEP_OUTPUT_ROOT/tables/final_table.md"
