#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_ROOT="${MDDM_SWEEP_STATE_ROOT:-${SCRATCH:-$EXPERIMENT_ROOT/.runtime}}"
export MDDM_SWEEP_VENV="${MDDM_SWEEP_VENV:-$STATE_ROOT/venv}"
export ML_RL_DLLM_REPO="${ML_RL_DLLM_REPO:-$STATE_ROOT/ml-rl-dllm}"
export MDDM_SWEEP_OUTPUT_ROOT="${MDDM_SWEEP_OUTPUT_ROOT:-$EXPERIMENT_ROOT/results}"
export DPO_POLICY_CHECKPOINT="$MDDM_SWEEP_OUTPUT_ROOT/checkpoints/dpo_policy/model.safetensors"

if [[ ! -x "$MDDM_SWEEP_VENV/bin/python" ]]; then
  echo "Missing experiment environment. Run: bash scripts/bootstrap_env.sh" >&2
  exit 2
fi
mkdir -p "$MDDM_SWEEP_OUTPUT_ROOT/logs"
"$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/preflight.py"
read -r BASE_MODEL_REVISION DPARALLEL_MODEL_REVISION < <(
  "$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/seal_model_revisions.py"
)
export BASE_MODEL_REVISION DPARALLEL_MODEL_REVISION

ARRAY_LIMIT="${MDDM_SWEEP_ARRAY_LIMIT:-1}"
BASELINE_ARRAY_JOB_ID="$(sbatch --parsable \
  --array="0-59%$ARRAY_LIMIT" \
  --gpus="${MDDM_SWEEP_GPUS:-a100:1}" \
  --output="$MDDM_SWEEP_OUTPUT_ROOT/logs/%A_%a.out" \
  --error="$MDDM_SWEEP_OUTPUT_ROOT/logs/%A_%a.err" \
  --export=ALL \
  "$EXPERIMENT_ROOT/slurm/sweep.sbatch")"
BASELINE_TABLE_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:$BASELINE_ARRAY_JOB_ID" \
  --output="$MDDM_SWEEP_OUTPUT_ROOT/logs/table_baseline_%j.out" \
  --error="$MDDM_SWEEP_OUTPUT_ROOT/logs/table_baseline_%j.err" \
  --export=ALL,MDDM_SWEEP_ALLOW_PARTIAL=1 \
  "$EXPERIMENT_ROOT/slurm/aggregate.sbatch")"
DPO_TRAIN_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:$BASELINE_TABLE_JOB_ID" \
  --gpus="${MDDM_SWEEP_GPUS:-a100:1}" \
  --output="$MDDM_SWEEP_OUTPUT_ROOT/logs/dpo_train_%j.out" \
  --error="$MDDM_SWEEP_OUTPUT_ROOT/logs/dpo_train_%j.err" \
  --export=ALL \
  "$EXPERIMENT_ROOT/slurm/train_dpo.sbatch")"
DPO_ARRAY_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:$DPO_TRAIN_JOB_ID" \
  --array="60-71%$ARRAY_LIMIT" \
  --gpus="${MDDM_SWEEP_GPUS:-a100:1}" \
  --output="$MDDM_SWEEP_OUTPUT_ROOT/logs/%A_%a.out" \
  --error="$MDDM_SWEEP_OUTPUT_ROOT/logs/%A_%a.err" \
  --export=ALL \
  "$EXPERIMENT_ROOT/slurm/sweep.sbatch")"
FINAL_TABLE_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:$DPO_ARRAY_JOB_ID" \
  --output="$MDDM_SWEEP_OUTPUT_ROOT/logs/table_final_%j.out" \
  --error="$MDDM_SWEEP_OUTPUT_ROOT/logs/table_final_%j.err" \
  --export=ALL,MDDM_SWEEP_ALLOW_PARTIAL=0 \
  "$EXPERIMENT_ROOT/slurm/aggregate.sbatch")"

echo "Submitted original 60-cell sweep: $BASELINE_ARRAY_JOB_ID"
echo "Submitted original 60-row table: $BASELINE_TABLE_JOB_ID"
echo "Submitted full-train DPO policy after baselines: $DPO_TRAIN_JOB_ID"
echo "Submitted 12-cell full-test DPO sweep: $DPO_ARRAY_JOB_ID"
echo "Submitted fail-closed final 72-row table: $FINAL_TABLE_JOB_ID"
echo "Results: $MDDM_SWEEP_OUTPUT_ROOT"
