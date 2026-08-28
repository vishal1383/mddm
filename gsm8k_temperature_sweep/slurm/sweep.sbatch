#!/usr/bin/env bash
#SBATCH --job-name=gsm8k-passk
#SBATCH --partition=gpu-preempt
#SBATCH --qos=long
#SBATCH --time=14-00:00:00
#SBATCH --gpus=1
#SBATCH --constraint=bf16&vram40
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array job}"
: "${MDDM_SWEEP_VENV:?Run scripts/bootstrap_env.sh and export MDDM_SWEEP_VENV}"
: "${ML_RL_DLLM_REPO:?Run scripts/bootstrap_env.sh and export ML_RL_DLLM_REPO}"
: "${MDDM_SWEEP_OUTPUT_ROOT:?Set a durable shared result directory}"

export HF_HOME="${HF_HOME:-${MDDM_SWEEP_STATE_ROOT:-${SCRATCH:-$EXPERIMENT_ROOT/.runtime}}/huggingface}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$MDDM_SWEEP_OUTPUT_ROOT"

requeue_near_timeout() {
  echo "Received Slurm USR1; records are already atomic, requesting requeue." >&2
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_near_timeout USR1

"$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/preflight.py"
srun --kill-on-bad-exit=1 "$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/evaluate.py" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --output-root "$MDDM_SWEEP_OUTPUT_ROOT"
