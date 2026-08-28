#!/usr/bin/env bash
#SBATCH --job-name=gsm8k-dpo-policy
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
: "${MDDM_SWEEP_VENV:?MDDM_SWEEP_VENV is required}"
: "${ML_RL_DLLM_REPO:?ML_RL_DLLM_REPO is required}"
: "${DPO_POLICY_CHECKPOINT:?DPO_POLICY_CHECKPOINT is required}"

requeue_near_timeout() {
  echo "Received Slurm USR1; DPO records/checkpoints are resumable, requesting requeue." >&2
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_near_timeout USR1

export HF_HOME="${HF_HOME:-${MDDM_SWEEP_STATE_ROOT:-${SCRATCH:-$EXPERIMENT_ROOT/.runtime}}/huggingface}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$(dirname "$DPO_POLICY_CHECKPOINT")"

srun --kill-on-bad-exit=1 "$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/train_dpo_policy.py" \
  --output-dir "$(dirname "$DPO_POLICY_CHECKPOINT")"
