#!/usr/bin/env bash
#SBATCH --job-name=gsm8k-passk-a100
#SBATCH --partition=gpu-preempt
#SBATCH --time=45:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=final_results/manifests/slurm-%j.out
#SBATCH --error=final_results/manifests/slurm-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export MDDM_SWEEP_STATE_ROOT="${MDDM_SWEEP_STATE_ROOT:-${SCRATCH:-$EXPERIMENT_ROOT/.runtime}}"
export MDDM_SWEEP_OUTPUT_ROOT="${MDDM_SWEEP_OUTPUT_ROOT:-$EXPERIMENT_ROOT/final_results}"
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

bash "$EXPERIMENT_ROOT/scripts/bootstrap_env.sh"
export MDDM_SWEEP_VENV="$MDDM_SWEEP_STATE_ROOT/venv"
export ML_RL_DLLM_REPO="$MDDM_SWEEP_STATE_ROOT/ml-rl-dllm"

exec bash "$EXPERIMENT_ROOT/scripts/run_sequential.sh"
