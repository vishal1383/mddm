#!/usr/bin/env bash
#SBATCH --job-name=gsm8k-passk-submit
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=final_results/manifests/slurm-%j.out
#SBATCH --error=final_results/manifests/slurm-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export MDDM_SWEEP_STATE_ROOT="${MDDM_SWEEP_STATE_ROOT:-${SCRATCH:-$EXPERIMENT_ROOT/.runtime}}"
export MDDM_SWEEP_OUTPUT_ROOT="${MDDM_SWEEP_OUTPUT_ROOT:-$EXPERIMENT_ROOT/final_results}"
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

: "${SFT_ADAPTER_PATH:?Export SFT_ADAPTER_PATH before sbatch}"
: "${UNMASKING_POLICY_CHECKPOINT:?Export UNMASKING_POLICY_CHECKPOINT before sbatch}"
: "${HF_TOKEN:?Export HF_TOKEN before sbatch}"

bash "$EXPERIMENT_ROOT/scripts/bootstrap_env.sh"
export MDDM_SWEEP_VENV="$MDDM_SWEEP_STATE_ROOT/venv"
export ML_RL_DLLM_REPO="$MDDM_SWEEP_STATE_ROOT/ml-rl-dllm"

bash "$EXPERIMENT_ROOT/scripts/submit.sh"
