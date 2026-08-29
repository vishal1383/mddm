#!/usr/bin/env bash
#SBATCH --job-name=gsm8k-passk-a100
#SBATCH --partition=gpu-preempt
#SBATCH --time=45:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=gsm8k_temperature_sweep/final_results/manifests/slurm-%j.out
#SBATCH --error=gsm8k_temperature_sweep/final_results/manifests/slurm-%j.err

set -euo pipefail

cd gsm8k_temperature_sweep
export MDDM_SWEEP_STATE_ROOT="${MDDM_SWEEP_STATE_ROOT:-$PWD/.runtime}"
export MDDM_SWEEP_OUTPUT_ROOT="${MDDM_SWEEP_OUTPUT_ROOT:-$PWD/final_results}"
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
export PIP_CACHE_DIR="$MDDM_SWEEP_STATE_ROOT/pip-cache"
export TMPDIR="$MDDM_SWEEP_STATE_ROOT/tmp"
export XDG_CACHE_HOME="$MDDM_SWEEP_STATE_ROOT/cache"
export HF_HOME="$MDDM_SWEEP_STATE_ROOT/huggingface"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME"

bash scripts/bootstrap_env.sh
export MDDM_SWEEP_VENV="$MDDM_SWEEP_STATE_ROOT/venv"
export ML_RL_DLLM_REPO="$MDDM_SWEEP_STATE_ROOT/ml-rl-dllm"

exec bash scripts/run_sequential.sh
