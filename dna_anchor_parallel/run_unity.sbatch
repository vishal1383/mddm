#!/usr/bin/env bash
#SBATCH --job-name=d3lm-anchors-4a100
#SBATCH --partition=gpu-preempt
#SBATCH --account=pi_mccallum_umass_edu
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100-80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=45:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=d3lm-anchors-4a100-%j.out
#SBATCH --error=d3lm-anchors-4a100-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
DNA_REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the mddm repository root}"
DNA_STATE_ROOT="${DNA_ANCHOR_STATE:-$DNA_REPO_ROOT/dna_anchor_census/.runtime/all_species_v1}"
DNA_PYTHON="${DNA_ANCHOR_PYTHON:-$DNA_REPO_ROOT/gsm8k_temperature_sweep/.runtime/venv/bin/python}"
test -x "$DNA_PYTHON"
test -f "$DNA_STATE_ROOT/dependencies.ready"
export HF_HOME="$DNA_STATE_ROOT/huggingface"
export HF_MODULES_CACHE="$DNA_STATE_ROOT/transformers_modules"
export MPLCONFIGDIR="$DNA_STATE_ROOT/matplotlib-$SLURM_JOB_ID"
mkdir -p "$MPLCONFIGDIR"
export PYTHONPATH="$DNA_STATE_ROOT/deps:$DNA_REPO_ROOT/dna_anchor_census:$DNA_REPO_ROOT/dna_anchor_parallel"
# Single coordinator sees exactly four A100s; spawned servers each see one UUID/device.
# The initial four reference replays must pass before any production work is queued.
exec "$DNA_PYTHON" "$DNA_REPO_ROOT/dna_anchor_parallel/run.py" \
  --state "$DNA_STATE_ROOT" --clients-per-gpu "${DNA_CLIENTS_PER_GPU:-4}" \
  --cpu-threads 2 "$@"
