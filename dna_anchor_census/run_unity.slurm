#!/usr/bin/env bash
#SBATCH --job-name=d3lm-anchors-both
#SBATCH --partition=gpu-preempt
#SBATCH --account=pi_mccallum_umass_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=45:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=d3lm-anchors-%j.out
#SBATCH --error=d3lm-anchors-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
DNA_REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the mddm repository root}"
DNA_STATE_ROOT="${DNA_ANCHOR_STATE:-$DNA_REPO_ROOT/dna_anchor_census/.runtime/all_species_v1}"
DNA_PYTHON="${DNA_ANCHOR_PYTHON:-$DNA_REPO_ROOT/gsm8k_temperature_sweep/.runtime/venv/bin/python}"
mkdir -p "$DNA_STATE_ROOT"
export HF_HOME="$DNA_STATE_ROOT/huggingface"
export HF_MODULES_CACHE="$DNA_STATE_ROOT/transformers_modules"
export PIP_CACHE_DIR="$DNA_STATE_ROOT/pip-cache"
test -x "$DNA_PYTHON"
"$DNA_PYTHON" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; name=torch.cuda.get_device_name(0); assert "A100" in name.upper(), name; print("Allocated exactly one GPU:",name,flush=True)'
if ! test -f "$DNA_STATE_ROOT/dependencies.ready"; then
  "$DNA_PYTHON" -m pip install --target "$DNA_STATE_ROOT/deps" -r "$DNA_REPO_ROOT/dna_anchor_census/requirements.txt"
  touch "$DNA_STATE_ROOT/dependencies.ready"
fi
export PYTHONPATH="$DNA_STATE_ROOT/deps:$DNA_REPO_ROOT/dna_anchor_census"
exec "$DNA_PYTHON" "$DNA_REPO_ROOT/dna_anchor_census/run.py" --state "$DNA_STATE_ROOT" --scope all_species --batch-size 32
