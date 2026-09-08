#!/usr/bin/env bash
#SBATCH --job-name=d3lm-anchor-plots
#SBATCH --partition=cpu
#SBATCH --account=pi_mccallum_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=d3lm-anchor-plots-%j.out
#SBATCH --error=d3lm-anchor-plots-%j.err
set -euo pipefail
DNA_REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the mddm repository root}"
DNA_STATE_ROOT="${DNA_ANCHOR_STATE:-$DNA_REPO_ROOT/dna_anchor_census/.runtime/all_species_v1}"
DNA_PYTHON="${DNA_ANCHOR_PYTHON:-$DNA_REPO_ROOT/gsm8k_temperature_sweep/.runtime/venv/bin/python}"
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR="$DNA_STATE_ROOT/matplotlib-plots-$SLURM_JOB_ID"
mkdir -p "$MPLCONFIGDIR"
export PYTHONPATH="$DNA_STATE_ROOT/deps:$DNA_REPO_ROOT/dna_anchor_census"
exec "$DNA_PYTHON" "$DNA_REPO_ROOT/dna_anchor_parallel/plots.py" --state "$DNA_STATE_ROOT"
