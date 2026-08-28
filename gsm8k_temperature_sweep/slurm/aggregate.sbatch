#!/usr/bin/env bash
#SBATCH --job-name=gsm8k-table
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
: "${MDDM_SWEEP_VENV:?MDDM_SWEEP_VENV is required}"
: "${MDDM_SWEEP_OUTPUT_ROOT:?MDDM_SWEEP_OUTPUT_ROOT is required}"

ARGS=(--output-root "$MDDM_SWEEP_OUTPUT_ROOT")
if [[ "${MDDM_SWEEP_ALLOW_PARTIAL:-0}" == "1" ]]; then
  ARGS+=(--allow-partial)
fi
ARGS+=(--table-stem "${MDDM_SWEEP_TABLE_STEM:-final_table}")
"$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/aggregate.py" "${ARGS[@]}"
