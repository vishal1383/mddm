#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_ROOT="${MDDM_SWEEP_STATE_ROOT:-${SCRATCH:-$EXPERIMENT_ROOT/.runtime}}"
export MDDM_SWEEP_VENV="${MDDM_SWEEP_VENV:-$STATE_ROOT/venv}"
export ML_RL_DLLM_REPO="${ML_RL_DLLM_REPO:-$STATE_ROOT/ml-rl-dllm}"
export MDDM_SWEEP_OUTPUT_ROOT="${MDDM_SWEEP_OUTPUT_ROOT:-$EXPERIMENT_ROOT/results}"

if [[ ! -x "$MDDM_SWEEP_VENV/bin/python" ]]; then
  echo "Missing experiment environment. Run: bash scripts/bootstrap_env.sh" >&2
  exit 2
fi
mkdir -p "$MDDM_SWEEP_OUTPUT_ROOT/logs"
"$MDDM_SWEEP_VENV/bin/python" "$EXPERIMENT_ROOT/preflight.py"

ARRAY_LIMIT="${MDDM_SWEEP_ARRAY_LIMIT:-8}"
ARRAY_JOB_ID="$(sbatch --parsable \
  --array="0-59%$ARRAY_LIMIT" \
  --output="$MDDM_SWEEP_OUTPUT_ROOT/logs/%A_%a.out" \
  --error="$MDDM_SWEEP_OUTPUT_ROOT/logs/%A_%a.err" \
  --export=ALL \
  "$EXPERIMENT_ROOT/slurm/sweep.sbatch")"
TABLE_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:$ARRAY_JOB_ID" \
  --output="$MDDM_SWEEP_OUTPUT_ROOT/logs/table_%j.out" \
  --error="$MDDM_SWEEP_OUTPUT_ROOT/logs/table_%j.err" \
  --export=ALL \
  "$EXPERIMENT_ROOT/slurm/aggregate.sbatch")"

echo "Submitted full 60-cell sweep: $ARRAY_JOB_ID"
echo "Submitted fail-closed final aggregation: $TABLE_JOB_ID"
echo "Results: $MDDM_SWEEP_OUTPUT_ROOT"
