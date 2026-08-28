#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_ROOT="${MDDM_SWEEP_STATE_ROOT:-${SCRATCH:-$EXPERIMENT_ROOT/.runtime}}"
VENV_PATH="${MDDM_SWEEP_VENV:-$STATE_ROOT/venv}"
POLICY_REPO="${ML_RL_DLLM_REPO:-$STATE_ROOT/ml-rl-dllm}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
POLICY_COMMIT="35e4830485f1821d57f9ac3f1a303f3d4531fb82"

mkdir -p "$STATE_ROOT"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.12 is required; set PYTHON_BIN to the Unity Python 3.12 executable." >&2
  exit 2
fi
if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi
"$VENV_PATH/bin/python" -m pip install --upgrade pip wheel setuptools
"$VENV_PATH/bin/python" -m pip install -r "$EXPERIMENT_ROOT/requirements.txt"

if [[ ! -d "$POLICY_REPO/.git" ]]; then
  git clone https://github.com/apple/ml-rl-dllm.git "$POLICY_REPO"
fi
git -C "$POLICY_REPO" fetch origin "$POLICY_COMMIT"
if [[ -n "$(git -C "$POLICY_REPO" status --porcelain)" ]]; then
  echo "Refusing to alter dirty upstream checkout: $POLICY_REPO" >&2
  exit 2
fi
git -C "$POLICY_REPO" switch --detach "$POLICY_COMMIT"

ML_RL_DLLM_REPO="$POLICY_REPO" PYTHONPATH="$EXPERIMENT_ROOT" \
  "$VENV_PATH/bin/python" -m unittest discover -s "$EXPERIMENT_ROOT/tests" -v

echo "Bootstrap complete. Export these values before submission:"
echo "export MDDM_SWEEP_VENV=$VENV_PATH"
echo "export ML_RL_DLLM_REPO=$POLICY_REPO"
