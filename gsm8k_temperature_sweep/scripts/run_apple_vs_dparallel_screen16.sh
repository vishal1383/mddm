#!/usr/bin/env bash
set -euo pipefail

cd /workspace/gsm8k_temperature_sweep

output_root=/workspace/gsm8k_temperature_sweep/final_results/screen16_apple_vs_dparallel
policy_repo=/workspace/Token2Token/.external/ml-rl-dllm

python evaluate_screen16.py \
  --method paper_policy \
  --policy-repo "$policy_repo" \
  --output-root "$output_root"

python evaluate_screen16.py \
  --method dparallel \
  --policy-repo "$policy_repo" \
  --output-root "$output_root"

python evaluate_screen16.py \
  --aggregate-only \
  --output-root "$output_root"
