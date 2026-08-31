#!/usr/bin/env bash
set -euo pipefail

cd /home/vishalg/Desktop/DhruveshProject

download_container=diffucoder-cpgrpo-download-10eval
active_container=apple-grpo-train128-v4-cycle01-a01
eval_container=diffucoder-vs-dparallel-gsm8k10-v1
output=Token2Token/outputs/diffucoder_vs_dparallel_gsm8k10/v1
log="$output/queue.log"
mkdir -p "$output"

exec > >(tee -a "$log") 2>&1
date -Is
echo "Waiting for the DiffuCoder checkpoint download."
while [[ "$(docker inspect --format '{{.State.Running}}' "$download_container")" == true ]]; do
  sleep 60
done
download_status="$(docker inspect --format '{{.State.ExitCode}}' "$download_container")"
if [[ "$download_status" -ne 0 ]]; then
  echo "Checkpoint download failed with exit code $download_status."
  exit "$download_status"
fi

echo "Waiting for the active local GRPO training container without interrupting it."
while [[ "$(docker inspect --format '{{.State.Running}}' "$active_container")" == true ]]; do
  sleep 60
done

if [[ -f "$output/table.md" ]]; then
  echo "Diagnostic already complete: $output/table.md"
  exit 0
fi
if docker inspect "$eval_container" >/dev/null 2>&1; then
  echo "Restarting the saved resumable evaluation container."
  docker start -a "$eval_container"
else
  echo "Launching the three-row 10-example diagnostic."
  docker run \
    --name "$eval_container" \
    --gpus all \
    --ipc=host \
    -w /workspace \
    -e HOME=/tmp \
    -e HF_HOME=/hf \
    -e DPARALLEL_HF_HOME=/hf_dparallel \
    -e HF_MODULES_CACHE=/tmp/hf-modules \
    -e HF_DATASETS_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH=/workspace \
    -v /home/vishalg/v260_hf_cache:/hf:ro \
    -v /home/vishalg/v260_hf_cache/datasets:/hf/datasets:rw \
    -v /home/vishalg/.cache/huggingface:/hf_dparallel:ro \
    -v /home/vishalg/Desktop/DhruveshProject:/workspace:rw \
    token2token-apple-grpo:v1 \
    bash -lc "python3 Token2Token/experiments/eval_diffucoder_vs_dparallel10.py --output '$output'"
fi

test -s "$output/table.md"
echo "Complete: $output/table.md"
date -Is
