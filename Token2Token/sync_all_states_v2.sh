#!/usr/bin/env bash
set -u

RUNNER_PID="${1:?pass the container runner PID}"
SOURCE="confident_borg:/workspace/DhruveshProject/outputs/token2token/all_states_confidence_v2/."
ARTIFACTS="/home/vishalg/Desktop/DhruveshProjectArtifacts/token2token/all_states_confidence_v2/"
REPO_COPY="/home/vishalg/Desktop/DhruveshProject/Token2Token/results/all_states_confidence_v2_raw/"

mkdir -p "$ARTIFACTS" "$REPO_COPY"

copy_outputs() {
  docker cp "$SOURCE" "$ARTIFACTS"
  docker cp "$SOURCE" "$REPO_COPY"
}

while docker exec confident_borg kill -0 "$RUNNER_PID" >/dev/null 2>&1; do
  copy_outputs
  sleep 60
done

copy_outputs
date --iso-8601=seconds | tee "$ARTIFACTS/COPY_COMPLETE" > "$REPO_COPY/COPY_COMPLETE"
