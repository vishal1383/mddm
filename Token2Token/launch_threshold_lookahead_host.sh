#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/vishalg/Desktop/DhruveshProject"
CONTAINER="confident_borg"
CONTAINER_ROOT="/workspace/DhruveshProject/outputs/token2token/threshold_lookahead_v7/full_train7473_t090_num099"
HOST_ROOT="$PROJECT/Token2Token/artifacts/threshold_lookahead_v7/full_train7473_t090_num099"
mkdir -p "$HOST_ROOT"

sync_outputs() {
  docker cp "$CONTAINER:$CONTAINER_ROOT/." "$HOST_ROOT/" >/dev/null 2>&1 || true
}

watch_sync() {
  while true; do
    sleep 300
    sync_outputs
  done
}

watch_sync &
watch_pid=$!
trap 'kill "$watch_pid" >/dev/null 2>&1 || true; sync_outputs' EXIT

cd "$PROJECT"
docker exec -w /workspace/DhruveshProject "$CONTAINER" \
  bash Token2Token/run_threshold_lookahead_overnight.sh \
  2>&1 | tee "$HOST_ROOT/host_runner.log"
