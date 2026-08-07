#!/usr/bin/env bash
# Run the remaining queue sequentially on the single GPU.
#
# Waits on explicit PIDs rather than pgrep patterns. A waiter written as
# `while pgrep -f run_decoder_sweep50; do ...` matches its own command line,
# because pgrep -f tests full command lines and the pattern is part of this
# script's invocation, so such a loop never terminates.
set -uo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

for pid in "$@"; do
  echo "waiting for pid $pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  echo "pid $pid finished"
done

echo "=== round 2 decoder sweep ==="
bash Token2Token/run_decoder_sweep50_round2.sh || echo "round 2 failed"

echo "=== v4a evaluation ==="
NAME=v4a bash Token2Token/run_parallel_unlock_v4.sh || echo "v4a eval failed"

echo "=== full base benchmark, 1319 examples ==="
BATCH=16 LIMIT=1319 bash Token2Token/run_full_benchmark.sh || echo "full benchmark failed"

echo "=== chain complete ==="
