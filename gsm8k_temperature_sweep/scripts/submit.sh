#!/usr/bin/env bash
# Backward-compatible in-allocation entrypoint. It never submits nested jobs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_sequential.sh"
