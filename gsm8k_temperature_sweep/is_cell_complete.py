#!/usr/bin/env python3
"""Exit 0 for a valid complete result, 1 if absent, and 3 if corrupt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from experiment_contract import OFFICIAL_TEST_EXAMPLES, SAMPLES, SCHEMA_VERSION, task_for_id, temperature_slug


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--task-id", required=True, type=int)
    args = parser.parse_args()
    method, temperature = task_for_id(args.task_id)
    cell = Path(args.output_root).resolve() / method / temperature_slug(temperature)
    summary_path = cell / "summary.json"
    contract_path = cell / "contract.json"
    if not summary_path.is_file():
        return 1
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        valid = (
            summary.get("schema") == SCHEMA_VERSION
            and summary.get("method") == method
            and float(summary.get("temperature", -1)) == temperature
            and bool(summary.get("complete"))
            and int(summary.get("examples", -1)) == OFFICIAL_TEST_EXAMPLES
            and int(summary.get("trajectories", -1)) == OFFICIAL_TEST_EXAMPLES * SAMPLES
            and summary.get("contract_sha256") == contract.get("contract_sha256")
        )
    except Exception as error:
        print(f"Invalid saved cell {cell}: {error}", file=sys.stderr)
        return 3
    if not valid:
        print(f"Invalid saved cell {cell}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
