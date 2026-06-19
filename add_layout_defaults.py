#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    layout_rows = read_jsonl(run_dir / "layout_control.jsonl")
    baselines = load_baselines(run_dir)

    for row in layout_rows:
        default = row.get("p_gt_default") or row.get("p_gt_before") or baselines[str(row["example_id"])]
        row["p_gt_default"] = default
        row.setdefault("p_gt_before", default)

    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{args.name}.jsonl"
    out_csv = out_dir / f"{args.name}.csv"
    write_jsonl(out_jsonl, layout_rows)
    write_csv(out_csv, layout_rows)
    print(f"Wrote {out_jsonl}")
    print(f"Wrote {out_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add default p_gt to cached layout-control rows.")
    parser.add_argument("run_dir", help="Directory containing layout_control.jsonl and cached probe rows")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--name", default="layout_control_with_default")
    return parser.parse_args()


def load_baselines(run_dir: Path) -> dict[str, list[float]]:
    for name in ["single_anchor.jsonl", "greedy_k.jsonl"]:
        path = run_dir / name
        if not path.exists():
            continue
        baselines: dict[str, list[float]] = {}
        for row in read_jsonl(path):
            if "p_gt_before" in row:
                baselines.setdefault(str(row["example_id"]), row["p_gt_before"])
        if baselines:
            return baselines
    raise FileNotFoundError(f"No p_gt_before found in {run_dir}/single_anchor.jsonl or greedy_k.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


if __name__ == "__main__":
    main()
