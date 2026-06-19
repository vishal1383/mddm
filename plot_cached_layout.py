#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mdm_probe.plots import plot_layout_pgt_results


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    layout_rows = read_jsonl(run_dir / "layout_control.jsonl")
    baselines = load_baselines(run_dir) if any("p_gt_before" not in row for row in layout_rows) else {}

    for row in layout_rows:
        example_id = str(row["example_id"])
        if "p_gt_before" not in row:
            row["p_gt_before"] = baselines[example_id]
        row.setdefault("p_gt_default", row["p_gt_before"])

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "plots_cached_layout_pgt"
    plot_layout_pgt_results(layout_rows, out_dir)
    print(f"Wrote cached layout p_gt plots to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cached layout-control p_gt curves.")
    parser.add_argument("run_dir", help="Directory containing layout_control.jsonl and cached probe rows")
    parser.add_argument("--out-dir", default=None, help="Plot output dir; default: RUN_DIR/plots_cached_layout_pgt")
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
    raise FileNotFoundError(f"No p_gt_before baseline found in {run_dir}/single_anchor.jsonl or greedy_k.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
