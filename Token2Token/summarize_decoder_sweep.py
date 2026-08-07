#!/usr/bin/env python3
"""Collect decoder-sweep summaries into one accuracy/latency table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    args = parse_args()
    rows = collect_rows(Path(args.sweep_dir))
    if not rows:
        raise SystemExit(f"no summary.json files under {args.sweep_dir}")
    rows = rank_rows(rows)
    report = render_markdown(rows)
    output = Path(args.output or Path(args.sweep_dir) / "sweep.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)


def collect_rows(sweep_dir: Path):
    rows = []
    for summary_path in sorted(sweep_dir.glob("*/summary.json")):
        for entry in json.loads(summary_path.read_text(encoding="utf-8")):
            rows.append({"config": summary_path.parent.name, **entry})
    return rows


def rank_rows(rows):
    return sorted(rows, key=lambda row: -tokens_per_forward(row))


def tokens_per_forward(row):
    return float(row.get("tokens_per_forward") or 0.0)


def pareto_front(rows):
    """Configs that no other config beats on both accuracy and tokens/forward."""
    front = []
    for row in rows:
        dominated = any(
            other is not row
            and float(other["accuracy"]) >= float(row["accuracy"])
            and tokens_per_forward(other) >= tokens_per_forward(row)
            and (
                float(other["accuracy"]) > float(row["accuracy"])
                or tokens_per_forward(other) > tokens_per_forward(row)
            )
            for other in rows
        )
        if not dominated:
            front.append(row)
    return front


def render_markdown(rows):
    front = {id(row) for row in pareto_front(rows)}
    lines = [
        "# Decoder sweep",
        "",
        "| Config | Threshold | Accuracy | Forwards/example | Tokens/forward |"
        " Seconds/example | Pareto |",
        "|---|---:|---:|---:|---:|---:|:--:|",
    ]
    for row in rows:
        examples = int(row["examples"]) or 1
        forwards = int(row["total_model_forwards"]) / examples
        seconds = float(row["elapsed_seconds"]) / examples
        lines.append(
            f"| {row['config']} | {float(row['confidence_threshold']):.2f} | "
            f"{int(row['correct'])}/{examples} = "
            f"{100 * float(row['accuracy']):.1f}% | {forwards:.1f} | "
            f"{tokens_per_forward(row):.3f} | {seconds:.2f} | "
            f"{'yes' if id(row) in front else ''} |"
        )
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    main()
