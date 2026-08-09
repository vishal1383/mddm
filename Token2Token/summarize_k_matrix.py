#!/usr/bin/env python3
"""Summarize a model-family by fixed-k evaluation matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.glob("*/k*/summary.json")):
        summaries = json.loads(path.read_text(encoding="utf-8"))
        if not summaries:
            continue
        summary = summaries[-1]
        rows.append(
            {
                "family": path.parent.parent.name,
                "k": int(path.parent.name.removeprefix("k")),
                **summary,
            }
        )
    rows.sort(key=lambda row: (row["family"], row["k"]))
    lines = [
        "# Full GSM8K k=1,2,3 Evaluation",
        "",
        "| Model | k | Correct | Accuracy | Forwards/example | Tokens/forward |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        examples = int(row["examples"])
        forwards = int(row["total_model_forwards"])
        lines.append(
            f"| {row['family']} | {row['k']} | {int(row['correct'])}/{examples} "
            f"| {100 * float(row['accuracy']):.2f}% "
            f"| {forwards / examples:.2f} "
            f"| {float(row['tokens_per_forward']):.3f} |"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
