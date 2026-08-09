#!/usr/bin/env python3
"""Select the fastest threshold that stays near held-out base accuracy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    args = parse_args()
    baseline = load_threshold(Path(args.baseline_summary), args.baseline_threshold)
    trained = load_summaries(Path(args.trained_summary))
    result = select_operating_point(
        baseline,
        trained,
        accuracy_tolerance=args.accuracy_tolerance,
        minimum_tokens_per_forward=args.minimum_tokens_per_forward,
    )
    payload = {"baseline": baseline, "candidates": trained, "selected": result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(render(payload))
    if result is None:
        raise SystemExit(2)


def load_summaries(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    return sorted(rows, key=lambda row: float(row["confidence_threshold"]))


def load_threshold(path, threshold):
    rows = load_summaries(path)
    matches = [
        row
        for row in rows
        if abs(float(row["confidence_threshold"]) - float(threshold)) < 1e-9
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one threshold {threshold} in {path}")
    return matches[0]


def select_operating_point(
    baseline, trained, *, accuracy_tolerance, minimum_tokens_per_forward
):
    accuracy_floor = float(baseline["accuracy"]) - float(accuracy_tolerance)
    eligible = [
        row
        for row in trained
        if float(row["accuracy"]) >= accuracy_floor
        and float(row["tokens_per_forward"]) >= minimum_tokens_per_forward
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            float(row["tokens_per_forward"]),
            float(row["accuracy"]),
            float(row["confidence_threshold"]),
        ),
    )


def render(payload):
    baseline = payload["baseline"]
    lines = [
        "# Held-out Threshold Selection",
        "",
        f"Base reference: {100 * float(baseline['accuracy']):.2f}% accuracy, "
        f"{float(baseline['tokens_per_forward']):.3f} tokens/forward.",
        "",
        "| Threshold | Accuracy | Tokens/forward |",
        "|---:|---:|---:|",
    ]
    for row in payload["candidates"]:
        lines.append(
            f"| {float(row['confidence_threshold']):.2f} | "
            f"{100 * float(row['accuracy']):.2f}% | "
            f"{float(row['tokens_per_forward']):.3f} |"
        )
    lines.append("")
    selected = payload["selected"]
    if selected is None:
        lines.append("No threshold passed the quality and throughput constraints.")
    else:
        lines.append(
            f"Selected threshold: {float(selected['confidence_threshold']):.2f}."
        )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--trained-summary", required=True)
    parser.add_argument("--baseline-threshold", type=float, default=0.95)
    parser.add_argument("--accuracy-tolerance", type=float, default=0.02)
    parser.add_argument("--minimum-tokens-per-forward", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.accuracy_tolerance < 0:
        parser.error("accuracy-tolerance must be nonnegative")
    if args.minimum_tokens_per_forward <= 0:
        parser.error("minimum-tokens-per-forward must be positive")
    return args


if __name__ == "__main__":
    main()
