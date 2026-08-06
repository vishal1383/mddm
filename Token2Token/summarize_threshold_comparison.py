#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path, threshold: float):
    rows = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        row
        for row in rows
        if abs(float(row["confidence_threshold"]) - threshold) < 1e-9
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one threshold {threshold} row in {path}")
    return add_latency_metrics(matches[0])


def add_latency_metrics(row):
    result = dict(row)
    examples = int(result["examples"])
    elapsed = float(result["elapsed_seconds"])
    forwards = int(result["total_model_forwards"])
    result["seconds_per_example"] = elapsed / examples if examples else None
    result["examples_per_second"] = examples / elapsed if elapsed else None
    result["forwards_per_example"] = forwards / examples if examples else None
    return result


def comparison(baseline, trained):
    base_latency = float(baseline["seconds_per_example"])
    trained_latency = float(trained["seconds_per_example"])
    return {
        "confidence_threshold": float(trained["confidence_threshold"]),
        "baseline": baseline,
        "trained": trained,
        "accuracy_change_pp": 100.0
        * (float(trained["accuracy"]) - float(baseline["accuracy"])),
        "latency_change_percent": 100.0
        * (trained_latency / base_latency - 1.0),
        "trained_speedup_vs_baseline": base_latency / trained_latency,
    }


def render_markdown(result):
    baseline = result["baseline"]
    trained = result["trained"]
    lines = [
        "# Threshold Decode Comparison",
        "",
        f"Confidence threshold: {result['confidence_threshold']:.2f}",
        "",
        "| Model | Accuracy | Correct | Seconds/example | Examples/second | Forwards/example | Tokens/forward |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (baseline, trained):
        lines.append(
            f"| {row['model_label']} | {100 * float(row['accuracy']):.2f}% | "
            f"{int(row['correct'])}/{int(row['examples'])} | "
            f"{float(row['seconds_per_example']):.4f} | "
            f"{float(row['examples_per_second']):.4f} | "
            f"{float(row['forwards_per_example']):.2f} | "
            f"{float(row['tokens_per_forward']):.3f} |"
        )
    lines.extend(
        [
            "",
            f"Accuracy change: {result['accuracy_change_pp']:+.2f} percentage points.",
            f"Latency change: {result['latency_change_percent']:+.2f}%.",
            f"Trained/base speed ratio: {result['trained_speedup_vs_baseline']:.3f}x.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare base and trained threshold-decoding evaluations"
    )
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--trained-summary", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = comparison(
        load_summary(args.baseline_summary, args.threshold),
        load_summary(args.trained_summary, args.threshold),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = render_markdown(result)
    (args.output_dir / "comparison.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
