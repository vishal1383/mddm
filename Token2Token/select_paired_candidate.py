#!/usr/bin/env python3
"""Select a checkpoint using paired quality and decoder-forward efficiency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from Token2Token.micro_trial_gate import evaluate_candidate
from Token2Token.paired_comparison import load_rows


def select_candidate(results):
    if not results:
        raise ValueError("at least one candidate is required")
    passing = [item for item in results if item["gate"]["passed"]]
    pool = passing or results
    selected = max(
        pool,
        key=lambda item: (
            item["gate"]["trained_correct"],
            item["gate"]["trained_tokens_per_forward"],
        ),
    )
    return selected, bool(passing)


def main():
    args = parse_args()
    baseline = load_rows(Path(args.baseline_predictions))
    results = []
    for label, predictions, adapter in args.candidate:
        gate = evaluate_candidate(
            baseline,
            load_rows(Path(predictions)),
            completion_length=args.completion_length,
            min_tokens_per_forward=args.minimum_tokens_per_forward,
            min_tokens_per_forward_gain=args.minimum_tokens_per_forward_gain,
            max_correct_loss=args.max_correct_loss,
            expected_threshold=args.expected_threshold,
            expected_numeric_threshold=args.expected_numeric_threshold,
            min_example_id=args.min_example_id,
            max_example_id=args.max_example_id,
        )
        results.append(
            {
                "label": label,
                "predictions": predictions,
                "adapter_path": adapter,
                "gate": gate,
            }
        )
    selected, any_passed = select_candidate(results)
    report = {
        "any_candidate_passed": any_passed,
        "selected_label": selected["label"],
        "selected_adapter_path": selected["adapter_path"],
        "candidates": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=3,
        metavar=("LABEL", "PREDICTIONS", "ADAPTER"),
        required=True,
    )
    parser.add_argument("--completion-length", type=int, default=128)
    parser.add_argument("--minimum-tokens-per-forward", type=float, default=4.0)
    parser.add_argument("--minimum-tokens-per-forward-gain", type=float, default=0.1)
    parser.add_argument("--max-correct-loss", type=int, default=1)
    parser.add_argument("--expected-threshold", type=float, required=True)
    parser.add_argument("--expected-numeric-threshold", type=float, required=True)
    parser.add_argument("--min-example-id", type=int)
    parser.add_argument("--max-example-id", type=int)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
