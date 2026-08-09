#!/usr/bin/env python3
"""Gate a short decoder trial on paired quality and model-forward efficiency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from Token2Token.main.paired_comparison import compare, filtered_shared_ids, load_rows


def evaluate_candidate(
    baseline_rows,
    trained_rows,
    *,
    completion_length,
    min_tokens_per_forward,
    min_tokens_per_forward_gain,
    max_correct_loss,
    expected_threshold=None,
    expected_numeric_threshold=None,
    min_example_id=None,
    max_example_id=None,
):
    shared = filtered_shared_ids(
        baseline_rows,
        trained_rows,
        min_example_id=min_example_id,
        max_example_id=max_example_id,
    )
    if not shared:
        raise ValueError("no overlapping example ids")

    baseline = [baseline_rows[key] for key in shared]
    trained = [trained_rows[key] for key in shared]
    if expected_threshold is not None:
        assert_matching_threshold(baseline, expected_threshold, "baseline")
        assert_matching_threshold(trained, expected_threshold, "trained")
    if expected_numeric_threshold is not None:
        assert_matching_numeric_threshold(
            baseline, expected_numeric_threshold, "baseline"
        )
        assert_matching_numeric_threshold(
            trained, expected_numeric_threshold, "trained"
        )
    paired = compare(baseline, trained)
    baseline_forwards = sum(int(row["model_forwards"]) for row in baseline)
    trained_forwards = sum(int(row["model_forwards"]) for row in trained)
    baseline_tpf = completion_length * len(shared) / baseline_forwards
    trained_tpf = completion_length * len(shared) / trained_forwards
    correct_delta = paired["trained_correct"] - paired["baseline_correct"]
    quality_pass = correct_delta >= -max_correct_loss
    absolute_speed_pass = trained_tpf >= min_tokens_per_forward
    relative_speed_pass = (
        trained_tpf - baseline_tpf >= min_tokens_per_forward_gain
    )

    return {
        "examples": len(shared),
        "min_example_id": min(map(int, shared)),
        "max_example_id": max(map(int, shared)),
        "confidence_threshold": expected_threshold,
        "numeric_threshold": expected_numeric_threshold,
        "baseline_correct": paired["baseline_correct"],
        "trained_correct": paired["trained_correct"],
        "correct_delta": correct_delta,
        "only_baseline_correct": paired["only_baseline_correct"],
        "only_trained_correct": paired["only_trained_correct"],
        "baseline_tokens_per_forward": baseline_tpf,
        "trained_tokens_per_forward": trained_tpf,
        "tokens_per_forward_delta": trained_tpf - baseline_tpf,
        "baseline_forwards_per_example": paired["baseline_forwards_per_example"],
        "trained_forwards_per_example": paired["trained_forwards_per_example"],
        "max_correct_loss": max_correct_loss,
        "minimum_tokens_per_forward": min_tokens_per_forward,
        "minimum_tokens_per_forward_gain": min_tokens_per_forward_gain,
        "quality_pass": quality_pass,
        "absolute_speed_pass": absolute_speed_pass,
        "relative_speed_pass": relative_speed_pass,
        "passed": quality_pass and absolute_speed_pass and relative_speed_pass,
    }


def assert_matching_threshold(rows, expected_threshold, label):
    observed = {float(row["confidence_threshold"]) for row in rows}
    if observed != {float(expected_threshold)}:
        raise ValueError(
            f"{label} threshold mismatch: expected {expected_threshold}, "
            f"observed {sorted(observed)}"
        )


def assert_matching_numeric_threshold(rows, expected_threshold, label):
    observed = {row.get("numeric_threshold") for row in rows}
    if observed != {float(expected_threshold)}:
        raise ValueError(
            f"{label} numeric threshold mismatch: expected {expected_threshold}, "
            f"observed {sorted(observed, key=lambda item: (item is None, item))}"
        )


def main() -> None:
    args = parse_args()
    result = evaluate_candidate(
        load_rows(Path(args.baseline_predictions)),
        load_rows(Path(args.trained_predictions)),
        completion_length=args.completion_length,
        min_tokens_per_forward=args.minimum_tokens_per_forward,
        min_tokens_per_forward_gain=args.minimum_tokens_per_forward_gain,
        max_correct_loss=args.max_correct_loss,
        expected_threshold=args.expected_threshold,
        expected_numeric_threshold=args.expected_numeric_threshold,
        min_example_id=args.min_example_id,
        max_example_id=args.max_example_id,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.fail_on_reject and not result["passed"]:
        raise SystemExit(2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--trained-predictions", required=True)
    parser.add_argument("--completion-length", type=int, default=128)
    parser.add_argument("--minimum-tokens-per-forward", type=float, default=4.0)
    parser.add_argument("--minimum-tokens-per-forward-gain", type=float, default=0.1)
    parser.add_argument("--max-correct-loss", type=int, default=1)
    parser.add_argument("--expected-threshold", type=float, required=True)
    parser.add_argument("--expected-numeric-threshold", type=float)
    parser.add_argument("--min-example-id", type=int)
    parser.add_argument("--max-example-id", type=int)
    parser.add_argument("--output")
    parser.add_argument("--fail-on-reject", action="store_true")
    args = parser.parse_args()
    if args.completion_length <= 0:
        parser.error("completion-length must be positive")
    if args.minimum_tokens_per_forward <= 0:
        parser.error("minimum-tokens-per-forward must be positive")
    if args.minimum_tokens_per_forward_gain < 0:
        parser.error("minimum-tokens-per-forward-gain must be nonnegative")
    if args.max_correct_loss < 0:
        parser.error("max-correct-loss must be nonnegative")
    return args


if __name__ == "__main__":
    main()
