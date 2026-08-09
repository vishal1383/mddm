#!/usr/bin/env python3
"""Paired base-vs-trained comparison on the same examples.

Accuracy deltas on 50 examples are mostly noise: at 68% the standard error of
an unpaired proportion is 6.6 pp, so a 4 pp gap says nothing. Both models
decode the same questions, so pair them. McNemar uses only the examples where
the two disagree, which is far more sensitive, and the forward-count
difference is compared per example rather than in aggregate.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
import random
import statistics


def main() -> None:
    args = parse_args()
    baseline = load_rows(Path(args.baseline_predictions))
    trained = load_rows(Path(args.trained_predictions))
    shared = filtered_shared_ids(
        baseline,
        trained,
        min_example_id=args.min_example_id,
        max_example_id=args.max_example_id,
    )
    if not shared:
        raise SystemExit("no overlapping example ids")
    result = compare(
        [baseline[key] for key in shared], [trained[key] for key in shared]
    )
    report = render_markdown(result, args.baseline_label, args.trained_label)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    print(report)


def filtered_shared_ids(
    baseline, trained, min_example_id=None, max_example_id=None
):
    """Return matched numeric IDs inside the requested half-open interval."""
    shared = sorted(set(baseline) & set(trained), key=int)
    if min_example_id is not None:
        shared = [key for key in shared if int(key) >= min_example_id]
    if max_example_id is not None:
        shared = [key for key in shared if int(key) < max_example_id]
    return shared


def load_rows(path: Path):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["example_id"])] = row
    return rows


def compare(baseline, trained):
    both = sum(1 for a, b in zip(baseline, trained) if a["correct"] and b["correct"])
    only_baseline = sum(
        1 for a, b in zip(baseline, trained) if a["correct"] and not b["correct"]
    )
    only_trained = sum(
        1 for a, b in zip(baseline, trained) if not a["correct"] and b["correct"]
    )
    neither = sum(
        1 for a, b in zip(baseline, trained) if not a["correct"] and not b["correct"]
    )
    forward_deltas = [
        int(b["model_forwards"]) - int(a["model_forwards"])
        for a, b in zip(baseline, trained)
    ]
    accuracy_deltas = [
        int(bool(b["correct"])) - int(bool(a["correct"]))
        for a, b in zip(baseline, trained)
    ]
    return {
        "examples": len(baseline),
        "baseline_correct": both + only_baseline,
        "trained_correct": both + only_trained,
        "both_correct": both,
        "only_baseline_correct": only_baseline,
        "only_trained_correct": only_trained,
        "neither_correct": neither,
        "mcnemar_p_value": mcnemar_p_value(only_baseline, only_trained),
        "accuracy_delta": mean_or_none(accuracy_deltas),
        "accuracy_delta_ci95": bootstrap_ci(accuracy_deltas),
        "baseline_forwards_per_example": mean_or_none(
            [int(row["model_forwards"]) for row in baseline]
        ),
        "trained_forwards_per_example": mean_or_none(
            [int(row["model_forwards"]) for row in trained]
        ),
        "forward_delta_per_example": mean_or_none(forward_deltas),
        "forward_delta_ci95": bootstrap_ci(forward_deltas),
    }


def mcnemar_p_value(only_baseline, only_trained):
    """Exact two-sided McNemar over the discordant pairs."""
    discordant = only_baseline + only_trained
    if discordant == 0:
        return 1.0
    extreme = max(only_baseline, only_trained)
    tail = sum(comb(discordant, k) for k in range(extreme, discordant + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def bootstrap_ci(values, samples=10_000, seed=0):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    count = len(values)
    means = sorted(
        sum(rng.choices(values, k=count)) / count for _ in range(samples)
    )
    return [means[int(0.025 * samples)], means[int(0.975 * samples) - 1]]


def mean_or_none(values):
    return statistics.fmean(values) if values else None


def render_markdown(result, baseline_label, trained_label):
    examples = result["examples"]
    low, high = result["forward_delta_ci95"] or (float("nan"), float("nan"))
    accuracy_low, accuracy_high = result["accuracy_delta_ci95"] or (
        float("nan"),
        float("nan"),
    )
    verdict = read_verdict(result)
    return "\n".join(
        [
            "# Paired comparison",
            "",
            f"Examples: {examples}",
            "",
            "| Metric | " + baseline_label + " | " + trained_label + " |",
            "|---|---:|---:|",
            f"| Correct | {result['baseline_correct']} | "
            f"{result['trained_correct']} |",
            f"| Accuracy | "
            f"{100 * result['baseline_correct'] / examples:.1f}% | "
            f"{100 * result['trained_correct'] / examples:.1f}% |",
            f"| Forwards/example | "
            f"{result['baseline_forwards_per_example']:.1f} | "
            f"{result['trained_forwards_per_example']:.1f} |",
            "",
            "## Quality (McNemar, paired)",
            "",
            f"- Both correct: {result['both_correct']}",
            f"- Only {baseline_label} correct: {result['only_baseline_correct']}",
            f"- Only {trained_label} correct: {result['only_trained_correct']}",
            f"- Neither correct: {result['neither_correct']}",
            f"- Two-sided exact p-value: {result['mcnemar_p_value']:.4f}",
            f"- Paired accuracy change: {100 * result['accuracy_delta']:+.2f} pp",
            f"- Bootstrap 95% CI: [{100 * accuracy_low:+.2f}, "
            f"{100 * accuracy_high:+.2f}] pp",
            "",
            "## Latency (paired)",
            "",
            f"- Mean forward change: {result['forward_delta_per_example']:+.2f} "
            f"per example",
            f"- Bootstrap 95% CI: [{low:+.2f}, {high:+.2f}]",
            "",
            "## Verdict",
            "",
            verdict,
            "",
        ]
    )


def read_verdict(result):
    quality = (
        "there is no statistically detectable quality change"
        if result["mcnemar_p_value"] >= 0.05
        else (
            "quality changed significantly"
            f" ({'in favour of trained' if result['only_trained_correct'] > result['only_baseline_correct'] else 'against trained'})"
        )
    )
    interval = result["forward_delta_ci95"]
    if interval is None:
        latency = "latency change not estimable"
    elif interval[1] < 0:
        latency = "latency improved significantly"
    elif interval[0] > 0:
        latency = "latency regressed significantly"
    else:
        latency = "latency change is not significant"
    return f"On these examples, {quality}, and {latency}."


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--trained-predictions", required=True)
    parser.add_argument("--baseline-label", default="base")
    parser.add_argument("--trained-label", default="trained")
    parser.add_argument("--min-example-id", type=int)
    parser.add_argument("--max-example-id", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    if (
        args.min_example_id is not None
        and args.max_example_id is not None
        and args.min_example_id >= args.max_example_id
    ):
        parser.error("min-example-id must be smaller than max-example-id")
    return args


if __name__ == "__main__":
    main()
