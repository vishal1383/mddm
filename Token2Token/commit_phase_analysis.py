#!/usr/bin/env python3
"""Relate how a completion was committed to whether it came out right.

Requires predictions written with --record-commit-phase. Each position carries
the rule that placed it: 1 catalyst, 2 cleanup, 3 first-forward threshold,
4 post-anchor unlock.

This is correlational. Harder questions plausibly produce a different mix of
commit rules for reasons that have nothing to do with the rules being harmful,
so a difference here supports the error-amplification hypothesis without
establishing it. It is still the only evidence available short of an
intervention that suppresses unlock commits and re-decodes.
"""
from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
import statistics

PHASE_NAMES = {
    0: "never committed",
    1: "catalyst",
    2: "cleanup",
    3: "first-forward threshold",
    4: "post-anchor unlock",
}


def main() -> None:
    args = parse_args()
    rows = [
        row
        for row in read_rows(Path(args.predictions))
        if isinstance(row.get("commit_phase"), list)
    ]
    if not rows:
        raise SystemExit(
            "no rows carry commit_phase; rerun the eval with --record-commit-phase"
        )
    report = analyse(rows)
    rendered = render_markdown(report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


def read_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def analyse(rows):
    shares = {phase: {"correct": [], "wrong": []} for phase in PHASE_NAMES}
    for row in rows:
        phases = row["commit_phase"]
        total = len(phases) or 1
        bucket = "correct" if row["correct"] else "wrong"
        for phase in PHASE_NAMES:
            shares[phase][bucket].append(
                sum(1 for value in phases if int(value) == phase) / total
            )
    comparisons = []
    for phase, name in PHASE_NAMES.items():
        correct = shares[phase]["correct"]
        wrong = shares[phase]["wrong"]
        if not correct or not wrong:
            continue
        if not any(correct) and not any(wrong):
            continue
        comparisons.append(
            {
                "phase": name,
                "correct_share": statistics.fmean(correct),
                "wrong_share": statistics.fmean(wrong),
                "difference": statistics.fmean(wrong) - statistics.fmean(correct),
                "welch_t": welch_t(wrong, correct),
            }
        )
    return {
        "examples": len(rows),
        "correct": sum(1 for row in rows if row["correct"]),
        "comparisons": sorted(
            comparisons, key=lambda item: -abs(item["difference"])
        ),
    }


def welch_t(left, right):
    """Welch t statistic; the samples have unequal size and variance."""
    if len(left) < 2 or len(right) < 2:
        return None
    left_variance = statistics.variance(left) / len(left)
    right_variance = statistics.variance(right) / len(right)
    denominator = sqrt(left_variance + right_variance)
    if denominator == 0:
        return None
    return (statistics.fmean(left) - statistics.fmean(right)) / denominator


def render_markdown(report):
    lines = [
        "# Commit-phase composition versus correctness",
        "",
        f"Examples: {report['examples']} "
        f"({report['correct']} correct, "
        f"{report['examples'] - report['correct']} wrong)",
        "",
        "Share of each completion's positions placed by each rule.",
        "",
        "| Commit rule | Correct answers | Wrong answers | Difference | Welch t |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report["comparisons"]:
        statistic = item["welch_t"]
        lines.append(
            f"| {item['phase']} | {100 * item['correct_share']:.1f}% | "
            f"{100 * item['wrong_share']:.1f}% | "
            f"{100 * item['difference']:+.1f} pp | "
            f"{'n/a' if statistic is None else f'{statistic:+.2f}'} |"
        )
    lines += [
        "",
        "A positive difference means the rule places a larger share of the "
        "completion when the answer comes out wrong. Treat |t| under about 2 "
        "as no signal, and read any signal as correlational: harder questions "
        "produce a different mix of commit rules for reasons unrelated to the "
        "rules being harmful.",
        "",
    ]
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    main()
