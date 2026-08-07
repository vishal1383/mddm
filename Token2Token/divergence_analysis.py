#!/usr/bin/env python3
"""Where two decoders' completions first disagree, and on what kind of token.

The claim from the 50-example sweep is that the post-anchor unlock forward
goes wrong specifically on *dependent* tokens -- arithmetic results, which only
become confident once their operands are on the canvas. Two examples cannot
support that. This categorises every divergence in a full run so the claim can
be checked or dropped.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

# A result position: the token right after "=" is the value a calculation
# produces, which is the class the unlock forward is hypothesised to harm.
RESULT_CONTEXT = re.compile(r"=\s*\$?-?[\d.,]*$")
OPERATOR_CONTEXT = re.compile(r"[-+*/x]\s*\$?[\d.,]*$")


def main() -> None:
    args = parse_args()
    baseline = load_rows(Path(args.baseline_predictions))
    trained = load_rows(Path(args.trained_predictions))
    shared = sorted(set(baseline) & set(trained), key=int)
    report = analyse(
        [baseline[key] for key in shared],
        [trained[key] for key in shared],
        args.baseline_label,
        args.trained_label,
    )
    rendered = render_markdown(report, args.baseline_label, args.trained_label)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


def load_rows(path: Path):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["example_id"])] = row
    return rows


def analyse(baseline, trained, baseline_label, trained_label):
    categories = Counter()
    outcomes = Counter()
    samples = []
    for base_row, trained_row in zip(baseline, trained):
        base_text = base_row["decoded_completion"]
        trained_text = trained_row["decoded_completion"]
        if base_text == trained_text:
            continue
        index = first_difference(base_text, trained_text)
        category = classify(base_text[:index])
        outcome = (
            "trained_only_correct"
            if trained_row["correct"] and not base_row["correct"]
            else "baseline_only_correct"
            if base_row["correct"] and not trained_row["correct"]
            else "same_outcome"
        )
        categories[(category, outcome)] += 1
        outcomes[outcome] += 1
        if outcome != "same_outcome" and len(samples) < 12:
            samples.append(
                {
                    "example_id": base_row["example_id"],
                    "category": category,
                    "outcome": outcome,
                    "context": base_text[max(0, index - 60) : index],
                    "baseline_next": base_text[index : index + 24],
                    "trained_next": trained_text[index : index + 24],
                }
            )
    return {
        "compared": len(baseline),
        "diverged": sum(outcomes.values()),
        "outcomes": outcomes,
        "categories": categories,
        "samples": samples,
    }


def first_difference(left, right):
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def classify(prefix):
    if RESULT_CONTEXT.search(prefix):
        return "calculation result"
    if OPERATOR_CONTEXT.search(prefix):
        return "operand"
    if prefix.endswith(("\n", ". ", ".")):
        return "sentence start"
    return "other prose"


def render_markdown(report, baseline_label, trained_label):
    lines = [
        "# Divergence analysis",
        "",
        f"Compared {report['compared']} examples; "
        f"{report['diverged']} produced different completions.",
        "",
        "| Outcome | Count |",
        "|---|---:|",
    ]
    for outcome, count in report["outcomes"].most_common():
        lines.append(f"| {outcome} | {count} |")
    lines += [
        "",
        "## First differing token, by kind and outcome",
        "",
        "| Token kind | Outcome | Count |",
        "|---|---|---:|",
    ]
    for (category, outcome), count in report["categories"].most_common():
        lines.append(f"| {category} | {outcome} | {count} |")
    if report["samples"]:
        lines += ["", "## Examples that changed the answer", ""]
        for sample in report["samples"]:
            lines += [
                f"- Example {sample['example_id']} ({sample['category']}, "
                f"{sample['outcome']})",
                f"  - context: `...{sample['context']}`",
                f"  - {baseline_label}: `{sample['baseline_next']}`",
                f"  - {trained_label}: `{sample['trained_next']}`",
            ]
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--trained-predictions", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--trained-label", default="trained")
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    main()
