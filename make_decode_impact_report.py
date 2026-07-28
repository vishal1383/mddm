#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out else run_dir / "decode_impact_report.md"

    aggregate_rows = read_csv(run_dir / "anchor_decode_aggregate.csv")
    timeline_rows = read_csv(run_dir / "anchor_answer_timeline.csv")
    token_effect_rows = read_csv_if_exists(run_dir / "anchor_token_effects_by_position.csv")
    greedy_change_rows = read_csv_if_exists(run_dir / "greedy_standard_accuracy_change.csv")
    policy_change_rows = read_csv_if_exists(run_dir / "policy_standard_accuracy_change.csv")

    if args.report_mode == "full":
        report = build_report(
            run_dir,
            aggregate_rows,
            timeline_rows,
            token_effect_rows,
            max_examples=args.max_examples,
            focus_policy=args.focus_policy,
        )
    else:
        report = build_greedy_standard_report(
            run_dir,
            aggregate_rows,
            timeline_rows,
            greedy_change_rows,
            policy_change_rows,
            max_examples=args.max_examples,
            focus_policy=args.focus_policy,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a polished Markdown report for cheated-anchor decode impact outputs."
    )
    parser.add_argument("run_dir", help="Directory containing anchor_decode_aggregate.csv and anchor_answer_timeline.csv")
    parser.add_argument("--out", default=None, help="Default: RUN_DIR/decode_impact_report.md")
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--focus-policy", default="greedy_ig")
    parser.add_argument(
        "--report-mode",
        choices=["greedy", "full"],
        default="greedy",
        help="Default is a simple greedy-vs-standard report. Use full for all policies/plots.",
    )
    return parser.parse_args()


def build_greedy_standard_report(
    run_dir: Path,
    aggregate_rows: list[dict[str, str]],
    timeline_rows: list[dict[str, str]],
    greedy_change_rows: list[dict[str, str]],
    policy_change_rows: list[dict[str, str]],
    *,
    max_examples: int,
    focus_policy: str,
) -> str:
    example_ids = sorted({row["example_id"] for row in timeline_rows}, key=natural_key)
    greedy_rows = greedy_change_rows or greedy_standard_rows(aggregate_rows, focus_policy)
    max_k = max((int(row["k"]) for row in greedy_rows), default=0)
    baseline_acc = greedy_rows[0]["standard_mean_answer_correct"] if greedy_rows else ""

    lines: list[str] = [
        "# Greedy Anchors vs Left-to-Right Control",
        "",
        (
            "This report only asks one thing: as we place greedy cheated anchors, "
            "how does final-answer accuracy change relative to the standard no-anchor decode? "
            "It also compares against a left-to-right prefix anchor control. "
            "Both curves are averaged over all examples/runs in this directory."
        ),
        "",
        f"- Run directory: `{run_dir}`",
        f"- Examples analyzed: `{len(example_ids)}`",
        f"- Anchor policy: `{focus_policy}`",
        "- Left-to-right control: `prefix` anchors, i.e. the first k anchorable completion tokens.",
        f"- Standard decode: `k=0`",
        f"- Largest greedy k in this run: `{max_k}`",
        f"- Standard mean answer accuracy: `{pct(baseline_acc)}`",
        "",
        "## Mean Accuracy Change",
        "",
    ]

    control_plot = run_dir / "plots" / "greedy_vs_left_to_right_accuracy_change.png"
    plot = control_plot if control_plot.exists() else run_dir / "plots" / "greedy_standard_mean_answer_accuracy_change.png"
    if plot.exists():
        lines.extend([f"![anchor policy accuracy change]({rel(plot, run_dir)})", ""])
    else:
        lines.extend([f"Missing plot: `{plot}`", ""])

    table_rows = policy_change_rows or greedy_rows
    if table_rows:
        lines.append(
            markdown_table(
                [
                    "policy",
                    "x",
                    "anchors placed",
                    "mean answer acc",
                    "change vs standard",
                    "95% CI",
                    "n",
                ],
                [
                    [
                        policy_label(row.get("policy", "")),
                        row["x_label"],
                        row["k"],
                        pct(row.get("mean_answer_correct")),
                        pp(row.get("mean_answer_accuracy_change_pp")),
                        pp(row.get("ci95_answer_accuracy_change_pp")),
                        row.get("n_examples", row.get("n", "")),
                    ]
                    for row in sorted(table_rows, key=lambda row: (policy_order(row.get("policy", "")), int(row["k"])))
                ],
            )
        )
        lines.append("")

    flip_summary = greedy_k_flip_summary(timeline_rows, focus_policy, max_k)
    if flip_summary:
        lines.extend(
            [
                f"## Concrete k={max_k} Cases",
                "",
                (
                    f"At `k={max_k}`, examples are counted relative to each example's own "
                    "standard no-anchor decode."
                ),
                "",
                markdown_table(
                    ["case type", "count"],
                    [
                        ["standard correct -> greedy wrong", str(len(flip_summary["harmed"]))],
                        ["standard wrong -> greedy correct", str(len(flip_summary["helped"]))],
                        ["unchanged", str(len(flip_summary["unchanged"]))],
                    ],
                ),
                "",
            ]
        )
        for title, key in [
            ("Greedy Makes a Correct Standard Answer Wrong", "harmed"),
            ("Greedy Fixes a Wrong Standard Answer", "helped"),
        ]:
            examples = flip_summary[key][: max_examples]
            if not examples:
                continue
            lines.extend([f"### {title}", ""])
            lines.append(
                markdown_table(
                    [
                        "example",
                        "gold",
                        "standard answer",
                        f"greedy k={max_k} answer",
                        "change",
                        "k anchor path",
                    ],
                    [
                        [
                            row["example_id"],
                            row["gold_answer"],
                            row["standard_decoded_answer"],
                            row["greedy_decoded_answer"],
                            signed(row["answer_correct_delta_from_baseline"]),
                            shorten(row["anchor_path"], 180),
                        ]
                        for row in examples
                    ],
                )
            )
            lines.append("")

    lines.extend(
        [
            "## Greedy Examples",
            "",
            (
                "Each example keeps only the greedy anchor path. "
                "`change vs standard` is answer correctness at that k minus answer correctness at k=0."
            ),
            "",
        ]
    )

    selected_examples = select_examples(timeline_rows, focus_policy, max_examples)
    by_example_policy: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in timeline_rows:
        by_example_policy[(row["example_id"], row["policy"])].append(row)

    for example_id in selected_examples:
        group = by_example_policy.get((example_id, focus_policy), [])
        if not group:
            continue
        group = sorted(group, key=lambda row: int(row["k"]))
        canvas_plot = run_dir / "plots" / "anchor_canvas" / f"{safe(example_id)}_{safe(focus_policy)}_anchor_canvas.png"
        lines.extend([f"### Example {example_id}", ""])
        if canvas_plot.exists():
            lines.extend([f"![{example_id} greedy anchor canvas]({rel(canvas_plot, run_dir)})", ""])
        lines.append(
            markdown_table(
                [
                    "x",
                    "anchor placed",
                    "anchor path",
                    "decoded answer",
                    "correct",
                    "change vs standard",
                    "token acc",
                ],
                [
                    [
                        "standard" if int(row["k"]) == 0 else row["k"],
                        row["action"],
                        row["anchor_path"],
                        row["decoded_answer"],
                        row["answer_correct"],
                        signed(row.get("answer_correct_delta_from_baseline")),
                        pct(row.get("token_accuracy")),
                    ]
                    for row in group
                ],
            )
        )
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            "- `greedy_standard_accuracy_change.csv`: one row per k for the mean accuracy-change plot.",
            "- `policy_standard_accuracy_change.csv`: mean curves for greedy IG and left-to-right prefix controls.",
            "- `policy_standard_accuracy_change_by_example.csv`: one row per example/run per k and policy.",
            "- `plots/greedy_vs_left_to_right_accuracy_change.png`: the main control comparison plot.",
            "- `plots/anchor_canvas/`: per-example greedy canvas illustrations.",
            "- `anchor_answer_timeline.csv`: per-example anchor placement -> decoded answer table.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(
    run_dir: Path,
    aggregate_rows: list[dict[str, str]],
    timeline_rows: list[dict[str, str]],
    token_effect_rows: list[dict[str, str]],
    *,
    max_examples: int,
    focus_policy: str,
) -> str:
    policies = sorted({row["policy"] for row in aggregate_rows})
    example_ids = sorted({row["example_id"] for row in timeline_rows}, key=natural_key)
    max_k = max((int(row["k"]) for row in aggregate_rows), default=0)

    lines: list[str] = []
    lines.extend(
        [
            "# Cheated-Anchor Decode Impact Report",
            "",
            "## Executive Summary",
            "",
            (
                "This report asks whether revealing ground-truth completion tokens as anchors "
                "changes the final decoded answer under standard masked confidence decoding."
            ),
            "",
            f"- Run directory: `{run_dir}`",
            f"- Examples analyzed: `{len(example_ids)}`",
            f"- Anchor policies: `{', '.join(policies)}`",
            f"- Largest k in this run: `{max_k}`",
            "",
        ]
    )

    final_rows = [row for row in aggregate_rows if int(row["k"]) == max_k]
    if final_rows:
        lines.extend(
            [
                "At the largest k, the headline comparison is:",
                "",
                markdown_table(
                    [
                        "policy",
                        "mean answer correct",
                        "mean answer changed",
                        "mean token acc",
                        "n",
                    ],
                    [
                        [
                            row["policy"],
                            pct(row.get("mean_answer_correct")),
                            pct(row.get("mean_answer_changed_from_baseline")),
                            pct(row.get("mean_token_accuracy")),
                            row.get("n", ""),
                        ]
                        for row in sorted(final_rows, key=lambda row: row["policy"])
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Method",
            "",
            "For each prompt, the script first runs a baseline decode with no completion anchors.",
            "Then, for each anchor policy and each `k`, it reveals the gold token at selected completion-token positions, runs the same decode, and extracts the final answer.",
            "",
            "Important columns:",
            "",
            "- `answer_correct`: extracted decoded answer matches the gold final answer.",
            "- `answer_changed_from_baseline`: extracted decoded answer differs from the no-anchor baseline answer.",
            "- `token_accuracy`: fraction of decoded completion tokens matching the gold completion tokens.",
            "- `anchor_path`: token positions and gold token text revealed at that step.",
            "",
        ]
    )

    lines.extend(["## Aggregate Plots", ""])
    for plot in [
        run_dir / "plots" / "aggregate_decode_answer_impact.png",
        run_dir / "plots" / "aggregate_mean_answer_change.png",
    ]:
        if plot.exists():
            lines.extend([f"![{plot.stem}]({rel(plot, run_dir)})", ""])
        else:
            lines.extend([f"Missing plot: `{plot}`", ""])

    lines.extend(
        [
            "## Anchor Token Effects",
            "",
            (
                "Here each newly placed anchor is assigned to a normalized token slot `t1...t10`. "
                "`mean Δ correct_ans` is the average change in final-answer correctness after anchoring tokens in that slot."
            ),
            "",
        ]
    )
    for plot in [
        run_dir / "plots" / "anchor_token_effects" / "aggregate_correct_change_by_anchor_position.png",
        run_dir / "plots" / "anchor_token_effects" / f"{safe(focus_policy)}_correct_change_by_anchor_position.png",
    ]:
        if plot.exists():
            lines.extend([f"![{plot.stem}]({rel(plot, run_dir)})", ""])

    focus_effect_rows = [
        row for row in token_effect_rows if row.get("policy") == focus_policy
    ] or token_effect_rows
    if focus_effect_rows:
        lines.append(
            markdown_table(
                [
                    "policy",
                    "slot",
                    "mean Δ correct_ans vs base",
                    "mean Δ correct_ans vs prev",
                    "mean answer correct",
                    "n",
                ],
                [
                    [
                        row["policy"],
                        row.get("position_bin_label", f"t{int(row['position_bin']) + 1}"),
                        signed(row.get("mean_correct_delta_from_baseline")),
                        signed(row.get("mean_correct_delta_from_previous")),
                        pct(row.get("mean_answer_correct")),
                        row.get("n", ""),
                    ]
                    for row in sorted(
                        focus_effect_rows,
                        key=lambda row: (row["policy"], int(row["position_bin"])),
                    )
                ],
            )
        )
        lines.append("")

    lines.extend(["## Aggregate Table", ""])
    lines.append(
        markdown_table(
            [
                "policy",
                "k",
                "answer correct",
                "answer changed",
                "token acc",
                "non-anchor token acc",
                "n",
            ],
            [
                [
                    row["policy"],
                    row["k"],
                    pct(row.get("mean_answer_correct")),
                    pct(row.get("mean_answer_changed_from_baseline")),
                    pct(row.get("mean_token_accuracy")),
                    pct(row.get("mean_non_anchor_token_accuracy")),
                    row.get("n", ""),
                ]
                for row in sorted(aggregate_rows, key=lambda row: (row["policy"], int(row["k"])))
            ],
        )
    )
    lines.append("")

    lines.extend(
        [
            "## Example Walkthroughs",
            "",
            (
                "Each walkthrough reads left-to-right over the gold completion token positions. "
                "Each canvas starts from all blanks `_`; revealed cells are cheated gold anchors. "
                "The right-side annotation shows the decoded final answer and the correctness delta."
            ),
            "",
        ]
    )

    selected_examples = select_examples(timeline_rows, focus_policy, max_examples)
    by_example_policy: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in timeline_rows:
        by_example_policy[(row["example_id"], row["policy"])].append(row)

    for example_id in selected_examples:
        lines.extend([f"### Example {example_id}", ""])
        for policy in preferred_policy_order(policies, focus_policy):
            group = by_example_policy.get((example_id, policy), [])
            if not group:
                continue
            group = sorted(group, key=lambda row: int(row["k"]))
            canvas_plot = run_dir / "plots" / "anchor_canvas" / f"{safe(example_id)}_{safe(policy)}_anchor_canvas.png"
            plot = run_dir / "plots" / "answer_timeline" / f"{safe(example_id)}_{safe(policy)}_anchor_answer_timeline.png"
            lines.extend([f"#### {policy}", ""])
            if canvas_plot.exists():
                lines.extend([f"![{example_id} {policy} anchor canvas]({rel(canvas_plot, run_dir)})", ""])
            elif plot.exists():
                lines.extend([f"![{example_id} {policy} timeline]({rel(plot, run_dir)})", ""])
            lines.append(
                markdown_table(
                    [
                        "k",
                        "anchor action",
                        "anchor path",
                        "decoded answer",
                        "correct",
                        "Δcorrect base",
                        "changed",
                        "token acc",
                    ],
                    [
                        [
                            row["k"],
                            row["action"],
                            row["anchor_path"],
                            row["decoded_answer"],
                            row["answer_correct"],
                            signed(row.get("answer_correct_delta_from_baseline")),
                            row["answer_changed_from_baseline"],
                            pct(row.get("token_accuracy")),
                        ]
                        for row in group
                    ],
                )
            )
            lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            "- `anchor_answer_story.md`: full per-example anchor story.",
            "- `anchor_answer_timeline.csv`: compact table for anchor placement -> answer.",
            "- `anchor_token_effects.csv`: one row per anchor placement, with correctness deltas.",
            "- `anchor_token_effects_by_position.csv`: aggregate mean correctness change by normalized token slot.",
            "- `anchor_decode_impact.csv`: full decoded text for every example/policy/k.",
            "- `anchor_decode_tokens.csv`: indexed gold completion tokens.",
            "- `plots/anchor_canvas/`: blank-canvas anchor placement examples.",
            "- `plots/anchor_token_effects/`: aggregate correctness-change plots by token slot.",
            "- `plots/answer_timeline/`: per-example token-position illustrations.",
            "",
        ]
    )
    return "\n".join(lines)


def select_examples(rows: list[dict[str, str]], focus_policy: str, max_examples: int) -> list[str]:
    by_example: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["policy"] == focus_policy:
            by_example[row["example_id"]].append(row)
    scored: list[tuple[int, int, str]] = []
    for example_id, group in by_example.items():
        changed = sum(str(row.get("answer_changed_from_baseline")) == "True" for row in group)
        incorrect = sum(str(row.get("answer_correct")) == "False" for row in group)
        scored.append((changed + incorrect, len(group), example_id))
    if not scored:
        return sorted({row["example_id"] for row in rows}, key=natural_key)[:max_examples]
    return [example_id for _, _, example_id in sorted(scored, key=lambda item: (-item[0], natural_key(item[2])))[:max_examples]]


def greedy_k_flip_summary(
    rows: list[dict[str, str]],
    focus_policy: str,
    k: int,
) -> dict[str, list[dict[str, str]]]:
    by_example: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["policy"] == focus_policy:
            by_example[row["example_id"]][int(row["k"])] = row

    summary: dict[str, list[dict[str, str]]] = {
        "harmed": [],
        "helped": [],
        "unchanged": [],
    }
    for example_id, group in sorted(by_example.items(), key=lambda item: natural_key(item[0])):
        if 0 not in group or k not in group:
            continue
        standard = group[0]
        greedy = group[k]
        standard_correct = bool_value(standard.get("answer_correct"))
        greedy_correct = bool_value(greedy.get("answer_correct"))
        out = {
            "example_id": example_id,
            "gold_answer": greedy.get("gold_answer", ""),
            "standard_decoded_answer": standard.get("decoded_answer", ""),
            "greedy_decoded_answer": greedy.get("decoded_answer", ""),
            "answer_correct_delta_from_baseline": greedy.get("answer_correct_delta_from_baseline", ""),
            "anchor_path": greedy.get("anchor_path", ""),
        }
        if standard_correct and not greedy_correct:
            summary["harmed"].append(out)
        elif not standard_correct and greedy_correct:
            summary["helped"].append(out)
        else:
            summary["unchanged"].append(out)
    return summary


def preferred_policy_order(policies: list[str], focus_policy: str) -> list[str]:
    order = [focus_policy, "prefix", "suffix", "middle_cluster", "maximally_separated"]
    return [policy for policy in order if policy in policies] + [
        policy for policy in policies if policy not in order
    ]


def policy_label(policy: str) -> str:
    labels = {
        "greedy_ig": "greedy IG",
        "prefix": "left-to-right prefix",
    }
    return labels.get(policy, policy)


def policy_order(policy: str) -> int:
    order = {
        "greedy_ig": 0,
        "prefix": 1,
    }
    return order.get(policy, 99)


def greedy_standard_rows(
    aggregate_rows: list[dict[str, str]],
    focus_policy: str,
) -> list[dict[str, str]]:
    focus_rows = [
        row for row in aggregate_rows
        if row.get("policy") == focus_policy and row.get("mean_answer_correct") not in {None, ""}
    ]
    if not focus_rows:
        return []
    focus_rows = sorted(focus_rows, key=lambda row: int(row["k"]))
    baseline = next((row for row in focus_rows if int(row["k"]) == 0), focus_rows[0])
    try:
        baseline_acc = float(baseline["mean_answer_correct"])
    except (TypeError, ValueError):
        return []

    out: list[dict[str, str]] = []
    for row in focus_rows:
        try:
            mean_acc = float(row["mean_answer_correct"])
        except (TypeError, ValueError):
            continue
        delta = mean_acc - baseline_acc
        out.append(
            {
                "policy": focus_policy,
                "k": row["k"],
                "x_label": "standard" if int(row["k"]) == 0 else row["k"],
                "standard_mean_answer_correct": str(baseline_acc),
                "mean_answer_correct": str(mean_acc),
                "mean_answer_accuracy_change": str(delta),
                "mean_answer_accuracy_change_pp": str(100.0 * delta),
                "n": row.get("n", ""),
            }
        )
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv(path)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(md_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(md_cell(str(value)) for value in row) + " |")
    return "\n".join(out)


def md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def bool_value(value: str | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def pct(value: str | None) -> str:
    if value in {None, ""}:
        return ""
    try:
        return f"{100.0 * float(value):.1f}%"
    except ValueError:
        return str(value)


def pp(value: str | None) -> str:
    if value in {None, ""}:
        return ""
    try:
        return f"{float(value):+.1f} pp"
    except ValueError:
        return str(value)


def signed(value: str | None) -> str:
    if value in {None, ""}:
        return ""
    try:
        return f"{float(value):+.2f}"
    except ValueError:
        return str(value)


def shorten(value: str, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "example"


def natural_key(value: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value))]


if __name__ == "__main__":
    main()
