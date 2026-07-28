#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timeline_rows = read_csv(run_dir / "anchor_answer_timeline.csv")
    impact_rows = read_csv(run_dir / "anchor_decode_impact.csv")
    policy_rows = read_csv(run_dir / "policy_standard_accuracy_change.csv")
    token_rows = read_csv(run_dir / "anchor_decode_tokens.csv")

    token_info = build_token_info(token_rows)
    questions = load_questions(Path(args.examples_jsonl)) if args.examples_jsonl else {}
    write_long_trajectory_csv(out_dir / "all_examples_k_trajectory.csv", timeline_rows)
    write_pivot_csv(out_dir / "all_examples_k_pivot.csv", timeline_rows)
    write_hypothesis_csv(out_dir / "hypothesis_experiments.csv")

    report = build_report(
        run_dir=run_dir,
        out_dir=out_dir,
        timeline_rows=timeline_rows,
        impact_rows=impact_rows,
        policy_rows=policy_rows,
        token_info=token_info,
        questions=questions,
        max_examples=args.max_examples,
    )
    report_path = out_dir / "completion_hypothesis_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report_path)
    print(out_dir / "all_examples_k_trajectory.csv")
    print(out_dir / "all_examples_k_pivot.csv")
    print(out_dir / "hypothesis_experiments.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a detailed k-by-k anchor decode report with hypotheses."
    )
    parser.add_argument(
        "run_dir",
        help="Full decode-impact output dir, e.g. outputs/decode_impact_all/llada-8b_gsm8k",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/decode_impact_analysis/llada-8b_gsm8k",
        help="Separate output directory for the report and appendices.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=1,
        help="Representative examples per case type in the Markdown report.",
    )
    parser.add_argument(
        "--examples-jsonl",
        help="Optional GSM8K-style JSONL containing question/prompt text in dataset order.",
    )
    return parser.parse_args()


def build_report(
    *,
    run_dir: Path,
    out_dir: Path,
    timeline_rows: list[dict[str, str]],
    impact_rows: list[dict[str, str]],
    policy_rows: list[dict[str, str]],
    token_info: dict[str, dict[str, Any]],
    questions: dict[str, str],
    max_examples: int,
) -> str:
    examples = sorted({row["example_id"] for row in timeline_rows}, key=natural_key)
    policies = ["greedy_ig", "prefix"]
    by_timeline = index_timeline(timeline_rows)
    by_impact = index_impact(impact_rows)
    walkthrough_id = select_walkthrough_example(by_timeline, by_impact, token_info)

    lines: list[str] = [
        "# Anchor Decode Completion Report",
        "",
        "This report summarizes the full GSM8K anchor-decode run and gives concrete k-by-k examples for the main failure and recovery patterns.",
        "",
        "## Headline",
        "",
        (
            "Greedy IG anchors are most useful early. Accuracy peaks around k=2-4, then declines as additional scattered anchors are added. "
            "The k=10 failures are mostly caused by isolated high-impact fragments, especially late numeric tokens, rather than by coherent answer copying."
        ),
        "",
        "## Main Table",
        "",
        main_accuracy_table(policy_rows),
        "",
        "## Prefix Sanity Check",
        "",
        prefix_sanity_check(by_timeline),
        "",
        "## How To Read The K Rows",
        "",
        (
            "Each k row is a fresh decode from the original masked completion canvas. "
            "For k=0, no anchors are placed. For k>0, the first k anchors from that policy are fixed to their gold tokens, "
            "then the remaining tokens are filled with the same standard confidence-greedy decoder. "
            "So the k=10 greedy row means: place the first 10 greedy-IG gold anchors, then decode the rest."
        ),
        "",
        "## Concrete K=10 Failure Walkthrough",
        "",
    ]
    if walkthrough_id:
        lines.extend(
            detailed_walkthrough_section(
                walkthrough_id,
                "greedy_ig",
                by_timeline[(walkthrough_id, "greedy_ig")],
                token_info.get(walkthrough_id, {}),
                questions.get(walkthrough_id, ""),
            )
        )
    else:
        lines.extend(["No clean greedy k=10 failure walkthrough was found.", ""])

    lines.extend(
        [
        "## Final Answer Anchor Pickup",
        "",
        final_answer_pickup_table(impact_rows, token_info, policies),
        "",
        (
            "Final-answer tokens are available in the target window for only a subset of examples. "
            "At k=10, greedy IG picks at least one final-answer number token in some examples, but usually not the whole answer span. "
            "Prefix anchors pick no final-answer tokens at k<=10 because they are early left-to-right anchors."
        ),
        "",
        "## K=10 Case Counts",
        "",
        case_count_table(by_timeline, policies, before_k=0, after_k=10),
        "",
        "<!-- pagebreak -->",
        "",
        "## K=2 To K=10 Transitions",
        "",
        (
            "This isolates the main phenomenon: k=2 is near the greedy peak, while k=10 is much lower. "
            "Greedy has many more regressions than recoveries after k=2."
        ),
        "",
        case_count_table(by_timeline, policies, before_k=2, after_k=10),
        "",
        "## First Harmful Anchor Analysis",
        "",
        first_harmful_anchor_table(by_timeline, token_info, policies),
        "",
        "## Hypotheses And Experiments",
        "",
        hypotheses_table(),
        "",
        "<!-- pagebreak -->",
        "",
        "## Additional K-By-K Examples",
        "",
        (
            "Each example uses the same horizontal form: accumulated anchor canvas -> decoded answer sentence -> extracted answer."
        ),
        "",
        ]
    )

    sections = [
        ("Greedy k=2 Correct -> k=10 Wrong", "greedy_ig", lambda rows: is_correct(rows, 2) and not is_correct(rows, 10)),
        ("Greedy k=2 Wrong -> k=10 Correct", "greedy_ig", lambda rows: (not is_correct(rows, 2)) and is_correct(rows, 10)),
    ]
    for title, policy, predicate in sections:
        selected = select_examples(by_timeline, policy, predicate, max_examples)
        lines.extend([f"### {title}", ""])
        if not selected:
            lines.extend(["No examples found.", ""])
            continue
        for example_id in selected:
            lines.extend(
                example_section(
                    example_id,
                    policy,
                    by_timeline[(example_id, policy)],
                    token_info.get(example_id, {}),
                    questions.get(example_id, ""),
                )
            )

    lines.extend(["### Greedy k=10 Final-Answer Fragment Harms", ""])
    for example_id in select_final_answer_fragment_harms(by_timeline, by_impact, token_info, max_examples):
        if example_id == walkthrough_id:
            continue
        lines.extend(
            example_section(
                example_id,
                "greedy_ig",
                by_timeline[(example_id, "greedy_ig")],
                token_info.get(example_id, {}),
                questions.get(example_id, ""),
            )
        )

    return "\n".join(lines)


def main_accuracy_table(policy_rows: list[dict[str, str]]) -> str:
    by = {(row["policy"], int(row["k"])): row for row in policy_rows}
    rows = []
    for k in range(0, 11):
        greedy = by.get(("greedy_ig", k), {})
        prefix = by.get(("prefix", k), {})
        rows.append(
            [
                str(k),
                pct(greedy.get("mean_answer_correct")),
                pp(greedy.get("mean_answer_accuracy_change_pp")),
                pct(prefix.get("mean_answer_correct")),
                pp(prefix.get("mean_answer_accuracy_change_pp")),
            ]
        )
    return md_table(
        ["k", "greedy IG acc", "greedy delta", "prefix acc", "prefix delta"],
        rows,
    )


def prefix_sanity_check(
    by_timeline: dict[tuple[str, str], dict[int, dict[str, str]]],
) -> str:
    comparisons = [
        paired_accuracy_comparison(by_timeline, "prefix", k - 1, k)
        for k in range(1, 11)
    ]
    rows = [
        [
            f"k={result['before_k']} -> k={result['after_k']}",
            str(result["recoveries"]),
            str(result["regressions"]),
            f"{100.0 * result['delta']:+.2f} pp",
            (
                f"[{100.0 * result['ci_low']:+.2f}, "
                f"{100.0 * result['ci_high']:+.2f}] pp"
            ),
            format_p_value(result["p_value"]),
        ]
        for result in comparisons
    ]
    overall = paired_accuracy_comparison(by_timeline, "prefix", 0, 10)
    corrected_alpha = 0.05 / len(comparisons)
    unadjusted_hits = sum(result["p_value"] < 0.05 for result in comparisons)
    corrected_hits = sum(result["p_value"] < corrected_alpha for result in comparisons)
    n = overall["n"]

    return "\n\n".join(
        [
            (
                f"The prefix policy uses deterministic left-to-right gold anchors over {n} examples, "
                "but every k is decoded afresh. Adding one correct token can therefore redirect the "
                "remaining confidence-greedy decode, so accuracy is not mathematically required to "
                "increase with k. These fluctuations are not Monte Carlo decoding noise."
            ),
            md_table(
                [
                    "comparison",
                    "wrong -> correct",
                    "correct -> wrong",
                    "paired change",
                    "95% CI",
                    "exact p",
                ],
                rows,
            ),
            (
                f"Only {unadjusted_hits}/10 adjacent changes have unadjusted p<0.05, and "
                f"{corrected_hits}/10 survive a Bonferroni threshold of {corrected_alpha:.3f}. "
                "The local rises and falls should therefore be treated as paired-sample uncertainty, "
                "not as a stable oscillating pattern."
            ),
            (
                f"The broader prefix effect is clearer: k=0 -> k=10 changes accuracy by "
                f"{100.0 * overall['delta']:+.2f} pp "
                f"(95% CI [{100.0 * overall['ci_low']:+.2f}, "
                f"{100.0 * overall['ci_high']:+.2f}] pp; "
                f"exact p={format_p_value(overall['p_value'])})."
            ),
        ]
    )


def paired_accuracy_comparison(
    by_timeline: dict[tuple[str, str], dict[int, dict[str, str]]],
    policy: str,
    before_k: int,
    after_k: int,
) -> dict[str, Any]:
    deltas: list[int] = []
    for (_, row_policy), group in by_timeline.items():
        if row_policy != policy or before_k not in group or after_k not in group:
            continue
        before = int(is_correct(group, before_k))
        after = int(is_correct(group, after_k))
        deltas.append(after - before)

    n = len(deltas)
    recoveries = sum(delta == 1 for delta in deltas)
    regressions = sum(delta == -1 for delta in deltas)
    mean_delta = sum(deltas) / n if n else 0.0
    if n > 1:
        variance = sum((delta - mean_delta) ** 2 for delta in deltas) / (n - 1)
        margin = 1.96 * math.sqrt(variance / n)
    else:
        margin = 0.0

    return {
        "before_k": before_k,
        "after_k": after_k,
        "n": n,
        "recoveries": recoveries,
        "regressions": regressions,
        "delta": mean_delta,
        "ci_low": mean_delta - margin,
        "ci_high": mean_delta + margin,
        "p_value": exact_mcnemar_p(recoveries, regressions),
    }


def exact_mcnemar_p(recoveries: int, regressions: int) -> float:
    discordant = recoveries + regressions
    if discordant == 0:
        return 1.0
    tail_end = min(recoveries, regressions)
    tail = sum(math.comb(discordant, i) for i in range(tail_end + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def format_p_value(value: float) -> str:
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def final_answer_pickup_table(
    impact_rows: list[dict[str, str]],
    token_info: dict[str, dict[str, Any]],
    policies: list[str],
) -> str:
    rows = []
    n_examples = len(token_info)
    answer_available = sum(1 for info in token_info.values() if info["answer_positions"])
    marker_available = sum(1 for info in token_info.values() if info["marker_positions"])
    for policy in policies:
        for k in [0, 2, 4, 10]:
            group = [
                row for row in impact_rows
                if row["policy"] == policy and int(row["k"]) == k
            ]
            any_answer = 0
            all_answer = 0
            marker_hit = 0
            for row in group:
                info = token_info.get(row["example_id"], {})
                answer_positions = set(info.get("answer_positions", []))
                marker_positions = set(info.get("marker_positions", []))
                anchors = set(json_list_int(row.get("anchors", "[]")))
                if answer_positions & anchors:
                    any_answer += 1
                if answer_positions and answer_positions <= anchors:
                    all_answer += 1
                if marker_positions & anchors:
                    marker_hit += 1
            rows.append(
                [
                    policy_label(policy),
                    str(k),
                    fraction(any_answer, len(group)),
                    fraction(any_answer, answer_available),
                    fraction(all_answer, answer_available),
                    fraction(marker_hit, marker_available),
                ]
            )
    intro = (
        f"Target window has mappable final-answer number tokens for `{answer_available}/{n_examples}` examples "
        f"and a `####` marker for `{marker_available}/{n_examples}` examples.\n\n"
    )
    return intro + md_table(
        [
            "policy",
            "k",
            "any final-answer token hit",
            "hit among available",
            "whole answer hit among available",
            "`####` hit among available",
        ],
        rows,
    )


def case_count_table(
    by_timeline: dict[tuple[str, str], dict[int, dict[str, str]]],
    policies: list[str],
    *,
    before_k: int,
    after_k: int,
) -> str:
    rows = []
    for policy in policies:
        counts = Counter()
        for (example_id, row_policy), group in by_timeline.items():
            if row_policy != policy or before_k not in group or after_k not in group:
                continue
            before = bool_value(group[before_k].get("answer_correct"))
            after = bool_value(group[after_k].get("answer_correct"))
            if before and after:
                counts["stay correct"] += 1
            elif before and not after:
                counts["regress"] += 1
            elif not before and after:
                counts["recover"] += 1
            else:
                counts["stay wrong"] += 1
        rows.append(
            [
                policy_label(policy),
                str(counts["stay correct"]),
                str(counts["regress"]),
                str(counts["recover"]),
                str(counts["stay wrong"]),
                signed_number(counts["recover"] - counts["regress"]),
            ]
        )
    return md_table(
        ["policy", "stay correct", "regress", "recover", "stay wrong", "net"],
        rows,
    )


def first_harmful_anchor_table(
    by_timeline: dict[tuple[str, str], dict[int, dict[str, str]]],
    token_info: dict[str, dict[str, Any]],
    policies: list[str],
) -> str:
    sections = []
    for policy in policies:
        first_k = Counter()
        token_types = Counter()
        late = 0
        n = 0
        for (example_id, row_policy), group in by_timeline.items():
            if row_policy != policy or 2 not in group or 10 not in group:
                continue
            if not (bool_value(group[2].get("answer_correct")) and not bool_value(group[10].get("answer_correct"))):
                continue
            for k in range(3, 11):
                row = group.get(k)
                if row and not bool_value(row.get("answer_correct")):
                    n += 1
                    first_k[k] += 1
                    tokens = json_list(row.get("new_anchor_token_texts", "[]"))
                    positions = json_list_int(row.get("new_anchor_positions", "[]"))
                    token = str(tokens[0]) if tokens else ""
                    position = positions[0] if positions else None
                    token_types[token_type(token)] += 1
                    target_len = int(token_info.get(example_id, {}).get("target_len") or 1)
                    if position is not None and target_len > 1 and position / (target_len - 1) > 0.75:
                        late += 1
                    break
        sections.append(
            "\n\n".join(
                [
                    f"**{policy_label(policy)}**",
                    f"k=2 correct -> k=10 wrong: {n} examples.",
                    f"First bad k counts: {counter_summary(first_k)}.",
                    f"First bad anchor types: {counter_summary(token_types)}.",
                    f"First bad anchor is late: {fraction(late, n)}.",
                ]
            )
        )
    return "\n\n".join(sections)


def hypotheses_table() -> str:
    rows = [
        [
            "Partial-number anchors",
            "Greedy often adds one digit from a larger number; harmed examples show `104 -> 1044`, `3000 -> 30000`, `80 -> 8`.",
            "When a selected anchor is numeric, expand to the whole contiguous number span.",
            "If regressions drop, isolated digits were the main failure mode.",
        ],
        [
            "Late-fragment anchors",
            "First harmful greedy anchor is late in the completion for most k2->k10 regressions.",
            "Run `greedy_ig_no_late`, excluding anchors after 75% of the target window.",
            "If k=5..10 stays closer to k=2 accuracy, late anchors are brittle.",
        ],
        [
            "Final-answer fragment anchors",
            "Greedy k10 harmed cases hit final-answer tokens more often than helped cases, but rarely the full answer span.",
            "Compare `ban_final_answer_digits` vs `force_full_final_answer_span`.",
            "If full-span helps but partial-span hurts, answer fragments are unsafe unless grouped.",
        ],
        [
            "Objective mismatch",
            "IG optimizes confidence/entropy over tokens, not final-answer correctness.",
            "On a subset, choose anchors by decoded-answer correctness or gold-answer probability proxy.",
            "If answer-aware anchors keep improving past k=2, IG is the wrong late-stage objective.",
        ],
        [
            "Gold-path conflict",
            "Gold rationale anchors may conflict with the model's own preferred solution path.",
            "Anchor the model's standard decoded tokens at the same positions instead of gold tokens.",
            "If harm disappears, the issue is conflict between gold rationale path and generated path.",
        ],
    ]
    sections = []
    for hypothesis, evidence, experiment, resolution in rows:
        sections.append(
            "\n\n".join(
                [
                    f"**{hypothesis}**",
                    f"Evidence: {evidence}",
                    f"Experiment: {experiment}",
                    f"What would resolve it: {resolution}",
                ]
            )
        )
    return "\n\n".join(sections)


def detailed_walkthrough_section(
    example_id: str,
    policy: str,
    group: dict[int, dict[str, str]],
    info: dict[str, Any],
    question: str,
) -> list[str]:
    gold = group[min(group)]["gold_answer"]
    first_bad_k = first_regression_k(group)
    target_text = shorten(info.get("target_text", ""), 900)
    lines = [
        f"### Example {example_id}: a partial final-answer anchor makes the answer worse",
        "",
        "**Question**",
        "",
        question.strip() or "Question text was not saved in the decode output.",
        "",
        "**Gold answer**",
        "",
        code(gold),
        "",
        "**Gold completion**",
        "",
        f"> {target_text}",
        "",
        "**Horizontal anchor-and-decode sequence**",
        "",
        (
            "The canvas shows the ten positions eventually selected by greedy IG, kept in their actual left-to-right "
            "order in the gold completion. `____` means that position has not yet been fixed. "
            "The number in parentheses is the token position."
        ),
        "",
        "```text",
        *horizontal_decode_lines(group, info),
        "```",
        "",
    ]
    if first_bad_k is not None:
        row = group[first_bad_k]
        positions = json_list_int(row.get("new_anchor_positions", "[]"))
        token = json_list(row.get("new_anchor_token_texts", "[]"))
        position = positions[0] if positions else None
        role = token_role(position, info) if position is not None else "unknown"
        before = group[first_bad_k - 1]
        lines.extend(
            [
                "**Why the answer gets worse**",
                "",
                (
                    f"The decisive transition is k={first_bad_k - 1} -> k={first_bad_k}. "
                    f"Before placing `{anchor_label(row)}`, the decoder returns "
                    f"`{before.get('decoded_answer', '')}`. After placing it, the decoder returns "
                    f"`{row.get('decoded_answer', '')}`. The new anchor is a {role}."
                ),
                "",
                (
                    "Only one token from the final number is fixed; the complete answer span is not fixed. The unanchored "
                    "positions can therefore decode a copy of the answer before the forced digit. The model satisfies the gold "
                    "digit constraint locally but produces the wrong complete number."
                ),
                "",
                (
                    "This is the concrete partial-number failure behind the aggregate decline: additional anchors can be individually "
                    "gold-correct while jointly steering the free decoder toward a worse final string."
                ),
                "",
            ]
        )
    return lines


def anchor_order_rows(group: dict[int, dict[str, str]], info: dict[str, Any]) -> list[list[str]]:
    rows = []
    for k, row in sorted(group.items()):
        if not (1 <= k <= 10):
            continue
        positions = json_list_int(row.get("new_anchor_positions", "[]"))
        tokens = json_list(row.get("new_anchor_token_texts", "[]"))
        for position, token in zip(positions, tokens):
            rows.append(
                [
                    f"k={k}",
                    str(position),
                    code(display_token(str(token))),
                    token_role(position, info),
                    shorten(target_context(position, info), 160),
                ]
            )
    return rows


def horizontal_decode_lines(
    group: dict[int, dict[str, str]],
    info: dict[str, Any],
) -> list[str]:
    final_row = group.get(10) or group[max(group)]
    slot_positions = sorted(set(json_list_int(final_row.get("anchors", "[]"))))
    token_by_pos = info.get("token_by_pos", {})
    lines = []
    for k, row in sorted(group.items()):
        if not (0 <= k <= 10):
            continue
        active = set(json_list_int(row.get("anchors", "[]")))
        canvas = " ".join(
            canvas_slot(position, str(token_by_pos.get(position, "")), info) if position in active else "____"
            for position in slot_positions
        )
        sentence = shorten(answer_sentence(row.get("decoded_text", ""), row.get("decoded_answer", "")), 170)
        answer = row.get("decoded_answer", "")
        status = "CORRECT" if bool_value(row.get("answer_correct")) else "WRONG"
        new_anchor = "no anchors" if k == 0 else f"add {anchor_label(row)}"
        lines.append(
            f"k={k:<2} {canvas}  ->  {sentence!r}  ->  answer {answer} ({status}); {new_anchor}"
        )
    return lines


def canvas_slot(position: int, token: str, info: dict[str, Any]) -> str:
    text = plain_token(token)
    if position in set(info.get("answer_positions", [])):
        text = f"FINAL:{text}"
    elif position in set(info.get("marker_positions", [])):
        text = f"MARKER:{text}"
    return f"[{text}({position})]"


def first_regression_k(group: dict[int, dict[str, str]]) -> int | None:
    for k in range(1, 11):
        previous = group.get(k - 1)
        current = group.get(k)
        if previous and current and bool_value(previous.get("answer_correct")) and not bool_value(current.get("answer_correct")):
            return k
    return None


def row_effect(row: dict[str, str], previous: dict[str, str] | None) -> str:
    correct = bool_value(row.get("answer_correct"))
    if previous is None:
        return "baseline correct" if correct else "baseline wrong"
    previous_correct = bool_value(previous.get("answer_correct"))
    if previous_correct and correct:
        return "still correct"
    if previous_correct and not correct:
        return "breaks answer here"
    if (not previous_correct) and correct:
        return "recovers answer"
    return "still wrong"


def anchor_label(row: dict[str, str]) -> str:
    positions = json_list_int(row.get("new_anchor_positions", "[]"))
    tokens = json_list(row.get("new_anchor_token_texts", "[]"))
    if not positions:
        return shorten(row.get("action", ""), 80)
    return ", ".join(f"{plain_token(str(token))}({position})" for position, token in zip(positions, tokens))


def plain_token(token: str) -> str:
    return token.replace("\n", "\\n").strip() or "<space>"


def answer_sentence(text: str, decoded_answer: str) -> str:
    raw = str(text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines() if line.strip()]
    if decoded_answer:
        for line in reversed(lines):
            if decoded_answer in line:
                return line
    normalized = re.sub(r"\s+", " ", raw).strip()
    if not normalized:
        return ""
    chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", normalized) if chunk.strip()]
    if decoded_answer:
        for chunk in reversed(chunks):
            if decoded_answer in chunk:
                return chunk
    return chunks[-1] if chunks else normalized


def token_role(position: int, info: dict[str, Any]) -> str:
    if position in set(info.get("answer_positions", [])):
        return "final-answer token"
    if position in set(info.get("marker_positions", [])):
        return "`####` marker"
    token = info.get("token_by_pos", {}).get(position, "")
    return token_type(str(token))


def target_context(position: int, info: dict[str, Any], radius: int = 4) -> str:
    tokens = info.get("tokens", [])
    pieces = []
    for pos, token in tokens:
        if position - radius <= pos <= position + radius:
            label = f"{pos}:{display_token(str(token))}"
            pieces.append(f"[{label}]" if pos == position else label)
    return " ".join(pieces)


def display_token(token: str) -> str:
    return repr(token.replace("\n", "\\n"))


def example_section(
    example_id: str,
    policy: str,
    group: dict[int, dict[str, str]],
    info: dict[str, Any],
    question: str,
) -> list[str]:
    gold = group[min(group)]["gold_answer"]
    lines = [
        f"#### Example {example_id} - {policy_label(policy)}",
        "",
        "**Question**",
        "",
        question.strip() or "Question text was not saved in the decode output.",
        "",
        "**Gold answer**",
        "",
        code(gold),
        "",
        "```text",
        *horizontal_decode_lines(group, info),
        "```",
        "",
    ]
    return lines


def select_examples(
    by_timeline: dict[tuple[str, str], dict[int, dict[str, str]]],
    policy: str,
    predicate,
    limit: int,
) -> list[str]:
    out = []
    for (example_id, row_policy), group in sorted(by_timeline.items(), key=lambda item: natural_key(item[0][0])):
        if row_policy == policy and predicate(group):
            out.append(example_id)
            if len(out) >= limit:
                break
    return out


def select_walkthrough_example(
    by_timeline: dict[tuple[str, str], dict[int, dict[str, str]]],
    by_impact: dict[tuple[str, str, int], dict[str, str]],
    token_info: dict[str, dict[str, Any]],
) -> str | None:
    scored: list[tuple[int, int, tuple[Any, ...], str]] = []
    for (example_id, policy), group in by_timeline.items():
        if policy != "greedy_ig" or 2 not in group or 10 not in group:
            continue
        if not (bool_value(group[2].get("answer_correct")) and not bool_value(group[10].get("answer_correct"))):
            continue
        info = token_info.get(example_id, {})
        answer_positions = set(info.get("answer_positions", []))
        marker_positions = set(info.get("marker_positions", []))
        impact = by_impact.get((example_id, policy, 10), {})
        anchors = set(json_list_int(impact.get("anchors", "[]")))
        if not (anchors & (answer_positions | marker_positions)):
            continue
        first_bad = first_regression_k(group) or 99
        bad_positions = set(json_list_int(group.get(first_bad, {}).get("new_anchor_positions", "[]")))
        if bad_positions & answer_positions and partial_answer_duplication(group, first_bad):
            priority = 0
        elif bad_positions & answer_positions and first_bad >= 8:
            priority = 1
        elif bad_positions & answer_positions:
            priority = 2
        elif bad_positions & marker_positions:
            priority = 3
        else:
            priority = 4
        scored.append((priority, -first_bad, natural_key(example_id), example_id))
    if scored:
        return sorted(scored)[0][3]

    fallback = select_examples(
        by_timeline,
        "greedy_ig",
        lambda rows: is_correct(rows, 2) and not is_correct(rows, 10),
        1,
    )
    return fallback[0] if fallback else None


def partial_answer_duplication(group: dict[int, dict[str, str]], k: int) -> bool:
    previous = normalize_number(group.get(k - 1, {}).get("decoded_answer", ""))
    current = normalize_number(group.get(k, {}).get("decoded_answer", ""))
    gold = normalize_number(group.get(k, {}).get("gold_answer", ""))
    if not previous or not current or not gold or previous != gold:
        return False
    return len(current) > len(gold) and gold in current


def select_final_answer_fragment_harms(
    by_timeline: dict[tuple[str, str], dict[int, dict[str, str]]],
    by_impact: dict[tuple[str, str, int], dict[str, str]],
    token_info: dict[str, dict[str, Any]],
    limit: int,
) -> list[str]:
    out = []
    for (example_id, policy), group in sorted(by_timeline.items(), key=lambda item: natural_key(item[0][0])):
        if policy != "greedy_ig" or 0 not in group or 10 not in group:
            continue
        if not (bool_value(group[0].get("answer_correct")) and not bool_value(group[10].get("answer_correct"))):
            continue
        impact = by_impact.get((example_id, policy, 10), {})
        anchors = set(json_list_int(impact.get("anchors", "[]")))
        answer_positions = set(token_info.get(example_id, {}).get("answer_positions", []))
        if anchors & answer_positions:
            out.append(example_id)
            if len(out) >= limit:
                break
    return out


def write_long_trajectory_csv(path: Path, timeline_rows: list[dict[str, str]]) -> None:
    rows = []
    for row in timeline_rows:
        rows.append(
            {
                "example_id": row["example_id"],
                "policy": row["policy"],
                "k": row["k"],
                "gold_answer": row["gold_answer"],
                "decoded_answer": row["decoded_answer"],
                "answer_correct": row["answer_correct"],
                "delta_vs_standard": row["answer_correct_delta_from_baseline"],
                "delta_vs_previous": row["answer_correct_delta_from_previous"],
                "new_anchor_positions": row["new_anchor_positions"],
                "new_anchor_token_texts": row["new_anchor_token_texts"],
                "anchor_path": row["anchor_path"],
                "decoded_text_preview": row["decoded_text_preview"],
            }
        )
    write_csv(path, rows)


def write_pivot_csv(path: Path, timeline_rows: list[dict[str, str]]) -> None:
    grouped = index_timeline(timeline_rows)
    rows = []
    for (example_id, policy), group in sorted(grouped.items(), key=lambda item: (natural_key(item[0][0]), item[0][1])):
        row = {
            "example_id": example_id,
            "policy": policy,
            "gold_answer": group[min(group)]["gold_answer"],
        }
        for k in range(0, 11):
            current = group.get(k, {})
            row[f"k{k}_answer"] = current.get("decoded_answer", "")
            row[f"k{k}_correct"] = current.get("answer_correct", "")
            row[f"k{k}_new_anchor"] = current.get("new_anchor_token_texts", "")
        rows.append(row)
    write_csv(path, rows)


def write_hypothesis_csv(path: Path) -> None:
    rows = [
        {"hypothesis": row[0], "evidence": row[1], "experiment": row[2], "resolution_signal": row[3]}
        for row in [
            [
                "Partial-number anchors",
                "Isolated digits appear in many regressions.",
                "Expand selected numeric anchors to contiguous number spans.",
                "Regressions decrease at k=5..10.",
            ],
            [
                "Late-fragment anchors",
                "First harmful greedy anchors are usually late.",
                "Exclude anchors after 75% of target window.",
                "Late-k accuracy improves or stays near k=2.",
            ],
            [
                "Final-answer fragment anchors",
                "Partial final-answer digit hits are over-represented in harms.",
                "Ban final-answer digits or force full final-answer span.",
                "Full-span helps while partial-span hurts.",
            ],
            [
                "Objective mismatch",
                "IG peak occurs early; later high-IG anchors hurt answer accuracy.",
                "Use answer-aware anchor scoring on a subset.",
                "Answer-aware anchors do not show same k=10 degradation.",
            ],
            [
                "Gold-path conflict",
                "Gold rationale may conflict with model-generated path.",
                "Anchor standard decoded tokens at same positions.",
                "Harms drop when anchors match model path.",
            ],
        ]
    ]
    write_csv(path, rows)


def build_token_info(token_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    by_example: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in token_rows:
        by_example[str(row["example_id"])].append(row)

    out: dict[str, dict[str, Any]] = {}
    for example_id, rows in by_example.items():
        rows = sorted(rows, key=lambda row: int(row["position"]))
        text = ""
        spans = []
        for row in rows:
            token_text = str(row["token_text"])
            start = len(text)
            text += token_text
            end = len(text)
            spans.append((int(row["position"]), start, end, token_text))

        marker_positions: list[int] = []
        answer_positions: list[int] = []
        answer_text = ""
        marker = re.search(r"####", text)
        if marker:
            marker_start, marker_end = marker.span()
            marker_positions = [
                pos for pos, start, end, _ in spans
                if start < marker_end and end > marker_start
            ]
            line_end = text.find("\n", marker.end())
            if line_end < 0:
                line_end = len(text)
            answer_match = re.search(r"-?\d[\d,]*(?:\.\d+)?", text[marker.end():line_end])
            if answer_match:
                answer_start = marker.end() + answer_match.start()
                answer_end = marker.end() + answer_match.end()
                answer_text = answer_match.group(0)
                answer_positions = [
                    pos for pos, start, end, _ in spans
                    if start < answer_end and end > answer_start
                ]

        out[example_id] = {
            "target_len": len(rows),
            "target_has_marker": bool(marker),
            "marker_positions": marker_positions,
            "answer_positions": answer_positions,
            "answer_text": answer_text,
            "target_text": text,
            "tokens": [(pos, token_text) for pos, _, _, token_text in spans],
            "token_by_pos": {pos: token_text for pos, _, _, token_text in spans},
        }
    return out


def index_timeline(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[int, dict[str, str]]]:
    out: dict[tuple[str, str], dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        out[(row["example_id"], row["policy"])][int(row["k"])] = row
    return out


def index_impact(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, str]]:
    return {
        (row["example_id"], row["policy"], int(row["k"])): row
        for row in rows
    }


def is_correct(group: dict[int, dict[str, str]], k: int) -> bool:
    return bool_value(group.get(k, {}).get("answer_correct"))


def token_type(token: str) -> str:
    text = token.strip()
    if text == "####":
        return "final_marker"
    if text in {"<<", ">>"} or "<<" in text or ">>" in text:
        return "calc_marker"
    if re.fullmatch(r"[-+]?[0-9][0-9,]*(?:\.[0-9]+)?", text):
        return "number"
    if text in {"+", "-", "*", "/", "=", "x", "×"} or any(op in text for op in ["+", "-", "*", "/", "="]):
        return "operator"
    if text == "":
        return "space"
    if re.fullmatch(r"\W+", text):
        return "punct"
    return "word"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_questions(path: Path) -> dict[str, str]:
    questions = {}
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row.get("example_id", index))
            question = row.get("question") or row.get("prompt") or ""
            questions[example_id] = str(question).strip()
    return questions


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def json_list(value: str) -> list[Any]:
    try:
        parsed = json.loads(value) if value else []
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def json_list_int(value: str) -> list[int]:
    out = []
    for item in json_list(value):
        try:
            out.append(int(item))
        except Exception:
            pass
    return out


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def normalize_number(value: Any) -> str:
    return re.sub(r"[,$\s]", "", str(value))


def pct(value: Any) -> str:
    if value in {None, ""}:
        return ""
    return f"{100.0 * float(value):.2f}%"


def pp(value: Any) -> str:
    if value in {None, ""}:
        return ""
    return f"{float(value):+.2f} pp"


def fraction(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return f"{numerator}/0"
    return f"{numerator}/{denominator} ({100.0 * numerator / denominator:.1f}%)"


def signed_float(value: Any) -> str:
    if value in {None, ""}:
        return ""
    return f"{float(value):+.0f}"


def signed_number(value: int) -> str:
    return f"{value:+d}"


def counter_summary(counter: Counter) -> str:
    return ", ".join(f"{key}:{value}" for key, value in counter.most_common())


def policy_label(policy: str) -> str:
    if policy == "greedy_ig":
        return "greedy IG"
    if policy == "prefix":
        return "prefix"
    return policy


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md_cell(str(cell)) for cell in row) + " |")
    return "\n".join(lines)


def md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def code(text: str) -> str:
    return "`" + text.replace("`", "'") + "`"


def shorten(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value)))


if __name__ == "__main__":
    main()
