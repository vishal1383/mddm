#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mdm_probe.plots import (
    plot_anchor_answer_timeline,
    plot_anchor_canvas_results,
    plot_anchor_token_effects,
    plot_greedy_standard_accuracy_change,
    plot_policy_standard_accuracy_change,
)
from run_anchor_decode_impact import (
    aggregate_decode_rows,
    aggregate_anchor_token_effect_rows,
    aggregate_greedy_standard_accuracy_change_rows,
    make_greedy_standard_accuracy_change_example_rows,
    make_policy_standard_accuracy_change_example_rows,
    make_anchor_token_effect_rows,
    make_answer_timeline_rows,
    has_final_answer_marker,
    preview_text,
    rescore_decode_rows,
    rescore_trajectory_rows,
    write_answer_story_markdown,
    write_rows,
)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    impact_rows = rescore_decode_rows(read_jsonl(run_dir / "anchor_decode_impact.jsonl"))
    token_rows = load_token_rows(run_dir, args.examples_jsonl)
    completion_window_rows = completion_window_rows_from_cached(impact_rows, token_rows)
    aggregate_rows = aggregate_decode_rows(impact_rows)
    timeline_rows = make_answer_timeline_rows(impact_rows)
    anchor_effect_rows = make_anchor_token_effect_rows(timeline_rows, token_rows)
    anchor_effect_aggregate_rows = aggregate_anchor_token_effect_rows(anchor_effect_rows)
    greedy_standard_example_rows = make_greedy_standard_accuracy_change_example_rows(timeline_rows)
    greedy_standard_rows = aggregate_greedy_standard_accuracy_change_rows(greedy_standard_example_rows)
    policy_standard_example_rows = make_policy_standard_accuracy_change_example_rows(
        timeline_rows,
        policies=["greedy_ig", "prefix"],
    )
    policy_standard_rows = aggregate_greedy_standard_accuracy_change_rows(policy_standard_example_rows)

    write_rows(run_dir / "anchor_decode_impact.jsonl", impact_rows)
    write_rows(run_dir / "anchor_decode_impact.csv", impact_rows)
    write_rows(run_dir / "anchor_decode_aggregate.jsonl", aggregate_rows)
    write_rows(run_dir / "anchor_decode_aggregate.csv", aggregate_rows)
    write_rows(run_dir / "anchor_decode_tokens.jsonl", token_rows)
    write_rows(run_dir / "anchor_decode_tokens.csv", token_rows)
    write_rows(run_dir / "completion_window_audit.jsonl", completion_window_rows)
    write_rows(run_dir / "completion_window_audit.csv", completion_window_rows)
    write_rows(run_dir / "anchor_answer_timeline.jsonl", timeline_rows)
    write_rows(run_dir / "anchor_answer_timeline.csv", timeline_rows)
    write_rows(run_dir / "anchor_token_effects.jsonl", anchor_effect_rows)
    write_rows(run_dir / "anchor_token_effects.csv", anchor_effect_rows)
    write_rows(run_dir / "anchor_token_effects_by_position.jsonl", anchor_effect_aggregate_rows)
    write_rows(run_dir / "anchor_token_effects_by_position.csv", anchor_effect_aggregate_rows)
    write_rows(run_dir / "greedy_standard_accuracy_change_by_example.jsonl", greedy_standard_example_rows)
    write_rows(run_dir / "greedy_standard_accuracy_change_by_example.csv", greedy_standard_example_rows)
    write_rows(run_dir / "greedy_standard_accuracy_change.jsonl", greedy_standard_rows)
    write_rows(run_dir / "greedy_standard_accuracy_change.csv", greedy_standard_rows)
    write_rows(run_dir / "policy_standard_accuracy_change_by_example.jsonl", policy_standard_example_rows)
    write_rows(run_dir / "policy_standard_accuracy_change_by_example.csv", policy_standard_example_rows)
    write_rows(run_dir / "policy_standard_accuracy_change.jsonl", policy_standard_rows)
    write_rows(run_dir / "policy_standard_accuracy_change.csv", policy_standard_rows)
    write_answer_story_markdown(run_dir / "anchor_answer_story.md", timeline_rows, token_rows)
    rescore_cached_trajectories(run_dir, impact_rows)

    if not args.no_plots:
        plot_greedy_standard_accuracy_change(
            greedy_standard_rows,
            run_dir / "plots",
            example_rows=greedy_standard_example_rows,
        )
        plot_policy_standard_accuracy_change(policy_standard_rows, run_dir / "plots")
        if not args.main_plots_only:
            plot_anchor_answer_timeline(timeline_rows, token_rows, run_dir / "plots" / "answer_timeline")
            plot_anchor_canvas_results(timeline_rows, token_rows, run_dir / "plots" / "anchor_canvas")
            plot_anchor_token_effects(
                anchor_effect_aggregate_rows,
                run_dir / "plots" / "anchor_token_effects",
            )

    print(f"Wrote anchor answer story artifacts to {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-example anchor-answer timelines from cached decode-impact outputs."
    )
    parser.add_argument("run_dir", help="Directory containing anchor_decode_impact.jsonl")
    parser.add_argument(
        "--examples-jsonl",
        default=None,
        help=(
            "examples.jsonl from the probe run, used for indexed gold completion tokens. "
            "If omitted, reuses RUN_DIR/anchor_decode_tokens.jsonl."
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--main-plots-only",
        action="store_true",
        help="Only refresh aggregate report plots; skip per-example canvas/timeline plots.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_token_rows(run_dir: Path, examples_jsonl: str | None) -> list[dict]:
    if examples_jsonl:
        return token_rows_from_examples(read_jsonl(Path(examples_jsonl)))
    return read_jsonl(run_dir / "anchor_decode_tokens.jsonl")


def token_rows_from_examples(example_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for example in example_rows:
        anchorable = set(example.get("anchor_positions") or [])
        for pos, (token_id, token_text) in enumerate(
            zip(example.get("completion_token_ids", []), example.get("completion_token_texts", []))
        ):
            rows.append(
                {
                    "example_id": str(example["example_id"]),
                    "position": pos,
                    "token_id": int(token_id),
                    "token_text": str(token_text),
                    "anchorable": pos in anchorable,
                }
            )
    return rows


def completion_window_rows_from_cached(
    impact_rows: list[dict],
    token_rows: list[dict],
) -> list[dict]:
    by_example: dict[str, dict] = {}
    for row in impact_rows:
        if int(row.get("k", 0)) == 0:
            by_example.setdefault(str(row["example_id"]), row)

    token_text_by_example: dict[str, list[str]] = {}
    for row in token_rows:
        token_text_by_example.setdefault(str(row["example_id"]), []).append(str(row["token_text"]))

    out = []
    for example_id, row in sorted(by_example.items(), key=lambda item: natural_key(item[0])):
        dataset = str(row.get("dataset", ""))
        target_text = "".join(token_text_by_example.get(example_id, []))
        full_text = str(row.get("gold_completion", ""))
        full_count = row.get("full_completion_token_count", "")
        target_count = len(token_text_by_example.get(example_id, []))
        full_has_marker = has_final_answer_marker(full_text, dataset)
        target_has_marker = has_final_answer_marker(target_text, dataset)
        out.append(
            {
                "example_id": example_id,
                "dataset": dataset,
                "full_completion_token_count": full_count,
                "target_completion_token_count": target_count,
                "completion_truncated": str(row.get("completion_truncated", "")).lower() == "true"
                or (full_has_marker and not target_has_marker),
                "full_has_final_answer_marker": full_has_marker,
                "target_has_final_answer_marker": target_has_marker,
                "target_tail": preview_text(target_text[-240:], limit=240),
            }
        )
    return out


def natural_key(value: str) -> tuple:
    parts = []
    for part in value.split("."):
        parts.append(int(part) if part.isdigit() else part)
    return tuple(parts)


def rescore_cached_trajectories(run_dir: Path, impact_rows: list[dict]) -> None:
    path = run_dir / "anchor_decode_trajectories.jsonl"
    if not path.exists():
        return
    rows = rescore_trajectory_rows(read_jsonl(path), impact_rows)
    write_rows(run_dir / "anchor_decode_trajectories.jsonl", rows)
    write_rows(run_dir / "anchor_decode_trajectories.csv", rows)


if __name__ == "__main__":
    main()
