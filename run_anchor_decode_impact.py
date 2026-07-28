#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import math
import json
from pathlib import Path
import re
from typing import Any

from mdm_probe.anchors import layout_anchors
from mdm_probe.datasets import load_examples
from mdm_probe.models import load_probe_model
from mdm_probe.plots import (
    plot_anchor_answer_timeline,
    plot_anchor_canvas_results,
    plot_anchor_token_effects,
    plot_decode_impact_results,
    plot_decode_trajectory_results,
    plot_greedy_standard_accuracy_change,
    plot_policy_standard_accuracy_change,
)
from mdm_probe.sampling import anchor_candidates, run_greedy_k_probe
from mdm_probe.types import DecodeResult, EncodedExample, ProbeExample


GREEDY_POLICY_NAMES = {
    "information_gain": "greedy_ig",
    "p_gt_gain": "greedy_p_gt",
    "max_p_gain": "greedy_max_p",
}


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        args.limit = None
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_probe_model(
        args.model,
        model_id=args.model_id,
        device=args.device,
        mask_token_id=args.mask_token_id,
        mask_token=args.mask_token,
        prompt_format=args.prompt_format,
    )
    examples = load_probe_examples(args)
    cached_greedy = load_cached_greedy(args.probe_run_dir) if args.probe_run_dir else {}
    policies = parse_policies(args.anchor_policies)

    decode_rows: list[dict] = []
    trajectory_rows: list[dict] = []
    token_rows: list[dict] = []
    completion_window_rows: list[dict] = []
    write_run_config(out_dir / "run_config.json", args)

    for index, example in enumerate(examples):
        max_completion_tokens = completion_limit_for_example(args, example)
        encoded = model.encode_example(
            example,
            max_completion_tokens=max_completion_tokens,
        )
        token_rows.extend(make_token_rows(encoded))
        completion_window_rows.append(make_completion_window_row(encoded, example.dataset))
        max_k = min(args.max_k, len(anchor_candidates(encoded)))
        print(
            f"[{index + 1}/{len(examples)}] {encoded.example_id}: "
            f"T={encoded.completion_length}, max_k={max_k}"
        )
        if completion_window_rows[-1]["full_has_final_answer_marker"] and not completion_window_rows[-1]["target_has_final_answer_marker"]:
            print(
                f"  warning: target window misses final-answer marker "
                f"(full_T={completion_window_rows[-1]['full_completion_token_count']}, "
                f"target_T={completion_window_rows[-1]['target_completion_token_count']})"
            )

        baseline = model.decode_completion(
            encoded,
            [],
            steps=args.decode_steps,
            tokens_per_step=args.tokens_per_step,
            suppress_mask_token=not args.allow_mask_output,
        )
        baseline_eval = evaluate_decode(encoded, baseline, [], example.dataset)
        if not args.no_trajectories:
            add_trajectory_rows(
                trajectory_rows,
                encoded,
                baseline,
                policy="baseline",
                k=0,
                eval_row=baseline_eval,
            )

        anchor_specs = build_anchor_specs(
            model,
            encoded,
            policies,
            max_k=max_k,
            batch_size=args.batch_size,
            score_reduction=args.score_reduction,
            selection_metric=args.select_metric,
            cached_greedy=cached_greedy.get(str(encoded.example_id)),
        )

        policy_names = sorted({policy for policy, _, _ in anchor_specs}) or ["baseline"]
        for policy in policy_names:
            decode_rows.append(
                make_decode_row(
                    encoded,
                    baseline,
                    policy=policy,
                    k=0,
                    anchors=[],
                    eval_row=baseline_eval,
                    baseline_eval=baseline_eval,
                    dataset=example.dataset,
                )
            )

        for policy, k, anchors in anchor_specs:
            result = model.decode_completion(
                encoded,
                anchors,
                steps=args.decode_steps,
                tokens_per_step=args.tokens_per_step,
                suppress_mask_token=not args.allow_mask_output,
            )
            eval_row = evaluate_decode(encoded, result, anchors, example.dataset)
            decode_rows.append(
                make_decode_row(
                    encoded,
                    result,
                    policy=policy,
                    k=k,
                    anchors=anchors,
                    eval_row=eval_row,
                    baseline_eval=baseline_eval,
                    dataset=example.dataset,
                )
            )
            if not args.no_trajectories:
                add_trajectory_rows(
                    trajectory_rows,
                    encoded,
                    result,
                    policy=policy,
                    k=k,
                    eval_row=eval_row,
                )

    aggregate_rows = aggregate_decode_rows(decode_rows)
    timeline_rows = make_answer_timeline_rows(decode_rows)
    anchor_effect_rows = make_anchor_token_effect_rows(timeline_rows, token_rows)
    anchor_effect_aggregate_rows = aggregate_anchor_token_effect_rows(anchor_effect_rows)
    greedy_standard_example_rows = make_greedy_standard_accuracy_change_example_rows(timeline_rows)
    greedy_standard_rows = aggregate_greedy_standard_accuracy_change_rows(greedy_standard_example_rows)
    policy_standard_example_rows = make_policy_standard_accuracy_change_example_rows(
        timeline_rows,
        policies=["greedy_ig", "prefix"],
    )
    policy_standard_rows = aggregate_greedy_standard_accuracy_change_rows(policy_standard_example_rows)

    write_rows(out_dir / "anchor_decode_impact.jsonl", decode_rows)
    write_rows(out_dir / "anchor_decode_impact.csv", decode_rows)
    write_rows(out_dir / "anchor_decode_aggregate.jsonl", aggregate_rows)
    write_rows(out_dir / "anchor_decode_aggregate.csv", aggregate_rows)
    write_rows(out_dir / "anchor_decode_trajectories.jsonl", trajectory_rows)
    write_rows(out_dir / "anchor_decode_trajectories.csv", trajectory_rows)
    write_rows(out_dir / "anchor_decode_tokens.jsonl", token_rows)
    write_rows(out_dir / "anchor_decode_tokens.csv", token_rows)
    write_rows(out_dir / "completion_window_audit.jsonl", completion_window_rows)
    write_rows(out_dir / "completion_window_audit.csv", completion_window_rows)
    write_rows(out_dir / "anchor_answer_timeline.jsonl", timeline_rows)
    write_rows(out_dir / "anchor_answer_timeline.csv", timeline_rows)
    write_rows(out_dir / "anchor_token_effects.jsonl", anchor_effect_rows)
    write_rows(out_dir / "anchor_token_effects.csv", anchor_effect_rows)
    write_rows(out_dir / "anchor_token_effects_by_position.jsonl", anchor_effect_aggregate_rows)
    write_rows(out_dir / "anchor_token_effects_by_position.csv", anchor_effect_aggregate_rows)
    write_rows(out_dir / "greedy_standard_accuracy_change_by_example.jsonl", greedy_standard_example_rows)
    write_rows(out_dir / "greedy_standard_accuracy_change_by_example.csv", greedy_standard_example_rows)
    write_rows(out_dir / "greedy_standard_accuracy_change.jsonl", greedy_standard_rows)
    write_rows(out_dir / "greedy_standard_accuracy_change.csv", greedy_standard_rows)
    write_rows(out_dir / "policy_standard_accuracy_change_by_example.jsonl", policy_standard_example_rows)
    write_rows(out_dir / "policy_standard_accuracy_change_by_example.csv", policy_standard_example_rows)
    write_rows(out_dir / "policy_standard_accuracy_change.jsonl", policy_standard_rows)
    write_rows(out_dir / "policy_standard_accuracy_change.csv", policy_standard_rows)
    write_answer_story_markdown(out_dir / "anchor_answer_story.md", timeline_rows, token_rows)

    if not args.no_plots:
        plot_greedy_standard_accuracy_change(
            greedy_standard_rows,
            plots_dir,
            example_rows=greedy_standard_example_rows,
        )
        plot_policy_standard_accuracy_change(policy_standard_rows, plots_dir)
        if not args.main_plots_only:
            plot_decode_impact_results(decode_rows, plots_dir)
            plot_anchor_answer_timeline(timeline_rows, token_rows, plots_dir / "answer_timeline")
            plot_anchor_canvas_results(timeline_rows, token_rows, plots_dir / "anchor_canvas")
            plot_anchor_token_effects(
                anchor_effect_aggregate_rows,
                plots_dir / "anchor_token_effects",
            )
            plot_decode_trajectory_results(
                trajectory_rows,
                plots_dir / "trajectories",
                max_groups=args.max_trajectory_plots,
            )

    print(f"Saved cheated-anchor decode impact outputs to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure how gold-token anchor cheating changes full MDM decodes."
    )
    parser.add_argument("--model", required=True, help="llada-8b, dream-7b, or HF model path")
    parser.add_argument("--model-id", default=None, help="Optional HF model id/path override")
    parser.add_argument("--dataset", default="gsm8k", help="gsm8k, humaneval, or local dataset name")
    parser.add_argument(
        "--probe-run-dir",
        default=None,
        help="Optional run_probe.py output dir; reuses examples.jsonl and greedy_k.jsonl when present.",
    )
    parser.add_argument("--data-path", default=None, help="Optional local .jsonl/.json/.csv")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--completion-field", default="completion")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=128,
        help="Completion decode length; use 0 for no truncation.",
    )
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument(
        "--anchor-policies",
        default="greedy,prefix,suffix,middle_cluster,maximally_separated",
        help="Comma-separated policies: greedy, prefix, suffix, middle_cluster, maximally_separated.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--score-reduction", choices=["sum", "mean"], default="mean")
    parser.add_argument(
        "--select-metric",
        choices=["information_gain", "p_gt_gain", "max_p_gain"],
        default="information_gain",
        help="Metric used when greedy anchors must be computed.",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=None,
        help="Number of confidence-decoding iterations. Default fills one token per step.",
    )
    parser.add_argument(
        "--tokens-per-step",
        type=int,
        default=None,
        help="Fill a fixed number of tokens per decode step instead of using --decode-steps.",
    )
    parser.add_argument("--out-dir", default="outputs/anchor_decode_impact")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--mask-token", default=None)
    parser.add_argument("--prompt-format", choices=["auto", "raw", "chat"], default="auto")
    parser.add_argument("--allow-mask-output", action="store_true")
    parser.add_argument("--max-trajectory-plots", type=int, default=50)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--main-plots-only",
        action="store_true",
        help="Only write aggregate report plots; skip per-example plot families.",
    )
    parser.add_argument(
        "--no-trajectories",
        action="store_true",
        help="Do not store per-step decode trajectory rows. Useful for large dataset runs.",
    )
    return parser.parse_args()


def load_probe_examples(args: argparse.Namespace) -> list[ProbeExample]:
    if args.probe_run_dir and args.data_path is None:
        examples = load_cached_examples(Path(args.probe_run_dir))
        if args.limit is not None:
            examples = examples[: args.limit]
        return examples

    return load_examples(
        args.dataset,
        split=args.split,
        limit=args.limit,
        data_path=args.data_path,
        prompt_field=args.prompt_field,
        completion_field=args.completion_field,
    )


def load_cached_examples(run_dir: Path) -> list[ProbeExample]:
    rows = read_jsonl(run_dir / "examples.jsonl")
    examples: list[ProbeExample] = []
    for row in rows:
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"example_id", "dataset", "prompt", "completion"}
        }
        metadata["cached_completion_tokens"] = row.get("completion_tokens")
        examples.append(
            ProbeExample(
                example_id=str(row["example_id"]),
                dataset=str(row.get("dataset", "cached")),
                prompt=str(row["prompt"]),
                completion=str(row["completion"]),
                metadata=metadata,
            )
        )
    return examples


def completion_limit_for_example(args: argparse.Namespace, example: ProbeExample) -> int | None:
    del example
    if args.max_completion_tokens == 0:
        return None
    return args.max_completion_tokens


def load_cached_greedy(run_dir_arg: str | None) -> dict[str, dict[int, list[int]]]:
    if run_dir_arg is None:
        return {}
    path = Path(run_dir_arg) / "greedy_k.jsonl"
    if not path.exists():
        return {}
    by_example: dict[str, dict[int, list[int]]] = defaultdict(dict)
    for row in read_jsonl(path):
        by_example[str(row["example_id"])][int(row["k"])] = [int(v) for v in row["anchors"]]
    return by_example


def parse_policies(raw: str) -> list[str]:
    policies = [part.strip() for part in raw.split(",") if part.strip()]
    allowed = {"greedy", "prefix", "suffix", "middle_cluster", "maximally_separated"}
    unknown = [policy for policy in policies if policy not in allowed]
    if unknown:
        raise ValueError(f"Unknown anchor policies: {unknown}. Allowed: {sorted(allowed)}")
    return policies


def build_anchor_specs(
    model,
    encoded: EncodedExample,
    policies: list[str],
    *,
    max_k: int,
    batch_size: int,
    score_reduction: str,
    selection_metric: str,
    cached_greedy: dict[int, list[int]] | None,
) -> list[tuple[str, int, list[int]]]:
    specs: list[tuple[str, int, list[int]]] = []
    eligible = anchor_candidates(encoded)
    if max_k <= 0:
        return specs

    if "greedy" in policies:
        greedy_name = GREEDY_POLICY_NAMES[selection_metric]
        greedy_by_k = cached_greedy or {}
        if not all(k in greedy_by_k for k in range(1, max_k + 1)):
            greedy_rows, _ = run_greedy_k_probe(
                model,
                encoded,
                max_k=max_k,
                batch_size=batch_size,
                score_reduction=score_reduction,
                selection_metric=selection_metric,
            )
            greedy_by_k = {
                int(row["k"]): [int(v) for v in row["anchors"]]
                for row in greedy_rows
            }
        for k in range(1, max_k + 1):
            if k in greedy_by_k:
                specs.append((greedy_name, k, valid_anchors(greedy_by_k[k], encoded.completion_length)))

    layout_policies = [policy for policy in policies if policy != "greedy"]
    for k in range(1, max_k + 1):
        layouts = layout_anchors(eligible, k)
        for policy in layout_policies:
            specs.append((policy, k, valid_anchors(layouts[policy], encoded.completion_length)))

    return specs


def valid_anchors(anchors: list[int], completion_length: int) -> list[int]:
    return sorted({int(pos) for pos in anchors if 0 <= int(pos) < completion_length})


def evaluate_decode(
    encoded: EncodedExample,
    result: DecodeResult,
    anchors: list[int],
    dataset: str,
) -> dict:
    anchor_set = set(anchors)
    matches = [
        int(pred == gold)
        for pred, gold in zip(result.completion_token_ids, encoded.completion_token_ids)
    ]
    non_anchor_matches = [
        matches[pos]
        for pos in range(len(matches))
        if pos not in anchor_set
    ]
    decoded_answer = extract_final_answer(result.completion_text, dataset)
    gold_answer = extract_final_answer(encoded.completion_text, dataset)
    answer_correct = answers_equal(decoded_answer, gold_answer)
    return {
        "token_accuracy": sum(matches) / len(matches) if matches else 0.0,
        "non_anchor_token_accuracy": (
            sum(non_anchor_matches) / len(non_anchor_matches)
            if non_anchor_matches
            else None
        ),
        "exact_token_match": bool(matches and all(matches)),
        "decoded_answer": decoded_answer,
        "gold_answer": gold_answer,
        "answer_correct": answer_correct,
        "decoded_text": result.completion_text,
    }


def make_decode_row(
    encoded: EncodedExample,
    result: DecodeResult,
    *,
    policy: str,
    k: int,
    anchors: list[int],
    eval_row: dict,
    baseline_eval: dict,
    dataset: str,
) -> dict:
    return {
        "example_id": encoded.example_id,
        "dataset": dataset,
        "policy": policy,
        "k": int(k),
        "anchors": list(anchors),
        "anchor_token_ids": [encoded.completion_token_ids[pos] for pos in anchors],
        "anchor_token_texts": [encoded.completion_token_texts[pos] for pos in anchors],
        "gold_anchor_fraction": len(anchors) / encoded.completion_length
        if encoded.completion_length
        else 0.0,
        "target_completion_token_count": encoded.completion_length,
        "full_completion_token_count": encoded.metadata.get(
            "full_completion_token_count",
            encoded.completion_length,
        ),
        "completion_truncated": bool(encoded.metadata.get("completion_truncated", False)),
        "full_has_final_answer_marker": has_final_answer_marker(encoded.completion_text, dataset),
        "target_has_final_answer_marker": has_final_answer_marker(
            "".join(encoded.completion_token_texts),
            dataset,
        ),
        "decode_steps": len(result.steps),
        "token_accuracy": eval_row["token_accuracy"],
        "non_anchor_token_accuracy": eval_row["non_anchor_token_accuracy"],
        "exact_token_match": eval_row["exact_token_match"],
        "decoded_answer": eval_row["decoded_answer"],
        "gold_answer": eval_row["gold_answer"],
        "answer_correct": eval_row["answer_correct"],
        "answer_changed_from_baseline": not answers_equal(
            eval_row["decoded_answer"],
            baseline_eval["decoded_answer"],
        ),
        "decoded_text_changed_from_baseline": normalize_text(result.completion_text)
        != normalize_text(baseline_eval.get("decoded_text", "")),
        "decoded_text": result.completion_text,
        "gold_completion": encoded.completion_text,
    }


def make_token_rows(encoded: EncodedExample) -> list[dict]:
    anchorable = set(encoded.anchor_positions or [])
    return [
        {
            "example_id": encoded.example_id,
            "position": pos,
            "token_id": int(token_id),
            "token_text": token_text,
            "anchorable": pos in anchorable,
        }
        for pos, (token_id, token_text) in enumerate(
            zip(encoded.completion_token_ids, encoded.completion_token_texts)
        )
    ]


def make_completion_window_row(encoded: EncodedExample, dataset: str) -> dict:
    target_text = "".join(encoded.completion_token_texts)
    full_count = int(
        encoded.metadata.get("full_completion_token_count", encoded.completion_length)
    )
    target_count = encoded.completion_length
    return {
        "example_id": encoded.example_id,
        "dataset": dataset,
        "full_completion_token_count": full_count,
        "target_completion_token_count": target_count,
        "completion_truncated": target_count < full_count,
        "full_has_final_answer_marker": has_final_answer_marker(encoded.completion_text, dataset),
        "target_has_final_answer_marker": has_final_answer_marker(target_text, dataset),
        "target_tail": preview_text(target_text[-240:], limit=240),
    }


def make_answer_timeline_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["example_id"]), str(row["policy"]))].append(row)

    timeline: list[dict] = []
    for (example_id, policy), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(row["k"]))
        baseline = next((row for row in group if int(row["k"]) == 0), group[0])
        baseline_correct_score = bool_score(baseline.get("answer_correct"))
        previous_anchors: set[int] = set()
        previous_correct_score = baseline_correct_score
        for row in group:
            anchors = [int(anchor) for anchor in row.get("anchors", [])]
            token_text_by_anchor = {
                int(anchor): str(token_text)
                for anchor, token_text in zip(anchors, row.get("anchor_token_texts", []))
            }
            new_positions = [anchor for anchor in anchors if anchor not in previous_anchors]
            new_token_texts = [token_text_by_anchor.get(anchor, "") for anchor in new_positions]
            answer_correct_score = bool_score(row.get("answer_correct"))
            delta_from_baseline = numeric_delta(answer_correct_score, baseline_correct_score)
            delta_from_previous = numeric_delta(answer_correct_score, previous_correct_score)
            if int(row["k"]) == 0:
                action = "standard decode (no anchors)"
            elif previous_anchors and not previous_anchors.issubset(set(anchors)):
                action = "set k anchors: " + ", ".join(
                    anchor_display(anchor, token_text_by_anchor.get(anchor, ""))
                    for anchor in anchors
                )
            else:
                action = "place anchor(s): " + ", ".join(
                    anchor_display(anchor, token_text_by_anchor.get(anchor, ""))
                    for anchor in new_positions
                )
            timeline.append(
                {
                    "example_id": example_id,
                    "dataset": row.get("dataset", ""),
                    "policy": policy,
                    "k": int(row["k"]),
                    "action": action,
                    "new_anchor_positions": new_positions,
                    "new_anchor_token_texts": new_token_texts,
                    "anchors": anchors,
                    "anchor_path": " -> ".join(
                        anchor_display(anchor, token_text_by_anchor.get(anchor, ""))
                        for anchor in anchors
                    ),
                    "decoded_answer": row.get("decoded_answer", ""),
                    "gold_answer": row.get("gold_answer", ""),
                    "answer_correct": row.get("answer_correct"),
                    "answer_correct_delta_from_baseline": delta_from_baseline,
                    "answer_correct_delta_from_previous": delta_from_previous,
                    "answer_changed_from_baseline": row.get("answer_changed_from_baseline"),
                    "token_accuracy": row.get("token_accuracy"),
                    "non_anchor_token_accuracy": row.get("non_anchor_token_accuracy"),
                    "decoded_text_preview": preview_text(str(row.get("decoded_text", ""))),
                    "decoded_text": row.get("decoded_text", ""),
                }
            )
            previous_anchors = set(anchors)
            previous_correct_score = answer_correct_score
    return timeline


def make_anchor_token_effect_rows(
    timeline_rows: list[dict],
    token_rows: list[dict],
    *,
    bins: int = 10,
) -> list[dict]:
    tokens_by_example: dict[str, dict[int, dict]] = defaultdict(dict)
    token_counts: dict[str, int] = defaultdict(int)
    for row in token_rows:
        example_id = str(row["example_id"])
        position = int(row["position"])
        tokens_by_example[example_id][position] = row
        token_counts[example_id] = max(token_counts[example_id], position + 1)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in timeline_rows:
        grouped[(str(row["example_id"]), str(row["policy"]))].append(row)

    out: list[dict] = []
    for (example_id, policy), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(row["k"]))
        if not group:
            continue
        baseline = next((row for row in group if int(row["k"]) == 0), group[0])
        baseline_correct = bool_score(baseline.get("answer_correct"))
        previous = baseline
        previous_anchors: set[int] = set()
        token_count = token_counts.get(example_id, 0)
        for row in group:
            k = int(row["k"])
            anchors = [int(anchor) for anchor in row.get("anchors", []) or []]
            if k == 0:
                previous = row
                previous_anchors = set(anchors)
                continue

            current_anchors = set(anchors)
            is_cumulative = previous_anchors.issubset(current_anchors)
            effect_positions = [
                anchor for anchor in anchors if anchor not in previous_anchors
            ] if is_cumulative else anchors
            if not effect_positions:
                previous = row
                previous_anchors = current_anchors
                continue

            current_correct = bool_score(row.get("answer_correct"))
            previous_correct = bool_score(previous.get("answer_correct"))
            current_token_accuracy = float_or_none(row.get("token_accuracy"))
            previous_token_accuracy = float_or_none(previous.get("token_accuracy"))
            token_accuracy_delta = numeric_delta(current_token_accuracy, previous_token_accuracy)
            for position in effect_positions:
                fraction = normalized_position(position, token_count)
                bin_index = min(bins - 1, max(0, int(fraction * bins))) if bins > 0 else 0
                token = tokens_by_example.get(example_id, {}).get(position, {})
                out.append(
                    {
                        "example_id": example_id,
                        "policy": policy,
                        "k": k,
                        "anchor_position": position,
                        "anchor_position_fraction": fraction,
                        "position_bin": bin_index,
                        "position_bin_label": f"t{bin_index + 1}",
                        "position_bin_center": (bin_index + 0.5) / bins if bins > 0 else 0.0,
                        "anchor_token_text": token.get("token_text", ""),
                        "anchor_update_type": "place" if is_cumulative else "set",
                        "decoded_answer": row.get("decoded_answer", ""),
                        "gold_answer": row.get("gold_answer", ""),
                        "answer_correct": row.get("answer_correct"),
                        "baseline_answer_correct": baseline.get("answer_correct"),
                        "previous_answer_correct": previous.get("answer_correct"),
                        "correct_delta_from_baseline": numeric_delta(
                            current_correct,
                            baseline_correct,
                        ),
                        "correct_delta_from_previous": numeric_delta(
                            current_correct,
                            previous_correct,
                        ),
                        "answer_changed_from_baseline": row.get("answer_changed_from_baseline"),
                        "token_accuracy": current_token_accuracy,
                        "token_accuracy_delta_from_previous": token_accuracy_delta,
                    }
                )
            previous = row
            previous_anchors = current_anchors
    return out


def aggregate_anchor_token_effect_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["policy"]), int(row["position_bin"]))].append(row)

    out: list[dict] = []
    for (policy, position_bin), group in sorted(grouped.items()):
        out.append(
            {
                "policy": policy,
                "position_bin": position_bin,
                "position_bin_label": f"t{position_bin + 1}",
                "position_bin_center": float(group[0]["position_bin_center"]),
                "n": len(group),
                "n_examples": len({str(row["example_id"]) for row in group}),
                "mean_correct_delta_from_baseline": mean_present(
                    row.get("correct_delta_from_baseline") for row in group
                ),
                "mean_correct_delta_from_previous": mean_present(
                    row.get("correct_delta_from_previous") for row in group
                ),
                "mean_answer_correct": mean_present(
                    bool_score(row.get("answer_correct")) for row in group
                ),
                "mean_answer_changed_from_baseline": mean_present(
                    bool_score(row.get("answer_changed_from_baseline")) for row in group
                ),
                "mean_token_accuracy": mean_present(row.get("token_accuracy") for row in group),
                "anchor_positions": sorted({int(row["anchor_position"]) for row in group}),
            }
        )
    return out


def make_greedy_standard_accuracy_change_rows(
    aggregate_rows: list[dict],
    *,
    policy: str = "greedy_ig",
) -> list[dict]:
    focus_rows = [
        row for row in aggregate_rows
        if str(row.get("policy")) == policy and row.get("mean_answer_correct") is not None
    ]
    if not focus_rows:
        return []
    focus_rows = sorted(focus_rows, key=lambda row: int(row["k"]))
    baseline = next((row for row in focus_rows if int(row["k"]) == 0), focus_rows[0])
    baseline_acc = float_or_none(baseline.get("mean_answer_correct"))
    if baseline_acc is None:
        return []

    out: list[dict] = []
    for row in focus_rows:
        mean_acc = float_or_none(row.get("mean_answer_correct"))
        if mean_acc is None:
            continue
        delta = mean_acc - baseline_acc
        out.append(
            {
                "policy": policy,
                "k": int(row["k"]),
                "x_label": "standard" if int(row["k"]) == 0 else str(row["k"]),
                "standard_mean_answer_correct": baseline_acc,
                "mean_answer_correct": mean_acc,
                "mean_answer_accuracy_change": delta,
                "mean_answer_accuracy_change_pp": 100.0 * delta,
                "n": row.get("n", ""),
            }
        )
    return out


def make_greedy_standard_accuracy_change_example_rows(
    timeline_rows: list[dict],
    *,
    policy: str = "greedy_ig",
) -> list[dict]:
    return make_policy_standard_accuracy_change_example_rows(timeline_rows, policies=[policy])


def make_policy_standard_accuracy_change_example_rows(
    timeline_rows: list[dict],
    *,
    policies: list[str],
) -> list[dict]:
    policy_set = set(policies)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in timeline_rows:
        policy = str(row.get("policy"))
        if policy in policy_set:
            grouped[(str(row["example_id"]), policy)].append(row)

    out: list[dict] = []
    for (example_id, policy), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(row["k"]))
        if not group:
            continue
        baseline = next((row for row in group if int(row["k"]) == 0), group[0])
        baseline_correct = bool_score(baseline.get("answer_correct"))
        if baseline_correct is None:
            continue
        for row in group:
            current_correct = bool_score(row.get("answer_correct"))
            if current_correct is None:
                continue
            delta = current_correct - baseline_correct
            k = int(row["k"])
            out.append(
                {
                    "example_id": example_id,
                    "policy": policy,
                    "k": k,
                    "x_label": "standard" if k == 0 else str(k),
                    "standard_answer_correct": baseline_correct,
                    "answer_correct": current_correct,
                    "answer_accuracy_change": delta,
                    "answer_accuracy_change_pp": 100.0 * delta,
                    "decoded_answer": row.get("decoded_answer", ""),
                    "gold_answer": row.get("gold_answer", ""),
                    "anchor_path": row.get("anchor_path", ""),
                    "token_accuracy": float_or_none(row.get("token_accuracy")),
                }
            )
    return out


def aggregate_greedy_standard_accuracy_change_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["policy"]), int(row["k"]))].append(row)

    out: list[dict] = []
    for (policy, k), group in sorted(grouped.items()):
        deltas = [float(row["answer_accuracy_change"]) for row in group]
        delta_pp = [100.0 * value for value in deltas]
        answer_correct = [float(row["answer_correct"]) for row in group]
        standard_correct = [float(row["standard_answer_correct"]) for row in group]
        n = len(group)
        std_delta = sample_std(deltas)
        sem_delta = (std_delta / math.sqrt(n)) if n > 0 else None
        ci95_delta = (1.96 * sem_delta) if sem_delta is not None else None
        out.append(
            {
                "policy": policy,
                "k": k,
                "x_label": "standard" if k == 0 else str(k),
                "standard_mean_answer_correct": mean_present(standard_correct),
                "mean_answer_correct": mean_present(answer_correct),
                "mean_answer_accuracy_change": mean_present(deltas),
                "mean_answer_accuracy_change_pp": mean_present(delta_pp),
                "std_answer_accuracy_change": std_delta,
                "std_answer_accuracy_change_pp": 100.0 * std_delta if std_delta is not None else None,
                "sem_answer_accuracy_change": sem_delta,
                "sem_answer_accuracy_change_pp": 100.0 * sem_delta if sem_delta is not None else None,
                "ci95_answer_accuracy_change": ci95_delta,
                "ci95_answer_accuracy_change_pp": 100.0 * ci95_delta if ci95_delta is not None else None,
                "n": n,
                "n_examples": len({str(row["example_id"]) for row in group}),
            }
        )
    return out


def write_answer_story_markdown(
    path: Path,
    timeline_rows: list[dict],
    token_rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens_by_example: dict[str, list[dict]] = defaultdict(list)
    for row in token_rows:
        tokens_by_example[str(row["example_id"])].append(row)
    by_example_policy: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in timeline_rows:
        by_example_policy[(str(row["example_id"]), str(row["policy"]))].append(row)

    lines = ["# Anchor Decode Answer Story", ""]
    for (example_id, policy), group in sorted(by_example_policy.items()):
        group = sorted(group, key=lambda row: int(row["k"]))
        lines.extend([f"## Example {example_id} - {policy}", ""])
        tokens = sorted(tokens_by_example.get(example_id, []), key=lambda row: int(row["position"]))
        if tokens:
            lines.extend(["Gold completion tokens:", ""])
            lines.append("```text")
            lines.append(format_token_strip(tokens))
            lines.append("```")
            lines.append("")
        lines.append("| k | action | anchor path | decoded answer | correct | delta correct base | changed | token acc | decoded preview |")
        lines.append("|---:|---|---|---|---:|---:|---:|---:|---|")
        for row in group:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["k"]),
                        md_cell(str(row["action"])),
                        md_cell(str(row["anchor_path"])),
                        md_cell(str(row["decoded_answer"])),
                        str(row["answer_correct"]),
                        format_signed(row.get("answer_correct_delta_from_baseline")),
                        str(row["answer_changed_from_baseline"]),
                        f"{float(row['token_accuracy']):.3f}"
                        if isinstance(row.get("token_accuracy"), (int, float))
                        else "",
                        md_cell(str(row["decoded_text_preview"])),
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def anchor_display(position: int, token_text: str) -> str:
    return f"{position}:{token_repr(token_text)}"


def token_repr(token_text: str) -> str:
    return repr(token_text.replace("\n", "\\n").replace("\t", "\\t"))


def preview_text(text: str, limit: int = 180) -> str:
    text = normalize_text(text)
    if len(text) > limit:
        return text[: limit - 1] + "..."
    return text


def format_token_strip(tokens: list[dict], tokens_per_line: int = 8) -> str:
    chunks: list[str] = []
    current: list[str] = []
    for row in tokens:
        current.append(f"{int(row['position']):02d}:{token_repr(str(row['token_text']))}")
        if len(current) >= tokens_per_line:
            chunks.append("  ".join(current))
            current = []
    if current:
        chunks.append("  ".join(current))
    return "\n".join(chunks)


def md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def format_signed(value: Any) -> str:
    value = float_or_none(value)
    if value is None:
        return ""
    return f"{value:+.0f}"


def bool_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return 1.0
    if text in {"false", "0", "no", "n"}:
        return 0.0
    return None


def float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def numeric_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def mean_present(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def sample_std(values: list[float]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if len(present) < 2:
        return 0.0 if present else None
    mean = sum(present) / len(present)
    variance = sum((value - mean) ** 2 for value in present) / (len(present) - 1)
    return math.sqrt(variance)


def normalized_position(position: int, token_count: int) -> float:
    if token_count <= 1:
        return 0.0
    return max(0.0, min(1.0, float(position) / float(token_count - 1)))


def add_trajectory_rows(
    rows: list[dict],
    encoded: EncodedExample,
    result: DecodeResult,
    *,
    policy: str,
    k: int,
    eval_row: dict,
) -> None:
    anchors = list(result.anchor_positions)
    for step in result.steps:
        for pos, token_id, token_text, confidence in zip(
            step.filled_positions,
            step.filled_token_ids,
            step.filled_token_texts,
            step.confidences,
        ):
            rows.append(
                {
                    "example_id": encoded.example_id,
                    "policy": policy,
                    "k": int(k),
                    "anchors": anchors,
                    "step": step.step,
                    "position": int(pos),
                    "token_id": int(token_id),
                    "token_text": token_text,
                    "gold_token_id": encoded.completion_token_ids[pos],
                    "gold_token_text": encoded.completion_token_texts[pos],
                    "matches_gold": int(token_id == encoded.completion_token_ids[pos]),
                    "confidence": float(confidence),
                    "remaining_masked": step.remaining_masked,
                    "decoded_answer": eval_row["decoded_answer"],
                    "gold_answer": eval_row["gold_answer"],
                    "answer_correct": eval_row["answer_correct"],
                }
            )


def rescore_decode_rows(rows: list[dict]) -> list[dict]:
    rescored = [dict(row) for row in rows]
    for row in rescored:
        dataset = str(row.get("dataset", ""))
        decoded_text = str(row.get("decoded_text", ""))
        gold_completion = str(row.get("gold_completion", ""))
        row["decoded_answer"] = extract_final_answer(decoded_text, dataset)
        row["gold_answer"] = extract_final_answer(gold_completion, dataset)
        row["answer_correct"] = answers_equal(row["decoded_answer"], row["gold_answer"])

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rescored:
        grouped[(str(row.get("example_id", "")), str(row.get("policy", "")))].append(row)
    for group in grouped.values():
        baseline = next((row for row in group if int(row.get("k", -1)) == 0), None)
        if baseline is None:
            continue
        baseline_answer = baseline.get("decoded_answer", "")
        for row in group:
            row["answer_changed_from_baseline"] = not answers_equal(
                row.get("decoded_answer", ""),
                baseline_answer,
            )
    return rescored


def rescore_trajectory_rows(rows: list[dict], decode_rows: list[dict]) -> list[dict]:
    by_decode = {
        (str(row.get("example_id", "")), str(row.get("policy", "")), int(row.get("k", 0))): row
        for row in decode_rows
    }
    rescored = []
    for row in rows:
        new_row = dict(row)
        decode_row = by_decode.get(
            (
                str(new_row.get("example_id", "")),
                str(new_row.get("policy", "")),
                int(new_row.get("k", 0)),
            )
        )
        if decode_row is not None:
            new_row["decoded_answer"] = decode_row.get("decoded_answer", "")
            new_row["gold_answer"] = decode_row.get("gold_answer", "")
            new_row["answer_correct"] = decode_row.get("answer_correct", "")
        rescored.append(new_row)
    return rescored


def extract_final_answer(text: str, dataset: str) -> str:
    dataset = dataset.lower()
    if dataset == "gsm8k":
        match = re.search(r"####\s*([^\n]+)", text)
        if match:
            answer_line = match.group(1)
            first_number = first_numberish(answer_line)
            if first_number is not None:
                return first_number
            return normalize_numberish(answer_line)
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        if numbers:
            return normalize_numberish(numbers[-1])
    return normalize_text(text)


def first_numberish(value: str) -> str | None:
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", value)
    if not match:
        return None
    return normalize_numberish(match.group(0))


def has_final_answer_marker(text: str, dataset: str) -> bool:
    if dataset.lower() == "gsm8k":
        return "####" in text
    return False


def normalize_numberish(value: str) -> str:
    value = value.strip().replace(",", "")
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" .:$")
    return value.lower()


def normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def answers_equal(left: Any, right: Any) -> bool:
    return normalize_text(str(left)) == normalize_text(str(right))


def aggregate_decode_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["policy"]), int(row["k"]))].append(row)

    out: list[dict] = []
    for (policy, k), group in sorted(groups.items()):
        aggregate = {"policy": policy, "k": k, "n": len(group)}
        for key in [
            "answer_correct",
            "answer_changed_from_baseline",
            "decoded_text_changed_from_baseline",
            "token_accuracy",
            "non_anchor_token_accuracy",
            "exact_token_match",
            "gold_anchor_fraction",
            "decode_steps",
        ]:
            values = [
                float(row[key])
                for row in group
                if isinstance(row.get(key), (bool, int, float))
            ]
            if values:
                aggregate[f"mean_{key}"] = sum(values) / len(values)
        out.append(aggregate)
    return out


def write_run_config(path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


if __name__ == "__main__":
    main()
