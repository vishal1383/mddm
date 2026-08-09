#!/usr/bin/env python3
"""Screen catalyst-teacher settings on empty canvases in one model load."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

import torch

from Token2Token.main.precompute_threshold_unlock_targets import (
    candidate_key,
    completion_logits_batch,
    confidence_summary,
    correct_threshold_positions,
    gold_log_probabilities,
    incorrect_threshold_positions,
    is_allowed_anchor_token,
    plausible_candidates,
)
from Token2Token.main.train import MODEL_ID, encode_record, load_base_model, record_stream


def main() -> None:
    args = parse_args()
    thresholds = parse_float_list(args.thresholds)
    ratios = parse_float_list(args.candidate_prob_ratios)
    penalties = parse_float_list(args.wrong_unlock_penalties, allow_zero=True)
    tokenizer, model, mask_token_id, device = load_base_model(
        args.model_id, args.device
    )
    model.eval()
    records = record_stream(args)
    selections = []

    with torch.no_grad():
        while len({item["example_id"] for item in selections}) < args.examples:
            source_record = next(records)
            encoded = encode_record(source_record, tokenizer, args)
            if encoded is None:
                continue
            prompt_ids, gold_ids, example_id = encoded
            canvas = [int(mask_token_id)] * len(gold_ids)
            candidates = [
                position
                for position, token_id in enumerate(gold_ids)
                if is_allowed_anchor_token(token_id, tokenizer)
            ]
            if not candidates:
                continue
            baseline_logits = completion_logits_batch(
                model, prompt_ids, [canvas], device
            )[0]
            gold_log_probabilities_by_position = gold_log_probabilities(
                baseline_logits, gold_ids, candidates, mask_token_id
            )
            baseline_prediction, baseline_confidence = confidence_summary(
                baseline_logits, mask_token_id
            )
            evaluated_positions = plausible_candidates(
                candidates,
                gold_log_probabilities_by_position,
                min(ratios),
            )
            candidate_outputs = {}
            for start in range(0, len(evaluated_positions), args.candidate_batch_size):
                positions = evaluated_positions[
                    start : start + args.candidate_batch_size
                ]
                candidate_canvases = []
                for position in positions:
                    candidate_canvas = list(canvas)
                    candidate_canvas[position] = int(gold_ids[position])
                    candidate_canvases.append(candidate_canvas)
                logits = completion_logits_batch(
                    model, prompt_ids, candidate_canvases, device
                )
                for row_index, position in enumerate(positions):
                    candidate_outputs[position] = confidence_summary(
                        logits[row_index], mask_token_id
                    )

            for threshold in thresholds:
                metrics = {
                    position: candidate_metrics(
                        canvas,
                        gold_ids,
                        baseline_prediction,
                        baseline_confidence,
                        candidate_outputs[position],
                        position,
                        mask_token_id,
                        threshold,
                        gold_log_probabilities_by_position[position],
                    )
                    for position in evaluated_positions
                }
                for ratio in ratios:
                    eligible = plausible_candidates(
                        candidates,
                        gold_log_probabilities_by_position,
                        ratio,
                    )
                    for penalty in penalties:
                        ranked = []
                        for position in eligible:
                            candidate = dict(metrics[position])
                            candidate["selection_score"] = (
                                candidate["correct_gain"]
                                - penalty * candidate["new_wrong"]
                            )
                            ranked.append(candidate)
                        best = max(ranked, key=candidate_key)
                        selections.append(
                            {
                                "example_id": str(example_id),
                                "threshold": threshold,
                                "candidate_prob_ratio": ratio,
                                "wrong_unlock_penalty": penalty,
                                "eligible_candidates": len(eligible),
                                **best,
                            }
                        )
            completed = len({item["example_id"] for item in selections})
            print(
                f"example={completed}/{args.examples} id={example_id} "
                f"candidates={len(candidates)} evaluated={len(evaluated_positions)}"
            )

    summaries = summarize_grid(selections)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "config": vars(args),
                "summaries": summaries,
                "selections": selections,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))


def candidate_metrics(
    canvas,
    gold_ids,
    baseline_prediction,
    baseline_confidence,
    candidate_output,
    position,
    mask_token_id,
    threshold,
    gold_log_probability,
):
    prediction, confidence = candidate_output
    common = {
        "candidate_position": position,
        "mask_token_id": mask_token_id,
        "threshold": threshold,
    }
    correct_before = correct_threshold_positions(
        canvas,
        gold_ids,
        baseline_prediction,
        baseline_confidence,
        **common,
    )
    correct_after = correct_threshold_positions(
        canvas, gold_ids, prediction, confidence, **common
    )
    wrong_before = incorrect_threshold_positions(
        canvas,
        gold_ids,
        baseline_prediction,
        baseline_confidence,
        **common,
    )
    wrong_after = incorrect_threshold_positions(
        canvas, gold_ids, prediction, confidence, **common
    )
    correct_before_positions = {
        int(item["gold_position"]) for item in correct_before
    }
    wrong_before_positions = {int(item["gold_position"]) for item in wrong_before}
    new_correct = sum(
        int(item["gold_position"]) not in correct_before_positions
        for item in correct_after
    )
    new_wrong = sum(
        int(item["gold_position"]) not in wrong_before_positions
        for item in wrong_after
    )
    return {
        "position": position,
        "gold_log_probability": float(gold_log_probability),
        "gold_probability": math.exp(float(gold_log_probability)),
        "correct_before": len(correct_before),
        "correct_after": len(correct_after),
        "correct_gain": len(correct_after) - len(correct_before),
        "new_correct": new_correct,
        "wrong_before": len(wrong_before),
        "wrong_after": len(wrong_after),
        "new_wrong": new_wrong,
        # candidate_key expects the detailed containers for its tie breakers.
        "newly_wrong": [None] * new_wrong,
    }


def summarize_grid(selections):
    grouped = defaultdict(list)
    for item in selections:
        key = (
            item["threshold"],
            item["candidate_prob_ratio"],
            item["wrong_unlock_penalty"],
        )
        grouped[key].append(item)
    summaries = []
    for (threshold, ratio, penalty), rows in grouped.items():
        summary = {
            "threshold": threshold,
            "candidate_prob_ratio": ratio,
            "wrong_unlock_penalty": penalty,
            "examples": len(rows),
            "mean_correct_gain": mean(rows, "correct_gain"),
            "mean_new_correct": mean(rows, "new_correct"),
            "mean_new_wrong": mean(rows, "new_wrong"),
            "mean_safe_gain": mean(rows, "selection_score"),
            "positive_gain_fraction": statistics.fmean(
                row["correct_gain"] > 0 for row in rows
            ),
            "burst_two_fraction": statistics.fmean(
                row["new_correct"] >= 2 for row in rows
            ),
            "mean_catalyst_probability": mean(rows, "gold_probability"),
            "mean_eligible_candidates": mean(rows, "eligible_candidates"),
        }
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda row: (
            row["mean_safe_gain"],
            row["mean_new_correct"],
            -row["mean_new_wrong"],
            row["mean_catalyst_probability"],
        ),
        reverse=True,
    )


def mean(rows, key):
    return statistics.fmean(float(row[key]) for row in rows)


def parse_float_list(value, allow_zero=False):
    items = [float(item.strip()) for item in value.split(",") if item.strip()]
    lower = 0 if allow_zero else 0.0
    if not items or any(item < lower or (not allow_zero and item == 0) for item in items):
        raise ValueError("list values must be positive")
    return items


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset", choices=("gsm8k", "lm1b"), default="gsm8k")
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--thresholds", default="0.70,0.80,0.90,0.95")
    parser.add_argument("--candidate-prob-ratios", default="0.30,0.50,0.70,0.90")
    parser.add_argument("--wrong-unlock-penalties", default="0,1")
    parser.add_argument("--candidate-batch-size", type=int, default=64)
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    parser.add_argument("--lm1b-prompt-tokens", type=int, default=32)
    parser.add_argument("--lm1b-dataset", default="FrankCCCCC/lm1b")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.examples <= 0 or args.candidate_batch_size <= 0:
        parser.error("examples and candidate-batch-size must be positive")
    return args


if __name__ == "__main__":
    main()
