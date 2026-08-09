#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

import torch

from Token2Token.precompute_anchor_targets import count_rows, source_fields
from Token2Token.train import MODEL_ID, encode_record, load_base_model, record_stream

TARGET_SOURCE = "frozen_base_threshold_gain_text_anchors"


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.resume:
        raise FileExistsError(f"{output} already exists; pass --resume to continue")
    write_metadata(output, args)

    completed = count_rows(output) if args.resume else 0
    tokenizer, model, mask_token_id, device = load_base_model(
        args.model_id, args.device
    )
    model.eval()
    records = record_stream(args)
    skipped = 0
    written = completed

    while written < args.examples:
        source_record = next(records)
        encoded = encode_record(source_record, tokenizer, args)
        if encoded is None:
            continue
        if skipped < completed:
            skipped += 1
            continue
        prompt_ids, gold_ids, example_id = encoded
        rounds, residual = threshold_unlock_trajectory(
            model,
            tokenizer,
            prompt_ids,
            gold_ids,
            mask_token_id,
            confidence_threshold=args.confidence_threshold,
            candidate_prob_ratio=args.candidate_prob_ratio,
            candidate_batch_size=args.candidate_batch_size,
            max_rounds=args.max_rounds,
            wrong_unlock_penalty=args.wrong_unlock_penalty,
            device=device,
            candidate_must_be_correct=args.candidate_must_be_correct,
            candidate_must_be_below_threshold=(
                args.candidate_must_be_below_threshold
            ),
            selection_mode=args.selection_mode,
            numeric_threshold=args.numeric_threshold,
        )
        row = {
            "dataset": args.dataset,
            "example_id": example_id,
            "target_source": TARGET_SOURCE,
            "selection_metric": (
                "decoder_highest_confidence_text_below"
                if args.selection_mode == "decoder"
                else "correct_95_after_minus_before"
            ),
            "selection_mode": args.selection_mode,
            "confidence_threshold": args.confidence_threshold,
            "numeric_threshold": args.numeric_threshold,
            "candidate_prob_ratio": args.candidate_prob_ratio,
            "wrong_unlock_penalty": args.wrong_unlock_penalty,
            "candidate_must_be_correct": args.candidate_must_be_correct,
            "candidate_must_be_below_threshold": (
                args.candidate_must_be_below_threshold
            ),
            "anchor_filter": (
                "predicted_alphabetic_tokens_only"
                if args.selection_mode == "decoder"
                else "alphabetic_tokens_only"
            ),
            "source": source_fields(args.dataset, source_record[1]),
            "prompt_ids": prompt_ids,
            "gold_ids": gold_ids,
            "rounds": rounds,
            "residual": residual,
        }
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        written += 1
        unlocked = sum(len(item["unlocked"]) for item in rounds)
        placed = sum(item["tokens_placed"] for item in rounds)
        mean_placed = placed / len(rounds) if rounds else 0.0
        print(
            f"example={written}/{args.examples} id={example_id} "
            f"tokens={len(gold_ids)} rounds={len(rounds)} "
            f"unlocked={unlocked} residual={len(residual)} "
            f"mean_placed={mean_placed:.3f}"
        )

    summary = summarize_target_file(output)
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def threshold_unlock_trajectory(
    model,
    tokenizer,
    prompt_ids,
    gold_ids,
    mask_token_id,
    *,
    confidence_threshold,
    candidate_prob_ratio,
    candidate_batch_size,
    max_rounds=None,
    wrong_unlock_penalty=0.0,
    device,
    candidate_must_be_correct=False,
    candidate_must_be_below_threshold=False,
    selection_mode="oracle",
    numeric_threshold=None,
):
    if not 0 < confidence_threshold < 1:
        raise ValueError("confidence threshold must be in (0, 1)")
    if not 0 < candidate_prob_ratio <= 1:
        raise ValueError("candidate probability ratio must be in (0, 1]")
    if candidate_batch_size <= 0:
        raise ValueError("candidate batch size must be positive")
    if max_rounds is not None and max_rounds <= 0:
        raise ValueError("max rounds must be positive")
    if wrong_unlock_penalty < 0:
        raise ValueError("wrong unlock penalty must be nonnegative")
    if selection_mode not in ("oracle", "decoder"):
        raise ValueError("selection mode must be 'oracle' or 'decoder'")
    if numeric_threshold is not None and not (
        confidence_threshold <= numeric_threshold < 1
    ):
        raise ValueError(
            "numeric threshold must be in [confidence threshold, 1)"
        )

    canvas = [int(mask_token_id)] * len(gold_ids)
    rounds = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            while mask_token_id in canvas and (
                max_rounds is None or len(rounds) < max_rounds
            ):
                remaining = [
                    position
                    for position, token_id in enumerate(canvas)
                    if token_id == mask_token_id
                ]
                baseline_logits = completion_logits_batch(
                    model, prompt_ids, [canvas], device
                )[0]
                baseline_prediction, baseline_confidence = confidence_summary(
                    baseline_logits, mask_token_id
                )
                decoder_cleanup = False
                if selection_mode == "decoder":
                    candidate_position, eligible_count, decoder_cleanup = (
                        decoder_catalyst_position(
                            remaining,
                            baseline_prediction,
                            baseline_confidence,
                            tokenizer,
                            confidence_threshold,
                        )
                    )
                    eligible = [candidate_position]
                    candidate_log_probabilities = gold_log_probabilities(
                        baseline_logits,
                        gold_ids,
                        eligible,
                        mask_token_id,
                    )
                else:
                    anchor_candidates = [
                        position
                        for position in remaining
                        if is_allowed_anchor_token(gold_ids[position], tokenizer)
                    ]
                    if not anchor_candidates:
                        break
                    candidate_log_probabilities = gold_log_probabilities(
                        baseline_logits,
                        gold_ids,
                        anchor_candidates,
                        mask_token_id,
                    )
                    eligible = plausible_candidates(
                        anchor_candidates,
                        candidate_log_probabilities,
                        candidate_prob_ratio,
                    )
                    eligible = inference_aligned_candidates(
                        eligible,
                        baseline_prediction,
                        baseline_confidence,
                        gold_ids,
                        confidence_threshold,
                        require_correct=candidate_must_be_correct,
                        require_below_threshold=(
                            candidate_must_be_below_threshold
                        ),
                    )
                    eligible_count = len(eligible)
                    if not eligible:
                        break
                del baseline_logits
                best = None
                for start in range(0, len(eligible), candidate_batch_size):
                    positions = eligible[start : start + candidate_batch_size]
                    candidate_canvases = []
                    for position in positions:
                        candidate_canvas = list(canvas)
                        candidate_canvas[position] = int(gold_ids[position])
                        candidate_canvases.append(candidate_canvas)
                    logits = completion_logits_batch(
                        model, prompt_ids, candidate_canvases, device
                    )
                    for row_index, position in enumerate(positions):
                        prediction, confidence = confidence_summary(
                            logits[row_index], mask_token_id
                        )
                        correct_before = correct_threshold_positions(
                            canvas,
                            gold_ids,
                            baseline_prediction,
                            baseline_confidence,
                            candidate_position=position,
                            mask_token_id=mask_token_id,
                            threshold=confidence_threshold,
                            tokenizer=tokenizer,
                            numeric_threshold=numeric_threshold,
                        )
                        correct_after = correct_threshold_positions(
                            canvas,
                            gold_ids,
                            prediction,
                            confidence,
                            candidate_position=position,
                            mask_token_id=mask_token_id,
                            threshold=confidence_threshold,
                            tokenizer=tokenizer,
                            numeric_threshold=numeric_threshold,
                        )
                        wrong_before = incorrect_threshold_positions(
                            canvas,
                            gold_ids,
                            baseline_prediction,
                            baseline_confidence,
                            candidate_position=position,
                            mask_token_id=mask_token_id,
                            threshold=confidence_threshold,
                            tokenizer=tokenizer,
                            numeric_threshold=numeric_threshold,
                        )
                        wrong_after = incorrect_threshold_positions(
                            canvas,
                            gold_ids,
                            prediction,
                            confidence,
                            candidate_position=position,
                            mask_token_id=mask_token_id,
                            threshold=confidence_threshold,
                            tokenizer=tokenizer,
                            numeric_threshold=numeric_threshold,
                        )
                        before_correct_positions = {
                            int(item["gold_position"]) for item in correct_before
                        }
                        before_wrong_positions = {
                            int(item["gold_position"]) for item in wrong_before
                        }
                        newly_unlocked = [
                            item
                            for item in correct_after
                            if int(item["gold_position"])
                            not in before_correct_positions
                        ]
                        newly_wrong = [
                            item
                            for item in wrong_after
                            if int(item["gold_position"])
                            not in before_wrong_positions
                        ]
                        correct_gain = len(correct_after) - len(correct_before)
                        candidate = {
                            "position": position,
                            "gold_log_probability": candidate_log_probabilities[
                                position
                            ],
                            "correct_before": len(correct_before),
                            "correct_after": correct_after,
                            "newly_unlocked": newly_unlocked,
                            "wrong_before": len(wrong_before),
                            "wrong_after": wrong_after,
                            "newly_wrong": newly_wrong,
                            "selection_score": (
                                correct_gain
                                - wrong_unlock_penalty * len(newly_wrong)
                            ),
                        }
                        if best is None or (
                            selection_mode == "oracle"
                            and candidate_key(candidate) > candidate_key(best)
                        ):
                            best = candidate

                if best is None:
                    raise RuntimeError("no candidate available for a non-empty canvas")
                catalyst_position = int(best["position"])
                catalyst_token_id = int(gold_ids[catalyst_position])
                unlocked = best["correct_after"]
                correct_before = int(best["correct_before"])
                correct_after = len(unlocked)
                rounds.append(
                    {
                        "round": len(rounds) + 1,
                        "catalyst": {
                            "gold_position": catalyst_position,
                            "token_id": catalyst_token_id,
                            "token": tokenizer.decode([catalyst_token_id]),
                            "gold_probability_before": math.exp(
                                float(best["gold_log_probability"])
                            ),
                            "predicted_token_id_before": int(
                                baseline_prediction[catalyst_position]
                            ),
                            "predicted_token_before": tokenizer.decode(
                                [int(baseline_prediction[catalyst_position])]
                            ),
                            "prediction_correct_before": (
                                int(baseline_prediction[catalyst_position])
                                == catalyst_token_id
                            ),
                            "confidence_before": float(
                                baseline_confidence[catalyst_position]
                            ),
                            "decoder_cleanup": decoder_cleanup,
                        },
                        "eligible_candidates": eligible_count,
                        "correct_95_before": correct_before,
                        "correct_95_after": correct_after,
                        "correct_95_gain": correct_after - correct_before,
                        "new_correct_unlocks": len(best["newly_unlocked"]),
                        "new_confident_mistakes": len(best["newly_wrong"]),
                        "wrong_95_before": int(best["wrong_before"]),
                        "wrong_95_after": len(best["wrong_after"]),
                        "selection_score": float(best["selection_score"]),
                        "tokens_placed": 1 + correct_after,
                        "unlocked": unlocked,
                        "newly_unlocked": best["newly_unlocked"],
                        "newly_wrong": best["newly_wrong"],
                    }
                )
                canvas[catalyst_position] = catalyst_token_id
                for item in unlocked:
                    position = int(item["gold_position"])
                    canvas[position] = int(gold_ids[position])
    finally:
        model.train(was_training)
    residual = [
        {
            "gold_position": position,
            "token_id": int(gold_ids[position]),
            "token": tokenizer.decode([int(gold_ids[position])]),
        }
        for position, token_id in enumerate(canvas)
        if token_id == mask_token_id
    ]
    return rounds, residual


def is_allowed_anchor_token(token_id, tokenizer):
    if int(token_id) in set(getattr(tokenizer, "all_special_ids", [])):
        return False
    text = tokenizer.decode([int(token_id)]).strip()
    return bool(text) and text.isalpha()


def decoder_catalyst_position(
    remaining,
    prediction,
    confidence,
    tokenizer,
    threshold,
):
    """Choose the position used by text-below/force-always inference."""
    eligible = [
        position
        for position in remaining
        if is_allowed_anchor_token(prediction[position], tokenizer)
        and float(confidence[position]) < threshold
    ]
    if eligible:
        position = max(eligible, key=lambda item: (confidence[item], -item))
        return position, len(eligible), False
    return min(remaining), 0, True


def token_confidence_threshold(
    token_id,
    tokenizer,
    threshold,
    numeric_threshold=None,
):
    if numeric_threshold is None:
        return threshold
    text = tokenizer.decode([int(token_id)])
    return numeric_threshold if any(character.isdigit() for character in text) else threshold


def completion_logits_batch(model, prompt_ids, canvases, device):
    input_ids = torch.tensor(
        [prompt_ids + canvas for canvas in canvases],
        device=device,
        dtype=torch.long,
    )
    outputs = model(input_ids=input_ids, use_cache=False)
    return outputs.logits[:, len(prompt_ids) : len(prompt_ids) + len(canvases[0])]


def gold_log_probabilities(logits, gold_ids, positions, mask_token_id):
    logits = logits.float()
    logits[:, mask_token_id] = -torch.inf
    selected = logits[positions]
    labels = torch.tensor(
        [gold_ids[position] for position in positions],
        device=logits.device,
        dtype=torch.long,
    )
    values = torch.log_softmax(selected, dim=-1).gather(1, labels.unsqueeze(1))
    return {
        position: float(value)
        for position, value in zip(positions, values.squeeze(1).cpu().tolist())
    }


def plausible_candidates(positions, log_probabilities, probability_ratio):
    best = max(log_probabilities[position] for position in positions)
    cutoff = best + math.log(probability_ratio)
    return [position for position in positions if log_probabilities[position] >= cutoff]


def inference_aligned_candidates(
    positions,
    prediction,
    confidence,
    gold_ids,
    threshold,
    *,
    require_correct=False,
    require_below_threshold=False,
):
    """Keep teacher catalysts that the configured inference rule can place."""
    return [
        position
        for position in positions
        if (
            not require_correct
            or int(prediction[position]) == int(gold_ids[position])
        )
        and (
            not require_below_threshold
            or float(confidence[position]) < threshold
        )
    ]


def confidence_summary(logits, mask_token_id):
    logits = logits.float()
    logits[:, mask_token_id] = -torch.inf
    maximum, prediction = logits.max(dim=-1)
    confidence = torch.exp(maximum - torch.logsumexp(logits, dim=-1))
    return prediction.cpu().tolist(), confidence.cpu().tolist()


def correct_threshold_positions(
    canvas,
    gold_ids,
    prediction,
    confidence,
    *,
    candidate_position,
    mask_token_id,
    threshold,
    tokenizer=None,
    numeric_threshold=None,
):
    unlocked = []
    for position, token_id in enumerate(canvas):
        if token_id != mask_token_id:
            continue
        predicted_token_id = int(prediction[position])
        position_threshold = token_confidence_threshold(
            predicted_token_id,
            tokenizer,
            threshold,
            numeric_threshold,
        )
        if position == candidate_position or confidence[position] < position_threshold:
            continue
        if predicted_token_id != int(gold_ids[position]):
            continue
        unlocked.append(
            {
                "gold_position": position,
                "token_id": int(gold_ids[position]),
                "predicted_token_id": predicted_token_id,
                "confidence": float(confidence[position]),
                "distance_from_catalyst": position - candidate_position,
                "normalized_position": position / max(1, len(canvas) - 1),
            }
        )
    unlocked.sort(key=lambda item: (-item["confidence"], item["gold_position"]))
    return unlocked


def incorrect_threshold_positions(
    canvas,
    gold_ids,
    prediction,
    confidence,
    *,
    candidate_position,
    mask_token_id,
    threshold,
    tokenizer=None,
    numeric_threshold=None,
):
    mistakes = []
    for position, token_id in enumerate(canvas):
        if token_id != mask_token_id:
            continue
        predicted_token_id = int(prediction[position])
        position_threshold = token_confidence_threshold(
            predicted_token_id,
            tokenizer,
            threshold,
            numeric_threshold,
        )
        if position == candidate_position or confidence[position] < position_threshold:
            continue
        if predicted_token_id == int(gold_ids[position]):
            continue
        mistakes.append(
            {
                "gold_position": position,
                "gold_token_id": int(gold_ids[position]),
                "predicted_token_id": predicted_token_id,
                "confidence": float(confidence[position]),
                "distance_from_catalyst": position - candidate_position,
            }
        )
    mistakes.sort(key=lambda item: (-item["confidence"], item["gold_position"]))
    return mistakes


def candidate_key(candidate):
    correct_after = candidate["correct_after"]
    after = (
        int(correct_after)
        if isinstance(correct_after, (int, float))
        else len(correct_after)
    )
    gain = after - int(candidate["correct_before"])
    return (
        float(candidate.get("selection_score", gain)),
        gain,
        after,
        -len(candidate.get("newly_wrong", [])),
        candidate["gold_log_probability"],
        -candidate["position"],
    )


def summarize_target_file(path: Path):
    examples = 0
    completion_tokens = 0
    rounds = 0
    unlocked = 0
    total_before = 0
    total_after = 0
    total_gain = 0
    causal_unlocks = 0
    new_confident_mistakes = 0
    placed_tokens = 0
    residual_tokens = 0
    examples_with_residual = 0
    zero_unlock_rounds = 0
    implied_forwards = 0
    histogram = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            examples += 1
            completion_tokens += len(record["gold_ids"])
            remaining = len(record["gold_ids"])
            for item in record["rounds"]:
                count = len(item["unlocked"])
                rounds += 1
                implied_forwards += 1
                remaining -= 1
                if remaining:
                    implied_forwards += 1
                remaining -= count
                unlocked += count
                total_before += int(item["correct_95_before"])
                total_after += int(item["correct_95_after"])
                total_gain += int(item["correct_95_gain"])
                causal_unlocks += len(
                    item.get(
                        "newly_unlocked",
                        [None] * max(0, int(item["correct_95_gain"])),
                    )
                )
                new_confident_mistakes += len(item.get("newly_wrong", []))
                placed_tokens += int(item["tokens_placed"])
                zero_unlock_rounds += int(count == 0)
                histogram[count] += 1
            residual = record.get("residual", [])
            residual_tokens += len(residual)
            examples_with_residual += int(bool(residual))
            remaining -= len(residual)
            if remaining != 0:
                raise ValueError(
                    f"incomplete trajectory for example {record.get('example_id')}"
                )
    return {
        "examples": examples,
        "completion_tokens": completion_tokens,
        "rounds": rounds,
        "unlocked_tokens": unlocked,
        "total_correct_95_before": total_before,
        "total_correct_95_after": total_after,
        "total_correct_95_gain": total_gain,
        "causal_new_correct_unlocks": causal_unlocks,
        "new_confident_mistakes": new_confident_mistakes,
        "mean_correct_95_gain": total_gain / rounds if rounds else 0.0,
        "placed_tokens": placed_tokens,
        "residual_tokens": residual_tokens,
        "examples_with_residual": examples_with_residual,
        "zero_unlock_rounds": zero_unlock_rounds,
        "zero_unlock_fraction": zero_unlock_rounds / rounds if rounds else 0.0,
        "mean_unlocked_per_round": unlocked / rounds if rounds else 0.0,
        "mean_tokens_placed_per_round": placed_tokens / rounds if rounds else 0.0,
        "implied_anchor_model_forwards": implied_forwards,
        "implied_cleanup_model_forwards": residual_tokens,
        "implied_model_forwards": implied_forwards + residual_tokens,
        "implied_tokens_per_forward": (
            completion_tokens / (implied_forwards + residual_tokens)
            if implied_forwards + residual_tokens
            else 0.0
        ),
        "unlock_count_histogram": dict(sorted(histogram.items())),
    }


def write_metadata(output: Path, args) -> None:
    metadata_path = output.with_suffix(".config.json")
    metadata = vars(args).copy()
    metadata["output"] = str(output)
    metadata["target_source"] = TARGET_SOURCE
    metadata["selection_scope"] = "full_completion_canvas"
    metadata["anchor_filter"] = (
        "predicted_alphabetic_tokens_only"
        if args.selection_mode == "decoder"
        else "alphabetic_tokens_only"
    )
    if metadata_path.exists() and args.resume:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        comparable = [
            "model_id",
            "dataset",
            "confidence_threshold",
            "numeric_threshold",
            "selection_mode",
            "candidate_prob_ratio",
            "max_completion_tokens",
            "max_rounds",
            "wrong_unlock_penalty",
            "candidate_must_be_correct",
            "candidate_must_be_below_threshold",
            "seed",
            "target_source",
            "selection_scope",
            "anchor_filter",
        ]
        if any(existing.get(key) != metadata.get(key) for key in comparable):
            raise ValueError("resume configuration does not match target metadata")
        existing["candidate_batch_size"] = args.candidate_batch_size
        existing["resume"] = True
        metadata_path.write_text(
            json.dumps(existing, indent=2) + "\n", encoding="utf-8"
        )
        return
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute anchors by global correct-95% after-minus-before gain"
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset", choices=["gsm8k", "lm1b"], default="gsm8k")
    parser.add_argument("--examples", type=int, default=7_473)
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--numeric-threshold", type=float)
    parser.add_argument(
        "--selection-mode",
        choices=("oracle", "decoder"),
        default="oracle",
    )
    parser.add_argument("--candidate-prob-ratio", type=float, default=0.7)
    parser.add_argument("--candidate-batch-size", type=int, default=8)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--wrong-unlock-penalty", type=float, default=0.0)
    parser.add_argument("--candidate-must-be-correct", action="store_true")
    parser.add_argument(
        "--candidate-must-be-below-threshold", action="store_true"
    )
    parser.add_argument("--lm1b-prompt-tokens", type=int, default=32)
    parser.add_argument("--lm1b-dataset", default="FrankCCCCC/lm1b")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/token2token/threshold_unlock/gsm8k_train_t095_gain_text_q07_max512.jsonl",
    )
    args = parser.parse_args()
    if args.examples <= 0 or args.candidate_batch_size <= 0:
        parser.error("examples and candidate-batch-size must be positive")
    if not 0 < args.confidence_threshold < 1:
        parser.error("confidence-threshold must be in (0, 1)")
    if args.numeric_threshold is not None and not (
        args.confidence_threshold <= args.numeric_threshold < 1
    ):
        parser.error(
            "numeric-threshold must be in [confidence-threshold, 1)"
        )
    if not 0 < args.candidate_prob_ratio <= 1:
        parser.error("candidate-prob-ratio must be in (0, 1]")
    if args.max_rounds is not None and args.max_rounds <= 0:
        parser.error("max-rounds must be positive")
    if args.wrong_unlock_penalty < 0:
        parser.error("wrong-unlock-penalty must be nonnegative")
    return args


if __name__ == "__main__":
    main()
