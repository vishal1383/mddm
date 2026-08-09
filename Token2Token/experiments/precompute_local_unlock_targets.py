#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch

from Token2Token.main.core import Target, select_targets
from Token2Token.main.precompute_anchor_targets import count_rows, source_fields
from Token2Token.main.train import (
    MODEL_ID,
    encode_record,
    load_base_model,
    record_stream,
)


TARGET_SOURCE = "frozen_base_local_top1_unlock"


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
        candidates = select_targets(
            gold_ids,
            tokenizer,
            count=len(gold_ids),
            max_right_fraction=args.target_right_fraction,
            policy="prefix",
            mask_token_id=mask_token_id,
        )
        targets, scores = greedy_local_unlock_targets(
            model,
            prompt_ids,
            gold_ids,
            candidates,
            mask_token_id,
            count=args.anchors,
            window_size=args.window_size,
            batch_size=args.candidate_batch_size,
            device=device,
        )
        if not targets:
            continue
        row = {
            "dataset": args.dataset,
            "example_id": example_id,
            "target_source": TARGET_SOURCE,
            "selection_metric": "local_top1_correct_gain",
            "source": source_fields(args.dataset, source_record[1]),
            "prompt_ids": prompt_ids,
            "gold_ids": gold_ids,
            "targets": [
                {
                    "rank": rank + 1,
                    "gold_position": target.gold_position,
                    "token_id": target.token_id,
                    "token": target.token_text,
                    "selection_score": float(scores[rank]["gain"]),
                    "correct_before": int(scores[rank]["before"]),
                    "correct_after": int(scores[rank]["after"]),
                    "top1_unlock_gain": int(scores[rank]["gain"]),
                    "window_start": int(scores[rank]["window_start"]),
                    "window_end": int(scores[rank]["window_end"]),
                    "window_size": args.window_size,
                }
                for rank, target in enumerate(targets)
            ],
        }
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        written += 1
        print(
            f"example={written}/{args.examples} id={example_id} "
            f"gains={[score['gain'] for score in scores]}"
        )


def greedy_local_unlock_targets(
    model,
    prompt_ids,
    gold_ids,
    candidates,
    mask_token_id,
    *,
    count,
    window_size,
    batch_size,
    device,
):
    if batch_size <= 0 or window_size <= 1:
        raise ValueError("candidate batch size must be positive and window > 1")
    count = min(int(count), len(candidates))
    canvas = [int(mask_token_id)] * len(gold_ids)
    selected = []
    scores = []
    selected_positions = set()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for rank in range(count):
                baseline = completion_argmax(
                    model, prompt_ids, [canvas], mask_token_id, device
                )[0]
                remaining = [
                    target
                    for target in candidates
                    if target.gold_position not in selected_positions
                ]
                best_target = None
                best_score = None
                for start in range(0, len(remaining), batch_size):
                    chunk = remaining[start : start + batch_size]
                    candidate_canvases = []
                    for target in chunk:
                        candidate_canvas = list(canvas)
                        candidate_canvas[target.gold_position] = target.token_id
                        candidate_canvases.append(candidate_canvas)
                    predictions = completion_argmax(
                        model,
                        prompt_ids,
                        candidate_canvases,
                        mask_token_id,
                        device,
                    )
                    for target, after_predictions in zip(chunk, predictions):
                        score = local_top1_gain(
                            canvas,
                            baseline,
                            after_predictions,
                            gold_ids,
                            target.gold_position,
                            mask_token_id,
                            window_size,
                        )
                        if best_score is None or (
                            score["gain"], score["after"]
                        ) > (best_score["gain"], best_score["after"]):
                            best_target = target
                            best_score = score
                if best_target is None or best_score is None:
                    break
                selected.append(
                    Target(
                        rank,
                        best_target.token_id,
                        best_target.token_text,
                        best_target.gold_position,
                    )
                )
                scores.append(best_score)
                selected_positions.add(best_target.gold_position)
                canvas[best_target.gold_position] = best_target.token_id
    finally:
        model.train(was_training)
    return selected, scores


def completion_argmax(model, prompt_ids, canvases, mask_token_id, device):
    input_ids = torch.tensor(
        [prompt_ids + canvas for canvas in canvases],
        device=device,
        dtype=torch.long,
    )
    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits[
        :, len(prompt_ids) : len(prompt_ids) + len(canvases[0])
    ].float()
    logits[:, :, mask_token_id] = -torch.inf
    return logits.argmax(dim=-1).cpu().tolist()


def local_top1_gain(
    canvas,
    before_predictions,
    after_predictions,
    gold_ids,
    candidate_position,
    mask_token_id,
    window_size,
):
    start, end = shifted_window(candidate_position, len(canvas), window_size)
    positions = [
        position
        for position in range(start, end)
        if position != candidate_position and canvas[position] == mask_token_id
    ]
    before = sum(
        before_predictions[position] == gold_ids[position] for position in positions
    )
    after = sum(
        after_predictions[position] == gold_ids[position] for position in positions
    )
    return {
        "before": before,
        "after": after,
        "gain": after - before,
        "window_start": start,
        "window_end": end,
    }


def shifted_window(position: int, length: int, window_size: int):
    width = min(int(window_size), length)
    start = position - width // 2
    start = max(0, min(start, length - width))
    return start, start + width


def write_metadata(output: Path, args) -> None:
    metadata_path = output.with_suffix(".config.json")
    metadata = vars(args).copy()
    metadata["output"] = str(output)
    metadata["target_source"] = TARGET_SOURCE
    if metadata_path.exists() and args.resume:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        comparable = [
            "model_id",
            "dataset",
            "anchors",
            "target_right_fraction",
            "window_size",
            "max_completion_tokens",
            "seed",
            "target_source",
        ]
        if any(existing.get(key) != metadata.get(key) for key in comparable):
            raise ValueError("resume configuration does not match target metadata")
        existing["candidate_batch_size"] = args.candidate_batch_size
        existing["resume"] = True
        metadata_path.write_text(
            json.dumps(existing, indent=2) + "\n", encoding="utf-8"
        )
        return
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute anchors by local top-1 gold-token unlock gain"
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset", choices=["gsm8k", "lm1b"], default="gsm8k")
    parser.add_argument("--examples", type=int, default=7_473)
    parser.add_argument("--anchors", type=int, default=2)
    parser.add_argument("--target-right-fraction", type=float, default=0.75)
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--candidate-batch-size", type=int, default=64)
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    parser.add_argument("--lm1b-prompt-tokens", type=int, default=32)
    parser.add_argument("--lm1b-dataset", default="FrankCCCCC/lm1b")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/token2token/anchor_targets/gsm8k_local_unlock.jsonl",
    )
    args = parser.parse_args()
    if args.examples <= 0 or args.anchors <= 0:
        parser.error("examples and anchors must be positive")
    if args.window_size <= 1:
        parser.error("window-size must be greater than one")
    if not 0 < args.target_right_fraction <= 1:
        parser.error("target-right-fraction must be in (0, 1]")
    return args


if __name__ == "__main__":
    main()
