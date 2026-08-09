#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch

from Token2Token.main.core import Target, select_targets
from Token2Token.main.train import (
    MODEL_ID,
    encode_record,
    load_base_model,
    record_stream,
)


TARGET_SOURCE = "frozen_base_confidence_rollout"


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
        targets, scores = greedy_rollout_targets(
            model,
            prompt_ids,
            gold_ids,
            candidates,
            mask_token_id,
            count=args.anchors,
            rollout_steps=args.rollout_steps,
            rollout_k=args.rollout_k,
            batch_size=args.rollout_batch_size,
            device=device,
        )
        if not targets:
            continue
        row = {
            "dataset": args.dataset,
            "example_id": example_id,
            "target_source": TARGET_SOURCE,
            "selection_metric": "confidence_rollout_correct_count",
            "source": source_fields(args.dataset, source_record[1]),
            "prompt_ids": prompt_ids,
            "gold_ids": gold_ids,
            "targets": [
                {
                    "rank": rank + 1,
                    "gold_position": target.gold_position,
                    "token_id": target.token_id,
                    "token": target.token_text,
                    "selection_score": float(scores[rank]["correct"]),
                    "rollout_correct": int(scores[rank]["correct"]),
                    "rollout_committed": int(scores[rank]["committed"]),
                    "rollout_steps": args.rollout_steps,
                    "rollout_k": args.rollout_k,
                }
                for rank, target in enumerate(targets)
            ],
        }
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        written += 1
        print(
            f"example={written}/{args.examples} id={example_id} "
            f"scores={[score['correct'] for score in scores]}"
        )


def greedy_rollout_targets(
    model,
    prompt_ids,
    gold_ids,
    candidates,
    mask_token_id,
    *,
    count,
    rollout_steps,
    rollout_k,
    batch_size,
    device,
):
    if batch_size <= 0 or rollout_steps <= 0 or rollout_k <= 0:
        raise ValueError("rollout batch size, steps, and k must be positive")
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
                remaining = [
                    target
                    for target in candidates
                    if target.gold_position not in selected_positions
                ]
                best_target = None
                best_correct = -1
                best_committed = 0
                for start in range(0, len(remaining), batch_size):
                    chunk = remaining[start : start + batch_size]
                    candidate_canvases = []
                    for target in chunk:
                        candidate_canvas = list(canvas)
                        candidate_canvas[target.gold_position] = target.token_id
                        candidate_canvases.append(candidate_canvas)
                    correct, committed = confidence_rollout_correct_counts(
                        model,
                        prompt_ids,
                        candidate_canvases,
                        gold_ids,
                        mask_token_id,
                        steps=rollout_steps,
                        tokens_per_step=rollout_k,
                        device=device,
                    )
                    for target, target_correct, target_committed in zip(
                        chunk, correct, committed
                    ):
                        if int(target_correct) > best_correct:
                            best_target = target
                            best_correct = int(target_correct)
                            best_committed = int(target_committed)
                if best_target is None:
                    break
                selected.append(
                    Target(
                        rank,
                        best_target.token_id,
                        best_target.token_text,
                        best_target.gold_position,
                    )
                )
                scores.append(
                    {"correct": best_correct, "committed": best_committed}
                )
                selected_positions.add(best_target.gold_position)
                canvas[best_target.gold_position] = best_target.token_id
    finally:
        model.train(was_training)
    return selected, scores


def confidence_rollout_correct_counts(
    model,
    prompt_ids,
    canvases,
    gold_ids,
    mask_token_id,
    *,
    steps,
    tokens_per_step,
    device,
):
    canvas_tensor = torch.tensor(canvases, device=device, dtype=torch.long)
    prompt = torch.tensor(prompt_ids, device=device, dtype=torch.long)
    prompts = prompt.unsqueeze(0).expand(len(canvases), -1)
    gold = torch.tensor(gold_ids, device=device, dtype=torch.long)
    correct = torch.zeros(len(canvases), device=device, dtype=torch.long)
    committed = torch.zeros(len(canvases), device=device, dtype=torch.long)

    for _ in range(steps):
        masked = canvas_tensor.eq(mask_token_id)
        remaining = int(masked.sum(dim=1).min().item())
        if remaining == 0:
            break
        input_ids = torch.cat([prompts, canvas_tensor], dim=1)
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[:, len(prompt_ids) :].float()
        logits[:, :, mask_token_id] = -torch.inf
        probabilities = torch.softmax(logits, dim=-1)
        confidence, predicted = probabilities.max(dim=-1)
        fill_count = min(tokens_per_step, remaining)
        ranked = confidence.masked_fill(~masked, -torch.inf)
        positions = torch.topk(ranked, fill_count, dim=-1).indices
        token_ids = torch.gather(predicted, 1, positions)
        gold_at_positions = gold.unsqueeze(0).expand(len(canvases), -1).gather(
            1, positions
        )
        correct += token_ids.eq(gold_at_positions).sum(dim=1)
        committed += fill_count
        canvas_tensor.scatter_(1, positions, token_ids)
    return correct.cpu().tolist(), committed.cpu().tolist()


def source_fields(dataset: str, row) -> dict:
    if dataset == "gsm8k":
        return {"question": str(row["question"]), "answer": str(row["answer"])}
    return {"text": str(row["text"])}


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
            "rollout_steps",
            "rollout_k",
            "max_completion_tokens",
            "seed",
            "target_source",
        ]
        if any(existing.get(key) != metadata.get(key) for key in comparable):
            raise ValueError("resume configuration does not match target metadata")
        existing["rollout_batch_size"] = args.rollout_batch_size
        existing["resume"] = True
        metadata_path.write_text(
            json.dumps(existing, indent=2) + "\n", encoding="utf-8"
        )
        return
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute anchors by standard-decoder rollout correctness"
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset", choices=["gsm8k", "lm1b"], default="gsm8k")
    parser.add_argument("--examples", type=int, default=7_473)
    parser.add_argument("--anchors", type=int, default=2)
    parser.add_argument("--target-right-fraction", type=float, default=0.75)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--rollout-k", type=int, default=2)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    parser.add_argument("--lm1b-prompt-tokens", type=int, default=32)
    parser.add_argument("--lm1b-dataset", default="FrankCCCCC/lm1b")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/token2token/anchor_targets/gsm8k_rollout_k2.jsonl",
    )
    args = parser.parse_args()
    if args.examples <= 0 or args.anchors <= 0:
        parser.error("examples and anchors must be positive")
    if not 0 < args.target_right_fraction <= 1:
        parser.error("target-right-fraction must be in (0, 1]")
    return args


if __name__ == "__main__":
    main()
