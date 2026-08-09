#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch

from Token2Token.main.core import select_targets
from Token2Token.main.train import (
    MODEL_ID,
    encode_record,
    greedy_ig_targets,
    load_base_model,
    record_stream,
)


TARGET_SOURCE = "frozen_base_greedy_ig"


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
        targets, ig_scores = greedy_ig_targets(
            model,
            prompt_ids,
            gold_ids,
            candidates,
            mask_token_id,
            count=args.anchors,
            batch_size=args.ig_batch_size,
            device=device,
        )
        if not targets:
            continue
        row = {
            "dataset": args.dataset,
            "example_id": example_id,
            "target_source": TARGET_SOURCE,
            "source": source_fields(args.dataset, source_record[1]),
            "prompt_ids": prompt_ids,
            "gold_ids": gold_ids,
            "targets": [
                {
                    "rank": rank + 1,
                    "gold_position": target.gold_position,
                    "token_id": target.token_id,
                    "token": target.token_text,
                    "ig_score": float(ig_scores[rank]),
                }
                for rank, target in enumerate(targets)
            ],
        }
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        written += 1
        print(
            f"example={written}/{args.examples} id={example_id} "
            f"anchors={len(targets)}"
        )


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
            "max_completion_tokens",
            "seed",
        ]
        if any(existing.get(key) != metadata.get(key) for key in comparable):
            raise ValueError("resume configuration does not match target metadata")
        existing["ig_batch_size"] = args.ig_batch_size
        existing["resume"] = True
        existing["target_source"] = TARGET_SOURCE
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
        description="Precompute frozen greedy-IG gold anchor order"
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset", choices=["gsm8k", "lm1b"], default="gsm8k")
    parser.add_argument("--examples", type=int, default=7_473)
    parser.add_argument("--anchors", type=int, default=5)
    parser.add_argument("--target-right-fraction", type=float, default=0.75)
    parser.add_argument("--ig-batch-size", type=int, default=64)
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    parser.add_argument("--lm1b-prompt-tokens", type=int, default=32)
    parser.add_argument("--lm1b-dataset", default="FrankCCCCC/lm1b")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/token2token/anchor_targets/gsm8k_train.jsonl",
    )
    args = parser.parse_args()
    if args.examples <= 0 or args.anchors <= 0 or args.ig_batch_size <= 0:
        parser.error("examples, anchors, and ig-batch-size must be positive")
    return args


if __name__ == "__main__":
    main()
