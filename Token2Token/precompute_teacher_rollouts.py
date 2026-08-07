#!/usr/bin/env python3
"""Cache frozen-base block-k1 trajectories for lookahead distillation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from Token2Token.eval_threshold_gsm8k import (
    batch_canvas_predictions,
    current_block,
    pad_prompts,
)
from Token2Token.train import MODEL_ID, _token_ids, load_base_model


TARGET_SOURCE = "frozen_base_block_k1_rollout"


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = count_rows(output) if output.exists() else 0
    if completed and not args.resume:
        raise FileExistsError(f"{output} already exists; pass --resume to continue")
    write_config(output, args)

    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="train").shuffle(
        seed=args.seed
    )
    if args.examples > len(dataset):
        raise ValueError(f"requested {args.examples} examples from {len(dataset)} rows")

    tokenizer, model, mask_token_id, device = load_base_model(
        args.model_id, args.device
    )
    model.eval()
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else mask_token_id
    )

    for start in range(completed, args.examples, args.batch_size):
        stop = min(start + args.batch_size, args.examples)
        source_rows = [dataset[index] for index in range(start, stop)]
        prompt_ids_batch = [encode_prompt(tokenizer, row["question"]) for row in source_rows]
        _, actions = teacher_rollout_actions(
            model,
            prompt_ids_batch,
            args.completion_length,
            args.block_length,
            mask_token_id,
            device,
            pad_token_id,
        )
        with output.open("a", encoding="utf-8") as handle:
            for index, (row, prompt_ids, row_actions) in enumerate(
                zip(source_rows, prompt_ids_batch, actions), start=start
            ):
                record = {
                    "dataset": "gsm8k",
                    "example_id": str(index),
                    "target_source": TARGET_SOURCE,
                    "completion_length": args.completion_length,
                    "block_length": args.block_length,
                    "prompt_ids": prompt_ids,
                    "source": {
                        "question": str(row["question"]),
                        "answer": str(row["answer"]),
                    },
                    "actions": row_actions,
                }
                validate_rollout_record(record)
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        print(f"examples={stop}/{args.examples}")

    summary = summarize(output)
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def teacher_rollout_actions(
    model,
    prompt_ids_batch,
    completion_length,
    block_length,
    mask_token_id,
    device,
    pad_token_id,
):
    """Run the exact block-k1 schedule used as the quality teacher."""
    prompts, prompt_attention = pad_prompts(
        prompt_ids_batch, pad_token_id, device
    )
    canvases = torch.full(
        (len(prompt_ids_batch), completion_length),
        int(mask_token_id),
        device=device,
        dtype=torch.long,
    )
    completion_attention = torch.ones_like(canvases)
    actions = [[] for _ in prompt_ids_batch]

    with torch.no_grad():
        for step in range(completion_length):
            masked = canvases.eq(mask_token_id)
            confidence, token_ids = batch_canvas_predictions(
                model,
                prompts,
                prompt_attention,
                canvases,
                completion_attention,
                mask_token_id,
            )
            eligible = masked & current_block(masked, block_length)
            scores = confidence.masked_fill(~eligible, -torch.inf)
            positions = scores.argmax(dim=1)
            rows = torch.arange(len(canvases), device=canvases.device)
            chosen_tokens = token_ids[rows, positions]
            chosen_confidence = confidence[rows, positions]
            canvases[rows, positions] = chosen_tokens
            for row_index in range(len(actions)):
                actions[row_index].append(
                    {
                        "step": step,
                        "position": int(positions[row_index]),
                        "token_id": int(chosen_tokens[row_index]),
                        "confidence": float(chosen_confidence[row_index]),
                    }
                )
    return canvases.detach().cpu().tolist(), actions


def encode_prompt(tokenizer, question):
    prompt = str(question).rstrip() + "\n"
    return list(
        map(
            int,
            _token_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=True,
                )
            ),
        )
    )


def validate_rollout_record(record) -> None:
    length = int(record["completion_length"])
    actions = record["actions"]
    positions = [int(item["position"]) for item in actions]
    if record.get("target_source") != TARGET_SOURCE:
        raise ValueError("invalid rollout target source")
    if len(actions) != length or sorted(positions) != list(range(length)):
        raise ValueError("rollout must fill each completion position exactly once")
    if any(int(item["step"]) != index for index, item in enumerate(actions)):
        raise ValueError("rollout steps must be consecutive")


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def summarize(path: Path):
    examples = 0
    actions = 0
    confidence = 0.0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            validate_rollout_record(record)
            examples += 1
            actions += len(record["actions"])
            confidence += sum(float(item["confidence"]) for item in record["actions"])
    return {
        "target_source": TARGET_SOURCE,
        "examples": examples,
        "actions": actions,
        "mean_teacher_confidence": confidence / actions if actions else 0.0,
    }


def write_config(output: Path, args) -> None:
    path = output.with_suffix(".config.json")
    config = {**vars(args), "target_source": TARGET_SOURCE}
    if path.exists() and args.resume:
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = (
            "model_id",
            "completion_length",
            "block_length",
            "seed",
            "target_source",
        )
        if any(existing.get(key) != config.get(key) for key in comparable):
            raise ValueError("resume configuration does not match rollout metadata")
        return
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--completion-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/token2token/lookahead_v6/base_block32_k1_train100.jsonl",
    )
    args = parser.parse_args()
    if min(args.examples, args.completion_length, args.block_length, args.batch_size) <= 0:
        parser.error("examples, lengths, and batch-size must be positive")
    return args


if __name__ == "__main__":
    main()
