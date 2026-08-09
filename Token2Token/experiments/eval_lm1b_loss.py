#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from Token2Token.main.train import MODEL_ID, _token_ids, completion_logits, load_base_model


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer, model, mask_token_id, device = load_base_model(
        args.model_id, args.device
    )
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path).to(device)
    model.eval()

    from datasets import load_dataset

    rows = load_dataset(args.dataset, split=args.split, streaming=True)
    totals = {
        ratio: {"nll": 0.0, "tokens": 0, "examples": 0}
        for ratio in args.mask_ratios
    }
    accepted = 0
    with torch.no_grad():
        for source_index, row in enumerate(rows):
            ids = _token_ids(tokenizer(str(row["text"]), add_special_tokens=True))
            if len(ids) <= args.prompt_tokens + 1:
                continue
            prompt_ids = ids[: args.prompt_tokens]
            gold_ids = ids[
                args.prompt_tokens : args.prompt_tokens + args.completion_tokens
            ]
            for ratio_index, ratio in enumerate(args.mask_ratios):
                generator = torch.Generator().manual_seed(
                    args.seed + source_index * len(args.mask_ratios) + ratio_index
                )
                masked = torch.rand(len(gold_ids), generator=generator) < ratio
                if not bool(masked.any()):
                    masked[0] = True
                canvas = [
                    mask_token_id if bool(masked[position]) else token_id
                    for position, token_id in enumerate(gold_ids)
                ]
                logits = completion_logits(model, prompt_ids, canvas, device)
                positions = torch.where(masked)[0].to(device)
                labels = torch.tensor(gold_ids, device=device, dtype=torch.long)[
                    positions
                ]
                nll = F.cross_entropy(
                    logits[positions].float(), labels, reduction="sum"
                )
                totals[ratio]["nll"] += float(nll.cpu())
                totals[ratio]["tokens"] += len(positions)
                totals[ratio]["examples"] += 1
            accepted += 1
            print(f"model={args.model_label} example={accepted}/{args.limit}")
            if accepted >= args.limit:
                break

    results = []
    for ratio in args.mask_ratios:
        total = totals[ratio]
        ce = total["nll"] / total["tokens"]
        results.append(
            {
                "mask_ratio": ratio,
                "examples": total["examples"],
                "masked_tokens": total["tokens"],
                "cross_entropy": ce,
                "masked_pseudo_perplexity": math.exp(ce),
            }
        )
    summary = {
        "model_label": args.model_label,
        "model_id": args.model_id,
        "adapter_path": args.adapter_path,
        "dataset": args.dataset,
        "split": args.split,
        "results": results,
    }
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_mask_ratios(value: str) -> list[float]:
    ratios = [float(item) for item in value.split(",") if item.strip()]
    if not ratios or any(ratio <= 0 or ratio > 1 for ratio in ratios):
        raise ValueError("mask ratios must satisfy 0 < ratio <= 1")
    return ratios


def parse_args():
    parser = argparse.ArgumentParser(
        description="Held-out LM1B masked CE and pseudo-perplexity"
    )
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--adapter-path")
    parser.add_argument("--dataset", default="FrankCCCCC/lm1b")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--completion-tokens", type=int, default=128)
    parser.add_argument("--mask-ratios", type=parse_mask_ratios, default="0.25,0.5,0.75,1.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.limit <= 0 or args.prompt_tokens <= 0 or args.completion_tokens <= 0:
        parser.error("limit and token lengths must be positive")
    return args


if __name__ == "__main__":
    main()
