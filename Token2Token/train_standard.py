#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from Token2Token.train import (
    MODEL_ID,
    completion_logits,
    encode_record,
    load_model,
    record_stream,
    save_adapter,
)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2) + "\n", encoding="utf-8"
    )

    tokenizer, model, mask_token_id, device = load_model(args)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    records = record_stream(args)
    log_path = output / "train.jsonl"
    step = 0
    started = time.time()

    while step < args.max_steps:
        encoded = encode_record(next(records), tokenizer, args)
        if encoded is None:
            continue
        prompt_ids, gold_ids, example_id = encoded
        for update_index in range(args.updates_per_example):
            if step >= args.max_steps:
                break
            mask_ratio = random.uniform(args.min_mask_ratio, args.max_mask_ratio)
            mask = torch.rand(len(gold_ids)) < mask_ratio
            if not bool(mask.any()):
                mask[random.randrange(len(gold_ids))] = True
            canvas = [
                mask_token_id if bool(mask[position]) else token_id
                for position, token_id in enumerate(gold_ids)
            ]
            positions = torch.where(mask)[0].to(device)
            labels = torch.tensor(gold_ids, device=device, dtype=torch.long)[positions]

            optimizer.zero_grad(set_to_none=True)
            with autocast(device, args.bf16):
                logits = completion_logits(model, prompt_ids, canvas, device)
                loss = masked_denoising_loss(logits, positions, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()

            step += 1
            row = {
                "step": step,
                "dataset": args.dataset,
                "example_id": example_id,
                "update_index": update_index + 1,
                "loss": float(loss.detach().cpu()),
                "mask_ratio": mask_ratio,
                "masked_tokens": int(mask.sum()),
                "elapsed_seconds": time.time() - started,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            print(
                f"step={step} loss={row['loss']:.4f} "
                f"mask_ratio={mask_ratio:.3f} masked={row['masked_tokens']}"
            )
            if args.save_every and step % args.save_every == 0:
                save_adapter(model, tokenizer, output / f"checkpoint-{step:06d}")

    save_adapter(model, tokenizer, output / "adapter-final")


def masked_denoising_loss(logits, positions, labels):
    return F.cross_entropy(logits[positions].float(), labels)


def autocast(device: str, enabled: bool):
    if str(device).startswith("cuda") and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standard masked-denoising LLaDA LoRA baseline"
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset", choices=["gsm8k", "lm1b"], default="gsm8k")
    parser.add_argument("--max-steps", type=int, default=7_473)
    parser.add_argument("--updates-per-example", type=int, default=1)
    parser.add_argument("--min-mask-ratio", type=float, default=0.15)
    parser.add_argument("--max-mask-ratio", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,attn_out")
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    parser.add_argument("--lm1b-prompt-tokens", type=int, default=32)
    parser.add_argument("--lm1b-dataset", default="FrankCCCCC/lm1b")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", default="outputs/token2token/standard_lora_gsm8k"
    )
    args = parser.parse_args()
    if not 0 < args.min_mask_ratio <= args.max_mask_ratio <= 1:
        parser.error("mask ratios must satisfy 0 < min <= max <= 1")
    return args


if __name__ == "__main__":
    main()
