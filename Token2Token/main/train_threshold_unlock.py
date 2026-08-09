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

from Token2Token.main.precompute_threshold_unlock_targets import TARGET_SOURCE
from Token2Token.main.train import MODEL_ID, load_model, save_adapter


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2) + "\n", encoding="utf-8"
    )
    records = read_records(Path(args.targets_file))
    if not records:
        raise ValueError(f"no threshold-unlock records in {args.targets_file}")
    validate_target_file(Path(args.targets_file), records)

    tokenizer, model, mask_token_id, device = load_model(args)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    log_path = output / "train.jsonl"
    step = 0
    started = time.time()

    while step < args.max_steps:
        record = records[step % len(records)]
        prompt_ids = list(map(int, record["prompt_ids"]))
        gold_ids = list(map(int, record["gold_ids"]))
        stages = trajectory_stages(record, mask_token_id)
        sampled = sample_stages(stages, args.stages_per_example, random)
        model_stages = list(sampled)
        standard_stage = None
        mask_ratio = None
        if args.standard_loss_weight:
            standard_stage, mask_ratio = random_denoising_stage(
                gold_ids,
                mask_token_id,
                args.min_mask_ratio,
                args.max_mask_ratio,
            )
            model_stages.append(standard_stage)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device, args.bf16):
            logits = batched_completion_logits(
                model,
                prompt_ids,
                [item["canvas"] for item in model_stages],
                device,
            )
            catalyst_logits = logits[: len(sampled)]
            catalyst_loss = trajectory_loss(catalyst_logits, sampled)
            standard_loss = catalyst_loss.detach() * 0.0
            if standard_stage is not None:
                standard_loss = stage_loss(logits[-1], standard_stage)
            loss = (
                args.catalyst_loss_weight * catalyst_loss
                + args.standard_loss_weight * standard_loss
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()

        step += 1
        global_step = args.step_offset + step
        row = {
            "step": global_step,
            "dataset": record["dataset"],
            "example_id": record["example_id"],
            "loss": float(loss.detach().cpu()),
            "catalyst_loss": float(catalyst_loss.detach().cpu()),
            "standard_loss": float(standard_loss.detach().cpu()),
            "catalyst_loss_weight": args.catalyst_loss_weight,
            "standard_loss_weight": args.standard_loss_weight,
            "sampled_stages": len(sampled),
            "mask_ratio": mask_ratio,
            "elapsed_seconds": time.time() - started,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        print(
            f"step={global_step} loss={row['loss']:.4f} "
            f"catalyst={row['catalyst_loss']:.4f} "
            f"standard={row['standard_loss']:.4f} "
            f"catalyst_stages={len(sampled)}"
        )
        if args.save_every and step % args.save_every == 0:
            save_adapter(model, tokenizer, output / f"checkpoint-{global_step:06d}")

    save_adapter(model, tokenizer, output / "adapter-final")


def trajectory_stages(record, mask_token_id):
    gold_ids = list(map(int, record["gold_ids"]))
    canvas = [int(mask_token_id)] * len(gold_ids)
    stages = []
    for expected_round, item in enumerate(record["rounds"], start=1):
        if int(item["round"]) != expected_round:
            raise ValueError("trajectory rounds must be consecutive")
        catalyst = item["catalyst"]
        catalyst_position = int(catalyst["gold_position"])
        catalyst_token_id = int(catalyst["token_id"])
        validate_gold_target(
            canvas,
            gold_ids,
            catalyst_position,
            catalyst_token_id,
            mask_token_id,
        )
        stages.append(
            {
                "kind": "catalyst",
                "round": expected_round,
                "canvas": list(canvas),
                "positions": [catalyst_position],
                "token_ids": [catalyst_token_id],
            }
        )
        canvas[catalyst_position] = catalyst_token_id

        for target in item["unlocked"]:
            position = int(target["gold_position"])
            token_id = int(target["token_id"])
            validate_gold_target(canvas, gold_ids, position, token_id, mask_token_id)
            canvas[position] = token_id

    residual = sorted(
        record.get("residual", []), key=lambda item: int(item["gold_position"])
    )
    for cleanup_index, target in enumerate(residual, start=1):
        position = int(target["gold_position"])
        token_id = int(target["token_id"])
        validate_gold_target(canvas, gold_ids, position, token_id, mask_token_id)
        stages.append(
            {
                "kind": "left_to_right_cleanup",
                "round": len(record["rounds"]) + cleanup_index,
                "canvas": list(canvas),
                "positions": [position],
                "token_ids": [token_id],
            }
        )
        canvas[position] = token_id

    if canvas != gold_ids:
        missing = [
            position
            for position, token_id in enumerate(canvas)
            if token_id == mask_token_id
        ]
        raise ValueError(f"threshold trajectory is incomplete at positions {missing}")
    return stages


def validate_gold_target(canvas, gold_ids, position, token_id, mask_token_id):
    if position < 0 or position >= len(gold_ids):
        raise ValueError(f"target position {position} is outside the completion")
    if canvas[position] != mask_token_id:
        raise ValueError(f"target position {position} was already filled")
    if int(gold_ids[position]) != token_id:
        raise ValueError("trajectory token does not match the gold completion")


def sample_stages(stages, count, rng):
    if count <= 0 or count >= len(stages):
        return list(stages)
    return sorted(rng.sample(stages, count), key=lambda item: item["round"])


def random_denoising_stage(gold_ids, mask_token_id, min_mask_ratio, max_mask_ratio):
    mask_ratio = random.uniform(min_mask_ratio, max_mask_ratio)
    mask = torch.rand(len(gold_ids)) < mask_ratio
    if not bool(mask.any()):
        mask[random.randrange(len(gold_ids))] = True
    positions = torch.where(mask)[0].tolist()
    canvas = [
        int(mask_token_id) if bool(mask[position]) else int(token_id)
        for position, token_id in enumerate(gold_ids)
    ]
    return (
        {
            "kind": "standard",
            "round": 0,
            "canvas": canvas,
            "positions": positions,
            "token_ids": [int(gold_ids[position]) for position in positions],
        },
        mask_ratio,
    )


def batched_completion_logits(model, prompt_ids, canvases, device):
    input_ids = torch.tensor(
        [prompt_ids + canvas for canvas in canvases],
        device=device,
        dtype=torch.long,
    )
    outputs = model(input_ids=input_ids, use_cache=False)
    return outputs.logits[:, len(prompt_ids) : len(prompt_ids) + len(canvases[0])]


def trajectory_loss(logits, stages):
    losses = [
        stage_token_losses(row_logits, stage)
        for row_logits, stage in zip(logits, stages)
    ]
    return torch.cat(losses).mean()


def stage_token_losses(logits, stage):
    positions = torch.tensor(stage["positions"], device=logits.device, dtype=torch.long)
    labels = torch.tensor(stage["token_ids"], device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits[positions].float(), labels, reduction="none")


def stage_loss(logits, stage):
    return stage_token_losses(logits, stage).mean()


def read_records(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_target_file(path: Path, records):
    metadata_path = path.with_suffix(".config.json")
    if not metadata_path.exists():
        raise ValueError(f"missing target metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("target_source") != TARGET_SOURCE:
        raise ValueError(
            f"expected {TARGET_SOURCE!r}, got {metadata.get('target_source')!r}"
        )
    invalid = {
        record.get("target_source")
        for record in records
        if record.get("target_source") != TARGET_SOURCE
    }
    if invalid:
        raise ValueError(f"invalid threshold target rows: {sorted(invalid)!r}")


def autocast(device, enabled):
    if str(device).startswith("cuda") and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LoRA on after-minus-before catalyst targets"
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--adapter-path")
    parser.add_argument("--targets-file", required=True)
    parser.add_argument("--max-steps", type=int, default=7_473)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--stages-per-example", type=int, default=8)
    parser.add_argument("--catalyst-loss-weight", type=float, default=1.0)
    parser.add_argument("--standard-loss-weight", type=float, default=0.0)
    parser.add_argument("--min-mask-ratio", type=float, default=0.15)
    parser.add_argument("--max-mask-ratio", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,attn_out")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="outputs/token2token/threshold_unlock/llada8b_gsm8k_t095_text_q07_anchor_only",
    )
    args = parser.parse_args()
    if args.max_steps <= 0 or args.stages_per_example <= 0:
        parser.error("max-steps and stages-per-example must be positive")
    if args.step_offset < 0:
        parser.error("step-offset must be non-negative")
    if not 0 < args.min_mask_ratio <= args.max_mask_ratio <= 1:
        parser.error("mask ratios must satisfy 0 < min <= max <= 1")
    weights = [
        args.catalyst_loss_weight,
        args.standard_loss_weight,
    ]
    if any(weight < 0 for weight in weights) or not any(weights):
        parser.error("loss weights must be non-negative with at least one positive")
    return args


if __name__ == "__main__":
    main()
