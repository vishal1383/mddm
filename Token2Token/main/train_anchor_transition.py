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
from Token2Token.main.train_threshold_unlock import (
    batched_completion_logits,
    random_denoising_stage,
    stage_loss,
    trajectory_loss,
    validate_gold_target,
    validate_target_file,
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
    records = read_records(Path(args.targets_file), args.record_limit)
    if not records:
        raise ValueError(f"no threshold-unlock records in {args.targets_file}")
    validate_target_file(Path(args.targets_file), records)

    tokenizer, model, mask_token_id, device = load_model(args)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    log_path = output / "train.jsonl"
    started = time.time()

    for step in range(1, args.max_steps + 1):
        record = records[(step - 1) % len(records)]
        prompt_ids = list(map(int, record["prompt_ids"]))
        gold_ids = list(map(int, record["gold_ids"]))
        transitions = anchor_transitions(record, mask_token_id, args.max_unlock_tokens)
        sampled = sample_transitions(transitions, args.transitions_per_example, random)
        anchor_stages = [item["anchor"] for item in sampled]
        post_anchor_stages = [
            item["post_anchor"] for item in sampled if item["post_anchor"] is not None
        ]
        standard_stage, mask_ratio = random_denoising_stage(
            gold_ids,
            mask_token_id,
            args.min_mask_ratio,
            args.max_mask_ratio,
        )
        model_stages = anchor_stages + post_anchor_stages + [standard_stage]

        teacher_logits = None
        if args.base_kl_weight:
            was_training = model.training
            model.eval()
            with torch.no_grad(), model.disable_adapter(), autocast(device, args.bf16):
                teacher_logits = batched_completion_logits(
                    model,
                    prompt_ids,
                    [standard_stage["canvas"]],
                    device,
                )[0].detach()
            if was_training:
                model.train()

        optimizer.zero_grad(set_to_none=True)
        with autocast(device, args.bf16):
            logits = batched_completion_logits(
                model,
                prompt_ids,
                [item["canvas"] for item in model_stages],
                device,
            )
            zero = logits.sum() * 0.0
            anchor_loss = zero
            if anchor_stages:
                anchor_loss = trajectory_loss(logits[: len(anchor_stages)], anchor_stages)
            unlock_start = len(anchor_stages)
            unlock_end = unlock_start + len(post_anchor_stages)
            unlock_loss = zero
            unlock_positions = []
            if post_anchor_stages:
                unlock_loss, unlock_positions = post_anchor_topk_loss(
                    logits[unlock_start:unlock_end],
                    post_anchor_stages,
                    gold_ids,
                    mask_token_id,
                    args.max_unlock_tokens,
                )
            standard_loss = stage_loss(logits[-1], standard_stage)
            base_kl_loss = zero
            if teacher_logits is not None:
                base_kl_loss = masked_kl_loss(
                    logits[-1],
                    teacher_logits,
                    standard_stage["canvas"],
                    mask_token_id,
                )
            loss = (
                args.standard_loss_weight * standard_loss
                + args.anchor_loss_weight * anchor_loss
                + args.unlock_loss_weight * unlock_loss
                + args.base_kl_weight * base_kl_loss
            )

        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()

        row = {
            "step": step,
            "dataset": record["dataset"],
            "example_id": record["example_id"],
            "loss": float(loss.detach().cpu()),
            "anchor_loss": float(anchor_loss.detach().cpu()),
            "unlock_loss": float(unlock_loss.detach().cpu()),
            "standard_loss": float(standard_loss.detach().cpu()),
            "base_kl_loss": float(base_kl_loss.detach().cpu()),
            "sampled_transitions": len(sampled),
            "anchor_stages": len(anchor_stages),
            "post_anchor_stages": len(post_anchor_stages),
            "unlock_tokens": sum(len(positions) for positions in unlock_positions),
            "mask_ratio": mask_ratio,
            "elapsed_seconds": time.time() - started,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        print(
            f"step={step} loss={row['loss']:.4f} "
            f"standard={row['standard_loss']:.4f} "
            f"anchor={row['anchor_loss']:.4f} unlock={row['unlock_loss']:.4f} "
            f"base_kl={row['base_kl_loss']:.4f} "
            f"transitions={len(sampled)} unlock_tokens={row['unlock_tokens']}"
        )
        if args.save_every and step % args.save_every == 0:
            save_adapter(model, tokenizer, output / f"checkpoint-{step:06d}")

    save_adapter(model, tokenizer, output / "adapter-final")


def anchor_transitions(record, mask_token_id, max_unlock_tokens):
    if max_unlock_tokens <= 0:
        raise ValueError("max_unlock_tokens must be positive")
    gold_ids = list(map(int, record["gold_ids"]))
    canvas = [int(mask_token_id)] * len(gold_ids)
    transitions = []
    for expected_round, item in enumerate(record["rounds"], start=1):
        if int(item["round"]) != expected_round:
            raise ValueError("trajectory rounds must be consecutive")
        catalyst = item["catalyst"]
        position = int(catalyst["gold_position"])
        token_id = int(catalyst["token_id"])
        validate_gold_target(canvas, gold_ids, position, token_id, mask_token_id)
        anchor_stage = {
            "kind": "anchor",
            "round": expected_round,
            "canvas": list(canvas),
            "positions": [position],
            "token_ids": [token_id],
        }

        post_anchor_canvas = list(canvas)
        post_anchor_canvas[position] = token_id
        masked_after_anchor = [
            index
            for index, canvas_token_id in enumerate(post_anchor_canvas)
            if canvas_token_id == mask_token_id
        ]
        post_anchor_stage = None
        if masked_after_anchor:
            post_anchor_stage = {
                "kind": "post_anchor_unlock",
                "round": expected_round,
                "canvas": post_anchor_canvas,
            }
        transitions.append(
            {
                "round": expected_round,
                "anchor": anchor_stage,
                "post_anchor": post_anchor_stage,
            }
        )

        canvas[position] = token_id
        for target in item["unlocked"]:
            unlocked_position = int(target["gold_position"])
            unlocked_token_id = int(target["token_id"])
            validate_gold_target(
                canvas,
                gold_ids,
                unlocked_position,
                unlocked_token_id,
                mask_token_id,
            )
            canvas[unlocked_position] = unlocked_token_id
    return transitions


def post_anchor_topk_loss(
    logits, stages, gold_ids, mask_token_id, max_unlock_tokens
):
    losses = []
    selected_rows = []
    labels = torch.tensor(gold_ids, device=logits.device, dtype=torch.long)
    for row_logits, stage in zip(logits, stages):
        positions = post_anchor_topk_positions(
            row_logits, stage["canvas"], mask_token_id, max_unlock_tokens
        )
        selected_rows.append(positions)
        if not positions:
            continue
        position_tensor = torch.tensor(
            positions, device=logits.device, dtype=torch.long
        )
        losses.append(
            F.cross_entropy(
                row_logits[position_tensor].float(),
                labels[position_tensor],
                reduction="none",
            )
        )
    if not losses:
        return logits.sum() * 0.0, selected_rows
    return torch.cat(losses).mean(), selected_rows


def post_anchor_topk_positions(logits, canvas, mask_token_id, count):
    masked = [
        position
        for position, token_id in enumerate(canvas)
        if int(token_id) == int(mask_token_id)
    ]
    if not masked:
        return []
    masked_logits = logits[masked].float().clone()
    masked_logits[:, int(mask_token_id)] = -torch.inf
    confidence = torch.softmax(masked_logits, dim=-1).amax(dim=-1)
    selected = confidence.topk(min(int(count), len(masked))).indices.tolist()
    return [masked[index] for index in selected]


def masked_kl_loss(student_logits, teacher_logits, canvas, mask_token_id):
    positions = [
        position
        for position, token_id in enumerate(canvas)
        if int(token_id) == int(mask_token_id)
    ]
    if not positions:
        return student_logits.sum() * 0.0
    position_tensor = torch.tensor(
        positions, device=student_logits.device, dtype=torch.long
    )
    student = student_logits[position_tensor].float()
    teacher = teacher_logits[position_tensor].float()
    return F.kl_div(
        F.log_softmax(student, dim=-1),
        F.softmax(teacher, dim=-1),
        reduction="batchmean",
    )


def sample_transitions(transitions, count, rng):
    if count <= 0 or count >= len(transitions):
        return list(transitions)
    return sorted(rng.sample(transitions, count), key=lambda item: item["round"])


def read_records(path: Path, limit=None):
    if not path.exists():
        raise FileNotFoundError(path)
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def autocast(device, enabled):
    if str(device).startswith("cuda") and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train anchors and capped post-anchor transitions with denoising CE"
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--adapter-path")
    parser.add_argument("--targets-file", required=True)
    parser.add_argument("--record-limit", type=int, default=7_473)
    parser.add_argument("--max-steps", type=int, default=7_473)
    parser.add_argument("--transitions-per-example", type=int, default=3)
    parser.add_argument("--max-unlock-tokens", type=int, default=2)
    parser.add_argument("--standard-loss-weight", type=float, default=1.0)
    parser.add_argument("--anchor-loss-weight", type=float, default=0.5)
    parser.add_argument("--unlock-loss-weight", type=float, default=1.0)
    parser.add_argument("--base-kl-weight", type=float, default=0.0)
    parser.add_argument("--min-mask-ratio", type=float, default=0.15)
    parser.add_argument("--max-mask-ratio", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,attn_out")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.record_limit is not None and args.record_limit <= 0:
        parser.error("record-limit must be positive")
    if args.max_steps <= 0:
        parser.error("max-steps must be positive")
    if args.transitions_per_example <= 0:
        parser.error("transitions-per-example must be positive")
    if args.max_unlock_tokens <= 0:
        parser.error("max-unlock-tokens must be positive")
    if args.standard_loss_weight <= 0:
        parser.error("standard-loss-weight must be positive")
    if (
        args.anchor_loss_weight < 0
        or args.unlock_loss_weight < 0
        or args.base_kl_weight < 0
    ):
        parser.error("anchor, unlock, and base KL loss weights must be nonnegative")
    return args


if __name__ == "__main__":
    main()
