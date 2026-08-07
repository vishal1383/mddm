#!/usr/bin/env python3
"""Teach one forward to imitate several frozen-teacher decode steps."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from Token2Token.eval_threshold_gsm8k import current_block
from Token2Token.precompute_teacher_rollouts import (
    TARGET_SOURCE,
    validate_rollout_record,
)
from Token2Token.train import MODEL_ID, load_model, save_adapter
from Token2Token.train_threshold_unlock import batched_completion_logits


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
        raise ValueError(f"no teacher rollouts in {args.targets_file}")

    tokenizer, model, mask_token_id, device = load_model(args)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    log_path = output / "train.jsonl"
    started = time.time()

    for step in range(1, args.max_steps + 1):
        record = records[(step - 1) % len(records)]
        stages = sample_rollout_stages(
            record,
            mask_token_id,
            args.lookahead,
            args.states_per_example,
            random,
        )
        prompt_ids = list(map(int, record["prompt_ids"]))
        canvases = [stage["canvas"] for stage in stages]

        was_training = model.training
        model.eval()
        with torch.no_grad(), model.disable_adapter(), autocast(device, args.bf16):
            teacher_logits = batched_completion_logits(
                model, prompt_ids, canvases, device
            ).detach()
        if was_training:
            model.train()

        optimizer.zero_grad(set_to_none=True)
        with autocast(device, args.bf16):
            logits = batched_completion_logits(model, prompt_ids, canvases, device)
            transition_loss = future_token_loss(
                logits, stages, mask_token_id
            )
            preserve_loss = decoder_kl_loss(
                logits, teacher_logits, stages, mask_token_id
            )
            loss = (
                args.transition_loss_weight * transition_loss
                + args.preserve_kl_weight * preserve_loss
            )

        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()

        metrics = transition_metrics(
            logits.detach(), stages, mask_token_id, int(record["block_length"])
        )
        row = {
            "step": step,
            "example_id": record["example_id"],
            "loss": float(loss.detach().cpu()),
            "transition_loss": float(transition_loss.detach().cpu()),
            "preserve_kl": float(preserve_loss.detach().cpu()),
            "states": len(stages),
            **metrics,
            "elapsed_seconds": time.time() - started,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        print(
            f"step={step} loss={row['loss']:.4f} "
            f"transition={row['transition_loss']:.4f} "
            f"preserve={row['preserve_kl']:.4f} "
            f"future_top1={row['future_top1_fraction']:.3f} "
            f"selected={row['target_selection_fraction']:.3f} "
            f"joint={row['joint_success_fraction']:.3f}"
        )
        if args.save_every and step % args.save_every == 0:
            save_adapter(model, tokenizer, output / f"checkpoint-{step:06d}")

    save_adapter(model, tokenizer, output / "adapter-final")


def sample_rollout_stages(record, mask_token_id, lookahead, count, rng):
    validate_rollout_record(record)
    actions = record["actions"]
    block_length = int(record["block_length"])
    valid_starts = [
        start
        for start in range(len(actions) - lookahead + 1)
        if start // block_length == (start + lookahead - 1) // block_length
    ]
    if not valid_starts:
        raise ValueError("rollout is shorter than the requested lookahead")
    if count >= len(valid_starts):
        starts = valid_starts
    else:
        starts = [valid_starts[0]]
        remaining = valid_starts[1:]
        starts.extend(rng.sample(remaining, min(count - 1, len(remaining))))
        starts.sort()
    return [make_rollout_stage(record, start, lookahead, mask_token_id) for start in starts]


def make_rollout_stage(record, start, lookahead, mask_token_id):
    actions = record["actions"]
    canvas = [int(mask_token_id)] * int(record["completion_length"])
    for action in actions[:start]:
        canvas[int(action["position"])] = int(action["token_id"])
    targets = [
        {
            "position": int(action["position"]),
            "token_id": int(action["token_id"]),
        }
        for action in actions[start : start + lookahead]
    ]
    if any(canvas[item["position"]] != int(mask_token_id) for item in targets):
        raise ValueError("lookahead target was already committed")
    return {"start": start, "canvas": canvas, "targets": targets}


def future_token_loss(logits, stages, mask_token_id):
    losses = []
    for row_logits, stage in zip(logits, stages):
        future = stage["targets"][1:]
        if not future:
            continue
        positions = torch.tensor(
            [item["position"] for item in future],
            device=logits.device,
            dtype=torch.long,
        )
        labels = torch.tensor(
            [item["token_id"] for item in future],
            device=logits.device,
            dtype=torch.long,
        )
        selected = row_logits[positions].float().clone()
        selected[:, int(mask_token_id)] = -torch.inf
        losses.append(F.cross_entropy(selected, labels, reduction="none"))
    if not losses:
        return logits.sum() * 0.0
    return torch.cat(losses).mean()


def target_selection_loss(
    logits, stages, mask_token_id, block_length, margin=0.0
):
    """Rank every teacher target above every competing position in its block."""
    losses = []
    for row_logits, stage in zip(logits, stages):
        targets = stage["targets"]
        if not targets:
            continue
        rows = row_logits.float().clone()
        rows[:, int(mask_token_id)] = -1e4
        log_probabilities = F.log_softmax(rows, dim=-1)
        confidence = log_probabilities.max(dim=-1).values
        masked = torch.tensor(
            [[int(token_id) == int(mask_token_id) for token_id in stage["canvas"]]],
            device=logits.device,
            dtype=torch.bool,
        )
        competitors = (masked & current_block(masked, block_length))[0]
        positions = torch.tensor(
            [int(item["position"]) for item in targets],
            device=logits.device,
            dtype=torch.long,
        )
        labels = torch.tensor(
            [int(item["token_id"]) for item in targets],
            device=logits.device,
            dtype=torch.long,
        )
        competitors[positions] = False
        if not bool(competitors.any()):
            continue
        target_scores = log_probabilities[positions, labels]
        hardest_competitor = confidence.masked_fill(
            ~competitors, -torch.inf
        ).max()
        weakest_target = target_scores.min()
        losses.append(F.softplus(hardest_competitor - weakest_target + margin))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def decoder_kl_loss(student_logits, teacher_logits, stages, mask_token_id):
    losses = []
    for student, teacher, stage in zip(student_logits, teacher_logits, stages):
        future_positions = {item["position"] for item in stage["targets"][1:]}
        positions = [
            position
            for position, token_id in enumerate(stage["canvas"])
            if int(token_id) == int(mask_token_id) and position not in future_positions
        ]
        if not positions:
            continue
        index = torch.tensor(positions, device=student_logits.device, dtype=torch.long)
        student_rows = student[index].float().clone()
        teacher_rows = teacher[index].float().clone()
        student_rows[:, int(mask_token_id)] = -1e4
        teacher_rows[:, int(mask_token_id)] = -1e4
        losses.append(
            F.kl_div(
                F.log_softmax(student_rows, dim=-1),
                F.softmax(teacher_rows, dim=-1),
                reduction="none",
            ).sum(dim=-1)
        )
    if not losses:
        return student_logits.sum() * 0.0
    return torch.cat(losses).mean()


def transition_metrics(logits, stages, mask_token_id, block_length):
    future_correct = 0
    future_count = 0
    selected_count = 0
    target_count = 0
    joint_success = 0
    with torch.no_grad():
        for row_logits, stage in zip(logits, stages):
            rows = row_logits.float().clone()
            rows[:, int(mask_token_id)] = -torch.inf
            probabilities = torch.softmax(rows, dim=-1)
            confidence, prediction = probabilities.max(dim=-1)
            masked = torch.tensor(
                [[int(token_id) == int(mask_token_id) for token_id in stage["canvas"]]],
                device=logits.device,
                dtype=torch.bool,
            )
            eligible = current_block(masked, block_length)[0] & masked[0]
            budget = min(len(stage["targets"]), int(eligible.sum()))
            chosen = set(
                confidence.masked_fill(~eligible, -torch.inf)
                .topk(budget)
                .indices.cpu()
                .tolist()
            )
            all_tokens_correct = True
            all_selected = True
            for target_index, target in enumerate(stage["targets"]):
                position = int(target["position"])
                token_id = int(target["token_id"])
                is_correct = int(prediction[position]) == token_id
                is_selected = position in chosen
                target_count += 1
                selected_count += int(is_selected)
                all_tokens_correct &= is_correct
                all_selected &= is_selected
                if target_index:
                    future_count += 1
                    future_correct += int(is_correct)
            joint_success += int(all_tokens_correct and all_selected)
    return {
        "future_top1_fraction": future_correct / future_count if future_count else 0.0,
        "target_selection_fraction": selected_count / target_count if target_count else 0.0,
        "joint_success_fraction": joint_success / len(stages) if stages else 0.0,
    }


def read_records(path: Path, limit=None):
    if not path.exists():
        raise FileNotFoundError(path)
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            validate_rollout_record(record)
            if record.get("target_source") != TARGET_SOURCE:
                raise ValueError("unexpected teacher-rollout provenance")
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def autocast(device, enabled):
    if str(device).startswith("cuda") and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--adapter-path")
    parser.add_argument("--targets-file", required=True)
    parser.add_argument("--record-limit", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--lookahead", type=int, default=2)
    parser.add_argument("--states-per-example", type=int, default=4)
    parser.add_argument("--transition-loss-weight", type=float, default=1.0)
    parser.add_argument("--preserve-kl-weight", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,attn_out")
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if min(
        args.record_limit,
        args.max_steps,
        args.lookahead,
        args.states_per_example,
    ) <= 0:
        parser.error("record-limit, steps, lookahead, and states must be positive")
    if args.lookahead < 2:
        parser.error("lookahead must be at least 2")
    if min(args.transition_loss_weight, args.preserve_kl_weight) < 0:
        parser.error("loss weights must be nonnegative")
    return args


if __name__ == "__main__":
    main()
