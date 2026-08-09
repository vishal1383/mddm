#!/usr/bin/env python3
"""Train one forward to place a cached catalyst and all of its unlocked tokens."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from Token2Token.main.train import MODEL_ID, load_model, save_adapter
from Token2Token.main.train_anchor_transition import read_records
from Token2Token.main.train_lookahead_distillation import target_selection_loss
from Token2Token.main.train_threshold_unlock import (
    batched_completion_logits,
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

    step = 0
    record_cursor = 0
    skipped_records = 0
    consecutive_skips = 0
    while step < args.max_steps:
        record = records[record_cursor % len(records)]
        record_cursor += 1
        prompt_ids = list(map(int, record["prompt_ids"]))
        stages = all_unlocked_stages(
            record, mask_token_id, args.completion_length
        )
        if not stages:
            skipped_records += 1
            consecutive_skips += 1
            print(
                f"skip example_id={record.get('example_id')} reason=no-catalyst-"
                f"within-first-{args.completion_length}"
            )
            if consecutive_skips >= len(records):
                raise ValueError(
                    "no record has a catalyst inside the training canvas"
                )
            continue
        consecutive_skips = 0
        step += 1
        sampled = sample_stages(stages, args.states_per_example, random)
        canvases = [stage["canvas"] for stage in sampled]

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
            target_loss = target_set_cross_entropy(logits, sampled, mask_token_id)
            selection_loss = target_selection_loss(
                logits,
                sampled,
                mask_token_id,
                args.completion_length,
                args.selection_margin,
            )
            preserve_loss = non_target_kl_loss(
                logits, teacher_logits, sampled, mask_token_id
            )
            loss = (
                args.target_loss_weight * target_loss
                + args.selection_loss_weight * selection_loss
                + args.preserve_kl_weight * preserve_loss
            )

        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()

        metrics = target_metrics(
            logits.detach(), sampled, mask_token_id, args.commit_threshold
        )
        row = {
            "step": step,
            "example_id": record["example_id"],
            "records_seen": record_cursor,
            "skipped_records": skipped_records,
            "loss": float(loss.detach().cpu()),
            "target_loss": float(target_loss.detach().cpu()),
            "selection_loss": float(selection_loss.detach().cpu()),
            "preserve_kl": float(preserve_loss.detach().cpu()),
            "states": len(sampled),
            "targets": sum(len(stage["targets"]) for stage in sampled),
            "unlocked_targets": sum(
                len(stage["targets"]) - 1 for stage in sampled
            ),
            **metrics,
            "elapsed_seconds": time.time() - started,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        print(
            f"step={step} loss={row['loss']:.4f} "
            f"target={row['target_loss']:.4f} "
            f"selection={row['selection_loss']:.4f} "
            f"preserve={row['preserve_kl']:.4f} "
            f"states={row['states']} targets={row['targets']} "
            f"unlocked={row['unlocked_targets']} "
            f"anchor_top1={row['anchor_top1_fraction']:.3f} "
            f"unlock_commit={row['unlocked_commit_fraction']:.3f}"
        )
        if args.save_every and step % args.save_every == 0:
            save_adapter(model, tokenizer, output / f"checkpoint-{step:06d}")

    save_adapter(model, tokenizer, output / "adapter-final")


def all_unlocked_stages(record, mask_token_id, completion_length):
    """Use each pre-anchor canvas to target its catalyst plus all U_after tokens."""
    if completion_length <= 0:
        raise ValueError("completion_length must be positive")
    gold_ids = list(map(int, record["gold_ids"]))
    width = min(len(gold_ids), int(completion_length))
    canvas = [int(mask_token_id)] * int(completion_length)
    stages = []

    for expected_round, item in enumerate(record["rounds"], start=1):
        if int(item["round"]) != expected_round:
            raise ValueError("trajectory rounds must be consecutive")
        catalyst = item["catalyst"]
        catalyst_position = int(catalyst["gold_position"])
        catalyst_token_id = int(catalyst["token_id"])
        committed_unlocked = [
            {
                "position": int(target["gold_position"]),
                "token_id": int(target["token_id"]),
            }
            for target in item["unlocked"]
            if int(target["gold_position"]) < width
        ]
        training_source = item.get("newly_unlocked", item["unlocked"])
        training_unlocked = [
            {
                "position": int(target["gold_position"]),
                "token_id": int(target["token_id"]),
            }
            for target in training_source
            if int(target["gold_position"]) < width
        ]
        corrections = [
            {
                "position": int(target["gold_position"]),
                "token_id": int(target["gold_token_id"]),
            }
            for target in item.get("newly_wrong", [])
            if int(target["gold_position"]) < width
        ]

        if catalyst_position < width:
            validate_target(
                canvas,
                gold_ids,
                catalyst_position,
                catalyst_token_id,
                mask_token_id,
            )
            for target in committed_unlocked:
                validate_target(
                    canvas,
                    gold_ids,
                    target["position"],
                    target["token_id"],
                    mask_token_id,
                )
            for target in corrections:
                validate_target(
                    canvas,
                    gold_ids,
                    target["position"],
                    target["token_id"],
                    mask_token_id,
                )
            stages.append(
                {
                    "round": expected_round,
                    "selection_score": float(item.get("selection_score", 0.0)),
                    "catalyst_prediction_correct_before": catalyst.get(
                        "prediction_correct_before"
                    ),
                    "canvas": list(canvas),
                    "targets": [
                        {
                            "position": catalyst_position,
                            "token_id": catalyst_token_id,
                            "kind": "catalyst",
                        },
                        *[
                            {**target, "kind": "unlocked"}
                            for target in training_unlocked
                        ],
                        *[
                            {**target, "kind": "correction"}
                            for target in corrections
                        ],
                    ],
                }
            )

        if catalyst_position < width:
            canvas[catalyst_position] = catalyst_token_id
        for target in committed_unlocked:
            canvas[target["position"]] = target["token_id"]

    return stages


def validate_target(canvas, gold_ids, position, token_id, mask_token_id):
    if position < 0 or position >= len(canvas) or position >= len(gold_ids):
        raise ValueError(f"target position {position} is outside the canvas")
    if int(canvas[position]) != int(mask_token_id):
        raise ValueError(f"target position {position} was already filled")
    if int(gold_ids[position]) != int(token_id):
        raise ValueError("cached target does not match the gold completion")


def sample_stages(stages, count, rng):
    """Always retain the earliest state and the largest available unlock burst."""
    if count <= 0:
        raise ValueError("stage count must be positive")
    if count == 1:
        return [stages[0]]
    if count >= len(stages):
        return list(stages)

    selected = {0}
    largest = max(
        range(len(stages)),
        key=lambda index: (len(stages[index]["targets"]), -index),
    )
    selected.add(largest)
    remaining = [index for index in range(len(stages)) if index not in selected]
    budget = count - len(selected)
    if budget > 0:
        selected.update(rng.sample(remaining, min(budget, len(remaining))))
    return [stages[index] for index in sorted(selected)]


def target_set_cross_entropy(logits, stages, mask_token_id):
    """Average within each cache round, then average rounds equally."""
    losses = []
    for row_logits, stage in zip(logits, stages):
        positions = torch.tensor(
            [target["position"] for target in stage["targets"]],
            device=logits.device,
            dtype=torch.long,
        )
        labels = torch.tensor(
            [target["token_id"] for target in stage["targets"]],
            device=logits.device,
            dtype=torch.long,
        )
        selected = row_logits[positions].float().clone()
        selected[:, int(mask_token_id)] = -torch.inf
        losses.append(F.cross_entropy(selected, labels))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def non_target_kl_loss(student_logits, teacher_logits, stages, mask_token_id):
    """Preserve frozen-base predictions outside the complete target set."""
    losses = []
    for student, teacher, stage in zip(student_logits, teacher_logits, stages):
        targets = {int(target["position"]) for target in stage["targets"]}
        positions = [
            position
            for position, token_id in enumerate(stage["canvas"])
            if int(token_id) == int(mask_token_id) and position not in targets
        ]
        if not positions:
            continue
        index = torch.tensor(
            positions, device=student_logits.device, dtype=torch.long
        )
        student_rows = student[index].float().clone()
        teacher_rows = teacher[index].float().clone()
        student_rows[:, int(mask_token_id)] = -1e4
        teacher_rows[:, int(mask_token_id)] = -1e4
        losses.append(
            F.kl_div(
                F.log_softmax(student_rows, dim=-1),
                F.softmax(teacher_rows, dim=-1),
                reduction="none",
            )
            .sum(dim=-1)
            .mean()
        )
    if not losses:
        return student_logits.sum() * 0.0
    return torch.stack(losses).mean()


def target_metrics(logits, stages, mask_token_id, commit_threshold):
    anchors = 0
    anchor_top1 = 0
    unlocked = 0
    unlocked_top1 = 0
    unlocked_commit = 0
    joint = 0

    with torch.no_grad():
        for row_logits, stage in zip(logits, stages):
            rows = row_logits.float().clone()
            rows[:, int(mask_token_id)] = -torch.inf
            probabilities = torch.softmax(rows, dim=-1)
            confidence, prediction = probabilities.max(dim=-1)
            stage_joint = True
            for target in stage["targets"]:
                position = int(target["position"])
                token_id = int(target["token_id"])
                correct = int(prediction[position]) == token_id
                if target["kind"] == "catalyst":
                    anchors += 1
                    anchor_top1 += int(correct)
                    stage_joint &= correct
                else:
                    unlocked += 1
                    unlocked_top1 += int(correct)
                    committable = correct and float(confidence[position]) >= float(
                        commit_threshold
                    )
                    unlocked_commit += int(committable)
                    stage_joint &= committable
            joint += int(stage_joint)

    return {
        "anchor_top1_fraction": anchor_top1 / anchors if anchors else 0.0,
        "unlocked_top1_fraction": unlocked_top1 / unlocked if unlocked else 0.0,
        "unlocked_commit_fraction": (
            unlocked_commit / unlocked if unlocked else 0.0
        ),
        "joint_commit_fraction": joint / len(stages) if stages else 0.0,
    }


def autocast(device, enabled):
    if str(device).startswith("cuda") and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--adapter-path")
    parser.add_argument("--targets-file", required=True)
    parser.add_argument("--record-limit", type=int, default=7_473)
    parser.add_argument("--max-steps", type=int, default=7_473)
    parser.add_argument("--completion-length", type=int, default=128)
    parser.add_argument("--states-per-example", type=int, default=3)
    parser.add_argument("--commit-threshold", type=float, default=0.95)
    parser.add_argument("--target-loss-weight", type=float, default=1.0)
    parser.add_argument("--selection-loss-weight", type=float, default=1.0)
    parser.add_argument("--selection-margin", type=float, default=0.1)
    parser.add_argument("--preserve-kl-weight", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,attn_out")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    positive = (
        args.record_limit,
        args.max_steps,
        args.completion_length,
        args.states_per_example,
    )
    if min(positive) <= 0:
        parser.error("record and canvas counts must be positive")
    if not 0 < args.commit_threshold < 1:
        parser.error("commit-threshold must be in (0, 1)")
    if min(
        args.target_loss_weight,
        args.selection_loss_weight,
        args.selection_margin,
        args.preserve_kl_weight,
    ) < 0:
        parser.error("loss weights and margin must be nonnegative")
    return args


if __name__ == "__main__":
    main()
