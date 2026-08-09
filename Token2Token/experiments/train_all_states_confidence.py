#!/usr/bin/env python3
"""Train every cached anchor state to cross a confidence boundary."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from Token2Token.main.train import MODEL_ID, load_model, save_adapter
from Token2Token.experiments.train_all_unlocked import (
    all_unlocked_stages,
    non_target_kl_loss,
)
from Token2Token.main.train_anchor_transition import read_records
from Token2Token.main.train_lookahead_distillation import target_selection_loss
from Token2Token.main.train_threshold_unlock import (
    batched_completion_logits,
    validate_target_file,
)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2) + "\n", encoding="utf-8"
    )

    records = read_records(Path(args.targets_file))
    records = records[args.record_start :]
    if args.record_limit is not None:
        records = records[: args.record_limit]
    if not records:
        raise ValueError("record selection is empty")
    validate_target_file(Path(args.targets_file), records)

    tokenizer, model, mask_token_id, device = load_model(args)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    log_path = output / "train.jsonl"
    started = time.time()
    step = 0
    states_seen = 0
    skipped_records = 0

    for epoch in range(1, args.epochs + 1):
        order = list(range(len(records)))
        if args.shuffle_records:
            rng.shuffle(order)
        for local_index in order:
            record = records[local_index]
            stages = all_unlocked_stages(
                record, mask_token_id, args.completion_length
            )
            stages = select_high_unlock_stages(
                stages,
                min_unlocked_targets=args.min_unlocked_targets,
                min_selection_score=args.min_selection_score,
                catalyst_correct_only=args.catalyst_correct_only,
                max_states=args.max_states_per_record,
            )
            if not stages:
                skipped_records += 1
                continue
            if args.shuffle_states:
                rng.shuffle(stages)
            for chunk_index, stages_chunk in enumerate(
                chunked(stages, args.states_per_update)
            ):
                if args.max_steps is not None and step >= args.max_steps:
                    break
                step += 1
                states_seen += len(stages_chunk)
                prompt_ids = list(map(int, record["prompt_ids"]))
                canvases = [stage["canvas"] for stage in stages_chunk]

                was_training = model.training
                model.eval()
                with torch.no_grad(), model.disable_adapter(), autocast(
                    device, args.bf16
                ):
                    teacher_logits = batched_completion_logits(
                        model, prompt_ids, canvases, device
                    ).detach()
                if was_training:
                    model.train()

                optimizer.zero_grad(set_to_none=True)
                with autocast(device, args.bf16):
                    logits = batched_completion_logits(
                        model, prompt_ids, canvases, device
                    )
                    anchor_ce = target_kind_cross_entropy(
                        logits, stages_chunk, mask_token_id, "catalyst"
                    )
                    unlock_ce = target_kind_cross_entropy(
                        logits, stages_chunk, mask_token_id, "unlocked"
                    )
                    correction_ce = target_kind_cross_entropy(
                        logits, stages_chunk, mask_token_id, "correction"
                    )
                    confidence_loss = target_confidence_margin_loss(
                        logits,
                        stages_chunk,
                        mask_token_id,
                        args.target_confidence,
                        target_kind=args.confidence_margin_kind,
                    )
                    selection_loss = target_selection_loss(
                        logits,
                        select_target_kind_stages(
                            stages_chunk, args.selection_targets
                        ),
                        mask_token_id,
                        args.completion_length,
                        args.selection_margin,
                    )
                    preserve_loss = non_target_kl_loss(
                        logits, teacher_logits, stages_chunk, mask_token_id
                    )
                    loss = (
                        args.anchor_ce_weight * anchor_ce
                        + args.unlock_ce_weight * unlock_ce
                        + args.correction_ce_weight * correction_ce
                        + args.confidence_margin_weight * confidence_loss
                        + args.selection_loss_weight * selection_loss
                        + args.preserve_kl_weight * preserve_loss
                    )

                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()

                metrics = target_confidence_metrics(
                    logits.detach(),
                    stages_chunk,
                    mask_token_id,
                    args.target_confidence,
                )
                row = {
                    "step": step,
                    "epoch": epoch,
                    "example_id": record["example_id"],
                    "record_index": args.record_start + local_index,
                    "chunk_index": chunk_index,
                    "loss": float(loss.detach().cpu()),
                    "anchor_ce": float(anchor_ce.detach().cpu()),
                    "unlock_ce": float(unlock_ce.detach().cpu()),
                    "correction_ce": float(correction_ce.detach().cpu()),
                    "confidence_margin": float(confidence_loss.detach().cpu()),
                    "selection_loss": float(selection_loss.detach().cpu()),
                    "preserve_kl": float(preserve_loss.detach().cpu()),
                    "states": len(stages_chunk),
                    "states_seen": states_seen,
                    "targets": sum(len(stage["targets"]) for stage in stages_chunk),
                    "skipped_records": skipped_records,
                    **metrics,
                    "elapsed_seconds": time.time() - started,
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                print(
                    f"step={step} states={states_seen} loss={row['loss']:.4f} "
                    f"anchor_ce={row['anchor_ce']:.4f} "
                    f"unlock_ce={row['unlock_ce']:.4f} "
                    f"correction_ce={row['correction_ce']:.4f} "
                    f"margin={row['confidence_margin']:.4f} "
                    f"anchor_cross={row['anchor_cross_fraction']:.3f} "
                    f"unlock_cross={row['unlock_cross_fraction']:.3f}"
                )
                if args.save_every and step % args.save_every == 0:
                    save_adapter(model, tokenizer, output / f"checkpoint-{step:06d}")
            if args.max_steps is not None and step >= args.max_steps:
                break
        if args.max_steps is not None and step >= args.max_steps:
            break

    if step == 0:
        raise ValueError("no eligible anchor states were found")
    save_adapter(model, tokenizer, output / "adapter-final")


def chunked(items, size):
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [items[start : start + size] for start in range(0, len(items), size)]


def select_high_unlock_stages(
    stages,
    min_unlocked_targets=0,
    min_selection_score=None,
    catalyst_correct_only=False,
    max_states=None,
):
    """Keep the cached catalysts that unlocked the largest correct bursts."""
    if catalyst_correct_only and any(
        stage.get("catalyst_prediction_correct_before") is None
        for stage in stages
    ):
        raise ValueError(
            "catalyst-correct-only requires prediction_correct_before metadata"
        )
    selected = [
        stage
        for stage in stages
        if sum(target["kind"] == "unlocked" for target in stage["targets"])
        >= min_unlocked_targets
        and (
            min_selection_score is None
            or float(stage.get("selection_score", 0.0)) >= min_selection_score
        )
        and (
            not catalyst_correct_only
            or bool(stage.get("catalyst_prediction_correct_before", True))
        )
    ]
    selected.sort(
        key=lambda stage: sum(
            target["kind"] == "unlocked" for target in stage["targets"]
        ),
        reverse=True,
    )
    if max_states is not None:
        selected = selected[:max_states]
    return selected


def target_kind_cross_entropy(logits, stages, mask_token_id, kind):
    """Give catalyst and unlocked sets independent per-state weight."""
    losses = []
    for row_logits, stage in zip(logits, stages):
        targets = [target for target in stage["targets"] if target["kind"] == kind]
        if not targets:
            continue
        positions = torch.tensor(
            [target["position"] for target in targets],
            device=logits.device,
            dtype=torch.long,
        )
        labels = torch.tensor(
            [target["token_id"] for target in targets],
            device=logits.device,
            dtype=torch.long,
        )
        selected = row_logits[positions].float().clone()
        selected[:, int(mask_token_id)] = -torch.inf
        losses.append(F.cross_entropy(selected, labels))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def target_confidence_margin_loss(
    logits, stages, mask_token_id, target_confidence, target_kind=None
):
    """Hinge on gold-vs-rest log odds, equivalent to a probability floor."""
    if not 0.5 < target_confidence < 1.0:
        raise ValueError("target confidence must be in (0.5, 1)")
    required_margin = math.log(target_confidence / (1.0 - target_confidence))
    losses = []
    for row_logits, stage in zip(logits, stages):
        targets = [
            target
            for target in stage["targets"]
            if target_kind is None or target["kind"] == target_kind
        ]
        if not targets:
            continue
        positions = torch.tensor(
            [target["position"] for target in targets],
            device=logits.device,
            dtype=torch.long,
        )
        labels = torch.tensor(
            [target["token_id"] for target in targets],
            device=logits.device,
            dtype=torch.long,
        )
        selected = row_logits[positions].float().clone()
        selected[:, int(mask_token_id)] = -torch.inf
        gold = selected.gather(1, labels.unsqueeze(1)).squeeze(1)
        others = selected.clone()
        others.scatter_(1, labels.unsqueeze(1), -torch.inf)
        log_odds = gold - torch.logsumexp(others, dim=-1)
        losses.append(F.relu(required_margin - log_odds).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def select_target_kind_stages(stages, target_kind):
    """Restrict a ranking loss to catalysts or keep every target."""
    if target_kind == "all":
        return stages
    return [
        {
            **stage,
            "targets": [
                target
                for target in stage["targets"]
                if target["kind"] == target_kind
            ],
        }
        for stage in stages
    ]


def target_confidence_metrics(
    logits, stages, mask_token_id, target_confidence
):
    values = {"catalyst": [], "unlocked": [], "correction": []}
    with torch.no_grad():
        for row_logits, stage in zip(logits, stages):
            rows = row_logits.float().clone()
            rows[:, int(mask_token_id)] = -torch.inf
            probabilities = torch.softmax(rows, dim=-1)
            for target in stage["targets"]:
                probability = float(
                    probabilities[int(target["position"]), int(target["token_id"])]
                )
                values[target["kind"]].append(probability)

    result = {}
    for kind, probabilities in values.items():
        prefix = {
            "catalyst": "anchor",
            "unlocked": "unlock",
            "correction": "correction",
        }[kind]
        result[f"{prefix}_mean_probability"] = (
            sum(probabilities) / len(probabilities) if probabilities else 0.0
        )
        result[f"{prefix}_cross_fraction"] = (
            sum(value >= target_confidence for value in probabilities)
            / len(probabilities)
            if probabilities
            else 0.0
        )
    return result


def autocast(device, enabled):
    if str(device).startswith("cuda") and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--adapter-path")
    parser.add_argument("--targets-file", required=True)
    parser.add_argument("--record-start", type=int, default=0)
    parser.add_argument("--record-limit", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--completion-length", type=int, default=128)
    parser.add_argument("--states-per-update", type=int, default=8)
    parser.add_argument("--min-unlocked-targets", type=int, default=0)
    parser.add_argument("--min-selection-score", type=float)
    parser.add_argument("--catalyst-correct-only", action="store_true")
    parser.add_argument("--max-states-per-record", type=int)
    parser.add_argument("--target-confidence", type=float, default=0.70)
    parser.add_argument("--anchor-ce-weight", type=float, default=1.0)
    parser.add_argument("--unlock-ce-weight", type=float, default=1.0)
    parser.add_argument("--correction-ce-weight", type=float, default=0.0)
    parser.add_argument("--confidence-margin-weight", type=float, default=1.0)
    parser.add_argument(
        "--confidence-margin-kind",
        choices=("catalyst", "unlocked"),
        default="unlocked",
    )
    parser.add_argument("--selection-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--selection-targets", choices=("all", "catalyst"), default="catalyst"
    )
    parser.add_argument("--selection-margin", type=float, default=0.1)
    parser.add_argument("--preserve-kl-weight", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,attn_out")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-records", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-states", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    positive = (args.epochs, args.completion_length, args.states_per_update)
    if min(positive) <= 0:
        parser.error("epoch and state dimensions must be positive")
    if args.record_start < 0:
        parser.error("record-start must be nonnegative")
    if args.record_limit is not None and args.record_limit <= 0:
        parser.error("record-limit must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("max-steps must be positive")
    if args.min_unlocked_targets < 0:
        parser.error("min-unlocked-targets must be nonnegative")
    if args.max_states_per_record is not None and args.max_states_per_record <= 0:
        parser.error("max-states-per-record must be positive")
    if not 0.5 < args.target_confidence < 1:
        parser.error("target-confidence must be in (0.5, 1)")
    weights = (
        args.anchor_ce_weight,
        args.unlock_ce_weight,
        args.correction_ce_weight,
        args.confidence_margin_weight,
        args.selection_loss_weight,
        args.preserve_kl_weight,
    )
    if min(weights) < 0 or not any(weights):
        parser.error("loss weights must be nonnegative with one positive")
    return args


if __name__ == "__main__":
    main()
