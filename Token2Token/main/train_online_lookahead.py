#!/usr/bin/env python3
"""Distil frozen-base token-to-token transitions on cached canvases."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time

import torch

from Token2Token.main.decode_policy import joint_topk_catalyst_mask
from Token2Token.main.eval_threshold_gsm8k import (
    allowed_prediction_mask,
    current_block,
    threshold_selection_mask,
)
from Token2Token.main.train import MODEL_ID, load_model, save_adapter
from Token2Token.main.train_anchor_transition import anchor_transitions, read_records
from Token2Token.main.train_lookahead_distillation import (
    decoder_kl_loss,
    future_token_loss,
    target_selection_loss,
    transition_metrics,
)
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
        raise ValueError(f"no cached canvases in {args.targets_file}")
    validate_target_file(Path(args.targets_file), records)

    tokenizer, model, mask_token_id, device = load_model(args)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    log_path = output / "train.jsonl"
    started = time.time()

    for step in range(1, args.max_steps + 1):
        record = records[(step - 1) % len(records)]
        prompt_ids = list(map(int, record["prompt_ids"]))
        canvases = sample_cached_canvases(
            record,
            mask_token_id,
            args.completion_length,
            args.states_per_example,
            args.max_unlock_tokens,
            random,
        )

        was_training = model.training
        model.eval()
        with torch.no_grad(), model.disable_adapter(), autocast(device, args.bf16):
            if args.teacher_policy == "threshold-catalyst":
                stages, teacher_logits = threshold_teacher_lookahead_stages(
                    model,
                    tokenizer,
                    prompt_ids,
                    canvases,
                    mask_token_id,
                    args.lookahead,
                    args.confidence_threshold,
                    args.numeric_threshold,
                    device,
                )
            else:
                stages, teacher_logits = teacher_lookahead_stages(
                    model,
                    prompt_ids,
                    canvases,
                    mask_token_id,
                    args.block_length,
                    args.lookahead,
                    device,
                )
        if was_training:
            model.train()

        optimizer.zero_grad(set_to_none=True)
        with autocast(device, args.bf16):
            logits = batched_completion_logits(model, prompt_ids, canvases, device)
            transition_loss = future_token_loss(logits, stages, mask_token_id)
            selection_loss = target_selection_loss(
                logits,
                stages,
                mask_token_id,
                args.block_length,
                args.selection_margin,
            )
            preserve_loss = decoder_kl_loss(
                logits, teacher_logits, stages, mask_token_id
            )
            loss = (
                args.transition_loss_weight * transition_loss
                + args.selection_loss_weight * selection_loss
                + args.preserve_kl_weight * preserve_loss
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()

        metrics = transition_metrics(
            logits.detach(), stages, mask_token_id, args.block_length
        )
        row = {
            "step": step,
            "example_id": record["example_id"],
            "loss": float(loss.detach().cpu()),
            "transition_loss": float(transition_loss.detach().cpu()),
            "selection_loss": float(selection_loss.detach().cpu()),
            "preserve_kl": float(preserve_loss.detach().cpu()),
            "states": len(stages),
            "teacher_targets": [stage["targets"] for stage in stages],
            **metrics,
            "elapsed_seconds": time.time() - started,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        print(
            f"step={step} loss={row['loss']:.4f} "
            f"transition={row['transition_loss']:.4f} "
            f"selection_loss={row['selection_loss']:.4f} "
            f"preserve={row['preserve_kl']:.4f} "
            f"future_top1={row['future_top1_fraction']:.3f} "
            f"selected={row['target_selection_fraction']:.3f} "
            f"joint={row['joint_success_fraction']:.3f}"
        )
        if args.save_every and step % args.save_every == 0:
            save_adapter(model, tokenizer, output / f"checkpoint-{step:06d}")

    save_adapter(model, tokenizer, output / "adapter-final")


def sample_cached_canvases(
    record,
    mask_token_id,
    completion_length,
    count,
    max_unlock_tokens,
    rng,
):
    """Always include the blank inference state, then sample cached states."""
    blank = [int(mask_token_id)] * completion_length
    pool = []
    for transition in anchor_transitions(record, mask_token_id, max_unlock_tokens):
        pool.append(normalize_canvas(transition["anchor"]["canvas"], completion_length, mask_token_id))
        if transition["post_anchor"] is not None:
            pool.append(
                normalize_canvas(
                    transition["post_anchor"]["canvas"],
                    completion_length,
                    mask_token_id,
                )
            )
    unique = []
    seen = {tuple(blank)}
    for canvas in pool:
        key = tuple(canvas)
        if key not in seen and int(mask_token_id) in canvas:
            seen.add(key)
            unique.append(canvas)
    sampled = rng.sample(unique, min(max(0, count - 1), len(unique)))
    return [blank, *sampled]


def normalize_canvas(canvas, completion_length, mask_token_id):
    normalized = list(map(int, canvas[:completion_length]))
    normalized.extend(
        [int(mask_token_id)] * max(0, completion_length - len(normalized))
    )
    return normalized


def teacher_lookahead_stages(
    model,
    prompt_ids,
    canvases,
    mask_token_id,
    block_length,
    lookahead,
    device,
):
    """Return the actions block-k1 would take over the next few forwards."""
    working = [list(map(int, canvas)) for canvas in canvases]
    targets = [[] for _ in working]
    initial_logits = None
    for _ in range(lookahead):
        logits = batched_completion_logits(model, prompt_ids, working, device)
        if initial_logits is None:
            initial_logits = logits.detach()
        actions = teacher_actions(logits, working, mask_token_id, block_length)
        for row_index, action in enumerate(actions):
            targets[row_index].append(action)
            working[row_index][action["position"]] = action["token_id"]
    stages = [
        {"canvas": list(canvas), "targets": row_targets}
        for canvas, row_targets in zip(canvases, targets)
    ]
    return stages, initial_logits


def threshold_teacher_lookahead_stages(
    model,
    tokenizer,
    prompt_ids,
    canvases,
    mask_token_id,
    lookahead,
    confidence_threshold,
    numeric_threshold,
    device,
):
    """Distil consecutive actions of the deployed top-one catalyst decoder."""
    working = [list(map(int, canvas)) for canvas in canvases]
    targets = [[] for _ in working]
    initial_logits = None
    allowed_cache = {}
    numeric_cache = {}
    for _ in range(lookahead):
        logits = batched_completion_logits(model, prompt_ids, working, device)
        if initial_logits is None:
            initial_logits = logits.detach()
        rows = logits.float().clone()
        rows[:, :, int(mask_token_id)] = -torch.inf
        probabilities = torch.softmax(rows, dim=-1)
        confidence, prediction = probabilities.max(dim=-1)
        masked = torch.tensor(
            [
                [int(token_id) == int(mask_token_id) for token_id in canvas]
                for canvas in working
            ],
            device=logits.device,
            dtype=torch.bool,
        )
        active = masked.any(dim=1)
        allowed = allowed_prediction_mask(
            prediction,
            masked,
            tokenizer,
            allowed_cache,
            catalyst_filter="text",
            min_length=0,
        )
        allowed &= confidence.lt(confidence_threshold)
        has_catalyst = allowed.any(dim=1) & active
        chosen = joint_topk_catalyst_mask(confidence, allowed, 1)
        chosen &= has_catalyst.unsqueeze(1)
        cleanup = active & ~has_catalyst
        leftmost = torch.zeros_like(masked)
        leftmost.scatter_(1, masked.long().argmax(dim=1, keepdim=True), True)
        chosen |= leftmost & cleanup.unsqueeze(1)

        burst_eligible = masked & ~chosen
        burst = threshold_selection_mask(
            burst_eligible,
            confidence,
            prediction,
            confidence_threshold,
            numeric_threshold,
            tokenizer,
            numeric_cache,
        )
        for row_index in range(len(working)):
            if not bool(active[row_index]):
                continue
            position = int(chosen[row_index].long().argmax())
            token_id = int(prediction[row_index, position])
            targets[row_index].append(
                {"position": position, "token_id": token_id}
            )
            working[row_index][position] = token_id
            for burst_position in burst[row_index].nonzero(as_tuple=False).flatten():
                index = int(burst_position)
                working[row_index][index] = int(prediction[row_index, index])

    stages = [
        {"canvas": list(canvas), "targets": row_targets}
        for canvas, row_targets in zip(canvases, targets)
    ]
    return stages, initial_logits


def teacher_actions(logits, canvases, mask_token_id, block_length):
    rows = logits.float().clone()
    rows[:, :, int(mask_token_id)] = -torch.inf
    probabilities = torch.softmax(rows, dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    masked = torch.tensor(
        [
            [int(token_id) == int(mask_token_id) for token_id in canvas]
            for canvas in canvases
        ],
        device=logits.device,
        dtype=torch.bool,
    )
    eligible = masked & current_block(masked, block_length)
    positions = confidence.masked_fill(~eligible, -torch.inf).argmax(dim=1)
    batch = torch.arange(len(canvases), device=logits.device)
    token_ids = prediction[batch, positions]
    return [
        {"position": int(position), "token_id": int(token_id)}
        for position, token_id in zip(positions, token_ids)
    ]


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
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--completion-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--lookahead", type=int, default=2)
    parser.add_argument(
        "--teacher-policy",
        choices=("block-k1", "threshold-catalyst"),
        default="block-k1",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--numeric-threshold", type=float, default=0.99)
    parser.add_argument("--states-per-example", type=int, default=4)
    parser.add_argument("--max-unlock-tokens", type=int, default=2)
    parser.add_argument("--transition-loss-weight", type=float, default=1.0)
    parser.add_argument("--selection-loss-weight", type=float, default=0.0)
    parser.add_argument("--selection-margin", type=float, default=0.1)
    parser.add_argument("--preserve-kl-weight", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,attn_out")
    parser.add_argument("--save-every", type=int, default=125)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    positive = (
        args.record_limit,
        args.max_steps,
        args.completion_length,
        args.block_length,
        args.lookahead,
        args.states_per_example,
        args.max_unlock_tokens,
    )
    if min(positive) <= 0:
        parser.error("counts and lengths must be positive")
    if args.lookahead < 2:
        parser.error("lookahead must be at least 2")
    if not 0 < args.confidence_threshold < 1:
        parser.error("confidence-threshold must be in (0, 1)")
    if not args.confidence_threshold <= args.numeric_threshold < 1:
        parser.error(
            "numeric-threshold must be in [confidence-threshold, 1)"
        )
    if min(
        args.transition_loss_weight,
        args.selection_loss_weight,
        args.selection_margin,
        args.preserve_kl_weight,
    ) < 0:
        parser.error("loss weights and selection margin must be nonnegative")
    return args


if __name__ == "__main__":
    main()
