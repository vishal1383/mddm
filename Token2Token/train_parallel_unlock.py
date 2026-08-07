#!/usr/bin/env python3
"""Train for parallel decodability instead of for the anchor transition.

The decoder commits a masked position once its confidence crosses the
threshold, so tokens/forward is set by how many positions clear the threshold
at once. This trainer reads the frozen base model's own prediction at every
masked position of a realistic canvas and only moves the adapter where moving
changes a commit decision:

  promote  base already ranks gold first but stays under the threshold.
           Raising it turns a non-commit into a correct commit, which buys
           throughput at no accuracy risk.
  repair   base is over the threshold on a non-gold token, so the decoder
           would irreversibly commit a mistake.
  preserve everything else, held at the base distribution with KL.

Canvases are replayed from the cached threshold-gain trajectory so they match
the partially-filled states the decoder actually visits.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from Token2Token.train import MODEL_ID, load_model, save_adapter
from Token2Token.train_anchor_transition import anchor_transitions, read_records
from Token2Token.train_threshold_unlock import (
    batched_completion_logits,
    random_denoising_stage,
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
    numeric_cache = {}
    started = time.time()

    for step in range(1, args.max_steps + 1):
        record = records[(step - 1) % len(records)]
        prompt_ids = list(map(int, record["prompt_ids"]))
        gold_ids = list(map(int, record["gold_ids"]))
        canvases = sample_canvases(
            record, gold_ids, mask_token_id, args, random
        )

        was_training = model.training
        model.eval()
        with torch.no_grad(), model.disable_adapter(), autocast(device, args.bf16):
            teacher_logits = batched_completion_logits(
                model, prompt_ids, canvases, device
            ).detach()
        if was_training:
            model.train()

        buckets = [
            bucket_positions(
                row_logits,
                canvas,
                gold_ids,
                mask_token_id,
                args.commit_threshold,
                args.promote_min_confidence,
                args.repair_max_gold_rank,
                tokenizer if args.protect_numeric_positions else None,
                numeric_cache,
            )
            for row_logits, canvas in zip(teacher_logits, canvases)
        ]

        optimizer.zero_grad(set_to_none=True)
        with autocast(device, args.bf16):
            logits = batched_completion_logits(model, prompt_ids, canvases, device)
            zero = logits.sum() * 0.0
            promote_loss = promote_objective(
                logits,
                [item["promote"] for item in buckets],
                gold_ids,
                zero,
                args.promote_loss,
                args.promote_target_confidence,
            )
            repair_loss = gold_cross_entropy(
                logits, [item["repair"] for item in buckets], gold_ids, zero
            )
            preserve_loss = preserve_kl(
                logits, teacher_logits, [item["preserve"] for item in buckets], zero
            )
            loss = (
                args.promote_loss_weight * promote_loss
                + args.repair_loss_weight * repair_loss
                + args.preserve_kl_weight * preserve_loss
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
            "promote_loss": float(promote_loss.detach().cpu()),
            "repair_loss": float(repair_loss.detach().cpu()),
            "preserve_loss": float(preserve_loss.detach().cpu()),
            "canvases": len(canvases),
            "promote_positions": sum(len(item["promote"]) for item in buckets),
            "repair_positions": sum(len(item["repair"]) for item in buckets),
            "preserve_positions": sum(len(item["preserve"]) for item in buckets),
            "elapsed_seconds": time.time() - started,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        print(
            f"step={step} loss={row['loss']:.4f} "
            f"promote={row['promote_loss']:.4f}({row['promote_positions']}) "
            f"repair={row['repair_loss']:.4f}({row['repair_positions']}) "
            f"preserve={row['preserve_loss']:.4f}({row['preserve_positions']})"
        )
        if args.save_every and step % args.save_every == 0:
            save_adapter(model, tokenizer, output / f"checkpoint-{step:06d}")

    save_adapter(model, tokenizer, output / "adapter-final")


def sample_canvases(record, gold_ids, mask_token_id, args, rng):
    """Canvases the decoder would plausibly visit, plus one random-mask canvas."""
    transitions = anchor_transitions(record, mask_token_id, args.max_unlock_tokens)
    pool = []
    for item in transitions:
        if args.canvas_source != "post-anchor":
            pool.append(item["anchor"]["canvas"])
        if item["post_anchor"] is not None:
            pool.append(item["post_anchor"]["canvas"])
    count = min(args.canvases_per_example, len(pool))
    canvases = rng.sample(pool, count) if count else []
    if args.random_denoising_canvas:
        stage, _ = random_denoising_stage(
            gold_ids, mask_token_id, args.min_mask_ratio, args.max_mask_ratio
        )
        canvases.append(stage["canvas"])
    if not canvases:
        canvases.append([int(mask_token_id)] * len(gold_ids))
    return canvases


def bucket_positions(
    teacher_logits,
    canvas,
    gold_ids,
    mask_token_id,
    commit_threshold,
    promote_min_confidence,
    repair_max_gold_rank=0,
    tokenizer=None,
    numeric_cache=None,
):
    masked = [
        position
        for position, token_id in enumerate(canvas)
        if int(token_id) == int(mask_token_id)
    ]
    if not masked:
        return {"promote": [], "repair": [], "preserve": []}
    index = torch.tensor(masked, device=teacher_logits.device, dtype=torch.long)
    rows = teacher_logits[index].float().clone()
    rows[:, int(mask_token_id)] = -torch.inf
    confidence, top1 = torch.softmax(rows, dim=-1).max(dim=-1)
    gold = torch.tensor(
        [int(gold_ids[position]) for position in masked],
        device=teacher_logits.device,
        dtype=torch.long,
    )
    ranks_gold = top1.eq(gold)
    committable = confidence.ge(commit_threshold)
    promote = ranks_gold & ~committable & confidence.ge(promote_min_confidence)
    repair = ~ranks_gold & committable
    if repair_max_gold_rank:
        # A confident non-gold prediction is usually the model phrasing the same
        # reasoning differently, not an error. Only repair where the model still
        # holds gold as a live alternative; otherwise cross-entropy here teaches
        # it to abandon its own coherent completion.
        gold_rank = rows.gt(rows.gather(1, gold.unsqueeze(1))).sum(dim=1)
        repair = repair & gold_rank.lt(repair_max_gold_rank)
    if tokenizer is not None:
        # V3 kept base's prose almost verbatim but slipped digits: it turned
        # "3 * $22.50 = $67.5" into "= $67". Numeric tokens carry the answer
        # and have no redundancy, so pin them to the base distribution instead
        # of letting the adapter move them.
        protected = numeric_tokens(
            top1.tolist() + gold.tolist(), tokenizer, numeric_cache
        )
        guard = torch.tensor(
            protected, device=teacher_logits.device, dtype=torch.bool
        ).view(2, -1).any(dim=0)
        promote = promote & ~guard
        repair = repair & ~guard
    preserve = ~(promote | repair)
    return {
        "promote": select(masked, promote),
        "repair": select(masked, repair),
        "preserve": select(masked, preserve),
    }


def numeric_tokens(token_ids, tokenizer, cache):
    if cache is None:
        cache = {}
    flags = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id not in cache:
            cache[token_id] = any(
                character.isdigit() for character in tokenizer.decode([token_id])
            )
        flags.append(cache[token_id])
    return flags


def select(positions, mask):
    keep = mask.detach().cpu().tolist()
    return [position for position, flag in zip(positions, keep) if flag]


def promote_objective(logits, rows, gold_ids, zero, objective, target_confidence):
    """Raise gold probability only until the position becomes committable.

    Cross-entropy keeps pushing toward probability 1 long after a position has
    crossed the commit threshold, which buys no extra tokens per forward and
    is how earlier runs inflated confidence into wrong commits. The hinge stops
    contributing as soon as the position would commit.
    """
    if objective == "ce":
        return gold_cross_entropy(logits, rows, gold_ids, zero)
    losses = []
    labels = torch.tensor(gold_ids, device=logits.device, dtype=torch.long)
    floor = torch.log(
        torch.tensor(target_confidence, device=logits.device, dtype=torch.float32)
    )
    for row_logits, positions in zip(logits, rows):
        if not positions:
            continue
        index = torch.tensor(positions, device=logits.device, dtype=torch.long)
        gold_log_probability = (
            F.log_softmax(row_logits[index].float(), dim=-1)
            .gather(1, labels[index].unsqueeze(1))
            .squeeze(1)
        )
        losses.append(torch.relu(floor - gold_log_probability))
    if not losses:
        return zero
    return torch.cat(losses).mean()


def gold_cross_entropy(logits, rows, gold_ids, zero):
    losses = []
    labels = torch.tensor(gold_ids, device=logits.device, dtype=torch.long)
    for row_logits, positions in zip(logits, rows):
        if not positions:
            continue
        index = torch.tensor(positions, device=logits.device, dtype=torch.long)
        losses.append(
            F.cross_entropy(
                row_logits[index].float(), labels[index], reduction="none"
            )
        )
    if not losses:
        return zero
    return torch.cat(losses).mean()


def preserve_kl(student_logits, teacher_logits, rows, zero):
    losses = []
    for student_row, teacher_row, positions in zip(
        student_logits, teacher_logits, rows
    ):
        if not positions:
            continue
        index = torch.tensor(positions, device=student_logits.device, dtype=torch.long)
        losses.append(
            F.kl_div(
                F.log_softmax(student_row[index].float(), dim=-1),
                F.softmax(teacher_row[index].float(), dim=-1),
                reduction="none",
            ).sum(dim=-1)
        )
    if not losses:
        return zero
    return torch.cat(losses).mean()


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
    parser.add_argument("--canvases-per-example", type=int, default=4)
    parser.add_argument(
        "--canvas-source", choices=("trajectory", "post-anchor"), default="trajectory"
    )
    parser.add_argument("--max-unlock-tokens", type=int, default=2)
    parser.add_argument("--commit-threshold", type=float, default=0.95)
    parser.add_argument("--promote-min-confidence", type=float, default=0.5)
    parser.add_argument("--promote-loss", choices=("hinge", "ce"), default="hinge")
    parser.add_argument("--promote-target-confidence", type=float, default=0.97)
    parser.add_argument("--promote-loss-weight", type=float, default=1.0)
    parser.add_argument("--repair-loss-weight", type=float, default=0.0)
    parser.add_argument("--repair-max-gold-rank", type=int, default=5)
    parser.add_argument(
        "--protect-numeric-positions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--preserve-kl-weight", type=float, default=1.0)
    parser.add_argument(
        "--random-denoising-canvas",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    if args.max_steps <= 0:
        parser.error("max-steps must be positive")
    if args.canvases_per_example <= 0:
        parser.error("canvases-per-example must be positive")
    if not 0 < args.commit_threshold < 1:
        parser.error("commit-threshold must be in (0, 1)")
    if not 0 <= args.promote_min_confidence < args.commit_threshold:
        parser.error("promote-min-confidence must be in [0, commit-threshold)")
    if not 0 < args.promote_target_confidence < 1:
        parser.error("promote-target-confidence must be in (0, 1)")
    if min(
        args.promote_loss_weight, args.repair_loss_weight, args.preserve_kl_weight
    ) < 0:
        parser.error("loss weights must be nonnegative")
    return args


if __name__ == "__main__":
    main()
