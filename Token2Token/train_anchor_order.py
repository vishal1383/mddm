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

from Token2Token.train import MODEL_ID, load_model, save_adapter


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
        raise ValueError(f"no anchor target records in {args.targets_file}")
    validate_target_provenance(Path(args.targets_file), records)

    tokenizer, model, mask_token_id, device = load_model(args)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    log_path = output / "train.jsonl"
    step = 0
    started = time.time()

    while step < args.max_steps:
        record = records[step % len(records)]
        prompt_ids = list(map(int, record["prompt_ids"]))
        gold_ids = list(map(int, record["gold_ids"]))
        targets = record["targets"][: args.anchors]
        validate_targets(gold_ids, targets)
        canvases, positions, token_ids = ordered_anchor_canvases(
            len(gold_ids), mask_token_id, targets
        )
        final_canvas = completed_anchor_canvas(canvases, positions, token_ids)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, args.bf16):
            anchor_loss, sequence_loss = batched_anchor_completion_losses(
                model,
                prompt_ids,
                canvases,
                final_canvas,
                positions,
                token_ids,
                gold_ids,
                mask_token_id,
                device,
            )
            loss = (
                args.anchor_loss_weight * anchor_loss
                + args.sequence_loss_weight * sequence_loss
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
            "anchor_loss": float(anchor_loss.detach().cpu()),
            "sequence_loss": float(sequence_loss.detach().cpu()),
            "anchor_loss_weight": args.anchor_loss_weight,
            "sequence_loss_weight": args.sequence_loss_weight,
            "anchors": len(targets),
            "anchor_tokens": [target["token"] for target in targets],
            "anchor_positions": [int(target["gold_position"]) for target in targets],
            "ig_scores": [float(target["ig_score"]) for target in targets],
            "elapsed_seconds": time.time() - started,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        print(
            f"step={global_step} loss={row['loss']:.4f} "
            f"anchor={row['anchor_loss']:.4f} sequence={row['sequence_loss']:.4f} "
            f"anchors={row['anchors']} "
            f"tokens={row['anchor_tokens']!r}"
        )
        if args.save_every and step % args.save_every == 0:
            save_adapter(model, tokenizer, output / f"checkpoint-{global_step:06d}")

    save_adapter(model, tokenizer, output / "adapter-final")


def anchor_target_loss(logits, position: int, token_id: int):
    target = torch.tensor([token_id], device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits[position].float().unsqueeze(0), target)


def ordered_anchor_canvases(length: int, mask_token_id: int, targets: list[dict]):
    canvas = [int(mask_token_id)] * length
    canvases = []
    positions = []
    token_ids = []
    for target in targets:
        position = int(target["gold_position"])
        token_id = int(target["token_id"])
        canvases.append(list(canvas))
        positions.append(position)
        token_ids.append(token_id)
        canvas[position] = token_id
    return canvases, positions, token_ids


def completed_anchor_canvas(canvases, positions, token_ids):
    canvas = list(canvases[-1])
    canvas[positions[-1]] = token_ids[-1]
    return canvas


def batched_anchor_completion_losses(
    model,
    prompt_ids,
    canvases,
    final_canvas,
    positions,
    token_ids,
    gold_ids,
    mask_token_id,
    device,
):
    model_canvases = canvases + [final_canvas]
    input_ids = torch.tensor(
        [prompt_ids + canvas for canvas in model_canvases],
        device=device,
        dtype=torch.long,
    )
    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits[
        :, len(prompt_ids) : len(prompt_ids) + len(canvases[0])
    ]
    return anchor_completion_losses(
        logits, final_canvas, positions, token_ids, gold_ids, mask_token_id
    )


def anchor_completion_losses(
    logits, final_canvas, positions, token_ids, gold_ids, mask_token_id
):
    device = logits.device
    anchor_count = len(positions)
    row_indices = torch.arange(anchor_count, device=device)
    position_indices = torch.tensor(positions, device=device, dtype=torch.long)
    labels = torch.tensor(token_ids, device=device, dtype=torch.long)
    selected_logits = logits[row_indices, position_indices].float()
    anchor_loss = F.cross_entropy(selected_logits, labels)

    canvas_tensor = torch.tensor(final_canvas, device=device, dtype=torch.long)
    gold = torch.tensor(gold_ids, device=device, dtype=torch.long)
    masked = canvas_tensor.eq(mask_token_id)
    sequence_loss = F.cross_entropy(
        logits[anchor_count, masked].float(), gold[masked]
    )
    return anchor_loss, sequence_loss


def validate_targets(gold_ids: list[int], targets: list[dict]) -> None:
    if not targets:
        raise ValueError("each record must contain at least one anchor target")
    for expected_rank, target in enumerate(targets, start=1):
        position = int(target["gold_position"])
        token_id = int(target["token_id"])
        if int(target["rank"]) != expected_rank:
            raise ValueError("anchor targets must be stored in consecutive IG order")
        if position < 0 or position >= len(gold_ids):
            raise ValueError(f"anchor position {position} is outside the completion")
        if gold_ids[position] != token_id:
            raise ValueError("anchor token does not match the gold completion")


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_target_provenance(path: Path, records: list[dict]) -> None:
    metadata_path = path.with_suffix(".config.json")
    if not metadata_path.exists():
        raise ValueError(f"missing target metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = "frozen_base_greedy_ig"
    if metadata.get("target_source") != expected:
        raise ValueError(
            f"anchor targets must come from {expected}; got "
            f"{metadata.get('target_source')!r}"
        )
    invalid = {
        record.get("target_source")
        for record in records
        if record.get("target_source") not in (None, expected)
    }
    if invalid:
        raise ValueError(f"non-frozen target rows found: {sorted(invalid)!r}")


def autocast(device: str, enabled: bool):
    if str(device).startswith("cuda") and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plain sequential IG-anchor CE trainer"
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--adapter-path")
    parser.add_argument("--targets-file", required=True)
    parser.add_argument("--max-steps", type=int, default=7_473)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--anchors", type=int, default=5)
    parser.add_argument("--anchor-loss-weight", type=float, default=1.0)
    parser.add_argument("--sequence-loss-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,attn_out")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", default="outputs/token2token/anchor_order_lora_gsm8k"
    )
    args = parser.parse_args()
    if args.max_steps <= 0 or args.anchors <= 0:
        parser.error("max-steps and anchors must be positive")
    if args.step_offset < 0:
        parser.error("step-offset must be non-negative")
    if args.anchor_loss_weight < 0 or args.sequence_loss_weight < 0:
        parser.error("loss weights must be non-negative")
    if args.anchor_loss_weight == 0 and args.sequence_loss_weight == 0:
        parser.error("at least one loss weight must be positive")
    return args


if __name__ == "__main__":
    main()
