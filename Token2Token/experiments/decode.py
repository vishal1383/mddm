#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from Token2Token.main.precompute_threshold_unlock_targets import is_allowed_anchor_token
from Token2Token.main.train import MODEL_ID, _token_ids, completion_logits, load_base_model


def main() -> None:
    args = parse_args()
    tokenizer, base_model, mask_token_id, device = load_base_model(
        args.model_id, args.device
    )
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, args.adapter_path).to(device)
    model.eval()
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
    )
    canvas = [mask_token_id] * args.completion_length
    if args.confidence_threshold is not None:
        decode_trace = threshold_unlock_decode(
            model,
            tokenizer,
            prompt_ids,
            canvas,
            mask_token_id,
            confidence_threshold=args.confidence_threshold,
            max_threshold_tokens=args.max_threshold_tokens,
            device=device,
        )
    else:
        decode_trace = confidence_decode(
            model,
            tokenizer,
            prompt_ids,
            canvas,
            mask_token_id,
            steps=args.decode_steps,
            tokens_per_step=args.tokens_per_step,
            device=device,
        )
    result = {
        "prompt": args.prompt,
        "decode_trace": decode_trace,
        "completion_token_ids": canvas,
        "completion": tokenizer.decode(canvas, skip_special_tokens=True),
    }
    rendered = json.dumps(result, ensure_ascii=True, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def confidence_decode(
    model,
    tokenizer,
    prompt_ids,
    canvas,
    mask_token_id,
    *,
    steps=None,
    tokens_per_step=None,
    device,
):
    if steps is None and tokens_per_step is None:
        raise ValueError("set steps or tokens_per_step")
    if steps is not None and steps <= 0:
        raise ValueError("decode_steps must be positive")
    if tokens_per_step is not None and tokens_per_step <= 0:
        raise ValueError("tokens_per_step must be positive")
    trace = []
    step = 0
    with torch.no_grad():
        while mask_token_id in canvas:
            step += 1
            masked = [
                position
                for position, token_id in enumerate(canvas)
                if token_id == mask_token_id
            ]
            if tokens_per_step is not None:
                fill_count = min(len(masked), tokens_per_step)
            else:
                remaining_steps = max(1, steps - step + 1)
                fill_count = min(
                    len(masked), max(1, math.ceil(len(masked) / remaining_steps))
                )
            logits = completion_logits(model, prompt_ids, canvas, device).float()
            logits[:, mask_token_id] = -torch.inf
            probabilities = torch.softmax(logits, dim=-1)
            confidence, token_ids = probabilities.max(dim=-1)
            positions = sorted(
                masked, key=lambda item: float(confidence[item]), reverse=True
            )[:fill_count]
            filled = []
            for position in positions:
                token_id = int(token_ids[position])
                canvas[position] = token_id
                filled.append(
                    {
                        "position": position,
                        "token_id": token_id,
                        "token": tokenizer.decode([token_id]),
                        "confidence": float(confidence[position]),
                    }
                )
            trace.append({"step": step, "filled": filled})
    return trace


def threshold_unlock_decode(
    model,
    tokenizer,
    prompt_ids,
    canvas,
    mask_token_id,
    *,
    confidence_threshold,
    max_threshold_tokens=None,
    device,
):
    if not 0 < confidence_threshold < 1:
        raise ValueError("confidence_threshold must be in (0, 1)")
    trace = []
    cycle = 0
    forward_pass = 0
    with torch.no_grad():
        while mask_token_id in canvas:
            cycle += 1
            confidence, token_ids = canvas_predictions(
                model, prompt_ids, canvas, mask_token_id, device
            )
            forward_pass += 1
            masked = [
                position
                for position, token_id in enumerate(canvas)
                if token_id == mask_token_id
            ]
            allowed = [
                position
                for position in masked
                if is_allowed_anchor_token(int(token_ids[position]), tokenizer)
            ]
            if not allowed:
                cleanup_position = min(masked)
                cleanup = fill_positions(
                    tokenizer,
                    canvas,
                    [cleanup_position],
                    token_ids,
                    confidence,
                )
                trace.append(
                    {
                        "cycle": cycle,
                        "forward": forward_pass,
                        "phase": "cleanup",
                        "filled": cleanup,
                    }
                )
                continue
            catalyst_position = max(
                allowed, key=lambda position: float(confidence[position])
            )
            catalyst = fill_positions(
                tokenizer,
                canvas,
                [catalyst_position],
                token_ids,
                confidence,
            )
            trace.append(
                {
                    "cycle": cycle,
                    "forward": forward_pass,
                    "phase": "catalyst",
                    "filled": catalyst,
                }
            )
            if mask_token_id not in canvas:
                break

            confidence, token_ids = canvas_predictions(
                model, prompt_ids, canvas, mask_token_id, device
            )
            forward_pass += 1
            unlocked_positions = sorted(
                (
                    position
                    for position, token_id in enumerate(canvas)
                    if token_id == mask_token_id
                    and float(confidence[position]) >= confidence_threshold
                ),
                key=lambda position: float(confidence[position]),
                reverse=True,
            )
            if max_threshold_tokens is not None:
                unlocked_positions = unlocked_positions[:max_threshold_tokens]
            unlocked = fill_positions(
                tokenizer,
                canvas,
                unlocked_positions,
                token_ids,
                confidence,
            )
            trace.append(
                {
                    "cycle": cycle,
                    "forward": forward_pass,
                    "phase": "threshold_unlock",
                    "threshold": confidence_threshold,
                    "filled": unlocked,
                }
            )
    return trace


def canvas_predictions(model, prompt_ids, canvas, mask_token_id, device):
    logits = completion_logits(model, prompt_ids, canvas, device).float()
    logits[:, mask_token_id] = -torch.inf
    probabilities = torch.softmax(logits, dim=-1)
    return probabilities.max(dim=-1)


def fill_positions(tokenizer, canvas, positions, token_ids, confidence):
    filled = []
    for position in positions:
        token_id = int(token_ids[position])
        canvas[position] = token_id
        filled.append(
            {
                "position": position,
                "token_id": token_id,
                "token": tokenizer.decode([token_id]),
                "confidence": float(confidence[position]),
            }
        )
    return filled


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standard confidence decode for an IG-anchor-trained adapter"
    )
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--completion-length", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--tokens-per-step", type=int)
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--max-threshold-tokens", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.max_threshold_tokens is not None and args.max_threshold_tokens <= 0:
        parser.error("max-threshold-tokens must be positive")
    return args


if __name__ == "__main__":
    main()
