#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from Token2Token.eval_gsm8k import extract_gsm8k_answer, read_jsonl
from Token2Token.precompute_threshold_unlock_targets import is_allowed_anchor_token
from Token2Token.train import MODEL_ID, _token_ids, load_base_model


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer, model, mask_token_id, device = load_base_model(
        args.model_id, args.device
    )
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path).to(device)
    model.eval()

    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    summary_path = output / "summary.json"
    summaries = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else []
    )
    for threshold in thresholds:
        tag = threshold_tag(threshold)
        rows_path = output / f"predictions_{tag}.jsonl"
        completed = load_completed(rows_path) if args.resume else set()
        existing = list(read_jsonl(rows_path)) if args.resume else []
        correct = sum(int(row["correct"]) for row in existing)
        evaluated = len(existing)
        total_forwards = sum(int(row["model_forwards"]) for row in existing)
        total_threshold_tokens = sum(int(row["threshold_tokens"]) for row in existing)
        started = time.time()
        pending = []
        for index, row in enumerate(dataset):
            example_id = str(index)
            if example_id in completed:
                continue
            if args.limit is not None and evaluated + len(pending) >= args.limit:
                break
            prompt = str(row["question"]).rstrip() + "\n"
            prompt_ids = _token_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=True,
                )
            )
            pending.append((example_id, row, prompt_ids))

        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            canvases, decode_stats = batch_threshold_unlock_decode(
                model,
                [item[2] for item in batch],
                args.completion_length,
                mask_token_id,
                confidence_threshold=threshold,
                tokenizer=tokenizer,
                device=device,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else mask_token_id
                ),
            )
            for (example_id, row, _), canvas, stats in zip(
                batch, canvases, decode_stats
            ):
                decoded = tokenizer.decode(canvas, skip_special_tokens=True)
                predicted_answer = extract_gsm8k_answer(decoded)
                gold_answer = extract_gsm8k_answer(str(row["answer"]))
                is_correct = predicted_answer == gold_answer
                evaluated += 1
                correct += int(is_correct)
                total_forwards += int(stats["model_forwards"])
                total_threshold_tokens += int(stats["threshold_tokens"])
                result = {
                    "model_label": args.model_label,
                    "confidence_threshold": threshold,
                    "example_id": example_id,
                    "question": row["question"],
                    "decoded_completion": decoded,
                    "predicted_answer": predicted_answer,
                    "gold_answer": gold_answer,
                    "correct": is_correct,
                    **stats,
                }
                with rows_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=True) + "\n")
                print(
                    f"model={args.model_label} threshold={threshold:.3f} "
                    f"example={evaluated} accuracy={correct / evaluated:.4f} "
                    f"tokens_per_forward={args.completion_length * evaluated / total_forwards:.3f}"
                )

        summary = {
            "model_label": args.model_label,
            "model_id": args.model_id,
            "adapter_path": args.adapter_path,
            "confidence_threshold": threshold,
            "examples": evaluated,
            "correct": correct,
            "accuracy": correct / evaluated if evaluated else 0.0,
            "completion_length": args.completion_length,
            "total_model_forwards": total_forwards,
            "tokens_per_forward": (
                args.completion_length * evaluated / total_forwards
                if total_forwards
                else 0.0
            ),
            "mean_threshold_tokens": (
                total_threshold_tokens / evaluated if evaluated else 0.0
            ),
            "elapsed_seconds": time.time() - started,
        }
        summaries = [
            item
            for item in summaries
            if float(item["confidence_threshold"]) != threshold
        ]
        summaries.append(summary)
        summaries.sort(key=lambda item: item["confidence_threshold"], reverse=True)
        summary_path.write_text(
            json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summaries, indent=2))


def batch_threshold_unlock_decode(
    model,
    prompt_ids_batch,
    completion_length,
    mask_token_id,
    *,
    confidence_threshold,
    tokenizer,
    device,
    pad_token_id,
):
    if not 0 < confidence_threshold < 1:
        raise ValueError("confidence_threshold must be in (0, 1)")
    max_prompt = max(len(prompt_ids) for prompt_ids in prompt_ids_batch)
    padded_prompts = []
    prompt_masks = []
    for prompt_ids in prompt_ids_batch:
        padding = max_prompt - len(prompt_ids)
        padded_prompts.append([pad_token_id] * padding + prompt_ids)
        prompt_masks.append([0] * padding + [1] * len(prompt_ids))
    prompts = torch.tensor(padded_prompts, device=device, dtype=torch.long)
    prompt_attention = torch.tensor(prompt_masks, device=device, dtype=torch.long)
    canvases = torch.full(
        (len(prompt_ids_batch), completion_length),
        int(mask_token_id),
        device=device,
        dtype=torch.long,
    )
    completion_attention = torch.ones_like(canvases)
    forwards = torch.zeros(len(canvases), device=device, dtype=torch.long)
    cycles = torch.zeros_like(forwards)
    catalyst_tokens = torch.zeros_like(forwards)
    cleanup_tokens = torch.zeros_like(forwards)
    threshold_tokens = torch.zeros_like(forwards)
    allowed_cache = {}

    with torch.no_grad():
        while bool(canvases.eq(mask_token_id).any()):
            masked = canvases.eq(mask_token_id)
            active = masked.any(dim=1)
            confidence, token_ids = batch_canvas_predictions(
                model,
                prompts,
                prompt_attention,
                canvases,
                completion_attention,
                mask_token_id,
            )
            forwards[active] += 1
            cycles[active] += 1
            allowed = allowed_prediction_mask(
                token_ids, masked, tokenizer, allowed_cache
            )
            has_anchor = allowed.any(dim=1) & active
            cleanup = active & ~has_anchor
            catalyst_positions = confidence.masked_fill(~allowed, -torch.inf).argmax(
                dim=1
            )
            leftmost_positions = masked.long().argmax(dim=1)
            catalyst_positions = torch.where(
                has_anchor, catalyst_positions, leftmost_positions
            )
            active_rows = torch.where(active)[0]
            active_positions = catalyst_positions[active]
            canvases[active_rows, active_positions] = token_ids[
                active_rows, active_positions
            ]
            catalyst_tokens[has_anchor] += 1
            cleanup_tokens[cleanup] += 1

            masked = canvases.eq(mask_token_id)
            unlock_active = masked.any(dim=1) & has_anchor
            if not bool(unlock_active.any()):
                if bool(masked.any()):
                    continue
                break
            confidence, token_ids = batch_canvas_predictions(
                model,
                prompts,
                prompt_attention,
                canvases,
                completion_attention,
                mask_token_id,
            )
            forwards[unlock_active] += 1
            selected = masked & confidence.ge(confidence_threshold)
            selected &= unlock_active.unsqueeze(1)
            threshold_tokens += selected.sum(dim=1)
            canvases[selected] = token_ids[selected]

    rows = []
    for index in range(len(canvases)):
        row_forwards = int(forwards[index])
        rows.append(
            {
                "cycles": int(cycles[index]),
                "model_forwards": row_forwards,
                "catalyst_tokens": int(catalyst_tokens[index]),
                "cleanup_tokens": int(cleanup_tokens[index]),
                "threshold_tokens": int(threshold_tokens[index]),
                "tokens_per_forward": completion_length / row_forwards,
            }
        )
    return canvases.detach().cpu().tolist(), rows


def allowed_prediction_mask(token_ids, masked, tokenizer, cache):
    token_rows = token_ids.detach().cpu().tolist()
    masked_rows = masked.detach().cpu().tolist()
    allowed = []
    for row_tokens, row_masked in zip(token_rows, masked_rows):
        row = []
        for token_id, is_masked in zip(row_tokens, row_masked):
            if not is_masked:
                row.append(False)
                continue
            token_id = int(token_id)
            if token_id not in cache:
                cache[token_id] = is_allowed_anchor_token(token_id, tokenizer)
            row.append(cache[token_id])
        allowed.append(row)
    return torch.tensor(allowed, device=masked.device, dtype=torch.bool)


def batch_canvas_predictions(
    model,
    prompts,
    prompt_attention,
    canvases,
    completion_attention,
    mask_token_id,
):
    input_ids = torch.cat([prompts, canvases], dim=1)
    attention_mask = torch.cat([prompt_attention, completion_attention], dim=1)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    prompt_length = prompts.shape[1]
    logits = outputs.logits[
        :, prompt_length : prompt_length + canvases.shape[1]
    ].float()
    logits[:, :, mask_token_id] = -torch.inf
    probabilities = torch.softmax(logits, dim=-1)
    return probabilities.max(dim=-1)


def parse_thresholds(value):
    thresholds = []
    for item in value.split(","):
        if not item.strip():
            continue
        threshold = float(item)
        if threshold not in thresholds:
            thresholds.append(threshold)
    if not thresholds or any(not 0 < item < 1 for item in thresholds):
        raise ValueError("thresholds must contain values in (0, 1)")
    return thresholds


def threshold_tag(threshold):
    return f"t{threshold:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def load_completed(path):
    return {str(row["example_id"]) for row in read_jsonl(path)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate catalyst plus confidence-threshold decoding on GSM8K"
    )
    parser.add_argument("--adapter-path")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--completion-length", type=int, default=128)
    parser.add_argument("--thresholds", default="0.95")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    parse_thresholds(args.thresholds)
    return args


if __name__ == "__main__":
    main()
