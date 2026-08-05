#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import torch

from Token2Token.train import MODEL_ID, _token_ids, load_base_model


def main() -> None:
    args = parse_args()
    k_values = parse_k_values(args.k_values)
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
    for k in k_values:
        rows_path = output / f"predictions_k{k}.jsonl"
        completed = load_completed(rows_path) if args.resume else set()
        correct, evaluated = existing_totals(rows_path) if args.resume else (0, 0)
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
            canvases = batch_confidence_decode(
                model,
                [item[2] for item in batch],
                args.completion_length,
                mask_token_id,
                tokens_per_step=k,
                device=device,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else mask_token_id
                ),
            )
            for (example_id, row, _), canvas in zip(batch, canvases):
                decoded = tokenizer.decode(canvas, skip_special_tokens=True)
                predicted_answer = extract_gsm8k_answer(decoded)
                gold_answer = extract_gsm8k_answer(str(row["answer"]))
                is_correct = predicted_answer == gold_answer
                evaluated += 1
                correct += int(is_correct)
                result = {
                    "model_label": args.model_label,
                    "tokens_per_step": k,
                    "example_id": example_id,
                    "question": row["question"],
                    "decoded_completion": decoded,
                    "predicted_answer": predicted_answer,
                    "gold_answer": gold_answer,
                    "correct": is_correct,
                }
                with rows_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=True) + "\n")
                print(
                    f"model={args.model_label} k={k} example={evaluated} "
                    f"accuracy={correct / evaluated:.4f}"
                )

        summary = {
            "model_label": args.model_label,
            "model_id": args.model_id,
            "adapter_path": args.adapter_path,
            "tokens_per_step": k,
            "examples": evaluated,
            "correct": correct,
            "accuracy": correct / evaluated if evaluated else 0.0,
            "completion_length": args.completion_length,
            "elapsed_seconds": time.time() - started,
        }
        summaries = [item for item in summaries if item["tokens_per_step"] != k]
        summaries.append(summary)
        summaries.sort(key=lambda item: item["tokens_per_step"])
        summary_path.write_text(
            json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summaries, indent=2))


def batch_confidence_decode(
    model,
    prompt_ids_batch,
    completion_length,
    mask_token_id,
    *,
    tokens_per_step,
    device,
    pad_token_id,
):
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step must be positive")
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

    with torch.no_grad():
        while bool((canvases == mask_token_id).any()):
            input_ids = torch.cat([prompts, canvases], dim=1)
            attention_mask = torch.cat([prompt_attention, completion_attention], dim=1)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits[:, max_prompt : max_prompt + completion_length].float()
            logits[:, :, mask_token_id] = -torch.inf
            probabilities = torch.softmax(logits, dim=-1)
            confidence, token_ids = probabilities.max(dim=-1)
            masked = canvases == mask_token_id
            remaining = int(masked[0].sum())
            fill_count = min(tokens_per_step, remaining)
            ranked_confidence = confidence.masked_fill(~masked, -torch.inf)
            positions = torch.topk(ranked_confidence, fill_count, dim=-1).indices
            selected_ids = torch.gather(token_ids, 1, positions)
            canvases.scatter_(1, positions, selected_ids)
    return canvases.detach().cpu().tolist()


def parse_k_values(value: str) -> list[int]:
    values = []
    for item in value.split(","):
        if not item.strip():
            continue
        parsed = int(item.strip())
        if parsed not in values:
            values.append(parsed)
    if not values or any(value <= 0 for value in values):
        raise ValueError("k-values must contain positive integers")
    return values


def extract_gsm8k_answer(text: str) -> str:
    marker = re.search(r"####\s*([^\n]+)", text)
    if marker:
        number = first_number(marker.group(1))
        if number is not None:
            return number
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return normalize_number(numbers[-1]) if numbers else ""


def first_number(text: str) -> str | None:
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", text)
    return normalize_number(match.group(0)) if match else None


def normalize_number(text: str) -> str:
    return text.strip().replace(",", "").strip(" .:$").lower()


def load_completed(path: Path) -> set[str]:
    return {str(row["example_id"]) for row in read_jsonl(path)}


def existing_totals(path: Path) -> tuple[int, int]:
    rows = list(read_jsonl(path))
    return sum(int(row["correct"]) for row in rows), len(rows)


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GSM8K accuracy sweep over top-k confidence decoding"
    )
    parser.add_argument("--adapter-path")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--completion-length", type=int, default=128)
    parser.add_argument("--k-values", default="5,4,3,2,1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    return args


if __name__ == "__main__":
    main()
