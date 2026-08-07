#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
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
        if not args.resume and rows_path.exists():
            # Rows are appended, so a rerun into a directory left behind by an
            # interrupted run would silently duplicate examples and corrupt
            # every aggregate computed from the file.
            rows_path.unlink()
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
            pad_token_id = (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else mask_token_id
            )
            if args.decoder == "topk":
                canvases, decode_stats = batch_topk_decode(
                    model,
                    [item[2] for item in batch],
                    args.completion_length,
                    mask_token_id,
                    tokens_per_step=args.tokens_per_step,
                    block_length=args.block_length,
                    device=device,
                    pad_token_id=pad_token_id,
                )
            else:
                canvases, decode_stats = batch_threshold_unlock_decode(
                    model,
                    [item[2] for item in batch],
                    args.completion_length,
                    mask_token_id,
                    confidence_threshold=threshold,
                    max_threshold_tokens=args.max_threshold_tokens,
                    commit_threshold_on_first_forward=(
                        args.commit_threshold_on_first_forward
                    ),
                    unlock_forward=args.unlock_forward,
                    catalyst_tokens_per_forward=args.catalyst_tokens_per_forward,
                    base_first_forward=(args.adapter_scope == "unlock"),
                    catalyst_filter=args.catalyst_filter,
                    catalyst_min_length=args.catalyst_min_length,
                    force_catalyst=args.force_catalyst,
                    record_commit_phase=args.record_commit_phase,
                    tokenizer=tokenizer,
                    device=device,
                    pad_token_id=pad_token_id,
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
            "max_threshold_tokens": args.max_threshold_tokens,
            "decoder": args.decoder,
            "tokens_per_step": args.tokens_per_step,
            "block_length": args.block_length,
            "commit_threshold_on_first_forward": (
                args.commit_threshold_on_first_forward
            ),
            "unlock_forward": args.unlock_forward,
            "catalyst_tokens_per_forward": args.catalyst_tokens_per_forward,
            "adapter_scope": args.adapter_scope,
            "catalyst_filter": args.catalyst_filter,
            "catalyst_min_length": args.catalyst_min_length,
            "force_catalyst": args.force_catalyst,
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
    max_threshold_tokens=None,
    commit_threshold_on_first_forward=False,
    unlock_forward=True,
    catalyst_tokens_per_forward=1,
    base_first_forward=False,
    catalyst_filter="text",
    catalyst_min_length=0,
    force_catalyst="always",
    record_commit_phase=False,
    tokenizer,
    device,
    pad_token_id,
):
    if not 0 < confidence_threshold < 1:
        raise ValueError("confidence_threshold must be in (0, 1)")
    if not unlock_forward and not commit_threshold_on_first_forward:
        raise ValueError(
            "disabling the unlock forward requires first-forward threshold commits"
        )
    if catalyst_tokens_per_forward <= 0:
        raise ValueError("catalyst_tokens_per_forward must be positive")
    if base_first_forward and not unlock_forward:
        # Gating the adapter to the unlock forward when there is no unlock
        # forward disables it for every forward, silently evaluating the base
        # model under a trained model's label.
        raise ValueError(
            "base_first_forward needs an unlock forward for the adapter to run"
        )
    if force_catalyst == "when-empty" and not commit_threshold_on_first_forward:
        # Skipping the forced commit is only safe when the same forward is the
        # one applying the threshold, otherwise a cycle can commit nothing.
        raise ValueError(
            "force_catalyst='when-empty' requires first-forward threshold commits"
        )
    prompts, prompt_attention = pad_prompts(prompt_ids_batch, pad_token_id, device)
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
    first_forward_tokens = torch.zeros_like(forwards)
    # 1 catalyst, 2 cleanup, 3 first-forward threshold, 4 unlock threshold.
    commit_phase = torch.zeros_like(canvases)
    allowed_cache = {}

    with torch.no_grad():
        while bool(canvases.eq(mask_token_id).any()):
            masked = canvases.eq(mask_token_id)
            active = masked.any(dim=1)
            with adapter_disabled(model, base_first_forward):
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
                token_ids,
                masked,
                tokenizer,
                allowed_cache,
                catalyst_filter,
                catalyst_min_length,
            )
            has_anchor = allowed.any(dim=1) & active
            cleanup = active & ~has_anchor
            forcing = active
            if force_catalyst == "when-empty":
                # The forced commit exists only to guarantee progress. When the
                # threshold already selects something, forcing an extra token
                # commits the most confident position that was still judged
                # not confident enough, which is the decoder's least reliable
                # commit of the cycle.
                forcing = active & ~(masked & confidence.ge(confidence_threshold)).any(
                    dim=1
                )
            scores = confidence.masked_fill(~allowed, -torch.inf)
            budget = min(int(catalyst_tokens_per_forward), scores.shape[1])
            chosen = torch.zeros_like(allowed)
            chosen.scatter_(1, scores.topk(budget, dim=1).indices, True)
            chosen &= allowed & (has_anchor & forcing).unsqueeze(1)
            leftmost = torch.zeros_like(allowed)
            leftmost.scatter_(1, masked.long().argmax(dim=1, keepdim=True), True)
            chosen |= leftmost & (cleanup & forcing).unsqueeze(1)
            canvases[chosen] = token_ids[chosen]
            commit_phase[chosen & has_anchor.unsqueeze(1)] = 1
            commit_phase[chosen & cleanup.unsqueeze(1)] = 2
            catalyst_tokens += (chosen & has_anchor.unsqueeze(1)).sum(dim=1)
            cleanup_tokens[cleanup & forcing] += 1

            if commit_threshold_on_first_forward:
                masked = canvases.eq(mask_token_id)
                selected = masked & confidence.ge(confidence_threshold)
                selected &= active.unsqueeze(1)
                selected = limit_threshold_selection(
                    selected, confidence, max_threshold_tokens
                )
                first_forward_tokens += selected.sum(dim=1)
                threshold_tokens += selected.sum(dim=1)
                canvases[selected] = token_ids[selected]
                commit_phase[selected] = 3

            masked = canvases.eq(mask_token_id)
            unlock_active = masked.any(dim=1) & has_anchor
            if not unlock_forward:
                if bool(masked.any()):
                    continue
                break
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
            selected = limit_threshold_selection(
                selected, confidence, max_threshold_tokens
            )
            threshold_tokens += selected.sum(dim=1)
            canvases[selected] = token_ids[selected]
            commit_phase[selected] = 4

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
                "first_forward_threshold_tokens": int(first_forward_tokens[index]),
                "tokens_per_forward": completion_length / row_forwards,
                **(
                    {"commit_phase": commit_phase[index].detach().cpu().tolist()}
                    if record_commit_phase
                    else {}
                ),
            }
        )
    return canvases.detach().cpu().tolist(), rows


def batch_topk_decode(
    model,
    prompt_ids_batch,
    completion_length,
    mask_token_id,
    *,
    tokens_per_step,
    block_length=None,
    device,
    pad_token_id,
):
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step must be positive")
    if block_length is not None and block_length <= 0:
        raise ValueError("block_length must be positive")
    prompts, prompt_attention = pad_prompts(prompt_ids_batch, pad_token_id, device)
    canvases = torch.full(
        (len(prompt_ids_batch), completion_length),
        int(mask_token_id),
        device=device,
        dtype=torch.long,
    )
    completion_attention = torch.ones_like(canvases)
    forwards = torch.zeros(len(canvases), device=device, dtype=torch.long)

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
            count = min(int(tokens_per_step), canvases.shape[1])
            eligible = masked
            if block_length is not None:
                eligible = masked & current_block(masked, block_length)
            scores = confidence.masked_fill(~eligible, -torch.inf)
            positions = scores.topk(count, dim=1).indices
            selected = torch.zeros_like(masked)
            selected.scatter_(1, positions, True)
            selected &= eligible & active.unsqueeze(1)
            canvases[selected] = token_ids[selected]

    rows = []
    for index in range(len(canvases)):
        row_forwards = int(forwards[index])
        rows.append(
            {
                "cycles": row_forwards,
                "model_forwards": row_forwards,
                "catalyst_tokens": 0,
                "cleanup_tokens": 0,
                "threshold_tokens": 0,
                "first_forward_threshold_tokens": 0,
                "tokens_per_forward": completion_length / row_forwards,
            }
        )
    return canvases.detach().cpu().tolist(), rows


def current_block(masked, block_length):
    """Restrict candidates to the leftmost block that still holds a mask.

    This is the semi-autoregressive schedule LLaDA is normally generated with:
    the completion is filled block by block, left to right, with confidence
    ordering applied only inside the active block. Pure global-confidence
    decoding is a different and weaker baseline, so quoting only that would
    overstate any decoder result.
    """
    width = masked.shape[1]
    index = torch.arange(width, device=masked.device).unsqueeze(0)
    first_masked = masked.long().argmax(dim=1, keepdim=True)
    block_start = (first_masked // block_length) * block_length
    return (index >= block_start) & (index < block_start + block_length)


def pad_prompts(prompt_ids_batch, pad_token_id, device):
    max_prompt = max(len(prompt_ids) for prompt_ids in prompt_ids_batch)
    padded_prompts = []
    prompt_masks = []
    for prompt_ids in prompt_ids_batch:
        padding = max_prompt - len(prompt_ids)
        padded_prompts.append([pad_token_id] * padding + prompt_ids)
        prompt_masks.append([0] * padding + [1] * len(prompt_ids))
    return (
        torch.tensor(padded_prompts, device=device, dtype=torch.long),
        torch.tensor(prompt_masks, device=device, dtype=torch.long),
    )


def adapter_disabled(model, enabled):
    """Run the catalyst forward on the frozen base so the adapter can only
    affect the post-anchor unlock forward."""
    if enabled and hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return nullcontext()


def limit_threshold_selection(selected, confidence, max_threshold_tokens):
    if max_threshold_tokens is None:
        return selected
    if max_threshold_tokens <= 0:
        raise ValueError("max_threshold_tokens must be positive")
    count = min(int(max_threshold_tokens), selected.shape[1])
    scores = confidence.masked_fill(~selected, -torch.inf)
    positions = scores.topk(count, dim=1).indices
    limited = torch.zeros_like(selected)
    limited.scatter_(1, positions, True)
    return selected & limited


def allowed_prediction_mask(
    token_ids, masked, tokenizer, cache, catalyst_filter="text", min_length=0
):
    if catalyst_filter == "any":
        # Control: the catalyst is just the most confident masked position,
        # so the decoder reduces to ordinary confidence-plus-threshold
        # decoding and any difference is attributable to the text filter.
        return masked
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
                cache[token_id] = is_allowed_anchor_token(
                    token_id, tokenizer
                ) and len(tokenizer.decode([token_id]).strip()) >= min_length
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
    parser.add_argument("--max-threshold-tokens", type=int)
    parser.add_argument("--decoder", choices=("catalyst", "topk"), default="catalyst")
    parser.add_argument("--tokens-per-step", type=int, default=1)
    parser.add_argument("--block-length", type=int)
    parser.add_argument(
        "--commit-threshold-on-first-forward",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--unlock-forward", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--catalyst-tokens-per-forward", type=int, default=1)
    parser.add_argument(
        "--catalyst-filter", choices=("text", "any"), default="text"
    )
    parser.add_argument("--catalyst-min-length", type=int, default=0)
    parser.add_argument(
        "--force-catalyst", choices=("always", "when-empty"), default="always"
    )
    parser.add_argument(
        "--record-commit-phase",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--adapter-scope", choices=("all", "unlock"), default="all"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.max_threshold_tokens is not None and args.max_threshold_tokens <= 0:
        parser.error("max-threshold-tokens must be positive")
    if args.tokens_per_step <= 0:
        parser.error("tokens-per-step must be positive")
    if args.block_length is not None and args.block_length <= 0:
        parser.error("block-length must be positive")
    if args.catalyst_tokens_per_forward <= 0:
        parser.error("catalyst-tokens-per-forward must be positive")
    if args.adapter_scope == "unlock" and not args.unlock_forward:
        parser.error("--adapter-scope unlock needs --unlock-forward")
    if args.force_catalyst == "when-empty" and not args.commit_threshold_on_first_forward:
        parser.error(
            "--force-catalyst when-empty requires --commit-threshold-on-first-forward"
        )
    if not args.unlock_forward and not args.commit_threshold_on_first_forward:
        parser.error(
            "--no-unlock-forward requires --commit-threshold-on-first-forward"
        )
    parse_thresholds(args.thresholds)
    return args


if __name__ == "__main__":
    main()
