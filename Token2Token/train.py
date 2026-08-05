#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time

import torch

from Token2Token.core import (
    Commit,
    Target,
    anchor_loss,
    make_canvas,
    position_priors,
    select_targets,
)


MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    tokenizer, model, mask_token_id, device = load_model(args)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    records = record_stream(args)
    log_path = output / "train.jsonl"
    step = 0
    example_count = 0
    started = time.time()

    while step < args.max_steps:
        record = next(records)
        encoded = encode_record(record, tokenizer, args)
        if encoded is None:
            continue
        prompt_ids, gold_ids, example_id = encoded
        candidates = select_targets(
            gold_ids,
            tokenizer,
            count=len(gold_ids),
            max_right_fraction=args.target_right_fraction,
            policy="prefix",
            mask_token_id=mask_token_id,
        )
        if not candidates:
            continue
        targets, ig_scores = greedy_ig_targets(
            model,
            prompt_ids,
            gold_ids,
            candidates,
            mask_token_id,
            count=args.anchors,
            batch_size=args.ig_batch_size,
            device=device,
        )
        example_count += 1
        commits = {}

        for round_index in range(len(targets)):
            if step >= args.max_steps:
                break
            stage_targets = targets[: round_index + 1]
            canvas = make_canvas(
                len(gold_ids), mask_token_id, targets, commits
            )
            priors = position_priors(
                stage_targets,
                commits,
                len(canvas),
                sigma=args.sigma,
                max_right_fraction=args.target_right_fraction,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast(device, args.bf16):
                logits = completion_logits(model, prompt_ids, canvas, device)
                losses, _, _ = anchor_loss(
                    logits,
                    stage_targets,
                    commits,
                    priors,
                    lambda_order=args.lambda_order,
                    lambda_ce=args.lambda_ce,
                )
            if not bool(torch.isfinite(losses.total)):
                raise FloatingPointError(
                    f"non-finite loss at step {step}: {losses.total}"
                )
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()

            step += 1
            target = targets[round_index]
            row = {
                "step": step,
                "dataset": args.dataset,
                "example_id": example_id,
                "loss": float(losses.total.detach().cpu()),
                "placement": float(losses.placement.detach().cpu()),
                "order": float(losses.order.detach().cpu()),
                "ce": float(losses.grounding.detach().cpu()),
                "anchor_token": target.token_text,
                "gold_position": target.gold_position,
                "ig_rank": round_index + 1,
                "ig_score": ig_scores[round_index],
                "input_anchors": _canvas_anchors(canvas, mask_token_id, tokenizer),
                "elapsed_seconds": time.time() - started,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            print(
                f"step={step} loss={row['loss']:.4f} place={row['placement']:.4f} "
                f"order={row['order']:.6f} ce={row['ce']:.4f} "
                f"ig={row['ig_score']:.4f} anchor={target.token_text!r} "
                f"gold_position={target.gold_position}"
            )
            commits[target.index] = Commit(
                target.index, target.gold_position, confidence=1.0
            )
            if args.save_every and step % args.save_every == 0:
                save_adapter(model, tokenizer, output / f"checkpoint-{step:06d}")

    save_adapter(model, tokenizer, output / "adapter-final")


def load_model(args):
    from peft import LoraConfig, PeftModel, get_peft_model

    tokenizer, model, mask_token_id, device = load_base_model(
        args.model_id, args.device
    )
    adapter_path = getattr(args, "adapter_path", None)
    if adapter_path:
        model = PeftModel.from_pretrained(
            model, adapter_path, is_trainable=True
        ).to(device)
        model.print_trainable_parameters()
        model.train()
        return tokenizer, model, mask_token_id, device

    available = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    requested = [name.strip() for name in args.lora_targets.split(",") if name.strip()]
    targets = [name for name in requested if name in available]
    if not targets:
        raise ValueError(f"LoRA targets {requested} not found; available: {sorted(available)}")
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
            bias="none",
        ),
    )
    model.print_trainable_parameters()
    model.train()
    return tokenizer, model, mask_token_id, device


def load_base_model(model_id, requested_device="auto"):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    _patch_tied_weights()
    device = (
        "cuda"
        if requested_device == "auto" and torch.cuda.is_available()
        else requested_device
    )
    if device == "auto":
        device = "cpu"
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if getattr(config, "use_cache", None) is None:
        config.use_cache = False
    _patch_llada_remote_compat(config, model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
    ).to(device)
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = getattr(config, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError("LLaDA tokenizer/config has no mask token id")
    return tokenizer, model, int(mask_token_id), str(device)


def record_stream(args):
    from datasets import load_dataset

    if args.dataset == "gsm8k":
        rows = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    else:
        rows = load_dataset(args.lm1b_dataset, split="train", streaming=True)
    rows = rows.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    while True:
        emitted = False
        for index, row in enumerate(rows):
            emitted = True
            yield index, row
        if not emitted:
            raise ValueError(f"dataset {args.dataset} is empty")


def encode_record(record, tokenizer, args):
    index, row = record
    if args.dataset == "gsm8k":
        prompt = str(row["question"]).rstrip() + "\n"
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
        prompt_ids = _token_ids(prompt_ids)
        gold_ids = _token_ids(
            tokenizer(str(row["answer"]).strip(), add_special_tokens=False)
        )[: args.max_completion_tokens]
        example_id = str(row.get("id", index))
    else:
        ids = _token_ids(tokenizer(str(row["text"]), add_special_tokens=True))
        if len(ids) <= args.lm1b_prompt_tokens + 1:
            return None
        prompt_ids = ids[: args.lm1b_prompt_tokens]
        gold_ids = ids[
            args.lm1b_prompt_tokens : args.lm1b_prompt_tokens + args.max_completion_tokens
        ]
        example_id = str(index)
    return list(map(int, prompt_ids)), list(map(int, gold_ids)), example_id


def _token_ids(encoded):
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(map(int, encoded))


def _canvas_anchors(canvas, mask_token_id, tokenizer):
    return [
        {
            "position": position,
            "token_id": token_id,
            "token": tokenizer.decode([token_id]),
        }
        for position, token_id in enumerate(canvas)
        if token_id != mask_token_id
    ]


def completion_logits(model, prompt_ids, canvas, device):
    input_ids = torch.tensor([prompt_ids + canvas], device=device, dtype=torch.long)
    outputs = model(input_ids=input_ids, use_cache=False)
    return outputs.logits[0, len(prompt_ids) : len(prompt_ids) + len(canvas)]


def greedy_ig_targets(
    model,
    prompt_ids,
    gold_ids,
    candidates,
    mask_token_id,
    *,
    count,
    batch_size,
    device,
):
    """Rank gold tokens by greedy entropy-reduction information gain."""

    if batch_size <= 0:
        raise ValueError("ig_batch_size must be positive")
    count = min(int(count), len(candidates))
    canvas = [int(mask_token_id)] * len(gold_ids)
    selected = []
    scores = []
    selected_positions = set()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            current_entropy = completion_entropies(
                model, prompt_ids, [canvas], device
            )[0]
            for rank in range(count):
                remaining_candidates = [
                    target
                    for target in candidates
                    if target.gold_position not in selected_positions
                ]
                best_target = None
                best_entropy = None
                best_score = float("-inf")
                for start in range(0, len(remaining_candidates), batch_size):
                    chunk = remaining_candidates[start : start + batch_size]
                    canvases = []
                    for target in chunk:
                        candidate_canvas = list(canvas)
                        candidate_canvas[target.gold_position] = target.token_id
                        canvases.append(candidate_canvas)
                    candidate_entropies = completion_entropies(
                        model, prompt_ids, canvases, device
                    )
                    for target, after_entropy in zip(chunk, candidate_entropies):
                        excluded = selected_positions | {target.gold_position}
                        remaining = [
                            position
                            for position in range(len(gold_ids))
                            if position not in excluded
                        ]
                        score = float(
                            (current_entropy[remaining] - after_entropy[remaining])
                            .sum()
                            .item()
                        )
                        if score > best_score:
                            best_target = target
                            best_entropy = after_entropy
                            best_score = score
                if best_target is None or best_entropy is None:
                    break
                selected.append(
                    Target(
                        rank,
                        best_target.token_id,
                        best_target.token_text,
                        best_target.gold_position,
                    )
                )
                scores.append(best_score)
                selected_positions.add(best_target.gold_position)
                canvas[best_target.gold_position] = best_target.token_id
                current_entropy = best_entropy
    finally:
        model.train(was_training)
    return selected, scores


def completion_entropies(model, prompt_ids, canvases, device):
    input_ids = torch.tensor(
        [prompt_ids + canvas for canvas in canvases],
        device=device,
        dtype=torch.long,
    )
    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits[
        :, len(prompt_ids) : len(prompt_ids) + len(canvases[0])
    ].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1).cpu()


def save_adapter(model, tokenizer, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def autocast(device: str, enabled: bool):
    if str(device).startswith("cuda") and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _patch_tied_weights():
    from transformers.modeling_utils import PreTrainedModel

    existing = getattr(PreTrainedModel, "all_tied_weights_keys", None)
    if isinstance(existing, property) and existing.fset is not None:
        return

    @property
    def keys(self):
        value = self.__dict__.get("_token2token_tied_keys", getattr(self, "_tied_weights_keys", []))
        return value if isinstance(value, dict) else {key: None for key in value or []}

    @keys.setter
    def keys(self, value):
        self.__dict__["_token2token_tied_keys"] = value

    PreTrainedModel.all_tied_weights_keys = keys


def _patch_llada_remote_compat(config, model_id: str):
    """Bridge the older LLaDA remote code to current Transformers releases."""

    try:
        from transformers import modeling_rope_utils

        rope_functions = getattr(modeling_rope_utils, "ROPE_INIT_FUNCTIONS", None)
        default_rope = getattr(
            modeling_rope_utils, "_compute_default_rope_parameters", None
        )
        if isinstance(rope_functions, dict) and "default" not in rope_functions:
            if default_rope is not None:
                rope_functions["default"] = default_rope
    except (ImportError, AttributeError):
        pass

    auto_map = getattr(config, "auto_map", {}) or {}
    class_ref = auto_map.get("AutoModel") or auto_map.get("AutoModelForCausalLM")
    if isinstance(class_ref, (list, tuple)):
        class_ref = next((ref for ref in reversed(class_ref) if ref), None)
    if not class_ref:
        return

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_class = get_class_from_dynamic_module(
        class_ref, model_id, trust_remote_code=True
    )
    original = getattr(model_class, "tie_weights", None)
    if original is None or getattr(original, "_token2token_compat", False):
        return

    def tie_weights(self, *args, **kwargs):
        del args, kwargs
        return original(self)

    tie_weights._token2token_compat = True
    model_class.tie_weights = tie_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal IG-anchor LoRA trainer")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset", choices=["gsm8k", "lm1b"], default="gsm8k")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--anchors", type=int, default=5)
    parser.add_argument("--target-right-fraction", type=float, default=0.75)
    parser.add_argument("--ig-batch-size", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--lambda-order", type=float, default=0.25)
    parser.add_argument("--lambda-ce", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,attn_out")
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    parser.add_argument("--lm1b-prompt-tokens", type=int, default=32)
    parser.add_argument("--lm1b-dataset", default="FrankCCCCC/lm1b")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/token2token/llada8b_gsm8k")
    return parser.parse_args()


if __name__ == "__main__":
    main()
