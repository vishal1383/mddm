#!/usr/bin/env python3
"""Resumable 10-prompt DiffuCoder-cpGRPO versus dParallel diagnostic.

This is deliberately a held-out training-split screen, not an official-test
result.  It reports both a matched block-32 entropy decoder and DiffuCoder's
model-card native sampler so model and decoder effects remain distinguishable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Sequence

import torch

from Token2Token.main.eval_gsm8k import extract_gsm8k_answer
from Token2Token.main.eval_full256_pass5_baselines import PROMPT_SUFFIX, _prompt_ids
from Token2Token.main.train import load_base_model


CANVAS_LENGTH = 256
BLOCK_LENGTH = 32
SCREEN_START = 7463
SCREEN_STOP = 7473
MODEL_IDS = {
    "dparallel_entropy": "Zigeng/dParallel-LLaDA-8B-instruct",
    "diffucoder_entropy": "apple/DiffuCoder-7B-cpGRPO",
    "diffucoder_native": "apple/DiffuCoder-7B-cpGRPO",
}
METHOD_LABELS = {
    "dparallel_entropy": "dParallel checkpoint + published entropy decoder",
    "diffucoder_entropy": "DiffuCoder-cpGRPO + matched entropy decoder",
    "diffucoder_native": "DiffuCoder-cpGRPO + model-card native sampler",
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stable_seed(master: int, method: str, index: int) -> int:
    payload = json.dumps([int(master), method, int(index)], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_snapshot(model_id: str) -> tuple[Path, str]:
    homes = [Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))]
    if os.environ.get("DPARALLEL_HF_HOME"):
        homes.append(Path(os.environ["DPARALLEL_HF_HOME"]))
    for home in homes:
        repository = home / "hub" / ("models--" + model_id.replace("/", "--"))
        reference = repository / "refs" / "main"
        if reference.is_file():
            revision = reference.read_text(encoding="utf-8").strip()
            snapshot = repository / "snapshots" / revision
            if snapshot.is_dir() and (snapshot / "model.safetensors.index.json").is_file():
                return snapshot, revision
        # Some manually synchronized caches retain a single immutable snapshot
        # but omit refs/main.  Accept it only when the choice is unambiguous.
        complete = sorted(
            path
            for path in (repository / "snapshots").glob("*")
            if path.is_dir() and (path / "model.safetensors.index.json").is_file()
        )
        if len(complete) == 1:
            return complete[0], complete[0].name
    raise FileNotFoundError(f"no complete pinned local checkpoint found for {model_id}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def active_block_mask(canvas: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    result = torch.zeros_like(canvas, dtype=torch.bool)
    for row in range(canvas.shape[0]):
        masked = canvas[row].eq(int(mask_token_id))
        positions = masked.nonzero(as_tuple=False).flatten()
        if not int(positions.numel()):
            continue
        block_start = (int(positions[0]) // BLOCK_LENGTH) * BLOCK_LENGTH
        result[row, block_start : block_start + BLOCK_LENGTH] = masked[
            block_start : block_start + BLOCK_LENGTH
        ]
    return result


@torch.no_grad()
def entropy_decode(
    model,
    prompt: Sequence[int],
    mask_token_id: int,
    device: str,
    *,
    temperature: float,
    entropy_threshold: float,
    shift_logits: bool,
) -> dict[str, Any]:
    prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
    canvas = torch.full((1, CANVAS_LENGTH), int(mask_token_id), dtype=torch.long, device=device)
    nfe = 0
    sync(device)
    started = perf_counter()
    while bool(canvas.eq(int(mask_token_id)).any()):
        inputs = torch.cat([prompt_tensor, canvas], dim=1)
        output = model(input_ids=inputs, use_cache=False)
        all_logits = output.logits
        if shift_logits:
            all_logits = torch.cat([all_logits[:, :1], all_logits[:, :-1]], dim=1)
        logits = all_logits[:, -CANVAS_LENGTH:].detach()
        active = active_block_mask(canvas, mask_token_id)[0]
        positions = active.nonzero(as_tuple=False).flatten()
        local = logits[0, positions].float().clone()
        local[:, int(mask_token_id)] = -torch.inf
        sampled = torch.multinomial((local / float(temperature)).softmax(dim=-1), 1).squeeze(-1)
        probability = local.double().softmax(dim=-1)
        log_probability = local.double().log_softmax(dim=-1)
        entropy = -(
            probability * log_probability.masked_fill(~torch.isfinite(log_probability), 0.0)
        ).sum(dim=-1)
        selected = entropy.le(float(entropy_threshold))
        selected[int(entropy.argmin())] = True
        commit_positions = positions[selected]
        canvas[0, commit_positions] = sampled[selected]
        nfe += 1
    sync(device)
    return {
        "tokens": canvas[0].detach().cpu().tolist(),
        "nfe": nfe,
        "latency_seconds": perf_counter() - started,
    }


@torch.no_grad()
def native_diffucoder_decode(
    model,
    prompt: Sequence[int],
    device: str,
    *,
    temperature: float,
    steps: int,
) -> dict[str, Any]:
    inputs = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
    attention = torch.ones_like(inputs)
    sync(device)
    started = perf_counter()
    output = model.diffusion_generate(
        inputs,
        attention_mask=attention,
        max_new_tokens=CANVAS_LENGTH,
        output_history=True,
        return_dict_in_generate=True,
        steps=int(steps),
        temperature=float(temperature),
        top_p=0.95,
        alg="entropy",
        alg_temp=0.0,
    )
    sync(device)
    history = getattr(output, "history", None)
    if history is None or len(history) != int(steps):
        raise RuntimeError("DiffuCoder native sampler returned an unexpected NFE history")
    tokens = output.sequences[0, len(prompt) : len(prompt) + CANVAS_LENGTH]
    return {
        "tokens": tokens.detach().cpu().tolist(),
        "nfe": len(history),
        "latency_seconds": perf_counter() - started,
    }


def decode_text(tokenizer, tokens: Sequence[int]) -> str:
    text = tokenizer.decode(list(tokens), skip_special_tokens=False)
    return text.split("<|dlm_pad|>", 1)[0]


def summarize(method: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    total_nfe = sum(int(record["nfe"]) for record in records)
    latency = sum(float(record["latency_seconds"]) for record in records)
    total_tokens = len(records) * CANVAS_LENGTH
    return {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "examples": len(records),
        "correct": sum(bool(record["correct"]) for record in records),
        "accuracy": sum(bool(record["correct"]) for record in records) / len(records),
        "micro_tokens_per_nfe": total_tokens / total_nfe,
        "mean_nfe": total_nfe / len(records),
        "total_nfe": total_nfe,
        "latency_seconds": latency,
        "tokens_per_second": total_tokens / latency,
    }


def write_table(path: Path, summaries: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# DiffuCoder-cpGRPO versus dParallel: GSM8K diagnostic",
        "",
        "Held-out GSM8K training indices 7463–7472; these are not official-test results.",
        "",
        "| Method | Correct | Accuracy | Tok/NFE | Mean NFE | Latency (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method_label']} | {row['correct']}/{row['examples']} | "
            f"{100 * row['accuracy']:.1f}% | {row['micro_tokens_per_nfe']:.3f} | "
            f"{row['mean_nfe']:.2f} | {row['latency_seconds']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    if args.start != SCREEN_START or args.stop != SCREEN_STOP:
        raise ValueError("this diagnostic is sealed to held-out train indices 7463:7473")
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dparallel_snapshot, dparallel_revision = local_snapshot(MODEL_IDS["dparallel_entropy"])
    diffucoder_snapshot, diffucoder_revision = local_snapshot(MODEL_IDS["diffucoder_native"])
    config = {
        "schema": "diffucoder_vs_dparallel_gsm8k10_v1",
        "split": "train",
        "start": args.start,
        "stop": args.stop,
        "sequence_policy": "256-token historical evaluation canvas; no training targets or cache are generated",
        "canvas_length": CANVAS_LENGTH,
        "block_length": BLOCK_LENGTH,
        "temperature": args.temperature,
        "entropy_threshold": args.entropy_threshold,
        "native_diffucoder_steps": args.native_steps,
        "native_diffucoder_top_p": 0.95,
        "seed": args.seed,
        "model_receipts": {
            MODEL_IDS["dparallel_entropy"]: {
                "revision": dparallel_revision,
                "weight_index_sha256": file_sha256(dparallel_snapshot / "model.safetensors.index.json"),
            },
            MODEL_IDS["diffucoder_native"]: {
                "revision": diffucoder_revision,
                "weight_index_sha256": file_sha256(diffucoder_snapshot / "model.safetensors.index.json"),
            },
        },
        "evaluator_sha256": file_sha256(Path(__file__).resolve()),
        "interpretation": {
            "dparallel_entropy": "published dParallel checkpoint and entropy transition",
            "diffucoder_entropy": "same block-32 entropy transition applied to DiffuCoder; Dream logits shifted by one as in native generation",
            "diffucoder_native": "exact model-card sampler: entropy, top_p=0.95, alg_temp=0, 256 steps",
        },
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    config["config_sha256"] = config_hash
    config_path = output_root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("output directory contains a different immutable diagnostic")
    atomic_json(config_path, config)

    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    if len(dataset) != 7473:
        raise ValueError("GSM8K train split changed")
    preflight_models: dict[str, Any] = {}
    for model_id, snapshot, revision in (
        (MODEL_IDS["dparallel_entropy"], dparallel_snapshot, dparallel_revision),
        (MODEL_IDS["diffucoder_native"], diffucoder_snapshot, diffucoder_revision),
    ):
        model_config = AutoConfig.from_pretrained(snapshot, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
        mask_token_id = getattr(tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            mask_token_id = getattr(model_config, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError(f"{model_id} has no mask token id")
        preflight_models[model_id] = {
            "revision": revision,
            "model_type": model_config.model_type,
            "mask_token_id": int(mask_token_id),
            "first_prompt_tokens": len(_prompt_ids(tokenizer, str(dataset[args.start]["question"]))),
        }
    atomic_json(
        output_root / "preflight.json",
        {
            "schema": "diffucoder_vs_dparallel_gsm8k10_preflight_v1",
            "config_sha256": config_hash,
            "dataset": "openai/gsm8k:main:train",
            "dataset_examples": len(dataset),
            "screen_indices": list(range(args.start, args.stop)),
            "models": preflight_models,
            "passed": True,
        },
    )
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required for the queued diagnostic")

    model_groups = (
        (dparallel_snapshot, ("dparallel_entropy",)),
        (diffucoder_snapshot, ("diffucoder_entropy", "diffucoder_native")),
    )
    all_summaries: list[dict[str, Any]] = []
    for snapshot, methods in model_groups:
        incomplete = [
            method
            for method in methods
            if len(list((output_root / method / "records").glob("*.json"))) < args.stop - args.start
        ]
        if incomplete:
            tokenizer, model, mask_token_id, device = load_base_model(str(snapshot), args.device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for method in incomplete:
                records_dir = output_root / method / "records"
                records_dir.mkdir(parents=True, exist_ok=True)
                for source_index in range(args.start, args.stop):
                    record_path = records_dir / f"{source_index:04d}.json"
                    if record_path.is_file():
                        continue
                    row = dict(dataset[source_index])
                    prompt = _prompt_ids(tokenizer, str(row["question"]))
                    trajectory_seed = stable_seed(args.seed, method, source_index)
                    seed_everything(trajectory_seed)
                    if method == "diffucoder_native":
                        decoded = native_diffucoder_decode(
                            model,
                            prompt,
                            device,
                            temperature=args.temperature,
                            steps=args.native_steps,
                        )
                    else:
                        decoded = entropy_decode(
                            model,
                            prompt,
                            mask_token_id,
                            device,
                            temperature=args.temperature,
                            entropy_threshold=args.entropy_threshold,
                            shift_logits=method == "diffucoder_entropy",
                        )
                    completion = decode_text(tokenizer, decoded["tokens"])
                    prediction = extract_gsm8k_answer(completion)
                    gold = extract_gsm8k_answer(str(row["answer"]))
                    record = {
                        "schema": config["schema"],
                        "config_sha256": config_hash,
                        "method": method,
                        "source_index": source_index,
                        "seed": trajectory_seed,
                        "completion": completion,
                        "predicted_answer": prediction,
                        "gold_answer": gold,
                        "correct": prediction == gold,
                        "nfe": int(decoded["nfe"]),
                        "generated_tokens": CANVAS_LENGTH,
                        "latency_seconds": float(decoded["latency_seconds"]),
                    }
                    atomic_json(record_path, record)
                    print(
                        json.dumps(
                            {
                                "method": method,
                                "completed": source_index - args.start + 1,
                                "target": args.stop - args.start,
                                "correct": record["correct"],
                                "nfe": record["nfe"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        if "model" in locals():
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for method in METHOD_LABELS:
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((output_root / method / "records").glob("*.json"))
        ]
        if len(records) != args.stop - args.start:
            raise RuntimeError(f"{method} is incomplete: {len(records)}/10")
        summary = summarize(method, records)
        atomic_json(output_root / method / "summary.json", summary)
        all_summaries.append(summary)
    atomic_json(output_root / "summary.json", {"config": config, "methods": all_summaries})
    write_table(output_root / "table.md", all_summaries)
    print((output_root / "table.md").read_text(encoding="utf-8"), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=SCREEN_START)
    parser.add_argument("--stop", type=int, default=SCREEN_STOP)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--entropy-threshold", type=float, default=0.5)
    parser.add_argument("--native-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.temperature <= 0 or args.native_steps <= 0:
        parser.error("temperature and native steps must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
