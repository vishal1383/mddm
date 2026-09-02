#!/usr/bin/env python3
"""Matched 16-example Apple-GRPO versus dParallel GSM8K evaluation.

This is intentionally separate from the full 1,319 x 10 production sweep. It
loads the same sealed artifacts and decoder implementation, but evaluates one
trajectory per official-test prompt at exact greedy token decoding and T=0.5.
Each example is an atomic resumable record.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch

from answers import extract_gsm8k_answer
from artifact_sources import BASE_MODEL_ID, DPARALLEL_MODEL_ID, PAPER_POLICY_FILENAME, PAPER_POLICY_REPO_ID
from evaluate import (
    PAPER_POLICY_ARCHITECTURE,
    atomic_json,
    decode_batch,
    load_model_and_tokenizer,
    load_policy,
    prompt_ids,
)
from experiment_contract import (
    BLOCK_LENGTH,
    CANVAS_LENGTH,
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_SPLIT,
    PROMPT_SUFFIX,
    canonical_sha256,
    file_sha256,
)


SCHEMA = "apple_dparallel_gsm8k_screen16_v1"
METHODS = ("paper_policy", "dparallel")
TEMPERATURES = (0.0, 0.5)
LABELS = {
    "paper_policy": "Unofficial Apple GRPO reproduction (reward-selected checkpoint)",
    "dparallel": "dParallel published checkpoint",
}


def _temperature_slug(value: float) -> str:
    return "greedy" if value == 0.0 else f"T{value:g}"


def _resolve_policy(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.policy_checkpoint:
        path = Path(args.policy_checkpoint).expanduser().resolve()
        if path.is_dir():
            path = path / "model.safetensors"
        source: dict[str, Any] = {"source": "local_override", "path": str(path)}
    else:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id=PAPER_POLICY_REPO_ID,
                filename=PAPER_POLICY_FILENAME,
                revision=args.paper_policy_revision,
                token=args.hf_token,
            )
        ).resolve()
        source = {
            "source": "huggingface",
            "repo_id": PAPER_POLICY_REPO_ID,
            "filename": PAPER_POLICY_FILENAME,
            "revision": args.paper_policy_revision,
        }
    if not path.is_file():
        raise FileNotFoundError(f"Apple-policy checkpoint is missing: {path}")
    source["sha256"] = file_sha256(path)
    return path, source


def _runtime_args(args: argparse.Namespace, method: str) -> tuple[SimpleNamespace, dict[str, Any] | None]:
    if method == "paper_policy":
        checkpoint, policy_receipt = _resolve_policy(args)
        model_id = BASE_MODEL_ID
        model_revision = args.base_revision
    else:
        checkpoint = None
        policy_receipt = None
        model_id = DPARALLEL_MODEL_ID
        model_revision = args.dparallel_revision
    runtime = SimpleNamespace(
        method=method,
        resolved_model_id=model_id,
        resolved_model_revision=model_revision,
        hf_token=args.hf_token,
        attn_implementation=args.attn_implementation,
        sft_adapter=None,
        policy_repo=args.policy_repo,
        resolved_policy_checkpoint=checkpoint,
        resolved_policy_checkpoint_receipt=policy_receipt,
    )
    return runtime, policy_receipt


def _summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 16:
        raise ValueError(f"expected 16 records, found {len(records)}")
    total_nfe = sum(int(row["base_forwards"]) for row in records)
    total_tokens = sum(int(row["generated_tokens"]) for row in records)
    total_seconds = sum(float(row["latency_seconds"]) for row in records)
    correct = sum(bool(row["correct"]) for row in records)
    return {
        "examples": len(records),
        "accuracy": correct / len(records),
        "correct": correct,
        "total_nfe": total_nfe,
        "mean_nfe": total_nfe / len(records),
        "micro_tokens_per_nfe": total_tokens / total_nfe,
        "total_generated_tokens": total_tokens,
        "synchronized_latency_seconds": total_seconds,
        "end_to_end_tokens_per_second": total_tokens / total_seconds,
    }


@torch.no_grad()
def evaluate_method(args: argparse.Namespace) -> dict[str, Any]:
    if args.method not in METHODS:
        raise ValueError(f"unsupported method: {args.method}")
    if args.start < 0 or args.stop - args.start != 16:
        raise ValueError("the matched screen must contain exactly 16 examples")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    policy_root = Path(args.policy_repo).expanduser().resolve()
    if args.method == "paper_policy" and not (policy_root / "common/models/policy.py").is_file():
        raise FileNotFoundError(f"invalid Apple policy repository: {policy_root}")

    runtime, policy_receipt = _runtime_args(args, args.method)
    contract = {
        "schema": SCHEMA,
        "method": args.method,
        "method_label": LABELS[args.method],
        "dataset": f"{DATASET_ID}:{DATASET_CONFIG}:{DATASET_SPLIT}",
        "source_indices": [args.start, args.stop],
        "temperatures": list(TEMPERATURES),
        "samples_per_example_temperature": 1,
        "canvas_length": CANVAS_LENGTH,
        "block_length": BLOCK_LENGTH,
        "prompt_suffix": PROMPT_SUFFIX,
        "seed": args.seed,
        "model_id": runtime.resolved_model_id,
        "model_revision": runtime.resolved_model_revision,
        "attention_implementation": args.attn_implementation,
        "policy_checkpoint": policy_receipt,
        "policy_architecture": PAPER_POLICY_ARCHITECTURE if args.method == "paper_policy" else None,
        "policy_temperature": args.policy_temperature if args.method == "paper_policy" else None,
        "policy_sampling": "independent Bernoulli; force policy argmax only if empty"
        if args.method == "paper_policy"
        else None,
        "dparallel_entropy_threshold": args.entropy_threshold if args.method == "dparallel" else None,
        "greedy_semantics": "exact token argmax at T_token=0; selector semantics unchanged",
        "one_base_forward_per_cycle": True,
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    root = Path(args.output_root).resolve() / args.method
    root.mkdir(parents=True, exist_ok=True)
    contract_path = root / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError(f"resume contract mismatch: {contract_path}")
    else:
        atomic_json(contract_path, contract)

    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT)
    if len(dataset) != 1319:
        raise ValueError(f"official GSM8K test size changed: {len(dataset)}")
    device = torch.device(args.device)
    tokenizer, model, mask_token_id, model_receipt = load_model_and_tokenizer(runtime, device)
    policy, loaded_policy_receipt = load_policy(runtime, device)
    atomic_json(root / "runtime_manifest.json", {
        **contract,
        "model": model_receipt,
        "policy": loaded_policy_receipt,
        "torch": torch.__version__,
    })

    summaries: list[dict[str, Any]] = []
    for temperature in TEMPERATURES:
        cell = root / _temperature_slug(temperature)
        records_dir = cell / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        for index in range(args.start, args.stop):
            record_path = records_dir / f"{index:04d}.json"
            if record_path.exists():
                saved = json.loads(record_path.read_text(encoding="utf-8"))
                if saved.get("contract_sha256") != contract["contract_sha256"]:
                    raise ValueError(f"incompatible resume record: {record_path}")
                continue
            row = dict(dataset[index])
            decoded = decode_batch(
                model,
                prompt_ids(tokenizer, str(row["question"])),
                method=args.method,
                policy=policy,
                temperature=temperature,
                policy_temperature=args.policy_temperature,
                confidence_threshold=0.9,
                entropy_threshold=args.entropy_threshold,
                canvas_length=CANVAS_LENGTH,
                block_length=BLOCK_LENGTH,
                samples=1,
                example_index=index,
                seed=args.seed,
                mask_token_id=mask_token_id,
                device=device,
            )
            completion = tokenizer.decode(decoded["canvases"][0], skip_special_tokens=True)
            prediction = extract_gsm8k_answer(completion)
            gold = extract_gsm8k_answer(str(row["answer"]))
            record = {
                "schema": SCHEMA,
                "contract_sha256": contract["contract_sha256"],
                "method": args.method,
                "temperature": temperature,
                "source_index": index,
                "predicted_answer": prediction,
                "gold_answer": gold,
                "correct": prediction == gold,
                "decoded_completion": completion,
                "base_forwards": int(decoded["nfe"][0]),
                "generated_tokens": CANVAS_LENGTH,
                "trace_sha256": decoded["trace_sha256"][0],
                "latency_seconds": float(decoded["latency_seconds"]),
                "base_forward_seconds": float(decoded["base_forward_seconds"]),
                "selector_seconds": float(decoded["selector_seconds"]),
            }
            atomic_json(record_path, record)
            print(json.dumps({
                "method": args.method,
                "temperature": temperature,
                "completed": index - args.start + 1,
                "target": 16,
                "correct": record["correct"],
                "nfe": record["base_forwards"],
            }, sort_keys=True), flush=True)
        records = [
            json.loads((records_dir / f"{index:04d}.json").read_text(encoding="utf-8"))
            for index in range(args.start, args.stop)
        ]
        summary = {
            "schema": SCHEMA,
            "contract_sha256": contract["contract_sha256"],
            "method": args.method,
            "method_label": LABELS[args.method],
            "temperature": temperature,
            **_summarize(records),
        }
        atomic_json(cell / "summary.json", summary)
        summaries.append(summary)
    result = {"schema": SCHEMA, "method": args.method, "cells": summaries, "complete": True}
    atomic_json(root / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        summary_path = root / method / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing completed method summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("complete"):
            raise ValueError(f"method is incomplete: {method}")
        rows.extend(summary["cells"])
    table = {
        "schema": SCHEMA,
        "dataset": "official GSM8K test examples 0..15",
        "rows": rows,
        "complete": True,
    }
    atomic_json(root / "final_table.json", table)
    markdown = [
        "| Method | Token decoding | Correct | Accuracy | Tok/NFE | Mean NFE | Tokens/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        decoding = "greedy" if float(row["temperature"]) == 0.0 else "T=0.5"
        markdown.append(
            f"| {row['method_label']} | {decoding} | {row['correct']}/16 | "
            f"{100 * row['accuracy']:.2f}% | {row['micro_tokens_per_nfe']:.3f} | "
            f"{row['mean_nfe']:.2f} | {row['end_to_end_tokens_per_second']:.2f} |"
        )
    (root / "final_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("\n".join(markdown), flush=True)
    return table


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--policy-repo", default=os.environ.get("ML_RL_DLLM_REPO"))
    parser.add_argument("--policy-checkpoint", default=os.environ.get("UNMASKING_POLICY_CHECKPOINT"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--base-revision", default="08b83a6feb34df1a6011b80c3c00c7563e963b07")
    parser.add_argument("--dparallel-revision", default="bbdd4fd017d7d5be141dcf276492d57b6166468f")
    parser.add_argument("--paper-policy-revision", default="12d570517c80fae7271773e668fc0d179b3ad155")
    # Paper setting for semi-autoregressive BL=32.  BL=256 full diffusion uses
    # policy temperature 1.0 instead; canvas length alone does not determine it.
    parser.add_argument("--policy-temperature", type=float, default=0.5)
    parser.add_argument("--entropy-threshold", type=float, default=0.5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if not args.aggregate_only and args.method is None:
        parser.error("--method is required unless --aggregate-only is used")
    if not args.aggregate_only and args.method == "paper_policy" and not args.policy_repo:
        parser.error("paper_policy requires --policy-repo or ML_RL_DLLM_REPO")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.aggregate_only:
        aggregate(args)
    else:
        evaluate_method(args)


if __name__ == "__main__":
    main()
