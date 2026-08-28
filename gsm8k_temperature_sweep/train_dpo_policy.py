#!/usr/bin/env python3
"""Train Apple's frozen-Base unmasking head with offline trajectory DPO."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from answers import extract_gsm8k_answer
from dpo_objective import DPO_SCHEMA_VERSION, multiplicative_reward, strict_preference_pairs
from evaluate import (
    BASE_MODEL_ID,
    MASK_TOKEN_ID,
    PAPER_POLICY_ARCHITECTURE,
    active_block_mask,
    atomic_json,
    build_policy,
    git_revision,
    prompt_ids,
)
from experiment_contract import BLOCK_LENGTH, CANVAS_LENGTH, DATASET_CONFIG, DATASET_ID, file_sha256


TRAIN_SPLIT = "train"
TRAIN_EXAMPLES = 7473
DEFAULT_THRESHOLDS = (0.30, 0.50, 0.70, 0.90)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def stack_trace(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("cannot serialize an empty trajectory")
    return {key: torch.stack([state[key] for state in states]) for key in states[0]}


@torch.no_grad()
def collect_threshold_group(
    model,
    prompt_token_ids: Sequence[int],
    thresholds: Sequence[float],
    *,
    mask_token_id: int,
    device: torch.device,
) -> tuple[list[list[int]], list[int], list[dict[str, torch.Tensor]]]:
    """Collect deterministic frozen-Base paths and all Bernoulli action observations."""
    group = len(thresholds)
    prompt = torch.tensor(prompt_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    canvas = torch.full((group, CANVAS_LENGTH), mask_token_id, dtype=torch.long, device=device)
    nfe = [0] * group
    traces: list[list[dict[str, torch.Tensor]]] = [[] for _ in range(group)]

    while bool(canvas.eq(mask_token_id).any()):
        rows = canvas.eq(mask_token_id).any(dim=1).nonzero(as_tuple=False).flatten()
        current = canvas.index_select(0, rows)
        inputs = torch.cat([prompt.expand(len(rows), -1), current], dim=1)
        output = model(input_ids=inputs, use_cache=False)
        logits = output.logits[:, -CANVAS_LENGTH:].detach().float()
        logits[..., mask_token_id] = -torch.inf
        log_normalizer = logits.logsumexp(dim=-1)
        maximum, top_tokens = logits.max(dim=-1)
        confidence = (maximum - log_normalizer).exp()
        active = active_block_mask(current, mask_token_id, BLOCK_LENGTH)

        for local_row, global_row_tensor in enumerate(rows):
            global_row = int(global_row_tensor)
            valid = active[local_row]
            action = valid & confidence[local_row].ge(float(thresholds[global_row]))
            if not bool(action.any()):
                fallback = confidence[local_row].masked_fill(~valid, -torch.inf).argmax()
                action[fallback] = True
            traces[global_row].append(
                {
                    "mask": current[local_row].eq(mask_token_id).cpu(),
                    "confidence": confidence[local_row].to(torch.float16).cpu(),
                    "timestep": torch.tensor([nfe[global_row] / CANVAS_LENGTH], dtype=torch.float32),
                    "valid": valid.cpu(),
                    "action": action.cpu(),
                }
            )
            canvas[global_row, action] = top_tokens[local_row, action]
            nfe[global_row] += 1
        del output, logits, inputs

    return canvas.cpu().tolist(), nfe, [stack_trace(trace) for trace in traces]


def validate_record(record: dict[str, Any], index: int, contract_sha256: str) -> None:
    if record.get("schema") != DPO_SCHEMA_VERSION:
        raise ValueError("preference record schema mismatch")
    if int(record.get("source_index", -1)) != index:
        raise ValueError("preference record example mismatch")
    if record.get("contract_sha256") != contract_sha256:
        raise ValueError("preference record contract mismatch")


def trajectory_log_probability(policy, trace: dict[str, torch.Tensor], device: torch.device, chunk: int) -> torch.Tensor:
    total = torch.zeros((), device=device, dtype=torch.float32)
    steps = int(trace["mask"].shape[0])
    for start in range(0, steps, chunk):
        stop = min(start + chunk, steps)
        masked = trace["mask"][start:stop].to(device=device, dtype=torch.bool)
        confidence = trace["confidence"][start:stop].to(device=device, dtype=torch.float32).unsqueeze(-1)
        timestep = trace["timestep"][start:stop].to(device=device, dtype=torch.float32)
        valid = trace["valid"][start:stop].to(device=device, dtype=torch.bool)
        action = trace["action"][start:stop].to(device=device, dtype=torch.float32)
        logits = policy(masked, confidence, timestep).float()
        log_probability = -F.binary_cross_entropy_with_logits(logits, action, reduction="none")
        total = total + (log_probability * valid).sum()
    return total


def save_final_policy(policy, output_dir: Path) -> Path:
    from safetensors.torch import save_file

    checkpoint = output_dir / "model.safetensors"
    temporary = output_dir / f"model.safetensors.tmp.{os.getpid()}"
    state = {key: value.detach().cpu().contiguous() for key, value in policy.state_dict().items()}
    save_file(state, str(temporary))
    os.replace(temporary, checkpoint)
    return checkpoint


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    records_dir = output_dir / "preferences"
    output_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi

    resolved_revision = HfApi(token=args.hf_token).model_info(BASE_MODEL_ID, revision=args.base_revision).sha
    if not resolved_revision:
        raise RuntimeError("could not resolve the frozen Base revision")
    contract = {
        "schema": DPO_SCHEMA_VERSION,
        "algorithm": "offline_trajectory_dpo",
        "dataset": f"{DATASET_ID}:{DATASET_CONFIG}:{TRAIN_SPLIT}",
        "examples": TRAIN_EXAMPLES,
        "base_model_id": BASE_MODEL_ID,
        "base_model_revision": str(resolved_revision),
        "base_trainable_parameters": 0,
        "canvas_length": CANVAS_LENGTH,
        "block_length": BLOCK_LENGTH,
        "behavior": "greedy_frozen_base_confidence_threshold_group",
        "behavior_thresholds": list(args.behavior_thresholds),
        "reward": "correct*((L-min(NFE,L)+1)/L)**alpha",
        "reward_alpha": args.reward_alpha,
        "dpo_beta": args.dpo_beta,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "seed": args.seed,
        "policy_architecture": PAPER_POLICY_ARCHITECTURE,
        "policy_repo_revision": git_revision(args.policy_repo),
        "trainer_sources_sha256": {
            name: file_sha256(Path(__file__).resolve().parent / name)
            for name in ("train_dpo_policy.py", "dpo_objective.py", "evaluate.py", "answers.py")
        },
    }
    from experiment_contract import canonical_sha256

    contract["contract_sha256"] = canonical_sha256(contract)
    contract_path = output_dir / "training_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8")) != contract:
        raise ValueError(f"DPO output has a different immutable contract: {output_dir}")
    atomic_json(contract_path, contract)
    final_checkpoint = output_dir / "model.safetensors"
    final_manifest = output_dir / "training_manifest.json"
    if final_checkpoint.is_file() and final_manifest.is_file():
        existing = json.loads(final_manifest.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("contract_sha256") == contract["contract_sha256"]:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return

    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=TRAIN_SPLIT)
    if len(dataset) != TRAIN_EXAMPLES:
        raise ValueError(f"expected {TRAIN_EXAMPLES} GSM8K training examples, found {len(dataset)}")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("full DPO collection requires CUDA")
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, revision=resolved_revision, trust_remote_code=True, token=args.hf_token
    )
    model = AutoModel.from_pretrained(
        BASE_MODEL_ID,
        revision=resolved_revision,
        trust_remote_code=True,
        token=args.hf_token,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("Base must remain frozen during preference collection")
    mask_token_id = int(tokenizer.mask_token_id if tokenizer.mask_token_id is not None else MASK_TOKEN_ID)
    if mask_token_id != MASK_TOKEN_ID:
        raise ValueError(f"unexpected LLaDA mask token {mask_token_id}")

    correct_paths = preference_pairs = total_paths = total_nfe = 0
    for index in range(TRAIN_EXAMPLES):
        record_path = records_dir / f"{index:04d}.pt"
        if record_path.is_file():
            record = torch.load(record_path, map_location="cpu", weights_only=True)
            validate_record(record, index, contract["contract_sha256"])
        else:
            row = dict(dataset[index])
            canvases, nfes, traces = collect_threshold_group(
                model,
                prompt_ids(tokenizer, str(row["question"])),
                args.behavior_thresholds,
                mask_token_id=mask_token_id,
                device=device,
            )
            completions = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in canvases]
            gold = extract_gsm8k_answer(str(row["answer"]))
            correctness = [extract_gsm8k_answer(text) == gold for text in completions]
            rewards = [
                multiplicative_reward(correct, nfe, CANVAS_LENGTH, args.reward_alpha)
                for correct, nfe in zip(correctness, nfes)
            ]
            pairs = strict_preference_pairs(rewards)
            record = {
                "schema": DPO_SCHEMA_VERSION,
                "contract_sha256": contract["contract_sha256"],
                "source_index": index,
                "nfe": nfes,
                "correct": correctness,
                "rewards": rewards,
                "pairs": pairs,
                "traces": traces,
            }
            atomic_torch_save(record_path, record)
        correct_paths += sum(bool(value) for value in record["correct"])
        preference_pairs += len(record["pairs"])
        total_paths += len(record["traces"])
        total_nfe += sum(int(value) for value in record["nfe"])
        if (index + 1) % 25 == 0 or index + 1 == TRAIN_EXAMPLES:
            atomic_json(
                output_dir / "collection_progress.json",
                {
                    "contract_sha256": contract["contract_sha256"],
                    "completed_examples": index + 1,
                    "total_examples": TRAIN_EXAMPLES,
                    "paths": total_paths,
                    "correct_paths": correct_paths,
                    "preference_pairs": preference_pairs,
                    "total_nfe": total_nfe,
                },
            )
            print(f"DPO collection {index + 1}/{TRAIN_EXAMPLES}; pairs={preference_pairs}", flush=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if preference_pairs <= 0:
        raise RuntimeError("offline collection produced no strict preferences")

    torch.manual_seed(args.seed)
    reference = build_policy(args.policy_repo, device).eval()
    policy = copy.deepcopy(reference).train()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=args.weight_decay
    )
    trainable_policy_parameters = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
    if trainable_policy_parameters <= 0:
        raise AssertionError("DPO policy has no trainable parameters")
    resume_path = output_dir / "trainer_resume.pt"
    start_epoch = start_index = updates = 0
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume.get("contract_sha256") != contract["contract_sha256"]:
            raise ValueError("DPO trainer resume contract mismatch")
        policy.load_state_dict(resume["policy"])
        optimizer.load_state_dict(resume["optimizer"])
        start_epoch = int(resume["epoch"])
        start_index = int(resume["next_index"])
        updates = int(resume["updates"])

    losses: list[float] = []
    for epoch in range(start_epoch, args.epochs):
        first_index = start_index if epoch == start_epoch else 0
        for index in range(first_index, TRAIN_EXAMPLES):
            record = torch.load(records_dir / f"{index:04d}.pt", map_location="cpu", weights_only=True)
            validate_record(record, index, contract["contract_sha256"])
            if not record["pairs"]:
                continue
            optimizer.zero_grad(set_to_none=True)
            pair_losses = []
            for chosen_index, rejected_index in record["pairs"]:
                chosen = record["traces"][int(chosen_index)]
                rejected = record["traces"][int(rejected_index)]
                chosen_logp = trajectory_log_probability(policy, chosen, device, args.state_chunk_size)
                rejected_logp = trajectory_log_probability(policy, rejected, device, args.state_chunk_size)
                with torch.no_grad():
                    reference_chosen = trajectory_log_probability(reference, chosen, device, args.state_chunk_size)
                    reference_rejected = trajectory_log_probability(reference, rejected, device, args.state_chunk_size)
                margin = (chosen_logp - rejected_logp) - (reference_chosen - reference_rejected)
                pair_losses.append(-F.logsigmoid(args.dpo_beta * margin))
            loss = torch.stack(pair_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach().cpu()))
            if updates % args.save_every_updates == 0:
                atomic_torch_save(
                    resume_path,
                    {
                        "contract_sha256": contract["contract_sha256"],
                        "policy": {key: value.detach().cpu() for key, value in policy.state_dict().items()},
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "next_index": index + 1,
                        "updates": updates,
                    },
                )
                print(f"DPO fit epoch={epoch + 1}/{args.epochs} example={index + 1} loss={losses[-1]:.6f}", flush=True)
        start_index = 0

    checkpoint = save_final_policy(policy.eval(), output_dir)
    manifest = {
        **contract,
        "complete": True,
        "collection_examples": TRAIN_EXAMPLES,
        "collection_paths": total_paths,
        "preference_pairs": preference_pairs,
        "correct_paths": correct_paths,
        "collection_total_nfe": total_nfe,
        "trainable_policy_parameters": trainable_policy_parameters,
        "optimizer_updates": updates,
        "mean_dpo_loss": sum(losses) / len(losses) if losses else None,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
    }
    atomic_json(final_manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-repo", default=os.environ.get("ML_RL_DLLM_REPO"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--base-revision", default=os.environ.get("BASE_MODEL_REVISION", "main"))
    parser.add_argument("--behavior-thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--reward-alpha", type=float, default=1.0)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--state-chunk-size", type=int, default=32)
    parser.add_argument("--save-every-updates", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="eager", choices=("sdpa", "eager", "flash_attention_2"))
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args(argv)
    if not args.policy_repo:
        parser.error("ML_RL_DLLM_REPO or --policy-repo is required")
    if len(set(args.behavior_thresholds)) < 2:
        parser.error("at least two distinct behavior thresholds are required")
    if args.epochs <= 0 or args.dpo_beta <= 0 or args.learning_rate <= 0:
        parser.error("epochs, DPO beta, and learning rate must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
