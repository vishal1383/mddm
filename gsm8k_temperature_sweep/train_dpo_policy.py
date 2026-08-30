#!/usr/bin/env python3
"""Train a logit-free hidden-state unmasking head with online trajectory DPO."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from answers import extract_gsm8k_answer
from dpo_objective import DPO_SCHEMA_VERSION, frontier_preference_pairs
from evaluate import (
    BASE_MODEL_ID,
    DPO_POLICY_ARCHITECTURE,
    MASK_TOKEN_ID,
    ProjectedHiddenSetPolicy,
    active_block_mask,
    atomic_json,
    prompt_ids,
    stable_seed,
)
from experiment_contract import BLOCK_LENGTH, CANVAS_LENGTH, DATASET_CONFIG, DATASET_ID, file_sha256


TRAIN_SPLIT = "train"
TRAIN_EXAMPLES = 7473
# The smart-initialized selector logit is -1.5.  These offsets explore
# approximately 5--20 commits per 32-token active block at initialization,
# covering the useful accuracy/throughput frontier without spending full-run
# compute on effectively serial trajectories.
DEFAULT_POLICY_BIASES = (-0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def stack_trace(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("cannot serialize an empty trajectory")
    return {key: torch.stack([state[key] for state in states]) for key in states[0]}


def trace_sha256(trace: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(trace["action"].contiguous().numpy().tobytes())
    digest.update(trace["valid"].contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_policy_checkpoint(policy: ProjectedHiddenSetPolicy, path: Path) -> Path:
    from safetensors.torch import save_file

    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    state = {key: value.detach().cpu().contiguous() for key, value in policy.state_dict().items()}
    save_file(state, str(temporary))
    os.replace(temporary, path)
    return path


@torch.no_grad()
def collect_hidden_policy_group(
    model,
    policy: ProjectedHiddenSetPolicy,
    prompt_token_ids: Sequence[int],
    policy_biases: Sequence[float],
    *,
    proposal_temperature: float,
    example_index: int,
    master_seed: int,
    mask_token_id: int,
    device: torch.device,
) -> tuple[list[list[int]], list[int], list[dict[str, torch.Tensor]]]:
    """Roll out select-then-sample paths without logit-derived position selection."""

    group = len(policy_biases)
    if group < 2 or proposal_temperature <= 0:
        raise ValueError("collection requires multiple policy biases and positive token temperature")
    prompt = torch.tensor(prompt_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    canvas = torch.full((group, CANVAS_LENGTH), mask_token_id, dtype=torch.long, device=device)
    nfe = [0] * group
    traces: list[list[dict[str, torch.Tensor]]] = [[] for _ in range(group)]
    policy.eval()

    while bool(canvas.eq(mask_token_id).any()):
        rows = canvas.eq(mask_token_id).any(dim=1).nonzero(as_tuple=False).flatten()
        current = canvas.index_select(0, rows)
        inputs = torch.cat([prompt.expand(len(rows), -1), current], dim=1)
        output = model(input_ids=inputs, use_cache=False, output_hidden_states=True)
        if not getattr(output, "hidden_states", None):
            raise RuntimeError("frozen Base did not return hidden states")
        logits = output.logits[:, -CANVAS_LENGTH:].detach().float()
        logits[..., mask_token_id] = -torch.inf
        base_hidden = output.hidden_states[-1][:, -CANVAS_LENGTH:].detach()
        projected_hidden = policy.project_hidden(base_hidden)
        active = active_block_mask(current, mask_token_id, BLOCK_LENGTH)
        masked = current.eq(mask_token_id)
        prior = [nfe[int(value)] for value in rows.tolist()]
        timestep = torch.tensor(prior, device=device, dtype=torch.float32).unsqueeze(-1) / CANVAS_LENGTH
        selector_logits = policy(masked, projected_hidden, timestep).float()

        for local_row, global_row_tensor in enumerate(rows):
            global_row = int(global_row_tensor)
            valid = active[local_row]
            positions = valid.nonzero(as_tuple=False).flatten()
            if not int(positions.numel()):
                raise AssertionError("active block contains no candidates")

            action_generator = torch.Generator(device=device)
            action_generator.manual_seed(
                stable_seed(master_seed, example_index, global_row, nfe[global_row], "dpo-selector")
            )
            biased_logits = selector_logits[local_row] + float(policy_biases[global_row])
            action = torch.bernoulli(torch.sigmoid(biased_logits), generator=action_generator).bool() & valid
            if not bool(action.any()):
                forced = biased_logits.masked_fill(~valid, -torch.inf).argmax()
                action[forced] = True

            # Draw token values only after the hidden-state policy has fixed
            # the commit set. The independent token RNG is never observed by
            # the selector.
            token_generator = torch.Generator(device=device)
            token_generator.manual_seed(
                stable_seed(master_seed, example_index, global_row, nfe[global_row], "dpo-token")
            )
            chosen_positions = action.nonzero(as_tuple=False).flatten()
            token_logits = logits[local_row, chosen_positions] / float(proposal_temperature)
            chosen_tokens = torch.multinomial(
                token_logits.softmax(dim=-1), 1, generator=token_generator
            ).squeeze(-1)

            traces[global_row].append(
                {
                    "mask": masked[local_row].cpu(),
                    "hidden": projected_hidden[local_row].to(torch.float16).cpu(),
                    "timestep": torch.tensor([nfe[global_row] / CANVAS_LENGTH], dtype=torch.float32),
                    "valid": valid.cpu(),
                    "action": action.cpu(),
                }
            )
            canvas[global_row, chosen_positions] = chosen_tokens
            nfe[global_row] += 1
        del output, logits, base_hidden, projected_hidden, inputs

    return canvas.cpu().tolist(), nfe, [stack_trace(trace) for trace in traces]


def trajectory_log_probability(
    policy: ProjectedHiddenSetPolicy,
    trace: dict[str, torch.Tensor],
    device: torch.device,
    chunk: int,
) -> torch.Tensor:
    total = torch.zeros((), device=device, dtype=torch.float32)
    steps = int(trace["mask"].shape[0])
    for start in range(0, steps, chunk):
        stop = min(start + chunk, steps)
        masked = trace["mask"][start:stop].to(device=device, dtype=torch.bool)
        hidden = trace["hidden"][start:stop].to(device=device, dtype=torch.float32)
        timestep = trace["timestep"][start:stop].to(device=device, dtype=torch.float32)
        valid = trace["valid"][start:stop].to(device=device, dtype=torch.bool)
        action = trace["action"][start:stop].to(device=device, dtype=torch.float32)
        logits = policy(masked, hidden, timestep).float()
        log_probability = -F.binary_cross_entropy_with_logits(logits, action, reduction="none")
        total = total + (log_probability * valid).sum()
    return total


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
        "algorithm": "online_onpolicy_hidden_state_trajectory_dpo",
        "dataset": f"{DATASET_ID}:{DATASET_CONFIG}:{TRAIN_SPLIT}",
        "examples": TRAIN_EXAMPLES,
        "base_model_id": BASE_MODEL_ID,
        "base_model_revision": str(resolved_revision),
        "base_trainable_parameters": 0,
        "canvas_length": CANVAS_LENGTH,
        "block_length": BLOCK_LENGTH,
        "selector_inputs": "frozen_base_final_hidden_state_mask_and_timestep_only",
        "selector_excludes": ["token_logits", "confidence", "entropy", "margin", "JSD", "dParallel"],
        "token_rule": "sample_selected_positions_from_frozen_base_full_conditional",
        "proposal_temperature": args.proposal_temperature,
        "behavior": "current_hidden_policy_with_action_rate_bias_sweep",
        "behavior_policy_biases": list(args.behavior_policy_biases),
        "preference_rule": "fastest_correct_over_fastest_incorrect_and_slowest_correct",
        "preference_balance": "one_safety_and_one_efficiency_pair_maximum_per_prompt",
        "reference_initialization": "frozen_copy_of_fresh_hidden_policy",
        "dpo_beta": args.dpo_beta,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "policy_architecture": DPO_POLICY_ARCHITECTURE,
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
        raise RuntimeError("full DPO training requires CUDA")
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
        raise AssertionError("Base must remain frozen")
    if int(model.config.hidden_size) != int(DPO_POLICY_ARCHITECTURE["base_hidden_dim"]):
        raise ValueError("Base hidden size differs from the sealed hidden-policy architecture")
    mask_token_id = int(tokenizer.mask_token_id if tokenizer.mask_token_id is not None else MASK_TOKEN_ID)
    if mask_token_id != MASK_TOKEN_ID:
        raise ValueError(f"unexpected LLaDA mask token {mask_token_id}")

    torch.manual_seed(args.seed)
    reference = ProjectedHiddenSetPolicy(DPO_POLICY_ARCHITECTURE).to(device).eval()
    policy = copy.deepcopy(reference).to(device).train()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=args.weight_decay
    )
    trainable_policy_parameters = sum(
        parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
    )
    if trainable_policy_parameters <= 0 or any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("invalid trainable-parameter boundary")

    resume_path = output_dir / "trainer_resume.pt"
    start_index = updates = correct_paths = preference_pairs = total_paths = total_nfe = 0
    loss_sum = 0.0
    loss_count = 0
    pair_kinds = {"safety": 0, "efficiency": 0}
    prompts_with_winner = 0
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume.get("contract_sha256") != contract["contract_sha256"]:
            raise ValueError("DPO trainer resume contract mismatch")
        policy.load_state_dict(resume["policy"])
        optimizer.load_state_dict(resume["optimizer"])
        start_index = int(resume["next_index"])
        updates = int(resume["updates"])
        correct_paths = int(resume["correct_paths"])
        preference_pairs = int(resume["preference_pairs"])
        total_paths = int(resume["total_paths"])
        total_nfe = int(resume["total_nfe"])
        pair_kinds = {key: int(value) for key, value in resume["pair_kinds"].items()}
        prompts_with_winner = int(resume["prompts_with_winner"])
        loss_sum = float(resume["loss_sum"])
        loss_count = int(resume["loss_count"])

    for index in range(start_index, TRAIN_EXAMPLES):
        row = dict(dataset[index])
        canvases, nfes, traces = collect_hidden_policy_group(
            model,
            policy,
            prompt_ids(tokenizer, str(row["question"])),
            args.behavior_policy_biases,
            proposal_temperature=args.proposal_temperature,
            example_index=index,
            master_seed=args.seed,
            mask_token_id=mask_token_id,
            device=device,
        )
        completions = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in canvases]
        gold = extract_gsm8k_answer(str(row["answer"]))
        predictions = [extract_gsm8k_answer(text) for text in completions]
        correctness = [prediction == gold for prediction in predictions]
        winner, pairs = frontier_preference_pairs(correctness, nfes)

        if pairs:
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            pair_losses = []
            for chosen_index, rejected_index, _kind in pairs:
                chosen = traces[int(chosen_index)]
                rejected = traces[int(rejected_index)]
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
            loss_sum += float(loss.detach().cpu())
            loss_count += 1
        policy.eval()

        correct_paths += sum(bool(value) for value in correctness)
        preference_pairs += len(pairs)
        total_paths += len(traces)
        total_nfe += sum(int(value) for value in nfes)
        prompts_with_winner += int(winner is not None)
        for _chosen, _rejected, kind in pairs:
            pair_kinds[str(kind)] += 1
        atomic_json(
            records_dir / f"{index:04d}.json",
            {
                "schema": DPO_SCHEMA_VERSION,
                "contract_sha256": contract["contract_sha256"],
                "source_index": index,
                "policy_biases": list(args.behavior_policy_biases),
                "gold_answer": gold,
                "predicted_answers": predictions,
                "correct": correctness,
                "nfe": nfes,
                "trace_sha256": [trace_sha256(trace) for trace in traces],
                "winner": winner,
                "pairs": pairs,
            },
        )

        completed = index + 1
        if completed % args.save_every_examples == 0 or completed == TRAIN_EXAMPLES:
            atomic_torch_save(
                resume_path,
                {
                    "contract_sha256": contract["contract_sha256"],
                    "policy": {key: value.detach().cpu() for key, value in policy.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "next_index": completed,
                    "updates": updates,
                    "correct_paths": correct_paths,
                    "preference_pairs": preference_pairs,
                    "total_paths": total_paths,
                    "total_nfe": total_nfe,
                    "pair_kinds": pair_kinds,
                    "prompts_with_winner": prompts_with_winner,
                    "loss_sum": loss_sum,
                    "loss_count": loss_count,
                },
            )
            atomic_json(
                output_dir / "training_progress.json",
                {
                    "contract_sha256": contract["contract_sha256"],
                    "completed_examples": completed,
                    "target_examples": TRAIN_EXAMPLES,
                    "optimizer_updates": updates,
                    "correct_paths": correct_paths,
                    "preference_pairs": preference_pairs,
                    "preference_pair_kinds": pair_kinds,
                    "prompts_with_winner": prompts_with_winner,
                    "total_nfe": total_nfe,
                },
            )
        print(
            json.dumps(
                {
                    "DPO_completed": completed,
                    "target": TRAIN_EXAMPLES,
                    "winner": winner is not None,
                    "pairs": len(pairs),
                    "updates": updates,
                    "mean_nfe": sum(nfes) / len(nfes),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    checkpoint = save_policy_checkpoint(policy.eval(), final_checkpoint)
    manifest = {
        **contract,
        "complete": True,
        "collection_examples": TRAIN_EXAMPLES,
        "collection_paths": total_paths,
        "preference_pairs": preference_pairs,
        "preference_pair_kinds": pair_kinds,
        "prompts_with_winner": prompts_with_winner,
        "correct_paths": correct_paths,
        "collection_total_nfe": total_nfe,
        "trainable_policy_parameters": trainable_policy_parameters,
        "optimizer_updates": updates,
        "mean_dpo_loss": loss_sum / loss_count if loss_count else None,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
    }
    atomic_json(final_manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--base-revision", default=os.environ.get("BASE_MODEL_REVISION", "main"))
    parser.add_argument("--behavior-policy-biases", type=float, nargs="+", default=DEFAULT_POLICY_BIASES)
    parser.add_argument("--proposal-temperature", type=float, default=1.0)
    parser.add_argument("--dpo-beta", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--state-chunk-size", type=int, default=32)
    parser.add_argument("--save-every-examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="eager", choices=("sdpa", "eager", "flash_attention_2"))
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.behavior_policy_biases)) < 4:
        parser.error("at least four distinct action-rate biases are required")
    if (
        args.proposal_temperature <= 0
        or args.dpo_beta <= 0
        or args.learning_rate <= 0
        or args.state_chunk_size <= 0
        or args.save_every_examples <= 0
    ):
        parser.error("temperatures, rates, chunks, and save intervals must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
