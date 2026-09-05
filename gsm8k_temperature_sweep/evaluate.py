#!/usr/bin/env python3
"""Run one full GSM8K method/temperature cell with ten sampled paths."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Any, Sequence
import subprocess

import torch
from torch import nn

from answers import extract_gsm8k_answer
from artifact_sources import (
    BASE_MODEL_ID,
    DPARALLEL_MODEL_ID,
    JUSTGRPO_MODEL_ID,
    SFT_ADAPTER_PATH,
)
from experiment_contract import (
    BLOCK_LENGTH,
    CANVAS_LENGTH,
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_SPLIT,
    METHOD_LABELS,
    METHODS,
    OFFICIAL_TEST_EXAMPLES,
    PROMPT_SUFFIX,
    SAMPLES,
    SCHEMA_VERSION,
    SEED,
    TEMPERATURES,
    canonical_sha256,
    file_sha256,
    resume_compatible_contract,
    summarize_records,
    task_for_id,
    temperature_slug,
)
from policy_specs import APPLE_POLICY_ARCHITECTURE


MASK_TOKEN_ID = 126336
# Backward-compatible name for the retained historical screen16 utility.
PAPER_POLICY_ARCHITECTURE = APPLE_POLICY_ARCHITECTURE
DPO_POLICY_ARCHITECTURE = {
    "policy_type": "projected_hidden_set",
    "base_hidden_dim": 4096,
    "projection_dim": 128,
    "hidden_dim": 128,
    "feedforward_dim": 512,
    "num_heads": 4,
    "dropout": 0.0,
    "smart_init": -1.5,
    "num_blocks": 2,
    "projection_seed": 104729,
    "full_context": True,
    "selector_inputs": "frozen_base_final_hidden_state_mask_and_timestep_only",
}
POLICY_METHODS = {"apple_policy_rl", "dpo_policy_v3"}
DPO_POLICY_TEMPERATURE = 1.0


class ProjectedHiddenSetPolicy(nn.Module):
    """Lightweight set selector over a fixed projection of frozen Base states."""

    def __init__(self, architecture: dict[str, Any]):
        super().__init__()
        base_dim = int(architecture["base_hidden_dim"])
        projection_dim = int(architecture["projection_dim"])
        hidden_dim = int(architecture["hidden_dim"])
        if projection_dim != hidden_dim:
            raise ValueError("projected-hidden v3 requires projection_dim == hidden_dim")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(architecture["projection_seed"]))
        projection = torch.randn(base_dim, projection_dim, generator=generator) / math.sqrt(base_dim)
        self.register_buffer("hidden_projection", projection, persistent=True)
        self.mask_embedding = nn.Embedding(2, hidden_dim)
        self.time_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(architecture["num_heads"]),
            dim_feedforward=int(architecture["feedforward_dim"]),
            dropout=float(architecture["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=int(architecture["num_blocks"]))
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.fill_(float(architecture["smart_init"]))

    def project_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.hidden_projection.shape[0]:
            raise ValueError(
                f"expected Base hidden size {self.hidden_projection.shape[0]}, found {hidden_states.shape[-1]}"
            )
        normalized = torch.nn.functional.layer_norm(
            hidden_states.detach().float(), (hidden_states.shape[-1],)
        )
        return normalized @ self.hidden_projection

    def forward(
        self, masked: torch.Tensor, projected_hidden: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        if projected_hidden.ndim != 3 or projected_hidden.shape[:2] != masked.shape:
            raise ValueError("projected hidden states must have shape [batch, length, width]")
        if timestep.shape != (masked.shape[0], 1):
            raise ValueError("timestep must have shape [batch, 1]")
        conditioning = self.mask_embedding(masked.long()) + self.time_mlp(timestep.float()).unsqueeze(1)
        states = self.context(projected_hidden.float() + conditioning)
        return self.output_proj(self.final_norm(states)).squeeze(-1)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def stable_seed(master: int, example: int, trajectory: int, cycle: int, stream: str) -> int:
    value = canonical_sha256(
        {"master": master, "example": example, "trajectory": trajectory, "cycle": cycle, "stream": stream}
    )
    return int(value[:16], 16) % (2**63 - 1)


def evaluator_source_receipt() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = ("evaluate.py", "answers.py", "experiment_contract.py")
    return {name: file_sha256(root / name) for name in names}


def active_block_mask(canvas: torch.Tensor, mask_token_id: int, block_length: int) -> torch.Tensor:
    active = torch.zeros_like(canvas, dtype=torch.bool)
    masked = canvas.eq(int(mask_token_id))
    for row in range(canvas.shape[0]):
        positions = masked[row].nonzero(as_tuple=False).flatten()
        if not int(positions.numel()):
            continue
        start = (int(positions[0]) // int(block_length)) * int(block_length)
        active[row, start : start + int(block_length)] = masked[row, start : start + int(block_length)]
    return active


@torch.no_grad()
def distribution_interaction_positions(
    logits: torch.Tensor, candidate_mask: torch.Tensor, mask_token_id: int
) -> list[tuple[int, ...]]:
    """Exact local mean-field solve used by the existing MDDM JSD decoder."""

    if logits.ndim != 3 or logits.shape[:2] != candidate_mask.shape:
        raise ValueError("JSD logits and candidates must align")
    result: list[tuple[int, ...]] = []
    log_two = math.log(2.0)
    for row in range(logits.shape[0]):
        positions = candidate_mask[row].nonzero(as_tuple=False).flatten()
        count = int(positions.numel())
        if not count:
            result.append(())
            continue
        local = logits[row, positions].detach().float().clone()
        local[:, int(mask_token_id)] = -torch.inf
        log_probability = local.log_softmax(dim=-1)
        probability = log_probability.exp()
        top_two = local.topk(2, dim=-1).values
        unary = top_two[:, 0] - top_two[:, 1]
        interaction = local.new_zeros((count, count))
        for start in range(0, count, min(4, count)):
            stop = min(start + min(4, count), count)
            right_log = log_probability[start:stop]
            right_probability = probability[start:stop]
            mixture_log = torch.logaddexp(log_probability[:, None, :], right_log[None, :, :]) - log_two
            left_kl = (
                probability[:, None, :]
                * (log_probability[:, None, :] - mixture_log).masked_fill(
                    ~torch.isfinite(log_probability[:, None, :]), 0.0
                )
            ).sum(dim=-1)
            right_kl = (
                right_probability[None, :, :]
                * (right_log[None, :, :] - mixture_log).masked_fill(
                    ~torch.isfinite(right_log[None, :, :]), 0.0
                )
            ).sum(dim=-1)
            interaction[:, start:stop] = (1.0 - 0.5 * (left_kl + right_kl) / log_two).clamp(0.0, 1.0)
        interaction.fill_diagonal_(0.0)
        maximum = interaction.max()
        if bool(maximum.gt(0)):
            interaction = interaction / maximum
        q = unary.sigmoid()
        tolerance = torch.finfo(q.dtype).eps * max(1, count)
        for _ in range(max(1, count * 4)):
            previous = q.clone()
            for index in range(count):
                q[index] = (unary[index] - torch.dot(interaction[index], q)).sigmoid()
            if float((q - previous).abs().max()) <= tolerance:
                break
        result.append(tuple(int(position) for position in positions[q.ge(0.5)].tolist()))
    return result


def sample_active_tokens(
    logits: torch.Tensor,
    active: torch.Tensor,
    temperature: float,
    *,
    example_index: int,
    trajectory_ids: Sequence[int],
    prior_nfe: Sequence[int],
    master_seed: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.full(logits.shape[:2], int(mask_token_id), dtype=torch.long, device=logits.device)
    top_confidence = torch.zeros(logits.shape[:2], dtype=torch.float32, device=logits.device)
    for local_row in range(logits.shape[0]):
        positions = active[local_row].nonzero(as_tuple=False).flatten()
        local = logits[local_row, positions].detach().float().clone()
        local[:, int(mask_token_id)] = -torch.inf
        maximum = local.max(dim=-1).values
        top_confidence[local_row, positions] = (maximum - local.logsumexp(dim=-1)).exp()
        generator = torch.Generator(device=logits.device)
        generator.manual_seed(
            stable_seed(master_seed, example_index, int(trajectory_ids[local_row]), int(prior_nfe[local_row]), "token")
        )
        selected = torch.multinomial((local / float(temperature)).softmax(dim=-1), 1, generator=generator).squeeze(-1)
        tokens[local_row, positions] = selected
    return tokens, top_confidence


def sample_committed_tokens(
    logits: torch.Tensor,
    commit_sets: Sequence[tuple[int, ...]],
    temperature: float,
    *,
    example_index: int,
    trajectory_ids: Sequence[int],
    prior_nfe: Sequence[int],
    master_seed: int,
    mask_token_id: int,
) -> torch.Tensor:
    """Sample only after a logit-free policy has fixed each commit set."""

    tokens = torch.full(logits.shape[:2], int(mask_token_id), dtype=torch.long, device=logits.device)
    for local_row, commit_set in enumerate(commit_sets):
        positions = torch.tensor(commit_set, dtype=torch.long, device=logits.device)
        local = logits[local_row, positions].detach().float().clone()
        local[:, int(mask_token_id)] = -torch.inf
        generator = torch.Generator(device=logits.device)
        generator.manual_seed(
            stable_seed(
                master_seed,
                example_index,
                int(trajectory_ids[local_row]),
                int(prior_nfe[local_row]),
                "token",
            )
        )
        selected = torch.multinomial((local / float(temperature)).softmax(dim=-1), 1, generator=generator)
        tokens[local_row, positions] = selected.squeeze(-1)
    return tokens


def policy_positions(
    policy,
    canvas: torch.Tensor,
    policy_features: torch.Tensor,
    active: torch.Tensor,
    prior_nfe: Sequence[int],
    trajectory_ids: Sequence[int],
    *,
    policy_temperature: float,
    example_index: int,
    master_seed: int,
    mask_token_id: int,
) -> list[tuple[int, ...]]:
    masked = canvas.eq(int(mask_token_id))
    if policy_features.ndim != 3 or policy_features.shape[:2] != canvas.shape:
        raise ValueError("policy features must have shape [batch, length, width]")
    timestep = (
        torch.tensor(prior_nfe, device=canvas.device, dtype=policy_features.dtype).unsqueeze(-1)
        / CANVAS_LENGTH
    )
    logits = policy(masked, policy_features, timestep).float() / float(policy_temperature)
    result: list[tuple[int, ...]] = []
    for row in range(canvas.shape[0]):
        generator = torch.Generator(device=canvas.device)
        generator.manual_seed(
            stable_seed(master_seed, example_index, int(trajectory_ids[row]), int(prior_nfe[row]), "policy")
        )
        draws = torch.bernoulli(torch.sigmoid(logits[row]), generator=generator).bool() & active[row]
        if not bool(draws.any()):
            forced = logits[row].masked_fill(~active[row], -torch.inf).argmax()
            draws[forced] = True
        result.append(tuple(int(value) for value in draws.nonzero(as_tuple=False).flatten().tolist()))
    return result


@torch.no_grad()
def decode_batch(
    model,
    prompt_ids: Sequence[int],
    *,
    method: str,
    policy,
    temperature: float,
    policy_temperature: float,
    confidence_threshold: float,
    entropy_threshold: float,
    canvas_length: int,
    block_length: int,
    samples: int,
    example_index: int,
    seed: int,
    mask_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    canvas = torch.full((samples, canvas_length), int(mask_token_id), dtype=torch.long, device=device)
    nfe = [0] * samples
    digests = [hashlib.sha256() for _ in range(samples)]
    base_seconds = selector_seconds = 0.0
    sync(device)
    started = perf_counter()
    while bool(canvas.eq(int(mask_token_id)).any()):
        rows = canvas.eq(int(mask_token_id)).any(dim=1).nonzero(as_tuple=False).flatten()
        current = canvas.index_select(0, rows)
        inputs = torch.cat([prompt.expand(len(rows), -1), current], dim=1)
        sync(device)
        base_started = perf_counter()
        output = model(
            input_ids=inputs,
            use_cache=False,
            output_hidden_states=method == "dpo_policy_v3",
        )
        sync(device)
        base_seconds += perf_counter() - base_started
        logits = output.logits[:, -canvas_length:].detach()
        selector_started = perf_counter()
        active = active_block_mask(current, mask_token_id, block_length)
        global_rows = [int(value) for value in rows.tolist()]
        prior = [nfe[value] for value in global_rows]
        selected_tokens = None
        top_confidence = None
        if method != "dpo_policy_v3":
            selected_tokens, top_confidence = sample_active_tokens(
                logits,
                active,
                temperature,
                example_index=example_index,
                trajectory_ids=global_rows,
                prior_nfe=prior,
                master_seed=seed,
                mask_token_id=mask_token_id,
            )
        commit_sets: list[tuple[int, ...]] = []
        if method in {"base", "lora_sft", "justgrpo", "jsd_mean_field"}:
            assert top_confidence is not None
            # Position confidence comes from the clean current top-1, exactly
            # as in standard confidence decoding. Temperature changes the
            # committed token draw, not which confidence statistic is gated.
            burst = active & top_confidence.ge(float(confidence_threshold))
            supplement = (
                distribution_interaction_positions(logits, active & ~burst, mask_token_id)
                if method == "jsd_mean_field"
                else [()] * len(rows)
            )
            for local_row in range(len(rows)):
                positions = burst[local_row].nonzero(as_tuple=False).flatten().tolist()
                if method == "jsd_mean_field" and positions:
                    positions.extend(supplement[local_row])
                if not positions:
                    fallback = int(top_confidence[local_row].masked_fill(~active[local_row], -torch.inf).argmax())
                    positions = [fallback]
                commit_sets.append(tuple(sorted(set(map(int, positions)))))
        elif method == "dparallel":
            for local_row in range(len(rows)):
                positions = active[local_row].nonzero(as_tuple=False).flatten()
                local = logits[local_row, positions].detach().double().clone()
                local[:, int(mask_token_id)] = -torch.inf
                probability = local.softmax(dim=-1)
                log_probability = local.log_softmax(dim=-1)
                entropy = -(
                    probability
                    * log_probability.masked_fill(~torch.isfinite(log_probability), 0.0)
                ).sum(dim=-1)
                selected = entropy.le(float(entropy_threshold))
                selected[int(entropy.argmin())] = True
                commit_sets.append(tuple(int(value) for value in positions[selected].tolist()))
        elif method in POLICY_METHODS:
            if policy is None:
                raise ValueError(f"{method} requires a loaded policy")
            if method == "dpo_policy_v3":
                if not getattr(output, "hidden_states", None):
                    raise RuntimeError("Base did not return hidden states for the logit-free selector")
                policy_features = policy.project_hidden(output.hidden_states[-1][:, -canvas_length:])
            else:
                clean = logits.float().clone()
                clean[..., int(mask_token_id)] = -torch.inf
                maximum = clean.max(dim=-1).values
                policy_features = (maximum - clean.logsumexp(dim=-1)).exp().unsqueeze(-1)
            commit_sets = policy_positions(
                policy,
                current,
                policy_features,
                active,
                prior,
                global_rows,
                policy_temperature=policy_temperature,
                example_index=example_index,
                master_seed=seed,
                mask_token_id=mask_token_id,
            )
            if method == "dpo_policy_v3":
                selected_tokens = sample_committed_tokens(
                    logits,
                    commit_sets,
                    temperature,
                    example_index=example_index,
                    trajectory_ids=global_rows,
                    prior_nfe=prior,
                    master_seed=seed,
                    mask_token_id=mask_token_id,
                )
        else:
            raise ValueError(f"unknown method: {method}")

        if selected_tokens is None:
            raise AssertionError("decoder did not sample committed token values")
        for local_row, global_row in enumerate(global_rows):
            positions = commit_sets[local_row]
            tokens = [int(selected_tokens[local_row, position]) for position in positions]
            if not positions or bool(current[local_row, list(positions)].ne(int(mask_token_id)).any()):
                raise AssertionError("decoder produced an invalid commit")
            canvas[global_row, list(positions)] = torch.tensor(tokens, device=device, dtype=torch.long)
            nfe[global_row] += 1
            digests[global_row].update(json.dumps([list(positions), tokens], separators=(",", ":")).encode())
            digests[global_row].update(b"\n")
        sync(device)
        selector_seconds += perf_counter() - selector_started
    sync(device)
    return {
        "canvases": canvas.detach().cpu().tolist(),
        "nfe": nfe,
        "trace_sha256": [digest.hexdigest() for digest in digests],
        "latency_seconds": perf_counter() - started,
        "base_forward_seconds": base_seconds,
        "selector_seconds": selector_seconds,
    }


def token_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, dict):
        tokenized = tokenized["input_ids"]
    if isinstance(tokenized, torch.Tensor):
        tokenized = tokenized.tolist()
    if tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    return list(map(int, tokenized))


def prompt_ids(tokenizer, question: str) -> list[int]:
    content = question.rstrip() + PROMPT_SUFFIX
    return token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True
        )
    )


def checkpoint_file(path_value: str) -> Path:
    if not path_value:
        raise ValueError("a local policy checkpoint path is required")
    path = Path(path_value).expanduser().resolve()
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"policy checkpoint not found: {path}")
    return path


def resolve_policy_checkpoint(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Resolve a locally trained selector and its immutable checksum."""
    if args.method == "dpo_policy_v3":
        checkpoint = checkpoint_file(args.dpo_policy_checkpoint)
        return checkpoint, {
            "source": "locally_trained_dpo",
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
        }
    checkpoint = checkpoint_file(getattr(args, "apple_policy_checkpoint", None))
    return checkpoint, {
        "source": "locally_trained_apple_grpo",
        "path": str(checkpoint),
        "sha256": file_sha256(checkpoint),
    }


def build_policy(
    policy_repo: str,
    device: torch.device,
    architecture: dict[str, Any] | None = None,
):
    upstream = Path(policy_repo).expanduser().resolve()
    if not (upstream / "common/models/policy.py").is_file():
        raise FileNotFoundError(f"official ml-rl-dllm checkout is invalid: {upstream}")
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from common.models.policy import DiTConfidencePolicy, PolicyHFWrapper

    config = dict(architecture or APPLE_POLICY_ARCHITECTURE)
    core = DiTConfidencePolicy(
        hidden_dim=config["hidden_dim"],
        feedforward_dim=config["feedforward_dim"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        time_embed_dim=config["time_embed_dim"],
        confidences_top_p=config["confidences_top_p"],
        smart_init=config["smart_init"],
        num_blocks=config["num_blocks"],
        time_period=config["time_period"],
    ).to(device)
    return PolicyHFWrapper(core, "dit_confidence").to(device)


def load_policy(args: argparse.Namespace, device: torch.device):
    if args.method not in POLICY_METHODS:
        return None, None
    from safetensors.torch import load_file

    architecture = (
        args.resolved_policy_architecture
        if args.method == "dpo_policy_v3"
        else APPLE_POLICY_ARCHITECTURE
    )
    policy = (
        ProjectedHiddenSetPolicy(architecture).to(device)
        if args.method == "dpo_policy_v3"
        else build_policy(args.policy_repo, device, architecture)
    )
    checkpoint = getattr(args, "resolved_policy_checkpoint", None)
    receipt = getattr(args, "resolved_policy_checkpoint_receipt", None)
    if checkpoint is None or receipt is None:
        checkpoint, receipt = resolve_policy_checkpoint(args)
    checkpoint = Path(checkpoint)
    missing, unexpected = policy.load_state_dict(load_file(str(checkpoint)), strict=False)
    if missing or unexpected:
        raise ValueError(f"policy checkpoint mismatch; missing={missing}, unexpected={unexpected}")
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy, dict(receipt)


def resolve_adapter_hash(adapter_path: str | None) -> dict[str, Any] | None:
    if not adapter_path:
        return None
    path = Path(adapter_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"LoRA adapter must be a durable local path: {path}")
    weight = path / "adapter_model.safetensors" if path.is_dir() else path
    if not weight.is_file():
        raise FileNotFoundError(f"LoRA adapter weights are missing: {weight}")
    config = path / "adapter_config.json" if path.is_dir() else path.parent / "adapter_config.json"
    if not config.is_file():
        raise FileNotFoundError(f"LoRA adapter config is missing: {config}")
    resolved = path.resolve()
    try:
        source = str(resolved.relative_to(Path(__file__).resolve().parent))
    except ValueError:
        source = str(resolved)
    return {
        "source": "bundled" if resolved == SFT_ADAPTER_PATH.resolve() else "local_override",
        "path": source,
        "weight_sha256": file_sha256(weight),
        "config_sha256": file_sha256(config),
    }


def git_revision(path_value: str) -> str:
    return subprocess.run(
        ["git", "-C", str(Path(path_value).expanduser().resolve()), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def resolve_model(method: str, token: str | None, requested_revision: str) -> tuple[str, str]:
    from huggingface_hub import HfApi

    model_id = (
        DPARALLEL_MODEL_ID
        if method == "dparallel"
        else JUSTGRPO_MODEL_ID if method == "justgrpo" else BASE_MODEL_ID
    )
    resolved = HfApi(token=token).model_info(model_id, revision=requested_revision).sha
    if not resolved:
        raise RuntimeError(f"could not resolve an immutable revision for {model_id}")
    return model_id, str(resolved)


def load_model_and_tokenizer(args: argparse.Namespace, device: torch.device):
    from transformers import AutoModel, AutoTokenizer

    model_id = args.resolved_model_id
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=args.resolved_model_revision, trust_remote_code=True, token=args.hf_token
    )
    model = AutoModel.from_pretrained(
        model_id,
        revision=args.resolved_model_revision,
        trust_remote_code=True,
        token=args.hf_token,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    if args.method == "lora_sft":
        if not args.sft_adapter:
            raise ValueError("lora_sft requires --sft-adapter")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.sft_adapter)).merge_and_unload()
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("evaluation model contains trainable parameters")
    mask_id = int(tokenizer.mask_token_id if tokenizer.mask_token_id is not None else MASK_TOKEN_ID)
    if mask_id != MASK_TOKEN_ID:
        raise ValueError(f"unexpected LLaDA mask token: expected {MASK_TOKEN_ID}, found {mask_id}")
    revision = getattr(model.config, "_commit_hash", None)
    if revision and str(revision) != args.resolved_model_revision:
        raise ValueError(f"loaded model revision {revision} differs from sealed revision {args.resolved_model_revision}")
    return tokenizer, model, mask_id, {"model_id": model_id, "resolved_revision": args.resolved_model_revision}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.method not in METHODS:
        raise ValueError(f"unknown method: {args.method}")
    if args.temperature not in TEMPERATURES:
        raise ValueError(f"temperature must be one of {TEMPERATURES}")
    if args.samples != SAMPLES or args.canvas_length != CANVAS_LENGTH or args.block_length != BLOCK_LENGTH:
        raise ValueError("the matched experiment is fixed to 10 paths, canvas 256, block 32")
    if args.stop != OFFICIAL_TEST_EXAMPLES or args.start != 0:
        raise ValueError("production cells must cover the complete official 1,319-example test split")

    output = Path(args.output_root).resolve() / args.method / temperature_slug(args.temperature)
    records_dir = output / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    source_receipt = evaluator_source_receipt()
    requested_revision = (
        args.dparallel_revision
        if args.method == "dparallel"
        else args.justgrpo_revision if args.method == "justgrpo" else args.base_revision
    )
    args.resolved_model_id, args.resolved_model_revision = resolve_model(args.method, args.hf_token, requested_revision)
    policy_checkpoint_receipt = None
    policy_training_receipt = None
    policy_repo_revision = None
    if args.method in POLICY_METHODS:
        policy_checkpoint_path, policy_checkpoint_receipt = resolve_policy_checkpoint(args)
        args.resolved_policy_checkpoint = policy_checkpoint_path
        args.resolved_policy_checkpoint_receipt = policy_checkpoint_receipt
        policy_repo_revision = git_revision(args.policy_repo) if args.method == "apple_policy_rl" else None
        if args.method in {"apple_policy_rl", "dpo_policy_v3"}:
            training_manifest = (
                policy_checkpoint_path.parent.parent / "training_manifest.json"
                if args.method == "apple_policy_rl"
                else policy_checkpoint_path.parent / "training_manifest.json"
            )
            if not training_manifest.is_file():
                raise FileNotFoundError(f"policy checkpoint has no training manifest: {training_manifest}")
            training_data = json.loads(training_manifest.read_text(encoding="utf-8"))
            if not training_data.get("complete"):
                raise ValueError(f"policy training is not complete: {training_manifest}")
            if args.method == "apple_policy_rl" and training_data.get(
                "selected_checkpoint_sha256"
            ) != policy_checkpoint_receipt["sha256"]:
                raise ValueError("Apple policy checkpoint hash differs from its training manifest")
            if args.method == "apple_policy_rl" and training_data.get(
                "apple_source_commit"
            ) != policy_repo_revision:
                raise ValueError("Apple training and evaluation source revisions differ")
            if training_data.get("base_model_revision") != args.resolved_model_revision:
                raise ValueError("policy checkpoint and evaluation Base revisions differ")
            args.resolved_policy_architecture = training_data.get("policy_architecture")
            expected_architecture = (
                DPO_POLICY_ARCHITECTURE
                if args.method == "dpo_policy_v3"
                else APPLE_POLICY_ARCHITECTURE
            )
            if args.resolved_policy_architecture != expected_architecture:
                raise ValueError("policy checkpoint architecture differs from the sealed architecture")
            policy_training_receipt = {
                "path": str(training_manifest),
                "sha256": file_sha256(training_manifest),
            }
    contract = {
        "schema": SCHEMA_VERSION,
        "method": args.method,
        "method_label": METHOD_LABELS[args.method],
        "temperature": args.temperature,
        "temperature_semantics": "categorical_softmax_logits_div_T",
        "policy_temperature": (
            DPO_POLICY_TEMPERATURE
            if args.method == "dpo_policy_v3"
            else args.policy_temperature if args.method == "apple_policy_rl" else None
        ),
        "policy_sampling": "independent_bernoulli_then_force_argmax_if_empty"
        if args.method in POLICY_METHODS
        else None,
        "dataset": f"{DATASET_ID}:{DATASET_CONFIG}:{DATASET_SPLIT}",
        "start": args.start,
        "stop": args.stop,
        "samples": args.samples,
        "canvas_length": args.canvas_length,
        "block_length": args.block_length,
        "prompt_suffix": PROMPT_SUFFIX,
        "seed": args.seed,
        "confidence_threshold": args.confidence_threshold if args.method in {"base", "jsd_mean_field", "justgrpo", "lora_sft"} else None,
        "confidence_statistic": "clean_current_top1_probability"
        if args.method in {"base", "jsd_mean_field", "justgrpo", "lora_sft"}
        else None,
        "dparallel_entropy_threshold": args.entropy_threshold if args.method == "dparallel" else None,
        "one_base_forward_per_cycle": True,
        "model_id": args.resolved_model_id,
        "model_revision": args.resolved_model_revision,
        "evaluator_sources_sha256": source_receipt,
        "sft_adapter": resolve_adapter_hash(args.sft_adapter) if args.method == "lora_sft" else None,
        "unmasking_policy_architecture": (
            args.resolved_policy_architecture
            if args.method == "dpo_policy_v3"
            else APPLE_POLICY_ARCHITECTURE if args.method == "apple_policy_rl" else None
        ),
        "unmasking_policy_checkpoint": policy_checkpoint_receipt,
        "unmasking_policy_training_manifest": policy_training_receipt,
        "unmasking_policy_repo_revision": policy_repo_revision,
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    contract_path = output / "contract.json"
    if contract_path.exists():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing_contract != contract:
            if not resume_compatible_contract(existing_contract, contract):
                raise ValueError(f"resume directory has a different contract: {output}")
            print(
                f"Reusing semantically compatible baseline contract for partial cell: {output}",
                flush=True,
            )
            contract = existing_contract
    atomic_json(contract_path, contract)

    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT)
    if len(dataset) != OFFICIAL_TEST_EXAMPLES:
        raise ValueError(f"official GSM8K test size changed: expected 1319, found {len(dataset)}")

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required (pass --allow-cpu only for a development smoke run)")
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer, model, mask_token_id, model_receipt = load_model_and_tokenizer(args, device)
    policy, policy_receipt = load_policy(args, device)
    atomic_json(
        output / "runtime_manifest.json",
        {**contract, "model": model_receipt, "policy_checkpoint": policy_receipt, "torch": torch.__version__},
    )

    for index in range(args.start, args.stop):
        record_path = records_dir / f"{index:04d}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("contract_sha256") != contract["contract_sha256"] or int(record.get("source_index", -1)) != index:
                raise ValueError(f"invalid resume record: {record_path}")
            continue
        row = dict(dataset[index])
        ids = prompt_ids(tokenizer, str(row["question"]))
        decoded = decode_batch(
            model,
            ids,
            method=args.method,
            policy=policy,
            temperature=args.temperature,
            policy_temperature=(
                DPO_POLICY_TEMPERATURE if args.method == "dpo_policy_v3" else args.policy_temperature
            ),
            confidence_threshold=args.confidence_threshold,
            entropy_threshold=args.entropy_threshold,
            canvas_length=args.canvas_length,
            block_length=args.block_length,
            samples=args.samples,
            example_index=index,
            seed=args.seed,
            mask_token_id=mask_token_id,
            device=device,
        )
        completions = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in decoded["canvases"]]
        predictions = [extract_gsm8k_answer(text) for text in completions]
        gold = extract_gsm8k_answer(str(row["answer"]))
        paths = []
        for sample_index in range(args.samples):
            paths.append(
                {
                    "sample_index": sample_index,
                    "temperature": args.temperature,
                    "decoded_completion": completions[sample_index],
                    "predicted_answer": predictions[sample_index],
                    "gold_answer": gold,
                    "correct": predictions[sample_index] == gold,
                    "base_forwards": int(decoded["nfe"][sample_index]),
                    "generated_tokens": args.canvas_length,
                    "trace_sha256": decoded["trace_sha256"][sample_index],
                }
            )
        record = {
            "schema": SCHEMA_VERSION,
            "contract_sha256": contract["contract_sha256"],
            "method": args.method,
            "temperature": args.temperature,
            "example_id": str(index),
            "source_index": index,
            "question": str(row["question"]),
            "gold_answer": gold,
            "paths": paths,
            "pass_at_5": any(path["correct"] for path in paths[:5]),
            "pass_at_10": any(path["correct"] for path in paths),
            "unique_normalized_answers_at_10": len(set(predictions)),
            "unique_traces_at_10": len(set(decoded["trace_sha256"])),
            "batch_latency_seconds": float(decoded["latency_seconds"]),
            "base_forward_seconds": float(decoded["base_forward_seconds"]),
            "selector_seconds": float(decoded["selector_seconds"]),
        }
        atomic_json(record_path, record)
        print(
            json.dumps(
                {
                    "method": args.method,
                    "temperature": args.temperature,
                    "completed": index + 1,
                    "target": args.stop,
                    "pass_at_10": record["pass_at_10"],
                    "micro_tokens_per_nfe_this_example": args.samples * args.canvas_length / sum(decoded["nfe"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    records = [json.loads((records_dir / f"{index:04d}.json").read_text(encoding="utf-8")) for index in range(args.start, args.stop)]
    summary = {
        "schema": SCHEMA_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "method": args.method,
        "method_label": METHOD_LABELS[args.method],
        "temperature": args.temperature,
        **summarize_records(records),
        "complete": len(records) == OFFICIAL_TEST_EXAMPLES,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sft-adapter", default=os.environ.get("SFT_ADAPTER_PATH", str(SFT_ADAPTER_PATH)))
    parser.add_argument("--apple-policy-checkpoint", default=os.environ.get("APPLE_POLICY_CHECKPOINT"))
    parser.add_argument("--dpo-policy-checkpoint", default=os.environ.get("DPO_POLICY_CHECKPOINT"))
    parser.add_argument("--policy-repo", default=os.environ.get("ML_RL_DLLM_REPO"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--base-revision", default=os.environ.get("BASE_MODEL_REVISION", "main"))
    parser.add_argument("--dparallel-revision", default=os.environ.get("DPARALLEL_MODEL_REVISION", "main"))
    parser.add_argument("--justgrpo-revision", default=os.environ.get("JUSTGRPO_MODEL_REVISION", "main"))
    parser.add_argument("--policy-temperature", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--entropy-threshold", type=float, default=0.50)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--canvas-length", type=int, default=CANVAS_LENGTH)
    parser.add_argument("--block-length", type=int, default=BLOCK_LENGTH)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=OFFICIAL_TEST_EXAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--attn-implementation", default="eager", choices=("sdpa", "eager", "flash_attention_2"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args(argv)
    if args.task_id is not None:
        mapped_method, mapped_temperature = task_for_id(args.task_id)
        if args.method is not None and args.method != mapped_method:
            parser.error("--method conflicts with --task-id")
        if args.temperature is not None and args.temperature != mapped_temperature:
            parser.error("--temperature conflicts with --task-id")
        args.method, args.temperature = mapped_method, mapped_temperature
    if args.method is None or args.temperature is None:
        parser.error("provide --task-id, or both --method and --temperature")
    if args.method == "lora_sft" and not args.sft_adapter:
        parser.error("lora_sft requires SFT_ADAPTER_PATH or --sft-adapter")
    if args.method == "apple_policy_rl" and not args.policy_repo:
        parser.error("apple_policy_rl requires ML_RL_DLLM_REPO")
    if args.method == "apple_policy_rl" and not args.apple_policy_checkpoint:
        parser.error("apple_policy_rl requires APPLE_POLICY_CHECKPOINT or --apple-policy-checkpoint")
    if args.method == "dpo_policy_v3" and not args.dpo_policy_checkpoint:
        parser.error("dpo_policy_v3 requires DPO_POLICY_CHECKPOINT")
    return args


if __name__ == "__main__":
    run(parse_args())
