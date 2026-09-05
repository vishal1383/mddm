#!/usr/bin/env python3
"""Train Apple's official BL32 confidence policy and seal the selected checkpoint.

The algorithm, policy, rollout, reward, and GRPO loss come from the pinned
``apple/ml-rl-dllm`` checkout.  This launcher supplies an immutable Base
snapshot, freezes it explicitly, and adds local checkpoint resume so a
preemptible one-A100 job can finish the paper's full one-epoch mixture run.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Sequence

from artifact_sources import BASE_MODEL_ID
from experiment_contract import file_sha256
from policy_specs import APPLE_POLICY_ARCHITECTURE


APPLE_COMMIT = "35e4830485f1821d57f9ac3f1a303f3d4531fb82"
TRAINING_SCHEMA = "apple_official_grpo_bl32_alpha03_v1"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git_revision(path: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def latest_resumable_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    if not output_dir.is_dir():
        return None
    for path in output_dir.iterdir():
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if not match or not path.is_dir():
            continue
        required = ("model.safetensors", "trainer_state.json", "optimizer.pt", "scheduler.pt")
        if all((path / name).is_file() for name in required):
            candidates.append((int(match.group(1)), path))
    return max(candidates)[1] if candidates else None


def completed_checkpoint(output_dir: Path) -> Path | None:
    manifest = output_dir / "training_manifest.json"
    selected = output_dir / "checkpoint-best" / "model.safetensors"
    if not manifest.is_file() or not selected.is_file():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not data.get("complete") or data.get("selected_checkpoint_sha256") != file_sha256(selected):
        return None
    return selected


def run(args: argparse.Namespace) -> Path:
    apple_root = args.apple_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if git_revision(apple_root) != APPLE_COMMIT:
        raise ValueError(f"Apple source must be pinned to {APPLE_COMMIT}")
    required_sources = (
        apple_root / "train/train.py",
        apple_root / "train/trainer.py",
        apple_root / "train/reward_func.py",
        apple_root / "common/config.py",
        apple_root / "common/models/policy.py",
        apple_root / "common/models/policy_layers.py",
        apple_root / "common/generation/generation.py",
        apple_root / "common/generation/sampling.py",
        apple_root / "data/data_utils.py",
    )
    if not all(path.is_file() for path in required_sources) or not config_path.is_file():
        raise FileNotFoundError("Apple checkout or training config is incomplete")

    from huggingface_hub import snapshot_download

    base_snapshot = Path(
        snapshot_download(
            repo_id=BASE_MODEL_ID,
            revision=args.base_revision,
            token=args.hf_token,
        )
    ).resolve()
    contract = {
        "schema": TRAINING_SCHEMA,
        "algorithm": "Apple official confidence-policy GRPO",
        "apple_source_commit": APPLE_COMMIT,
        "apple_source_sha256": {path.relative_to(apple_root).as_posix(): file_sha256(path) for path in required_sources},
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "base_model_id": BASE_MODEL_ID,
        "base_model_revision": args.base_revision,
        "base_snapshot": str(base_snapshot),
        "dataset": "GSM8K train + MATH train, proportionally mixed (~15K examples)",
        "epochs": 1,
        "alpha_compute_reward": 0.3,
        "block_length": 32,
        "completion_length": 256,
        "num_generations": 8,
        "policy_architecture": APPLE_POLICY_ARCHITECTURE,
        "trainable_base_parameters": 0,
        "one_gpu_adaptation": "per-device/global batch 16; official policy, rollout, reward, and loss unchanged",
    }
    contract_path = output_dir / "training_contract.json"
    if contract_path.is_file():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError(f"existing Apple training contract differs: {contract_path}")
    else:
        atomic_json(contract_path, contract)

    already_complete = completed_checkpoint(output_dir)
    if already_complete is not None:
        manifest_data = json.loads(
            (output_dir / "training_manifest.json").read_text(encoding="utf-8")
        )
        if any(manifest_data.get(key) != value for key, value in contract.items()):
            raise ValueError("completed Apple checkpoint has a stale training contract")
        print(f"Apple GRPO training already complete: {already_complete}", flush=True)
        return already_complete

    # Apple's reward module imports the third-party package named ``evaluate``.
    # Preload that package without allowing this experiment's evaluate.py to
    # shadow it on sys.path.
    experiment_root = Path(__file__).resolve().parent
    original_sys_path = list(sys.path)
    try:
        sys.path = [
            entry
            for entry in sys.path
            if Path(entry or os.getcwd()).resolve() != experiment_root
        ]
        hf_evaluate = importlib.import_module("evaluate")
        if not hasattr(hf_evaluate, "load"):
            raise ImportError("the Hugging Face evaluate package was shadowed")
    finally:
        sys.path = original_sys_path

    if str(apple_root) not in sys.path:
        sys.path.insert(0, str(apple_root))
    apple_train = importlib.import_module("train.train")
    parser = apple_train.TrlParser((apple_train.Config, apple_train.ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config(
        args=["--config", str(config_path)], fail_with_unknown_args=False
    )
    grpo_config.output_dir = str(output_dir)
    grpo_config.model_path = str(base_snapshot)
    grpo_config.resume_from_checkpoint = True

    if (
        grpo_config.dataset != "gsm8k_and_math"
        or grpo_config.policy_type != "dit_confidence"
        or grpo_config.remasking != "policy"
        or grpo_config.sampling_mode != "bernoulli"
        or float(grpo_config.alpha_compute_reward) != 0.3
    ):
        raise ValueError("config no longer matches the sealed Apple BL32 alpha=0.3 run")

    original_auto_model = apple_train.AutoModel
    original_trainer = apple_train.Trainer

    class FrozenAutoModel:
        @classmethod
        def from_pretrained(cls, *model_args, **model_kwargs):
            model = original_auto_model.from_pretrained(*model_args, **model_kwargs)
            model.requires_grad_(False)
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise AssertionError("Apple environment model was not frozen")
            return model

    class ResumableTrainer(original_trainer):
        last_instance = None

        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            type(self).last_instance = self

        def train(self, *trainer_args, **trainer_kwargs):
            resume = latest_resumable_checkpoint(output_dir)
            if resume is not None:
                trainer_kwargs["resume_from_checkpoint"] = str(resume)
                print(f"Resuming Apple GRPO from {resume}", flush=True)
            return super().train(*trainer_args, **trainer_kwargs)

    apple_train.AutoModel = FrozenAutoModel
    apple_train.Trainer = ResumableTrainer
    try:
        apple_train.main(grpo_config=grpo_config, model_config=model_config)
    finally:
        apple_train.AutoModel = original_auto_model
        apple_train.Trainer = original_trainer

    selected = output_dir / "checkpoint-best" / "model.safetensors"
    if not selected.is_file():
        raise FileNotFoundError(f"official trainer did not save checkpoint-best: {selected}")
    instance = ResumableTrainer.last_instance
    final_step = int(instance.state.global_step) if instance is not None else None
    manifest = {
        **contract,
        "complete": True,
        "final_global_step": final_step,
        "selected_checkpoint": str(selected),
        "selected_checkpoint_sha256": file_sha256(selected),
    }
    atomic_json(output_dir / "training_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return selected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apple-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
