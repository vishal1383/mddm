#!/usr/bin/env python3
"""Fail before submission when required code or checkpoint artifacts are absent."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Sequence

from artifact_sources import SFT_ADAPTER_PATH
from experiment_contract import METHODS, TEMPERATURES, task_for_id, task_matrix


PINNED_POLICY_COMMIT = "35e4830485f1821d57f9ac3f1a303f3d4531fb82"


def require_local_policy_checkpoint(value: str | None) -> Path:
    if not value:
        raise ValueError("DPO_POLICY_CHECKPOINT is required")
    path = Path(value).expanduser().resolve()
    checkpoint = path / "model.safetensors" if path.is_dir() else path
    if not checkpoint.is_file():
        raise FileNotFoundError(f"policy checkpoint is missing: {checkpoint}")
    return checkpoint


def require_adapter(value: str | None) -> str:
    if not value:
        raise ValueError("SFT_ADAPTER_PATH is required")
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"LoRA adapter must be a durable local path visible to compute nodes: {path}"
        )
    config = path / "adapter_config.json" if path.is_dir() else path.parent / "adapter_config.json"
    weight = path / "adapter_model.safetensors" if path.is_dir() else path
    if not config.is_file() or not weight.is_file():
        raise FileNotFoundError(f"LoRA adapter_config.json or adapter_model.safetensors is missing under {path}")
    return str(path.resolve())


def require_policy_repo(value: str | None) -> tuple[Path, str]:
    if not value:
        raise ValueError("ML_RL_DLLM_REPO is required; run scripts/bootstrap_env.sh")
    path = Path(value).expanduser().resolve()
    if not (path / "common/models/policy.py").is_file():
        raise FileNotFoundError(f"invalid ml-rl-dllm checkout: {path}")
    revision = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if revision != PINNED_POLICY_COMMIT:
        raise ValueError(f"ml-rl-dllm must be pinned to {PINNED_POLICY_COMMIT}; found {revision}")
    return path, revision


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-adapter", default=os.environ.get("SFT_ADAPTER_PATH", str(SFT_ADAPTER_PATH)))
    parser.add_argument("--apple-policy-checkpoint", default=os.environ.get("APPLE_POLICY_CHECKPOINT"))
    parser.add_argument("--dpo-policy-checkpoint", default=os.environ.get("DPO_POLICY_CHECKPOINT"))
    parser.add_argument("--require-apple", action="store_true")
    parser.add_argument("--require-dpo", action="store_true")
    parser.add_argument("--policy-repo", default=os.environ.get("ML_RL_DLLM_REPO"))
    args = parser.parse_args(argv)
    adapter = require_adapter(args.sft_adapter)
    apple_checkpoint = (
        require_local_policy_checkpoint(args.apple_policy_checkpoint) if args.require_apple else None
    )
    if apple_checkpoint is not None:
        manifest = apple_checkpoint.parent.parent / "training_manifest.json"
        manifest_data = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
        if not manifest_data.get("complete"):
            raise ValueError(f"Apple policy has no complete training manifest: {manifest}")
        from experiment_contract import file_sha256

        if manifest_data.get("selected_checkpoint_sha256") != file_sha256(apple_checkpoint):
            raise ValueError("Apple policy checkpoint hash differs from its training manifest")
    dpo_checkpoint = require_local_policy_checkpoint(args.dpo_policy_checkpoint) if args.require_dpo else None
    repo, revision = require_policy_repo(args.policy_repo)
    matrix = task_matrix()
    if len(matrix) != 28 or len(METHODS) != 7 or TEMPERATURES != (0.1, 0.5, 0.8, 1.2):
        raise AssertionError("expected a 7 x 4 = 28 task matrix")
    if len(set(matrix)) != len(matrix) or task_for_id(27) != (METHODS[-1], TEMPERATURES[-1]):
        raise AssertionError("task matrix is not a one-to-one deterministic mapping")
    print(
        json.dumps(
            {
                "ok": True,
                "tasks": len(matrix),
                "sft_adapter": adapter,
                "apple_policy_checkpoint": str(apple_checkpoint) if apple_checkpoint else None,
                "dpo_policy_checkpoint": str(dpo_checkpoint) if dpo_checkpoint else None,
                "policy_repo": str(repo),
                "policy_repo_revision": revision,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
