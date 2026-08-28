#!/usr/bin/env python3
"""Fail before submission when required code or checkpoint artifacts are absent."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Sequence

from experiment_contract import METHODS, TEMPERATURES, task_for_id, task_matrix


PINNED_POLICY_COMMIT = "35e4830485f1821d57f9ac3f1a303f3d4531fb82"


def require_policy_checkpoint(value: str | None) -> Path:
    if not value:
        raise ValueError("UNMASKING_POLICY_CHECKPOINT is required")
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
    parser.add_argument("--sft-adapter", default=os.environ.get("SFT_ADAPTER_PATH"))
    parser.add_argument("--policy-checkpoint", default=os.environ.get("UNMASKING_POLICY_CHECKPOINT"))
    parser.add_argument("--policy-repo", default=os.environ.get("ML_RL_DLLM_REPO"))
    args = parser.parse_args(argv)
    adapter = require_adapter(args.sft_adapter)
    checkpoint = require_policy_checkpoint(args.policy_checkpoint)
    repo, revision = require_policy_repo(args.policy_repo)
    matrix = task_matrix()
    if len(matrix) != 60 or len(METHODS) != 5 or len(TEMPERATURES) != 12:
        raise AssertionError("expected a 5 x 12 = 60 task matrix")
    if len(set(matrix)) != len(matrix) or task_for_id(59) != (METHODS[-1], TEMPERATURES[-1]):
        raise AssertionError("task matrix is not a one-to-one deterministic mapping")
    print(
        json.dumps(
            {
                "ok": True,
                "tasks": len(matrix),
                "sft_adapter": adapter,
                "policy_checkpoint": str(checkpoint),
                "policy_repo": str(repo),
                "policy_repo_revision": revision,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
